"""dag_kb.retrieval.temporal -- deterministic "no stale memory" retrieval.

End-to-end through query, LLM deliberately left out (per the experiment's own
scope: "everything until query ... mocking retrieval" -- these tests build a
real KnowledgeDAG via the real gate/add_record path, with real dates, and
assert on classify_facts()'s output directly, the exact context an LLM would
receive, without ever invoking one).

Scenarios mirror the motivating doctor questions verbatim:
  - "what medication has the patient taken in the past?"    -> PAST
  - "what medication is the patient taking now?"             -> CURRENT (+ CURRENT_STALE, disclosed)
  - "what chronic diseases does the patient have?"           -> CURRENT (+ CURRENT_STALE, disclosed)
  - "is the patient sick now?" with a 3-month-old last record -> CURRENT_STALE, staleness note present
  - same question with a from-yesterday record                -> CURRENT, no staleness note
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dag_kb import (
    Cardinality,
    NodeState,
    Recency,
    Record,
    Tier,
    classify_facts,
    filter_recency,
)
from dag_kb.graph import KnowledgeDAG
from dag_kb.ids import compute_record_id
from dag_kb.store import InMemoryRecordStore

NOW = datetime(2026, 9, 14, tzinfo=timezone.utc)  # fixed clock, deterministic tests


def _relation(subject_ref, predicate, object_ref, *, seq, valid_from, source_id="doc"):
    key = f"{subject_ref}::{predicate.lower()}::{object_ref}"
    rid = compute_record_id(semantic_key=key, predicate=predicate, obj=object_ref,
                            parent_ids=(), owner=key, owner_seq=seq)
    return Record(record_id=rid, semantic_key=key, predicate=predicate, obj=object_ref,
                  parent_ids=(), owner=key, owner_seq=seq, tier=Tier.T4, auto_update=True,
                  record_type="asserted", valid_from=valid_from, source_id=source_id,
                  raw_citation="c", txn_id="t",
                  subject_ref=subject_ref, object_ref=object_ref, cardinality=Cardinality.MANY)


def _dag_with_medication_history() -> KnowledgeDAG:
    """patient_7714: metformin started 2 years ago then discontinued (superseded
    by a dosage-supersede-like ARCHIVED state via explicit set_state, simulating
    "no longer prescribed"), ibuprofen taken once a year ago (never revisited --
    ACTIVE but old, MANY-cardinality: no natural supersession), and biotin
    started yesterday (ACTIVE, fresh)."""
    store = InMemoryRecordStore()
    dag = KnowledgeDAG(store)

    metformin = _relation("patient_7714", "Treated_With", "metformin", seq=0,
                          valid_from=NOW - timedelta(days=730), source_id="doc:visit_2024")
    ibuprofen = _relation("patient_7714", "Treated_With", "ibuprofen", seq=1,
                          valid_from=NOW - timedelta(days=365), source_id="doc:visit_2025a")
    biotin = _relation("patient_7714", "Treated_With", "biotin", seq=2,
                       valid_from=NOW - timedelta(days=1), source_id="doc:visit_yesterday")

    for r in (metformin, ibuprofen, biotin):
        store.put(r)
        dag.add_record(r, state=NodeState.ACTIVE)

    # metformin was explicitly discontinued -- superseded, PAST regardless of date
    dag.set_state(metformin.record_id, NodeState.ARCHIVED,
                  actor=__import__("dag_kb").Actor.SYSTEM_1, reason="discontinued")
    return dag


def test_medication_taken_in_the_past():
    dag = _dag_with_medication_history()
    facts = classify_facts(dag, "patient_7714", now=NOW)
    past = filter_recency(facts, Recency.PAST)
    assert {f.object_label for f in past} == {"metformin"}
    assert past[0].as_text().startswith("[PAST / SUPERSEDED]")


def test_medication_taking_now_includes_stale_but_active_with_disclosure():
    dag = _dag_with_medication_history()
    facts = classify_facts(dag, "patient_7714", now=NOW, freshness_days=90)
    current = filter_recency(facts, Recency.CURRENT, Recency.CURRENT_STALE)
    labels = {f.object_label for f in current}
    # biotin (1 day old) is fresh; ibuprofen (365 days old, never revisited,
    # still ACTIVE) is "current" in the data but must be flagged stale --
    # metformin must NOT appear here, it's PAST (discontinued).
    assert labels == {"biotin", "ibuprofen"}

    biotin = next(f for f in current if f.object_label == "biotin")
    ibuprofen = next(f for f in current if f.object_label == "ibuprofen")
    assert biotin.recency is Recency.CURRENT and biotin.staleness_note is None
    assert ibuprofen.recency is Recency.CURRENT_STALE
    assert "365 days ago" in ibuprofen.staleness_note
    assert "may be outdated" in ibuprofen.staleness_note
    assert "[CURRENT, but as of" in ibuprofen.as_text()


def _dag_with_diagnosis(days_old: int):
    store = InMemoryRecordStore()
    dag = KnowledgeDAG(store)
    dx = _relation("patient_x", "Diagnosed_With", "ards", seq=0,
                   valid_from=NOW - timedelta(days=days_old), source_id="doc:visit_x")
    store.put(dx)
    dag.add_record(dx, state=NodeState.ACTIVE)
    return dag


def test_chronic_disease_query_flags_a_3_month_old_record_as_stale():
    # "what chronic diseases does the patient have?" / "is the patient sick now?"
    # -- last diagnosis record is 3 months (≈92 days) old
    dag = _dag_with_diagnosis(days_old=92)
    facts = classify_facts(dag, "patient_x", now=NOW, freshness_days=90)
    assert len(facts) == 1 and facts[0].recency is Recency.CURRENT_STALE
    assert facts[0].days_old == 92
    note = facts[0].staleness_note
    assert "92 days ago" in note and "may be outdated" in note
    # the exact disclosure shape requested: date + explicit "may be outdated"
    assert facts[0].doc_date.date().isoformat() in note


def test_diagnosis_from_yesterday_has_no_staleness_disclosure():
    dag = _dag_with_diagnosis(days_old=1)
    facts = classify_facts(dag, "patient_x", now=NOW, freshness_days=90)
    assert facts[0].recency is Recency.CURRENT
    assert facts[0].staleness_note is None
    assert facts[0].as_text() == "patient_x — Diagnosed_With — ards [source: doc:visit_x]"


def test_source_and_date_are_always_attached_to_every_fact():
    dag = _dag_with_medication_history()
    facts = classify_facts(dag, "patient_7714", now=NOW)
    for f in facts:
        assert f.doc_date is not None
        assert f.source_ids  # never empty -- provenance is mandatory, per the design goal
        assert f.days_old is not None


def test_freshness_window_is_configurable_not_hardcoded():
    # the exact "3 months" boundary is a policy knob, not a magic number baked
    # into the classifier -- a 30-day window makes the same 92-day-old record
    # stale, a 365-day window makes it current.
    dag = _dag_with_diagnosis(days_old=92)
    tight = classify_facts(dag, "patient_x", now=NOW, freshness_days=30)[0]
    loose = classify_facts(dag, "patient_x", now=NOW, freshness_days=365)[0]
    assert tight.recency is Recency.CURRENT_STALE
    assert loose.recency is Recency.CURRENT and loose.staleness_note is None


def test_no_facts_for_unknown_entity_fails_closed_not_error():
    store = InMemoryRecordStore()
    dag = KnowledgeDAG(store)
    assert classify_facts(dag, "nonexistent_patient", now=NOW) == []


def test_facts_sorted_most_relevant_first():
    # CURRENT, then CURRENT_STALE, then PAST -- and newest-first within each
    # group -- regardless of commit order.
    dag = _dag_with_medication_history()
    facts = classify_facts(dag, "patient_7714", now=NOW)
    assert [f.object_label for f in facts] == ["biotin", "ibuprofen", "metformin"]
    assert [f.recency for f in facts] == [Recency.CURRENT, Recency.CURRENT_STALE, Recency.PAST]
