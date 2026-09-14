"""dag_kb.retrieval.frequency.most_common -- flood-then-tally aggregation for
"N most common X of Y" / "all X of Y" queries (e.g. "3 most common symptoms
of ARDS"), as distinct from BoundedRetriever's single-path nearest-context
retrieval.
"""
from __future__ import annotations

from datetime import datetime, timezone

from dag_kb import Cardinality, NodeState, Record, Tier, most_common
from dag_kb.graph import KnowledgeDAG
from dag_kb.ids import compute_record_id
from dag_kb.store import InMemoryRecordStore


def _relation(subject_ref, predicate, object_ref, *, seq):
    key = f"{subject_ref}::{predicate.lower()}::{object_ref}"
    rid = compute_record_id(semantic_key=key, predicate=predicate, obj=object_ref,
                            parent_ids=(), owner=key, owner_seq=seq)
    return Record(record_id=rid, semantic_key=key, predicate=predicate, obj=object_ref,
                  parent_ids=(), owner=key, owner_seq=seq, tier=Tier.T4, auto_update=True,
                  record_type="asserted", valid_from=datetime.now(timezone.utc),
                  source_id="s", raw_citation="c", txn_id="t",
                  subject_ref=subject_ref, object_ref=object_ref, cardinality=Cardinality.MANY)


def _dag_with_patients() -> KnowledgeDAG:
    """3 patients diagnosed with ARDS (fever x3, cough x2, dyspnea x1) and one
    patient diagnosed with something else entirely (should not be counted)."""
    store = InMemoryRecordStore()
    dag = KnowledgeDAG(store)
    seq = 0
    facts = [
        ("patient_1", "Diagnosed_With", "ards"),
        ("patient_1", "Presents_With", "fever"),
        ("patient_1", "Presents_With", "cough"),
        ("patient_2", "Diagnosed_With", "ards"),
        ("patient_2", "Presents_With", "fever"),
        ("patient_2", "Presents_With", "cough"),
        ("patient_3", "Diagnosed_With", "ards"),
        ("patient_3", "Presents_With", "fever"),
        ("patient_3", "Presents_With", "dyspnea"),
        ("patient_4", "Diagnosed_With", "flu"),      # different disease
        ("patient_4", "Presents_With", "fever"),       # must NOT be counted for ards
    ]
    for subj, pred, obj in facts:
        r = _relation(subj, pred, obj, seq=seq)
        seq += 1
        store.put(r)
        dag.add_record(r, state=NodeState.ACTIVE)
    return dag


def test_most_common_symptoms_via_diagnosis_flood():
    dag = _dag_with_patients()
    ranked = most_common(dag, "ards", via_predicates=("Diagnosed_With",),
                         target_predicates=("Presents_With",))
    # fever: 3, cough: 2, dyspnea: 1 -- and the flu patient's fever must not leak in
    assert ranked == [("fever", 3), ("cough", 2), ("dyspnea", 1)]


def test_top_n_truncates_without_changing_order():
    dag = _dag_with_patients()
    ranked = most_common(dag, "ards", via_predicates=("Diagnosed_With",),
                         target_predicates=("Presents_With",), top_n=2)
    assert ranked == [("fever", 3), ("cough", 2)]


def test_predicate_matching_is_case_insensitive():
    dag = _dag_with_patients()
    ranked = most_common(dag, "ards", via_predicates=("diagnosed_with",),
                         target_predicates=("presents_with",))
    assert ranked[0] == ("fever", 3)


def test_unrelated_disease_contributes_nothing():
    dag = _dag_with_patients()
    ranked = most_common(dag, "flu", via_predicates=("Diagnosed_With",),
                         target_predicates=("Presents_With",))
    assert ranked == [("fever", 1)]


def test_no_via_predicates_aggregates_direct_edges_of_seed():
    # e.g. disease <-Symptom_Of- symptom recorded with no patient in between
    store = InMemoryRecordStore()
    dag = KnowledgeDAG(store)
    for i, obj in enumerate(("fever", "fever", "cough")):
        r = _relation("ards", "Has_Common_Symptom", obj, seq=i)
        store.put(r)
        dag.add_record(r, state=NodeState.ACTIVE)
    ranked = most_common(dag, "ards", target_predicates=("Has_Common_Symptom",))
    assert ranked == [("fever", 2), ("cough", 1)]


def test_empty_seed_returns_empty_list():
    dag = _dag_with_patients()
    assert most_common(dag, "nonexistent_entity", via_predicates=("Diagnosed_With",),
                       target_predicates=("Presents_With",)) == []
