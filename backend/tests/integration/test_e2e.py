"""End-to-end evaluation of the full deterministic memory pipeline.

Best-practice shape: one deterministic scenario walked stage by stage, with an
explicit assertion at every stage boundary (extraction -> resolution -> delta ->
gate -> DAG state -> cascade -> PLANFENCE -> retrieval -> SYSTEM_2 -> revert),
plus a guardrail matrix that feeds malformed LLM output at each stage and proves
nothing ever auto-commits on garbage.

All LLM behaviour is supplied by MockProvider, so the run is fully deterministic.
Extractor replies are NVIDIA-format Python-literal lists of 5-tuples (see
pipeline.stage1_extract / tests.conftest.nvidia_triplets).
"""
from __future__ import annotations

import json

import pytest

from conftest import nvidia_triplets
from dag_kb import (
    Cardinality,
    Delta,
    Disposition,
    NodeState,
    QueueItemType,
    Routing,
    Tier,
    ValidationOutcome,
    compute_distance,
)
from llm.mock import UNAVAILABLE, MockProvider
from pipeline import semantic_key
from service.assembly import build_memory_layer

# "max building height" canonicalizes to a curated functional (single-valued)
# attribute -> cardinality ONE, so a new value supersedes the old one.
KEY = semantic_key("Sector 4", "max building height", cardinality=Cardinality.ONE)


def _mock(*, triplets, delta="SUPERSEDING_CHANGE"):
    return MockProvider(
        {
            "INPUT TEXT": nvidia_triplets(triplets),
            "Compare the NEW fact": json.dumps({"delta": delta, "rationale": "value changed"}),
            "named entities": json.dumps({"entities": ["Sector 4"]}),
            "using ONLY the FACTS": json.dumps({"answer": "45 stories"}),
        },
        default="",
    )


T_30 = [("Sector 4", "Zone", "max building height", "30 stories", "METRIC")]
T_45 = [("Sector 4", "Zone", "max building height", "45 stories", "METRIC")]


# --------------------------------------------------------------------------- #
# 1. the happy path, asserted stage by stage
# --------------------------------------------------------------------------- #
def test_full_pipeline_walkthrough(tmp_path):
    mem = build_memory_layer(provider=_mock(triplets=T_30), data_dir=str(tmp_path / "kb"))

    # stage 0-1: source policy + extraction ------------------------------- #
    raw = mem.pipeline._registry.from_text("Res 402: 30 stories in Sector 4", "vendor_doc:v1")
    assert raw.tier is Tier.T2 and raw.auto_update is True          # policy applied
    drafts = mem.pipeline._extractor.extract(raw)
    assert len(drafts) == 1 and drafts[0].semantic_key == KEY       # validated draft
    assert drafts[0].obj == "30 stories"
    assert drafts[0].subject_ref == "sector_4"                      # entity ref resolved

    # stage 2-5: resolve -> (no active) -> gate AUTO_COMMIT -------------- #
    report1 = mem.pipeline.ingest_text("Res 402: 30 stories in Sector 4", "vendor_doc:v1")
    assert report1.summary()["by_disposition"] == {"AUTO_COMMIT": 1}
    v0 = mem.heads.get(KEY).record_id
    assert mem.store.get(v0).obj == "30 stories"
    assert mem.dag.get_state(v0) is NodeState.ACTIVE
    assert mem.dag.is_entity("sector_4")                             # entity node created

    # a derived T3 child so we can watch the cascade -------------------- #
    from datetime import datetime, timezone

    from dag_kb import TripletDraft
    child = TripletDraft(semantic_key="sector_4::buildable", predicate="buildable",
                         obj="yes", tier=Tier.T3, auto_update=True, source_id="derived:d1",
                         raw_citation="derived", valid_from=datetime.now(timezone.utc),
                         parent_ids=(v0,), record_type="derived")
    child_id = mem.gate.submit(child, Delta.SUPERSEDING_CHANGE,
                               writer_identity="derived:d1").committed_record_id

    # stage 1-5 again: amendment -> delta SUPERSEDING_CHANGE -> AUTO_COMMIT #
    mem.pipeline._extractor._llm = _mock(triplets=T_45)           # swap mock payload
    mem.pipeline._delta._llm = mem.pipeline._extractor._llm
    report2 = mem.pipeline.ingest_text("Res 511: 45 stories in Sector 4", "vendor_doc:v1")
    assert report2.summary()["by_disposition"] == {"AUTO_COMMIT": 1}
    v1 = mem.heads.get(KEY).record_id
    assert v1 != v0 and mem.store.get(v1).obj == "45 stories"

    # stage 6: bounded cascade flipped the direct dependent ------------- #
    assert mem.dag.get_state(v0) is NodeState.ARCHIVED
    assert mem.dag.get_state(child_id) is NodeState.TBD
    assert mem.queue.pending(kind=QueueItemType.REDERIVE)

    # Path B: PLANFENCE on the child (stale — cites v0) ---------------- #
    res = mem.planfence.validate(child_id, [KEY], replanned=False)
    assert res.outcome is ValidationOutcome.REPLAN_REQUIRED
    assert res.heads[KEY] == v1 and res.frontier[KEY] == v0

    # retrieval never serves the archived / TBD nodes ---------------- #
    ctx_ids = {n for n, _ in mem.retriever.context(v1, threshold=10.0)}
    assert v0 not in ctx_ids and child_id not in ctx_ids

    # persistence: cold restart sees the same head + states ---------- #
    mem2 = build_memory_layer(provider=_mock(triplets=T_45), data_dir=str(tmp_path / "kb"))
    assert mem2.heads.get(KEY).record_id == v1
    assert mem2.dag.get_state(v0) is NodeState.ARCHIVED

    # revert to v0 (T2 -> SYSTEM_2), head moves back --------------- #
    from dag_kb import Actor
    mem2.reverter.revert(KEY, v0, actor=Actor.SYSTEM_2)
    assert mem2.heads.get(KEY).record_id == v0
    assert mem2.dag.get_state(v0) is NodeState.ACTIVE


# --------------------------------------------------------------------------- #
# 1b. commits carry real tier-confidence edge distances (weighted-DAG retrieval)
# --------------------------------------------------------------------------- #
def test_committed_edges_carry_tier_distance_not_default(tmp_path):
    mem = build_memory_layer(provider=_mock(triplets=T_30), data_dir=str(tmp_path / "kb"))
    mem.pipeline.ingest_text("Res 402: 30 stories in Sector 4", "vendor_doc:v1")
    v0 = mem.heads.get(KEY).record_id
    record = mem.store.get(v0)

    # the SUBJECT edge (sector_4 -> relation node) was set by the gate's
    # real compute_distance(tier) call, not add_record's 1.0 default
    edge = mem.dag.nx.get_edge_data("sector_4", v0)
    assert edge is not None and edge["distance"] != 1.0
    assert edge["distance"] == pytest.approx(compute_distance(record.tier), rel=1e-3)

    # a T2 (vendor_doc) fact is always closer than a T4 (conversation) one --
    # tier confidence alone, no time dependence left
    assert compute_distance(Tier.T2) < compute_distance(Tier.T4)


def test_corroboration_tightens_distance_across_independent_commits(tmp_path):
    # two independent "patients" both presenting with Fever, committed one
    # at a time through the real gate -- the second, corroborating commit
    # must land a shorter (tighter) distance than the first, uncorroborated one.
    row_a = [("Patient A", "PERSON", "Presents_With", "Fever", "CONDITION")]
    row_b = [("Patient B", "PERSON", "Presents_With", "Fever", "CONDITION")]

    mem = build_memory_layer(provider=_mock(triplets=row_a), data_dir=str(tmp_path / "kb"))
    mem.pipeline.ingest_text("Patient A presented with fever.", "conversation:a")
    key_a = semantic_key("Patient A", "Presents_With", cardinality=Cardinality.MANY, obj="Fever")
    v_a = mem.heads.get(key_a).record_id
    d_a = mem.dag.nx.get_edge_data(v_a, "fever")["distance"]
    assert d_a == pytest.approx(compute_distance(Tier.T4))   # first assertion -- uncorroborated

    mem.pipeline._extractor._llm = _mock(triplets=row_b)
    mem.pipeline._delta._llm = mem.pipeline._extractor._llm
    mem.pipeline.ingest_text("Patient B presented with fever.", "conversation:b")
    key_b = semantic_key("Patient B", "Presents_With", cardinality=Cardinality.MANY, obj="Fever")
    v_b = mem.heads.get(key_b).record_id
    d_b = mem.dag.nx.get_edge_data(v_b, "fever")["distance"]

    assert d_b < d_a   # patient B's fact is now corroborated by patient A's
    assert d_b == pytest.approx(compute_distance(Tier.T4, corroboration_count=1))


# --------------------------------------------------------------------------- #
# 2. guardrail matrix — malformed LLM output never auto-commits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("responses, expect_key_committed", [
    # extractor returns an error channel (a dict, not a list)
    ({"INPUT TEXT": json.dumps({"error": "cannot comply"})}, False),
    # extractor returns prose, not a Python literal list
    ({"INPUT TEXT": "here are your triplets:"}, False),
    # extractor unavailable
    ({"INPUT TEXT": UNAVAILABLE}, False),
    # extractor emits a null-ish object value
    ({"INPUT TEXT": nvidia_triplets(
        [("Sector 4", "Zone", "max building height", "N/A", "METRIC")])}, False),
])
def test_guardrails_fail_closed(responses, expect_key_committed):
    mem = build_memory_layer(provider=MockProvider(responses, default=""))
    report = mem.pipeline.ingest_text("anything about Sector 4", "vendor_doc:v1")
    dispositions = {r.disposition for r in report.results}
    assert Disposition.AUTO_COMMIT not in dispositions
    assert Disposition.REJECT_FAIL_CLOSED in dispositions
    assert mem.queue.pending(routing=Routing.SYSTEM_2)             # routed for a human
    assert (mem.heads.get(KEY) is not None) is expect_key_committed


def test_delta_garbage_routes_to_system_2_not_commit():
    # first fact commits fine; the amendment's delta call returns an invalid label
    mem = build_memory_layer(provider=MockProvider({
        "INPUT TEXT": nvidia_triplets(T_30),
        "Compare the NEW fact": json.dumps({"delta": "PROBABLY"}),
    }, default=""))
    mem.pipeline.ingest_text("Res 402: 30 stories", "vendor_doc:v1")
    v0 = mem.heads.get(KEY).record_id

    mem.pipeline._extractor._llm = MockProvider({
        "INPUT TEXT": nvidia_triplets(T_45),
        "Compare the NEW fact": json.dumps({"delta": "PROBABLY"}),
    }, default="")
    mem.pipeline._delta._llm = mem.pipeline._extractor._llm
    report = mem.pipeline.ingest_text("Res 511: 45 stories", "vendor_doc:v1")

    assert Disposition.REJECT_FAIL_CLOSED in {r.disposition for r in report.results}
    assert mem.heads.get(KEY).record_id == v0                     # unchanged
