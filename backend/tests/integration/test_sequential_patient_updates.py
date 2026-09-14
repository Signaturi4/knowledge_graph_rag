"""Integration test: Sequential multi-document patient update sequences.

Validates deterministic memory layer behavior across temporal EMR document streams:
- Authority tier lattice: T1 Physician > T2 Nurse (Nurse cannot supersede Physician).
- Medication protocol switching: old regimen is ARCHIVED, new regimen is ACTIVE with SUPERSEDES edge.
- Malformed extraction fail-closed: bad extractor output routes to DEAD_LETTER without corrupting DAG.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

from dag_kb import Delta, Disposition, EdgeType, NodeState, QueueItemType, Tier
from dag_kb.store import InMemoryRecordStore
from llm.mock import MockProvider
from pipeline.stage0_sources import RawItem, SourcePolicy, SourceRegistry
from service.assembly import build_memory_layer

CLINICAL_POLICIES = {
    "Primary_Care_Physician": SourcePolicy(Tier.T1, True, "clinical_physician"),
    "ED_Attending_Physician": SourcePolicy(Tier.T1, True, "clinical_physician"),
    "General_Surgeon": SourcePolicy(Tier.T1, True, "clinical_specialist"),
    "Triage_Nurse": SourcePolicy(Tier.T2, True, "clinical_triage"),
    "Laboratory_System": SourcePolicy(Tier.T1, True, "clinical_lab"),
}


def test_sequential_medication_supersession():
    """Verify PT-1049 trajectory: Metformin initiated -> Metformin discontinued & Semaglutide started."""
    registry = SourceRegistry(CLINICAL_POLICIES)
    mem = build_memory_layer(registry=registry)

    # Document 1: Prescribe Metformin
    doc1_triplets = (
        "[('PT-1049', 'PERSON', 'Diagnosed_With', 'T2DM', 'CONDITION'), "
        "('PT-1049', 'PERSON', 'Active_Treatment', 'Metformin', 'TREATMENT')]"
    )
    mem.pipeline._extractor._llm = MockProvider(default=doc1_triplets)
    mem.pipeline._delta._llm = MockProvider(default='{"delta": "SUPERSEDING_CHANGE", "rationale": "new Rx"}')

    raw1 = RawItem(
        text="Initiating Metformin 500mg BID for T2DM.",
        tier=Tier.T1,
        auto_update=True,
        source_id="DOC-C1-003",
        valid_from=datetime(2024, 3, 18, 14, 30, tzinfo=timezone.utc),
        canonical_subject="PT-1049",
    )
    rep1 = mem.pipeline.ingest_raw(raw1)
    assert rep1.drafts == 2
    assert all(r.disposition == Disposition.AUTO_COMMIT for r in rep1.results)

    head_rx1 = mem.heads.get("pt_1049::active_treatment")
    assert head_rx1 is not None
    assert mem.dag.get_state(head_rx1.record_id) == NodeState.ACTIVE
    rec_metformin_id = head_rx1.record_id

    # Document 2: Discontinue Metformin, Start Semaglutide
    doc2_triplets = "[('PT-1049', 'PERSON', 'Active_Treatment', 'Semaglutide', 'TREATMENT')]"
    mem.pipeline._extractor._llm = MockProvider(default=doc2_triplets)

    raw2 = RawItem(
        text="Discontinue Metformin due to side effects. Start Semaglutide.",
        tier=Tier.T1,
        auto_update=True,
        source_id="DOC-C1-004",
        valid_from=datetime(2024, 9, 20, 10, 0, tzinfo=timezone.utc),
        canonical_subject="PT-1049",
    )
    rep2 = mem.pipeline.ingest_raw(raw2)
    assert rep2.drafts == 1
    assert rep2.results[0].disposition == Disposition.AUTO_COMMIT

    head_rx2 = mem.heads.get("pt_1049::active_treatment")
    assert head_rx2.record_id != rec_metformin_id
    rec_semaglutide_id = head_rx2.record_id

    # Verify state transitions and edges
    assert mem.dag.get_state(rec_semaglutide_id) == NodeState.ACTIVE
    assert mem.dag.get_state(rec_metformin_id) == NodeState.ARCHIVED

    # Verify SUPERSEDES edge in the graph
    edges = [
        (u, v, d["kind"])
        for u, v, d in mem.dag.nx.edges(data=True)
        if d.get("kind") == EdgeType.SUPERSEDES.value
    ]
    assert (rec_semaglutide_id, rec_metformin_id, EdgeType.SUPERSEDES.value) in edges


def test_sequential_authority_tier_rejection():
    """Verify PT-3392 trajectory: Nurse (T2) cannot overwrite Attending Physician (T1)."""
    registry = SourceRegistry(CLINICAL_POLICIES)
    mem = build_memory_layer(registry=registry)

    # Document 1: Attending Physician (Tier T1) diagnoses Viral Gastroenteritis
    doc1_triplets = "[('PT-3392', 'PERSON', 'Current_Diagnosis', 'Viral Gastroenteritis', 'CONDITION')]"
    mem.pipeline._extractor._llm = MockProvider(default=doc1_triplets)
    mem.pipeline._delta._llm = MockProvider(default='{"delta": "SUPERSEDING_CHANGE", "rationale": "update"}')

    raw1 = RawItem(
        text="Diagnosis: Viral Gastroenteritis.",
        tier=Tier.T1,
        auto_update=True,
        source_id="DOC-A1-002",
        valid_from=datetime(2026, 5, 10, 19, 45, tzinfo=timezone.utc),
        canonical_subject="PT-3392",
    )
    rep1 = mem.pipeline.ingest_raw(raw1)
    assert rep1.results[0].disposition == Disposition.AUTO_COMMIT
    initial_diagnosis_id = rep1.results[0].committed_record_id

    # Document 2: Triage Nurse (Tier T2) notes different impression
    doc2_triplets = "[('PT-3392', 'PERSON', 'Current_Diagnosis', 'Appendicitis Suspected', 'CONDITION')]"
    mem.pipeline._extractor._llm = MockProvider(default=doc2_triplets)

    raw2 = RawItem(
        text="Patient returns with RLQ pain. Appendicitis suspected.",
        tier=Tier.T2,  # Weaker authority tier
        auto_update=True,
        source_id="DOC-A1-003",
        valid_from=datetime(2026, 5, 11, 6, 10, tzinfo=timezone.utc),
        canonical_subject="PT-3392",
    )
    rep2 = mem.pipeline.ingest_raw(raw2)
    # Must be rejected because T2 < T1 authority!
    assert rep2.results[0].disposition == Disposition.NO_OP
    assert "precedence: reject" in rep2.results[0].detail

    # Head must remain the Attending Physician's diagnosis
    head = mem.heads.get("pt_3392::current_diagnosis")
    assert head.record_id == initial_diagnosis_id


def test_malformed_extraction_fail_closed():
    """Verify PT-7714 fail-closed behavior on unparseable/truncated LLM response."""
    registry = SourceRegistry(CLINICAL_POLICIES)
    mem = build_memory_layer(registry=registry)

    # Broken output from model (truncated list)
    mem.pipeline._extractor._llm = MockProvider(default="[('PT-7714', 'PERSON', 'Diagnosed_With'")

    raw = RawItem(
        text="The prior diagnosis is revoked.",
        tier=Tier.T1,
        auto_update=True,
        source_id="DOC-E1-006",
        valid_from=datetime(2026, 7, 28, 14, 0, tzinfo=timezone.utc),
        canonical_subject="PT-7714",
    )
    rep = mem.pipeline.ingest_raw(raw)
    assert rep.drafts == 0
    assert rep.results[0].disposition == Disposition.REJECT_FAIL_CLOSED

    # Work queue must carry DEAD_LETTER
    dead_letters = mem.queue.pending(kind=QueueItemType.DEAD_LETTER)
    assert len(dead_letters) == 1
    assert dead_letters[0].semantic_key == "<extract:DOC-E1-006>"
