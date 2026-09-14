"""Cross-cutting -- persistence survives a process restart (STATUS.md P0 #2).

A fresh ``build_memory_layer(data_dir=...)`` must reload records + heads + the
DAG (node state + edges) so the second "process" sees exactly what the first
committed.
"""
from __future__ import annotations

from conftest import nvidia_triplets
from dag_kb import Delta, NodeState, Tier
from service.assembly import build_memory_layer
from llm.mock import MockProvider


def _prov():
    return MockProvider({
        "INPUT TEXT": nvidia_triplets([
            ("Sector 4", "Zone", "zoning limit", "30 stories", "METRIC"),
        ]),
    }, default="")


def test_restart_reloads_heads_and_dag_state(tmp_path):
    data_dir = str(tmp_path / "kb")

    # --- process 1: ingest + supersede ---
    m1 = build_memory_layer(provider=_prov(), data_dir=data_dir)
    m1.pipeline.ingest_text("res 402: 30 stories", "vendor_doc:v1")   # T2 auto
    key = "sector_4::zoning_limit"
    v0 = m1.heads.get(key).record_id
    # supersede within tier
    m1.gate.submit(
        __import__("dag_kb").TripletDraft(
            semantic_key=key, predicate="zoning limit", obj="45 stories",
            tier=Tier.T2, auto_update=True, source_id="vendor_doc:v1",
            raw_citation="res 511",
            valid_from=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        ),
        Delta.SUPERSEDING_CHANGE, writer_identity="vendor_doc:v1",
    )
    v1 = m1.heads.get(key).record_id
    assert v1 != v0
    assert m1.dag.get_state(v0) is NodeState.ARCHIVED

    # --- process 2: cold start on the same data_dir ---
    m2 = build_memory_layer(provider=_prov(), data_dir=data_dir)
    assert m2.heads.get(key).record_id == v1                 # head persisted
    assert m2.store.get(v0).obj == "30 stories"              # records persisted
    assert m2.dag.get_state(v1) is NodeState.ACTIVE          # DAG state reloaded
    assert m2.dag.get_state(v0) is NodeState.ARCHIVED        # not lost on restart
    # lineage edge survived
    assert v0 in [d for d in m2.dag.nx.successors(v1)]
