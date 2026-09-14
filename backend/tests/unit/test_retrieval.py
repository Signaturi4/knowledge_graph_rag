"""Phase 5 -- R5.1 / R5.2 : bounded Dijkstra retrieval over the ACTIVE view.

Reproduces the PRD section 3 reference numbers.
"""
from __future__ import annotations

from datetime import datetime, timezone

from dag_kb import Actor, BoundedRetriever, KnowledgeDAG, NodeState, Record, Tier
from dag_kb.ids import compute_record_id
from dag_kb.store import InMemoryRecordStore


def _rec(key):
    rid = compute_record_id(semantic_key=key, predicate="p", obj=key,
                            parent_ids=(), owner="o", owner_seq=0)
    return Record(record_id=rid, semantic_key=key, predicate="p", obj=key, parent_ids=(),
                  owner="o", owner_seq=0, tier=Tier.T2, auto_update=True,
                  record_type="asserted", valid_from=datetime.now(timezone.utc),
                  source_id="s", raw_citation="c", txn_id="t")


def test_threshold_excludes_far_nodes_prd_section_3():
    store = InMemoryRecordStore()
    dag = KnowledgeDAG(store)
    ids = {}
    for k in ("root", "A", "B", "C", "D"):
        r = _rec(k)
        store.put(r)
        dag.add_record(r, state=NodeState.ACTIVE)
        ids[k] = r.record_id

    # PRD weights: root->A 1.5, root->B 4.0, A->C 2.0, C->D 1.0
    dag.add_edge_typed(ids["root"], ids["A"], __import__("dag_kb").EdgeType.DERIVED_FROM, distance=1.5)
    dag.add_edge_typed(ids["root"], ids["B"], __import__("dag_kb").EdgeType.DERIVED_FROM, distance=4.0)
    dag.add_edge_typed(ids["A"], ids["C"], __import__("dag_kb").EdgeType.DERIVED_FROM, distance=2.0)
    dag.add_edge_typed(ids["C"], ids["D"], __import__("dag_kb").EdgeType.DERIVED_FROM, distance=1.0)

    ctx = dict(BoundedRetriever(dag).context(ids["root"], threshold=4.0))
    got = {k: round(ctx.get(ids[k]), 2) for k in ("root", "A", "B", "C") if ids[k] in ctx}
    assert got == {"root": 0.0, "A": 1.5, "C": 3.5, "B": 4.0}
    assert ids["D"] not in ctx                       # distance 4.5 > threshold


def test_archived_nodes_never_in_context():
    store = InMemoryRecordStore()
    dag = KnowledgeDAG(store)
    root, arch = _rec("root2"), _rec("arch")
    for r in (root, arch):
        store.put(r)
        dag.add_record(r, state=NodeState.ACTIVE)
    dag.add_edge_typed(root.record_id, arch.record_id, __import__("dag_kb").EdgeType.DERIVED_FROM, distance=1.0)
    dag.set_state(arch.record_id, NodeState.ARCHIVED, actor=Actor.SYSTEM_1, reason="x")
    ctx = BoundedRetriever(dag).context(root.record_id, threshold=10.0)
    assert arch.record_id not in dict(ctx)
