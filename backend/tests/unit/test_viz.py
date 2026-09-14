"""Visualization: pyvis HTML render + node-link JSON export."""
from __future__ import annotations

from datetime import datetime, timezone

from dag_kb import EdgeType, KnowledgeDAG, NodeState, Record, Tier
from dag_kb.ids import compute_record_id
from dag_kb.store import InMemoryRecordStore
from viz.pyvis_render import _edge_length, graph_json, render_html


def _rec(key):
    rid = compute_record_id(semantic_key=key, predicate="p", obj=key,
                            parent_ids=(), owner="o", owner_seq=0)
    return Record(record_id=rid, semantic_key=key, predicate="p", obj=key, parent_ids=(),
                  owner="o", owner_seq=0, tier=Tier.T2, auto_update=True,
                  record_type="asserted", valid_from=datetime.now(timezone.utc),
                  source_id="s", raw_citation="c", txn_id="t")


def _dag():
    store = InMemoryRecordStore()
    dag = KnowledgeDAG(store)
    a, b = _rec("a"), _rec("b")
    store.put(a); dag.add_record(a, state=NodeState.ACTIVE)
    store.put(b); dag.add_record(b, state=NodeState.ACTIVE)
    return dag, a, b


def test_render_html_contains_nodes_and_vis_network():
    dag, a, b = _dag()
    html = render_html(dag, physics=False)
    assert "<html" in html.lower()
    assert a.record_id in html and b.record_id in html
    assert "vis-network" in html.lower() or "network" in html.lower()


def test_graph_json_shape():
    dag, a, b = _dag()
    data = graph_json(dag)
    ids = {n["id"] for n in data["nodes"]}
    assert {a.record_id, b.record_id} <= ids
    assert all(n["state"] == "ACTIVE" for n in data["nodes"])


def test_graph_json_exposes_edge_distance():
    dag, a, b = _dag()
    dag.add_edge_typed(a.record_id, b.record_id, EdgeType.DERIVED_FROM, distance=3.0)
    data = graph_json(dag)
    edge = next(e for e in data["edges"] if e["source"] == a.record_id and e["target"] == b.record_id)
    assert edge["distance"] == 3.0


def test_edge_length_scales_with_distance_and_is_capped():
    tight, loose = _edge_length(1.0), _edge_length(5.0)
    assert tight < loose                        # closer (lower distance) -> shorter spring
    assert _edge_length(1000.0) <= 400.0         # capped, never flies off-screen
    assert _edge_length(0.0) > 0.0               # never collapses to zero-length


def test_render_html_carries_distance_into_edge_length():
    dag, a, b = _dag()
    dag.add_edge_typed(a.record_id, b.record_id, EdgeType.DERIVED_FROM, distance=5.0)
    html = render_html(dag, physics=False, include_bookkeeping_edges=True)
    assert f'"length": {_edge_length(5.0)}' in html or str(_edge_length(5.0)) in html
