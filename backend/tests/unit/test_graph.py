"""Phase 2 -- R2.* : frontier full-closure, missing intermediate, cycle, active view, GraphML."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from dag_kb import LineageIncomplete, NotADag
from dag_kb.graph import KnowledgeDAG
from dag_kb.ids import compute_record_id
from dag_kb.store import InMemoryRecordStore
from dag_kb import Actor, EdgeType, NodeState, Record, Tier


def rec(key, seq, parents=()):
    rid = compute_record_id(semantic_key=key, predicate="p", obj=seq,
                            parent_ids=parents, owner="o", owner_seq=seq)
    return Record(record_id=rid, semantic_key=key, predicate="p", obj=seq,
                  parent_ids=tuple(parents), owner="o", owner_seq=seq, tier=Tier.T2,
                  auto_update=True, record_type="derived" if parents else "asserted",
                  valid_from=datetime.now(timezone.utc), source_id="s",
                  raw_citation="c", txn_id="t")


def test_frontier_depth_8_and_missing_intermediate():
    store = InMemoryRecordStore()
    dag = KnowledgeDAG(store)
    chain = []
    prev = None
    for i in range(9):  # keys k0..k8, each derived from the previous
        r = rec(f"k{i}", i, parents=(prev,) if prev else ())
        store.put(r)
        dag.add_record(r, state=NodeState.ACTIVE)
        chain.append(r)
        prev = r.record_id

    root = chain[-1].record_id
    f = dag.frontier(root, [f"k{i}" for i in range(9)])
    assert f["k0"] == chain[0].record_id and f["k8"] == chain[8].record_id

    # remove an intermediate node -> lineage incomplete
    dag.remove_node(chain[4].record_id)
    with pytest.raises(LineageIncomplete):
        dag.frontier(root, ["k0"])


def test_cycle_rejected():
    store = InMemoryRecordStore()
    dag = KnowledgeDAG(store)
    a = rec("a", 0)
    store.put(a)
    dag.add_record(a, state=NodeState.ACTIVE)
    b = rec("b", 0, parents=(a.record_id,))
    store.put(b)
    dag.add_record(b, state=NodeState.ACTIVE)
    with pytest.raises(NotADag):
        dag.add_edge_typed(a.record_id, b.record_id, EdgeType.DERIVED_FROM)


def test_active_view_excludes_non_active():
    store = InMemoryRecordStore()
    dag = KnowledgeDAG(store)
    a = rec("a", 0)
    store.put(a)
    dag.add_record(a, state=NodeState.ACTIVE)
    b = rec("b", 0)
    store.put(b)
    dag.add_record(b, state=NodeState.ACTIVE)
    dag.set_state(b.record_id, NodeState.ARCHIVED, actor=Actor.SYSTEM_1, reason="x")
    view = dag.active_view()
    assert a.record_id in view and b.record_id not in view


def test_graphml_round_trip_preserves_state_and_tier(tmp_path):
    store = InMemoryRecordStore()
    dag = KnowledgeDAG(store)
    a = rec("a", 0)
    store.put(a)
    dag.add_record(a, state=NodeState.ACTIVE)
    b = rec("b", 0, parents=(a.record_id,))
    store.put(b)
    dag.add_record(b, state=NodeState.ACTIVE)
    dag.set_state(a.record_id, NodeState.ARCHIVED, actor=Actor.SYSTEM_1, reason="x")

    p = os.path.join(tmp_path, "dag.graphml")
    dag.write_graphml(p)
    dag2 = KnowledgeDAG.read_graphml(p, store)
    assert dag2.get_state(a.record_id) is NodeState.ARCHIVED
    assert dag2.get_state(b.record_id) is NodeState.ACTIVE
    assert dag2.nx.nodes[b.record_id]["tier"] is Tier.T2


def test_illegal_transition_blocked():
    store = InMemoryRecordStore()
    dag = KnowledgeDAG(store)
    a = rec("a", 0)
    store.put(a)
    dag.add_record(a, state=NodeState.ACTIVE)
    dag.set_state(a.record_id, NodeState.DELETED, actor=Actor.SYSTEM_2, reason="x")
    with pytest.raises(NotADag):  # DELETED -> TBD is not allowed
        dag.set_state(a.record_id, NodeState.TBD, actor=Actor.SYSTEM_1, reason="x")


# --------------------------------------------------------------------------- #
# corroboration_count -- distinct-subject in-degree feeding graph/distance.py
# --------------------------------------------------------------------------- #
def _relation(subject_ref, predicate, object_ref, *, seq):
    key = f"{subject_ref}::{predicate.lower()}::{object_ref}"
    rid = compute_record_id(semantic_key=key, predicate=predicate, obj=object_ref,
                            parent_ids=(), owner=key, owner_seq=seq)
    return Record(record_id=rid, semantic_key=key, predicate=predicate, obj=object_ref,
                  parent_ids=(), owner=key, owner_seq=seq, tier=Tier.T4, auto_update=True,
                  record_type="asserted", valid_from=datetime.now(timezone.utc),
                  source_id="s", raw_citation="c", txn_id="t",
                  subject_ref=subject_ref, object_ref=object_ref)


def test_corroboration_count_unknown_entity_is_zero():
    dag = KnowledgeDAG(InMemoryRecordStore())
    assert dag.corroboration_count("nonexistent") == 0


def test_corroboration_count_counts_distinct_subjects_only():
    store = InMemoryRecordStore()
    dag = KnowledgeDAG(store)
    facts = [
        ("patient_1", "Presents_With", "fever"),
        ("patient_2", "Presents_With", "fever"),
        ("patient_2", "Diagnosed_With", "fever"),   # same subject, 2nd relation -- no double count
        ("patient_3", "Presents_With", "fever"),
    ]
    for i, (subj, pred, obj) in enumerate(facts):
        r = _relation(subj, pred, obj, seq=i)
        store.put(r)
        dag.add_record(r, state=NodeState.ACTIVE)
    assert dag.corroboration_count("fever") == 3   # patient_1, patient_2, patient_3
    assert dag.corroboration_count("patient_1") == 0   # nothing points AT a patient


def test_corroboration_count_ignores_non_active_relations():
    store = InMemoryRecordStore()
    dag = KnowledgeDAG(store)
    r1 = _relation("patient_1", "Presents_With", "fever", seq=0)
    r2 = _relation("patient_2", "Presents_With", "fever", seq=1)
    store.put(r1); dag.add_record(r1, state=NodeState.ACTIVE)
    store.put(r2); dag.add_record(r2, state=NodeState.ACTIVE)
    assert dag.corroboration_count("fever") == 2
    dag.set_state(r2.record_id, NodeState.ARCHIVED, actor=Actor.SYSTEM_1, reason="superseded")
    assert dag.corroboration_count("fever") == 1   # archived subject no longer counted
