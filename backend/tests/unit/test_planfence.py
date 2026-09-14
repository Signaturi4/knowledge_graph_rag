"""Phase 3 -- R3.* : PLANFENCE reproduces the paper's Table 1 stale-plan result."""
from __future__ import annotations

from datetime import datetime, timezone

from dag_kb.graph import KnowledgeDAG
from dag_kb import HeadIndex
from dag_kb.ids import compute_record_id
from dag_kb import PlanFence
from dag_kb.store import InMemoryRecordStore
from dag_kb import HeadEntry, NodeState, Record, Tier, ValidationOutcome


def rec(key, seq, obj, parents=()):
    rid = compute_record_id(semantic_key=key, predicate="p", obj=obj,
                            parent_ids=parents, owner="o", owner_seq=seq)
    return Record(record_id=rid, semantic_key=key, predicate="p", obj=obj,
                  parent_ids=tuple(parents), owner="o", owner_seq=seq, tier=Tier.T1,
                  auto_update=True, record_type="derived" if parents else "asserted",
                  valid_from=datetime.now(timezone.utc), source_id="s",
                  raw_citation="c", txn_id="t")


def _fixture():
    store = InMemoryRecordStore()
    dag = KnowledgeDAG(store)
    heads = HeadIndex()
    pf = PlanFence(dag, heads)

    r_v1 = rec("requirement", 0, "r_v1")
    store.put(r_v1)
    dag.add_record(r_v1, state=NodeState.ACTIVE)
    heads.update(HeadEntry("requirement", r_v1.record_id, "o", 0), writer_identity="o")

    plan = rec("plan", 0, "p(r_v1)", parents=(r_v1.record_id,))
    store.put(plan)
    dag.add_record(plan, state=NodeState.ACTIVE)
    return store, dag, heads, pf, r_v1, plan


def test_fresh_state_but_stale_plan_requires_replan():
    store, dag, heads, pf, r_v1, plan = _fixture()

    # requirement revised: r_v2 becomes the head; plan still cites r_v1
    r_v2 = rec("requirement", 1, "r_v2")
    store.put(r_v2)
    dag.add_record(r_v2, state=NodeState.ACTIVE)
    dag.set_state(r_v1.record_id, NodeState.ARCHIVED, actor=__import__("dag_kb").Actor.SYSTEM_1,
                  reason="superseded")
    heads.update(HeadEntry("requirement", r_v2.record_id, "o", 1), writer_identity="o")

    # a freshness-only check ("is r_v2 the head? yes") would PASS. PLANFENCE must not.
    res = pf.validate(plan.record_id, ["requirement"], replanned=False)
    assert res.outcome is ValidationOutcome.REPLAN_REQUIRED
    assert res.mismatched_keys == ("requirement",)
    assert res.heads["requirement"] == r_v2.record_id

    # re-bind the plan to r_v2, validate again -> AUTHORIZED
    plan2 = rec("plan", 1, "p(r_v2)", parents=(r_v2.record_id,))
    store.put(plan2)
    dag.add_record(plan2, state=NodeState.ACTIVE)
    res2 = pf.validate(plan2.record_id, ["requirement"], replanned=True)
    assert res2.outcome is ValidationOutcome.AUTHORIZED

    # a SECOND revision before re-validation -> BLOCKED, no replan loop
    r_v3 = rec("requirement", 2, "r_v3")
    store.put(r_v3)
    dag.add_record(r_v3, state=NodeState.ACTIVE)
    heads.update(HeadEntry("requirement", r_v3.record_id, "o", 2), writer_identity="o")
    res3 = pf.validate(plan2.record_id, ["requirement"], replanned=True)
    assert res3.outcome is ValidationOutcome.BLOCKED


def test_missing_head_blocks():
    store, dag, heads, pf, r_v1, plan = _fixture()
    res = pf.validate(plan.record_id, ["requirement", "nonexistent"], replanned=False)
    assert res.outcome is ValidationOutcome.BLOCKED


def test_incomplete_lineage_blocks():
    store, dag, heads, pf, r_v1, plan = _fixture()
    dag.remove_node(r_v1.record_id)  # break the plan's lineage
    res = pf.validate(plan.record_id, ["requirement"], replanned=False)
    assert res.outcome is ValidationOutcome.BLOCKED
