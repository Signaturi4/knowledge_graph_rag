"""Phase 4 -- R4.* : one-replan-then-block orchestration."""
from __future__ import annotations

from datetime import datetime, timezone

from dag_kb import (
    Actor,
    HeadEntry,
    KnowledgeDAG,
    NodeState,
    PlanFence,
    Record,
    ReplanOrchestrator,
    Tier,
    ValidationOutcome,
)
from dag_kb.ids import compute_record_id
from dag_kb.store import HeadIndex, InMemoryRecordStore


def _rec(key, seq, obj, parents=()):
    rid = compute_record_id(semantic_key=key, predicate="p", obj=obj,
                            parent_ids=parents, owner="o", owner_seq=seq)
    return Record(record_id=rid, semantic_key=key, predicate="p", obj=obj,
                  parent_ids=tuple(parents), owner="o", owner_seq=seq, tier=Tier.T1,
                  auto_update=True, record_type="derived" if parents else "asserted",
                  valid_from=datetime.now(timezone.utc), source_id="s",
                  raw_citation="c", txn_id="t")


def _stack():
    store, dag, heads = InMemoryRecordStore(), None, HeadIndex()
    dag = KnowledgeDAG(store)
    r0 = _rec("req", 0, "v0")
    store.put(r0)
    dag.add_record(r0, state=NodeState.ACTIVE)
    heads.update(HeadEntry("req", r0.record_id, "o", 0), writer_identity="o")
    plan0 = _rec("plan", 0, "p(v0)", parents=(r0.record_id,))
    store.put(plan0)
    dag.add_record(plan0, state=NodeState.ACTIVE)
    return store, dag, heads, ReplanOrchestrator(PlanFence(dag, heads)), r0, plan0


def test_one_replan_then_authorized():
    store, dag, heads, orch, r0, plan0 = _stack()
    # requirement moves to v1
    r1 = _rec("req", 1, "v1")
    store.put(r1)
    dag.add_record(r1, state=NodeState.ACTIVE)
    dag.set_state(r0.record_id, NodeState.ARCHIVED, actor=Actor.SYSTEM_1, reason="x")
    heads.update(HeadEntry("req", r1.record_id, "o", 1), writer_identity="o")

    calls = {"n": 0}

    def recompute(fresh_heads):
        calls["n"] += 1
        new_plan = _rec("plan", calls["n"], "p(v1)", parents=(fresh_heads["req"],))
        store.put(new_plan)
        dag.add_record(new_plan, state=NodeState.ACTIVE)
        return new_plan.record_id

    result, root = orch.run(plan0.record_id, ["req"], recompute)
    assert result.outcome is ValidationOutcome.AUTHORIZED
    assert calls["n"] == 1                       # recompute invoked exactly once


def test_second_change_during_replan_blocks():
    store, dag, heads, orch, r0, plan0 = _stack()
    r1 = _rec("req", 1, "v1")
    store.put(r1); dag.add_record(r1, state=NodeState.ACTIVE)
    dag.set_state(r0.record_id, NodeState.ARCHIVED, actor=Actor.SYSTEM_1, reason="x")
    heads.update(HeadEntry("req", r1.record_id, "o", 1), writer_identity="o")

    def recompute(fresh_heads):
        # a THIRD version lands before the recomputed plan is validated
        r2 = _rec("req", 2, "v2")
        store.put(r2); dag.add_record(r2, state=NodeState.ACTIVE)
        heads.update(HeadEntry("req", r2.record_id, "o", 2), writer_identity="o")
        new_plan = _rec("plan", 9, "p(v1)", parents=(fresh_heads["req"],))
        store.put(new_plan); dag.add_record(new_plan, state=NodeState.ACTIVE)
        return new_plan.record_id

    result, _ = orch.run(plan0.record_id, ["req"], recompute)
    assert result.outcome is ValidationOutcome.BLOCKED


def test_already_valid_plan_skips_recompute():
    store, dag, heads, orch, r0, plan0 = _stack()
    called = []
    result, root = orch.run(plan0.record_id, ["req"], lambda h: called.append(1))
    assert result.outcome is ValidationOutcome.AUTHORIZED
    assert called == []
    assert root == plan0.record_id


def test_r44_propagates_to_downstream_dependents_on_success():
    """A successful replan flags direct dependents of the STALE root TBD and
    queues a REDERIVE item, so they don't silently keep citing a superseded root."""
    from dag_kb import Cascade, AuditLog, WorkQueue, QueueItemType

    store, dag, heads, _, r0, plan0 = _stack()
    audit = AuditLog()
    queue = WorkQueue()
    cascade = Cascade(dag, queue, audit)
    orch = ReplanOrchestrator(PlanFence(dag, heads), dag=dag, cascade=cascade)

    # a downstream node derived from plan0 (e.g. a report built on that plan)
    downstream = _rec("report", 0, "r(plan0)", parents=(plan0.record_id,))
    store.put(downstream)
    dag.add_record(downstream, state=NodeState.ACTIVE)

    r1 = _rec("req", 1, "v1")
    store.put(r1); dag.add_record(r1, state=NodeState.ACTIVE)
    dag.set_state(r0.record_id, NodeState.ARCHIVED, actor=Actor.SYSTEM_1, reason="x")
    heads.update(HeadEntry("req", r1.record_id, "o", 1), writer_identity="o")

    def recompute(fresh_heads):
        new_plan = _rec("plan", 1, "p(v1)", parents=(fresh_heads["req"],))
        store.put(new_plan)
        dag.add_record(new_plan, state=NodeState.ACTIVE)
        return new_plan.record_id

    result, new_root = orch.run(plan0.record_id, ["req"], recompute)
    assert result.outcome is ValidationOutcome.AUTHORIZED
    assert new_root != plan0.record_id

    assert dag.get_state(downstream.record_id) is NodeState.TBD
    assert queue.pending(kind=QueueItemType.REDERIVE)
