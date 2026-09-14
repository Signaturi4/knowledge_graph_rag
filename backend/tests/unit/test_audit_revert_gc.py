"""Phases 7 & 6 -- R7.* revert, R6.* garbage collection."""
from __future__ import annotations

from dag_kb import Actor, Delta, NodeState, Tier


def _commit_chain(mem, draft):
    """v0(30) -> v1(45) -> v2(60) on an auto T2 key, with a T3 dependent."""
    r0 = mem.gate.submit(draft(key="z::lim", obj=30, tier=Tier.T2, auto=True,
                               source_id="vendor_doc:v"),
                         Delta.SUPERSEDING_CHANGE, writer_identity="vendor_doc:v")
    dep = draft(key="z::use", obj="ok", tier=Tier.T3, auto=True, source_id="derived:d",
                parents=(r0.committed_record_id,))
    mem.gate.submit(dep, Delta.SUPERSEDING_CHANGE, writer_identity="derived:d")
    r1 = mem.gate.submit(draft(key="z::lim", obj=45, tier=Tier.T2, auto=True, days_ago=-1,
                               source_id="vendor_doc:v"),
                         Delta.SUPERSEDING_CHANGE, writer_identity="vendor_doc:v")
    r2 = mem.gate.submit(draft(key="z::lim", obj=60, tier=Tier.T2, auto=True, days_ago=-2,
                               source_id="vendor_doc:v"),
                         Delta.SUPERSEDING_CHANGE, writer_identity="vendor_doc:v")
    return r0.committed_record_id, r1.committed_record_id, r2.committed_record_id


def test_revert_moves_head_and_flags_dependents(mem, draft):
    v0, v1, v2 = _commit_chain(mem, draft)
    assert mem.heads.get("z::lim").record_id == v2

    mem.reverter.revert("z::lim", v0, actor=Actor.SYSTEM_2)
    assert mem.heads.get("z::lim").record_id == v0
    assert mem.dag.get_state(v0) is NodeState.ACTIVE
    assert mem.dag.get_state(v2) is NodeState.ARCHIVED
    # audit recorded the reinstatement
    assert any(r.to_state is NodeState.ACTIVE and r.actor is Actor.SYSTEM_2
               for r in mem.audit.for_node(v0))


def test_revert_of_t0_requires_system_2(mem, draft):
    import pytest

    from dag_kb import RevertRejected

    mem.gate.submit(draft(obj=30), Delta.SUPERSEDING_CHANGE, writer_identity="municipal:res-402")
    r1 = mem.gate.submit(draft(obj=45, days_ago=-1), Delta.SUPERSEDING_CHANGE,
                         writer_identity="municipal:res-402")
    # (T0 supersession is queued, not committed) -> commit via approval to get an ARCHIVED v0
    approved = mem.gate.approve_review(r1.queue_item.id)
    v0 = mem.store.all_versions("sector_4::zoning_limit")[0].record_id
    with pytest.raises(RevertRejected):
        mem.reverter.revert("sector_4::zoning_limit", v0, actor=Actor.SYSTEM_1)
    mem.reverter.revert("sector_4::zoning_limit", v0, actor=Actor.SYSTEM_2)  # ok
    assert mem.heads.get("sector_4::zoning_limit").record_id == v0


def test_revert_txn_rolls_back_a_whole_ingest(mem, draft):
    """Two keys committed under one txn_id, both roll back to their pre-txn head."""
    from dag_kb import Actor

    txn = "txn_shared"
    a0 = mem.gate.submit(draft(key="z::a", obj=1, tier=Tier.T2, auto=True, source_id="vendor_doc:v"),
                         Delta.SUPERSEDING_CHANGE, writer_identity="vendor_doc:v", txn_id="txn_seed")
    b0 = mem.gate.submit(draft(key="z::b", obj=1, tier=Tier.T2, auto=True, source_id="vendor_doc:v"),
                         Delta.SUPERSEDING_CHANGE, writer_identity="vendor_doc:v", txn_id="txn_seed")
    # one bad ingest bumps both keys under txn
    mem.gate.submit(draft(key="z::a", obj=2, tier=Tier.T2, auto=True, days_ago=-1, source_id="vendor_doc:v"),
                    Delta.SUPERSEDING_CHANGE, writer_identity="vendor_doc:v", txn_id=txn)
    mem.gate.submit(draft(key="z::b", obj=2, tier=Tier.T2, auto=True, days_ago=-1, source_id="vendor_doc:v"),
                    Delta.SUPERSEDING_CHANGE, writer_identity="vendor_doc:v", txn_id=txn)
    assert mem.store.get(mem.heads.get("z::a").record_id).obj == 2

    affected = mem.reverter.revert_txn(txn, actor=Actor.SYSTEM_2)
    assert set(affected) == {"z::a", "z::b"}
    assert mem.store.get(mem.heads.get("z::a").record_id).obj == 1
    assert mem.store.get(mem.heads.get("z::b").record_id).obj == 1


def test_commit_rolls_back_on_head_conflict(mem, draft):
    """If heads.update raises mid-commit, the DAG node is removed and prev stays ACTIVE."""
    from dag_kb import HeadConflict, NodeState as NS

    r0 = mem.gate.submit(draft(key="z::c", obj=1, tier=Tier.T2, auto=True, source_id="vendor_doc:v"),
                         Delta.SUPERSEDING_CHANGE, writer_identity="vendor_doc:v")
    v0 = r0.committed_record_id

    # make the next heads.update fail
    orig = mem.heads.update
    mem.heads.update = lambda *a, **k: (_ for _ in ()).throw(HeadConflict("boom"))
    try:
        with __import__("pytest").raises(HeadConflict):
            mem.gate.submit(draft(key="z::c", obj=2, tier=Tier.T2, auto=True, days_ago=-1,
                                  source_id="vendor_doc:v"),
                            Delta.SUPERSEDING_CHANGE, writer_identity="vendor_doc:v")
    finally:
        mem.heads.update = orig

    assert mem.heads.get("z::c").record_id == v0        # head unchanged
    assert mem.dag.get_state(v0) is NS.ACTIVE           # prev not archived
    # no orphan ACTIVE node for the failed candidate
    assert len([n for n in mem.dag.nodes(NS.ACTIVE)
                if mem.dag.nx.nodes[n]["semantic_key"] == "z::c"]) == 1


def test_gc_evicts_unlocked_superseded_past_retention(mem, draft):
    mem.gc._retention = __import__("datetime").timedelta(seconds=-1)  # everything is "old"
    v0, v1, v2 = _commit_chain(mem, draft)

    # v0 is locked: its direct dependent z::use is TBD after the first supersession
    evicted = mem.gc.sweep()
    assert v0 not in evicted            # locked by TBD dependent
    assert v1 in evicted               # superseded, unlocked, old
    assert not mem.dag.has_node(v1)
    assert mem.store.is_cold(v1)
