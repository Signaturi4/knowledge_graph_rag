"""Phase 9 -- R9.3-R9.8 : the reconciliation gate."""
from __future__ import annotations

from dag_kb import Delta, Disposition, NodeState, QueueItemType, Routing, Tier


def test_first_fact_auto_commits(mem, draft):
    r = mem.gate.submit(draft(), Delta.SUPERSEDING_CHANGE, writer_identity="municipal:res-402")
    assert r.disposition is Disposition.AUTO_COMMIT
    head = mem.heads.get("sector_4::zoning_limit")
    assert head and mem.store.get(head.record_id).obj == 30
    assert mem.dag.get_state(head.record_id) is NodeState.ACTIVE


def test_identical_is_noop(mem, draft):
    mem.gate.submit(draft(), Delta.SUPERSEDING_CHANGE, writer_identity="municipal:res-402")
    r = mem.gate.submit(draft(), Delta.IDENTICAL, writer_identity="municipal:res-402")
    assert r.disposition is Disposition.NO_OP


def test_t0_supersession_queues_for_system_2(mem, draft):
    mem.gate.submit(draft(obj=30), Delta.SUPERSEDING_CHANGE, writer_identity="municipal:res-402")
    r = mem.gate.submit(draft(obj=45, days_ago=-1), Delta.SUPERSEDING_CHANGE,
                        writer_identity="municipal:res-402")
    assert r.disposition is Disposition.QUEUE_SYSTEM_2
    assert mem.heads.get("sector_4::zoning_limit").owner_seq == 0  # not committed
    assert mem.queue.pending(routing=Routing.SYSTEM_2, kind=QueueItemType.SUPERSESSION_REVIEW)


def test_auto_tier_supersession_commits_and_cascades(mem, draft):
    # T2 parent fact
    p = mem.gate.submit(draft(key="zone::limit", obj=30, tier=Tier.T2, auto=True,
                              source_id="vendor_doc:v1"),
                        Delta.SUPERSEDING_CHANGE, writer_identity="vendor_doc:v1")
    parent_id = p.committed_record_id

    # a derived child that cites the parent
    child = draft(key="zone::buildable", obj="ok", tier=Tier.T3, auto=True,
                  source_id="derived:d1", parents=(parent_id,))
    c = mem.gate.submit(child, Delta.SUPERSEDING_CHANGE, writer_identity="derived:d1")
    child_id = c.committed_record_id
    assert mem.dag.get_state(child_id) is NodeState.ACTIVE

    # now supersede the parent -> child must flip to TBD + REDERIVE queued
    r = mem.gate.submit(draft(key="zone::limit", obj=45, tier=Tier.T2, auto=True,
                              days_ago=-1, source_id="vendor_doc:v1"),
                        Delta.SUPERSEDING_CHANGE, writer_identity="vendor_doc:v1")
    assert r.disposition is Disposition.AUTO_COMMIT
    assert mem.dag.get_state(parent_id) is NodeState.ARCHIVED
    assert mem.dag.get_state(child_id) is NodeState.TBD
    assert mem.queue.pending(kind=QueueItemType.REDERIVE)


def test_contradiction_flags_active(mem, draft):
    mem.gate.submit(draft(), Delta.SUPERSEDING_CHANGE, writer_identity="municipal:res-402")
    head_id = mem.heads.get("sector_4::zoning_limit").record_id
    r = mem.gate.submit(draft(obj=0), Delta.CONTRADICTION, writer_identity="municipal:res-402")
    assert r.disposition is Disposition.FLAG_CONTRADICTION
    assert mem.dag.get_state(head_id) is NodeState.FLAGGED


def test_malformed_draft_fails_closed(mem, draft):
    bad = draft()
    bad.obj = None  # missing required field
    r = mem.gate.submit(bad, Delta.SUPERSEDING_CHANGE, writer_identity="municipal:res-402")
    assert r.disposition is Disposition.REJECT_FAIL_CLOSED
    assert r.queue_item and r.queue_item.kind is QueueItemType.DEAD_LETTER
    assert mem.heads.get("sector_4::zoning_limit") is None


def test_derived_draft_missing_parent_fails_closed(mem, draft):
    bad = draft(key="x::y", tier=Tier.T3, auto=True, source_id="derived:d",
                parents=("rec_does_not_exist",))
    r = mem.gate.submit(bad, Delta.SUPERSEDING_CHANGE, writer_identity="derived:d")
    assert r.disposition is Disposition.REJECT_FAIL_CLOSED


def test_system_2_approval_commits(mem, draft):
    mem.gate.submit(draft(obj=30), Delta.SUPERSEDING_CHANGE, writer_identity="municipal:res-402")
    r = mem.gate.submit(draft(obj=45, days_ago=-1), Delta.SUPERSEDING_CHANGE,
                        writer_identity="municipal:res-402")
    item_id = r.queue_item.id
    approved = mem.gate.approve_review(item_id)
    assert approved.disposition is Disposition.AUTO_COMMIT
    assert mem.store.get(mem.heads.get("sector_4::zoning_limit").record_id).obj == 45
    assert not mem.queue.pending(kind=QueueItemType.SUPERSESSION_REVIEW)
