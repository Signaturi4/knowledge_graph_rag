"""Section 6b stage 4-5 -- the reconciliation gate (adapted PLANFENCE Algorithm 1).

Same structural logic as the use-path gate -- mismatch detection, a single
bounded repair decision, fail closed -- but the trigger is *data arrival* and the
layer is *memory maintenance*.

    Input : draft N_d for key x, delta in {IDENTICAL, REFINEMENT,
            SUPERSEDING_CHANGE, CONTRADICTION}
    1  N_a <- heads.get(x)
    2  if N_d malformed / x unresolved / tier missing / derived w/ missing parent:
           dead-letter; return REJECT_FAIL_CLOSED
    3  if N_a is None:                       return AUTO_COMMIT
    4  if delta == IDENTICAL:                return NO_OP  (append citation)
    5  if delta == CONTRADICTION:            FLAG both; CRE lock; return FLAG_CONTRADICTION
    6  if precedence(N_d, N_a) == REJECT:    return NO_OP
    7  if auto_allowed(N_a, N_d):            return AUTO_COMMIT   (SYSTEM_1)
    8  enqueue SUPERSESSION_REVIEW; notify;  return QUEUE_SYSTEM_2 (SYSTEM_2)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from ..errors import HeadConflict
from ..graph.distance import compute_distance
from ..ids import compute_record_id, new_txn_id
from .precedence import auto_allowed, precedence

log = logging.getLogger("dagkb.write.gate")
from ..types import (
    Actor,
    Delta,
    Disposition,
    EdgeType,
    GateResult,
    HeadEntry,
    NodeState,
    Precedence,
    QueueItem,
    QueueItemType,
    Record,
    Routing,
    TripletDraft,
)

if TYPE_CHECKING:
    from ..lifecycle.audit import AuditLog
    from .cascade import Cascade
    from ..graph import KnowledgeDAG
    from ..store.heads import HeadIndex
    from .queue import WorkQueue
    from ..store import RecordStore


class ReconciliationGate:
    def __init__(
        self,
        store: "RecordStore",
        heads: "HeadIndex",
        dag: "KnowledgeDAG",
        queue: "WorkQueue",
        audit: "AuditLog",
        cascade: "Cascade",
        *,
        notifier: Callable[[dict], None] | None = None,
        on_commit: Callable[[str], None] | None = None,
    ) -> None:
        self._store = store
        self._heads = heads
        self._dag = dag
        self._queue = queue
        self._audit = audit
        self._cascade = cascade
        self._notify = notifier or (lambda _payload: None)
        self._on_commit = on_commit or (lambda _rid: None)

    @property
    def queue(self) -> "WorkQueue":
        return self._queue

    # ------------------------------------------------------------------ #
    def submit(
        self,
        draft: TripletDraft,
        delta: Delta,
        *,
        writer_identity: str,
        txn_id: str | None = None,
    ) -> GateResult:
        txn_id = txn_id or new_txn_id()
        log.info("GATE submit | txn=%s key=%s delta=%s writer=%s obj=%r",
                 txn_id, draft.semantic_key, delta.value, writer_identity, draft.obj)
        result = self._submit_inner(draft, delta, writer_identity=writer_identity, txn_id=txn_id)
        log.info("GATE result | txn=%s key=%s -> %s (%s)",
                 txn_id, draft.semantic_key, result.disposition.value, result.detail)
        return result

    def _submit_inner(
        self,
        draft: TripletDraft,
        delta: Delta,
        *,
        writer_identity: str,
        txn_id: str,
    ) -> GateResult:
        key = draft.semantic_key

        # --- line 2: fail closed -------------------------------------- #
        bad = self._malformed_reason(draft, writer_identity)
        if bad:
            item = self._queue.enqueue(QueueItem(
                kind=QueueItemType.DEAD_LETTER, routed_to=Routing.SYSTEM_2,
                semantic_key=key, payload={"reason": bad, "draft": _draft_summary(draft)},
            ))
            return GateResult(Disposition.REJECT_FAIL_CLOSED, key, queue_item=item, detail=bad)

        active_entry = self._heads.get(key)

        # --- line 3: first fact for x -------------------------------- #
        if active_entry is None:
            rid = self._commit(draft, prev=None, writer_identity=writer_identity, txn_id=txn_id)
            return GateResult(Disposition.AUTO_COMMIT, key, committed_record_id=rid,
                              detail="first version")

        # fail closed if the head points at a record we cannot load
        if not self._store.has(active_entry.record_id):
            item = self._queue.enqueue(QueueItem(
                kind=QueueItemType.DEAD_LETTER, routed_to=Routing.SYSTEM_2, semantic_key=key,
                payload={"reason": f"head {active_entry.record_id} missing from store",
                         "draft": _draft_summary(draft)},
            ))
            return GateResult(Disposition.REJECT_FAIL_CLOSED, key, queue_item=item,
                              detail="active head record missing")
        active = self._store.get(active_entry.record_id)

        # --- line 4: identical -------------------------------------- #
        if delta is Delta.IDENTICAL:
            # Update citation, frequency, and document date on the active record node
            doc_date_str = (
                draft.valid_from.isoformat()
                if hasattr(draft.valid_from, "isoformat")
                else str(draft.valid_from)
            )
            self._dag.record_mention(
                active.record_id,
                source_id=draft.source_id,
                doc_date=doc_date_str,
            )
            self._audit_noop(active.record_id, txn_id, f"identical re-ingest from {draft.source_id}; mention recorded")
            return GateResult(Disposition.NO_OP, key, detail="identical")

        # --- line 5: contradiction --------------------------------- #
        if delta is Delta.CONTRADICTION:
            self._dag.set_state(active.record_id, NodeState.FLAGGED, actor=Actor.CRE,
                                reason="contradiction detected at ingest", audit=self._audit,
                                txn_id=txn_id)
            item = self._queue.enqueue(QueueItem(
                kind=QueueItemType.CONTRADICTION, routed_to=Routing.SYSTEM_2, semantic_key=key,
                payload={"active": active.record_id, "draft": _draft_summary(draft),
                         "llm_assessment": draft.provenance.get("llm_assessment", "")},
            ))
            self._notify({"type": "CONTRADICTION", "semantic_key": key,
                          "active": active.record_id, "draft": _draft_summary(draft)})
            return GateResult(Disposition.FLAG_CONTRADICTION, key, queue_item=item)

        # --- line 6: precedence ----------------------------------- #
        candidate = self._candidate_record(draft, active.owner_seq + 1, writer_identity, txn_id)
        if precedence(candidate, active) is Precedence.REJECT:
            self._audit_noop(active.record_id, txn_id, "draft lower authority / older; kept as variant")
            return GateResult(Disposition.NO_OP, key, detail="precedence: reject")

        # --- line 7: auto-commit (SYSTEM_1) ---------------------- #
        if auto_allowed(active, candidate):
            rid = self._commit(draft, prev=active, writer_identity=writer_identity, txn_id=txn_id)
            return GateResult(Disposition.AUTO_COMMIT, key, committed_record_id=rid,
                              detail="auto-allowed supersession")

        # --- line 8: queue for SYSTEM_2 ------------------------- #
        item = self._queue.enqueue(QueueItem(
            kind=QueueItemType.SUPERSESSION_REVIEW, routed_to=Routing.SYSTEM_2, semantic_key=key,
            payload={"active": active.record_id, "active_obj": active.obj,
                     "draft": _draft_summary(draft),
                     "llm_assessment": draft.provenance.get("llm_assessment", ""),
                     "writer_identity": writer_identity, "txn_id": txn_id},
        ))
        self._notify({"type": "SUPERSESSION_REVIEW", "semantic_key": key,
                      "active_obj": active.obj, "draft": _draft_summary(draft)})
        return GateResult(Disposition.QUEUE_SYSTEM_2, key, queue_item=item)

    # ------------------------------------------------------------------ #
    def approve_review(self, item_id: str, *, actor: Actor = Actor.SYSTEM_2) -> GateResult:
        """Commit a SYSTEM_2-approved supersession."""
        item = self._queue.get(item_id)
        if item is None or item.kind is not QueueItemType.SUPERSESSION_REVIEW:
            raise KeyError(item_id)
        p = item.payload
        # the head may have moved since the item was queued -> re-check
        current = self._heads.get(item.semantic_key)
        if current is None or current.record_id != p["active"]:
            return GateResult(Disposition.QUEUE_SYSTEM_2, item.semantic_key, queue_item=item,
                              detail="head moved since review was queued; re-review required")
        active = self._store.get(p["active"])
        draft = _draft_from_summary(p["draft"])
        try:
            rid = self._commit(draft, prev=active, writer_identity=p["writer_identity"],
                               txn_id=p["txn_id"])
        except HeadConflict:
            return GateResult(Disposition.QUEUE_SYSTEM_2, item.semantic_key, queue_item=item,
                              detail="commit conflicted; re-review required")
        self._queue.resolve(item_id, "approve")
        return GateResult(Disposition.AUTO_COMMIT, item.semantic_key, committed_record_id=rid,
                          detail="SYSTEM_2 approved")

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _malformed_reason(self, draft: TripletDraft, writer_identity: str) -> str | None:
        if not draft.required_fields_present():
            return "missing required fields"
        if not writer_identity:
            return "no writer identity"
        for pid in draft.parent_ids:
            if not self._store.has(pid):
                return f"derived draft cites missing parent {pid}"
        return None

    def _candidate_record(
        self, draft: TripletDraft, owner_seq: int, owner: str, txn_id: str
    ) -> Record:
        rid = compute_record_id(
            semantic_key=draft.semantic_key, predicate=draft.predicate, obj=draft.obj,
            parent_ids=draft.parent_ids, owner=owner, owner_seq=owner_seq,
        )
        return draft.to_record(record_id=rid, owner=owner, owner_seq=owner_seq, txn_id=txn_id)

    def _commit(
        self,
        draft: TripletDraft,
        *,
        prev: Record | None,
        writer_identity: str,
        txn_id: str,
    ) -> str:
        owner_seq = 0 if prev is None else prev.owner_seq + 1
        record = self._candidate_record(draft, owner_seq, writer_identity, txn_id)
        log.info("COMMIT begin | txn=%s key=%s new_id=%s prev_id=%s owner_seq=%d",
                 txn_id, record.semantic_key, record.record_id,
                 prev.record_id if prev else "(none)", owner_seq)

        # All-or-nothing: apply mutations while recording undo steps; on ANY
        # failure, unwind in reverse so no partial head move survives (R9.6).
        undo: list[Callable[[], None]] = []
        try:
            self._store.put(record)  # idempotent; nothing to undo

            # tier confidence, tightened by corroboration (how many other
            # subjects already reference this record's object entity) ->
            # Dijkstra edge weight for BoundedRetriever; see graph/distance.py
            corroboration = self._dag.corroboration_count(record.object_ref) \
                if record.object_ref else 0
            dist = compute_distance(record.tier, corroboration_count=corroboration)
            self._dag.add_record(record, state=NodeState.ACTIVE, audit=self._audit,
                                 actor=Actor.SYSTEM_1, reason="commit", distance=dist)
            undo.append(lambda: self._dag.remove_node(record.record_id)
                        if self._dag.has_node(record.record_id) else None)

            self._heads.update(
                HeadEntry(record.semantic_key, record.record_id, writer_identity, owner_seq),
                writer_identity=writer_identity,
            )
            if prev is not None:
                undo.append(lambda: self._heads.force_set(
                    HeadEntry(prev.semantic_key, prev.record_id, prev.owner, prev.owner_seq)))

            if prev is not None:
                self._dag.add_edge_typed(record.record_id, prev.record_id, EdgeType.SUPERSEDES)
                self._dag.set_state(prev.record_id, NodeState.ARCHIVED, actor=Actor.SYSTEM_1,
                                    reason=f"superseded by {record.record_id}", audit=self._audit,
                                    txn_id=txn_id)
                undo.append(lambda: self._dag.set_state(
                    prev.record_id, NodeState.ACTIVE, actor=Actor.SYSTEM_1,
                    reason="rollback", audit=self._audit, txn_id=txn_id))
        except Exception:
            for step in reversed(undo):
                try:
                    step()
                except Exception:  # best effort; keep unwinding
                    pass
            raise

        # The commit is now durable. The bounded cascade is a *follow-up* -- a
        # failure here must not roll back a valid head move, so it runs outside
        # the atomic section and is resilient per-dependent.
        if prev is not None:
            self._cascade.flag_direct_dependents(prev.record_id, record.semantic_key, txn_id=txn_id)

        self._on_commit(record.record_id)
        return record.record_id

    def _audit_noop(self, record_id: str, txn_id: str, reason: str) -> None:
        from ..types import AuditRow

        st = self._dag.get_state(record_id) if self._dag.has_node(record_id) else None
        self._audit.record(AuditRow(record_id, st, st, Actor.SYSTEM_1, txn_id, reason))


# --------------------------------------------------------------------------- #
def _draft_summary(d: TripletDraft) -> dict:
    return {
        "semantic_key": d.semantic_key, "predicate": d.predicate, "obj": d.obj,
        "tier": int(d.tier), "auto_update": d.auto_update, "source_id": d.source_id,
        "raw_citation": d.raw_citation, "valid_from": d.valid_from.isoformat(),
        "parent_ids": list(d.parent_ids), "record_type": d.record_type,
        "provenance": dict(d.provenance),
    }


def _draft_from_summary(s: dict) -> TripletDraft:
    from datetime import datetime

    from ..types import Tier

    return TripletDraft(
        semantic_key=s["semantic_key"], predicate=s["predicate"], obj=s["obj"],
        tier=Tier(int(s["tier"])), auto_update=bool(s["auto_update"]),
        source_id=s["source_id"], raw_citation=s["raw_citation"],
        valid_from=datetime.fromisoformat(s["valid_from"]),
        parent_ids=tuple(s.get("parent_ids", ())), record_type=s.get("record_type", "asserted"),
        provenance=dict(s.get("provenance", {})),
    )
