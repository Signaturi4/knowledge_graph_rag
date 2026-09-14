"""Phase 7 -- revert.

Reverting any prior version is a first-class, itself-reversible head move over
retained immutable records (technical-plan section 6e; PRD section 2.2
"compensable transaction rollbacks").
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable

from ..errors import RevertRejected
from ..graph.distance import compute_distance
from ..types import Actor, EdgeType, HeadEntry, NodeState, Tier, utcnow
from .audit import AuditLog

if TYPE_CHECKING:  # avoid import cycles
    from ..graph import KnowledgeDAG
    from ..store import RecordStore
    from ..store.heads import HeadIndex
    from ..write.cascade import Cascade

log = logging.getLogger("dagkb.lifecycle.revert")


class Reverter:
    """``revert`` / ``revert_txn`` over the immutable spine."""

    def __init__(
        self,
        store: "RecordStore",
        heads: "HeadIndex",
        dag: "KnowledgeDAG",
        cascade: "Cascade",
        audit: AuditLog,
        *,
        retention_days: int = 3650,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self._store = store
        self._heads = heads
        self._dag = dag
        self._cascade = cascade
        self._audit = audit
        self._retention = timedelta(days=retention_days)
        self._on_change = on_change or (lambda: None)

    # ------------------------------------------------------------------ #
    def revert(self, semantic_key: str, target_record_id: str, *, actor: Actor) -> None:
        log.info("REVERT request | key=%s target=%s actor=%s", semantic_key, target_record_id, actor.value)

        if not self._store.has(target_record_id):
            log.warning("REVERT rejected | key=%s target=%s reason=unknown record",
                       semantic_key, target_record_id)
            raise RevertRejected(f"unknown record {target_record_id}")
        target = self._store.get(target_record_id)
        if target.semantic_key != semantic_key:
            log.warning("REVERT rejected | key=%s target=%s reason=key mismatch (record belongs to %s)",
                       semantic_key, target_record_id, target.semantic_key)
            raise RevertRejected("target record does not belong to that semantic key")

        state = self._dag.get_state(target_record_id) if self._dag.has_node(target_record_id) else NodeState.ARCHIVED
        if state not in (NodeState.ARCHIVED, NodeState.DELETED):
            log.warning("REVERT rejected | key=%s target=%s reason=state is %s, not ARCHIVED/DELETED",
                       semantic_key, target_record_id, state)
            raise RevertRejected(f"target state {state} is not ARCHIVED/DELETED")

        age = utcnow() - _aware(target.created_at)
        if age > self._retention:
            log.warning("REVERT rejected | key=%s target=%s reason=past retention (age=%s > %s)",
                       semantic_key, target_record_id, age, self._retention)
            raise RevertRejected("target is past the retention window")

        if target.tier in (Tier.T0, Tier.T1, Tier.T2) and actor is not Actor.SYSTEM_2:
            log.warning("REVERT rejected | key=%s target=%s reason=tier %s requires SYSTEM_2, got %s",
                       semantic_key, target_record_id, target.tier.name, actor.value)
            raise RevertRejected(f"tier {target.tier.name} revert requires SYSTEM_2")

        current = self._heads.get(semantic_key)
        if current is not None and current.record_id == target_record_id:
            log.info("REVERT no-op | key=%s target=%s is already the head", semantic_key, target_record_id)
            return  # already the head

        # rehydrate from cold storage if GC archived it
        if self._store.is_cold(target_record_id):
            self._store.mark_cold(target_record_id, False)
        if not self._dag.has_node(target_record_id):
            corroboration = self._dag.corroboration_count(target.object_ref) \
                if target.object_ref else 0
            dist = compute_distance(target.tier, corroboration_count=corroboration)
            self._dag.add_record(target, state=NodeState.ARCHIVED, audit=self._audit,
                                 actor=actor, reason="rehydrate for revert", distance=dist)

        new_seq = (current.owner_seq + 1) if current else target.owner_seq
        self._heads.update(_head_of(target, new_seq), writer_identity=target.owner)

        if current is not None:
            self._dag.set_state(current.record_id, NodeState.ARCHIVED, actor=actor,
                                reason=f"revert to {target_record_id}", audit=self._audit)
            self._dag.add_edge_typed(target_record_id, current.record_id, EdgeType.REVERTED_FROM)

        self._dag.set_state(target_record_id, NodeState.ACTIVE, actor=actor,
                            reason="reverted to active", audit=self._audit)

        if current is not None:
            self._cascade.flag_direct_dependents(current.record_id, semantic_key, actor=actor)

        self._on_change()
        log.info("REVERT committed | key=%s new_head=%s prev_head=%s",
                semantic_key, target_record_id, current.record_id if current else "(none)")

    def revert_txn(self, txn_id: str, *, actor: Actor) -> list[str]:
        """Roll every semantic key touched by one ingest transaction back to its
        pre-txn head. Returns the affected keys."""
        log.info("REVERT_TXN request | txn=%s actor=%s", txn_id, actor.value)
        touched = {r.semantic_key: r for r in self._store if r.txn_id == txn_id}
        affected: list[str] = []
        for key, rec in touched.items():
            prior = [v for v in self._store.all_versions(key) if v.owner_seq < rec.owner_seq]
            if not prior:
                log.info("REVERT_TXN skip | txn=%s key=%s has no prior version", txn_id, key)
                continue
            self.revert(key, prior[-1].record_id, actor=actor)
            affected.append(key)
        log.info("REVERT_TXN done | txn=%s affected=%s", txn_id, affected)
        return affected


# --------------------------------------------------------------------------- #
def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _head_of(record, owner_seq: int) -> HeadEntry:
    return HeadEntry(
        semantic_key=record.semantic_key,
        record_id=record.record_id,
        owner=record.owner,
        owner_seq=owner_seq,
    )
