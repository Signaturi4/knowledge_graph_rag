"""Phase 6 -- continuous garbage collector (technical-plan section 2.4 / Phase 6).

Sweeps the DAG for fully-superseded, unlocked tombstones past the retention
window and moves them to cold storage, keeping the live graph small. Never
touches ACTIVE or FLAGGED nodes; never breaks a live DERIVED_FROM chain.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from ..types import Actor, AuditRow, EdgeType, NodeState, utcnow

if TYPE_CHECKING:
    from .audit import AuditLog
    from ..graph import KnowledgeDAG
    from ..write.queue import WorkQueue
    from ..store import RecordStore

log = logging.getLogger("dagkb.lifecycle.gc")


class GarbageCollector:
    def __init__(
        self,
        dag: "KnowledgeDAG",
        store: "RecordStore",
        queue: "WorkQueue",
        audit: "AuditLog",
        *,
        cold_dir: str,
        retention_days: int = 30,
        batch: int = 100,
    ) -> None:
        self._dag = dag
        self._store = store
        self._queue = queue
        self._audit = audit
        self._cold_dir = cold_dir
        self._retention = timedelta(days=retention_days)
        self._batch = batch
        self._stop = threading.Event()
        os.makedirs(cold_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    def _is_locked(self, record_id: str) -> bool:
        """Locked if a TBD node still derives from it, or an unresolved queue
        item references it."""
        for dep in self._dag.direct_dependents(record_id):
            if self._dag.get_state(dep) is NodeState.TBD:
                return True
        return self._queue.references_record(record_id)

    def _evictable(self, record_id: str) -> bool:
        st = self._dag.get_state(record_id)
        if st not in (NodeState.ARCHIVED, NodeState.DELETED):
            return False
        if self._is_locked(record_id):
            return False
        rec = self._store.get(record_id)
        # fully superseded: a newer version of the same key exists
        newer = [v for v in self._store.all_versions(rec.semantic_key) if v.owner_seq > rec.owner_seq]
        if not newer:
            return False
        age = utcnow() - (rec.created_at if rec.created_at.tzinfo else rec.created_at.replace(tzinfo=timezone.utc))
        return age > self._retention

    def sweep(self) -> list[str]:
        candidates = list(self._dag.nodes(NodeState.ARCHIVED)) + list(self._dag.nodes(NodeState.DELETED))
        log.info("GC sweep start | candidates=%d batch_cap=%d retention=%s",
                len(candidates), self._batch, self._retention)
        evicted: list[str] = []
        for rid in candidates:
            if len(evicted) >= self._batch:
                log.info("GC sweep hit batch cap (%d), stopping early", self._batch)
                break
            if not self._dag.has_node(rid):
                continue
            if not self._evictable(rid):
                log.debug("GC skip | record=%s locked/not-evictable", rid)
                continue
            rec = self._store.get(rid)
            path = os.path.join(self._cold_dir, f"{rec.txn_id}.jsonl")
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"record_id": rid, "semantic_key": rec.semantic_key,
                                     "obj": rec.obj, "owner_seq": rec.owner_seq,
                                     "evicted_at": utcnow().isoformat()}, default=str) + "\n")
            self._store.mark_cold(rid, True)
            self._dag.remove_node(rid)
            self._audit.record(AuditRow(rid, NodeState.ARCHIVED, None, Actor.SYSTEM_1,
                                        rec.txn_id, "gc: archived to cold storage"))
            evicted.append(rid)
            log.info("GC evicted | record=%s key=%s -> %s", rid, rec.semantic_key, path)
        log.info("GC sweep done | evicted=%d", len(evicted))
        return evicted

    # ------------------------------------------------------------------ #
    def run_forever(self, interval_s: float = 3600.0) -> None:
        log.info("GC daemon started | interval_s=%s", interval_s)
        while not self._stop.wait(interval_s):
            try:
                self.sweep()
            except Exception:  # a daemon must not die on one bad sweep
                log.exception("GC sweep failed; will retry next interval")
        log.info("GC daemon stopped")

    def start(self, interval_s: float = 3600.0) -> threading.Thread:
        t = threading.Thread(target=self.run_forever, args=(interval_s,), daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()
