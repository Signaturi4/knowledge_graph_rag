"""Section 6b stage 6 -- the bounded dependency cascade.

On every ACTIVE -> ARCHIVED transition, direct (depth-1) dependents are flagged
TBD and a REDERIVE work item is enqueued. Transitive spread is queue-driven --
one hop per processed item -- never a recursive walk at write time
(technical-plan R9.7; paper section 2.1 "unrelated dependents shielded").
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..types import Actor, NodeState, QueueItem, QueueItemType, Routing, Tier

log = logging.getLogger("dagkb.write.cascade")

if TYPE_CHECKING:
    from ..lifecycle.audit import AuditLog
    from ..graph import KnowledgeDAG
    from .queue import WorkQueue


def _route_for_tier(tier: Tier) -> Routing:
    # T3 derived nodes re-derive automatically; everything else needs review.
    return Routing.SYSTEM_1 if tier is Tier.T3 else Routing.SYSTEM_2


class Cascade:
    def __init__(self, dag: "KnowledgeDAG", queue: "WorkQueue", audit: "AuditLog") -> None:
        self._dag = dag
        self._queue = queue
        self._audit = audit

    def flag_direct_dependents(
        self,
        archived_record_id: str,
        changed_key: str,
        *,
        actor: Actor = Actor.SYSTEM_1,
        txn_id: str = "-",
    ) -> list[str]:
        """Returns the node ids that were moved to TBD.

        Resilient per dependent: one bad transition does not abort the rest, and
        because this runs *after* the commit is durable it can never corrupt a
        valid head move.
        """
        flagged: list[str] = []
        for dep in self._dag.direct_dependents(archived_record_id):
            try:
                if self._dag.get_state(dep) is not NodeState.ACTIVE:
                    continue
                self._dag.set_state(
                    dep, NodeState.TBD, actor=actor,
                    reason=f"parent {changed_key} changed; update in progress",
                    audit=self._audit, txn_id=txn_id,
                )
                tier = self._dag.nx.nodes[dep]["tier"]
                self._queue.enqueue(QueueItem(
                    kind=QueueItemType.REDERIVE,
                    routed_to=_route_for_tier(tier),
                    semantic_key=self._dag.nx.nodes[dep]["semantic_key"],
                    payload={"node_id": dep, "changed_parent": changed_key,
                             "archived_parent": archived_record_id},
                ))
                flagged.append(dep)
                log.info("CASCADE flag TBD | archived=%s changed_key=%s dependent=%s",
                        archived_record_id, changed_key, dep)
            except Exception as exc:  # noqa: BLE001 -- resilience is the point
                log.warning("CASCADE step failed for dependent=%s: %s", dep, exc)
                continue
        if flagged:
            log.info("CASCADE done | archived=%s changed_key=%s flagged=%d",
                    archived_record_id, changed_key, len(flagged))
        return flagged
