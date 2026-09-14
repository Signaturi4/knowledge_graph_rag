"""Phase 4 -- one-replan-then-block orchestration (technical-plan section A.3).

A ``REPLAN_REQUIRED(H)`` result is turned into exactly one recompute against the
fresh heads, then re-validated with ``replanned=True``. A second dependency
change during revalidation blocks (Algorithm 1 lines 8-9).

R4.4 -- on a successful replan, the validated fresh root is propagated to
*downstream* dependents (nodes derived from the stale root) via the same
bounded, per-dependent-resilient cascade used for data supersession: they are
flagged TBD and a REDERIVE item is queued, rather than silently left pointing
at a root that is about to be superseded by the caller.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Iterable

from ..types import Actor, ValidationOutcome, ValidationResult

if TYPE_CHECKING:
    from ..graph import KnowledgeDAG
    from ..write.cascade import Cascade
    from .planfence import PlanFence

log = logging.getLogger("dagkb.validate.replan")

# recompute(fresh_heads: dict[key -> record_id]) -> new_root_id
Recompute = Callable[[dict[str, str]], str]


class ReplanOrchestrator:
    def __init__(
        self,
        planfence: "PlanFence",
        *,
        dag: "KnowledgeDAG | None" = None,
        cascade: "Cascade | None" = None,
    ) -> None:
        self._pf = planfence
        self._dag = dag
        self._cascade = cascade

    def run(
        self,
        root_id: str,
        declared_deps: Iterable[str],
        recompute: Recompute,
    ) -> tuple[ValidationResult, str]:
        """Returns ``(final_result, effective_root_id)``.

        ``recompute`` is called **at most once**. It must produce a new consumer
        node whose parent_ids bind to the supplied fresh heads.
        """
        deps = list(declared_deps)
        first = self._pf.validate(root_id, deps, replanned=False)
        if first.outcome is not ValidationOutcome.REPLAN_REQUIRED:
            return first, root_id

        log.info("REPLAN triggered | stale_root=%s deps=%s reason=%s", root_id, deps, first.reason)
        new_root = recompute(dict(first.heads))
        second = self._pf.validate(new_root, deps, replanned=True)
        log.info("REPLAN result | stale_root=%s new_root=%s -> %s",
                root_id, new_root, second.outcome.value)

        if second.outcome is ValidationOutcome.AUTHORIZED and new_root != root_id:
            self._propagate(root_id, new_root)

        return second, new_root

    # ------------------------------------------------------------------ #
    def _propagate(self, stale_root: str, new_root: str) -> None:
        """R4.4: flag direct dependents of the stale root so they re-derive
        against the freshly-validated root before their own next use."""
        if self._dag is None or self._cascade is None or not self._dag.has_node(stale_root):
            return
        key = self._dag.nx.nodes[stale_root].get("semantic_key", stale_root)
        flagged = self._cascade.flag_direct_dependents(stale_root, key, actor=Actor.SYSTEM_1)
        if flagged:
            log.info("REPLAN propagated | stale_root=%s new_root=%s flagged_dependents=%s",
                    stale_root, new_root, flagged)
