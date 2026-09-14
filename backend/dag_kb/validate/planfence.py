"""Phase 3 -- the use-path gate (PLANFENCE Algorithm 1, technical-plan section 1.4).

Validates that a consumer (a plan, or a derived triplet about to be served or
recomputed) still descends from the current heads, scoped to its declared
dependencies. Reads only; the sole write is the ``validated_at`` stamp.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

from ..errors import LineageIncomplete
from ..types import ValidationOutcome, ValidationResult

if TYPE_CHECKING:
    from ..graph import KnowledgeDAG
    from ..store.heads import HeadIndex

log = logging.getLogger("dagkb.validate.planfence")


class PlanFence:
    def __init__(self, dag: "KnowledgeDAG", heads: "HeadIndex") -> None:
        self._dag = dag
        self._heads = heads
        self._clock = 0

    def validate(
        self,
        root_id: str,
        declared_deps: Iterable[str],
        *,
        replanned: bool = False,
    ) -> ValidationResult:
        deps = list(dict.fromkeys(declared_deps))  # dedupe, keep order
        log.info("PLANFENCE validate | root=%s deps=%s replanned=%s", root_id, deps, replanned)
        result = self._validate_inner(root_id, deps, replanned=replanned)
        log.info("PLANFENCE result | root=%s -> %s (%s) frontier=%s heads=%s",
                 root_id, result.outcome.value, result.reason, result.frontier, result.heads)
        return result

    def _validate_inner(
        self, root_id: str, deps: list[str], *, replanned: bool
    ) -> ValidationResult:
        # 1: exact parents -> frontier ; incomplete lineage blocks
        try:
            frontier = self._dag.frontier(root_id, deps)
        except LineageIncomplete as exc:
            return ValidationResult(ValidationOutcome.BLOCKED, {}, {}, reason=f"lineage: {exc}")

        # 2: owner heads ; any missing/malformed blocks
        found, missing = self._heads.batch_get(deps)
        if missing:
            return ValidationResult(
                ValidationOutcome.BLOCKED, frontier, {}, reason=f"no head for {missing}"
            )
        heads = {k: h.record_id for k, h in found.items()}

        # 3-4: compare frontier to heads, and confirm the root derives from F
        mismatched = tuple(k for k in deps if frontier[k] != heads[k])
        derives = self._dag.root_derives_from(root_id, frontier.values())

        if mismatched or not derives:
            reason = (
                f"stale dependencies {list(mismatched)}" if mismatched
                else "root does not derive from its frontier"
            )
            if replanned:
                return ValidationResult(ValidationOutcome.BLOCKED, frontier, heads,
                                        mismatched, reason=f"{reason} (already replanned)")
            return ValidationResult(ValidationOutcome.REPLAN_REQUIRED, frontier, heads,
                                    mismatched, reason=reason)

        # 5: authorized
        self._clock += 1
        if self._dag.has_node(root_id):
            self._dag.stamp_validated(root_id, self._clock)
        return ValidationResult(ValidationOutcome.AUTHORIZED, frontier, heads,
                                reason=f"validated@{self._clock}")
