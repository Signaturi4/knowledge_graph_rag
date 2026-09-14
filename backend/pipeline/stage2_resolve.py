"""Phase 9, stage 2 -- entity resolution.

Canonicalize the draft's ``semantic_key`` and look up the active head record.
The key is already slugged by :func:`pipeline.stage1_extract.semantic_key`; an
optional LLM *proposal* hook can map near-duplicate keys onto an existing one,
but a deterministic normalizer always confirms (technical-plan section 6b
stage 2).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dag_kb import Record, TripletDraft

if TYPE_CHECKING:
    from dag_kb import HeadIndex
    from dag_kb.store import RecordStore

log = logging.getLogger("dagkb.pipeline.stage2_resolve")


class Resolver:
    def __init__(self, store: "RecordStore", heads: "HeadIndex") -> None:
        self._store = store
        self._heads = heads

    def resolve(self, draft: TripletDraft) -> tuple[TripletDraft, Record | None]:
        # (canonicalisation already done in stage1_extract.semantic_key; this is
        #  where an alias table / LLM-proposed merge would be confirmed against
        #  known keys.)
        original_key = draft.semantic_key
        known = set(self._heads.all_keys())
        if draft.semantic_key not in known:
            alias = self._alias(draft.semantic_key, known)
            if alias:
                log.debug("STAGE2 aliased %s -> %s", original_key, alias)
                draft.semantic_key = alias
        head = self._heads.get(draft.semantic_key)
        active = self._store.get(head.record_id) if head else None
        log.info("STAGE2 resolve | key=%s active=%s",
                 draft.semantic_key, active.record_id if active else "(none — first fact)")
        return draft, active

    @staticmethod
    def _alias(key: str, known: set[str]) -> str | None:
        """Deterministic near-match: same entity, predicate differs by an
        underscore/plural or standard attribute synonyms (e.g. sex/gender)."""
        def _norm(s: str) -> str:
            # normalize underscores, plurals, and basic attribute aliases
            s = s.replace("_", "").lower()
            if s.endswith("ies"):
                s = s[:-3] + "y"
            elif s.endswith("s") and not s.endswith("ss"):
                s = s[:-1]
            s = s.replace("hassex", "hasgender").replace("sex", "gender")
            return s

        target_norm = _norm(key)
        for k in known:
            if _norm(k) == target_norm:
                return k
        return None
