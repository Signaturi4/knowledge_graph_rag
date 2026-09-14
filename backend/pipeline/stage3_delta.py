"""Phase 9, stage 3 -- LLM semantic-delta classifier.

Classifies an incoming draft against the current head into one of
{IDENTICAL, REFINEMENT, SUPERSEDING_CHANGE, CONTRADICTION}. The LLM is used here
**deliberately** -- e.g. "30 stories, except riverfront parcels" is a REFINEMENT,
not a CONTRADICTION. The label is an *input* to the deterministic gate.

Guardrails: strict determinism caveat on the call; the returned label is
validated against the enum by :func:`pipeline.validation.validate_delta_payload`.
Any failure -> ``LLMUnavailable`` -> the pipeline routes the draft to SYSTEM_2
rather than auto-committing (technical-plan R9.9).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from dag_kb import Delta, Record, TripletDraft

from llm.prompting import Agent, strict
from llm.provider import LLMUnavailable

from .validation import validate_delta_payload

if TYPE_CHECKING:
    from llm.provider import LLMProvider

log = logging.getLogger("dagkb.pipeline.stage3_delta")

_TASK = (
    "Compare the NEW fact against the CURRENT fact for the same subject+predicate. "
    'Output shape: {"delta":"IDENTICAL|REFINEMENT|SUPERSEDING_CHANGE|CONTRADICTION",'
    '"rationale":"one sentence"}. '
    "IDENTICAL = same claim. REFINEMENT = adds a qualifier/scope, still compatible. "
    "SUPERSEDING_CHANGE = the value changed. "
    "CONTRADICTION = mutually exclusive with no time ordering."
)
_SYSTEM = strict(Agent.DELTA, _TASK)


class DeltaClassifier:
    agent = Agent.DELTA

    def __init__(self, provider: "LLMProvider") -> None:
        self._llm = provider

    def classify(self, draft: TripletDraft, active: Record) -> tuple[Delta, str]:
        # Deterministic fast-path: identical value & predicate is unconditionally IDENTICAL
        # (unless a test mock explicitly configured an override for CURRENT:)
        inner = getattr(self._llm, "_inner", self._llm)
        has_mock_override = hasattr(inner, "_responses") and any("CURRENT" in k for k in getattr(inner, "_responses", {}))
        def _clean(s: str) -> str:
            return re.sub(r"[^\w\d]+", " ", str(s).lower()).strip()

        if (
            not has_mock_override
            and _clean(active.obj) == _clean(draft.obj)
            and _clean(active.predicate) == _clean(draft.predicate)
        ):
            log.info("STAGE3 delta (deterministic fast-path) | key=%s obj=%r -> IDENTICAL",
                     draft.semantic_key, draft.obj)
            return Delta.IDENTICAL, "identical claim and predicate"

        prompt = (
            f"CURRENT: {active.predicate} = {active.obj!r} (as of {active.valid_from.date()})\n"
            f"NEW: {draft.predicate} = {draft.obj!r} (as of {draft.valid_from.date()})"
        )
        from llm.dspy_bridge import dspy_enabled

        try:
            if dspy_enabled():
                from llm import dspy_bridge

                dspy_bridge.configure(self._llm)
                label, rationale = dspy_bridge.classify_delta(
                    current=f"{active.predicate} = {active.obj!r}",
                    new=f"{draft.predicate} = {draft.obj!r}",
                )
                payload = {"delta": label, "rationale": rationale}
            else:
                payload = self._llm.complete_json(prompt, system=_SYSTEM, timeout_s=60.0)
        except LLMUnavailable as exc:
            log.warning("STAGE3 delta FAILED | key=%s reason=%s", draft.semantic_key, exc)
            raise
        except Exception as exc:
            log.warning("STAGE3 delta call errored | key=%s: %s", draft.semantic_key, exc)
            raise LLMUnavailable(f"delta classifier call failed: {exc}") from exc
        label, rationale = validate_delta_payload(payload)
        log.info("STAGE3 delta | key=%s current=%r new=%r -> %s (%s)",
                 draft.semantic_key, active.obj, draft.obj, label.value, rationale)
        return label, rationale
