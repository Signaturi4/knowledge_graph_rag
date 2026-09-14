"""DSPy integration -- typed, self-validating LLM programs on top of the provider seam.

Why DSPy: instead of hand-formatting a JSON prompt and then shape-checking the
reply, each agent is a ``dspy.Signature`` with typed / ``Literal`` output fields.
DSPy formats the prompt, parses the completion, and enforces the field types --
that is the prompt hardening. Our ``pipeline.validation`` checks stay as a second
belt.

Opt-in: set ``USE_DSPY=1``. When DSPy is absent or off, the agents fall back to
the strict hand-rolled prompts. Determinism: the wrapped LM runs at
``temperature=0`` with DSPy caching disabled, and the strict caveat from
``llm.prompting`` is attached to every signature instruction.
"""

from __future__ import annotations

import os
from typing import Any

import pydantic


class Triplet(pydantic.BaseModel):
    """Module-level (not nested in a function) so DSPy/pydantic can resolve
    ``list[Triplet]`` when building the JSON schema for the output field --
    a class defined inside a function is not resolvable there and raises
    ``PydanticUserError: ... is not fully defined``."""

    subject: str
    subject_type: str = ""
    predicate: str
    object: str
    object_type: str = ""


_ENABLED: bool | None = None


def dspy_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        if os.environ.get("USE_DSPY", "0") != "1":
            _ENABLED = False
        else:
            try:
                import dspy  # noqa: F401

                _ENABLED = True
            except Exception:
                _ENABLED = False
    return _ENABLED


def _build_lm(provider):
    import dspy

    class _ProviderLM(dspy.BaseLM):
        """Routes DSPy completions through our LLMProvider (claude_bridge etc.)."""

        forward_contract = "typed_lm"

        def __init__(self) -> None:
            super().__init__(model="provider/claude_bridge", temperature=0.0, cache=False)
            self._provider = provider

        def forward(self, request: "dspy.LMRequest") -> "dspy.LMResponse":
            # DSPy 3.3's typed_lm contract hands us LMMessage objects (pydantic
            # models with .role/.content attributes), not plain dicts -- accept
            # either shape defensively.
            def _role(m):
                return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)

            def _content(m):
                return m.get("content") if isinstance(m, dict) else getattr(m, "content", "")

            msgs = getattr(request, "messages", None) or []
            system = "\n".join(_content(m) for m in msgs if _role(m) == "system")
            user = "\n".join(_content(m) for m in msgs if _role(m) != "system")
            text = self._provider.complete(user, system=system or None, timeout_s=90.0)
            return dspy.LMResponse.from_text(text, model=self.model)

    return _ProviderLM()


def configure(provider) -> None:
    """Point DSPy at our provider with a JSON adapter. Idempotent."""
    import dspy

    dspy.configure(lm=_build_lm(provider), adapter=dspy.JSONAdapter())


# --------------------------------------------------------------------------- #
# programs -- return the SAME shapes pipeline.validation expects
# --------------------------------------------------------------------------- #
def _programs():
    import dspy
    from typing import Literal

    from .prompting import STRICT_CAVEAT

    class ExtractTriples(dspy.Signature):
        __doc__ = (STRICT_CAVEAT + "\nExtract knowledge-graph triplets from the text. "
                   "Entities must not be generic, numeric, or purely temporal. "
                   "Copy every value verbatim from the text; never invent one.")
        text: str = dspy.InputField()
        triplets: list[Triplet] = dspy.OutputField(desc="all triplets found in the text")

    class ClassifyDelta(dspy.Signature):
        __doc__ = (STRICT_CAVEAT + "\nCompare the NEW fact against the CURRENT fact for the "
                   "same subject+predicate. IDENTICAL=same claim; REFINEMENT=adds a compatible "
                   "qualifier; SUPERSEDING_CHANGE=value changed; CONTRADICTION=mutually exclusive "
                   "with no time ordering.")
        current: str = dspy.InputField()
        new: str = dspy.InputField()
        delta: Literal["IDENTICAL", "REFINEMENT", "SUPERSEDING_CHANGE", "CONTRADICTION"] = dspy.OutputField()
        rationale: str = dspy.OutputField(desc="one sentence")

    class ExtractEntities(dspy.Signature):
        __doc__ = (STRICT_CAVEAT + "\nList the named entities in the query. Each entity MUST be "
                   "a verbatim substring of the query.")
        query: str = dspy.InputField()
        entities: list[str] = dspy.OutputField()

    class GroundedAnswer(dspy.Signature):
        __doc__ = (STRICT_CAVEAT + "\nAnswer the query using ONLY the facts. If unsupported, "
                   'answer exactly "insufficient grounded facts".')
        facts: str = dspy.InputField()
        query: str = dspy.InputField()
        answer: str = dspy.OutputField()

    return {
        "extract": (dspy.Predict(ExtractTriples), Triplet),
        "delta": dspy.Predict(ClassifyDelta),
        "entities": dspy.Predict(ExtractEntities),
        "answer": dspy.Predict(GroundedAnswer),
    }


_CACHE: dict[str, Any] = {}


def _get(name: str):
    if not _CACHE:
        _CACHE.update(_programs())
    return _CACHE[name]


# public helpers -------------------------------------------------------------- #
def extract_triplets(text: str) -> list[dict]:
    prog, _ = _get("extract")
    out = prog(text=text)
    return [t.model_dump() if hasattr(t, "model_dump") else dict(t) for t in out.triplets]


def classify_delta(current: str, new: str) -> tuple[str, str]:
    out = _get("delta")(current=current, new=new)
    return out.delta, getattr(out, "rationale", "")


def extract_entities(query: str) -> list[str]:
    out = _get("entities")(query=query)
    return [e for e in out.entities if isinstance(e, str)]


def grounded_answer(facts: str, query: str) -> str:
    return _get("answer")(facts=facts, query=query).answer
