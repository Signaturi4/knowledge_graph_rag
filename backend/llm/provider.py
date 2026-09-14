"""Phase 8 -- the LLM provider seam.

One swappable interface for every model call. ``dag_kb`` never imports this;
only ``pipeline/`` (extraction, delta classification) and the graph-QA router do.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class LLMUnavailable(RuntimeError):
    """The model could not be reached: binary missing, logged out, timeout,
    non-zero exit, or unparseable output. Callers decide the fallback -- they
    must never treat this as an empty/soft answer (technical-plan R8.3 / R9.9)."""


@runtime_checkable
class LLMProvider(Protocol):
    def complete(self, prompt: str, *, system: str | None = None,
                 model: str | None = None, timeout_s: float = 60.0) -> str: ...

    def complete_json(self, prompt: str, *, system: str | None = None,
                      model: str | None = None, timeout_s: float = 90.0) -> Any: ...
