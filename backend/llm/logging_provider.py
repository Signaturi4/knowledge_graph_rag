"""Wraps any LLMProvider to log every call: which agent, what prompt, what came
back, how long it took. This is the single choke point that makes the whole
system's LLM traffic observable -- every extraction, delta classification,
entity lookup, and answer synthesis passes through here.

INFO: one line per call -- agent, truncated prompt/response, duration.
DEBUG: full, untruncated system + prompt + response.
Set LOG_LLM_CALLS=0 to disable (the wrapper still forwards calls, just quietly).
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from .provider import LLMProvider, LLMUnavailable
from observability import truncate

log = logging.getLogger("dagkb.llm")

_AGENT_TAG = re.compile(r"^\[(\w+)\]")


def _agent_of(system: str | None) -> str:
    if not system:
        return "unknown"
    m = _AGENT_TAG.match(system.strip())
    return m.group(1) if m else "unknown"


class LoggingProvider:
    """Decorator: same LLMProvider interface, with logging around every call."""

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.name = getattr(inner, "name", inner.__class__.__name__)
        self._enabled = os.environ.get("LOG_LLM_CALLS", "1") == "1"

    def _log_call(self, kind: str, prompt: str, system: str | None, fn):
        agent = _agent_of(system)
        if not self._enabled:
            return fn()
        log.debug("-> [%s] %s call | system=%r prompt=%r", agent, kind, system, prompt)
        t0 = time.monotonic()
        try:
            result = fn()
        except LLMUnavailable as exc:
            dt = (time.monotonic() - t0) * 1000
            log.warning("<- [%s] %s FAILED in %.0fms: %s", agent, kind, dt, exc)
            raise
        dt = (time.monotonic() - t0) * 1000
        log.info("[%s] %s (%.0fms) prompt=%s -> reply=%s",
                 agent, kind, dt, truncate(prompt, 160), truncate(result, 240))
        log.debug("<- [%s] %s full reply: %r", agent, kind, result)
        return result

    def complete(self, prompt: str, *, system: str | None = None,
                 model: str | None = None, timeout_s: float = 60.0) -> str:
        return self._log_call(
            "complete", prompt, system,
            lambda: self._inner.complete(prompt, system=system, model=model, timeout_s=timeout_s),
        )

    def complete_json(self, prompt: str, *, system: str | None = None,
                      model: str | None = None, timeout_s: float = 90.0) -> Any:
        def _call():
            val = self._inner.complete_json(prompt, system=system, model=model, timeout_s=timeout_s)
            return val
        agent = _agent_of(system)
        if not self._enabled:
            return _call()
        log.debug("-> [%s] complete_json call | system=%r prompt=%r", agent, system, prompt)
        t0 = time.monotonic()
        try:
            result = _call()
        except LLMUnavailable as exc:
            dt = (time.monotonic() - t0) * 1000
            log.warning("<- [%s] complete_json FAILED in %.0fms: %s", agent, dt, exc)
            raise
        dt = (time.monotonic() - t0) * 1000
        log.info("[%s] complete_json (%.0fms) prompt=%s -> reply=%s",
                 agent, dt, truncate(prompt, 160), truncate(result, 240))
        log.debug("<- [%s] complete_json full reply: %r", agent, result)
        return result
