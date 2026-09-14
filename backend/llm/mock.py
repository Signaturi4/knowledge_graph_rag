"""Phase 8 -- deterministic provider for tests. No subprocess, no network.

Match rules, in order: exact prompt key, then first substring key found in the
prompt, then ``default``. Raise :class:`LLMUnavailable` for a key mapped to the
sentinel :data:`UNAVAILABLE`.
"""

from __future__ import annotations

import json
from typing import Any

from .provider import LLMUnavailable

UNAVAILABLE = object()


class MockProvider:
    name = "mock"

    def __init__(self, responses: dict[str, Any] | None = None, *, default: Any = "") -> None:
        self._responses = responses or {}
        self._default = default
        self.calls: list[str] = []

    def _resolve(self, prompt: str, system: str | None = None) -> Any:
        hay = f"{system or ''}\n{prompt}"
        self.calls.append(prompt)
        if prompt in self._responses:
            val = self._responses[prompt]
        else:
            val = next(
                (v for k, v in self._responses.items() if k in hay),
                self._default,
            )
        if val is UNAVAILABLE:
            raise LLMUnavailable("mock: configured unavailable")
        return val

    def complete(self, prompt: str, *, system: str | None = None,
                 model: str | None = None, timeout_s: float = 60.0) -> str:
        val = self._resolve(prompt, system)
        return val if isinstance(val, str) else json.dumps(val)

    def complete_json(self, prompt: str, *, system: str | None = None,
                      model: str | None = None, timeout_s: float = 90.0) -> Any:
        val = self._resolve(prompt, system)
        if not isinstance(val, str):
            return val
        try:
            return json.loads(val)
        except ValueError as exc:  # unconfigured / non-JSON default -> fail closed like a real provider
            raise LLMUnavailable(f"mock: no JSON response configured for this prompt ({exc})") from exc
