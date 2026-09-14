"""Phase 8 -- default provider: the keyless Claude Code CLI.

Delegates to ``_claude_bridge.py`` (vendored verbatim from
``/Users/signatur4ik/Desktop/Determenistic_memory_layer/claude_bridge/claude.py``),
which shells ``claude -p`` using the CLI's own authenticated session. No API key,
no SDK. Set ``CLAUDE_BIN`` to the absolute binary path for PATH-less services.
"""

from __future__ import annotations

import subprocess
from typing import Any

from . import _claude_bridge as cb
from .provider import LLMUnavailable


class ClaudeBridgeProvider:
    name = "claude_bridge"

    def __init__(self, model: str | None = None) -> None:
        self._model = model
        self._fallback = None

    def _get_fallback(self):
        if self._fallback is None:
            try:
                from .nvidia_provider import NvidiaProvider
                self._fallback = NvidiaProvider()
            except Exception:
                self._fallback = False
        return self._fallback if self._fallback is not False else None

    def complete(self, prompt: str, *, system: str | None = None,
                 model: str | None = None, timeout_s: float = 60.0) -> str:
        try:
            return cb.ask_claude(prompt, system=system, model=model or self._model,
                                 timeout_s=timeout_s)
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
            fb = self._get_fallback()
            if fb:
                try:
                    return fb.complete(prompt, system=system, timeout_s=timeout_s)
                except Exception:
                    pass
            raise LLMUnavailable(f"claude_bridge: {exc}") from exc

    def complete_json(self, prompt: str, *, system: str | None = None,
                      model: str | None = None, timeout_s: float = 90.0) -> Any:
        try:
            return cb.ask_claude_json(prompt, system=system, model=model or self._model,
                                      timeout_s=timeout_s)
        except (subprocess.SubprocessError, FileNotFoundError, OSError, ValueError) as exc:
            fb = self._get_fallback()
            if fb:
                try:
                    return fb.complete_json(prompt, system=system, timeout_s=timeout_s)
                except Exception:
                    pass
            raise LLMUnavailable(f"claude_bridge json: {exc}") from exc
