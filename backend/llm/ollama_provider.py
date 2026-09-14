"""Phase 8 -- Ollama Local Provider.

Interfaces with a running local Ollama instance (default: http://localhost:11434).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests
from .provider import LLMUnavailable

log = logging.getLogger("dagkb.llm.ollama")


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        model: str = "orcarouter/Qwen3.8-27B-Uncensored:mlx-4bit",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        timeout_s: float = 120.0,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=timeout_s,
            )
            if resp.status_code != 200:
                raise LLMUnavailable(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            return data["message"]["content"]
        except requests.RequestException as exc:
            raise LLMUnavailable(f"Ollama connection error: {exc}") from exc

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        timeout_s: float = 120.0,
    ) -> Any:
        raw = self.complete(prompt, system=system, model=model, timeout_s=timeout_s)
        s, e = raw.find("{"), raw.rfind("}")
        if s == -1 or e == -1:
            raise LLMUnavailable(f"no JSON in Ollama output: {raw[:200]}")
        try:
            return json.loads(raw[s : e + 1])
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"malformed JSON in Ollama output: {exc}") from exc
