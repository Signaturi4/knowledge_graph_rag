"""Phase 8 -- NVIDIA NIM Cloud Provider with 40 RPM Rate Limiter.

Supports cloud models like `meta/llama-3.2-11b-vision-instruct` via
NVIDIA NIM / integrate.api.nvidia.com with rate limiting to prevent
HTTP 429 quota exhaustion (strict 40 requests/minute ceiling).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

import requests
from .provider import LLMUnavailable

log = logging.getLogger("dagkb.llm.nvidia")


class RateLimiter:
    """Thread-safe rate limiter ensuring max_per_minute is never exceeded."""

    def __init__(self, max_per_minute: int = 38) -> None:
        self.interval = 60.0 / float(max_per_minute)
        self.lock = threading.Lock()
        self.last_call = 0.0

    def acquire(self) -> None:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                sleep_time = self.interval - elapsed
                log.debug("RateLimiter: pacing call for %.2fs", sleep_time)
                time.sleep(sleep_time)
            self.last_call = time.time()


_NVIDIA_RATE_LIMITER = RateLimiter(max_per_minute=36)


class NvidiaProvider:
    name = "nvidia"

    def __init__(self, model: str = "meta/llama-3.2-11b-vision-instruct") -> None:
        self.model = model
        self.api_key = os.environ.get("NVIDIA_API_KEY", "")

        # Fallback: check GTC25_DLI/.env if not in environment
        if not self.api_key.startswith("nvapi-"):
            gtc_env = os.path.join(os.path.dirname(__file__), "..", "..", "GTC25_DLI", ".env")
            if os.path.exists(gtc_env):
                try:
                    with open(gtc_env) as f:
                        for line in f:
                            if line.strip().startswith("NVIDIA_API_KEY="):
                                self.api_key = line.strip().split("=", 1)[1].strip()
                                os.environ["NVIDIA_API_KEY"] = self.api_key
                                break
                except Exception:
                    pass

        if not self.api_key.startswith("nvapi-"):
            raise LLMUnavailable("NVIDIA_API_KEY not set or invalid")

        self.endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        timeout_s: float = 60.0,
    ) -> str:
        _NVIDIA_RATE_LIMITER.acquire()
        target_model = model or self.model

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": target_model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.1,
        }

        try:
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=timeout_s)
            if resp.status_code == 429:
                raise LLMUnavailable("NVIDIA API 429: rate limit exceeded (40 RPM cap)")
            if resp.status_code != 200:
                raise LLMUnavailable(f"NVIDIA API HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.RequestException as exc:
            raise LLMUnavailable(f"NVIDIA request error: {exc}") from exc

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        timeout_s: float = 90.0,
    ) -> Any:
        raw = self.complete(prompt, system=system, model=model, timeout_s=timeout_s)
        s, e = raw.find("{"), raw.rfind("}")
        if s == -1 or e == -1:
            raise LLMUnavailable(f"no JSON in NVIDIA output: {raw[:200]}")
        try:
            return json.loads(raw[s : e + 1])
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"malformed JSON in NVIDIA output: {exc}") from exc
