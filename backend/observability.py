"""Centralized, verbose logging for the whole deterministic memory layer.

One call, ``configure_logging()``, sets a consistent format across every module
so a single log stream lets you follow one ingest or one query end-to-end:
raw input in -> extraction LLM call + raw reply -> validated drafts -> gate
decision -> DAG mutation -> cascade -> (query side) entities LLM call -> facts
retrieved -> answer LLM call + raw reply -> answer out.

Env vars:
  LOG_LEVEL      DEBUG | INFO | WARNING | ...   (default INFO)
  LOG_LLM_CALLS  1 | 0                           (default 1 -- log every LLM call)
  LOG_FORMAT     "text" | "json"                 (default "text")

DEBUG additionally prints full, untruncated prompts/system/responses; INFO
prints one-line, truncated summaries. Call this once, as early as possible
(service/main.py, scripts, eval harness) before any other module logs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from collections import deque
from typing import Any

_CONFIGURED = False


# --------------------------------------------------------------------------- #
# In-memory ring buffer so a UI can show "the full logs" without tailing a file.
# --------------------------------------------------------------------------- #
class RingBufferHandler(logging.Handler):
    def __init__(self, capacity: int = 5000) -> None:
        super().__init__()
        self._buf: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            row = {
                "ts": self.formatTime_(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        except Exception:
            return
        with self._lock:
            self._buf.append(row)

    @staticmethod
    def formatTime_(record: logging.LogRecord) -> str:
        import datetime

        return datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).isoformat()

    def snapshot(self, *, limit: int = 500, level: str | None = None,
                contains: str | None = None, logger_prefix: str | None = None) -> list[dict]:
        with self._lock:
            rows = list(self._buf)
        if level:
            order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            min_idx = order.index(level.upper()) if level.upper() in order else 0
            rows = [r for r in rows if order.index(r["level"]) >= min_idx if r["level"] in order]
        if logger_prefix:
            rows = [r for r in rows if r["logger"].startswith(logger_prefix)]
        if contains:
            needle = contains.lower()
            rows = [r for r in rows if needle in r["message"].lower() or needle in r["logger"].lower()]
        return rows[-limit:]


_RING: RingBufferHandler | None = None


def get_ring_buffer() -> RingBufferHandler:
    """Available even before configure_logging() runs (tests, tools)."""
    global _RING
    if _RING is None:
        _RING = RingBufferHandler()
    return _RING


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    fmt = os.environ.get("LOG_FORMAT", "text").lower()

    root = logging.getLogger()
    root.setLevel(lvl)
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)-28s | %(message)s", "%H:%M:%S",
        ))
    root.addHandler(handler)
    root.addHandler(get_ring_buffer())

    # keep third-party libraries quiet unless we're at DEBUG
    if lvl != "DEBUG":
        for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "langchain"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger("dagkb.observability").info(
        "logging configured: level=%s format=%s log_llm_calls=%s",
        lvl, fmt, os.environ.get("LOG_LLM_CALLS", "1"),
    )


def truncate(s: str, n: int = 240) -> str:
    s = s if isinstance(s, str) else str(s)
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f"…(+{len(s) - n} chars)"
