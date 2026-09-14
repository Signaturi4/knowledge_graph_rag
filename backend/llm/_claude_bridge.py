"""
claude_bridge — call the Claude Code CLI headlessly from Python.

Same idea as claude.mjs: no API key, no SDK. It runs `claude -p "<prompt>"` as a
subprocess and reads the answer from stdout, using the CLI's own authenticated
session. Use this from data/ML scripts that already run in Python.

Requirements: the `claude` binary installed and already logged in on this machine.
Set CLAUDE_BIN to its path, or leave it and rely on PATH.

Usage:
    from claude import ask_claude, ask_claude_json
    text = ask_claude("Write a haiku about retrieval.")
    obj  = ask_claude_json('Return {"ok": true} as minified JSON.')
"""
from __future__ import annotations

import json
import os
import subprocess

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")


def ask_claude(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    timeout_s: float = 60.0,
) -> str:
    """Send one prompt, return the model's trimmed text answer.

    Raises subprocess.CalledProcessError on non-zero exit, TimeoutExpired on timeout.
    """
    full = f"{system}\n\n{prompt}" if system else prompt
    args = [CLAUDE_BIN, "-p", full]
    if model:
        args += ["--model", model]
    out = subprocess.run(
        args, capture_output=True, text=True, timeout=timeout_s, check=True
    )
    return out.stdout.strip()


def ask_claude_json(prompt: str, **kwargs):
    """Ask for JSON and parse it, slicing the first bracket to the last so prose or
    code fences around the JSON do not break parsing."""
    kwargs.setdefault("timeout_s", 90.0)
    out = ask_claude(prompt, **kwargs)
    obj_start, arr_start = out.find("{"), out.find("[")
    use_array = arr_start != -1 and (obj_start == -1 or arr_start < obj_start)
    start = arr_start if use_array else obj_start
    end = out.rfind("]") if use_array else out.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON found in Claude output: {out[:200]}")
    return json.loads(out[start : end + 1])


def ask_claude_or_fallback(prompt: str, fallback: str, **kwargs) -> str:
    """Never-throws wrapper: model text, or the fallback if the CLI is missing,
    not logged in, times out, or errors."""
    try:
        text = ask_claude(prompt, **kwargs)
        return text if len(text) > 10 else fallback
    except Exception:
        return fallback


if __name__ == "__main__":
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else "Reply with exactly the word: OK"
    print(ask_claude(p))
