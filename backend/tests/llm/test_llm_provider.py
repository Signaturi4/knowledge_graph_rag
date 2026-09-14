"""Phase 8 -- R8.* : provider seam. Mock is offline; claude_bridge is opt-in."""
from __future__ import annotations

import os

import pytest

from llm import get_provider
from llm.mock import UNAVAILABLE, MockProvider
from llm.provider import LLMProvider, LLMUnavailable


def test_mock_is_offline_and_typed():
    p = MockProvider({"ping": "pong", "json me": {"a": 1}}, default="")
    assert isinstance(p, LLMProvider)
    assert p.complete("say ping please") == "pong"
    assert p.complete_json("please json me now") == {"a": 1}


def test_mock_unavailable_raises():
    p = MockProvider({"boom": UNAVAILABLE})
    with pytest.raises(LLMUnavailable):
        p.complete("boom now")


def test_factory_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    p = get_provider(cached=False)
    assert getattr(p, "name", "") == "mock"


@pytest.mark.skipif(
    os.environ.get("RUN_CLAUDE_BRIDGE_TESTS") != "1",
    reason="set RUN_CLAUDE_BRIDGE_TESTS=1 to exercise the real claude CLI",
)
def test_claude_bridge_roundtrip():
    p = get_provider("claude_bridge", cached=False)
    assert p.complete("Reply with exactly the word: OK").strip() == "OK"
    obj = p.complete_json('Return ONLY minified JSON {"ok": true}')
    assert obj == {"ok": True}
