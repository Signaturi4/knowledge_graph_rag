"""A minimal LangChain ``BaseChatModel`` wrapping ``claude_bridge`` (keyless
``claude -p`` subprocess), so RAGAS's ``evaluate()`` can use it as the judge
LLM via ``ragas.llms.LangchainLLMWrapper`` -- no OpenAI key, no network API,
same keyless model this whole project already runs on.

Standalone by design: this file has no dependency on the main project's
``llm/`` package or its venv. RAGAS's own dependency chain (langchain-core /
langchain-community / langchain-openai, all version-pinned to each other in
ways that conflict with this project's main venv -- see
``requirements-ragas.txt``) needs its own isolated venv (``eval/.ragas_venv/``,
built via ``eval/setup_ragas_env.sh``), so this module -- and
``score_with_ragas.py``, which imports it -- only ever runs under that venv's
Python, never the project's own.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")


def _ask_claude(prompt: str, *, model: str | None = None, timeout_s: float = 120.0) -> str:
    """Verbatim reimplementation of claude_bridge/claude.py::ask_claude --
    duplicated (not imported) so this module has zero dependency on the main
    project's package layout or venv; see module docstring."""
    args = [CLAUDE_BIN, "-p", prompt]
    if model:
        args += ["--model", model]
    out = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s, check=True)
    return out.stdout.strip()


class ClaudeBridgeChatModel(BaseChatModel):
    """LangChain-compatible chat model over the keyless ``claude`` CLI.

    RAGAS's metric prompts (faithfulness statement generation/verification,
    context precision/recall judgments) are single-turn -- messages are
    flattened to one prompt string, which is all ``claude -p`` accepts.
    """

    model_name: Optional[str] = None
    timeout_s: float = 120.0

    @property
    def _llm_type(self) -> str:
        return "claude-bridge"

    def _flatten(self, messages: List[BaseMessage]) -> str:
        return "\n\n".join(f"{m.type}: {m.content}" for m in messages)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._flatten(messages)
        text = _ask_claude(prompt, model=self.model_name, timeout_s=self.timeout_s)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
