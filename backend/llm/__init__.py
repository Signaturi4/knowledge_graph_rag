"""LLM provider seam. ``get_provider()`` picks the implementation from the
``LLM_PROVIDER`` env var: ``claude_bridge`` (default) | ``nvidia`` | ``mock``.
"""

from __future__ import annotations

import os

from .provider import LLMProvider, LLMUnavailable

__all__ = ["LLMProvider", "LLMUnavailable", "get_provider"]

_SINGLETON: LLMProvider | None = None


def get_provider(name: str | None = None, *, cached: bool = True) -> LLMProvider:
    global _SINGLETON
    if cached and name is None and _SINGLETON is not None:
        return _SINGLETON

    choice = (name or os.environ.get("LLM_PROVIDER", "claude_bridge")).lower()
    if choice == "claude_bridge":
        from .claude_bridge_provider import ClaudeBridgeProvider

        provider: LLMProvider = ClaudeBridgeProvider(model=os.environ.get("LLM_MODEL"))
    elif choice == "nvidia":
        from .nvidia_provider import NvidiaProvider

        provider = NvidiaProvider()
    elif choice in ("mlx", "mlx_lora"):
        from .mlx_provider import MlxLoraProvider

        provider = MlxLoraProvider(
            adapter_path=os.environ.get(
                "MLX_ADAPTER_PATH",
                os.path.join(os.path.dirname(__file__), "..", "..", "GTC25_DLI", "model", "loras", "Meta-Llama-3-8B-Instruct-PMC-LoRA-mlx")
            )
        )
    elif choice == "ollama":
        from .ollama_provider import OllamaProvider

        provider = OllamaProvider()
    elif choice == "mock":
        from .mock import MockProvider

        provider = MockProvider(default="")
    else:  # pragma: no cover
        raise ValueError(f"unknown LLM_PROVIDER {choice!r}")

    from .logging_provider import LoggingProvider

    provider = LoggingProvider(provider)

    if cached and name is None:
        _SINGLETON = provider
    return provider
