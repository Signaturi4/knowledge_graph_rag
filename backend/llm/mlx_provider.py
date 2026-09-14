"""Phase 8 -- MLX LoRA provider for Apple Silicon (M4 / MPS).

Loads a 4-bit base model (e.g., Meta-Llama-3-8B-Instruct-4bit) plus optional LoRA
adapters (e.g., Meta-Llama-3-8B-Instruct-PMC-LoRA-mlx) and runs high-efficiency,
local inference without external network dependencies or token limits.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from typing import Any

from .provider import LLMUnavailable

log = logging.getLogger("dagkb.llm.mlx")


class MlxLoraProvider:
    name = "mlx_lora"

    def __init__(
        self,
        model_path: str = "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        adapter_path: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_lm import load

            log.info("Loading MLX model: %s (adapter=%s)", self.model_path, self.adapter_path)
            self._model, self._tokenizer = load(
                self.model_path,
                adapter_path=self.adapter_path,
            )
            log.info("MLX model loaded successfully.")
        except Exception as exc:
            log.error("Failed to load MLX model: %s", exc)
            raise LLMUnavailable(f"MLX load failed: {exc}") from exc

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        timeout_s: float = 90.0,
    ) -> str:
        self._ensure_loaded()
        from mlx_lm import generate

        if prompt.startswith("<|begin_of_text|>"):
            formatted_prompt = prompt
        else:
            task_prefix = "Extract all clinical entity-relation-entity triplets from the following patient case as a valid Python list:\n\n"
            content = f"{task_prefix}{prompt}" if not prompt.startswith(task_prefix) else prompt

            formatted_prompt = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )

        try:
            raw_output = generate(
                self._model,
                self._tokenizer,
                prompt=formatted_prompt,
                max_tokens=512,
                verbose=False,
            )
            if "<|eot_id|>" in raw_output:
                raw_output = raw_output[: raw_output.find("<|eot_id|>")]
            return raw_output.strip()
        except Exception as exc:
            log.error("MLX generate error: %s", exc)
            raise LLMUnavailable(f"MLX generation failed: {exc}") from exc

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        timeout_s: float = 90.0,
    ) -> Any:
        raw = self.complete(prompt, system=system, timeout_s=timeout_s)
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(raw[s : e + 1])
            except json.JSONDecodeError:
                pass
        if s != -1 and e != -1:
            try:
                return ast.literal_eval(raw[s : e + 1])
            except Exception:
                pass
        raise LLMUnavailable(f"no valid JSON in MLX output: {raw[:200]}")
