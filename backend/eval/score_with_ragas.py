"""Score a list of QA records with RAGAS, using the keyless claude_bridge judge
LLM (``ragas_llm_wrapper.ClaudeBridgeChatModel``) -- no OpenAI key.

Runs only under the isolated ``eval/.ragas_venv`` (see
``eval/setup_ragas_env.sh`` / ``eval/requirements-ragas.txt``); the main
project venv does not, and should not, have ``ragas`` installed -- its
langchain-family pins conflict with the rest of this project's dependencies
(discovered the hard way: installing ragas into the shared venv broke
langgraph/langchain-openai there). ``run_eval.py`` (main venv) shells out to
this script as a subprocess rather than importing ragas directly.

Deliberately LLM-only metrics -- faithfulness, context_precision,
context_recall -- no answer_relevancy / semantic-similarity metric, since
those need an embeddings model and this project has none (embeddings would
also cut against its stated no-vector-store design; RAGAS's LLM-judge
metrics don't need one).

Usage:
  .ragas_venv/bin/python score_with_ragas.py records.json scores.json [--model claude-haiku-4-5-20251001]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("records_path")
    ap.add_argument("scores_path")
    ap.add_argument("--model", default=None)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    args = ap.parse_args()

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall, faithfulness

    from ragas_llm_wrapper import ClaudeBridgeChatModel
    from ragas.llms import LangchainLLMWrapper

    with open(args.records_path) as f:
        records = json.load(f)

    if not records:
        json.dump({"skipped": "no records to score"}, open(args.scores_path, "w"))
        return 0

    ds = Dataset.from_list([{
        "question": r["question"],
        "answer": r["answer"] or "",
        "contexts": r["context"] or ["(no grounded context)"],
        "ground_truth": r["ground_truth"],
    } for r in records])

    evaluator_llm = LangchainLLMWrapper(
        ClaudeBridgeChatModel(model_name=args.model, timeout_s=args.timeout_s)
    )
    metrics = [faithfulness, context_precision, context_recall]
    for m in metrics:
        m.llm = evaluator_llm

    try:
        result = evaluate(ds, metrics=metrics)
        df = result.to_pandas()
        scores = {
            "aggregate": {k: round(float(v), 3) for k, v in df.mean(numeric_only=True).items()},
            "per_record": df.to_dict(orient="records"),
        }
    except Exception as exc:  # noqa: BLE001 -- report the failure, don't crash the caller
        scores = {"error": f"{type(exc).__name__}: {exc}"}

    with open(args.scores_path, "w") as f:
        json.dump(scores, f, indent=2, default=str)
    print(json.dumps(scores.get("aggregate", scores), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
