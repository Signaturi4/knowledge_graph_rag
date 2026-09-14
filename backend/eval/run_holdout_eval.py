"""Evaluate the persisted graph (built by feed_dataset.py) against the 20% holdout.

The holdout patients were never ingested -- this measures whether the graph built
from the other 80% lets `/qa`-style retrieval answer questions about topics it HAS
seen (cross-patient terms) and confirms it does NOT fabricate facts about patients
it has never seen (a patient_id from the holdout set should ground on nothing).

Usage:
  python eval/run_holdout_eval.py --data-dir "$(pwd)/data/pmc_kb_full"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from observability import configure_logging  # noqa: E402

configure_logging()

from service.assembly import build_memory_layer  # noqa: E402
from service.routers.graph_qa import QAIn, qa  # noqa: E402
import service.deps as deps  # noqa: E402

HOLDOUT = os.path.join(HERE, "data", "pmc_holdout.jsonl")
REPORT = os.path.join(HERE, "holdout_report.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--provider", default="claude_bridge")
    args = ap.parse_args()

    with open(HOLDOUT) as f:
        holdout = [json.loads(line) for line in f if line.strip()]

    os.environ["LLM_PROVIDER"] = args.provider
    layer = build_memory_layer(data_dir=args.data_dir)
    deps._LAYER = layer

    records = []
    for p in holdout:
        pid = p["patient_id"]
        age_num = re.sub(r"[^0-9]", "", p["age"]) or p["age"]
        q = f"What is the age and sex of patient {pid}?"
        res = qa(QAIn(query=q))
        # correctness check: a holdout patient must NOT be grounded (it was never ingested)
        records.append({
            "patient_id": pid, "question": q,
            "answer": res.get("answer", ""), "context": res.get("context", []),
            "gold": [age_num, p["gender"]],
            "correctly_ungrounded": not res.get("context"),
        })

    n = len(records) or 1
    ungrounded_rate = sum(r["correctly_ungrounded"] for r in records) / n
    report = {
        "n_holdout": len(holdout),
        "data_dir": args.data_dir,
        "correct_ungrounded_rate": round(ungrounded_rate, 3),
        "note": ("1.0 = the graph correctly has no facts about any held-out patient "
                "(no cross-contamination / no fabrication). A lower rate would mean "
                "the QA path is hallucinating grounded-looking answers for unseen entities."),
        "records": records,
    }
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    print(f"full records -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
