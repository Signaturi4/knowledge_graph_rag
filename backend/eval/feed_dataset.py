"""Feed the PMC-Patients cut into a persisted knowledge graph, 80/20 split.

80% of the cut is ingested through the full deterministic pipeline (real LLM
extraction, canonical per-patient subject hint, gate/cascade/DAG persistence).
The remaining 20% is held out -- never ingested -- and written to
`eval/data/pmc_holdout.jsonl` for `eval/run_holdout_eval.py` to probe against
the resulting graph.

Usage:
  python eval/feed_dataset.py --split 0.8 --data-dir "$(pwd)/data/pmc_kb_full"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from observability import configure_logging  # noqa: E402

configure_logging()

from service.assembly import build_memory_layer  # noqa: E402

CUT = os.path.join(HERE, "data", "pmc_cut.jsonl")
HOLDOUT = os.path.join(HERE, "data", "pmc_holdout.jsonl")
MANIFEST = os.path.join(HERE, "data", "feed_manifest.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=float, default=0.8, help="fraction fed into the graph")
    ap.add_argument("--provider", default="claude_bridge")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="skip patients that already have records in the store "
                         "(the previous run was interrupted mid-way)")
    args = ap.parse_args()

    with open(CUT) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    n_train = int(round(len(rows) * args.split))
    train, holdout = rows[:n_train], rows[n_train:]

    with open(HOLDOUT, "w") as f:
        for r in holdout:
            f.write(json.dumps(r) + "\n")

    os.environ["LLM_PROVIDER"] = args.provider
    layer = build_memory_layer(data_dir=args.data_dir)

    already_done: set[str] = set()
    if args.resume:
        already_done = {r.source_id.split("pmc_", 1)[-1] for r in layer.store
                        if r.source_id.startswith("conversation:pmc_")}
        if already_done:
            print(f"resuming: {len(already_done)} patients already in the store, skipping them",
                 flush=True)

    t0 = time.monotonic()
    results = []
    todo = [p for p in train if str(p["patient_id"]) not in already_done]
    for i, p in enumerate(todo, 1):
        pid = p["patient_id"]
        subj = f"patient_{pid}"
        rep = layer.pipeline.ingest_text(
            p["text"], f"conversation:pmc_{pid}", canonical_subject=subj,
        )
        results.append(rep.summary())
        print(f"[{i}/{len(todo)}] patient={pid} drafts={rep.drafts} "
             f"dispositions={rep.summary()['by_disposition']} "
             f"elapsed={time.monotonic()-t0:.0f}s", flush=True)
        layer.snapshot_graphml()  # checkpoint after every patient, not just at the end

    layer.snapshot_graphml()

    manifest = {
        "cut": CUT,
        "n_total": len(rows),
        "n_train": len(train),
        "n_holdout": len(holdout),
        "train_patient_ids": [p["patient_id"] for p in train],
        "holdout_patient_ids": [p["patient_id"] for p in holdout],
        "data_dir": args.data_dir,
        "provider": args.provider,
        "elapsed_s": round(time.monotonic() - t0, 1),
        "final_semantic_keys": len(layer.heads.all_keys()),
        "final_nodes": len(layer.dag.nodes()),
        "pending_system_2": len(layer.queue.pending()),
        "ingest_summaries": results,
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n== feed complete ==")
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ("ingest_summaries", "train_patient_ids", "holdout_patient_ids")},
                     indent=2))
    print(f"manifest -> {MANIFEST}")
    print(f"holdout set -> {HOLDOUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
