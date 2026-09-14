"""Evaluate the full pipeline on the PMC-Patients cut.

Two modes isolate the two quality dimensions:

  --mode structured   inject clean (patient::attr = value) triplets, skipping LLM
                      extraction. Measures retrieval + answer quality only.
  --mode extracted    run the real LLM extraction pipeline. Measures the whole
                      Path A + Path B, and exposes the Phase 9a key-normalization
                      gap on real prose.

Always-on deterministic metrics: grounding_rate, answer_rate, keyword_recall,
context_precision_proxy. Optional RAGAS metrics with --ragas (faithfulness,
answer_relevancy, context_precision, context_recall) if `ragas` is installed.

Usage:
  python eval/run_eval.py --n 12 --mode structured
  python eval/run_eval.py --n 8  --mode extracted --provider claude_bridge --ragas
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from observability import configure_logging  # noqa: E402

configure_logging()

from dag_kb import Delta, TripletDraft  # noqa: E402
from service.assembly import build_memory_layer  # noqa: E402

CUT = os.path.join(HERE, "data", "pmc_cut.jsonl")
REPORT = os.path.join(HERE, "report.json")


def _load(n: int) -> list[dict]:
    with open(CUT) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:n]


def _provider(name: str):
    if name == "mock":
        from llm.mock import MockProvider

        # canned: entity echo + a generic grounded answer
        return MockProvider({
            "USER QUERY": json.dumps({"entities": []}),
            "using ONLY the FACTS": json.dumps({"answer": "see facts"}),
            "INPUT TEXT": json.dumps({"triplets": []}),
        }, default="")
    os.environ["LLM_PROVIDER"] = name
    from llm import get_provider

    return get_provider(name, cached=False)


# --------------------------------------------------------------------------- #
def _ingest_structured(layer, patients: list[dict]) -> None:
    now = datetime.now(timezone.utc)
    for p in patients:
        pid = p["patient_id"]
        facts = [("age", p["age"]), ("sex", p["gender"])]
        if p["condition_hint"]:
            facts.append(("condition", p["condition_hint"]))
        for attr, val in facts:
            d = TripletDraft(
                semantic_key=f"patient_{pid}::{attr}", predicate=attr, obj=val,
                tier=__import__("dag_kb").Tier.T2, auto_update=True,
                source_id=f"vendor_doc:pmc_{pid}", raw_citation=f"PMC-Patients case {pid}",
                valid_from=now, provenance={"subject": f"patient {pid}"},
            )
            layer.gate.submit(d, Delta.SUPERSEDING_CHANGE, writer_identity=f"vendor_doc:pmc_{pid}")


def _ingest_extracted(layer, patients: list[dict]) -> dict:
    """Real LLM extraction, one canonical_subject per patient (``patient_{pid}``)
    so the probes' "patient {pid}" questions resolve to the entity the
    extractor actually creates -- without this the extractor is free to name
    the subject however it infers from prose ("the patient", a descriptive
    phrase, ...), which reliably breaks entity matching in `/qa` and silently
    produces a 0% grounding rate that reflects an eval-harness bug, not the
    system's actual retrieval quality (found the hard way: a first run of
    this eval, unpatched, scored grounding_rate=0.0 across the board)."""
    disp: dict[str, int] = {}
    for p in patients:
        pid = p["patient_id"]
        rep = layer.pipeline.ingest_text(p["text"], f"conversation:pmc_{pid}",
                                         canonical_subject=f"patient_{pid}")
        for r in rep.results:
            disp[r.disposition.value] = disp.get(r.disposition.value, 0) + 1
    return disp


# --------------------------------------------------------------------------- #
def _probes(patients: list[dict]) -> list[dict]:
    out = []
    for p in patients:
        pid = p["patient_id"]
        age_num = re.sub(r"[^0-9]", "", p["age"]) or p["age"]
        out.append({
            "patient_id": pid,
            "question": f"What is the age and sex of patient {pid}?",
            "gold": [age_num, p["gender"]],
        })
        if p["condition_hint"]:
            out.append({
                "patient_id": pid,
                "question": f"What condition is recorded for patient {pid}?",
                "gold": [p["condition_hint"]],
            })
    return out


def _ask(layer, question: str) -> dict:
    from service.routers.graph_qa import qa, QAIn

    import service.deps as deps
    deps._LAYER = layer
    return qa(QAIn(query=question))


_NO_CTX = {"No grounded knowledge-graph facts match this query.", "insufficient grounded facts"}


def _score(records: list[dict]) -> dict:
    n = len(records) or 1
    grounded = sum(1 for r in records if r["context"])
    answered = sum(1 for r in records if r["answer"] and r["answer"] not in _NO_CTX)
    kw_hits = 0
    prec_sum = 0.0
    for r in records:
        ans = (r["answer"] or "").lower()
        if all(str(g).lower() in ans for g in r["gold"]):
            kw_hits += 1
        if r["context"]:
            hit = sum(1 for c in r["context"] if f"patient {r['patient_id']}" in c.lower()
                      or f"patient_{r['patient_id']}" in c.lower())
            prec_sum += hit / len(r["context"])
    return {
        "probes": n,
        "grounding_rate": round(grounded / n, 3),
        "answer_rate": round(answered / n, 3),
        "keyword_recall": round(kw_hits / n, 3),
        "context_precision_proxy": round(prec_sum / n, 3),
    }


def _ragas(records: list[dict], provider_name: str, *, model: str | None = None) -> dict:
    """Delegates to ``score_with_ragas.py`` run under the isolated
    ``eval/.ragas_venv`` (see that file + ``requirements-ragas.txt`` for why:
    ragas's langchain-family pins conflict with this project's main venv).
    Metrics: faithfulness, context_precision, context_recall -- LLM-only, no
    embeddings model (this project has none, by design)."""
    import subprocess
    import tempfile

    ragas_python = os.path.join(HERE, ".ragas_venv", "bin", "python")
    if not os.path.exists(ragas_python):
        return {"skipped": f"no {ragas_python} -- run eval/setup_ragas_env.sh first"}

    payload = [{
        "question": r["question"],
        "answer": r["answer"],
        "context": r["context"],
        "ground_truth": " ".join(map(str, r["gold"])),
    } for r in records]

    with tempfile.TemporaryDirectory() as td:
        records_path = os.path.join(td, "records.json")
        scores_path = os.path.join(td, "scores.json")
        with open(records_path, "w") as f:
            json.dump(payload, f)
        cmd = [ragas_python, os.path.join(HERE, "score_with_ragas.py"), records_path, scores_path]
        if model:
            cmd += ["--model", model]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            return {"error": "ragas scoring subprocess timed out (1800s)"}
        if proc.returncode != 0:
            return {"error": f"ragas subprocess failed: {proc.stderr[-2000:]}"}
        with open(scores_path) as f:
            return json.load(f)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--mode", choices=["structured", "extracted"], default="structured")
    ap.add_argument("--provider", default="claude_bridge")
    ap.add_argument("--ragas", action="store_true")
    ap.add_argument("--ragas-model", default="claude-haiku-4-5-20251001",
                    help="judge model for RAGAS scoring (smallest/fastest by default)")
    ap.add_argument("--data-dir", default=None,
                    help="persist the KB here (sqlite + graphml + audit); omit for in-memory")
    args = ap.parse_args()

    patients = _load(args.n)
    provider = _provider(args.provider)
    layer = build_memory_layer(provider=provider, data_dir=args.data_dir)

    ingest_disp = {}
    if args.mode == "structured":
        _ingest_structured(layer, patients)
    else:
        ingest_disp = _ingest_extracted(layer, patients)
    layer.snapshot_graphml()

    records = []
    for pr in _probes(patients):
        res = _ask(layer, pr["question"])
        records.append({**pr, "answer": res.get("answer", ""),
                        "context": res.get("context", []),
                        "entities": res.get("entities", [])})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "zhengyun21/PMC-Patients (byte-range cut)",
        "n_patients": len(patients),
        "mode": args.mode,
        "provider": args.provider,
        "semantic_keys": len(layer.heads.all_keys()),
        "ingest_dispositions": ingest_disp,
        "metrics": _score(records),
    }
    if args.ragas:
        report["ragas"] = _ragas(records, args.provider, model=args.ragas_model)

    with open(REPORT, "w") as f:
        json.dump({**report, "records": records}, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nfull records -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
