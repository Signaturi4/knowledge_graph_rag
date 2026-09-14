"""Side-by-side, real (no mocking) comparison of NVIDIA's original extraction
(`eval/reference_preprocessor.py`) vs. this project's current, modified
extraction (`pipeline/stage1_extract.py`) on the same real document, through
the same live LLM backend (default: claude_bridge).

Usage:
  python eval/compare_reference.py --doc-index 0 --provider claude_bridge
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from observability import configure_logging  # noqa: E402

configure_logging()

from llm import get_provider  # noqa: E402
from eval.reference_preprocessor import extract_triples as reference_extract  # noqa: E402
from pipeline.stage0_sources import RawItem  # noqa: E402
from pipeline.stage1_extract import Extractor  # noqa: E402
from dag_kb import Tier  # noqa: E402
from datetime import datetime, timezone  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-index", type=int, default=0)
    ap.add_argument("--provider", default="claude_bridge")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "pmc_cut.jsonl"))
    args = ap.parse_args()

    with open(args.data) as f:
        lines = [json.loads(line) for line in f]
    patient = lines[args.doc_index]
    text = patient.get("patient") or patient.get("text")
    pid = patient.get("patient_id", args.doc_index)

    provider = get_provider(args.provider)

    print(f"=== INPUT (patient_{pid}, {len(text)} chars) ===")
    print(text[:600] + ("..." if len(text) > 600 else ""))
    print()

    print("=== REFERENCE (NVIDIA original taxonomy, unmodified) ===")
    ref_rows = reference_extract(text, provider)
    for r in ref_rows:
        print(f"  ({r['subject']!r}, {r['subject_type']!r}, {r['relation']!r}, "
              f"{r['object']!r}, {r['object_type']!r})")
    print(f"  -> {len(ref_rows)} triplets")
    print()

    print("=== PRODUCTION (this project's clinical-extended extractor) ===")
    raw = RawItem(text=text, source_id=f"conversation:compare_{pid}", tier=Tier.T4,
                 auto_update=True, citation=f"compare_reference doc {pid}",
                 valid_from=datetime.now(timezone.utc), canonical_subject=f"patient_{pid}")
    drafts = Extractor(provider).extract(raw)
    for d in drafts:
        print(f"  {d.subject_ref!r} --[{d.predicate}]--> {d.obj!r} "
              f"(object_ref={d.object_ref!r}, cardinality={d.cardinality.value})")
    print(f"  -> {len(drafts)} drafts")
    print()

    ref_relations = {r["relation"] for r in ref_rows}
    prod_predicates = {d.predicate for d in drafts}
    print("=== DIFF ===")
    print(f"  reference relation verbs used: {sorted(ref_relations)}")
    print(f"  production predicates used:    {sorted(prod_predicates)}")
    print(f"  clinical-only predicates seen: {sorted(prod_predicates - ref_relations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
