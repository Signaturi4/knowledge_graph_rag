"""End-to-end smoke: real claude_bridge extraction -> reconciliation gate -> DAG.

Run:  LLM_PROVIDER=claude_bridge  CLAUDE_BIN=/abs/path/claude  python scripts/smoke_dagkb_e2e.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from observability import configure_logging  # noqa: E402

configure_logging()

from service.assembly import build_memory_layer  # noqa: E402
from dag_kb import ValidationOutcome  # noqa: E402

mem = build_memory_layer()

print("== ingest 1 ==")
r1 = mem.pipeline.ingest_text(
    "City Council Resolution 402 sets the maximum building height in Sector 4 to 30 stories, "
    "effective 2026-01-01.",
    source_id="municipal:res-402",
)
print(r1.summary())
for k in mem.heads.all_keys():
    h = mem.heads.get(k)
    print(f"  head {k} -> {mem.store.get(h.record_id).obj!r}  (tier {mem.store.get(h.record_id).tier.name})")

print("\n== ingest 2 (amendment) ==")
r2 = mem.pipeline.ingest_text(
    "Resolution 511 amends the Sector 4 maximum building height to 45 stories as of 2026-06-01.",
    source_id="municipal:res-511",
)
print(r2.summary())
print("  pending SYSTEM_2 items:", [(i.kind.value, i.semantic_key) for i in mem.queue.pending()])

print("\n== PLANFENCE demo ==")
keys = mem.heads.all_keys()
if keys:
    root = mem.heads.get(keys[0]).record_id
    res = mem.planfence.validate(root, [keys[0]], replanned=False)
    print(f"  validate({keys[0]}) -> {res.outcome.value}  ({res.reason})")

print("\nOK")
