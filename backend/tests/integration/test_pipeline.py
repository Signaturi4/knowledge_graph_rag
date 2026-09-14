"""Phase 9 -- the ingestion pipeline end-to-end with a mock LLM.

Extractor replies are NVIDIA-format Python-literal lists of 5-tuples
(subject, subject_type, predicate, object, object_type) -- see
pipeline.stage1_extract / tests.conftest.nvidia_triplets.
"""
from __future__ import annotations

import json

from dag_kb import Cardinality, Disposition, Routing
from service.assembly import build_memory_layer
from llm.mock import MockProvider, UNAVAILABLE
from pipeline import semantic_key

from conftest import nvidia_triplets


def _extractor_responses(rows):
    return {"INPUT TEXT": nvidia_triplets(rows)}


def test_ingest_text_commits_first_facts():
    prov = MockProvider(_extractor_responses([
        ("Sector 4", "Zone", "zoning limit", "30 stories", "METRIC"),
    ]))
    mem = build_memory_layer(provider=prov)
    report = mem.pipeline.ingest_text("City Council Resolution 402 ...", "municipal:res-402")
    assert report.drafts == 1
    assert report.results[0].disposition is Disposition.AUTO_COMMIT
    # "zoning limit" is a curated functional (single-valued) attribute -> ONE
    key = semantic_key("Sector 4", "zoning limit", cardinality=Cardinality.ONE)
    assert mem.heads.get(key) is not None


def test_ingest_second_value_uses_delta_then_gate():
    prov = MockProvider({
        "INPUT TEXT": nvidia_triplets([
            ("Sector 4", "Zone", "zoning limit", "45 stories", "METRIC"),
        ]),
        "CURRENT:": json.dumps({"delta": "SUPERSEDING_CHANGE", "rationale": "value changed"}),
    })
    mem = build_memory_layer(provider=prov)
    # seed v0
    mem.pipeline.ingest_text("res 402: 30 stories", "vendor_doc:v1")  # T2 auto
    # (mock returns the 45-stories triplet again) -> delta -> gate
    r2 = mem.pipeline.ingest_text("amendment: 45 stories", "vendor_doc:v1")
    disp = r2.results[0].disposition
    assert disp is Disposition.AUTO_COMMIT  # T2 within-tier auto


def test_extractor_unavailable_routes_to_system_2_not_commit():
    prov = MockProvider({"INPUT TEXT": UNAVAILABLE})
    mem = build_memory_layer(provider=prov)
    report = mem.pipeline.ingest_text("anything", "municipal:res-402")
    assert report.results[0].disposition is Disposition.REJECT_FAIL_CLOSED
    assert mem.queue.pending(routing=Routing.SYSTEM_2)
    assert mem.heads.all_keys() == []
