"""Phase 10 -- the FastAPI surface boots keyless and round-trips."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from conftest import nvidia_triplets
from service.assembly import build_memory_layer
from llm.mock import MockProvider


@pytest.fixture
def client(monkeypatch):
    prov = MockProvider({
        "INPUT TEXT": nvidia_triplets([
            ("Sector 4", "Zone", "zoning limit", "30 stories", "METRIC"),
        ]),
        "named entities": json.dumps({"entities": ["Sector 4"]}),        # entity agent
        "using ONLY the FACTS": json.dumps({"answer": "30 stories"}),     # answerer agent
    }, default="grounded answer")
    import service.deps as deps

    deps.reset_layer()
    monkeypatch.setattr(deps, "_LAYER", build_memory_layer(provider=prov))
    monkeypatch.setenv("GC_ENABLED", "0")
    from service.main import app

    with TestClient(app) as c:
        yield c
    deps.reset_layer()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ingest_then_head(client):
    r = client.post("/ingest/text", json={"text": "Resolution 402 ...", "source_id": "municipal:res-402"})
    assert r.status_code == 200
    assert r.json()["by_disposition"].get("AUTO_COMMIT") == 1

    # "zoning limit" is a curated functional (single-valued) attribute -> ONE
    # cardinality, so the key stays a plain 'entity::attribute' pair.
    h = client.get("/memory/head/sector_4::zoning_limit")
    assert h.status_code == 200 and h.json()["object"] == "30 stories"


def test_qa_uses_graph_only(client):
    client.post("/ingest/text", json={"text": "x", "source_id": "municipal:res-402"})
    r = client.post("/qa", json={"query": "What is the Sector 4 zoning limit?"})
    assert r.status_code == 200
    body = r.json()
    assert body["entities"] == ["Sector 4"]


def test_logs_and_stats_endpoints(client):
    client.post("/ingest/text", json={"text": "Resolution 402 ...", "source_id": "municipal:res-402"})

    stats = client.get("/logs/stats").json()
    # two entity nodes (Sector 4, and "30 stories" -- aligned with NVIDIA's
    # lc_graph.py, a typed object is a node like any other, no METRIC carve-out)
    # + one reified relation node (zoning limit)
    assert stats["entities_total"] == 2
    assert stats["relations_total"] == 1
    assert stats["nodes_total"] == 3
    assert stats["documents_ingested"] == 1
    assert stats["nodes_by_state"] == {"ACTIVE": 1}   # entity nodes carry no state

    logs = client.get("/logs", params={"contains": "GATE"}).json()
    assert logs["count"] >= 1
    assert any("GATE" in line["message"] for line in logs["lines"])
