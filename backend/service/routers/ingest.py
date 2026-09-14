"""Path A -- ingestion endpoints."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..deps import get_layer

router = APIRouter()


class TextIn(BaseModel):
    text: str
    source_id: str
    owner: str | None = None
    canonical_subject: str | None = None


class DirIn(BaseModel):
    directory: str
    source_id: str = "arxiv"
    owner: str | None = None


@router.post("/text")
def ingest_text(body: TextIn):
    layer = get_layer()
    report = layer.pipeline.ingest_text(body.text, body.source_id, owner=body.owner,
                                        canonical_subject=body.canonical_subject)
    if os.environ.get("DATA_DIR"):
        layer.snapshot_graphml(os.path.join(os.environ["DATA_DIR"], "dag.graphml"))
    return report.summary()


@router.post("/directory")
def ingest_directory(body: DirIn):
    layer = get_layer()
    if not os.path.isdir(body.directory):
        raise HTTPException(404, f"no such directory {body.directory}")
    pdfs = [f for f in os.listdir(body.directory) if f.lower().endswith(".pdf")]
    if not pdfs:
        raise HTTPException(400, "no PDF files in directory")
    summaries = []
    for name in pdfs:
        rep = layer.pipeline.ingest_pdf(os.path.join(body.directory, name), body.source_id)
        summaries.append(rep.summary())
    if os.environ.get("DATA_DIR"):
        layer.snapshot_graphml(os.path.join(os.environ["DATA_DIR"], "dag.graphml"))
    return {"ingested": summaries}


@router.get("/queue")
def queue(routing: str | None = None, kind: str | None = None):
    from dag_kb import QueueItemType, Routing

    layer = get_layer()
    r = Routing(routing) if routing else None
    k = QueueItemType(kind) if kind else None
    return [
        {"id": i.id, "kind": i.kind.value, "routed_to": i.routed_to.value,
         "semantic_key": i.semantic_key, "payload": i.payload}
        for i in layer.queue.pending(routing=r, kind=k)
    ]
