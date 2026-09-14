"""SYSTEM_2 review queue -- approve / reject a pending supersession."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dag_kb import QueueItemType, Routing

from ..deps import get_layer

router = APIRouter()


@router.get("/queue")
def queue():
    layer = get_layer()
    return [
        {"id": i.id, "kind": i.kind.value, "semantic_key": i.semantic_key, "payload": i.payload}
        for i in layer.queue.pending(routing=Routing.SYSTEM_2)
    ]


class Decision(BaseModel):
    verdict: str  # approve | reject | defer


@router.post("/{item_id}")
def decide(item_id: str, body: Decision):
    layer = get_layer()
    item = layer.queue.get(item_id)
    if item is None:
        raise HTTPException(404, "unknown queue item")
    if body.verdict == "approve" and item.kind is QueueItemType.SUPERSESSION_REVIEW:
        res = layer.gate.approve_review(item_id)
        return {"ok": True, "committed": res.committed_record_id}
    layer.queue.resolve(item_id, body.verdict)
    return {"ok": True, "resolution": body.verdict}
