"""Read the memory graph; revert versions."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dag_kb import RevertRejected
from dag_kb import Actor

from ..deps import get_layer

router = APIRouter()


@router.get("/head/{semantic_key}")
def head(semantic_key: str):
    layer = get_layer()
    h = layer.heads.get(semantic_key)
    if not h:
        raise HTTPException(404, "no such semantic key")
    r = layer.store.get(h.record_id)
    return {"semantic_key": semantic_key, "record_id": h.record_id, "owner_seq": h.owner_seq,
            "predicate": r.predicate, "object": r.obj, "tier": r.tier.name,
            "state": layer.dag.get_state(h.record_id).value, "raw_citation": r.raw_citation}


@router.get("/node/{record_id}")
def node(record_id: str):
    layer = get_layer()
    if not layer.store.has(record_id):
        raise HTTPException(404, "unknown record")
    r = layer.store.get(record_id)
    state = layer.dag.get_state(record_id).value if layer.dag.has_node(record_id) else "EVICTED"
    return {"record_id": record_id, "semantic_key": r.semantic_key, "object": r.obj,
            "owner_seq": r.owner_seq, "parent_ids": list(r.parent_ids), "tier": r.tier.name,
            "state": state, "txn_id": r.txn_id}


@router.get("/versions/{semantic_key}")
def versions(semantic_key: str):
    layer = get_layer()
    out = []
    for r in layer.store.all_versions(semantic_key):
        st = layer.dag.get_state(r.record_id).value if layer.dag.has_node(r.record_id) else "EVICTED"
        out.append({"record_id": r.record_id, "owner_seq": r.owner_seq, "object": r.obj,
                    "state": st, "valid_from": r.valid_from.isoformat()})
    return out


@router.get("/lineage/{record_id}")
def lineage(record_id: str, deps: str = ""):
    layer = get_layer()
    keys = [k for k in deps.split(",") if k]
    try:
        return layer.dag.frontier(record_id, keys) if keys else {
            "direct_dependents": layer.dag.direct_dependents(record_id)
        }
    except Exception as exc:  # LineageIncomplete etc
        raise HTTPException(409, str(exc))


class RevertIn(BaseModel):
    semantic_key: str
    target_record_id: str
    actor: str = "SYSTEM_2"


@router.post("/revert")
def revert(body: RevertIn):
    layer = get_layer()
    try:
        layer.reverter.revert(body.semantic_key, body.target_record_id,
                              actor=Actor(body.actor))
    except RevertRejected as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True, "head": layer.heads.get(body.semantic_key).record_id}


class RevertTxnIn(BaseModel):
    txn_id: str
    actor: str = "SYSTEM_2"


@router.post("/revert-txn")
def revert_txn(body: RevertTxnIn):
    layer = get_layer()
    affected = layer.reverter.revert_txn(body.txn_id, actor=Actor(body.actor))
    return {"ok": True, "reverted_keys": affected}
