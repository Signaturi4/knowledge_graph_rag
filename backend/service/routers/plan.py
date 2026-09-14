"""PLANFENCE use-path endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ..deps import get_layer

router = APIRouter()


class ValidateIn(BaseModel):
    root_id: str
    declared_deps: list[str]
    replanned: bool = False


@router.post("/validate")
def validate(body: ValidateIn):
    layer = get_layer()
    res = layer.planfence.validate(body.root_id, body.declared_deps, replanned=body.replanned)
    return {
        "outcome": res.outcome.value,
        "frontier": dict(res.frontier),
        "heads": dict(res.heads),
        "mismatched_keys": list(res.mismatched_keys),
        "reason": res.reason,
    }
