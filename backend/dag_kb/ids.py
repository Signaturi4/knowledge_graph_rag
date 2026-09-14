"""Content-addressed record identity.

A ``record_id`` is a deterministic hash over the fields that define *this exact
version*: the semantic key, the value, the sorted parent IDs, and the owner
sequence number. Two ingests of the same fact with the same lineage collapse to
the same id (idempotent); any change to value or lineage yields a new id, which
is what makes the parent links in :class:`dag_kb.types.Record` immutable and
verifiable.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_record_id(
    *,
    semantic_key: str,
    predicate: str,
    obj: Any,
    parent_ids: Iterable[str],
    owner: str,
    owner_seq: int,
) -> str:
    h = hashlib.sha256()
    h.update(_canonical(
        {
            "k": semantic_key,
            "p": predicate,
            "o": obj,
            "parents": sorted(parent_ids),
            "owner": owner,
            "seq": owner_seq,
        }
    ).encode("utf-8"))
    return "rec_" + h.hexdigest()[:32]


def new_txn_id() -> str:
    import uuid

    return "txn_" + uuid.uuid4().hex[:16]
