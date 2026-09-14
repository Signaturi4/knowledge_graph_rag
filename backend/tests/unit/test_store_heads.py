"""Phase 1 -- R1.* : immutable store + head guard, on both back-ends."""
from __future__ import annotations

import dataclasses
import sqlite3
from datetime import datetime, timezone

import pytest

from dag_kb import HeadConflict, ImmutableViolation
from dag_kb import HeadIndex
from dag_kb.ids import compute_record_id
from dag_kb.store import InMemoryRecordStore, SqliteRecordStore
from dag_kb import HeadEntry, Record, Tier


def _rec(seq: int, obj, owner="municipal"):
    rid = compute_record_id(semantic_key="k", predicate="p", obj=obj,
                            parent_ids=(), owner=owner, owner_seq=seq)
    return Record(record_id=rid, semantic_key="k", predicate="p", obj=obj, parent_ids=(),
                  owner=owner, owner_seq=seq, tier=Tier.T1, auto_update=True,
                  record_type="asserted", valid_from=datetime.now(timezone.utc),
                  source_id="s", raw_citation="c", txn_id="t")


@pytest.mark.parametrize("factory", [InMemoryRecordStore, lambda: SqliteRecordStore(":memory:")])
def test_add_only_and_versions(factory):
    store = factory()
    r1, r2, r3 = _rec(0, 10), _rec(1, 20), _rec(2, 30)
    for r in (r1, r2, r3):
        store.put(r)
    assert [r.owner_seq for r in store.all_versions("k")] == [0, 1, 2]
    # idempotent
    assert store.put(r2) == r2.record_id
    # superseded still readable
    assert store.get(r1.record_id).obj == 10


@pytest.mark.parametrize("factory", [InMemoryRecordStore, lambda: SqliteRecordStore(":memory:")])
def test_immutable_violation(factory):
    store = factory()
    r = _rec(0, 10)
    store.put(r)
    clash = dataclasses.replace(r, obj=999)  # same record_id, different content
    with pytest.raises(ImmutableViolation):
        store.put(clash)


def test_head_guard_identity_and_monotonic():
    h = HeadIndex()
    e0 = HeadEntry("k", "rec_a", "municipal", 0)
    h.update(e0, writer_identity="municipal")

    with pytest.raises(HeadConflict):  # wrong writer
        h.update(HeadEntry("k", "rec_b", "municipal", 1), writer_identity="intruder")
    with pytest.raises(HeadConflict):  # non-increasing seq
        h.update(HeadEntry("k", "rec_b", "municipal", 0), writer_identity="municipal")

    h.update(HeadEntry("k", "rec_b", "municipal", 1), writer_identity="municipal")
    assert h.get("k").record_id == "rec_b"


def test_head_persists_across_reload():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    h1 = HeadIndex(conn)
    h1.update(HeadEntry("k", "rec_a", "municipal", 3), writer_identity="municipal")
    h2 = HeadIndex(conn)
    assert h2.get("k").owner_seq == 3


def test_batch_get_reports_missing():
    h = HeadIndex()
    h.update(HeadEntry("a", "rec_a", "o", 0), writer_identity="o")
    found, missing = h.batch_get(["a", "b"])
    assert set(found) == {"a"} and missing == ["b"]
