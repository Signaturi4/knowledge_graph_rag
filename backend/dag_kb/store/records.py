"""Phase 1 -- ADD-only immutable record spine.

Two back-ends behind one ABC (:class:`InMemoryRecordStore`, :class:`SqliteRecordStore`);
identical test vectors must pass on both (technical-plan Phase 2: "any DB substrate
must pass the identical vectors").

Contract (technical-plan section 1.5):
  - records are content-addressed and never mutated or deleted by the store;
  - ``put`` is idempotent on an identical ``record_id``;
  - re-``put`` of the same id with different content raises :class:`ImmutableViolation`;
  - GC archival is a separate ``mark_cold`` flag, not a delete.
"""

from __future__ import annotations

import abc
import json
import sqlite3
from datetime import datetime
from typing import Iterator

from ..errors import ImmutableViolation
from ..types import Cardinality, Record, Tier


# --------------------------------------------------------------------------- #
# (de)serialization
# --------------------------------------------------------------------------- #
def record_to_row(r: Record) -> dict:
    return {
        "record_id": r.record_id,
        "semantic_key": r.semantic_key,
        "predicate": r.predicate,
        "obj": json.dumps(r.obj, default=str),
        "parent_ids": json.dumps(list(r.parent_ids)),
        "owner": r.owner,
        "owner_seq": r.owner_seq,
        "tier": int(r.tier),
        "auto_update": int(r.auto_update),
        "record_type": r.record_type,
        "valid_from": r.valid_from.isoformat(),
        "source_id": r.source_id,
        "raw_citation": r.raw_citation,
        "txn_id": r.txn_id,
        "subject_ref": r.subject_ref,
        "object_ref": r.object_ref if r.object_ref is not None else "",
        "cardinality": r.cardinality.value,
        "provenance": json.dumps(dict(r.provenance), default=str),
        "created_at": r.created_at.isoformat(),
    }


def row_to_record(row: dict) -> Record:
    return Record(
        record_id=row["record_id"],
        semantic_key=row["semantic_key"],
        predicate=row["predicate"],
        obj=json.loads(row["obj"]),
        parent_ids=tuple(json.loads(row["parent_ids"])),
        owner=row["owner"],
        owner_seq=int(row["owner_seq"]),
        tier=Tier(int(row["tier"])),
        auto_update=bool(row["auto_update"]),
        record_type=row["record_type"],
        valid_from=datetime.fromisoformat(row["valid_from"]),
        source_id=row["source_id"],
        raw_citation=row["raw_citation"],
        txn_id=row["txn_id"],
        subject_ref=row.get("subject_ref", "") or "",
        object_ref=(row.get("object_ref") or None) or None,
        cardinality=Cardinality(row.get("cardinality") or Cardinality.MANY.value),
        provenance=json.loads(row["provenance"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _identity_blob(r: Record) -> str:
    """Everything that must not change for a fixed record_id."""
    row = record_to_row(r)
    for volatile in ("created_at", "provenance", "raw_citation", "source_id", "valid_from"):
        row.pop(volatile, None)
    return json.dumps(row, sort_keys=True)


# --------------------------------------------------------------------------- #
# ABC
# --------------------------------------------------------------------------- #
class RecordStore(abc.ABC):
    @abc.abstractmethod
    def put(self, record: Record) -> str: ...

    @abc.abstractmethod
    def get(self, record_id: str) -> Record: ...

    @abc.abstractmethod
    def has(self, record_id: str) -> bool: ...

    @abc.abstractmethod
    def all_versions(self, semantic_key: str) -> list[Record]:
        """Every stored version of a key, ascending by owner_seq."""

    @abc.abstractmethod
    def mark_cold(self, record_id: str, cold: bool = True) -> None: ...

    @abc.abstractmethod
    def is_cold(self, record_id: str) -> bool: ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Record]: ...


# --------------------------------------------------------------------------- #
# in-memory
# --------------------------------------------------------------------------- #
class InMemoryRecordStore(RecordStore):
    def __init__(self) -> None:
        self._records: dict[str, Record] = {}
        self._cold: set[str] = set()

    def put(self, record: Record) -> str:
        existing = self._records.get(record.record_id)
        if existing is not None:
            if _identity_blob(existing) != _identity_blob(record):
                raise ImmutableViolation(
                    f"record_id {record.record_id} already stored with different content"
                )
            return record.record_id
        self._records[record.record_id] = record
        return record.record_id

    def get(self, record_id: str) -> Record:
        return self._records[record_id]

    def has(self, record_id: str) -> bool:
        return record_id in self._records

    def all_versions(self, semantic_key: str) -> list[Record]:
        rs = [r for r in self._records.values() if r.semantic_key == semantic_key]
        return sorted(rs, key=lambda r: r.owner_seq)

    def mark_cold(self, record_id: str, cold: bool = True) -> None:
        if cold:
            self._cold.add(record_id)
        else:
            self._cold.discard(record_id)

    def is_cold(self, record_id: str) -> bool:
        return record_id in self._cold

    def __iter__(self) -> Iterator[Record]:
        return iter(list(self._records.values()))


# --------------------------------------------------------------------------- #
# sqlite
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    record_id    TEXT PRIMARY KEY,
    semantic_key TEXT NOT NULL,
    predicate    TEXT NOT NULL,
    obj          TEXT NOT NULL,
    parent_ids   TEXT NOT NULL,
    owner        TEXT NOT NULL,
    owner_seq    INTEGER NOT NULL,
    tier         INTEGER NOT NULL,
    auto_update  INTEGER NOT NULL,
    record_type  TEXT NOT NULL,
    valid_from   TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    raw_citation TEXT NOT NULL,
    txn_id       TEXT NOT NULL,
    subject_ref  TEXT NOT NULL DEFAULT '',
    object_ref   TEXT NOT NULL DEFAULT '',
    cardinality  TEXT NOT NULL DEFAULT 'many',
    provenance   TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    cold         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_records_key ON records(semantic_key, owner_seq);
"""


class SqliteRecordStore(RecordStore):
    def __init__(
        self,
        path: str = ":memory:",
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """``conn`` lets the store share one connection with :class:`HeadIndex`
        (same file, one lock). Otherwise a connection is opened on ``path``."""
        if conn is not None:
            self._conn = conn
        else:
            self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def put(self, record: Record) -> str:
        cur = self._conn.execute(
            "SELECT * FROM records WHERE record_id = ?", (record.record_id,)
        )
        existing = cur.fetchone()
        if existing is not None:
            if _identity_blob(row_to_record(dict(existing))) != _identity_blob(record):
                raise ImmutableViolation(
                    f"record_id {record.record_id} already stored with different content"
                )
            return record.record_id
        row = record_to_row(record)
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        self._conn.execute(f"INSERT INTO records ({cols}) VALUES ({marks})", tuple(row.values()))
        self._conn.commit()
        return record.record_id

    def get(self, record_id: str) -> Record:
        cur = self._conn.execute("SELECT * FROM records WHERE record_id = ?", (record_id,))
        row = cur.fetchone()
        if row is None:
            raise KeyError(record_id)
        return row_to_record(dict(row))

    def has(self, record_id: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM records WHERE record_id = ?", (record_id,))
        return cur.fetchone() is not None

    def all_versions(self, semantic_key: str) -> list[Record]:
        cur = self._conn.execute(
            "SELECT * FROM records WHERE semantic_key = ? ORDER BY owner_seq", (semantic_key,)
        )
        return [row_to_record(dict(r)) for r in cur.fetchall()]

    def mark_cold(self, record_id: str, cold: bool = True) -> None:
        self._conn.execute(
            "UPDATE records SET cold = ? WHERE record_id = ?", (int(cold), record_id)
        )
        self._conn.commit()

    def is_cold(self, record_id: str) -> bool:
        cur = self._conn.execute("SELECT cold FROM records WHERE record_id = ?", (record_id,))
        row = cur.fetchone()
        return bool(row and row["cold"])

    def __iter__(self) -> Iterator[Record]:
        cur = self._conn.execute("SELECT * FROM records")
        return iter([row_to_record(dict(r)) for r in cur.fetchall()])
