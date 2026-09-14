"""Phase 1 -- the semantic router: semantic_key -> currently-authorized record_id.

This is ``H(x)`` in the paper. The owner guard is the whole safety of the write
path (technical-plan section 1.5, Table 3):

    an owner accepts a head update only when the writer identity matches the
    declared owner and the sequence number strictly increases; duplicate owner
    sequences are conflicts.

Optionally persisted to the same sqlite file as the record store.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable

from ..errors import HeadConflict
from ..types import HeadEntry


_SCHEMA = """
CREATE TABLE IF NOT EXISTS heads (
    semantic_key TEXT PRIMARY KEY,
    record_id    TEXT NOT NULL,
    owner        TEXT NOT NULL,
    owner_seq    INTEGER NOT NULL
);
"""


class HeadIndex:
    """In-memory head index with optional sqlite persistence.

    Pass ``conn`` (a live :class:`sqlite3.Connection`, typically the record
    store's) to persist and reload; omit it for a pure in-memory index.
    """

    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self._conn = conn
        self._heads: dict[str, HeadEntry] = {}
        if conn is not None:
            conn.executescript(_SCHEMA)
            conn.commit()
            for row in conn.execute("SELECT * FROM heads"):
                self._heads[row["semantic_key"] if isinstance(row, sqlite3.Row) else row[0]] = (
                    HeadEntry(
                        semantic_key=row["semantic_key"],
                        record_id=row["record_id"],
                        owner=row["owner"],
                        owner_seq=int(row["owner_seq"]),
                    )
                    if isinstance(row, sqlite3.Row)
                    else HeadEntry(row[0], row[1], row[2], int(row[3]))
                )

    # ------------------------------------------------------------------ #
    def update(self, entry: HeadEntry, *, writer_identity: str) -> HeadEntry:
        """Move the head. Raises :class:`HeadConflict` if the guard fails."""
        if writer_identity != entry.owner:
            raise HeadConflict(
                f"writer {writer_identity!r} != declared owner {entry.owner!r} for {entry.semantic_key}"
            )
        current = self._heads.get(entry.semantic_key)
        if current is not None and entry.owner_seq <= current.owner_seq:
            raise HeadConflict(
                f"owner_seq {entry.owner_seq} does not exceed current {current.owner_seq} "
                f"for {entry.semantic_key}"
            )
        self._heads[entry.semantic_key] = entry
        if self._conn is not None:
            self._conn.execute(
                "INSERT INTO heads (semantic_key, record_id, owner, owner_seq) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(semantic_key) DO UPDATE SET record_id=excluded.record_id, "
                "owner=excluded.owner, owner_seq=excluded.owner_seq",
                (entry.semantic_key, entry.record_id, entry.owner, entry.owner_seq),
            )
            self._conn.commit()
        return entry

    def force_set(self, entry: HeadEntry) -> None:
        """Bypass the monotonic guard. Internal use only -- transactional
        rollback of a failed commit (dag_kb.write.gate). Never call from
        application code."""
        self._heads[entry.semantic_key] = entry
        if self._conn is not None:
            self._conn.execute(
                "INSERT INTO heads (semantic_key, record_id, owner, owner_seq) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(semantic_key) DO UPDATE SET record_id=excluded.record_id, "
                "owner=excluded.owner, owner_seq=excluded.owner_seq",
                (entry.semantic_key, entry.record_id, entry.owner, entry.owner_seq),
            )
            self._conn.commit()

    def get(self, semantic_key: str) -> HeadEntry | None:
        return self._heads.get(semantic_key)

    def batch_get(self, keys: Iterable[str]) -> tuple[dict[str, HeadEntry], list[str]]:
        """Return ``(found, missing)``. A non-empty ``missing`` list means the
        caller must fail closed (PLANFENCE Algorithm 1, line 6)."""
        found: dict[str, HeadEntry] = {}
        missing: list[str] = []
        for k in keys:
            h = self._heads.get(k)
            if h is None:
                missing.append(k)
            else:
                found[k] = h
        return found, missing

    def all_keys(self) -> list[str]:
        return list(self._heads)
