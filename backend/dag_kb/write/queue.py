"""Section 6b stage 7 -- the reconciliation work queue.

SYSTEM_1 items are processed automatically; SYSTEM_2 items wait for a human
verdict; CONTRADICTION items are held for the Conflict Resolution Engine.
Optionally persisted as JSONL.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Iterable

from ..types import QueueItem, QueueItemType, Routing


class WorkQueue:
    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self._items: dict[str, QueueItem] = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    self._items[d["id"]] = QueueItem(
                        kind=QueueItemType(d["kind"]),
                        routed_to=Routing(d["routed_to"]),
                        semantic_key=d["semantic_key"],
                        payload=d.get("payload", {}),
                        id=d["id"],
                        resolved=d.get("resolved", False),
                        resolution=d.get("resolution", ""),
                    )
        elif path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # ------------------------------------------------------------------ #
    def enqueue(self, item: QueueItem) -> QueueItem:
        self._items[item.id] = item
        self._flush()
        return item

    def get(self, item_id: str) -> QueueItem | None:
        return self._items.get(item_id)

    def pending(
        self,
        *,
        routing: Routing | None = None,
        kind: QueueItemType | None = None,
    ) -> list[QueueItem]:
        out = [i for i in self._items.values() if not i.resolved]
        if routing is not None:
            out = [i for i in out if i.routed_to is routing]
        if kind is not None:
            out = [i for i in out if i.kind is kind]
        return out

    def resolve(self, item_id: str, resolution: str = "done") -> None:
        it = self._items[item_id]
        it.resolved = True
        it.resolution = resolution
        self._flush()

    def references_record(self, record_id: str) -> bool:
        """Used by GC: is any unresolved item locking this record?"""
        for it in self._items.values():
            if it.resolved:
                continue
            if record_id in (it.payload.get("node_id"), it.payload.get("record_id")):
                return True
        return False

    def all(self) -> Iterable[QueueItem]:
        return list(self._items.values())

    # ------------------------------------------------------------------ #
    def _flush(self) -> None:
        if not self.path:
            return
        with open(self.path, "w", encoding="utf-8") as fh:
            for it in self._items.values():
                d = asdict(it)
                d["kind"] = it.kind.value
                d["routed_to"] = it.routed_to.value
                d.pop("created_at", None)
                fh.write(json.dumps(d, default=str) + "\n")
