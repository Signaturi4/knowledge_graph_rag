"""Phase 7 -- append-only audit log.

Every state transition is recorded (technical-plan section 6d). One JSON line per
transition; never rewritten.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from ..types import AuditRow


class AuditLog:
    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self._rows: list[AuditRow] = []
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def record(self, row: AuditRow) -> None:
        self._rows.append(row)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                d = asdict(row)
                d["from_state"] = row.from_state.value if row.from_state else None
                d["to_state"] = row.to_state.value if row.to_state else None
                d["actor"] = row.actor.value
                d["ts"] = row.ts.isoformat()
                fh.write(json.dumps(d) + "\n")

    def rows(self) -> list[AuditRow]:
        return list(self._rows)

    def for_node(self, node_id: str) -> list[AuditRow]:
        return [r for r in self._rows if r.node_id == node_id]
