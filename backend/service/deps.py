"""Shared application state: one process-wide :class:`MemoryLayer`."""

from __future__ import annotations

import os

from .assembly import MemoryLayer, build_memory_layer

_LAYER: MemoryLayer | None = None


def get_layer() -> MemoryLayer:
    global _LAYER
    if _LAYER is None:
        data_dir = os.environ.get(
            "DATA_DIR",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        )
        dagkb_dir = os.path.join(data_dir, "dagkb") if not data_dir.endswith("dagkb") else data_dir
        _LAYER = build_memory_layer(
            data_dir=dagkb_dir,
            retention_days=int(os.environ.get("RETENTION_DAYS", "30")),
        )
    return _LAYER


def reset_layer() -> None:  # tests
    global _LAYER
    _LAYER = None
