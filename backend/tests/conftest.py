from __future__ import annotations

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

from datetime import datetime, timedelta, timezone

import pytest

from dag_kb import Tier, TripletDraft
from service.assembly import build_memory_layer
from llm.mock import MockProvider


@pytest.fixture
def mem():
    """In-memory stack with a mock LLM."""
    return build_memory_layer(provider=MockProvider(default=""))


def nvidia_triplets(rows: list[tuple]) -> str:
    """Build a mock extractor reply in the format ``pipeline.stage1_extract``
    actually expects: a Python-literal list of 5-tuples
    ``(subject, subject_type, predicate, object, object_type)``, parsed via
    ``ast.literal_eval`` (NVIDIA's ``preprocessor.py::process_response``
    contract) -- NOT JSON."""
    return repr([tuple(r) for r in rows])


@pytest.fixture
def draft():
    def _make(key="sector_4::zoning_limit", obj=30, *, tier=Tier.T0, auto=False,
              days_ago=0, source_id="municipal:res-402", parents=()):
        return TripletDraft(
            semantic_key=key,
            predicate="zoning_limit",
            obj=obj,
            tier=tier,
            auto_update=auto,
            source_id=source_id,
            raw_citation="City Council Resolution #402",
            valid_from=datetime.now(timezone.utc) - timedelta(days=days_ago),
            parent_ids=tuple(parents),
        )

    return _make
