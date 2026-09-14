"""Ingestion pipeline (Path A). Depends on ``dag_kb`` and ``llm`` only."""

from __future__ import annotations

from .stage3_delta import DeltaClassifier
from .stage1_extract import Extractor, semantic_key
from .ingest import IngestionPipeline, IngestReport
from .stage2_resolve import Resolver
from .stage0_sources import DEFAULT_POLICIES, RawItem, SourcePolicy, SourceRegistry

__all__ = [
    "DeltaClassifier", "Extractor", "semantic_key", "IngestionPipeline",
    "IngestReport", "Resolver", "RawItem", "SourcePolicy", "SourceRegistry",
    "DEFAULT_POLICIES",
]
