"""Composition root -- wire the whole deterministic memory layer into one object.

Used by the FastAPI service and the test suite. In-memory by default; pass
``data_dir`` for a persisted stack: sqlite records + heads (one shared
connection), JSONL audit + queue, a GraphML snapshot of the DAG that is
**reloaded on startup** and rewritten after every commit / revert, and a
cold-storage archive directory.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Callable

from dag_kb import (
    AuditLog,
    BoundedRetriever,
    Cascade,
    GarbageCollector,
    HeadIndex,
    KnowledgeDAG,
    PlanFence,
    ReconciliationGate,
    ReplanOrchestrator,
    Reverter,
    WorkQueue,
)
from dag_kb.store import InMemoryRecordStore, RecordStore, SqliteRecordStore
from llm.provider import LLMProvider
from pipeline import IngestionPipeline, Resolver, SourceRegistry

_GRAPHML = "dag.graphml"


@dataclass
class MemoryLayer:
    provider: LLMProvider
    store: RecordStore
    heads: HeadIndex
    dag: KnowledgeDAG
    audit: AuditLog
    queue: WorkQueue
    cascade: Cascade
    gate: ReconciliationGate
    planfence: PlanFence
    replan: ReplanOrchestrator
    retriever: BoundedRetriever
    reverter: Reverter
    gc: GarbageCollector
    pipeline: IngestionPipeline
    data_dir: str | None = None

    def snapshot_graphml(self, path: str | None = None) -> None:
        path = path or (os.path.join(self.data_dir, _GRAPHML) if self.data_dir else None)
        if path:
            self.dag.write_graphml(path)


def build_memory_layer(
    *,
    data_dir: str | None = None,
    provider: LLMProvider | None = None,
    retention_days: int = 30,
    notifier: Callable[[dict], None] | None = None,
    registry: SourceRegistry | None = None,
) -> MemoryLayer:
    if provider is None:
        from llm import get_provider

        provider = get_provider()
    from llm.logging_provider import LoggingProvider

    if not isinstance(provider, LoggingProvider):
        provider = LoggingProvider(provider)

    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        conn = sqlite3.connect(os.path.join(data_dir, "records.sqlite"), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        store: RecordStore = SqliteRecordStore(conn=conn)
        heads = HeadIndex(conn)
        audit = AuditLog(os.path.join(data_dir, "audit.jsonl"))
        queue = WorkQueue(os.path.join(data_dir, "queue.jsonl"))
        cold_dir = os.path.join(data_dir, "cold")
        graphml_path = os.path.join(data_dir, _GRAPHML)
    else:
        store = InMemoryRecordStore()
        heads = HeadIndex()
        audit = AuditLog()
        queue = WorkQueue()
        cold_dir = os.path.join(os.getcwd(), ".cold_mem")
        graphml_path = None

    # Reload the DAG (node state + edges) from the last snapshot so a restart is
    # consistent with the persisted records + heads (STATUS.md P0 #2).
    if graphml_path and os.path.exists(graphml_path):
        dag = KnowledgeDAG.read_graphml(graphml_path, store)
    else:
        dag = KnowledgeDAG(store)

    cascade = Cascade(dag, queue, audit)

    def _snapshot(*_a) -> None:
        if graphml_path:
            dag.write_graphml(graphml_path)

    gate = ReconciliationGate(store, heads, dag, queue, audit, cascade,
                              notifier=notifier, on_commit=_snapshot)
    planfence = PlanFence(dag, heads)
    replan = ReplanOrchestrator(planfence, dag=dag, cascade=cascade)
    retriever = BoundedRetriever(dag)
    reverter = Reverter(store, heads, dag, cascade, audit,
                        retention_days=retention_days, on_change=_snapshot)
    gc = GarbageCollector(dag, store, queue, audit, cold_dir=cold_dir,
                          retention_days=retention_days)
    resolver = Resolver(store, heads)
    pipe = IngestionPipeline(provider, resolver, gate, registry=registry)

    return MemoryLayer(provider, store, heads, dag, audit, queue, cascade, gate,
                       planfence, replan, retriever, reverter, gc, pipe, data_dir)
