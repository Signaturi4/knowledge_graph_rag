"""Deterministic DAG knowledgebase core -- the public surface.

Pure library: depends only on ``networkx``, the stdlib, and ``sqlite3``. No LLM,
no HTTP, no FastAPI -- this is the PRD section 3 "NetworkX prototyping standard"
and stays runnable in isolation.

Internal layout (import from here, not from the submodules):

    dag_kb/
      types.py errors.py ids.py          -- foundations
      store/     records.py  heads.py    -- Phase 1: immutable spine + head index
      graph/     dag.py                  -- Phase 2: the DAG + frontier traversal
      write/     precedence gate cascade queue   -- Path A: the write path
      validate/  planfence  replan       -- Path B: the use path (PLANFENCE)
      retrieval/ bounded.py              -- Phase 5: bounded-context retrieval
      lifecycle/ audit  revert  gc       -- Phases 6-7: audit, revert, GC
"""

from __future__ import annotations

from .errors import (
    DagKBError,
    HeadConflict,
    ImmutableViolation,
    LineageIncomplete,
    MalformedDraft,
    NotADag,
    RevertRejected,
)
from .graph import TIER_BASE_DISTANCE, KnowledgeDAG, compute_distance
from .lifecycle import AuditLog, GarbageCollector, Reverter
from .retrieval import BoundedRetriever, Recency, TemporalFact, classify_facts, filter_recency, most_common
from .store import HeadIndex, InMemoryRecordStore, RecordStore, SqliteRecordStore
from .types import (
    Actor,
    AuditRow,
    Cardinality,
    Delta,
    Disposition,
    EdgeType,
    EntityInfo,
    GateResult,
    HeadEntry,
    NodeState,
    Precedence,
    QueueItem,
    QueueItemType,
    Record,
    Routing,
    Tier,
    TripletDraft,
    ValidationOutcome,
    ValidationResult,
    utcnow,
)
from .validate import PlanFence, ReplanOrchestrator
from .write import Cascade, ReconciliationGate, auto_allowed, precedence

__all__ = [
    # types
    "Actor", "AuditRow", "Cardinality", "Delta", "Disposition", "EdgeType",
    "EntityInfo", "GateResult", "HeadEntry", "NodeState", "Precedence",
    "QueueItem", "QueueItemType", "Record", "Routing", "Tier", "TripletDraft",
    "ValidationOutcome", "ValidationResult", "utcnow",
    # errors
    "DagKBError", "HeadConflict", "ImmutableViolation", "LineageIncomplete",
    "MalformedDraft", "NotADag", "RevertRejected",
    # store / graph
    "RecordStore", "InMemoryRecordStore", "SqliteRecordStore", "HeadIndex",
    "KnowledgeDAG", "compute_distance", "TIER_BASE_DISTANCE",
    # write path
    "precedence", "auto_allowed", "ReconciliationGate", "Cascade",
    # use path
    "PlanFence", "ReplanOrchestrator",
    # retrieval / lifecycle
    "BoundedRetriever", "most_common", "Recency", "TemporalFact", "classify_facts",
    "filter_recency", "AuditLog", "Reverter", "GarbageCollector",
]

# WorkQueue lives in write/ but is a general utility -- re-export explicitly.
from .write.queue import WorkQueue  # noqa: E402

__all__.append("WorkQueue")
