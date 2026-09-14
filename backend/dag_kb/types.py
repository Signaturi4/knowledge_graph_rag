"""Core value types for the DAG knowledgebase.

Grounding: *Fresh Memory, Stale Plans* (PLANFENCE), object model in section 3.1.
Alignment: docs/PRD-DAG.md and docs/technical-plan-dag-knowledgebase.md sections
6b-6e.

The atomic versioned unit is a **triplet slot** -- a semantic key
``(entity, attribute)`` whose object/value is the versioned content. One
immutable :class:`Record` per version; the mutable ``state`` lives on the graph
node, never on the record.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Mapping


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class Tier(enum.IntEnum):
    """Authority tier. Lower value == stronger authority (T0 is strongest)."""

    T0 = 0  # canonical / of-record: statute, signed contract, ordinance
    T1 = 1  # official internal: approved policy, SOP, released spec
    T2 = 2  # verified external: vendor docs, standards body
    T3 = 3  # derived / inferred: pipeline output from other nodes
    T4 = 4  # observational: agent conversation history, user chat & email


class NodeState(enum.Enum):
    """Mutable lifecycle state of a content node (technical plan section 6d)."""

    ACTIVE = "ACTIVE"        # current ground truth; exactly one per semantic key
    TBD = "TBD"              # parent changed; re-derivation queued / in progress
    FLAGGED = "FLAGGED"      # contradiction; locked by Conflict Resolution Engine
    ARCHIVED = "ARCHIVED"    # historical fact, retained for audit / revert
    DELETED = "DELETED"      # soft-deleted; metadata retained, content GC-able

    @property
    def served(self) -> bool:
        """Whether a node in this state is eligible for active retrieval."""
        return self is NodeState.ACTIVE


class Delta(enum.Enum):
    """Semantic relationship of an incoming draft to the active node.

    Produced by the LLM delta layer (technical plan section 6b, stage 3);
    consumed -- never produced -- by the deterministic reconciliation gate.
    """

    IDENTICAL = "IDENTICAL"
    REFINEMENT = "REFINEMENT"
    SUPERSEDING_CHANGE = "SUPERSEDING_CHANGE"
    CONTRADICTION = "CONTRADICTION"


class Disposition(enum.Enum):
    """Outcome of the reconciliation gate (technical plan section 6b)."""

    NO_OP = "NO_OP"
    AUTO_COMMIT = "AUTO_COMMIT"
    QUEUE_SYSTEM_2 = "QUEUE_SYSTEM_2"
    FLAG_CONTRADICTION = "FLAG_CONTRADICTION"
    REJECT_FAIL_CLOSED = "REJECT_FAIL_CLOSED"


class Precedence(enum.Enum):
    """Result of the pure precedence function (draft vs active)."""

    SUPERSEDE = "SUPERSEDE"
    REJECT = "REJECT"
    COEXIST = "COEXIST"


class Actor(enum.Enum):
    """Who caused a state transition (audit trail)."""

    SYSTEM_1 = "SYSTEM_1"   # automatic
    SYSTEM_2 = "SYSTEM_2"   # human-in-the-loop
    CRE = "CRE"             # Conflict Resolution Engine
    REVERT = "REVERT"


class Routing(enum.Enum):
    SYSTEM_1 = "SYSTEM_1"
    SYSTEM_2 = "SYSTEM_2"


class QueueItemType(enum.Enum):
    SUPERSESSION_REVIEW = "SUPERSESSION_REVIEW"
    CONTRADICTION = "CONTRADICTION"
    REDERIVE = "REDERIVE"
    DEAD_LETTER = "DEAD_LETTER"


class EdgeType(str, enum.Enum):
    DERIVED_FROM = "DERIVED_FROM"    # child -> exact parent record (immutable spine)
    SUPERSEDES = "SUPERSEDES"        # new version -> prior version
    REVERTED_FROM = "REVERTED_FROM"  # reinstated version -> version it displaced
    SUBJECT = "SUBJECT"              # subject entity -> relation (reified triple) node
    OBJECT = "OBJECT"                # relation (reified triple) node -> object entity


class Cardinality(str, enum.Enum):
    """Does a new value for this (subject, predicate) SUPERSEDE the old one
    (a functional attribute -- age, current diagnosis, zoning limit), or
    COEXIST alongside it as an independent graph edge (a relation -- symptom,
    therapy received, participates_in)? Decided deterministically from the
    predicate string (technical-plan STATUS.md "multi-valued-fact modeling"),
    never by the LLM."""

    ONE = "one"    # semantic_key = subject::predicate            (current behavior)
    MANY = "many"  # semantic_key = subject::predicate::object    (independent edges)


RecordType = str  # "asserted" | "derived"


@dataclass(frozen=True, slots=True)
class EntityInfo:
    """A lightweight identity anchor -- NOT versioned the way Records are.

    Entities are the NVIDIA-repo base pattern (``entities.csv``: entity_id,
    entity_name, entity_type). They are upserted idempotently
    (:meth:`dag_kb.graph.KnowledgeDAG.ensure_entity`) the first time any
    relation references them; correcting an entity's type later is itself
    just another (low-cardinality) relation, not a special case.
    """

    entity_id: str     # canonical slug, e.g. "sector_4", "patient_42"
    label: str         # display name as first seen, e.g. "Sector 4"
    entity_type: str = ""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# The immutable record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Record:
    """One immutable version of a **reified triple**: ``subject_ref --predicate--> object``.

    This is the PLANFENCE object model (content-addressed, versioned, tiered,
    lineage-tracked -- see :mod:`dag_kb.ids`) applied to a knowledge-graph EDGE
    instead of a flat attribute slot (the NVIDIA-repo base pattern: entities as
    nodes, relations as edges -- see ``utils/lc_graph.py`` /
    ``from_pandas_edgelist(source=subject, target=object, edge_attr=relation)``).

    ``subject_ref`` is always the canonical id of the subject entity node.
    ``object_ref`` is the canonical id of the object entity node when the
    object is itself an entity worth a graph node (the common case for the
    NVIDIA-style categories -- PERSON, ORG, GPE, PRODUCT, CONCEPT, ...); it is
    ``None`` when the object is a bare literal value (a number, a short
    free-text value) with no reuse value as a node.

    ``cardinality`` decides how ``semantic_key`` is built (technical-plan
    STATUS.md "multi-valued-fact modeling"): :data:`Cardinality.ONE` keys on
    ``subject::predicate`` so a new value supersedes the old one (a functional
    attribute); :data:`Cardinality.MANY` keys on ``subject::predicate::object``
    so each distinct object is an independently-versioned edge that coexists
    with its siblings (a true graph relation).

    ``parent_ids`` are the *exact* record IDs this version was derived from --
    the lineage spine PLANFENCE walks. Empty for raw asserted facts.
    """

    record_id: str
    semantic_key: str            # cardinality-aware identity, see class docstring
    predicate: str
    obj: Any                     # the versioned value / object (display form)
    parent_ids: tuple[str, ...]
    owner: str
    owner_seq: int               # monotone per semantic_key
    tier: Tier
    auto_update: bool            # per-node override; can only *lower* autonomy
    record_type: RecordType
    valid_from: datetime
    source_id: str
    raw_citation: str
    txn_id: str
    subject_ref: str = ""                        # canonical subject entity id
    object_ref: str | None = None                # canonical object entity id, if entity-shaped
    cardinality: Cardinality = Cardinality.MANY
    provenance: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)

    @property
    def is_derived(self) -> bool:
        return self.record_type == "derived" or bool(self.parent_ids)


# --------------------------------------------------------------------------- #
# The mutable draft (pre-reconciliation)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TripletDraft:
    """Normalized output of extraction (technical plan section 6b, stage 1).

    Carries everything the reconciliation gate needs except ``owner_seq`` and
    ``record_id``, which are assigned at commit time.
    """

    semantic_key: str
    predicate: str
    obj: Any
    tier: Tier
    auto_update: bool
    source_id: str
    raw_citation: str
    valid_from: datetime = field(default_factory=utcnow)
    parent_ids: tuple[str, ...] = ()
    record_type: RecordType = "asserted"
    subject_ref: str = ""
    object_ref: str | None = None
    cardinality: Cardinality = Cardinality.MANY
    provenance: dict[str, Any] = field(default_factory=dict)

    def required_fields_present(self) -> bool:
        # subject_ref is not enforced here: this dataclass is also used directly
        # by unit tests exercising pure DAG/versioning mechanics with no entity
        # graph attached. The extraction pipeline (pipeline.stage1_extract) is
        # the layer that must never produce a draft with an empty subject_ref --
        # see pipeline.validation.validate_triplet_draft, which does enforce it.
        return bool(
            self.semantic_key
            and self.predicate
            and self.obj is not None
            and isinstance(self.tier, Tier)
            and self.source_id
        )

    def to_record(self, *, record_id: str, owner: str, owner_seq: int, txn_id: str) -> Record:
        return Record(
            record_id=record_id,
            semantic_key=self.semantic_key,
            predicate=self.predicate,
            obj=self.obj,
            parent_ids=tuple(self.parent_ids),
            owner=owner,
            owner_seq=owner_seq,
            tier=self.tier,
            auto_update=self.auto_update,
            record_type="derived" if self.parent_ids else self.record_type,
            valid_from=self.valid_from,
            source_id=self.source_id,
            raw_citation=self.raw_citation,
            txn_id=txn_id,
            subject_ref=self.subject_ref,
            object_ref=self.object_ref,
            cardinality=self.cardinality,
            provenance=dict(self.provenance),
        )


# --------------------------------------------------------------------------- #
# Head index entry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class HeadEntry:
    """The currently-authorized version pointer for a semantic key -- H(x)."""

    semantic_key: str
    record_id: str
    owner: str
    owner_seq: int              # monotone; strictly increases on every head move

    def bumped(self, *, record_id: str, owner: str) -> "HeadEntry":
        return replace(self, record_id=record_id, owner=owner, owner_seq=self.owner_seq + 1)


# --------------------------------------------------------------------------- #
# Work queue item
# --------------------------------------------------------------------------- #
def _new_id(prefix: str) -> str:
    import uuid

    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass(slots=True)
class QueueItem:
    kind: QueueItemType
    routed_to: Routing
    semantic_key: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: _new_id("q"))
    created_at: datetime = field(default_factory=utcnow)
    resolved: bool = False
    resolution: str = ""            # SYSTEM_2 verdict: approve | reject | edit | defer


# --------------------------------------------------------------------------- #
# Audit row
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class AuditRow:
    node_id: str
    from_state: NodeState | None
    to_state: NodeState | None
    actor: Actor
    txn_id: str
    reason: str
    ts: datetime = field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# Reconciliation gate result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class GateResult:
    disposition: Disposition
    semantic_key: str
    committed_record_id: str | None = None
    queue_item: QueueItem | None = None
    detail: str = ""


# --------------------------------------------------------------------------- #
# PLANFENCE (use-path) validation result
# --------------------------------------------------------------------------- #
class ValidationOutcome(enum.Enum):
    AUTHORIZED = "AUTHORIZED"
    REPLAN_REQUIRED = "REPLAN_REQUIRED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    outcome: ValidationOutcome
    frontier: Mapping[str, str]            # F_a(x)
    heads: Mapping[str, str]               # H(x)
    mismatched_keys: tuple[str, ...] = ()
    reason: str = ""
