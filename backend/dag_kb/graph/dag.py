"""Phase 2 -- the DAG over ``networkx.DiGraph``.

Two kinds of node now live in the same graph (the NVIDIA-repo base pattern --
entities as nodes, relations as edges -- with the PLANFENCE machinery applied
per-edge instead of per-attribute-slot; see ``dag_kb.types.Record`` docstring):

  - **entity** nodes (``kind="entity"``): a stable identity anchor -- a real
    graph node with no versioning of its own, upserted idempotently by
    :meth:`ensure_entity`.
  - **record** nodes (``kind="record"``): one immutable version of a reified
    triple, exactly as before -- state/tier/lineage/audit all unchanged.

A record's ``SUBJECT``/``OBJECT`` edges connect it to its subject/object entity
nodes (``subject_entity --SUBJECT--> record --OBJECT--> object_entity``),
giving the graph real subject->predicate->object connectivity on top of the
existing:

  - immutable ``DERIVED_FROM`` spine (child -> exact parent record);
  - full transitive-closure ``frontier`` resolution (technical-plan Table 11:
    a depth-1 check is unsafe; a missing intermediate must block);
  - depth-1 ``direct_dependents`` for the bounded cascade;
  - an ACTIVE-only view so retrieval never serves TBD / FLAGGED / ARCHIVED nodes;
  - GraphML persistence carrying node ``state`` + ``tier``.

``KnowledgeDAG`` *wraps* ``nx.DiGraph`` (composition, not subclass); every mutator
ends by asserting the lineage spine is still acyclic (SUBJECT/OBJECT/SUPERSEDES/
REVERTED_FROM edges are excluded from that check -- see :meth:`_assert_dag`).
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Iterable

import networkx as nx

from ..errors import LineageIncomplete, NotADag
from ..types import Actor, AuditRow, EdgeType, EntityInfo, NodeState, Tier, utcnow

if TYPE_CHECKING:
    from ..lifecycle.audit import AuditLog
    from ..store import RecordStore


# Allowed state transitions (technical-plan section 6d). None == node creation.
_TRANSITIONS: dict[NodeState | None, set[NodeState]] = {
    None: {NodeState.ACTIVE, NodeState.ARCHIVED},
    NodeState.ACTIVE: {NodeState.ARCHIVED, NodeState.FLAGGED, NodeState.TBD, NodeState.DELETED},
    NodeState.TBD: {NodeState.ACTIVE, NodeState.FLAGGED, NodeState.ARCHIVED, NodeState.DELETED},
    NodeState.FLAGGED: {NodeState.ACTIVE, NodeState.ARCHIVED, NodeState.DELETED},
    NodeState.ARCHIVED: {NodeState.ACTIVE, NodeState.DELETED},
    NodeState.DELETED: {NodeState.ACTIVE},
}


class KnowledgeDAG:
    def __init__(self, store: "RecordStore") -> None:
        self._store = store
        self._g = nx.DiGraph()

    # ------------------------------------------------------------------ #
    # structure
    # ------------------------------------------------------------------ #
    @property
    def nx(self) -> nx.DiGraph:
        return self._g

    def has_node(self, record_id: str) -> bool:
        return self._g.has_node(record_id)

    def is_entity(self, node_id: str) -> bool:
        return self._g.has_node(node_id) and self._g.nodes[node_id].get("kind") == "entity"

    # -- entities (NVIDIA base pattern: entities.csv -- id, name, type) --- #
    def ensure_entity(self, entity_id: str, *, label: str, entity_type: str = "") -> None:
        """Idempotent upsert. Entities are identity anchors, not versioned
        records -- correcting one's type later is just another relation."""
        if self._g.has_node(entity_id):
            if self._g.nodes[entity_id].get("kind") != "entity":
                raise NotADag(f"{entity_id} already exists as a non-entity node")
            if entity_type and not self._g.nodes[entity_id].get("entity_type"):
                self._g.nodes[entity_id]["entity_type"] = entity_type
            return
        self._g.add_node(entity_id, kind="entity", label=label, entity_type=entity_type)

    def entity_info(self, entity_id: str) -> EntityInfo:
        d = self._g.nodes[entity_id]
        return EntityInfo(entity_id=entity_id, label=d.get("label", entity_id),
                          entity_type=d.get("entity_type", ""))

    # -- records (reified, versioned triples) ----------------------------- #
    def add_record(
        self,
        record,
        *,
        state: NodeState = NodeState.ACTIVE,
        audit: "AuditLog | None" = None,
        actor: Actor = Actor.SYSTEM_1,
        reason: str = "ingest",
        distance: float = 1.0,
    ) -> None:
        """Add a node and its immutable DERIVED_FROM edges to declared parents,
        plus SUBJECT/OBJECT edges to the entity graph when the record carries
        ``subject_ref``/``object_ref`` (see ``dag_kb.types.Record``)."""
        rid = record.record_id
        if self._g.has_node(rid):
            return
        doc_date = (
            record.valid_from.isoformat()
            if hasattr(record.valid_from, "isoformat")
            else str(record.valid_from)
        )
        source_id = record.source_id or ""
        self._g.add_node(
            rid,
            kind="record",
            state=state,
            tier=record.tier,
            semantic_key=record.semantic_key,
            predicate=record.predicate,
            record_id=rid,
            validated_at=None,
            source_id=source_id,
            source_ids=json.dumps([source_id] if source_id else []),
            doc_date=doc_date,
            doc_dates=json.dumps([doc_date] if doc_date else []),
            mention_count=1,
        )
        for pid in record.parent_ids:
            if not self._g.has_node(pid):
                if not self._store.has(pid):
                    self._g.remove_node(rid)
                    raise LineageIncomplete(f"parent {pid} of {rid} is not in the store")
                parent = self._store.get(pid)
                p_date = (
                    parent.valid_from.isoformat()
                    if hasattr(parent.valid_from, "isoformat")
                    else str(parent.valid_from)
                )
                self._g.add_node(
                    pid, kind="record", state=NodeState.ARCHIVED, tier=parent.tier,
                    semantic_key=parent.semantic_key, predicate=parent.predicate,
                    record_id=pid, validated_at=None,
                    source_id=parent.source_id or "",
                    source_ids=json.dumps([parent.source_id] if parent.source_id else []),
                    doc_date=p_date,
                    doc_dates=json.dumps([p_date] if p_date else []),
                    mention_count=1,
                )
            self._g.add_edge(rid, pid, kind=EdgeType.DERIVED_FROM.value, distance=distance)

        subject_ref = getattr(record, "subject_ref", "") or ""
        object_ref = getattr(record, "object_ref", None)
        if subject_ref:
            self.ensure_entity(subject_ref, label=str(record.provenance.get("subject", subject_ref)))
            self._g.add_edge(
                subject_ref, rid, kind=EdgeType.SUBJECT.value, distance=distance,
                source_id=source_id, doc_date=doc_date, mention_count=1,
            )
        if object_ref:
            self.ensure_entity(object_ref, label=str(record.provenance.get("object_label", object_ref)),
                               entity_type=str(record.provenance.get("object_type", "")))
            self._g.add_edge(
                rid, object_ref, kind=EdgeType.OBJECT.value, distance=distance,
                source_id=source_id, doc_date=doc_date, mention_count=1,
            )

        self._assert_dag()
        if audit is not None:
            audit.record(AuditRow(rid, None, state, actor, record.txn_id, reason))

    def record_mention(self, record_id: str, *, source_id: str, doc_date: str) -> None:
        """Record an identical re-mention of a fact: increment frequency and append source document & date."""
        if not self._g.has_node(record_id):
            return
        nd = self._g.nodes[record_id]
        nd["mention_count"] = int(nd.get("mention_count", 1)) + 1

        raw_s = nd.get("source_ids", "[]")
        try:
            s_list = json.loads(raw_s) if isinstance(raw_s, str) else list(raw_s)
        except Exception:
            s_list = [raw_s] if raw_s else []
        if source_id and source_id not in s_list:
            s_list.append(source_id)
        nd["source_ids"] = json.dumps(s_list)
        nd["source_id"] = source_id

        raw_d = nd.get("doc_dates", "[]")
        try:
            d_list = json.loads(raw_d) if isinstance(raw_d, str) else list(raw_d)
        except Exception:
            d_list = [raw_d] if raw_d else []
        if doc_date and doc_date not in d_list:
            d_list.append(doc_date)
        nd["doc_dates"] = json.dumps(d_list)
        nd["doc_date"] = doc_date

        for u, v, ed in self._g.in_edges(record_id, data=True):
            ed["mention_count"] = nd["mention_count"]
            ed["source_ids"] = nd["source_ids"]
            ed["doc_dates"] = nd["doc_dates"]
        for u, v, ed in self._g.out_edges(record_id, data=True):
            if ed.get("kind") == EdgeType.OBJECT.value:
                ed["mention_count"] = nd["mention_count"]
                ed["source_ids"] = nd["source_ids"]
                ed["doc_dates"] = nd["doc_dates"]

    def add_edge_typed(self, src: str, dst: str, kind: EdgeType, *, distance: float = 1.0) -> None:
        self._g.add_edge(src, dst, kind=kind.value, distance=distance)
        self._assert_dag()

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self, record_id: str) -> NodeState:
        return self._g.nodes[record_id]["state"]

    def set_state(
        self,
        record_id: str,
        new_state: NodeState,
        *,
        actor: Actor,
        reason: str,
        audit: "AuditLog | None" = None,
        txn_id: str = "-",
    ) -> None:
        cur = self._g.nodes[record_id]["state"]
        if cur is new_state:
            return
        if new_state not in _TRANSITIONS.get(cur, set()):
            raise NotADag(f"illegal transition {cur} -> {new_state} for {record_id}")
        self._g.nodes[record_id]["state"] = new_state
        if audit is not None:
            audit.record(AuditRow(record_id, cur, new_state, actor, txn_id, reason))

    def stamp_validated(self, record_id: str, logical_ts: int) -> None:
        self._g.nodes[record_id]["validated_at"] = logical_ts

    # ------------------------------------------------------------------ #
    # lineage
    # ------------------------------------------------------------------ #
    def _derived_from_successors(self, node: str) -> Iterable[str]:
        for _, dst, data in self._g.out_edges(node, data=True):
            if data.get("kind") == EdgeType.DERIVED_FROM.value:
                yield dst

    def frontier(self, root_id: str, dep_keys: Iterable[str]) -> dict[str, str]:
        """F_a(x): for each key in ``dep_keys`` the record_id on the lineage of
        ``root_id`` whose semantic_key == key. Full transitive closure.

        Raises :class:`LineageIncomplete` if ``root_id`` is unknown, a parent is
        missing from the store, or a key is never reached.
        """
        want = set(dep_keys)
        if not self._g.has_node(root_id):
            raise LineageIncomplete(f"unknown lineage root {root_id}")
        found: dict[str, str] = {}
        seen: set[str] = set()
        stack = [root_id]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            k = self._g.nodes[n].get("semantic_key")
            if k in want and k not in found:
                found[k] = n
            for succ in self._derived_from_successors(n):
                if not self._g.has_node(succ):
                    raise LineageIncomplete(f"missing lineage node {succ}")
                stack.append(succ)
        missing = want - set(found)
        if missing:
            raise LineageIncomplete(f"dependency keys unreachable from {root_id}: {sorted(missing)}")
        return found

    def root_derives_from(self, root_id: str, frontier_ids: Iterable[str]) -> bool:
        """True iff every id in ``frontier_ids`` is reachable from ``root_id``
        along DERIVED_FROM edges (PLANFENCE Algorithm 1, line 7)."""
        reach: set[str] = set()
        stack = [root_id]
        while stack:
            n = stack.pop()
            if n in reach:
                continue
            reach.add(n)
            stack.extend(self._derived_from_successors(n))
        return all(fid in reach for fid in frontier_ids)

    def direct_dependents(self, record_id: str) -> list[str]:
        """Depth-1 children over DERIVED_FROM (nodes that cite this record)."""
        out = []
        for src, _, data in self._g.in_edges(record_id, data=True):
            if data.get("kind") == EdgeType.DERIVED_FROM.value:
                out.append(src)
        return out

    # -- entity-graph traversal (SUBJECT/OBJECT edges) -------------------- #
    def relations_of(self, entity_id: str, *, as_subject: bool = True,
                     as_object: bool = True) -> list[str]:
        """record_ids of relations touching this entity, ACTIVE or not."""
        out: list[str] = []
        if as_subject:
            for _, dst, d in self._g.out_edges(entity_id, data=True):
                if d.get("kind") == EdgeType.SUBJECT.value:
                    out.append(dst)
        if as_object:
            for src, _, d in self._g.in_edges(entity_id, data=True):
                if d.get("kind") == EdgeType.OBJECT.value:
                    out.append(src)
        return out

    def neighbors_of(self, entity_id: str, *, active_only: bool = True) -> list[tuple[str, str, str]]:
        """(predicate, direction, other_entity_id) for every relation touching
        this entity -- the graph-QA retrieval primitive."""
        out: list[tuple[str, str, str]] = []
        for rid in self.relations_of(entity_id):
            if active_only and self._g.nodes[rid].get("state") is not NodeState.ACTIVE:
                continue
            pred = self._g.nodes[rid].get("predicate", "")
            for _, obj_id, d in self._g.out_edges(rid, data=True):
                if d.get("kind") == EdgeType.OBJECT.value and obj_id != entity_id:
                    out.append((pred, "->", obj_id))
            for subj_id, _, d in self._g.in_edges(rid, data=True):
                if d.get("kind") == EdgeType.SUBJECT.value and subj_id != entity_id:
                    out.append((pred, "<-", subj_id))
        return out

    def corroboration_count(self, entity_id: str) -> int:
        """How many *distinct* subject entities already have an ACTIVE
        relation pointing at ``entity_id`` (any predicate) -- the "how many
        independent things already reference this" signal used to tighten
        edge distance (see ``graph/distance.py``). 0 if the entity doesn't
        exist yet or nothing points at it."""
        if not self.is_entity(entity_id):
            return 0
        return len({other for _, direction, other in self.neighbors_of(entity_id)
                    if direction == "<-"})

    # ------------------------------------------------------------------ #
    # views / persistence
    # ------------------------------------------------------------------ #
    def active_view(self) -> nx.DiGraph:
        """Entity nodes (always -- they carry no state) plus record nodes whose
        state is ACTIVE. This is what retrieval/QA/visualization traverse so a
        stale, contradicted, or archived fact is never served, while the
        entity graph around it stays connected."""
        keep = [
            n for n, d in self._g.nodes(data=True)
            if d.get("kind") == "entity" or d.get("state") is NodeState.ACTIVE
        ]
        return self._g.subgraph(keep).copy()

    def nodes(self, state: NodeState | None = None, *, kind: str | None = None) -> list[str]:
        items = self._g.nodes(data=True)
        if kind is not None:
            items = [(n, d) for n, d in items if d.get("kind") == kind]
        if state is None:
            return [n for n, _ in items] if kind is not None else list(self._g.nodes)
        return [n for n, d in items if d.get("state") is state]

    def remove_node(self, record_id: str) -> None:
        self._g.remove_node(record_id)

    def write_graphml(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        h = nx.DiGraph()
        for n, d in self._g.nodes(data=True):
            if d.get("kind") == "entity":
                h.add_node(n, kind="entity", label=d.get("label", ""),
                          entity_type=d.get("entity_type", ""))
            else:
                h.add_node(
                    n, kind="record",
                    state=d["state"].value,
                    tier=int(d["tier"]) if isinstance(d["tier"], Tier) else int(d["tier"]),
                    semantic_key=d.get("semantic_key", ""),
                    predicate=d.get("predicate", ""),
                    validated_at="" if d.get("validated_at") is None else str(d["validated_at"]),
                    source_id=str(d.get("source_id", "")),
                    source_ids=str(d.get("source_ids", "[]")),
                    doc_date=str(d.get("doc_date", "")),
                    doc_dates=str(d.get("doc_dates", "[]")),
                    mention_count=int(d.get("mention_count", 1)),
                )
        for u, v, d in self._g.edges(data=True):
            h.add_edge(
                u, v, kind=d.get("kind", ""),
                distance=float(d.get("distance", 1.0)),
                source_id=str(d.get("source_id", "")),
                doc_date=str(d.get("doc_date", "")),
                mention_count=int(d.get("mention_count", 1)),
            )
        nx.write_graphml(h, path)

    @classmethod
    def read_graphml(cls, path: str, store: "RecordStore") -> "KnowledgeDAG":
        raw = nx.read_graphml(path)
        dag = cls(store)
        for n, d in raw.nodes(data=True):
            if d.get("kind") == "entity":
                dag._g.add_node(n, kind="entity", label=d.get("label", ""),
                                entity_type=d.get("entity_type", ""))
            else:
                dag._g.add_node(
                    n, kind="record",
                    state=NodeState(d["state"]),
                    tier=Tier(int(d["tier"])),
                    semantic_key=d.get("semantic_key", ""),
                    predicate=d.get("predicate", ""),
                    record_id=n,
                    validated_at=(int(d["validated_at"]) if d.get("validated_at") not in ("", None) else None),
                    source_id=str(d.get("source_id", "")),
                    source_ids=str(d.get("source_ids", "[]")),
                    doc_date=str(d.get("doc_date", "")),
                    doc_dates=str(d.get("doc_dates", "[]")),
                    mention_count=int(d.get("mention_count", 1)),
                )
        for u, v, d in raw.edges(data=True):
            dag._g.add_edge(
                u, v, kind=d.get("kind", ""),
                distance=float(d.get("distance", 1.0)),
                source_id=str(d.get("source_id", "")),
                doc_date=str(d.get("doc_date", "")),
                mention_count=int(d.get("mention_count", 1)),
            )
        dag._assert_dag()
        return dag

    # ------------------------------------------------------------------ #
    def _assert_dag(self) -> None:
        """The acyclicity constraint applies to the *lineage* spine only.

        ``SUPERSEDES`` / ``REVERTED_FROM`` are historical back-pointers, and
        ``SUBJECT``/``OBJECT`` entity-graph edges legitimately cycle (A causes
        B, B causes A is valid knowledge); only ``DERIVED_FROM`` must stay
        acyclic (technical-plan section 1.2).
        """
        spine = nx.DiGraph()
        spine.add_nodes_from(self._g.nodes)
        spine.add_edges_from(
            (u, v) for u, v, d in self._g.edges(data=True)
            if d.get("kind") == EdgeType.DERIVED_FROM.value
        )
        if not nx.is_directed_acyclic_graph(spine):
            raise NotADag("graph mutation introduced a cycle in the DERIVED_FROM spine")
