"""Time/source-aware retrieval -- "no stale memory," the same design goal as
PLANFENCE's "no stale plans," applied to retrieval instead of action
authorization.

PLANFENCE eliminates a plan silently acting on a fact that's since changed
(technical-plan-dag-knowledgebase.md secs 1.1/1.3). The retrieval-side
version of that risk isn't a plan citing a stale record ID -- it's a
clinician (or an LLM answering on their behalf) reading "Patient X: ARDS"
with no way to tell whether that was recorded an hour ago or eight months
ago, and treating it as equally current either way. This module closes that
gap deterministically: no LLM decides what's current vs. stale, a pure
function of node state + ``doc_date`` does.

Two axes, and this module keeps them explicitly separate rather than
collapsing them into one ACTIVE/ARCHIVED bit:

1. **Superseded or not** (already tracked by ``NodeState``): only meaningful
   for ``Cardinality.ONE`` facts (a value that gets replaced -- e.g. "current
   diagnosis"). Superseded = ``ARCHIVED`` = unambiguously PAST.
2. **Stale or not**: a ``Cardinality.MANY`` fact (e.g. "medications
   received") never supersedes -- three different medications over a year
   all sit ACTIVE, coexisting. Without this axis, a medication started
   yesterday and one discontinued a year ago look identical. This module
   reads each fact's ``doc_date`` and flags ACTIVE-but-old facts as
   ``CURRENT_STALE`` with an explicit disclosure string, instead of silently
   presenting both with equal confidence -- directly the "as of my last
   record of 3 months ago ... this might be outdated" behaviour requested.

No LLM anywhere in this module. It returns a fully-formed, machine-checkable
classification; an LLM (if used at all) only phrases what this already
decided -- it does not get to decide currency itself.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..types import NodeState

if TYPE_CHECKING:
    from .. import KnowledgeDAG

DEFAULT_FRESHNESS_DAYS = 90


class Recency(enum.Enum):
    CURRENT = "CURRENT"                # ACTIVE, within the freshness window
    CURRENT_STALE = "CURRENT_STALE"    # ACTIVE, but older than the freshness window -- disclose
    PAST = "PAST"                      # ARCHIVED / FLAGGED / DELETED -- superseded, historical only


@dataclass(frozen=True)
class TemporalFact:
    record_id: str
    subject_id: str
    subject_label: str
    predicate: str
    object_label: str
    state: NodeState
    doc_date: datetime | None          # None if the record carries no parseable date
    days_old: int | None               # None if doc_date is None
    source_ids: tuple[str, ...]
    recency: Recency
    staleness_note: str | None         # set iff recency is CURRENT_STALE

    def as_text(self) -> str:
        """Human/LLM-facing rendering -- the disclosure is baked into the
        text itself so a downstream LLM cannot drop it by only reading a
        subset of fields."""
        base = f"{self.subject_label} — {self.predicate} — {self.object_label}"
        if self.source_ids:
            base += f" [source: {', '.join(self.source_ids)}]"
        if self.recency is Recency.PAST:
            return f"[PAST / SUPERSEDED] {base}"
        if self.recency is Recency.CURRENT_STALE:
            return f"[CURRENT, but {self.staleness_note}] {base}"
        return base


def _parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _source_ids(node_data: dict) -> tuple[str, ...]:
    raw = node_data.get("source_ids", "[]")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else list(raw)
    except (ValueError, TypeError):
        parsed = [raw] if raw else []
    return tuple(s for s in parsed if s)


def _classify_one(
    dag: "KnowledgeDAG", record_id: str, subject_id: str, subject_label: str,
    *, now: datetime, freshness_days: int,
) -> TemporalFact:
    nd = dag.nx.nodes[record_id]
    state: NodeState = nd.get("state", NodeState.ACTIVE)
    predicate = nd.get("predicate", "")
    doc_date = _parse_date(nd.get("doc_date"))
    days_old = (now - doc_date).days if doc_date is not None else None

    object_label = ""
    for _, obj_id, d in dag.nx.out_edges(record_id, data=True):
        if d.get("kind") == "OBJECT":
            object_label = dag.entity_info(obj_id).label if dag.is_entity(obj_id) else obj_id
            break
    if not object_label:
        key = nd.get("semantic_key", "")
        object_label = key.split("::")[-1] if "::" in key else key

    if state is not NodeState.ACTIVE:
        recency, note = Recency.PAST, None
    elif days_old is None:
        recency, note = Recency.CURRENT, None
    elif days_old <= freshness_days:
        recency, note = Recency.CURRENT, None
    else:
        recency = Recency.CURRENT_STALE
        date_str = doc_date.date().isoformat()
        note = (f"as of {date_str} ({days_old} days ago) — no more recent record exists; "
               f"this may be outdated")

    return TemporalFact(
        record_id=record_id, subject_id=subject_id, subject_label=subject_label,
        predicate=predicate, object_label=object_label, state=state,
        doc_date=doc_date, days_old=days_old, source_ids=_source_ids(nd),
        recency=recency, staleness_note=note,
    )


def classify_facts(
    dag: "KnowledgeDAG",
    entity_id: str,
    *,
    now: datetime | None = None,
    freshness_days: int = DEFAULT_FRESHNESS_DAYS,
) -> list[TemporalFact]:
    """Every relation where ``entity_id`` is the subject, classified by
    recency. Includes PAST (superseded) facts -- callers filter by
    ``.recency`` for a specific question shape (see ``filter_recency``)."""
    if now is None:
        now = datetime.now(timezone.utc)
    if not dag.is_entity(entity_id):
        return []
    subject_label = dag.entity_info(entity_id).label
    facts = [
        _classify_one(dag, rid, entity_id, subject_label, now=now, freshness_days=freshness_days)
        for rid in dag.relations_of(entity_id, as_subject=True, as_object=False)
    ]
    # most clinically relevant first: CURRENT, then CURRENT_STALE (still
    # active but flagged), then PAST (superseded) -- newest first within
    # each group, not the enum's incidental alphabetical order.
    rank = {Recency.CURRENT: 0, Recency.CURRENT_STALE: 1, Recency.PAST: 2}
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)  # avoids datetime.min OverflowError on .timestamp()
    facts.sort(key=lambda f: (rank[f.recency], -(f.doc_date or epoch).timestamp()))
    return facts


def filter_recency(facts: list[TemporalFact], *recencies: Recency) -> list[TemporalFact]:
    """Convenience: ``filter_recency(facts, Recency.CURRENT, Recency.CURRENT_STALE)``
    for "what is X now" (includes stale-but-still-current, disclosed);
    ``filter_recency(facts, Recency.PAST)`` for "what did X used to have."""
    wanted = set(recencies)
    return [f for f in facts if f.recency in wanted]
