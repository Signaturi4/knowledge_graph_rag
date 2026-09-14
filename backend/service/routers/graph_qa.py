"""KG-only question answering. No vector search, no Milvus, no reranker.

Entities are pulled from the query via the LLM provider (strict determinism
caveat + payload validation), matched to entity nodes in the real graph
(``dag_kb.graph.KnowledgeDAG`` -- entities as nodes, relations as edges, the
NVIDIA-repo base pattern), expanded ``depth`` hops over ACTIVE
subject->predicate->object edges, and the resulting facts are the only context
handed to the answerer. Retrieval is PLANFENCE-gated when a lineage root is
supplied.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter
from pydantic import BaseModel

from dag_kb import NodeState, ValidationOutcome
from llm import LLMUnavailable
from llm.prompting import Agent, strict
from observability import truncate
from pipeline.validation import validate_entities_payload

from ..deps import get_layer

router = APIRouter()
log = logging.getLogger("dagkb.qa")

import json

_ENTITY_SYS = strict(
    Agent.ENTITY,
    'Output shape: {"entities":["..."]} -- the named entities in the USER QUERY. '
    "Each entity MUST be a verbatim substring of the query.",
)
_ANSWER_SYS = strict(
    Agent.ANSWERER,
    "Answer the QUERY using ONLY the FACTS block. If the facts do not support an "
    'answer, output {"answer":"insufficient grounded facts"}. Otherwise output '
    '{"answer":"<concise answer>"}. When dates or source document citations are '
    "present in the facts for the relevant events or medications/supplements, "
    "explicitly include those dates and document IDs in your answer.",
)


class QAIn(BaseModel):
    query: str
    depth: int = 2
    lineage_root: str | None = None
    declared_deps: list[str] = []
    include_history: bool = False


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def _format_fact(dag, store, rid: str, is_archived: bool = False) -> str | None:
    nd = dag._g.nodes.get(rid, {})
    if not nd:
        return None

    # Derive true subject from incoming SUBJECT edge
    subj_slug = ""
    for u, _, d in dag._g.in_edges(rid, data=True):
        if d.get("kind") in ("SUBJECT", "subject"):
            subj_slug = u
            break
    subj_label = dag.entity_info(subj_slug).label if dag.is_entity(subj_slug) else subj_slug

    # Derive object and predicate
    rec = store.get(rid) if store else None
    pred = nd.get("predicate") or (rec.predicate if rec else "rel")
    obj = (rec.obj if rec else None)
    if not obj:
        for _, v, d in dag._g.out_edges(rid, data=True):
            if d.get("kind") in ("OBJECT", "object"):
                obj = dag.entity_info(v).label if dag.is_entity(v) else v
                break
    if not obj:
        obj = nd.get("semantic_key", "").split("::")[-1]

    # Format dates
    raw_dates = nd.get("doc_dates", "[]")
    try:
        dates = json.loads(raw_dates) if isinstance(raw_dates, str) else list(raw_dates)
    except Exception:
        dates = [nd.get("doc_date")] if nd.get("doc_date") else []

    clean_dates: list[str] = sorted(list({d.split("T")[0] for d in dates if d}))

    # Source documents
    raw_sources = nd.get("source_ids", "[]")
    try:
        sources = json.loads(raw_sources) if isinstance(raw_sources, str) else list(raw_sources)
    except Exception:
        sources = [nd.get("source_id")] if nd.get("source_id") else []
    clean_sources = [s for s in sources if s]

    mentions = int(nd.get("mention_count", 1))

    meta_parts: list[str] = []
    if clean_sources:
        if len(clean_sources) == 1:
            meta_parts.append(f"Doc: {clean_sources[0]}")
        else:
            meta_parts.append(f"Docs: {', '.join(clean_sources)}")
    if clean_dates:
        if len(clean_dates) == 1:
            meta_parts.append(f"Date: {clean_dates[0]}")
        else:
            meta_parts.append(f"Dates: {', '.join(clean_dates)}")
    if mentions > 1:
        meta_parts.append(f"{mentions} mentions")

    meta_str = f" [{ ' | '.join(meta_parts) }]" if meta_parts else ""
    prefix = "[ARCHIVED / SUPERSEDED] " if is_archived else ""
    return f"{prefix}{subj_label} — {pred} — {obj}{meta_str}"


def _facts_for_entities(layer, entities: list[str], depth: int = 2,
                        include_history: bool = False, query: str = "") -> list[str]:
    dag = layer.dag
    store = layer.store

    # Auto-detect historical query intent (e.g. prior diagnoses, revoked claims)
    if not include_history and query:
        hist_kw = r"\b(history|prior|previous|previously|earlier|before|past|initial|superseded|revoked|resolved)\b"
        if re.search(hist_kw, query, re.IGNORECASE):
            include_history = True

    start: set[str] = set()
    for e in entities:
        sl = _slug(e)
        if dag.is_entity(sl):
            start.add(sl)
            continue
        # Check patient aliases: e.g. "patient_7714" -> "pt_7714" or "7714" -> "pt_7714"
        pt_norm = re.sub(r"^patient_", "pt_", sl)
        if dag.is_entity(pt_norm):
            start.add(pt_norm)
            continue
        if re.match(r"^\d+$", sl) and dag.is_entity(f"pt_{sl}"):
            start.add(f"pt_{sl}")
            continue
        # Case-insensitive entity label search
        for n, d in dag._g.nodes(data=True):
            if d.get("kind") == "entity" and d.get("label", "").lower() == e.strip().lower():
                start.add(n)
                break

    if not start:
        return []

    facts_dict: dict[str, str] = {}
    seen_entities = set(start)
    frontier = set(start)

    for hop in range(max(1, depth)):
        next_frontier: set[str] = set()
        for ent_id in frontier:
            # 1. Outgoing relations where ent_id is subject
            for rid in dag.relations_of(ent_id, as_subject=True, as_object=False):
                nd = dag._g.nodes.get(rid, {})
                state = nd.get("state")
                is_archived = (state != NodeState.ACTIVE)
                if is_archived and not include_history:
                    continue

                formatted = _format_fact(dag, store, rid, is_archived=is_archived)
                if formatted:
                    facts_dict[rid] = formatted

                # Hop expansion: follow only outgoing object entities
                for _, obj_id, d in dag._g.out_edges(rid, data=True):
                    if d.get("kind") in ("OBJECT", "object") and obj_id not in seen_entities:
                        # Avoid hopping into other patient root entities or demographic values
                        obj_type = dag._g.nodes.get(obj_id, {}).get("entity_type", "")
                        if obj_type != "PATIENT" and obj_id not in ("female", "male"):
                            next_frontier.add(obj_id)

            # 2. Inbound relations where ent_id is object (seed entities only)
            if hop == 0:
                for rid in dag.relations_of(ent_id, as_subject=False, as_object=True):
                    nd = dag._g.nodes.get(rid, {})
                    state = nd.get("state")
                    is_archived = (state != NodeState.ACTIVE)
                    if is_archived and not include_history:
                        continue
                    # Ensure inbound relation does not originate from an unrelated patient
                    for subj_id, _, d in dag._g.in_edges(rid, data=True):
                        if d.get("kind") in ("SUBJECT", "subject") and subj_id != ent_id:
                            s_type = dag._g.nodes.get(subj_id, {}).get("entity_type", "")
                            if s_type == "PATIENT":
                                continue
                            formatted = _format_fact(dag, store, rid, is_archived=is_archived)
                            if formatted:
                                facts_dict[rid] = formatted

        seen_entities |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    # Separate active vs archived for clean LLM prompt hierarchy
    active_facts = [f for f in facts_dict.values() if not f.startswith("[ARCHIVED")]
    archived_facts = [f for f in facts_dict.values() if f.startswith("[ARCHIVED")]
    active_facts.sort()
    archived_facts.sort()
    return active_facts + archived_facts


@router.get("/models")
def models():
    return {"models": [getattr(get_layer().provider, "name", "llm")]}


@router.post("")
def qa(body: QAIn):
    layer = get_layer()
    provider = layer.provider
    log.info("QA received | query=%s depth=%d lineage_root=%s history=%s",
             truncate(body.query, 200), body.depth, body.lineage_root, body.include_history)

    gate_status = None
    if body.lineage_root and body.declared_deps:
        res = layer.planfence.validate(body.lineage_root, body.declared_deps, replanned=False)
        gate_status = res.outcome.value
        log.info("QA planfence gate | root=%s deps=%s -> %s (%s)",
                body.lineage_root, body.declared_deps, gate_status, res.reason)
        if res.outcome is not ValidationOutcome.AUTHORIZED:
            return {"answer": f"context withheld: lineage {res.outcome.value} ({res.reason})",
                    "context": [], "entities": [], "gate": gate_status}

    try:
        payload = provider.complete_json(f"USER QUERY:\n{body.query}", system=_ENTITY_SYS,
                                         timeout_s=45.0)
        entities = validate_entities_payload(payload, body.query)
    except LLMUnavailable as exc:
        log.warning("QA entity extraction unavailable | query=%s reason=%s",
                   truncate(body.query, 120), exc)
        entities = []
    log.info("QA entities resolved | query=%s -> %s", truncate(body.query, 120), entities)

    facts = _facts_for_entities(layer, entities, depth=body.depth,
                                include_history=body.include_history, query=body.query)
    log.info("QA retrieval | entities=%s -> %d facts: %s", entities, len(facts),
             truncate(" | ".join(facts), 400))
    if not facts:
        log.info("QA no grounded facts | query=%s", truncate(body.query, 120))
        return {"answer": "No grounded knowledge-graph facts match this query.",
                "context": [], "entities": entities, "gate": gate_status}

    block = "\n".join(f"- {f}" for f in facts)
    try:
        payload = provider.complete_json(f"FACTS:\n{block}\n\nQUERY: {body.query}",
                                         system=_ANSWER_SYS, timeout_s=60.0)
        answer = payload.get("answer", "") if isinstance(payload, dict) else str(payload)
    except LLMUnavailable as exc:
        log.warning("QA answerer unavailable | query=%s reason=%s", truncate(body.query, 120), exc)
        answer = f"(LLM unavailable: {exc}) Facts:\n{block}"

    log.info("QA answered | query=%s -> answer=%s", truncate(body.query, 120), truncate(answer, 300))
    return {"answer": answer, "context": facts, "entities": entities, "gate": gate_status}
