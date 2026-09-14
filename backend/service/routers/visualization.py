"""Render the live KnowledgeDAG as an interactive graph (pyvis / vis-network.js).

Endpoints:
- GET /visualization/graph.html: Interactive HTML for whole graph or specific patient subgraph.
- GET /visualization/graph.json: Node-link JSON for d3/cytoscape.
- GET /visualization/patients: List of detected patient entities in the graph.
- GET /visualization/patient/{patient_id}/summary: Summary of facts for the specified patient.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse, JSONResponse

from dag_kb import EdgeType, NodeState
from viz.pyvis_render import get_patient_entities, graph_json, render_html

from ..deps import get_layer

router = APIRouter()
log = logging.getLogger("dagkb.viz")


@router.get("/patients")
def list_patients():
    """Return all patient entity anchors detected in the DAG."""
    layer = get_layer()
    patients = get_patient_entities(layer.dag)
    return JSONResponse({"patients": patients})


@router.get("/graph.html", response_class=HTMLResponse)
def graph_html(
    patient_id: Optional[str] = Query(None, description="Patient ID to filter subgraph (e.g. PT-1049)"),
    include_history: bool = Query(True, description="Whether to include superseded/archived records"),
    physics: bool = Query(True, description="Enable ForceAtlas2 physics"),
):
    layer = get_layer()
    log.info("VIZ request | format=html patient=%s include_history=%s", patient_id, include_history)
    html_content = render_html(
        layer.dag,
        patient_id=patient_id,
        include_history=include_history,
        physics=physics,
    )
    return HTMLResponse(html_content)


@router.get("/patient/{patient_id}/summary")
def patient_summary(patient_id: str):
    """Structured clinical summary for a specific patient."""
    layer = get_layer()
    dag = layer.dag
    store = layer.store
    slug = patient_id.lower().replace("-", "_")

    active_facts = []
    archived_facts = []
    flagged_facts = []

    for nid, d in dag.nx.nodes(data=True):
        if d.get("kind") == "entity":
            continue
        key = d.get("semantic_key", "")
        if key.startswith(f"{slug}::"):
            st: NodeState = d.get("state", NodeState.ACTIVE)
            rec = store.get(nid) if store.has(nid) else None

            import json
            raw_s = d.get("source_ids", "[]")
            try:
                s_list = json.loads(raw_s) if isinstance(raw_s, str) else list(raw_s)
            except Exception:
                s_list = [raw_s] if raw_s else []
            if not s_list and rec and rec.source_id:
                s_list = [rec.source_id]

            raw_d = d.get("doc_dates", "[]")
            try:
                d_list = json.loads(raw_d) if isinstance(raw_d, str) else list(raw_d)
            except Exception:
                d_list = [raw_d] if raw_d else []
            if not d_list and rec and rec.valid_from:
                d_list = [rec.valid_from.isoformat()]

            item = {
                "record_id": nid,
                "semantic_key": key,
                "predicate": d.get("predicate", ""),
                "object": rec.obj if rec else "N/A",
                "tier": rec.tier.name if rec else "N/A",
                "valid_from": rec.valid_from.isoformat() if rec else None,
                "source_id": rec.source_id if rec else None,
                "mention_count": int(d.get("mention_count", 1)),
                "source_ids": s_list,
                "doc_dates": d_list,
            }
            if st is NodeState.ACTIVE:
                active_facts.append(item)
            elif st is NodeState.ARCHIVED:
                archived_facts.append(item)
            elif st is NodeState.FLAGGED:
                flagged_facts.append(item)

    # Detect supersession pairs
    supersessions = []
    for u, v, ed in dag.nx.edges(data=True):
        if ed.get("kind") == EdgeType.SUPERSEDES.value:
            ukey = dag.nx.nodes[u].get("semantic_key", "")
            if ukey.startswith(f"{slug}::"):
                rec_new = store.get(u) if store.has(u) else None
                rec_old = store.get(v) if store.has(v) else None
                supersessions.append({
                    "new_record": {"id": u, "predicate": rec_new.predicate if rec_new else "", "obj": rec_new.obj if rec_new else ""},
                    "superseded_record": {"id": v, "predicate": rec_old.predicate if rec_old else "", "obj": rec_old.obj if rec_old else ""},
                })

    return JSONResponse({
        "patient_id": patient_id,
        "active_count": len(active_facts),
        "archived_count": len(archived_facts),
        "flagged_count": len(flagged_facts),
        "active_facts": active_facts,
        "archived_facts": archived_facts,
        "flagged_facts": flagged_facts,
        "supersessions": supersessions,
    })


@router.get("/graph.json")
def graph_json_endpoint():
    layer = get_layer()
    data = graph_json(layer.dag)
    log.info("VIZ request | format=json nodes=%d edges=%d", len(data["nodes"]), len(data["edges"]))
    return JSONResponse(data)
