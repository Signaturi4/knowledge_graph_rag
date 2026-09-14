"""Interactive HTML rendering of the KnowledgeDAG via pyvis (vis-network.js) --
the same JS library graphify's HTML export uses.

Two node kinds (see ``dag_kb.graph.KnowledgeDAG``):
  - **entity** nodes -- blue squares, labeled by name -- the NVIDIA-repo base
    graph (entities.csv).
  - **record** nodes -- small dots colored by lifecycle state, labeled by
    predicate -- the versioned, reified triple sitting on the
    subject-entity->record->object-entity path (SUBJECT/OBJECT edges).

Bookkeeping edges (DERIVED_FROM = lineage spine, SUPERSEDES/REVERTED_FROM =
version history) are drawn thin and dashed so the subject->predicate->object
relations read as the dominant shape of the graph.

Layout: force-directed physics (ForceAtlas2) already pulls entities sharing
several relations/paths closer together and pushes unconnected ones apart --
that clustering is emergent from graph topology (shared neighbors), not
something bolted on. What was missing is per-edge tightness: every edge used
one uniform spring length regardless of the fact's real weight. Each edge's
``distance`` (tier confidence x recency decay, see ``dag_kb/graph/distance.py``,
already on every edge and in ``dag.graphml``) is now mapped to vis-network's
per-edge ``length``, so a fresh, high-authority relation pulls its two
entities in tight and a stale/low-tier one relaxes them further apart -- the
same signal BoundedRetriever's Dijkstra already prioritizes, made visible.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dag_kb import EdgeType, NodeState

if TYPE_CHECKING:
    from dag_kb import KnowledgeDAG

log = logging.getLogger("dagkb.viz")

_STATE_COLOR = {
    NodeState.ACTIVE: "#2e7d32",     # green  -- ground truth
    NodeState.TBD: "#f9a825",        # amber  -- update in progress
    NodeState.FLAGGED: "#c62828",    # red    -- contradiction, locked
    NodeState.ARCHIVED: "#9e9e9e",   # gray   -- historical
    NodeState.DELETED: "#212121",    # black  -- removed
}

_ENTITY_COLOR = "#1565c0"   # blue -- identity anchors (NVIDIA-repo base graph)

_EDGE_STYLE = {
    EdgeType.SUBJECT.value: {"dashes": False, "color": "#455a64", "width": 2},
    EdgeType.OBJECT.value: {"dashes": False, "color": "#455a64", "width": 2},
    EdgeType.DERIVED_FROM.value: {"dashes": True, "color": "#90a4ae", "width": 1},
    EdgeType.SUPERSEDES.value: {"dashes": True, "color": "#bdbdbd", "width": 1},
    EdgeType.REVERTED_FROM.value: {"dashes": True, "color": "#8e24aa", "width": 1},
}

# distance -> vis-network spring length. TIER_BASE_DISTANCE runs ~1.0 (T0) to
# ~5.0 (T4); this scale keeps a fresh T0 edge snug (~40px) and a stale/T4 one
# comfortably loose (~200px) without either collapsing nodes on top of each
# other or flying off-screen once decay stacks up.
_DISTANCE_TO_LENGTH_SCALE = 40.0
_MAX_EDGE_LENGTH = 400.0


def _edge_length(distance: float) -> float:
    return min(max(distance, 0.1) * _DISTANCE_TO_LENGTH_SCALE, _MAX_EDGE_LENGTH)


def get_patient_entities(dag: "KnowledgeDAG") -> list[dict]:
    """Find all patient entity nodes in the graph."""
    patients = []
    for n, d in dag.nx.nodes(data=True):
        if d.get("kind") == "entity":
            lid = n.lower()
            label = str(d.get("label", n))
            if lid.startswith("pt_") or lid.startswith("pt-") or lid.startswith("patient_") or "pt-" in label.lower() or "patient" in label.lower():
                patients.append({"id": n, "label": label, "entity_type": d.get("entity_type", "PERSON")})
    return sorted(patients, key=lambda x: x["label"])


def get_patient_subgraph(dag: "KnowledgeDAG", patient_id: str, *, include_history: bool = True) -> nx.DiGraph:
    """Extract the induced subgraph centered on a specific patient."""
    base_g = dag.nx
    slug_id = patient_id.lower().replace("-", "_")

    # Find the target patient entity node
    patient_node = None
    for n in base_g.nodes():
        if n.lower() == slug_id or n.lower() == patient_id.lower() or n.lower() == f"patient_{slug_id}":
            patient_node = n
            break

    sub_nodes = set()
    if patient_node and base_g.has_node(patient_node):
        sub_nodes.add(patient_node)

    # Find all record nodes belonging to this patient
    for n, d in base_g.nodes(data=True):
        if d.get("kind") != "entity":
            key = d.get("semantic_key", "")
            if key.startswith(f"{slug_id}::") or (patient_node and key.startswith(f"{patient_node}::")):
                st = d.get("state")
                if not include_history and st != NodeState.ACTIVE:
                    continue
                sub_nodes.add(n)
                # If connected to object entity, include it
                for _, target, edata in base_g.out_edges(n, data=True):
                    if edata.get("kind") == EdgeType.OBJECT.value:
                        sub_nodes.add(target)
                # If connected to subject entity, include it
                for src, _, edata in base_g.in_edges(n, data=True):
                    if edata.get("kind") == EdgeType.SUBJECT.value:
                        sub_nodes.add(src)

    # If include_history, trace SUPERSEDES, DERIVED_FROM, and REVERTED_FROM edges
    if include_history:
        history_nodes = set()
        for rn in list(sub_nodes):
            if base_g.has_node(rn) and base_g.nodes[rn].get("kind") != "entity":
                for _, v, edata in base_g.out_edges(rn, data=True):
                    if edata.get("kind") in (EdgeType.SUPERSEDES.value, EdgeType.DERIVED_FROM.value, EdgeType.REVERTED_FROM.value):
                        history_nodes.add(v)
                        # and that node's object entity
                        for _, obj_target, oed in base_g.out_edges(v, data=True):
                            if oed.get("kind") == EdgeType.OBJECT.value:
                                history_nodes.add(obj_target)
        sub_nodes.update(history_nodes)

    if not sub_nodes and patient_node:
        sub_nodes.add(patient_node)

    return base_g.subgraph(sub_nodes).copy()


def render_html(
    dag: "KnowledgeDAG",
    *,
    patient_id: str | None = None,
    height: str = "750px",
    physics: bool = True,
    include_bookkeeping_edges: bool = False,
    include_history: bool = True,
) -> str:
    """Return a standalone HTML document visualizing the current DAG or patient subgraph."""
    from pyvis.network import Network

    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#111111",
        notebook=False,
        cdn_resources="in_line",
    )
    if physics:
        net.force_atlas_2based(gravity=-60, spring_length=120)

    if patient_id:
        g = get_patient_subgraph(dag, patient_id, include_history=include_history)
        include_bookkeeping_edges = include_history
    else:
        g = dag.nx

    for n, d in g.nodes(data=True):
        if d.get("kind") == "entity":
            is_patient = n.lower().startswith("pt_") or "pt-" in str(d.get("label", "")).lower()
            color = "#f57c00" if is_patient else _ENTITY_COLOR
            size = 24 if is_patient else 18
            net.add_node(
                n,
                label=d.get("label", n)[:32],
                title=f"entity_id={n}\ntype={d.get('entity_type', '')}",
                color=color,
                shape="box",
                size=size,
                font={"size": 14 if is_patient else 12, "color": "#ffffff" if is_patient else "#0d1b2a", "face": "system-ui"},
            )
        else:
            state: NodeState = d.get("state", NodeState.ACTIVE)
            tier = d.get("tier")
            key = d.get("semantic_key", "")
            predicate = d.get("predicate") or (key.split("::")[1] if "::" in key else key)
            mentions = int(d.get("mention_count", 1))

            import json
            raw_s = d.get("source_ids", "[]")
            try:
                s_list = json.loads(raw_s) if isinstance(raw_s, str) else list(raw_s)
            except Exception:
                s_list = [raw_s] if raw_s else []
            s_str = ", ".join(s_list) if s_list else d.get("source_id", "N/A")

            raw_dates = d.get("doc_dates", "[]")
            try:
                dates_list = json.loads(raw_dates) if isinstance(raw_dates, str) else list(raw_dates)
            except Exception:
                dates_list = [raw_dates] if raw_dates else []
            dates_str = "\n  - " + "\n  - ".join(dates_list) if dates_list else d.get("doc_date", "N/A")

            display_label = f"{predicate} ({mentions}x)" if mentions > 1 else predicate

            net.add_node(
                n,
                label=display_label or n[:12],
                title=(
                    f"record_id={n}\n"
                    f"semantic_key={key}\n"
                    f"state={state.value}\n"
                    f"mention_count={mentions}\n"
                    f"source_docs={s_str}\n"
                    f"doc_dates={dates_str}\n"
                    f"tier={getattr(tier, 'name', tier)}\n"
                    f"validated_at={d.get('validated_at')}"
                ),
                color=_STATE_COLOR.get(state, "#607d8b"),
                shape="dot",
                size=14 if mentions > 1 else (11 if state == NodeState.ACTIVE else 8),
            )

    for u, v, d in g.edges(data=True):
        kind = d.get("kind")
        if kind in (EdgeType.DERIVED_FROM.value, EdgeType.SUPERSEDES.value,
                   EdgeType.REVERTED_FROM.value) and not include_bookkeeping_edges:
            continue
        style = _EDGE_STYLE.get(kind, {"dashes": False, "color": "#bdbdbd", "width": 1})
        distance = float(d.get("distance", 1.0))
        label = "SUPERSEDES" if kind == EdgeType.SUPERSEDES.value else ""
        edge_source = d.get("source_id", "")
        edge_date = d.get("doc_date", "")
        edge_mentions = d.get("mention_count", "")
        edge_title = f"kind={kind}\ndistance={distance:.2f}"
        if edge_source:
            edge_title += f"\nsource={edge_source}"
        if edge_date:
            edge_title += f"\ndoc_date={edge_date}"
        if edge_mentions and int(edge_mentions) > 1:
            edge_title += f"\nmentions={edge_mentions}x"

        net.add_edge(
            u, v, **style, arrows="to", length=_edge_length(distance),
            title=edge_title,
            label=label,
            font={"size": 9, "color": "#757575"},
        )

    html = net.generate_html(notebook=False)
    log.info("VIZ rendered | patient=%s nodes=%d edges=%d",
             patient_id or "(all)", g.number_of_nodes(), g.number_of_edges())
    return html


def graph_json(dag: "KnowledgeDAG") -> dict:
    """Node-link JSON (d3 / cytoscape-friendly) as a lighter alternative to HTML."""
    g = dag.nx
    nodes = []
    for n, d in g.nodes(data=True):
        if d.get("kind") == "entity":
            nodes.append({"id": n, "kind": "entity", "label": d.get("label", n),
                          "entity_type": d.get("entity_type", "")})
        else:
            nodes.append({"id": n, "kind": "record", "semantic_key": d.get("semantic_key", ""),
                          "state": d.get("state", NodeState.ACTIVE).value,
                          "tier": getattr(d.get("tier"), "name", d.get("tier"))})
    edges = [{"source": u, "target": v, "kind": d.get("kind", ""),
             "distance": float(d.get("distance", 1.0))} for u, v, d in g.edges(data=True)]
    return {"nodes": nodes, "edges": edges}
