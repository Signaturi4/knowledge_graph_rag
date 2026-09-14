"""Full-log viewer + headline knowledgebase stats, for debugging the running system.

GET /logs/stats  -- node/edge/doc/queue counts (the "main numbers")
GET /logs        -- recent log lines across every stage (ingest, retrieval, gate,
                     revert, GC, HTTP), filterable by level / logger prefix / substring
"""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter

from dag_kb import EdgeType, NodeState
from observability import get_ring_buffer

from ..deps import get_layer

router = APIRouter()


@router.get("/stats")
def stats():
    layer = get_layer()
    g = layer.dag.nx

    state_counts = Counter(d["state"].value for _, d in g.nodes(data=True) if d.get("kind") != "entity")
    edge_counts = Counter(d.get("kind", "?") for _, _, d in g.edges(data=True))
    entities = sum(1 for _, d in g.nodes(data=True) if d.get("kind") == "entity")
    docs = {r.source_id for r in layer.store}
    tiers = Counter(r.tier.name for r in layer.store)

    return {
        "entities_total": entities,
        "relations_total": g.number_of_nodes() - entities,
        "nodes_total": g.number_of_nodes(),
        "nodes_by_state": dict(state_counts),
        "edges_total": g.number_of_edges(),
        "edges_by_kind": dict(edge_counts),
        "dependencies": edge_counts.get(EdgeType.DERIVED_FROM.value, 0),
        "semantic_keys": len(layer.heads.all_keys()),
        "documents_ingested": len(docs),
        "records_total": sum(1 for _ in layer.store),
        "records_by_tier": dict(tiers),
        "queue_pending": len(layer.queue.pending()),
        "queue_by_kind": dict(Counter(i.kind.value for i in layer.queue.pending())),
        "audit_rows": len(layer.audit.rows()),
    }


@router.get("")
def logs(limit: int = 500, level: str | None = None, contains: str | None = None,
         logger: str | None = None):
    rows = get_ring_buffer().snapshot(limit=limit, level=level, contains=contains,
                                      logger_prefix=logger)
    return {"count": len(rows), "lines": rows}
