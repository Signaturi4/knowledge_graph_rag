"""Render a persisted knowledge graph to a standalone HTML file + print headline stats.

Usage:
  python eval/export_viz.py --data-dir "$(pwd)/data/pmc_kb_full" --out eval/pmc_graph.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from service.assembly import build_memory_layer  # noqa: E402
from viz.pyvis_render import render_html  # noqa: E402
from dag_kb import EdgeType  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "graph.html"))
    args = ap.parse_args()

    layer = build_memory_layer(data_dir=args.data_dir)
    g = layer.dag.nx

    html = render_html(layer.dag, physics=g.number_of_nodes() < 800)
    with open(args.out, "w") as f:
        f.write(html)

    state_counts = Counter(d["state"].value for _, d in g.nodes(data=True) if d.get("kind") != "entity")
    edge_counts = Counter(d.get("kind", "?") for _, _, d in g.edges(data=True))
    entities = sum(1 for _, d in g.nodes(data=True) if d.get("kind") == "entity")
    docs = {r.source_id for r in layer.store}
    stats = {
        "entities_total": entities,
        "relations_total": g.number_of_nodes() - entities,
        "nodes_total": g.number_of_nodes(),
        "nodes_by_state": dict(state_counts),
        "edges_total": g.number_of_edges(),
        "edges_by_kind": dict(edge_counts),
        "dependencies": edge_counts.get(EdgeType.DERIVED_FROM.value, 0),
        "semantic_keys": len(layer.heads.all_keys()),
        "documents_ingested": len(docs),
        "pending_system_2": len(layer.queue.pending()),
    }
    print(json.dumps(stats, indent=2))
    print(f"\nvisualization -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
