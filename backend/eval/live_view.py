"""A standalone, auto-refreshing live view of a growing knowledge graph.

Decoupled from the main API service on purpose: the main service caches one
MemoryLayer for the life of the process (by design -- see service/deps.py), so
it would never see writes made by a *separate* process (e.g. eval/feed_dataset.py
running in the background). This server re-reads the persisted state (sqlite +
GraphML) from disk on every single request instead, so the page genuinely
reflects whatever the feed job has committed up to that instant.

Usage:
  python eval/live_view.py --data-dir "$(pwd)/data/pmc_kb_full" --port 8078
Then open http://localhost:8078/ and leave the tab open -- it refreshes itself.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402

from dag_kb import EdgeType, KnowledgeDAG  # noqa: E402
from dag_kb.store import SqliteRecordStore  # noqa: E402
from viz.pyvis_render import render_html  # noqa: E402

app = FastAPI(title="Live Knowledge Graph View")
DATA_DIR = os.environ.get("LIVE_VIEW_DATA_DIR", "")
REFRESH_S = int(os.environ.get("LIVE_VIEW_REFRESH_S", "8"))


def _fresh_read():
    """Open a throwaway read-only view of the persisted store + DAG. Never
    holds a long-lived handle, so it always sees the writer process's latest
    commits (each commit is fsync'd sqlite + an immediate GraphML rewrite)."""
    store = SqliteRecordStore(os.path.join(DATA_DIR, "records.sqlite"))
    graphml = os.path.join(DATA_DIR, "dag.graphml")
    if os.path.exists(graphml):
        dag = KnowledgeDAG.read_graphml(graphml, store)
    else:
        dag = KnowledgeDAG(store)
    return store, dag


def _stats(store, dag) -> dict:
    g = dag.nx
    state_counts = Counter(d["state"].value for _, d in g.nodes(data=True) if d.get("kind") != "entity")
    edge_counts = Counter(d.get("kind", "?") for _, _, d in g.edges(data=True))
    entities = sum(1 for _, d in g.nodes(data=True) if d.get("kind") == "entity")
    docs = {r.source_id for r in store}
    return {
        "entities_total": entities,
        "relations_total": g.number_of_nodes() - entities,
        "nodes_total": g.number_of_nodes(),
        "nodes_by_state": dict(state_counts),
        "edges_total": g.number_of_edges(),
        "edges_by_kind": dict(edge_counts),
        "dependencies": edge_counts.get(EdgeType.DERIVED_FROM.value, 0),
        "documents_ingested": len(docs),
        "records_total": sum(1 for _ in store),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/stats")
def stats():
    store, dag = _fresh_read()
    return JSONResponse(_stats(store, dag))


@app.get("/graph.html", response_class=HTMLResponse)
def graph_only():
    _, dag = _fresh_read()
    return HTMLResponse(render_html(dag, physics=dag.nx.number_of_nodes() < 600))


@app.get("/", response_class=HTMLResponse)
def index():
    try:
        store, dag = _fresh_read()
        s = _stats(store, dag)
    except Exception as exc:  # transient sqlite lock while the writer commits -> just retry on refresh
        return HTMLResponse(
            f"<html><head><meta http-equiv='refresh' content='{REFRESH_S}'></head>"
            f"<body style='font-family:system-ui;padding:2em'>"
            f"<p>Transient read error (the writer is mid-commit) -- retrying in {REFRESH_S}s:</p>"
            f"<pre>{exc}</pre></body></html>"
        )
    graph = render_html(dag, physics=dag.nx.number_of_nodes() < 600)
    bar = f"""
    <div style="font-family:system-ui;padding:10px 16px;background:#0f172a;color:#e2e8f0;
                display:flex;gap:28px;align-items:center;flex-wrap:wrap">
      <b style="font-size:15px">🔴 Live Knowledge Graph</b>
      <span>Entities: <b>{s['entities_total']}</b></span>
      <span>Relations: <b>{s['relations_total']}</b></span>
      <span>Edges: <b>{s['edges_total']}</b></span>
      <span>Documents ingested: <b>{s['documents_ingested']}</b></span>
      <span>Records (all versions): <b>{s['records_total']}</b></span>
      <span style="opacity:.7">as of {s['generated_at']} · refreshing every {REFRESH_S}s</span>
      <span style="opacity:.7">{ {k: v for k, v in s['nodes_by_state'].items()} }</span>
    </div>
    """
    # inject the status bar + an auto-refresh tag into pyvis's <head>
    graph = graph.replace("<head>", f"<head>\n<meta http-equiv=\"refresh\" content=\"{REFRESH_S}\">",
                          1)
    graph = graph.replace("<body>", f"<body>\n{bar}", 1)
    return HTMLResponse(graph)


def main() -> int:
    global DATA_DIR, REFRESH_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--port", type=int, default=8078)
    ap.add_argument("--refresh-s", type=int, default=8)
    args = ap.parse_args()

    DATA_DIR = args.data_dir
    REFRESH_S = args.refresh_s

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
