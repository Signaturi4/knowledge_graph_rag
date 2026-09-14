"""As-is smoke test of the knowledge_graph_rag SUBSTRATE we are keeping.

Runs the exact NetworkX assembly + GraphML round-trip + graph-QA retrieval path
used by backend/routers/ui_backend.py and backend/routers/chat.py, but with a
fixed sample triple set instead of the NVIDIA LLM extractor. No NVIDIA API key,
no Milvus, no vector search.

Exercised, unchanged from the upstream repo:
  - utils.lc_graph.save_triples_to_csvs  (triples -> entities/relations/triples CSV)
  - nx.from_pandas_edgelist + relabel    (CSV -> DiGraph, as in ui_backend)
  - nx.write_graphml / nx.read_graphml   (persistence)
  - NetworkxEntityGraph.get_entity_knowledge  (multi-hop retrieval, as in chat.py)
"""
from __future__ import annotations

import os
import sys

import networkx as nx
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.abspath(os.path.join(HERE, "..", "backend"))
sys.path.insert(0, BACKEND)

DATA_DIR = os.environ["DATA_DIR"]

# --- 1. sample triples in the repo's own 5-tuple shape ---------------------- #
# (subject, subject_type, relation, object, object_type)
SAMPLE_TRIPLES = [
    {"subject": "URDFormer", "subject_type": "TOOL", "relation": "Relate_To",
     "object": "Vision Transformer", "object_type": "CONCEPT"},
    {"subject": "Vision Transformer", "subject_type": "CONCEPT", "relation": "Has",
     "object": "Attention Mechanism", "object_type": "CONCEPT"},
    {"subject": "URDFormer", "subject_type": "TOOL", "relation": "Operate_In",
     "object": "Robotics", "object_type": "FIELD"},
]


def main() -> int:
    os.makedirs(DATA_DIR, exist_ok=True)

    # save_triples_to_csvs chdir's into ./data relative to cwd; match ui_backend
    os.chdir(os.path.dirname(DATA_DIR))

    from utils.lc_graph import save_triples_to_csvs  # noqa: E402  (repo module)

    rows = [tuple(t.values()) for t in SAMPLE_TRIPLES]
    save_triples_to_csvs(rows)
    print("[1/4] wrote entities.csv / relations.csv / triples.csv")

    # --- 2. CSV -> DiGraph, identical to ui_backend.background_task --------- #
    triples_df = pd.read_csv(os.path.join(DATA_DIR, "triples.csv"))
    entities_df = pd.read_csv(os.path.join(DATA_DIR, "entities.csv"))
    relations_df = pd.read_csv(os.path.join(DATA_DIR, "relations.csv"))

    entity_name_map = entities_df.set_index("entity_id")["entity_name"].to_dict()
    relation_name_map = relations_df.set_index("relation_id")["relation_name"].to_dict()

    G = nx.from_pandas_edgelist(
        triples_df, source="entity_id_1", target="entity_id_2",
        edge_attr="relation_id", create_using=nx.DiGraph,
    )
    G = nx.relabel_nodes(G, entity_name_map)
    edge_attributes = nx.get_edge_attributes(G, "relation_id")
    nx.set_edge_attributes(
        G,
        {(u, v): relation_name_map[edge_attributes[(u, v)]]
         for u, v in G.edges() if edge_attributes[(u, v)] in relation_name_map},
        "relation",
    )
    print(f"[2/4] built DiGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # --- 3. GraphML round-trip ------------------------------------------------ #
    graphml_path = os.path.join(DATA_DIR, "knowledge_graph.graphml")
    nx.write_graphml(G, graphml_path)
    G2 = nx.read_graphml(graphml_path)
    print(f"[3/4] GraphML round-trip ok -> {graphml_path} ({G2.number_of_nodes()} nodes)")

    # --- 4. graph-QA retrieval path from chat.py ---------------------------- #
    from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

    graph = NetworkxEntityGraph(G2)
    knowledge = graph.get_entity_knowledge("URDFormer", depth=2)
    print("[4/4] get_entity_knowledge('URDFormer', depth=2):")
    for line in knowledge:
        print("      ", line)

    assert G2.number_of_nodes() == 4, G2.number_of_nodes()
    assert any("URDFormer" in k for k in knowledge)
    print("\nSMOKE OK - substrate (NetworkX KG assembly + retrieval) works unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
