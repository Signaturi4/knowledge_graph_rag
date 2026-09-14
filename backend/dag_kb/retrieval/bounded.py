"""Phase 5 -- bounded-context retrieval (PRD section 3).

Weighted Dijkstra from a root over the ``distance`` edge weight, filtered to a
threshold, over the ACTIVE-only view so TBD / FLAGGED / ARCHIVED nodes are never
served. This is the deterministic replacement for probabilistic vector search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from ..graph import KnowledgeDAG


class BoundedRetriever:
    def __init__(self, dag: "KnowledgeDAG") -> None:
        self._dag = dag

    def context(
        self,
        root_id: str,
        threshold: float,
        *,
        weight: str = "distance",
    ) -> list[tuple[str, float]]:
        """Nodes within ``threshold`` of ``root_id``, ascending by distance.

        Mirrors the PRD reference:
            paths = nx.single_source_dijkstra_path_length(G, root, weight='distance')
            {n: d for n, d in paths.items() if d <= threshold}
        """
        view = self._dag.active_view()
        if root_id not in view:
            return []
        # dijkstra needs weights on every edge; default any missing to 1.0
        for _, _, data in view.edges(data=True):
            data.setdefault(weight, 1.0)
        lengths = nx.single_source_dijkstra_path_length(view, root_id, weight=weight)
        within = [(n, d) for n, d in lengths.items() if d <= threshold]
        within.sort(key=lambda t: t[1])
        return within

    def triples(self, root_id: str, threshold: float) -> list[str]:
        """Human-readable relations for the bounded context (feeds graph-QA)."""
        view = self._dag.active_view()
        keep = {n for n, _ in self.context(root_id, threshold)}
        out = []
        for u, v, d in view.edges(data=True):
            if u in keep and v in keep:
                sk_u = view.nodes[u].get("semantic_key", u)
                sk_v = view.nodes[v].get("semantic_key", v)
                out.append(f"{sk_u} -[{d.get('kind', 'REL')}]-> {sk_v}")
        return out
