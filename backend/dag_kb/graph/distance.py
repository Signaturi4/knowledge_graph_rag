"""Deterministic edge-distance scoring for BoundedRetriever's weighted Dijkstra
(``dag_kb/retrieval/bounded.py``, PRD section 3).

That retriever, and every edge ``KnowledgeDAG`` creates, already carry a
``distance`` attribute end to end (graph, graphml persistence, Dijkstra
traversal, threshold filtering) -- it was just never fed anything but the
uniform default of 1.0. This module computes the real value.

Grounded in one signal already established in this codebase, not a learned
embedding model: authority tier (``dag_kb.types.Tier``) -- PLANFENCE's own
confidence ordering (T0 canonical ... T4 observational). It is already an
``IntEnum`` by construction (T0=0 strongest ... T4=4 weakest), so it is
exactly the "confidence score" signal the retriever needs, with zero new
modeling. Fully deterministic and reproducible, consistent with this
project's fail-closed / auditable design.

No recency decay: an earlier version of this module also decayed distance by
record age. Removed -- there is no use case for it here. Medical facts in
this KG (what a treatment does, what a disease presents with) don't go stale
the way operational/business facts do; a treatment that worked 10 years ago
is still valid evidence today, and the same diagnosis can recur. Time-since-
ingestion is not a confidence signal for this domain.

Corroboration, not decay, is the second signal: an object entity independently
reached from several distinct subjects (e.g. "Fever" reached by
``Presents_With`` from several different patients, not just one) is better
supported than one asserted a single time, so it should retrieve ahead of it
at the same tier. This is exactly the graph-degree signal already computable
from ``KnowledgeDAG.neighbors_of`` -- no new modeling, no training, no
embeddings, just counting distinct subjects already on record for that
entity at commit time.

Why not PyKEEN (or a trained KG-embedding model in general): PyKEEN's
``nn.weighting`` module weights relations/entities *during training* (e.g.
for R-GCN-style message passing) -- it does not export a portable per-edge
confidence scalar. Producing one would require training a stochastic
embedding model (non-deterministic without a pinned seed, and wanting far
more triples than a single-document or even the full PMC-Patients cut ingest
here produces) to approximate a signal -- authority -- that is already exact
and on hand. Three other libraries floated in discussion (DNoKGraphRAG,
Progeni, "Synthadoc" as an edge-weighting tool) did not resolve to anything
real/matching via Context7 lookup and are not used.
"""

from __future__ import annotations

import math

from ..types import Tier

# Confidence base distance per tier, in PLANFENCE authority order.
TIER_BASE_DISTANCE: dict[Tier, float] = {
    Tier.T0: 1.0,
    Tier.T1: 2.0,
    Tier.T2: 3.0,
    Tier.T3: 4.0,
    Tier.T4: 5.0,
}


def compute_distance(tier: Tier, *, corroboration_count: int = 0) -> float:
    """Deterministic edge distance = tier confidence, tightened by
    corroboration.

        distance = TIER_BASE_DISTANCE[tier] / sqrt(1 + corroboration_count)

    Higher authority (T0) is always closer than lower authority (T4) at
    equal corroboration. ``corroboration_count`` is the number of *other*,
    already-committed distinct subject entities independently pointing at
    the same object entity (e.g. how many other patients already have a
    ``Presents_With`` edge into "Fever" before this one) -- sqrt gives
    diminishing returns, so the tenth corroborating patient tightens things
    less than the second did, while a same-tier fact with more independent
    support still retrieves ahead of one asserted only once. 0 (the default)
    reproduces plain tier-only distance for a first-ever assertion.
    """
    base = TIER_BASE_DISTANCE[tier]
    return base / math.sqrt(1 + max(0, corroboration_count))
