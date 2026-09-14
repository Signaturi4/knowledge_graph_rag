"""Frequency-count aggregation over the entity graph -- for "what are all the
X of Y" / "N most common X of Y" style queries (e.g. "what are the symptoms
of ARDS", "3 most common symptoms of COVID-19"). Distinct from
``BoundedRetriever``, which answers "what is near this one node" with a
distance-thresholded single path; this answers "count occurrences across
every entity connected to this one, one predicate-hop out."

Reused mechanic, deliberately narrow: the two-phase flood-then-tally shape is
the "information flooding" traversal from GeoRDF2Vec (Boeckling, Paulheim &
Detzler 2025, arXiv:2504.17099, Algorithm 1) -- expand a frontier of nodes
reachable from a seed via a matching predicate, in one pass, rather than a
single fixed hop. We take *only* that traversal shape. We do not take
RDF2Vec's embeddings/word2vec training, or the geographic distance weighting
the paper builds on top of it: this graph has no spatial geometries, and per
the stated use case the ingested medical facts are assumed always-true
(patient case reports), so there is no noisy/uncertain signal to down-weight
the way GeoRDF2Vec down-weights geographically distant edges. What is reused
is the shape "flood from a seed along a matching relation, then aggregate
over what was reached" -- here the aggregation is an exact count instead of a
learned vector, which is what a "3 most common X" query actually needs:
deterministic, auditable, no training, no embeddings.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import KnowledgeDAG


def most_common(
    dag: "KnowledgeDAG",
    seed_entity_id: str,
    *,
    target_predicates: tuple[str, ...],
    via_predicates: tuple[str, ...] = (),
    top_n: int | None = None,
) -> list[tuple[str, int]]:
    """Count how often each distinct ``target_predicates``-object recurs
    among the entities connected to ``seed_entity_id``.

    Two phases, both case-insensitive predicate matching:

    1. Flood: if ``via_predicates`` is given, collect every entity that
       points at the seed through one of those predicates (e.g. every
       patient ``Diagnosed_With`` this disease). If empty, the seed itself
       is the sole "flooded" entity -- for facts recorded directly on it
       (e.g. disease <-Symptom_Of- symptom with no intermediate patient).
    2. Tally: for every flooded entity, count the object of each outgoing
       relation whose predicate is in ``target_predicates`` (e.g. each
       patient's ``Presents_With`` symptoms), keyed by entity label.

    Returns ``(label, count)`` pairs, most frequent first; ties broken by
    label for a stable, reproducible order.
    """
    via = {p.lower() for p in via_predicates}
    target = {p.lower() for p in target_predicates}

    if via:
        flooded = {
            other_id
            for predicate, direction, other_id in dag.neighbors_of(seed_entity_id)
            if direction == "<-" and predicate.lower() in via
        }
    else:
        flooded = {seed_entity_id}

    counts: Counter[str] = Counter()
    for entity_id in flooded:
        for predicate, direction, other_id in dag.neighbors_of(entity_id):
            if direction == "->" and predicate.lower() in target:
                counts[dag.entity_info(other_id).label] += 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:top_n] if top_n is not None else ranked
