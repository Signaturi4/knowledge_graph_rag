"""dag_kb.graph.distance -- deterministic tier + corroboration edge weighting
for BoundedRetriever's weighted Dijkstra (PRD section 3). No recency decay:
this domain's facts (what a treatment does, what a disease presents with)
don't go stale the way operational facts do, so time never factors in --
only authority tier and independent corroboration do.
"""
from __future__ import annotations

import math

import pytest

from dag_kb import TIER_BASE_DISTANCE, Tier, compute_distance


def test_tier_ordering_strongest_authority_is_closest():
    dists = [compute_distance(t) for t in (Tier.T0, Tier.T1, Tier.T2, Tier.T3, Tier.T4)]
    assert dists == sorted(dists)          # strictly increasing T0 -> T4
    assert len(set(dists)) == 5            # every tier distinct


def test_distance_equals_tier_base_with_zero_corroboration():
    for tier in Tier:
        assert compute_distance(tier) == pytest.approx(TIER_BASE_DISTANCE[tier])
        assert compute_distance(tier, corroboration_count=0) == pytest.approx(TIER_BASE_DISTANCE[tier])


def test_distance_is_stable_no_hidden_time_dependence():
    # calling twice (however far apart in wall-clock time) must be identical --
    # there is no age/decay input left to vary it.
    assert compute_distance(Tier.T2) == compute_distance(Tier.T2)


def test_corroboration_tightens_distance_monotonically():
    d0 = compute_distance(Tier.T4, corroboration_count=0)
    d1 = compute_distance(Tier.T4, corroboration_count=1)
    d5 = compute_distance(Tier.T4, corroboration_count=5)
    assert d0 > d1 > d5    # more independent support -> shorter distance


def test_corroboration_has_diminishing_returns():
    # the jump from 0->1 corroborating subject is bigger than 9->10
    d0, d1 = compute_distance(Tier.T4, corroboration_count=0), compute_distance(Tier.T4, corroboration_count=1)
    d9, d10 = compute_distance(Tier.T4, corroboration_count=9), compute_distance(Tier.T4, corroboration_count=10)
    assert (d0 - d1) > (d9 - d10)


def test_corroboration_matches_sqrt_formula():
    base = TIER_BASE_DISTANCE[Tier.T3]
    assert compute_distance(Tier.T3, corroboration_count=3) == pytest.approx(base / math.sqrt(4))


def test_well_corroborated_lower_tier_can_beat_uncorroborated_higher_tier():
    # a T4 fact independently backed by many subjects can still retrieve
    # ahead of a lone, unsupported T0 fact -- corroboration is real signal,
    # not just a tie-breaker.
    lone_t0 = compute_distance(Tier.T0, corroboration_count=0)
    corroborated_t4 = compute_distance(Tier.T4, corroboration_count=50)
    assert corroborated_t4 < lone_t0
