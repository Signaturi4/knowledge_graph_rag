"""Section 6b -- R9.1 / R9.2 : precedence + auto_allowed are pure and correct."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dag_kb.ids import compute_record_id
from dag_kb import auto_allowed, precedence
from dag_kb import Precedence, Record, Tier


def rec(tier, days_ago=0, auto=True, seq=0):
    return Record(
        record_id=compute_record_id(semantic_key="k", predicate="p", obj=seq,
                                    parent_ids=(), owner="o", owner_seq=seq),
        semantic_key="k", predicate="p", obj=seq, parent_ids=(), owner="o", owner_seq=seq,
        tier=tier, auto_update=auto, record_type="asserted",
        valid_from=datetime.now(timezone.utc) - timedelta(days=days_ago),
        source_id="s", raw_citation="c", txn_id="t",
    )


def test_precedence_by_tier_then_recency():
    assert precedence(rec(Tier.T0), rec(Tier.T2)) is Precedence.SUPERSEDE
    assert precedence(rec(Tier.T4), rec(Tier.T1)) is Precedence.REJECT
    assert precedence(rec(Tier.T2, days_ago=0), rec(Tier.T2, days_ago=5)) is Precedence.SUPERSEDE
    assert precedence(rec(Tier.T2, days_ago=5), rec(Tier.T2, days_ago=0)) is Precedence.REJECT


def test_auto_allowed_rules():
    assert auto_allowed(rec(Tier.T0), rec(Tier.T0)) is False              # T0 never auto
    assert auto_allowed(rec(Tier.T2, auto=False), rec(Tier.T2)) is False  # flag off
    assert auto_allowed(rec(Tier.T2), rec(Tier.T2)) is True               # within tier
    assert auto_allowed(rec(Tier.T2), rec(Tier.T4)) is False              # weaker draft
    assert auto_allowed(rec(Tier.T1), rec(Tier.T1)) is True
    assert auto_allowed(rec(Tier.T1), rec(Tier.T2)) is False
    assert auto_allowed(rec(Tier.T3), rec(Tier.T4)) is True               # derived: always
