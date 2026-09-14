"""Section 6b/6c -- the deterministic precedence + auto-commit rules.

Pure functions. No LLM, no I/O. The LLM delta label is an *input* to the gate;
whether a draft supersedes the head, and whether that can happen without a human,
are decided only here (technical-plan section 3.2: "this declaration belongs to
application code rather than generated prose").
"""

from __future__ import annotations

from ..types import Precedence, Record, Tier


def precedence(draft: Record, active: Record) -> Precedence:
    """Does ``draft`` supersede the current head ``active``?

    Rule (technical-plan R9.1), tier lattice with T0 strongest (lowest int):
      - stronger tier               -> SUPERSEDE
      - weaker tier                 -> REJECT
      - same tier, newer valid_from -> SUPERSEDE
      - same tier, older valid_from -> REJECT
      - same tier, same instant     -> COEXIST
    """
    if draft.tier < active.tier:
        return Precedence.SUPERSEDE
    if draft.tier > active.tier:
        return Precedence.REJECT
    if draft.valid_from > active.valid_from:
        return Precedence.SUPERSEDE
    if draft.valid_from < active.valid_from:
        return Precedence.REJECT
    return Precedence.COEXIST


def auto_allowed(active: Record, draft: Record) -> bool:
    """May the supersession commit without SYSTEM_2 review? (technical-plan R9.2 / 6c)

      - T0 head            -> never automatic
      - either auto_update flag false -> never automatic
      - T1 head            -> draft.tier must be <= T1
      - T2 / T4 head       -> draft.tier must be <= active.tier
      - T3 head (derived)  -> automatic
    """
    if active.tier is Tier.T0:
        return False
    if not (active.auto_update and draft.auto_update):
        return False
    if active.tier is Tier.T1:
        return draft.tier <= Tier.T1
    if active.tier in (Tier.T2, Tier.T4):
        return draft.tier <= active.tier
    if active.tier is Tier.T3:
        return True
    return False
