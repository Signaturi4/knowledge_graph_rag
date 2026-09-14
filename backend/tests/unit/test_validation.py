"""Per-stage guardrails (pipeline.validation) + strict determinism caveat."""
from __future__ import annotations

import pytest

from dag_kb import Delta, MalformedDraft, Tier, TripletDraft, utcnow
from llm.provider import LLMUnavailable
from pipeline.validation import (
    validate_delta_payload,
    validate_entities_payload,
    validate_extraction_payload,
    validate_triplet_draft,
)


# --- extraction payload ---------------------------------------------------- #
def test_extraction_accepts_dicts_and_tuples():
    out = validate_extraction_payload({"triplets": [
        {"subject": "A", "predicate": "rel", "object": "B"},
        ["C", "T", "rel2", "D", "T"],
    ]})
    assert {o["subject"] for o in out} == {"A", "C"}


@pytest.mark.parametrize("bad", [
    {"error": "cannot comply"},
    {"nope": 1},
    "not json object",
    {"triplets": [{"subject": "", "predicate": "p", "object": "o"}]},   # all rows invalid
    {"triplets": [{"subject": "x", "predicate": "p", "object": "nan"}]},
])
def test_extraction_rejects_garbage(bad):
    with pytest.raises(LLMUnavailable):
        validate_extraction_payload(bad)


def test_extraction_rejects_value_laden_predicate_seen_live():
    """Regression: a live run against claude_bridge produced exactly this shape --
    the object's value and an 'effective ...' date folded into the predicate,
    with the real object left as a generic location. This must never become a
    committed draft: it silently breaks supersession (a differently-worded
    duplicate can never match this predicate again)."""
    bad = {"triplets": [{
        "subject": "Resolution 511",
        "predicate": "sets maximum building height to 45 stories, effective 2026-06-01, in",
        "object": "Sector 4",
    }]}
    with pytest.raises(LLMUnavailable):
        validate_extraction_payload(bad)


def test_semantic_key_canonicalizes_common_predicate_abbreviations():
    """Regression: a live run extracted 'maximum building height' from one
    document and 'max building height' from its amendment -- two perfectly
    correct extractions that must still collide on one semantic key. Both are
    functional (single-valued) attributes, so cardinality=ONE."""
    from dag_kb import Cardinality
    from pipeline.stage1_extract import semantic_key

    assert (semantic_key("Sector 4", "maximum building height", cardinality=Cardinality.ONE)
            == semantic_key("Sector 4", "max building height", cardinality=Cardinality.ONE)
            == "sector_4::maximum_building_height")


def test_semantic_key_many_cardinality_keys_by_object():
    """A MANY-cardinality predicate (a true graph relation, e.g. 'symptom')
    must NOT collide across distinct objects -- each is an independent edge."""
    from dag_kb import Cardinality
    from pipeline.stage1_extract import semantic_key

    fever = semantic_key("patient_0", "symptom", cardinality=Cardinality.MANY, obj="fever")
    cough = semantic_key("patient_0", "symptom", cardinality=Cardinality.MANY, obj="dry cough")
    assert fever != cough
    assert fever == "patient_0::symptom::fever"


# --- triplet draft ------------------------------------------------------- #
def test_triplet_draft_requires_entity_attribute_key():
    d = TripletDraft(semantic_key="noseparator", predicate="p", obj="o", tier=Tier.T4,
                     auto_update=True, source_id="s", raw_citation="c", valid_from=utcnow())
    with pytest.raises(MalformedDraft):
        validate_triplet_draft(d)


def test_derived_draft_needs_parents():
    d = TripletDraft(semantic_key="a::b", predicate="p", obj="o", tier=Tier.T3,
                     auto_update=True, source_id="s", raw_citation="c", valid_from=utcnow(),
                     record_type="derived")
    with pytest.raises(MalformedDraft):
        validate_triplet_draft(d)


# --- delta label ------------------------------------------------------- #
def test_delta_payload_enum_enforced():
    assert validate_delta_payload({"delta": "REFINEMENT", "rationale": "x"})[0] is Delta.REFINEMENT
    with pytest.raises(LLMUnavailable):
        validate_delta_payload({"delta": "MAYBE"})
    with pytest.raises(LLMUnavailable):
        validate_delta_payload({"error": "n/a"})


# --- entities ------------------------------------------------------- #
def test_entities_must_be_substrings_of_query():
    got = validate_entities_payload({"entities": ["Sector 4", "Mars", "  zoning "]},
                                    "what is the sector 4 zoning limit")
    assert got == ["Sector 4", "zoning"]                # "Mars" dropped (not in query)


# --- strict caveat present on every agent ----------------------------- #
def test_every_agent_prompt_carries_the_determinism_contract():
    from llm.prompting import STRICT_CAVEAT, Agent, strict

    for a in Agent:
        s = strict(a, "do the thing")
        assert STRICT_CAVEAT in s
        assert f"[{a.value}]" in s
