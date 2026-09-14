"""pipeline.stage1_extract -- NVIDIA-reused parsing + our additions
(cardinality, entity-vs-literal classification, canonical_subject override)."""
from __future__ import annotations

import pytest

from dag_kb import Cardinality
from llm.provider import LLMUnavailable
from pipeline.stage1_extract import (
    _is_entity_shaped,
    _is_functional,
    _nvidia_process_response,
    semantic_key,
)


# --- NVIDIA-format parsing (ast.literal_eval on a list of 5-tuples) ----- #
def test_parses_clean_tuple_list():
    reply = "[('Sector 4', 'Zone', 'zoning limit', '30 stories', 'METRIC')]"
    rows = _nvidia_process_response(reply)
    assert rows == [{"subject": "Sector 4", "subject_type": "Zone", "predicate": "zoning limit",
                     "object": "30 stories", "object_type": "METRIC"}]


def test_bracket_slices_prose_wrapped_reply():
    reply = "Here you go:\n```\n[('A', 'ORG', 'Has', 'B', 'PRODUCT')]\n```\nHope that helps!"
    rows = _nvidia_process_response(reply)
    assert rows[0]["subject"] == "A" and rows[0]["object"] == "B"


def test_skips_malformed_tuple_keeps_rest():
    reply = "[('A', 'ORG', 'Has', 'B', 'PRODUCT'), ('bad', 'tuple')]"
    rows = _nvidia_process_response(reply)
    assert len(rows) == 1 and rows[0]["subject"] == "A"


@pytest.mark.parametrize("reply", [
    '{"error": "cannot comply"}',   # a dict, not a list -- must not be iterated as 5-char keys
    "not parseable at all {{{",
    "",
])
def test_non_list_or_unparseable_reply_fails_closed(reply):
    with pytest.raises(LLMUnavailable):
        _nvidia_process_response(reply)


# --- cardinality classification (our addition) --------------------------- #
def test_functional_predicates_recognized():
    assert _is_functional("zoning limit")
    assert _is_functional("max building height")
    assert _is_functional("age")
    assert not _is_functional("symptom")
    assert not _is_functional("participates_in")


# --- entity-vs-literal classification (our addition) --------------------- #
def test_entity_shaped_vs_literal():
    assert _is_entity_shaped("COVID-19", "CONDITION") is True
    assert _is_entity_shaped("Sector 4", "GPE") is True
    # aligned with NVIDIA's own lc_graph.py::save_triples_to_csvs, which
    # applies zero type-based filtering downstream -- a typed object (even
    # METRIC) is a node like any other, so repeats (e.g. "Oxygen Saturation"
    # cited by several relations) share one entity via slug dedup.
    assert _is_entity_shaped("30 stories", "METRIC") is True
    # only an *untyped* object falls back to the numeric/percent heuristic
    assert _is_entity_shaped("10%", "") is False


# --- semantic_key cardinality shape -------------------------------------- #
def test_semantic_key_one_vs_many_shape():
    one = semantic_key("Sector 4", "zoning limit", cardinality=Cardinality.ONE, obj="30 stories")
    many = semantic_key("patient_0", "symptom", cardinality=Cardinality.MANY, obj="fever")
    assert one == "sector_4::zoning_limit"          # object not part of the key
    assert many == "patient_0::symptom::fever"       # object IS part of the key
