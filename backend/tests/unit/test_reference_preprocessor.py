"""eval.reference_preprocessor -- locks in the faithful NVIDIA-original port
(no bracket-slicing, no fail-closed guard, no clinical taxonomy) so the
baseline used by eval/compare_reference.py doesn't silently drift from what
the original script actually did. All LLM calls are mocked -- deterministic.
"""
from __future__ import annotations

import pytest

from eval.reference_preprocessor import (
    _ORIGINAL_SYSTEM_PROMPT,
    extract_triples,
    generate_qa_pair,
    get_llm_as_a_judge_score,
    judge_prompt_template,
    process_response,
)
from llm.mock import MockProvider


# -- process_response: the original's exact (unguarded) behavior ----------- #
def test_process_response_parses_clean_tuple_list():
    reply = "[('Apple Inc.', 'ORG', 'Introduce', 'iPhone 14', 'PRODUCT')]"
    rows = process_response(reply)
    assert rows == [{"subject": "Apple Inc.", "subject_type": "ORG", "relation": "Introduce",
                     "object": "iPhone 14", "object_type": "PRODUCT"}]


def test_process_response_skips_malformed_tuple_keeps_rest():
    reply = "[('A', 'ORG', 'Has', 'B', 'PRODUCT'), ('bad', 'tuple')]"
    rows = process_response(reply)
    assert len(rows) == 1 and rows[0]["subject"] == "A"


def test_process_response_does_not_bracket_slice_unlike_production():
    # the original has no defensive bracket-slicing through prose/fences --
    # a wrapped reply is NOT valid Python and raises, unlike production's
    # _nvidia_process_response which recovers it.
    reply = "Here you go:\n```\n[('A', 'ORG', 'Has', 'B', 'PRODUCT')]\n```"
    with pytest.raises(SyntaxError):
        process_response(reply)


def test_process_response_does_not_fail_closed_on_dict_unlike_production():
    # the original iterates whatever ast.literal_eval returns -- a dict's
    # keys unpack character-by-character instead of raising. This is the
    # exact bug production's isinstance-list guard was added to prevent;
    # reproduced here (not fixed) because this module is a faithful baseline.
    reply = '{"error": "cannot comply"}'
    rows = process_response(reply)
    # "error" (5 chars) unpacks into the 5-tuple fields -- garbage, not a raise
    assert rows == [{"subject": "e", "subject_type": "r", "relation": "r",
                     "object": "o", "object_type": "r"}]


# -- extract_triples: same prompt, adapted LLM call ------------------------- #
def test_extract_triples_uses_original_system_prompt_and_provider_seam():
    prov = MockProvider({
        "Apple Inc. is introducing": "[('Apple Inc.', 'ORG', 'Introduce', 'iPhone 14', 'PRODUCT')]",
    }, default="[]")
    rows = extract_triples("Apple Inc. is introducing the iPhone 14.", prov)
    assert rows == [{"subject": "Apple Inc.", "subject_type": "ORG", "relation": "Introduce",
                     "object": "iPhone 14", "object_type": "PRODUCT"}]
    # the exact user text is the prompt (unlike production, which prefixes a
    # canonical_subject instruction) -- the taxonomy goes through `system=`
    assert prov.calls == ["Apple Inc. is introducing the iPhone 14."]


def test_original_taxonomy_has_no_clinical_types_or_verbs():
    # the reference's system prompt must NOT contain production's additions --
    # otherwise this stops being a valid baseline for the comparison.
    assert "CONDITION" not in _ORIGINAL_SYSTEM_PROMPT
    assert "TREATMENT" not in _ORIGINAL_SYSTEM_PROMPT
    assert "Diagnosed_With" not in _ORIGINAL_SYSTEM_PROMPT
    assert "Presents_With" not in _ORIGINAL_SYSTEM_PROMPT


# -- judge_prompt_template / get_llm_as_a_judge_score ----------------------- #
def test_judge_prompt_template_formats_with_question_and_answer():
    filled = judge_prompt_template().format(question="Q?", answer="A.")
    assert "Question: Q?" in filled and "Answer: A." in filled
    assert "Total rating:" in filled


def test_get_llm_as_a_judge_score_calls_provider_with_filled_prompt():
    prov = MockProvider({
        "Question: What is X?": "Feedback:::\nEvaluation: solid\nTotal rating: 3",
    }, default="")
    out = get_llm_as_a_judge_score("What is X?", "X is Y.", prov)
    assert "Total rating: 3" in out


# -- generate_qa_pair -------------------------------------------------------- #
def test_generate_qa_pair_parses_json_reply():
    prov = MockProvider({
        "some paragraph": '{"question": "Why?", "answer": "Because."}',
    }, default="{}")
    pair = generate_qa_pair("some paragraph of context", prov)
    assert pair == {"question": "Why?", "answer": "Because."}


def test_generate_qa_pair_returns_none_on_unparseable_reply():
    prov = MockProvider({"garbage in": "not json at all"}, default="not json")
    assert generate_qa_pair("garbage in, garbage out", prov) is None
