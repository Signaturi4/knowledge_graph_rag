"""A faithful reproduction of NVIDIA's original ``backend/utils/preprocessor.py``
(SPDX Apache-2.0, NVIDIA CORPORATION & AFFILIATES) -- kept as a reference/ground-
truth baseline, separate from the production extractor.

**What's identical to the original, verbatim:** the four function bodies below
are the same logic and the same prompt text (system prompt, entity taxonomy,
relation-verb set, worked example, judge rubric, QA-pair-generation
instruction) as NVIDIA's file. Nothing here has our clinical extension
(CONDITION/TREATMENT types, Diagnosed_With/Presents_With/... verbs) or our
robustness hardening (bracket-slicing through prose/fences, the
non-list-reply fail-closed guard) -- see the "Known, deliberate differences"
note at the bottom for exactly what production (`pipeline/stage1_extract.py`)
does differently and why.

**What's adapted, and only this:** the original chains
`ChatPromptTemplate | ChatNVIDIA | StrOutputParser`, which requires an
`NVIDIA_API_KEY` and the `langchain_nvidia_ai_endpoints` package -- neither of
which this project uses (LLM_PROVIDER=claude_bridge is keyless). Swapped for
a direct call through this project's `LLMProvider.complete()` seam so the
exact same prompts run against the exact same live `claude` CLI backend the
rest of the system uses. `judge_prompt_template()` needed no adaptation at
all -- it was already provider-agnostic (just a `PromptTemplate`-shaped
string); reproduced here as a plain `.format()`-able string instead of
importing `langchain.prompts.PromptTemplate` for one call site.

This module exists to answer one question with evidence, not assertion:
*does our current, modified extraction pipeline still match or exceed the
quality of NVIDIA's original, unmodified one on the same real input?* See
`eval/compare_reference.py` for the side-by-side runner and
`tests/unit/test_reference_preprocessor.py` for the parsing-behavior lock-in.
"""

from __future__ import annotations

import ast
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm.provider import LLMProvider

# -- verbatim from preprocessor.py::extract_triples's system message -------- #
_ORIGINAL_SYSTEM_PROMPT = """Note that the entities should not be generic, numerical, or temporal (like dates or percentages). Entities must be classified into the following categories:
- ORG: Organizations other than government or regulatory bodies
- ORG/GOV: Government bodies (e.g., "United States Government")
- ORG/REG: Regulatory bodies (e.g., "Food and Drug Administration")
- PERSON: Individuals (e.g., "Marie Curie")
- GPE: Geopolitical entities such as countries, cities, etc. (e.g., "Germany")
- INSTITUTION: Academic or research institutions (e.g., "Harvard University")
- PRODUCT: Products or services (e.g., "CRISPR technology")
- EVENT: Specific and Material Events (e.g., "Nobel Prize", "COVID-19 pandemic")
- FIELD: Academic fields or disciplines (e.g., "Quantum Physics")
- METRIC: Research metrics or indicators (e.g., "Impact Factor"), numerical values like "10%" is not a METRIC;
- TOOL: Research tools or methods (e.g., "Gene Sequencing", "Surveys")
- CONCEPT: Abstract ideas or notions or themes (e.g., "Quantum Entanglement", "Climate Change")

The relationships 'r' between these entities must be represented by one of the following relation verbs set: Has, Announce, Operate_In, Introduce, Produce, Control, Participates_In, Impact, Positive_Impact_On, Negative_Impact_On, Relate_To, Is_Member_Of, Invests_In, Raise, Decrease.

Remember to conduct entity disambiguation, consolidating different phrases or acronyms that refer to the same entity (for instance, "MIT" and "Massachusetts Institute of Technology" should be unified as "MIT"). Simplify each entity of the triplet to be less than four words. However, always make sure it is a sensible entity name and not a single letter or NAN value.

From this text, your output Must be in python list of tuple with each tuple made up of ['h', 'type', 'r', 'o', 'type'], each element of the tuple is the string, where the relationship 'r' must be in the given relation verbs set above. Only output the list. As an Example, consider the following news excerpt:
                        Input :'Apple Inc. is set to introduce the new iPhone 14 in the technology sector this month. The product's release is likely to positively impact Apple's stock value.'
                        OUTPUT : ```
                            [('Apple Inc.', 'COMP', 'Introduce', 'iPhone 14', 'PRODUCT'),
                            ('Apple Inc.', 'COMP', 'Operate_In', 'Technology Sector', 'SECTOR'),
                            ('iPhone 14', 'PRODUCT', 'Positive_Impact_On', 'Apple's Stock Value', 'FIN_INSTRUMENT')]
                        ```
      The output structure must not be anything apart from above OUTPUT structure. NEVER REPLY WITH any element as NAN. Just leave out the triple if you think it's not worth including or does not have an object. Do not provide ANY additional explanations, if it's not a Python parseable list of tuples, you will be penalized severely. Make the best possible decisions given the context."""


def process_response(triplets_str: str) -> list[dict]:
    """Verbatim port of ``preprocessor.py::process_response``: parse a Python-
    literal list of 5-tuples into dicts, skipping malformed tuples. Unlike
    production's `_nvidia_process_response`, this does NOT bracket-slice
    through prose/code-fences first, and does NOT fail closed on a non-list
    literal (e.g. a bare error dict) -- reproduced exactly as written, not
    hardened, so it is a true baseline for the comparison."""
    triplets_list = ast.literal_eval(triplets_str)
    json_triplets = []
    for triplet in triplets_list:
        try:
            subject, subject_type, relation, obj, object_type = triplet
            json_triplets.append({
                "subject": subject, "subject_type": subject_type,
                "relation": relation, "object": obj, "object_type": object_type,
            })
        except ValueError:
            continue
    return json_triplets


def extract_triples(text: str, llm: "LLMProvider") -> list[dict]:
    """Verbatim prompt/logic port of ``preprocessor.py::extract_triples``,
    with the LangChain/ChatNVIDIA chain swapped for this project's
    ``LLMProvider.complete()`` seam (see module docstring)."""
    response = llm.complete(text, system=_ORIGINAL_SYSTEM_PROMPT, timeout_s=90.0)
    return process_response(response)


# -- verbatim from preprocessor.py::judge_prompt_template ------------------- #
_JUDGE_PROMPT_TEMPLATE = """
    You will be given a user_question and system_answer couple.
    Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
    Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question.

    Here is the scale you should use to build your answer:
    1: The system_answer is terrible: completely irrelevant to the question asked, or very partial
    2: The system_answer is mostly not helpful: misses some key aspects of the question
    3: The system_answer is mostly helpful: provides support, but still could be improved
    4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question

    Provide your feedback as follows:

    Feedback:::
    Evaluation: (your rationale for the rating, as a text)
    Total rating: (your rating, as a number between 1 and 4)

    You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

    Now here are the question and answer.

    Question: {question}
    Answer: {answer}

    Provide your feedback. If you give a correct rating, I'll tip you $200.
    Feedback:::
    Evaluation:
    """


def judge_prompt_template() -> str:
    """Verbatim port of ``preprocessor.py::judge_prompt_template``. The
    original wrapped this string in a ``langchain.prompts.PromptTemplate``
    purely so ``.format(question=..., answer=...)`` worked with variable
    validation; a plain ``str`` needs no adaptation for that, so
    ``judge_prompt_template().format(question=q, answer=a)`` works the same
    way without the LangChain dependency."""
    return _JUDGE_PROMPT_TEMPLATE


def get_llm_as_a_judge_score(question: str, answer: str, llm: "LLMProvider") -> str:
    """Not in the original file directly (that logic lived in
    ``legacy/evaluation.py::get_llm_as_a_judge_scores``, which this reference
    module does not otherwise reproduce) -- included here as the minimal glue
    needed to actually exercise ``judge_prompt_template()`` end to end against
    our LLM seam. Returns the judge's raw feedback text (rating parsing is the
    caller's job, matching the original's own "Total rating: N" text format)."""
    prompt = judge_prompt_template().format(question=question, answer=answer)
    return llm.complete(prompt, timeout_s=60.0)


def generate_qa_pair(text: str, llm: "LLMProvider") -> dict | None:
    """Verbatim prompt/logic port of ``preprocessor.py::generate_qa_pair``:
    one synthetic, multi-step-reasoning question/answer pair per input
    paragraph, for building an eval set without hand-labeling. ``None`` on
    unparseable JSON, exactly like the original's bare ``except: return
    None`` (kept, not hardened, for a faithful baseline)."""
    system = ("You are a synthetic data generation model responsible for creating high "
             "quality question and answer pairs from text content provided to you. Given "
             "the paragraph as an input, create one high quality and highly complex "
             "question answer pair. The question should require a large portion of the "
             "context and multi-step advanced reasoning to answer. Make sure it is "
             "something a human may ask while reading this document. The answer should "
             "be highly detailed and comprehensive. Your output should be in a json "
             "format of one question answer pair. Restrict the question to the context "
             "information provided. Do not print anything else. The output MUST be JSON "
             "parseable.")
    response = llm.complete(text, system=system, timeout_s=90.0)
    try:
        return json.loads(response)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Known, deliberate differences from this reference vs. production
# (`pipeline/stage1_extract.py`) -- not omissions, intentional divergence:
#
# 1. Entity taxonomy: production adds CONDITION/TREATMENT for clinical text;
#    this reference keeps the original's general business/news taxonomy only.
# 2. Relation verbs: production adds 6 clinical verbs (Diagnosed_With,
#    Presents_With, Treated_With, Caused_By, Symptom_Of, Underwent); this
#    reference keeps the original 14-verb set only.
# 3. Parsing robustness: production's `_nvidia_process_response` bracket-
#    slices through code-fence/prose wrapping and fails closed
#    (`LLMUnavailable`) on a non-list literal (see the dict-as-triplet-list
#    character-unpacking bug this guarded against). This reference's
#    `process_response` is the original's exact, unguarded behavior.
# 4. Graph wiring: production additionally classifies cardinality
#    (ONE/MANY), entity-vs-literal shape, and slugs subject/object into real
#    graph nodes (`dag_kb`'s entity/relation model). This reference stops at
#    the flat dict list the original NVIDIA script also stopped at -- it
#    does not build a graph at all.
# --------------------------------------------------------------------------- #
