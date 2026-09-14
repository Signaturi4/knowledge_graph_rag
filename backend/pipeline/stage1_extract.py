"""Phase 9, stage 1 -- LLM extraction of triplet drafts.

Extraction prompt, entity taxonomy, closed relation-verb set, and output
parsing are reused near-verbatim from the upstream NVIDIA repo's
``backend/utils/preprocessor.py::extract_triples`` / ``process_response``:
Python list-of-5-tuples output (``ast.literal_eval``, NOT JSON), the same
entity categories, the same relation-verb vocabulary, the same disambiguation
instruction. That is the base graph-construction pattern this project builds
on (entities as nodes, relations as edges -- see ``dag_kb.graph.KnowledgeDAG``).

Layered on top (additions, not replacements of the reused core):
  - the strict determinism caveat (:func:`llm.prompting.strict`) and our
    provider abstraction instead of a hard-coded ``ChatNVIDIA`` call;
  - ``canonical_subject`` — narrative/coreference text names one stable
    subject verbatim, fixing the dominant key-fragmentation failure mode on
    prose (session 3 finding);
  - deterministic (non-LLM) classification of each triplet into an entity
    edge vs a literal attribute, and a cardinality (ONE supersedes / MANY
    coexists) -- NVIDIA's own graph has no versioning at all (every triple is
    just an edge, i.e. implicitly always "many"); reifying each triple as a
    versioned PLANFENCE record is what makes supersession for functional
    attributes (age, zoning limit, ...) possible, and that needs this
    classification since the LLM is not trusted to decide commit/auto shape;
  - the ``_predicate_is_value_laden`` guardrail (defence in depth).

It produces STRUCTURE only; it never decides commit/auto -- that is the
deterministic gate. Guardrails: every returned payload is shape-validated by
:func:`pipeline.validation.validate_extraction_payload` before any draft is
built. Any failure raises ``LLMUnavailable`` -> the pipeline routes to SYSTEM_2.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import TYPE_CHECKING

from dag_kb import Cardinality, Tier, TripletDraft

from llm.prompting import Agent, strict
from llm.provider import LLMUnavailable
from observability import truncate

from .stage0_sources import RawItem
from .validation import validate_extraction_payload, validate_triplet_draft

if TYPE_CHECKING:
    from llm.provider import LLMProvider

log = logging.getLogger("dagkb.pipeline.stage1_extract")

# --------------------------------------------------------------------------- #
# Reused near-verbatim from the NVIDIA repo's
# backend/utils/preprocessor.py::extract_triples system prompt.
# --------------------------------------------------------------------------- #
_NVIDIA_TASK = """Note that the entities should not be generic, numerical, or temporal (like dates or percentages). Entities must be classified into the following categories:
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
- CONDITION: A medical condition, diagnosis, or symptom (e.g., "ARDS", "hypertension")
- TREATMENT: A therapy, drug, or medical procedure (e.g., "cyclophosphamide", "plasma exchange")

The relationships 'r' between these entities must be represented by one of the following relation verbs set: Has, Announce, Operate_In, Introduce, Produce, Control, Participates_In, Impact, Positive_Impact_On, Negative_Impact_On, Relate_To, Is_Member_Of, Invests_In, Raise, Decrease, Diagnosed_With, Presents_With, Treated_With, Caused_By, Symptom_Of, Underwent, Resolves, Discontinues, Revokes.

Remember to conduct entity disambiguation, consolidating different phrases or acronyms that refer to the same entity (for instance, "MIT" and "Massachusetts Institute of Technology" should be unified as "MIT"). Simplify each entity of the triplet to be less than four words. However, always make sure it is a sensible entity name and not a single letter or NAN value.

From this text, your output Must be in python list of tuple with each tuple made up of ['h', 'type', 'r', 'o', 'type'], each element of the tuple is the string, where the relationship 'r' must be in the given relation verbs set above. Only output the list. As an Example, consider the following news excerpt:
                        Input :'Apple Inc. is set to introduce the new iPhone 14 in the technology sector this month. The product's release is likely to positively impact Apple's stock value.'
                        OUTPUT : ```
                            [('Apple Inc.', 'ORG', 'Introduce', 'iPhone 14', 'PRODUCT'),
                            ('Apple Inc.', 'ORG', 'Operate_In', 'Technology Sector', 'FIELD'),
                            ('iPhone 14', 'PRODUCT', 'Positive_Impact_On', 'Apple's Stock Value', 'METRIC')]
                        ```
      The output structure must not be anything apart from above OUTPUT structure. NEVER REPLY WITH any element as NAN. Just leave out the triple if you think it's not worth including or does not have an object. Do not provide ANY additional explanations, if it's not a Python parseable list of tuples, you will be penalized severely. Make the best possible decisions given the context."""

_SYSTEM = strict(Agent.EXTRACTOR, _NVIDIA_TASK)

_MAX_CHARS = 6000

# Cardinality is our addition (technical-plan STATUS.md "multi-valued-fact
# modeling"): NVIDIA's own graph never supersedes -- every triple is just an
# edge, i.e. implicitly always MANY. We default to that, and only treat a
# small, curated set of genuinely functional (single-valued) attributes as
# ONE, so a new value supersedes the old one instead of coexisting with it.
_FUNCTIONAL_PREDICATES = {
    "age", "sex", "gender", "date_of_birth", "name", "status",
    "current_status", "diagnosis", "current_diagnosis", "diagnosed_with",
    "has_diagnosis", "primary_diagnosis", "active_medication", "active_treatment",
    "clinical_status", "disease_status", "zoning_limit",
    "maximum_building_height", "max_building_height", "current_value",
}

_NUMERIC_RE = re.compile(r"^[\s\d.,%$/-]+$")


def _slug(s: str, *, canon: bool = False) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip().lower()).strip("_")
    if canon and s:
        s = "_".join(_WORD_CANON.get(w, w) for w in s.split("_"))
    return s or "unknown"


_WORD_CANON = {
    "max": "maximum", "min": "minimum", "qty": "quantity", "num": "number",
    "amt": "amount", "addr": "address", "id": "identifier",
}


def _is_functional(predicate: str) -> bool:
    slugged = _slug(predicate, canon=True)
    if slugged in _FUNCTIONAL_PREDICATES:
        return True
    if slugged.startswith("has_") and slugged[4:] in _FUNCTIONAL_PREDICATES:
        return True
    return False


def _is_entity_shaped(obj: str, object_type: str) -> bool:
    """Is the object of this triplet worth its own graph node?

    Aligned with NVIDIA's own downstream graph construction
    (utils/lc_graph.py::save_triples_to_csvs): it applies **no** type-based
    filtering at all -- every triple's subject and object becomes a node,
    deduped only by exact string value (``pd.concat([...]).unique()``). So a
    METRIC-typed object like "Oxygen Saturation" that recurs across several
    triples collapses onto the same shared node instead of staying a
    disconnected literal per-triple. NVIDIA's "numerical values like 10% is
    not a METRIC" remark is *prompt-time* guidance to the extractor about
    what counts as an entity in the first place (_NVIDIA_TASK above); it is
    not a post-hoc filter here, so we don't reimplement it as one. We keep
    the numeric/currency/percent heuristic only as a defensive fallback for
    the rare case the extractor omits object_type entirely.
    """
    if not isinstance(obj, str) or not obj.strip():
        return False
    ot = object_type.strip().lower()
    if ot == "":
        return not _NUMERIC_RE.match(obj.strip())
    return True


def semantic_key(subject: str, predicate: str, *, cardinality: Cardinality = Cardinality.MANY,
                 obj: str = "") -> str:
    """A triplet slot identity.

    ONE   -> 'entity::attribute'                 (a new value supersedes the old one)
    MANY  -> 'entity::attribute::object'          (each distinct object is independent)
    """
    base = f"{_slug(subject)}::{_slug(predicate, canon=True)}"
    if cardinality is Cardinality.ONE:
        return base
    return f"{base}::{_slug(str(obj))}"


class Extractor:
    agent = Agent.EXTRACTOR

    def __init__(self, provider: "LLMProvider") -> None:
        self._llm = provider

    def extract(self, raw: RawItem) -> list[TripletDraft]:
        drafts: list[TripletDraft] = []
        chunks = _chunks(raw.text, _MAX_CHARS)
        log.info("STAGE1 extract start | source_id=%s chunks=%d", raw.source_id, len(chunks))
        for ci, chunk in enumerate(chunks):
            prompt_text = f"INPUT TEXT:\n{chunk}"
            if raw.canonical_subject:
                prompt_text = (
                    f"The subject of this document is canonically named "
                    f"'{raw.canonical_subject}'. Whenever a triplet's subject is that "
                    f"same real-world thing -- including when referred to as 'the patient', "
                    f"'he'/'she', or a descriptive role like 'the subject' or 'individual' -- use "
                    f"exactly '{raw.canonical_subject}' as 'h', not the descriptive phrase.\n\n"
                    + prompt_text
                )
            try:
                raw_reply = self._llm.complete(prompt_text, system=_SYSTEM, timeout_s=90.0)
                rows = _nvidia_process_response(raw_reply)
            except LLMUnavailable as exc:
                log.warning("STAGE1 extract FAILED | source_id=%s chunk=%d/%d reason=%s",
                           raw.source_id, ci + 1, len(chunks), exc)
                raise
            except Exception as exc:  # transport / parse -> fail closed (R9.9)
                log.warning("STAGE1 extract call errored | source_id=%s chunk=%d/%d: %s",
                           raw.source_id, ci + 1, len(chunks), exc)
                raise LLMUnavailable(f"extractor call failed: {exc}") from exc

            valid_rows = validate_extraction_payload({"triplets": rows})
            log.info("STAGE1 extract chunk %d/%d | source_id=%s raw_triplets=%d -> valid=%d",
                     ci + 1, len(chunks), raw.source_id, len(rows), len(valid_rows))
            for t in valid_rows:
                subject = t["subject"]
                if raw.canonical_subject and _SELF_REF.match(subject.strip()):
                    subject = raw.canonical_subject

                cardinality = Cardinality.ONE if _is_functional(t["predicate"]) else Cardinality.MANY
                obj_str = t["object"] if isinstance(t["object"], str) else str(t["object"])
                entity_shaped = _is_entity_shaped(obj_str, t["object_type"])

                key = semantic_key(subject, t["predicate"], cardinality=cardinality, obj=obj_str)
                draft = TripletDraft(
                    semantic_key=key,
                    predicate=t["predicate"],
                    obj=t["object"],
                    tier=raw.tier,
                    auto_update=raw.auto_update,
                    source_id=raw.source_id,
                    raw_citation=raw.citation,
                    valid_from=raw.valid_from,
                    subject_ref=_slug(subject),
                    object_ref=_slug(obj_str) if entity_shaped else None,
                    cardinality=cardinality,
                    provenance={
                        "subject": subject,
                        "subject_type": t["subject_type"],
                        "object_label": obj_str,
                        "object_type": t["object_type"],
                        "extractor_agent": self.agent.value,
                        "extractor_impl": getattr(self._llm, "name", "llm"),
                    },
                )
                try:
                    validate_triplet_draft(draft)
                except Exception as exc:  # malformed after normalization -> skip this one
                    log.debug("STAGE1 dropped malformed draft | %s: %s", draft.semantic_key, exc)
                    continue
                log.debug("STAGE1 draft | key=%s predicate=%r object=%r tier=%s cardinality=%s",
                         draft.semantic_key, draft.predicate, draft.obj, draft.tier.name,
                         cardinality.value)
                drafts.append(draft)
        log.info("STAGE1 extract done | source_id=%s drafts=%d", raw.source_id, len(drafts))
        return drafts


_SELF_REF = re.compile(
    r"^(the\s+)?(patient(\s+[\w\d]+)?|pt[-\s]?[\w\d]+|subject|individual|client|person|he|she|him|her|his|the\s+man|the\s+woman|"
    r"a\s+\d+[\s-]*y.*|\d+[\s-]*y.*|a\s+\d+[\s-]*year[\s-]*old.*|this\s+\d+[\s-]*year[\s-]*old.*)$",
    re.I,
)


def _chunks(text: str, size: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    return [text[i : i + size] for i in range(0, len(text), size)]


def _nvidia_process_response(triplets_str: str) -> list[dict]:
    """Ported from NVIDIA's ``preprocessor.py::process_response``: parse a
    Python list-of-5-tuples reply via ``ast.literal_eval`` (not JSON). Adds
    defensive bracket-slicing first (the claude_bridge / claude-code-CLI path
    sometimes wraps the list in prose or code fences; NVIDIA's original code,
    written for a raw ChatNVIDIA completion, assumes a clean reply)."""
    s = triplets_str.strip()
    start, end = s.find("["), s.rfind("]")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]
    try:
        triplets_list = ast.literal_eval(s)
    except (ValueError, SyntaxError) as exc:
        raise LLMUnavailable(f"extractor reply is not a Python literal list: {exc}") from exc

    # Fail closed on anything that isn't a list (e.g. a bare {"error": "..."}
    # dict, or prose ast.literal_eval happens to accept). Without this check a
    # dict's keys would silently be iterated as if they were 5-char strings
    # and unpacked character-by-character into a garbage triplet.
    if not isinstance(triplets_list, list):
        raise LLMUnavailable(
            f"extractor reply parsed to {type(triplets_list).__name__}, not a list: {s[:200]!r}"
        )

    out: list[dict] = []
    for triplet in triplets_list:
        try:
            subject, subject_type, relation, obj, object_type = triplet
        except (ValueError, TypeError):
            continue  # skip the malformed triplet, keep the rest (NVIDIA's original behaviour)
        out.append({
            "subject": subject, "subject_type": subject_type,
            "predicate": relation, "object": obj, "object_type": object_type,
        })
    return out
