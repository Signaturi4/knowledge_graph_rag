"""Per-stage guardrails for the ingestion pipeline.

Every stage that consumes LLM output validates the *shape* of that output before
it is allowed to influence the graph. A validation failure is raised as
``LLMUnavailable`` so the pipeline routes the item to SYSTEM_2 instead of
committing on garbage (technical-plan R9.9).

Hand-rolled (no jsonschema dependency) but explicit and total.
"""

from __future__ import annotations

from typing import Any

from dag_kb import Delta, MalformedDraft, TripletDraft

from llm.provider import LLMUnavailable

_VALID_DELTAS = {d.value for d in Delta}
_MAX_STR = 512


def _predicate_is_value_laden(pred: str) -> bool:
    """Catches the extraction failure mode seen live: the LLM folds the object's
    value / a date / 'effective as of' phrasing into the predicate string instead
    of putting it in `object` (e.g. "sets_max_height_to_45_stories_effective_..."),
    which makes the predicate unique per-call and breaks supersession detection."""
    words = pred.split()
    if len(words) > 6:
        return True
    if any(ch.isdigit() for ch in pred) and len(words) > 3:
        return True
    if any(w in pred.lower() for w in ("effective", "as of", "as_of")):
        return True
    return False


def _clean_str(v: Any, field: str) -> str:
    if not isinstance(v, str):
        raise LLMUnavailable(f"{field!r} is not a string: {type(v).__name__}")
    s = v.strip()
    if not s:
        raise LLMUnavailable(f"{field!r} is empty")
    if len(s) > _MAX_STR:
        raise LLMUnavailable(f"{field!r} exceeds {_MAX_STR} chars")
    if s.lower() in {"nan", "none", "null", "n/a", "unknown"}:
        raise LLMUnavailable(f"{field!r} is a null-ish placeholder: {s!r}")
    return s


# --------------------------------------------------------------------------- #
# stage 1 -- extraction payload
# --------------------------------------------------------------------------- #
def validate_extraction_payload(payload: Any) -> list[dict]:
    """Return a list of clean ``{subject, predicate, object, subject_type,
    object_type}`` dicts. Raise if the payload is not a recognizable triplet
    container or if it carries an ``{"error": ...}`` channel."""
    if isinstance(payload, dict):
        if "error" in payload:
            raise LLMUnavailable(f"extractor error channel: {payload['error']}")
        rows = payload.get("triplets")
        if rows is None:
            raise LLMUnavailable("extraction payload has no 'triplets' key")
    elif isinstance(payload, list):
        rows = payload
    else:
        raise LLMUnavailable(f"extraction payload is {type(payload).__name__}, not object/array")

    if not isinstance(rows, list):
        raise LLMUnavailable("'triplets' is not a list")

    clean: list[dict] = []
    for i, r in enumerate(rows):
        try:
            if isinstance(r, dict):
                subj = _clean_str(r.get("subject"), f"triplets[{i}].subject")
                pred = _clean_str(r.get("predicate", r.get("relation")), f"triplets[{i}].predicate")
                obj = r.get("object")
                obj = _clean_str(obj, f"triplets[{i}].object") if isinstance(obj, str) else obj
                if obj is None or obj == "":
                    raise LLMUnavailable(f"triplets[{i}].object is empty")
                if _predicate_is_value_laden(pred):
                    raise LLMUnavailable(
                        f"triplets[{i}].predicate {pred!r} looks value-laden "
                        "(value/date folded in instead of using `object`)"
                    )
                clean.append({
                    "subject": subj, "predicate": pred, "object": obj,
                    "subject_type": str(r.get("subject_type", "")),
                    "object_type": str(r.get("object_type", "")),
                })
            elif isinstance(r, (list, tuple)) and len(r) >= 3:
                if len(r) >= 5:
                    subj, st, pred, obj, ot = r[0], r[1], r[2], r[3], r[4]
                else:
                    subj, pred, obj = r[0], r[1], r[2]
                    st = ot = ""
                pred = _clean_str(pred, f"triplets[{i}][2]")
                if _predicate_is_value_laden(pred):
                    raise LLMUnavailable(f"triplets[{i}] predicate {pred!r} looks value-laden")
                clean.append({
                    "subject": _clean_str(subj, f"triplets[{i}][0]"),
                    "predicate": pred,
                    "object": _clean_str(obj, f"triplets[{i}][3]") if isinstance(obj, str) else obj,
                    "subject_type": str(st), "object_type": str(ot),
                })
            else:
                raise LLMUnavailable(f"triplets[{i}] is not a dict or 3+-tuple")
        except LLMUnavailable:
            # one bad row does not sink the batch; skip it (logged by caller)
            continue
    if not clean:
        raise LLMUnavailable("extraction produced no valid triplets")
    return clean


# --------------------------------------------------------------------------- #
# stage 2 -- draft invariants (also enforced deterministically by the gate)
# --------------------------------------------------------------------------- #
def validate_triplet_draft(d: TripletDraft) -> None:
    if not d.required_fields_present():
        raise MalformedDraft(f"draft missing required fields: {d.semantic_key!r}")
    if "::" not in d.semantic_key:
        raise MalformedDraft(f"semantic_key is not 'entity::attribute': {d.semantic_key!r}")
    if d.record_type not in ("asserted", "derived"):
        raise MalformedDraft(f"bad record_type {d.record_type!r}")
    if d.record_type == "derived" and not d.parent_ids:
        raise MalformedDraft("derived draft has no parent_ids")


# --------------------------------------------------------------------------- #
# stage 3 -- delta label
# --------------------------------------------------------------------------- #
def validate_delta_payload(payload: Any) -> tuple[Delta, str]:
    if not isinstance(payload, dict):
        raise LLMUnavailable(f"delta payload is {type(payload).__name__}, not object")
    if "error" in payload:
        raise LLMUnavailable(f"delta error channel: {payload['error']}")
    label = str(payload.get("delta", "")).upper()
    if label not in _VALID_DELTAS:
        raise LLMUnavailable(f"delta label not in {_VALID_DELTAS}: {label!r}")
    return Delta(label), str(payload.get("rationale", ""))[:_MAX_STR]


# --------------------------------------------------------------------------- #
# query-time -- entity list
# --------------------------------------------------------------------------- #
def validate_entities_payload(payload: Any, query: str) -> list[str]:
    if isinstance(payload, dict) and "error" in payload:
        raise LLMUnavailable(f"entity extractor error channel: {payload['error']}")
    ents = payload.get("entities") if isinstance(payload, dict) else payload
    if not isinstance(ents, list):
        raise LLMUnavailable("entities payload is not a list")
    out: list[str] = []
    for e in ents:
        if isinstance(e, str) and e.strip() and e.strip().lower() in query.lower():
            out.append(e.strip())
    return out
