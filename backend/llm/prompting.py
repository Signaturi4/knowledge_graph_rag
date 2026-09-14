"""Strict determinism contract for every LLM call site ("agent").

Each agent -- the triplet extractor, the semantic-delta classifier, the query
entity-extractor, the grounded answerer -- prepends :data:`STRICT_CAVEAT` to its
system prompt. This is the "local-command caveat" that keeps behaviour
deterministic and machine-parseable: JSON only, no chain-of-thought, no invention,
and an explicit typed error channel instead of a free-form apology.

Agents are named (see :class:`Agent`) so their calls are attributable in logs and
so a test can assert *which* agent produced a bad payload.
"""

from __future__ import annotations

import enum

STRICT_CAVEAT = (
    "DETERMINISM CONTRACT — follow exactly:\n"
    "1. Output ONLY the single JSON value specified by the task. "
    "No prose, no markdown, no code fences, no preamble, no trailing commentary.\n"
    "2. Do NOT include reasoning, chain-of-thought, or explanations outside the JSON.\n"
    "3. Be deterministic: identical input MUST produce identical output. "
    "Do not sample alternatives, do not add variety.\n"
    "4. Never invent identifiers, dates, numbers, or entities that are not present "
    "in the provided input. Copy values verbatim from the input.\n"
    '5. If you cannot comply with the task, output exactly {"error":"<short reason>"} '
    "and nothing else.\n"
)


class Agent(str, enum.Enum):
    EXTRACTOR = "extractor"          # text -> triplet drafts
    DELTA = "delta_classifier"       # (draft, active) -> delta label
    ENTITY = "entity_extractor"      # query -> entity list
    ANSWERER = "answerer"            # facts + query -> grounded answer


def strict(agent: Agent, task_system: str) -> str:
    """Wrap a task-specific system prompt with the determinism contract."""
    return f"[{agent.value}]\n{STRICT_CAVEAT}\nTASK:\n{task_system}"
