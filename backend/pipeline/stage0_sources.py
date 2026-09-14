"""Phase 9, stage 0 -- raw input + authority-tier tagging.

Each source is registered once with its tier, its ``auto_update`` default, and a
tag. Extraction/ingest never guess these -- they come from application config
(technical-plan section 6c).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime

from dag_kb import Tier, utcnow
from observability import truncate

log = logging.getLogger("dagkb.pipeline.stage0_sources")


@dataclass(slots=True)
class RawItem:
    text: str
    tier: Tier
    auto_update: bool
    source_id: str
    tag: str = ""
    valid_from: datetime = field(default_factory=utcnow)
    citation: str = ""
    # Optional stable entity name for this document (e.g. "patient_42"). When set,
    # stage1 extraction is told to use this exact string as the subject whenever a
    # fact is about "the" entity the document describes -- the fix for narrative
    # text where the entity is only ever referred to by pronoun/description
    # ("the patient", "he", "the subject"), never by a stable name.
    canonical_subject: str | None = None


@dataclass(frozen=True, slots=True)
class SourcePolicy:
    tier: Tier
    auto_update: bool
    tag: str


# Sensible defaults; override per deployment.
DEFAULT_POLICIES: dict[str, SourcePolicy] = {
    "statute":       SourcePolicy(Tier.T0, False, "canonical"),
    "contract":      SourcePolicy(Tier.T0, False, "canonical"),
    "municipal":     SourcePolicy(Tier.T0, False, "canonical"),
    "policy":        SourcePolicy(Tier.T1, False, "internal"),
    "sop":           SourcePolicy(Tier.T1, True,  "internal"),
    "vendor_doc":    SourcePolicy(Tier.T2, True,  "external"),
    "arxiv":         SourcePolicy(Tier.T2, True,  "external"),
    "derived":       SourcePolicy(Tier.T3, True,  "derived"),
    "conversation":  SourcePolicy(Tier.T4, True,  "observational"),
    "email":         SourcePolicy(Tier.T4, True,  "observational"),
}


class SourceRegistry:
    def __init__(self, policies: dict[str, SourcePolicy] | None = None) -> None:
        self._policies = dict(DEFAULT_POLICIES)
        if policies:
            self._policies.update(policies)

    def register(self, source_id: str, policy: SourcePolicy) -> None:
        self._policies[source_id] = policy

    def policy(self, source_id: str) -> SourcePolicy:
        if source_id in self._policies:
            return self._policies[source_id]
        prefix = source_id.split(":", 1)[0]
        return self._policies.get(prefix, SourcePolicy(Tier.T4, True, "observational"))

    # -- builders ---------------------------------------------------------- #
    def from_text(self, text: str, source_id: str, *, citation: str = "",
                 canonical_subject: str | None = None) -> RawItem:
        p = self.policy(source_id)
        log.info("STAGE0 raw input | source_id=%s tier=%s auto_update=%s chars=%d canonical_subject=%s text=%s",
                 source_id, p.tier.name, p.auto_update, len(text), canonical_subject, truncate(text))
        return RawItem(text=text, tier=p.tier, auto_update=p.auto_update,
                       source_id=source_id, tag=p.tag, citation=citation or source_id,
                       canonical_subject=canonical_subject)

    def from_pdf(self, path: str, source_id: str) -> RawItem:
        from pypdf import PdfReader

        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return self.from_text(text, source_id, citation=os.path.basename(path))

    def from_conversation(self, turns: list[dict], source_id: str = "conversation") -> RawItem:
        text = "\n".join(f"{t.get('role', '?')}: {t.get('content', '')}" for t in turns)
        return self.from_text(text, source_id, citation="agent conversation")
