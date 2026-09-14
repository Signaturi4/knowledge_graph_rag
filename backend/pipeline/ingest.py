"""Phase 9 -- ingestion orchestrator (Path A, stages 1 -> 7).

    RawItem --extract--> TripletDraft[]
             --resolve--> (draft, N_active)
             --delta-----> Delta            (only when N_active exists)
             --gate------> GateResult        (deterministic disposition)

LLM failure at extract/delta => the affected draft is routed to SYSTEM_2 via a
DEAD_LETTER item; it is never auto-committed (technical-plan R9.9).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dag_kb.ids import new_txn_id
from dag_kb import Delta, Disposition, GateResult, QueueItem, QueueItemType, Routing

from .stage3_delta import DeltaClassifier
from .stage1_extract import Extractor
from .stage2_resolve import Resolver
from .stage0_sources import RawItem, SourceRegistry

if TYPE_CHECKING:
    from dag_kb import ReconciliationGate
    from llm.provider import LLMProvider

log = logging.getLogger("dagkb.pipeline.orchestrator")


@dataclass(slots=True)
class IngestReport:
    txn_id: str
    source_id: str
    drafts: int
    results: list[GateResult]

    def summary(self) -> dict:
        by = {}
        for r in self.results:
            by[r.disposition.value] = by.get(r.disposition.value, 0) + 1
        return {"txn_id": self.txn_id, "source_id": self.source_id,
                "drafts": self.drafts, "by_disposition": by}


class IngestionPipeline:
    def __init__(
        self,
        provider: "LLMProvider",
        resolver: Resolver,
        gate: "ReconciliationGate",
        *,
        extractor_provider: "LLMProvider | None" = None,
        delta_provider: "LLMProvider | None" = None,
        registry: SourceRegistry | None = None,
        default_owner: str = "ingest",
    ) -> None:
        self._extractor = Extractor(extractor_provider or provider)
        self._delta = DeltaClassifier(delta_provider or provider)
        self._resolver = resolver
        self._gate = gate
        self._registry = registry or SourceRegistry()
        self._owner = default_owner

    # -- entry points --------------------------------------------------- #
    def ingest_text(self, text: str, source_id: str, *, owner: str | None = None,
                    canonical_subject: str | None = None) -> IngestReport:
        return self._run(self._registry.from_text(text, source_id,
                                                   canonical_subject=canonical_subject), owner)

    def ingest_pdf(self, path: str, source_id: str, *, owner: str | None = None) -> IngestReport:
        return self._run(self._registry.from_pdf(path, source_id), owner)

    def ingest_raw(self, raw: RawItem, *, owner: str | None = None) -> IngestReport:
        return self._run(raw, owner)

    # -- core loop ---------------------------------------------------- #
    def _run(self, raw: RawItem, owner: str | None) -> IngestReport:
        from llm.provider import LLMUnavailable

        writer = owner or self._owner
        txn_id = new_txn_id()
        log.info("INGEST start | txn=%s source_id=%s writer=%s", txn_id, raw.source_id, writer)

        try:
            drafts = self._extractor.extract(raw)
        except LLMUnavailable as exc:
            item = self._gate.queue.enqueue(QueueItem(
                kind=QueueItemType.DEAD_LETTER, routed_to=Routing.SYSTEM_2,
                semantic_key=f"<extract:{raw.source_id}>",
                payload={"reason": f"extractor unavailable: {exc}", "source_id": raw.source_id},
            ))
            log.warning("INGEST aborted at STAGE1 | txn=%s source_id=%s reason=%s",
                       txn_id, raw.source_id, exc)
            return IngestReport(txn_id, raw.source_id, 0, [
                GateResult(Disposition.REJECT_FAIL_CLOSED, f"<extract:{raw.source_id}>",
                           queue_item=item, detail=str(exc))
            ])

        results: list[GateResult] = []
        for draft in drafts:
            draft, active = self._resolver.resolve(draft)
            if active is None:
                r = self._gate.submit(draft, Delta.SUPERSEDING_CHANGE,
                                      writer_identity=writer, txn_id=txn_id)
                log.info("INGEST gate | txn=%s key=%s -> %s (%s)",
                        txn_id, draft.semantic_key, r.disposition.value, r.detail)
                results.append(r)
                continue
            try:
                label, rationale = self._delta.classify(draft, active)
                draft.provenance["llm_assessment"] = f"{label.value}: {rationale}"
            except LLMUnavailable as exc:
                item = self._gate.queue.enqueue(QueueItem(
                    kind=QueueItemType.DEAD_LETTER, routed_to=Routing.SYSTEM_2,
                    semantic_key=draft.semantic_key,
                    payload={"reason": f"delta classifier unavailable: {exc}"},
                ))
                log.warning("INGEST aborted at STAGE3 | txn=%s key=%s reason=%s",
                           txn_id, draft.semantic_key, exc)
                results.append(GateResult(Disposition.REJECT_FAIL_CLOSED, draft.semantic_key,
                                          queue_item=item, detail=str(exc)))
                continue
            r = self._gate.submit(draft, label, writer_identity=writer, txn_id=txn_id)
            log.info("INGEST gate | txn=%s key=%s -> %s (%s)",
                    txn_id, draft.semantic_key, r.disposition.value, r.detail)
            results.append(r)

        by = {}
        for r in results:
            by[r.disposition.value] = by.get(r.disposition.value, 0) + 1
        log.info("INGEST done | txn=%s source_id=%s drafts=%d dispositions=%s",
                txn_id, raw.source_id, len(drafts), by)
        return IngestReport(txn_id, raw.source_id, len(drafts), results)
