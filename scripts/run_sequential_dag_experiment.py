"""Sequential Multi-Document DAG Ingestion Experiment.

Runs patient trajectories from test.json through the deterministic memory pipeline:
- Extractor: Local Apple Silicon MLX LoRA model (Meta-Llama-3-8B-Instruct-PMC-LoRA-mlx)
- Delta Classifier: NVIDIA NIM (meta/llama-3.2-11b-vision-instruct) with 40 RPM rate limiting
- Memory Layer: Versioned KnowledgeDAG + Reconciliation Gate + PlanFence
Logs and analyzes node lifecycle transitions and consistency across temporal updates.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

# Ensure backend is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from dag_kb import NodeState, EdgeType, Tier
from dag_kb.store import InMemoryRecordStore
from pipeline.stage0_sources import RawItem, SourcePolicy, SourceRegistry
from service.assembly import build_memory_layer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("sequential_experiment")

# Clinical source policies
CLINICAL_POLICIES = {
    "Primary_Care_Physician": SourcePolicy(Tier.T1, True, "clinical_physician"),
    "General_Surgeon": SourcePolicy(Tier.T1, True, "clinical_specialist"),
    "Nephrologist": SourcePolicy(Tier.T1, True, "clinical_specialist"),
    "Endocrinologist": SourcePolicy(Tier.T1, True, "clinical_specialist"),
    "ED_Attending_Physician": SourcePolicy(Tier.T1, True, "clinical_physician"),
    "Radiologist": SourcePolicy(Tier.T1, True, "clinical_imaging"),
    "Laboratory_System": SourcePolicy(Tier.T1, True, "clinical_lab"),
    "Triage_Nurse": SourcePolicy(Tier.T2, True, "clinical_triage"),
}


def run_experiment(test_json_path: str, output_report_path: str) -> dict:
    with open(test_json_path) as f:
        patients_data = json.load(f)

    log.info("Loaded %d patients from %s", len(patients_data), test_json_path)

    # Initialize providers
    from llm.mlx_provider import MlxLoraProvider
    from llm.nvidia_provider import NvidiaProvider
    from llm.mock import MockProvider

    adapter_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "GTC25_DLI",
        "model",
        "loras",
        "Meta-Llama-3-8B-Instruct-PMC-LoRA-mlx",
    )

    log.info("Initializing MLX LoRA Extractor Provider...")
    mlx_extractor = MlxLoraProvider(adapter_path=adapter_path)

    log.info("Initializing NVIDIA NIM Delta Provider (with 40 RPM limit)...")
    try:
        nvidia_delta = NvidiaProvider(model="meta/llama-3.2-11b-vision-instruct")
        # Quick ping test
        nvidia_delta.complete("Hello", timeout_s=10.0)
        log.info("NVIDIA Delta Provider connected successfully.")
    except Exception as exc:
        log.warning("NVIDIA Delta Provider unavailable (%s), falling back to mock.", exc)
        nvidia_delta = MockProvider(default='{"delta": "SUPERSEDING_CHANGE", "rationale": "clinical update"}')

    full_experiment_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "patients": {},
        "summary": {},
    }

    registry = SourceRegistry(CLINICAL_POLICIES)

    for patient in patients_data:
        pid = patient["patient_id"]
        description = patient.get("description", "")
        docs = patient.get("emr_documents", [])
        log.info("=== Starting Patient %s: %s (%d docs) ===", pid, description, len(docs))

        # Fresh memory layer per patient trajectory
        mem = build_memory_layer(registry=registry)
        # Inject custom extractor and delta classifier into the pipeline
        mem.pipeline._extractor._llm = mlx_extractor
        mem.pipeline._delta._llm = nvidia_delta

        patient_log = {
            "patient_id": pid,
            "description": description,
            "documents": [],
            "final_dag_state": {},
        }

        for doc_idx, doc in enumerate(docs, 1):
            doc_id = doc["doc_id"]
            author_role = doc.get("author_role", "Primary_Care_Physician")
            doc_type = doc.get("document_type", "EMR")
            ts_str = doc.get("timestamp", "2026-01-01T00:00:00Z")
            content = doc.get("content", "")

            valid_from = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            policy = registry.policy(author_role)

            raw = RawItem(
                text=content,
                tier=policy.tier,
                auto_update=policy.auto_update,
                source_id=doc_id,
                tag=doc_type,
                valid_from=valid_from,
                citation=f"{doc_type} by {author_role}",
                canonical_subject=pid,
            )

            t0 = time.time()
            report = mem.pipeline.ingest_raw(raw)
            dt = time.time() - t0

            # Inspect DAG state after this document
            dag = mem.dag
            store = mem.store
            heads = mem.heads

            active_nodes = []
            archived_nodes = []
            flagged_nodes = []
            tbd_nodes = []

            for nid in dag.nx.nodes():
                if not dag.has_node(nid) or dag.is_entity(nid):
                    continue
                st = dag.get_state(nid)
                rec = store.get(nid) if store.has(nid) else None
                summary_item = {
                    "record_id": nid,
                    "semantic_key": dag.nx.nodes[nid].get("semantic_key", ""),
                    "predicate": dag.nx.nodes[nid].get("predicate", ""),
                    "obj": rec.obj if rec else "N/A",
                    "state": st.value,
                    "tier": rec.tier.name if rec else "N/A",
                }
                if st is NodeState.ACTIVE:
                    active_nodes.append(summary_item)
                elif st is NodeState.ARCHIVED:
                    archived_nodes.append(summary_item)
                elif st is NodeState.FLAGGED:
                    flagged_nodes.append(summary_item)
                elif st is NodeState.TBD:
                    tbd_nodes.append(summary_item)

            # Supersession edges
            supersession_edges = []
            for u, v, d in dag.nx.edges(data=True):
                if d.get("kind") == EdgeType.SUPERSEDES.value:
                    supersession_edges.append({"new_record": u, "superseded_record": v})

            doc_entry = {
                "step": doc_idx,
                "doc_id": doc_id,
                "document_type": doc_type,
                "author_role": author_role,
                "timestamp": ts_str,
                "content": content,
                "latency_s": round(dt, 2),
                "drafts_count": report.drafts,
                "dispositions": [
                    {
                        "semantic_key": r.semantic_key,
                        "disposition": r.disposition.value,
                        "detail": r.detail,
                        "committed_record_id": r.committed_record_id,
                    }
                    for r in report.results
                ],
                "dag_snapshot": {
                    "active_count": len(active_nodes),
                    "archived_count": len(archived_nodes),
                    "flagged_count": len(flagged_nodes),
                    "tbd_count": len(tbd_nodes),
                    "active_nodes": active_nodes,
                    "archived_nodes": archived_nodes,
                    "flagged_nodes": flagged_nodes,
                    "supersessions": supersession_edges,
                },
            }
            patient_log["documents"].append(doc_entry)
            log.info(
                "Patient %s Doc %d/%d [%s] -> %d drafts | Active=%d, Archived=%d, Flagged=%d",
                pid,
                doc_idx,
                len(docs),
                doc_id,
                report.drafts,
                len(active_nodes),
                len(archived_nodes),
                len(flagged_nodes),
            )

        # Final state per patient
        patient_log["final_dag_state"] = {
            "total_records": len(list(mem.store)),
            "active_heads": {k: mem.heads.get(k).record_id for k in mem.heads.all_keys()},
            "queue_items": [
                {"id": q.id, "kind": q.kind.value, "key": q.semantic_key, "payload": q.payload}
                for q in mem.queue.pending()
            ],
            "audit_trail_count": len(mem.audit.rows()),
        }
        full_experiment_report["patients"][pid] = patient_log

    with open(output_report_path, "w") as f:
        json.dump(full_experiment_report, f, indent=2)

    log.info("Experiment report written to %s", output_report_path)
    return full_experiment_report


if __name__ == "__main__":
    test_json = os.path.join(os.path.dirname(__file__), "..", "GTC25_DLI", "notebooks", "test.json")
    out_json = os.path.join(os.path.dirname(__file__), "..", "backend", "data", "sequential_update_report.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    run_experiment(test_json, out_json)
