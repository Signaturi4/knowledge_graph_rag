# Deterministic DAG Knowledgebase -- FastAPI service.
#
# Rebuilt from the NVIDIA knowledge_graph_rag backend: same FastAPI + NetworkX
# substrate, but the LLM provider is claude_bridge (keyless) and there is no
# vector search / Milvus. See docs/integration-plan-knowledge_graph_rag.md.

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from dotenv import load_dotenv

load_dotenv()

from observability import configure_logging

configure_logging()  # before any other module logs

from service.routers import graph_qa, ingest, logs, memory, plan, review, visualization  # noqa: E402
from service.deps import get_layer  # noqa: E402

logger = logging.getLogger("dagkb")


@asynccontextmanager
async def lifespan(app: FastAPI):
    layer = get_layer()
    gc_thread = None
    if os.environ.get("GC_ENABLED", "1") == "1":
        gc_thread = layer.gc.start(interval_s=float(os.environ.get("GC_INTERVAL_S", "3600")))
        logger.info("garbage collector started")
    yield
    if gc_thread is not None:
        layer.gc.stop()


app = FastAPI(title="Deterministic DAG Knowledgebase", lifespan=lifespan)


@app.middleware("http")
async def _log_requests(request, call_next):
    import time

    t0 = time.monotonic()
    logger.info("HTTP -> %s %s", request.method, request.url.path)
    response = await call_next(request)
    dt = (time.monotonic() - t0) * 1000
    logger.info("HTTP <- %s %s %d (%.0fms)",
               request.method, request.url.path, response.status_code, dt)
    return response


app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(memory.router, prefix="/memory", tags=["memory"])
app.include_router(plan.router, prefix="/plan", tags=["planfence"])
app.include_router(review.router, prefix="/review", tags=["system-2"])
app.include_router(graph_qa.router, prefix="/qa", tags=["qa"])
app.include_router(visualization.router, prefix="/visualization", tags=["visualization"])
app.include_router(logs.router, prefix="/logs", tags=["logs"])


@app.get("/health")
def health():
    layer = get_layer()
    return {
        "status": "ok",
        "provider": os.environ.get("LLM_PROVIDER", "claude_bridge"),
        "semantic_keys": len(layer.heads.all_keys()),
        "nodes": len(layer.dag.nodes()),
        "pending_system_2": len(layer.queue.pending()),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
