"""
src/app/main.py

FastAPI backend for DevDoc RAG.
Features:
- Lifespan auto-seeding of 35 diverse mock records when rag_logs.db is empty
- /ask endpoint for RAG queries (vector, keyword, or hybrid RRF)
- /feedback endpoint for 👍 / 👎 user ratings
- /metrics/logs and /metrics/summary for feeding the Streamlit dashboard via HTTP
- /health endpoint verifying Qdrant, Ollama, and SQLite connectivity
- Interactive Swagger UI at /docs
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.app.db import (
    DEFAULT_DB_PATH,
    get_analytics_summary,
    get_recent_logs,
    init_db,
    log_feedback,
    seed_mock_logs_if_empty,
)
from src.app.rag import get_rag_pipeline
from src.generation.ollama_client import get_ollama_client
from src.retrieval.search import get_searcher

load_dotenv(find_dotenv())
load_dotenv(".env")  # Fallback: also try root .env for cloud deployments


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan handler (Directive 2):
    Initializes SQLite database and auto-seeds 35 mock records if empty on startup.
    """
    print("Initializing DevDoc RAG Database...")
    init_db(DEFAULT_DB_PATH)
    seed_mock_logs_if_empty(DEFAULT_DB_PATH, count=35)
    yield
    print("DevDoc RAG Application shutting down.")


app = FastAPI(
    title="DevDoc RAG API",
    description="Technical Documentation Assistant API powered by Qdrant Hybrid Search & Local Ollama",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------

class AskRequest(BaseModel):
    query: str = Field(..., description="Developer question or technical query", min_length=2)
    source_lib: Optional[str] = Field(None, description="Optional library filter (e.g. docker, fastapi, pytorch, pydantic)")
    retrieval_method: Optional[str] = Field(None, description="Retrieval strategy: 'vector', 'hybrid_rrf', or 'text' (defaults to RETRIEVAL_METHOD env var)")
    top_k: int = Field(4, ge=1, le=10, description="Number of context chunks to retrieve")
    rewrite_query: bool = Field(False, description="Enable Query Rewriting using LLM (Best Practice)")


class AskResponse(BaseModel):
    log_id: str
    query: str
    rewritten_query: Optional[str] = None
    response: str
    latency_ms: float
    source_lib: str
    retrieval_method: str
    retrieved_chunks: list[dict[str, Any]]


class FeedbackRequest(BaseModel):
    log_id: str = Field(..., description="The query log ID returned by /ask")
    rating: int = Field(..., description="+1 for Thumbs Up, -1 for Thumbs Down")
    comment: Optional[str] = Field(None, description="Optional user comment or reason")


class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get("/", tags=["Info"])
def root_info() -> dict[str, Any]:
    return {
        "service": "DevDoc RAG API",
        "version": "0.1.0",
        "status": "online",
        "documentation": "/docs",
        "endpoints": ["/ask", "/feedback", "/metrics/summary", "/metrics/logs", "/health"],
    }


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, Any]:
    """Health check validating connectivity to Qdrant, Ollama, and SQLite."""
    try:
        ollama_info = get_ollama_client().health_check()
    except Exception as e:
        ollama_info = {"status": "error", "error": str(e)}

    qdrant_status = "disconnected"
    qdrant_points = 0
    try:
        searcher = get_searcher()
        coll_info = searcher.client.get_collection(searcher.collection_name)
        qdrant_status = "connected"
        qdrant_points = coll_info.points_count
    except Exception as e:
        qdrant_status = f"error: {e}"

    sqlite_status = "ok" if os.path.exists(DEFAULT_DB_PATH) else "db_file_pending"

    overall = "healthy"
    if qdrant_status != "connected":
        overall = "degraded"
    if ollama_info.get("status") not in ("connected", "degraded"):
        overall = "degraded"

    return {
        "status": overall,
        "qdrant": {"status": qdrant_status, "points_count": qdrant_points},
        "ollama": ollama_info,
        "sqlite": {"status": sqlite_status, "path": DEFAULT_DB_PATH},
    }


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask_question(req: AskRequest) -> AskResponse:
    """Answers a developer question using the RAG pipeline."""
    try:
        pipeline = get_rag_pipeline()
        result = pipeline.answer(
            query=req.query,
            source_lib=req.source_lib,
            retrieval_method=req.retrieval_method,
            top_k=req.top_k,
            enable_query_rewriting=req.rewrite_query,
            log_to_db=True,
        )
        return AskResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing RAG pipeline: {str(e)}",
        )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Submits user feedback (+1 or -1) for an answer."""
    if req.rating not in (1, -1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rating must be either 1 (positive) or -1 (negative).",
        )
    try:
        fb_id = log_feedback(
            query_id=req.log_id,
            rating=req.rating,
            comment=req.comment,
            db_path=DEFAULT_DB_PATH,
        )
        return FeedbackResponse(feedback_id=fb_id, status="success")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record feedback: {str(e)}",
        )


@app.get("/metrics/summary", tags=["Monitoring"])
def metrics_summary() -> dict[str, Any]:
    """Returns aggregated metrics for the 5 dashboard charts."""
    try:
        return get_analytics_summary(DEFAULT_DB_PATH)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute metrics summary: {str(e)}",
        )


@app.get("/metrics/logs", tags=["Monitoring"])
def metrics_logs(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    """Returns recent query logs for the dashboard."""
    try:
        return get_recent_logs(limit=limit, db_path=DEFAULT_DB_PATH)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch logs: {str(e)}",
        )


def start() -> None:
    """CLI entrypoint for running the API server."""
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("src.app.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    start()
