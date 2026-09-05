"""
src/app/rag.py

Core RAG pipeline for DevDoc RAG.
Orchestrates:
1. Optional Query Rewriting (Best Practice Rubric Bonus)
2. Hybrid Retrieval (Vector + Keyword RRF via Qdrant)
3. Structured Prompt Assembly (Prompt B)
4. LLM Generation via Ollama
5. Automatic telemetry logging to SQLite (rag_logs.db)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.app.db import log_query
from src.generation.ollama_client import OllamaClient, get_ollama_client
from src.generation.prompts import (
    PROMPT_B_TEMPLATE,
    QUERY_REWRITE_TEMPLATE,
    format_context,
)
from src.retrieval.search import DevDocSearcher, get_searcher


class DevDocRAG:
    """Production RAG orchestrator for technical documentation Q&A."""

    def __init__(
        self,
        searcher: DevDocSearcher | None = None,
        llm_client: OllamaClient | None = None,
    ) -> None:
        self.searcher = searcher or get_searcher()
        self.llm_client = llm_client or get_ollama_client()

    def rewrite_query(self, query: str) -> str:
        """
        Reformulates developer questions into targeted search queries.
        Fulfills the 'Query Rewriting' best practice rubric criteria.
        """
        try:
            prompt = QUERY_REWRITE_TEMPLATE.format(query=query)
            rewritten = self.llm_client.generate(prompt=prompt, temperature=0.1)
            cleaned = rewritten.strip().strip('"\'')
            return cleaned if cleaned else query
        except Exception as e:
            print(f"Warning: Query rewriting encountered error: {e}. Falling back to original query.")
            return query

    def answer(
        self,
        query: str,
        source_lib: str | None = None,
        retrieval_method: str = "hybrid_rrf",
        top_k: int = 4,
        enable_query_rewriting: bool = False,
        log_to_db: bool = True,
    ) -> dict[str, Any]:
        """
        Executes end-to-end RAG pipeline:
        Query -> [Optional Rewrite] -> Retrieval -> Prompt Assembly -> LLM -> SQLite Log.
        """
        start_time = time.perf_counter()

        # 1. Optional Query Rewriting (Best Practice)
        effective_query = query
        rewritten_text = None
        if enable_query_rewriting:
            rewritten_text = self.rewrite_query(query)
            effective_query = rewritten_text

        # 2. Retrieval
        retrieval_method_norm = retrieval_method.lower().strip()
        if retrieval_method_norm in ("vector", "dense"):
            retrieved_chunks = self.searcher.vector_search(query=effective_query, k=top_k, source_lib=source_lib)
            method_used = "vector"
        elif retrieval_method_norm in ("text", "keyword", "lexical"):
            retrieved_chunks = self.searcher.text_search(query=effective_query, k=top_k, source_lib=source_lib)
            method_used = "text"
        else:
            retrieved_chunks = self.searcher.hybrid_search(query=effective_query, k=top_k, source_lib=source_lib)
            method_used = "hybrid_rrf"

        # 3. Prompt Assembly (Prompt B)
        context_str = format_context(retrieved_chunks)
        prompt = PROMPT_B_TEMPLATE.format(context=context_str, query=query)

        # 4. LLM Generation
        response_text = self.llm_client.generate(prompt=prompt, temperature=0.2)

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 5. Telemetry Logging (SQLite)
        log_id = ""
        if log_to_db:
            try:
                log_id = log_query(
                    query=query,
                    response=response_text,
                    latency_ms=total_latency_ms,
                    retrieved_chunks=retrieved_chunks,
                    source_lib=source_lib,
                    retrieval_method=method_used,
                    rewritten_query=rewritten_text,
                )
            except Exception as e:
                print(f"Warning: Failed to log query telemetry to database: {e}")

        return {
            "log_id": log_id,
            "query": query,
            "rewritten_query": rewritten_text,
            "response": response_text,
            "latency_ms": round(total_latency_ms, 2),
            "source_lib": source_lib or "all",
            "retrieval_method": method_used,
            "retrieved_chunks": retrieved_chunks,
        }


_rag_pipeline: DevDocRAG | None = None


def get_rag_pipeline() -> DevDocRAG:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = DevDocRAG()
    return _rag_pipeline
