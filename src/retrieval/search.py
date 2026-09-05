"""Unified retrieval engine for DevDoc RAG: Dense Vector, BM25 Keyword, and Hybrid RRF."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

from dotenv import find_dotenv, load_dotenv
from qdrant_client import QdrantClient, models

from src.embeddings.embedder import ONNXEmbedder, get_embedder

load_dotenv(find_dotenv())

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


class DevDocSearcher:
    """Unified search interface supporting vector, keyword, and hybrid RRF retrieval."""

    def __init__(
        self,
        collection_name: str = "devdoc",
        embedder: Any | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.embedder = embedder or get_embedder()

        if client is not None:
            self.client = client
        else:
            qdrant_url = os.getenv("QDRANT_API_URL") or os.getenv("Qdrant_API_URL")
            qdrant_key = os.getenv("QDRANT_API_KEY") or os.getenv("Qdrant_API_KEY")
            storage_path = os.getenv("QDRANT_STORAGE_PATH", "src/data/qdrant_storage")

            if qdrant_url:
                self.client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
            else:
                self.client = QdrantClient(path=storage_path)

    def _build_filter(self, source_lib: str | None = None) -> models.Filter | None:
        """Build optional payload filter for source library."""
        if not source_lib:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="source_lib",
                    match=models.MatchValue(value=source_lib.lower().strip()),
                )
            ]
        )

    def vector_search(
        self,
        query: str,
        k: int = 5,
        source_lib: str | None = None,
    ) -> list[dict[str, Any]]:
        """Dense semantic search using fastembed ONNX embeddings and Cosine distance."""
        query_vector = self.embedder.embed_query(query)
        filter_condition = self._build_filter(source_lib)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=filter_condition,
            limit=k,
            with_payload=True,
        )

        results: list[dict[str, Any]] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                {
                    "chunk_id": payload.get("chunk_id", str(point.id)),
                    "score": float(point.score),
                    "source_lib": payload.get("source_lib", ""),
                    "file_path": payload.get("file_path", ""),
                    "section_title": payload.get("section_title", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "content": payload.get("content", ""),
                    "char_count": payload.get("char_count", len(payload.get("content", ""))),
                    "retrieval_method": "vector",
                }
            )
        return results

    def text_search(
        self,
        query: str,
        k: int = 5,
        source_lib: str | None = None,
    ) -> list[dict[str, Any]]:
        """Lexical full-text search leveraging Qdrant payload text index and token matching."""
        raw_tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", query)]
        meaningful_tokens = [w for w in raw_tokens if w not in STOP_WORDS]
        tokens = meaningful_tokens if meaningful_tokens else raw_tokens
        if not tokens:
            return []

        must_conditions: list[models.Condition] = []
        if source_lib:
            must_conditions.append(
                models.FieldCondition(
                    key="source_lib",
                    match=models.MatchValue(value=source_lib.lower().strip()),
                )
            )

        primary_conditions = list(must_conditions) + [
            models.FieldCondition(
                key="content",
                match=models.MatchText(text=query),
            )
        ]
        matched_records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(must=primary_conditions),
            limit=max(k * 3, 25),
            with_payload=True,
        )

        if not matched_records:
            should_conditions = [
                models.FieldCondition(
                    key="content",
                    match=models.MatchText(text=token),
                )
                for token in tokens
            ]
            matched_records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=must_conditions,
                    should=should_conditions,
                ),
                limit=max(k * 4, 50),
                with_payload=True,
            )

        scored_records: list[tuple[float, Any]] = []
        for record in matched_records:
            payload = record.payload or {}
            content = payload.get("content", "").lower()
            term_hits = sum(content.count(token) for token in tokens)
            unique_hits = sum(1 for token in tokens if token in content)
            lexical_score = (unique_hits * 10.0) + min(term_hits, 20)
            scored_records.append((lexical_score, record))

        scored_records.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, record in scored_records[:k]:
            payload = record.payload or {}
            results.append(
                {
                    "chunk_id": payload.get("chunk_id", str(record.id)),
                    "score": float(score),
                    "source_lib": payload.get("source_lib", ""),
                    "file_path": payload.get("file_path", ""),
                    "section_title": payload.get("section_title", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "content": payload.get("content", ""),
                    "char_count": payload.get("char_count", len(payload.get("content", ""))),
                    "retrieval_method": "text",
                }
            )
        return results

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        source_lib: str | None = None,
        candidate_pool: int = 20,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Combines Vector and Text retrieval with Reciprocal Rank Fusion (RRF)."""
        vector_candidates = self.vector_search(query=query, k=candidate_pool, source_lib=source_lib)
        text_candidates = self.text_search(query=query, k=candidate_pool, source_lib=source_lib)

        rrf_scores: dict[str, float] = defaultdict(float)
        candidate_docs: dict[str, dict[str, Any]] = {}
        vector_ranks: dict[str, int] = {}
        text_ranks: dict[str, int] = {}

        for rank, item in enumerate(vector_candidates):
            cid = item["chunk_id"]
            rrf_scores[cid] += 1.0 / (rrf_k + rank + 1)
            candidate_docs[cid] = item
            vector_ranks[cid] = rank + 1

        for rank, item in enumerate(text_candidates):
            cid = item["chunk_id"]
            rrf_scores[cid] += 1.0 / (rrf_k + rank + 1)
            if cid not in candidate_docs:
                candidate_docs[cid] = item
            text_ranks[cid] = rank + 1

        sorted_candidates = sorted(rrf_scores.items(), key=lambda entry: entry[1], reverse=True)

        results: list[dict[str, Any]] = []
        for cid, score in sorted_candidates[:k]:
            doc = dict(candidate_docs[cid])
            doc["score"] = round(score, 6)
            doc["rrf_vector_rank"] = vector_ranks.get(cid, None)
            doc["rrf_text_rank"] = text_ranks.get(cid, None)
            doc["retrieval_method"] = "hybrid_rrf"
            results.append(doc)

        return results


_searcher: DevDocSearcher | None = None


def get_searcher() -> DevDocSearcher:
    global _searcher
    if _searcher is None:
        _searcher = DevDocSearcher()
    return _searcher


def vector_search(query: str, k: int = 5, source_lib: str | None = None) -> list[dict[str, Any]]:
    return get_searcher().vector_search(query=query, k=k, source_lib=source_lib)


def text_search(query: str, k: int = 5, source_lib: str | None = None) -> list[dict[str, Any]]:
    return get_searcher().text_search(query=query, k=k, source_lib=source_lib)


def hybrid_search(
    query: str,
    k: int = 5,
    source_lib: str | None = None,
    candidate_pool: int = 20,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    return get_searcher().hybrid_search(
        query=query,
        k=k,
        source_lib=source_lib,
        candidate_pool=candidate_pool,
        rrf_k=rrf_k,
    )

