"""Unified retrieval engine for DevDoc RAG: Dense Vector, BM25S Keyword, and Hybrid RRF."""

from __future__ import annotations

import os
import re
import threading
from collections import defaultdict
from typing import Any

import bm25s
from bm25s.tokenization import Tokenizer
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

# Matches the old regex exactly: keeps hyphenated/underscored identifiers
# (source_lib, scikit-learn, get_searcher, ...) intact instead of splitting them,
# which matters a lot for a technical-docs corpus.
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_\-]{2,}")


def _splitter(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# How many times to repeat section_title text when building the indexed text,
# to approximate field boosting (BM25S has no native per-field weighting).
_TITLE_BOOST_REPEATS = 3

# How many points to pull per Qdrant scroll page when building the BM25S index.
_SCROLL_PAGE_SIZE = 512


class DevDocSearcher:
    """Unified search interface supporting vector, keyword (BM25S), and hybrid RRF retrieval."""

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

        # BM25S state, built lazily from the Qdrant collection.
        self._bm25_retriever: bm25s.BM25 | None = None
        self._bm25_docs: list[dict[str, Any]] | None = None
        self._bm25_tokenizer: Tokenizer | None = None
        self._bm25_lock = threading.Lock()

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

    # ------------------------------------------------------------------
    # BM25S-backed text search
    # ------------------------------------------------------------------

    def _fetch_all_points(self) -> list[dict[str, Any]]:
        """Page through the entire Qdrant collection and return payload dicts."""
        docs: list[dict[str, Any]] = []
        offset = None

        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=_SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for record in records:
                payload = record.payload or {}
                docs.append(
                    {
                        "chunk_id": payload.get("chunk_id", str(record.id)),
                        "source_lib": (payload.get("source_lib") or "").lower().strip(),
                        "file_path": payload.get("file_path", ""),
                        "section_title": payload.get("section_title", ""),
                        "chunk_index": payload.get("chunk_index", 0),
                        "content": payload.get("content", ""),
                        "char_count": payload.get("char_count", len(payload.get("content", ""))),
                    }
                )

            if offset is None:
                break

        return docs

    @staticmethod
    def _doc_to_indexed_text(doc: dict[str, Any]) -> str:
        """Concatenate fields into one text blob, repeating the title to boost it."""
        title = doc.get("section_title", "") or ""
        content = doc.get("content", "") or ""
        boosted_title = (title + " ") * _TITLE_BOOST_REPEATS
        return f"{boosted_title}{content}"

    def _build_bm25_index(self) -> tuple[bm25s.BM25, list[dict[str, Any]], Tokenizer]:
        docs = self._fetch_all_points()
        tokenizer = Tokenizer(stemmer=None, stopwords=list(STOP_WORDS), splitter=_splitter)

        retriever = bm25s.BM25(corpus=docs)
        if docs:
            texts = [self._doc_to_indexed_text(d) for d in docs]
            corpus_tokens = tokenizer.tokenize(texts)
            retriever.index(corpus_tokens)

        return retriever, docs, tokenizer

    def _ensure_bm25_index(self) -> None:
        if self._bm25_retriever is None:
            with self._bm25_lock:
                if self._bm25_retriever is None:
                    self._bm25_retriever, self._bm25_docs, self._bm25_tokenizer = (
                        self._build_bm25_index()
                    )

    def rebuild_text_index(self) -> None:
        """Force a rebuild of the BM25S index from the current Qdrant collection.

        Call this after ingesting new documents into Qdrant, since the cached
        BM25S index otherwise won't know about them.
        """
        with self._bm25_lock:
            self._bm25_retriever, self._bm25_docs, self._bm25_tokenizer = (
                self._build_bm25_index()
            )

    def text_search(
        self,
        query: str,
        k: int = 5,
        source_lib: str | None = None,
    ) -> list[dict[str, Any]]:
        """Lexical search backed by bm25s (real Okapi BM25, whole-corpus ranked)."""
        self._ensure_bm25_index()

        if not self._bm25_docs:
            return []

        # Rank against the FULL corpus first, then filter by source_lib.
        # This is the key difference from the old implementation: nothing
        # gets truncated or dropped before scoring happens.
        pool_size = len(self._bm25_docs)
        if source_lib:
            k_pool = min(pool_size, max(k * 20, 200))
        else:
            k_pool = min(pool_size, k)

        query_tokens = self._bm25_tokenizer.tokenize([query], update_vocab=False)
        doc_batches, score_batches = self._bm25_retriever.retrieve(
            query_tokens, corpus=self._bm25_docs, k=k_pool
        )

        docs = doc_batches[0]
        scores = score_batches[0]

        normalized_source_lib = source_lib.lower().strip() if source_lib else None

        results: list[dict[str, Any]] = []
        for doc, score in zip(docs, scores):
            if normalized_source_lib and doc.get("source_lib") != normalized_source_lib:
                continue

            content = doc.get("content", "")
            results.append(
                {
                    "chunk_id": doc.get("chunk_id", ""),
                    "score": float(score),
                    "source_lib": doc.get("source_lib", ""),
                    "file_path": doc.get("file_path", ""),
                    "section_title": doc.get("section_title", ""),
                    "chunk_index": doc.get("chunk_index", 0),
                    "content": content,
                    "char_count": doc.get("char_count", len(content)),
                    "retrieval_method": "text",
                }
            )
            if len(results) >= k:
                break

        return results

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        source_lib: str | None = None,
        candidate_pool: int = 30,
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
    candidate_pool: int = 30,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    return get_searcher().hybrid_search(
        query=query,
        k=k,
        source_lib=source_lib,
        candidate_pool=candidate_pool,
        rrf_k=rrf_k,
    )