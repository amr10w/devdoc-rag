"""
src/embeddings/embedder.py

Unified text embedder supporting:
1. OllamaEmbedder: For models like qwen3-embedding (aligned with ingested Qdrant vectors)
2. ONNXEmbedder: High-performance, lightweight FastEmbed ONNX engine
"""

from __future__ import annotations

import os
from typing import Any, Iterable, List, Optional, Union

from dotenv import find_dotenv, load_dotenv
import numpy as np
import ollama
from fastembed import TextEmbedding

load_dotenv(find_dotenv())

DEFAULT_ONNX_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:latest")
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

DEFAULT_VECTOR_DIMENSION = int(
    os.getenv("VECTOR_DIMENSION",384)
)

MODEL_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "intfloat/multilingual-e5-large": 1024,
    "qwen3-embedding:latest": 384,
}


class OllamaEmbedder:
    """
    Ollama-powered embedding engine.
    Produces vector embeddings with dimensions configurable via environment variable or argument.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_EMBED_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        dimensions: int | None = None,
    ) -> None:
        self.model_name = os.getenv("OLLAMA_EMBED_MODEL", model_name)
        raw_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_URL") or base_url
        if "ollama.com" in raw_url:
            raw_url = "http://localhost:11434"
        self.base_url = raw_url.rstrip("/")
        self.dimensions = dimensions if dimensions is not None else DEFAULT_VECTOR_DIMENSION
        self.client = ollama.Client(host=self.base_url)
        self._fallback_embedder: ONNXEmbedder | None = None

    @property
    def dimension(self) -> int:
        return self.dimensions

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query string using Ollama with ONNX fallback."""
        if not text or not text.strip():
            return [0.0] * self.dimension
        try:
            res = self.client.embed(
                model=self.model_name,
                input=text,
                truncate=True,
                dimensions=self.dimensions,
            )
            return res["embeddings"][0]
        except Exception:
            if self._fallback_embedder is None:
                self._fallback_embedder = ONNXEmbedder()
            return self._fallback_embedder.embed_query(text)

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embeds a batch of documents using Ollama with ONNX fallback."""
        if not texts:
            return []
        try:
            res = self.client.embed(
                model=self.model_name,
                input=texts,
                truncate=True,
                dimensions=self.dimensions,
            )
            return res["embeddings"]
        except Exception:
            if self._fallback_embedder is None:
                self._fallback_embedder = ONNXEmbedder()
            return self._fallback_embedder.embed_documents(texts, batch_size=batch_size)


class ONNXEmbedder:
    """
    ONNX-powered embedding engine using FastEmbed.
    Provides PyTorch-free fast inference.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ONNX_MODEL,
        max_length: int = 512,
        threads: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.threads = threads
        self.cache_dir = cache_dir
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(
                model_name=self.model_name,
                max_length=self.max_length,
                threads=self.threads,
                cache_dir=self.cache_dir,
            )
        return self._model

    @property
    def dimension(self) -> int:
        if self.model_name in MODEL_DIMENSIONS:
            return MODEL_DIMENSIONS[self.model_name]
        test_vec = self.embed_query("test")
        return len(test_vec)

    def embed_documents(
        self,
        texts: Union[List[str], Iterable[str]],
        batch_size: int = 128,
        parallel: int | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        embeddings_generator = self.model.embed(
            documents=texts,
            batch_size=batch_size,
            parallel=parallel,
        )
        return [vec.tolist() if isinstance(vec, np.ndarray) else list(vec) for vec in embeddings_generator]

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self.dimension
        query_gen = self.model.query_embed(query=text)
        first_vec = next(query_gen)
        return first_vec.tolist() if isinstance(first_vec, np.ndarray) else list(first_vec)


def get_embedder(provider: str | None = None) -> Union[ONNXEmbedder, OllamaEmbedder]:
    """
    Factory helper returning appropriate embedder based on provider or EMBEDDING_PROVIDER env var.
    Defaults to ONNXEmbedder (FastEmbed BAAI/bge-small-en-v1.5) for fast in-process embedding.
    """
    prov = provider or os.getenv("EMBEDDING_PROVIDER", "onnx")
    if prov.lower() == "ollama":
        try:
            return OllamaEmbedder()
        except Exception:
            return ONNXEmbedder()
    return ONNXEmbedder()
