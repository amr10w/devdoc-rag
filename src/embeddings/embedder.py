"""
src/embeddings/embedder.py

High-performance, lightweight ONNX-powered text embedder using FastEmbed.
Designed for both offline document chunk ingestion and low-latency online RAG query embedding.
"""

from typing import Iterable, List, Optional, Union
import numpy as np
from fastembed import TextEmbedding


DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Dimension lookup for common FastEmbed models
MODEL_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "intfloat/multilingual-e5-large": 1024,
}


class ONNXEmbedder:
    """
    ONNX-powered embedding engine.
    Wraps FastEmbed's TextEmbedding to provide fast, PyTorch-free dense vector generation.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 512,
        threads: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.threads = threads
        self.cache_dir = cache_dir
        self._model: Optional[TextEmbedding] = None

    @property
    def model(self) -> TextEmbedding:
        """Lazy-loads the TextEmbedding ONNX session on first use."""
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
        """Returns the embedding vector dimension."""
        if self.model_name in MODEL_DIMENSIONS:
            return MODEL_DIMENSIONS[self.model_name]
        # Fallback: compute dynamically from a dummy embedding
        test_vec = self.embed_query("test")
        return len(test_vec)

    def embed_documents(
        self,
        texts: Union[List[str], Iterable[str]],
        batch_size: int = 128,
        parallel: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Embeds a batch of document chunks for ingestion.
        Returns a list of float vectors.
        """
        if not texts:
            return []

        embeddings_generator = self.model.embed(
            documents=texts,
            batch_size=batch_size,
            parallel=parallel,
        )
        return [vec.tolist() if isinstance(vec, np.ndarray) else list(vec) for vec in embeddings_generator]

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query string for online RAG search with minimal latency.
        """
        if not text or not text.strip():
            # Return a zero vector if query is empty
            return [0.0] * self.dimension

        query_gen = self.model.query_embed(query=text)
        first_vec = next(query_gen)
        return first_vec.tolist() if isinstance(first_vec, np.ndarray) else list(first_vec)


def get_embedder(model_name: str = DEFAULT_MODEL_NAME) -> ONNXEmbedder:
    """Factory helper returning a configured ONNXEmbedder instance."""
    return ONNXEmbedder(model_name=model_name)
