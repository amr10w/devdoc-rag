# ONNX Embeddings Module

This module provides the **`ONNXEmbedder`** engine powered by **`fastembed`** (Qdrant's official ONNX embedding library). It is used across both the offline ingestion pipeline and the real-time RAG query flow.

---

## 1. Why ONNX for RAG Embeddings?

- **Zero PyTorch / CUDA footprint**: Eliminates 2–3 GB of heavy ML dependencies.
- **Fast CPU & GPU Inference**: Optimized ONNX runtime with quantized weights.
- **Sub-10ms Query Latency**: Instant user question vectorization for FastAPI `/ask`.
- **Cross-Platform & Cloud-Ready**: Easily deploys inside minimal Docker containers (~150MB).

---

## 2. Default Model: `BAAI/bge-small-en-v1.5`

- **Vector Dimension**: `384`
- **Distance Metric**: `Cosine`
- **Max Sequence Length**: `512` tokens
- **Benchmark Performance**: Top tier on MTEB retrieval benchmarks among lightweight models.

---

## 3. Usage Examples

### Embed Query (Online RAG)
```python
from src.embeddings import ONNXEmbedder

embedder = ONNXEmbedder()
query_vector = embedder.embed_query("How do dependencies work in FastAPI?")
print(f"Dimension: {len(query_vector)}") # 384
```

### Embed Batch of Documents (Offline Ingestion)
```python
from src.embeddings import ONNXEmbedder

embedder = ONNXEmbedder()
chunks = ["FastAPI is a modern web framework...", "Docker volumes persist data..."]
vectors = embedder.embed_documents(chunks, batch_size=64)
print(f"Generated {len(vectors)} vectors of dimension {len(vectors[0])}")
```
