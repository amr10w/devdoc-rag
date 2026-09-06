# DevDoc Embeddings Module

This module provides a unified embedding interface supporting two complementary engines:
1. **`OllamaEmbedder`**: Powered by Ollama using **`qwen3-embedding:latest`** (384 dimensions). This is the primary embedding engine used for the official retrieval benchmarks and knowledge base indexing.
2. **`ONNXEmbedder`**: Powered by **`fastembed`** using `BAAI/bge-small-en-v1.5` (384 dimensions). Provides lightweight, sub-10ms CPU inference with zero PyTorch or CUDA dependencies.

The module provides an automated factory function [`get_embedder()`](embedder.py) that resolves the active provider from the `EMBEDDING_PROVIDER` environment variable (`ollama` or `onnx`).

---

## 1. Provider Comparison & Specifications

| Feature | `OllamaEmbedder` | `ONNXEmbedder` |
| :--- | :--- | :--- |
| **Model** | `qwen3-embedding:latest` | `BAAI/bge-small-en-v1.5` |
| **Vector Dimension** | `384` (Configurable via `VECTOR_DIMENSION`) | `384` |
| **Distance Metric** | Cosine | Cosine |
| **Execution Mode** | Remote / Local Ollama API daemon | In-process ONNX Runtime (C++) |
| **Dependencies** | `ollama` Python client | `fastembed`, `onnxruntime` |
| **Primary Use Case** | Benchmark evaluation, high semantic recall | Serverless / zero-daemon deployment, sub-10ms latency |
| **Failover Support** | Automatically falls back to ONNX if Ollama is unreachable | Standalone local execution |

---

## 2. Configuration & Environment Variables

Configure embeddings via `.env` or system environment variables:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `EMBEDDING_PROVIDER` | `ollama` | Active provider: `ollama` or `onnx` |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:latest` | Embedding model tag served by Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama daemon endpoint URL |
| `VECTOR_DIMENSION` | `384` | Embedding vector dimension (must match Qdrant collection) |

---

## 3. Usage Examples

### Recommended: Factory Initialization
```python
from src.embeddings import get_embedder

# Automatically loads provider configured in EMBEDDING_PROVIDER ('ollama' or 'onnx')
embedder = get_embedder()

# Vectorize a search query
query_vector = embedder.embed_query("How do dependencies work in FastAPI?")
print(f"Provider: {type(embedder).__name__}, Dimension: {len(query_vector)}")
```

### Direct `OllamaEmbedder` Usage (Evaluation Benchmark Model)
```python
from src.embeddings.embedder import OllamaEmbedder

embedder = OllamaEmbedder(model_name="qwen3-embedding:latest", dimensions=384)

# Single query embedding
query_vector = embedder.embed_query("How to mount static files in FastAPI?")

# Batch document embedding (offline ingestion)
documents = [
    "FastAPI is a modern, fast web framework for building APIs.",
    "Docker volumes provide persistent storage for containerized applications."
]
vectors = embedder.embed_documents(documents, batch_size=64)
print(f"Generated {len(vectors)} vectors of dimension {len(vectors[0])}")
```

### Direct `ONNXEmbedder` Usage (FastEmbed In-Process)
```python
from src.embeddings.embedder import ONNXEmbedder

embedder = ONNXEmbedder(model_name="BAAI/bge-small-en-v1.5")
query_vector = embedder.embed_query("docker compose networking overview")
print(f"Dimension: {len(query_vector)}") # 384
```

---

## 4. Benchmark Performance (`qwen3-embedding:latest`)

In the official 139-query retrieval benchmark evaluated across 8 technical libraries:
- **Dense Vector Search (`qwen3-embedding:latest`)**:
  - Hit Rate @ 1: **35.25%**
  - Hit Rate @ 5: **62.59%**
  - Hit Rate @ 10: **66.19%**
  - MRR: **0.4569**
- **Hybrid RRF (`qwen3-embedding:latest` + BM25S)**:
  - Hit Rate @ 1: **43.17%**
  - Hit Rate @ 5: **71.94%**
  - Hit Rate @ 10: **80.58%**
  - MRR: **0.5565**

Combining `qwen3-embedding:latest` with BM25S lexical search provides an **+14.39% Hit Rate @ 10 boost** over dense search alone, establishing the hybrid strategy as the production standard.

