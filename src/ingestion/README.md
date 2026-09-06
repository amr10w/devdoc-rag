# Ingestion & Qdrant Storage Pipeline

This module handles the end-to-end ingestion pipeline:
1. **Document Loading & Structural Chunking** ([`chunker.py`](chunker.py)): Hierarchically parses markdown documentation in `src/data/raw` (preserving code blocks within context) into `src/data/processed/chunks.json` (over 20,000 chunks across 8 frameworks).
2. **Dense Vector Generation** ([`embedder.py`](../embeddings/embedder.py)): Vectorizes chunks using the configured provider (`EMBEDDING_PROVIDER`: `ollama` with `qwen3-embedding:latest` or `onnx` with FastEmbed).
3. **Qdrant Storage & Indexing** ([`ingest_qdrant.py`](ingest_qdrant.py)): Idempotently indexes embeddings and text payloads into Qdrant Cloud or local embedded storage.

> [!NOTE]
> **Embedding Architecture & Dimension Alignment**:
> - **Provider**: Configured via `EMBEDDING_PROVIDER` (`ollama` or `onnx`).
> - **Default Benchmark Model**: **Ollama `qwen3-embedding:latest`** (or `BAAI/bge-small-en-v1.5` via FastEmbed).
> - **Vector Dimension**: `384` (configurable via `VECTOR_DIMENSION`), precisely matching the Qdrant `devdoc` collection configuration with Cosine distance.
>
> 🔗 **Download Raw Documentation & Chunks (`chunks.json`)**: [Google Drive Archive](https://drive.google.com/drive/folders/1LZDnexkJH9sGl_u1uLAMm1xik4nWmHud?usp=sharing)

---

## 1. Quick Start & Execution

### Prerequisites (Ollama Mode)
Start the Ollama daemon and ensure `qwen3-embedding:latest` is pulled:
```bash
ollama pull qwen3-embedding:latest
ollama serve
```

### Full Ingestion to Qdrant
```bash
# Ingest all chunks using the configured embedding provider
python -m src.ingestion.ingest_qdrant --recreate
```
*(Or via console script: `devdoc-ingest --recreate`)*

### Quick Test Run (e.g. First 100 Chunks)
```bash
python -m src.ingestion.ingest_qdrant --limit 100 --recreate
```

### Resuming an Ingestion Run
For large corpus uploads, you can resume ingestion from a specific batch without restarting:
```bash
# Resume from batch 104
python -m src.ingestion.ingest_qdrant --start-batch 104
```

---

## 2. Configuration & Environment Variables

The pipeline automatically loads credentials from `src/.env` or root `.env`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `EMBEDDING_PROVIDER` | `ollama` | Embedding provider (`ollama` or `onnx`) |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:latest` | Embedding model tag (384 dimensions) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama daemon endpoint URL |
| `VECTOR_DIMENSION` | `384` | Embedding vector dimension |
| `QDRANT_API_URL` | *(Optional)* | Qdrant Cloud Cluster URL (falls back to local disk) |
| `QDRANT_API_KEY` | *(Optional)* | Qdrant Cloud API Key |
| `QDRANT_STORAGE_PATH` | `src/data/qdrant_storage` | Local disk fallback storage path |

---

## 3. Qdrant Collection Schema (`devdoc`)

- **Vectors**:
  - `size`: 384 (Cosine distance)
- **Payload Indices**:
  - `content`: Full-text index (`TokenizerType.WORD`, lowercase) for BM25 keyword search.
  - `source_lib`: Keyword index (`fastapi`, `docker`, `pydantic`, `pytorch`, `qdrant`, `postgresql`, `sqlalchemy`, `transformers`).
  - `file_path`: Keyword index.
- **Point IDs**: Deterministic UUIDs generated from `chunk_id` for idempotent, overwrite-safe upserts.

---

## 4. CLI Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--chunks-path` | `src/data/processed/chunks.json` | Processed chunks JSON file |
| `--collection-name` | `devdoc` | Qdrant collection name |
| `--batch-size` | `128` | Batch size for embedding and upload |
| `--start-batch` | `None` | Batch number to resume from |
| `--offset` | `0` | Number of chunks to skip before starting ingestion |
| `--limit` | `None` | Optional limit for quick testing |
| `--recreate` | `False` | Force delete and recreate collection |
| `--embedding-provider` | `ollama` (or env) | Embedding provider (`ollama` or `onnx`) |
| `--vector-dim` | `384` (or env) | Vector dimension for collection |
| `--storage-path` | `src/data/qdrant_storage` | Local disk fallback storage path |

---

## 5. Downstream Evaluation Impact

The collection ingested with `qwen3-embedding:latest` directly powers the retrieval benchmarks:
- **Lexical Keyword (BM25S)**: 73.38% Hit Rate @ 10, MRR 0.5944
- **Dense Vector (`qwen3-embedding:latest`)**: 66.19% Hit Rate @ 10, MRR 0.4569
- **Hybrid RRF**: **80.58% Hit Rate @ 10**, MRR 0.5565
See [`src/evaluation/README.md`](../evaluation/README.md) for full benchmark details.

