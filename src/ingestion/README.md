# Ingestion & Qdrant Storage Pipeline

This module handles the end-to-end ingestion pipeline:
1. **Document Loading & Structural Chunking** ([`chunker.py`](chunker.py)) from markdown files in `src/data/raw`.
2. **Dense Vector Generation** using the configured provider (`EMBEDDING_PROVIDER`: `onnx` / FastEmbed or `ollama`).
3. **Qdrant Storage & Indexing** ([`ingest_qdrant.py`](ingest_qdrant.py)) into Qdrant Cloud or local embedded storage.

> [!NOTE]
> **Embedding Architecture**:
> - Embedder is selected dynamically via `EMBEDDING_PROVIDER` env variable (`onnx` or `ollama`).
> - Vector dimension is configurable via `VECTOR_DIMENSION` env variable (default: `384`), aligned with collection configurations in Qdrant with Cosine distance.

---

## 1. Quick Start & Execution

### Prerequisites
Start the local Ollama server before running ingestion:
```bash
ollama serve
```

### Full Ingestion to Qdrant
```bash
devdoc-ingest --recreate
```
*(Or via standard Python module syntax: `python -m src.ingestion.ingest_qdrant --recreate`)*

### Quick Test Run (e.g. First 100 Chunks)
```bash
devdoc-ingest --limit 100 --recreate
```

---

## 2. Configuration & Environment Variables

The pipeline automatically loads credentials from `src/.env` or root `.env`:

| Variable | Description |
| :--- | :--- |
| `Qdrant_API_URL` / `QDRANT_API_URL` | Qdrant Cloud Cluster URL |
| `Qdrant_API_KEY` / `QDRANT_API_KEY` | Qdrant Cloud API Key |
| `QDRANT_STORAGE_PATH` | Local disk fallback directory (default: `src/data/qdrant_storage`) |
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://localhost:11434`) |
| `OLLAMA_EMBED_MODEL` | Ollama embedding model (default: `qwen3-embedding:latest`) |

---

## 3. Qdrant Collection Schema (`devdoc`)

- **Vectors**:
  - `size`: 384 (Cosine distance)
- **Payload Indices**:
  - `content`: Full-text index (`TokenizerType.WORD`, lowercase) for BM25 keyword search.
  - `source_lib`: Keyword index (e.g. `fastapi`, `docker`, `pydantic`).
  - `file_path`: Keyword index.

---

## 4. CLI Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--chunks-path` | `src/data/processed/chunks.json` | Processed chunks JSON file |
| `--collection-name` | `devdoc` | Qdrant collection name |
| `--batch-size` | `128` | Batch size for embedding and upload |
| `--start-batch` | `None` | Batch number to resume from (e.g. `104` to skip the first 104 batches) |
| `--offset` | `0` | Number of chunks to skip before starting ingestion |
| `--limit` | `None` | Optional limit for quick testing |
| `--recreate` | `False` | Force delete and recreate collection |
| `--embedding-provider` | `onnx` (or env) | Embedding provider (`onnx` or `ollama`) |
| `--vector-dim` | `384` (or env) | Vector dimension for collection |
| `--storage-path` | `src/data/qdrant_storage` | Local disk fallback storage path |
