# Ingestion & Qdrant Storage Pipeline

This module handles the end-to-end ingestion pipeline:
1. **Document Loading & Structural Chunking** ([`chunker.py`](chunker.py)) from markdown files in `src/data/raw`.
2. **Dense Vector Generation** ([`src.embeddings`](../embeddings/README.md)) using lightweight ONNX embeddings (`BAAI/bge-small-en-v1.5`, 384-d).
3. **Qdrant Storage & Indexing** ([`ingest_qdrant.py`](ingest_qdrant.py)) into Qdrant Cloud or local embedded storage.

---

## 1. Quick Start & Execution

### Full Ingestion to Qdrant Cloud
```bash
conda activate llm-zoomcamp
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

---

## 3. Qdrant Collection Schema (`devdoc_chunks`)

- **Vectors**:
  - `size`: 384 (Cosine distance)
- **Payload Indices**:
  - `content`: Full-text index (`TokenizerType.WORD`, lowercase) for BM25 keyword search in Phase 4.
  - `source_lib`: Keyword index (e.g. `fastapi`, `docker`, `pydantic`).
  - `file_path`: Keyword index.

---

## 4. CLI Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--chunks-path` | `src/data/processed/chunks.json` | Processed chunks JSON file |
| `--collection-name` | `devdoc_chunks` | Qdrant collection name |
| `--batch-size` | `128` | Batch size for embedding and upload |
| `--limit` | `None` | Optional limit for quick testing |
| `--recreate` | `False` | Force delete and recreate collection |
