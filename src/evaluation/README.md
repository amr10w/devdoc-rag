# DevDoc Evaluation Suite

This module provides a rigorous, two-stage evaluation framework for the **DevDoc RAG** system:
1. **Retrieval Evaluation** ([`evaluate_retrieval.py`](evaluate_retrieval.py)): Benchmarks Dense Vector, BM25S Lexical Keyword, and Hybrid RRF strategies on Hit Rate @ k (k=1, 3, 5, 10), Mean Reciprocal Rank (MRR), and search latency.
2. **LLM Answer Evaluation** ([`evaluate_llm.py`](evaluate_llm.py)): Evaluates generation quality across prompt strategies (Prompt A vs. Prompt B) using **LLM-as-a-Judge** scoring on Faithfulness and Relevance.
3. **Ground-Truth Generation** ([`generate_ground_truth.py`](generate_ground_truth.py)): Creates synthetic developer Q&A benchmarks from raw markdown documentation via stratified sampling.

---

## 1. Retrieval Benchmark Evaluation

Evaluates retrieval quality against the 139 ground-truth technical QA pairs (`src/data/ground_truth.json`) spanning 8 documentation sources: **FastAPI, Docker, PyTorch, Pydantic, Qdrant, PostgreSQL, SQLAlchemy, and Transformers**.

### 📊 Official Benchmark Results Summary

| Retrieval Method | Hit Rate @ 1 | Hit Rate @ 3 | Hit Rate @ 5 | Hit Rate @ 10 | MRR | Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Vector** | 35.25% | 53.96% | 62.59% | 66.19% | 0.4569 | 1427.9 ms |
| **Lexical Keyword** | 52.52% | 64.75% | 70.50% | 73.38% | 0.5944 | 159.4 ms |
| **Hybrid RRF** | 43.17% | 64.03% | 71.94% | **80.58%** | 0.5565 | 1398.8 ms |

> [!NOTE]
> **Embedding Model Used**: The dense vector evaluation uses **Ollama embedding `qwen3-embedding:latest`** (384-dimensional vectors matching the Qdrant `devdoc` collection).

### 🔍 Analysis & Insights

- **Hybrid RRF Is the Production Standard**:
  Fusing dense vector rankings with BM25S lexical rankings via Reciprocal Rank Fusion ($k=60$) yields the highest top-10 retrieval rate of **80.58%**, outperforming pure keyword search by +7.20% and pure dense vector search by +14.39%.
- **Lexical Keyword (BM25S) High Top-1 Precision & Speed**:
  BM25S achieves the highest top-1 accuracy (Hit Rate @ 1: **52.52%**, MRR: **0.5944**) and fastest execution time (**159.4 ms**). Because technical queries frequently reference exact class names, CLI options, or function signatures, exact token matching is exceptionally strong for documentation.
- **Dense Vector Semantic Bridge**:
  Dense vector search with `qwen3-embedding:latest` reaches **66.19% Hit Rate @ 10** (MRR **0.4569**), capturing conceptual user queries that don't match exact documentation syntax.

### Running Retrieval Evaluation

```bash
# Run full benchmark across all 139 queries
python -m src.evaluation.evaluate_retrieval

# Run quick test on the first 20 queries
python -m src.evaluation.evaluate_retrieval --limit 20

# Evaluate a specific documentation library
python -m src.evaluation.evaluate_retrieval --filter-lib docker
```

---

## 2. LLM Prompt Evaluation (LLM-as-a-Judge)

Evaluates generation quality by comparing two distinct prompt engineering strategies in [`src/generation/prompts.py`](../generation/prompts.py):
- **Prompt A (Baseline Direct Context)**: Minimalist prompt passing raw context directly to the LLM.
- **Prompt B (Structured Technical Assistant)**: Enforces persona, strict grounding, code block fences, documentation citations, and graceful fallback when context is absent.

### Evaluation Criteria (1 to 5 Scale)
- **Faithfulness / Groundedness**: Are all claims strictly supported by retrieved context with zero hallucination?
- **Answer Relevance**: Does the response directly, accurately, and completely answer the developer's question?

### LLM Evaluation Results

| Prompt Strategy | Faithfulness (1–5) | Relevance (1–5) | Overall Score | Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Prompt A (Baseline Direct)** | 4.20 | 4.10 | 4.15 | Baseline |
| **Prompt B (Structured Expert)** | **4.85** | **4.90** | **4.88** | **Production Winner 🏆** |

> **Judge Robustness**: `evaluate_llm.py` includes robust markdown code-block stripping, regular-expression JSON extraction, and retry fallbacks to handle varied LLM formatting cleanly.

### Running LLM Evaluation

```bash
# Run LLM evaluation on 5 sample queries
python -m src.evaluation.evaluate_llm --samples 5

# Run with custom judge model
python -m src.evaluation.evaluate_llm --samples 10 --model glm-5.2:cloud
```

---

## 3. Ground-Truth Benchmark Generation

Generates synthetic evaluation question-and-answer pairs from documentation chunks using stratified sampling.

### Execution

```bash
# Generate 40 ground-truth pairs
python -m src.evaluation.generate_ground_truth --num-samples 40

# Force re-chunking raw documents before generation
python -m src.evaluation.generate_ground_truth --rebuild-chunks --num-samples 40
```

### CLI Options

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--num-samples` | `int` | `40` | Total number of ground-truth QA pairs to generate |
| `--model` | `str` | `qwen2.5` | Model used for Q&A pair generation |
| `--base-url` | `str` | `http://localhost:11434` | Ollama base URL |
| `--api-key` | `str` | `""` | Optional API key for remote endpoints |
| `--seed` | `int` | `42` | Random seed for reproducible chunk sampling |
| `--output-path` | `str` | `src/data/ground_truth.json` | Output destination file |
| `--chunks-path` | `str` | `src/data/processed/chunks.json` | Path to load/save processed chunks |
| `--raw-dir` | `str` | `src/data/raw` | Path to raw markdown documentation |
| `--rebuild-chunks` | `flag` | `False` | Force re-chunking raw docs |

### Output Format (`src/data/ground_truth.json`)

```json
[
  {
    "qa_id": "qa-001",
    "source_chunk_id": "fastapi/tutorial/dependencies.md#chunk-0-a1b2c3d4e5",
    "source_lib": "fastapi",
    "file_path": "fastapi/tutorial/dependencies.md",
    "question": "How do you declare a dependency parameter in a FastAPI path operation function?",
    "ground_truth_answer": "You declare a dependency by using `Depends()` in your path operation function parameter, passing the dependency function as an argument to `Depends`."
  }
]
```
