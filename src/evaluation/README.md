# Evaluation Benchmark Generation

This module provides tools for generating synthetic ground-truth question-and-answer benchmarks for evaluating the **DevDoc RAG** system (retrieval hit rate, MRR, and LLM answer quality).

---

## 1. Quick Start & Execution Modes

### Mode A: Standard Python Package Installation (Recommended for Deployment)
Install the repository as an editable package in your environment:
```bash
pip install -e .
```
Then run the benchmark generator directly using the console script:
```bash
devdoc-ground-truth --num-samples 40
```
Or via standard Python module execution:
```bash
python -m src.evaluation.generate_ground_truth --num-samples 40
```

### Mode B: Using `uv`
```bash
uv run python -m src.evaluation.generate_ground_truth
```

---

## 2. What Happens Automatically

When you run the generator:
1. Detects raw documentation in `src/data/raw` (or `data/raw`).
2. Automatically parses and chunks the markdown files into `src/data/processed/chunks.json` if not already generated.
3. Performs **stratified sampling** across all available libraries (`fastapi`, `docker`, `pydantic`, `pytorch`, `qdrant`, `sqlalchemy`, `transformers`, `postgresql`).
4. Connects to Ollama to generate realistic developer question-and-answer pairs.
5. Saves the final benchmark to `src/data/ground_truth.json`.

---

## 2. Configuration & Environment Variables

The script automatically detects `.env` files in `src/.env` or the project root `.env`.

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` / `OLLAMA_API_URL` | `http://localhost:11434` | Ollama server URL (local or remote/cloud) |
| `OLLAMA_MODEL` | `qwen2.5` | Model used for Q&A pair generation |
| `OLLAMA_API_KEY` | *(None)* | Bearer API token if using authenticated cloud endpoints |

Example `src/.env`:
```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5
```

---

## 3. CLI Command Options

You can customize the generation process using command-line arguments:

```bash
uv run python -m src.evaluation.generate_ground_truth [OPTIONS]
```

### Available Options

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--num-samples` | `int` | `40` | Total number of ground-truth QA pairs to generate |
| `--model` | `str` | `qwen2.5` | LLM model name |
| `--base-url` | `str` | `http://localhost:11434` | Ollama base URL |
| `--api-key` | `str` | `""` | Optional API key for remote endpoints |
| `--seed` | `int` | `42` | Random seed for reproducible chunk sampling |
| `--output-path` | `str` | `src/data/ground_truth.json` | Destination file for the generated benchmark |
| `--chunks-path` | `str` | `src/data/processed/chunks.json` | Path to load/save processed chunks |
| `--raw-dir` | `str` | `src/data/raw` | Path to raw markdown documentation |
| `--rebuild-chunks` | `flag` | `False` | Force re-chunking raw docs even if `chunks.json` exists |

---

## 4. Usage Examples

### Run a quick test (5 sample pairs)
```bash
conda activate llm-zoomcamp
uv run python -m src.evaluation.generate_ground_truth --num-samples 5
```

### Force re-chunking of raw documents
```bash
conda activate llm-zoomcamp
uv run python -m src.evaluation.generate_ground_truth --rebuild-chunks --num-samples 40
```

### Run with a specific model and seed for reproducibility
```bash
conda activate llm-zoomcamp
uv run python -m src.evaluation.generate_ground_truth --model qwen2.5 --seed 123 --num-samples 30
```

---

## 5. Output Format (`src/data/ground_truth.json`)

The output is saved as a JSON array of `GroundTruthQA` objects:

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
