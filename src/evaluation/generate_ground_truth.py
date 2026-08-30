"""
src/evaluation/generate_ground_truth.py

Generates a synthetic ground-truth Q&A benchmark dataset from the documentation knowledge base
using an LLM via Ollama (local or remote/cloud).
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import find_dotenv, load_dotenv
import requests
from tqdm.auto import tqdm

from src.models import ChunkModel, GroundTruthQA, QAGenerationResponse

# 1. Environment Loading: Check src/.env, root .env, or find_dotenv()

load_dotenv(find_dotenv())

DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_URL") or "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")
DEFAULT_OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

PROMPT_TEMPLATE = """You are a senior software engineer creating a high-quality evaluation benchmark for a technical documentation RAG system.

Context Information:
- Library / Framework: {source_lib}
- Topic / Section: {section_title}

Documentation Snippet:
\"\"\"
{content}
\"\"\"

Task:
1. Generate ONE realistic, specific technical question a developer would ask about {source_lib} that can be answered accurately and completely based ONLY on this documentation snippet.
   - The question must be standalone and natural (e.g. "How do I create a custom middleware in FastAPI?", "What is the command to inspect Docker volume details?").
   - DO NOT mention "the snippet", "the text", "according to the document", or "in section X".
2. Generate a concise, factual, and complete ground-truth answer based directly on the facts and code examples in the snippet.

Respond ONLY with a valid JSON object matching this exact schema:
{{
  "question": "The realistic developer question here",
  "ground_truth_answer": "The concise, factual ground-truth answer here"
}}
"""


def extract_and_validate_qa(raw_text: str) -> Optional[Dict[str, str]]:
    """
    Extracts and parses JSON from raw LLM output, handling markdown blocks,
    preambles, think tags, unescaped newlines, and schema validation.
    """
    if not raw_text or not raw_text.strip():
        return None

    # 1. Remove reasoning / think tags if present (e.g. DeepSeek/Qwen/GLM reasoning models)
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()

    def try_validate(obj: Any) -> Optional[Dict[str, str]]:
        if not isinstance(obj, dict):
            return None
        q = obj.get("question") or obj.get("developer_question") or obj.get("query")
        a = (
            obj.get("ground_truth_answer")
            or obj.get("ground_truth")
            or obj.get("answer")
            or obj.get("factual_answer")
        )
        if q and a and str(q).strip() and str(a).strip():
            return {
                "question": str(q).strip(),
                "ground_truth_answer": str(a).strip(),
            }
        return None

    # 2. Check if there is an explicit ```json ... ``` codeblock
    codeblock_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned, flags=re.IGNORECASE)
    if codeblock_match:
        cand = codeblock_match.group(1)
        try:
            data = json.loads(cand, strict=False)
            res = try_validate(data)
            if res:
                return res
        except Exception:
            pass

    # 3. Direct JSON load after stripping fences with strict=False (allows raw newlines in string literals)
    try:
        no_fences = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        no_fences = re.sub(r"\s*```$", "", no_fences)
        data = json.loads(no_fences, strict=False)
        res = try_validate(data)
        if res:
            return res
    except Exception:
        pass

    # 4. Regex extraction of outermost JSON object with trailing comma cleanup
    match = re.search(r"(\{[\s\S]*\})", cleaned)
    if match:
        json_candidate = match.group(1)
        json_clean = re.sub(r",\s*([\}\]])", r"\1", json_candidate)
        try:
            data = json.loads(json_clean, strict=False)
            res = try_validate(data)
            if res:
                return res
        except Exception:
            pass

    # 5. Regex field fallback if internal unescaped quotes broke json.loads
    q_match = re.search(r"\"(?:question|developer_question|query)\"\s*:\s*\"([^\"]*?)\"", cleaned)
    a_match = re.search(
        r"\"(?:ground_truth_answer|ground_truth|answer|factual_answer)\"\s*:\s*\"([\s\S]*?)\"\s*\}",
        cleaned,
    )
    if q_match and a_match:
        q_val = q_match.group(1).strip()
        a_val = a_match.group(1).strip()
        if q_val and a_val:
            return {"question": q_val, "ground_truth_answer": a_val}

    return None


def query_ollama(
    prompt: str,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    model: str = DEFAULT_OLLAMA_MODEL,
    api_key: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 3,
) -> Dict[str, str]:
    """
    Queries Ollama or an Ollama/OpenAI compatible endpoint with retries and JSON format mode.
    """
    clean_base_url = base_url.rstrip("/")
    api_key = api_key or DEFAULT_OLLAMA_API_KEY

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "num_predict": 2048,
        },
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                f"{clean_base_url}/api/generate",
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            res_data = response.json()
            raw_text = res_data.get("response", "")

            # If Ollama returns empty or format is nested
            if not raw_text and "message" in res_data:
                raw_text = res_data["message"].get("content", "")

            parsed = extract_and_validate_qa(raw_text)
            if parsed:
                return parsed
            else:
                last_error = ValueError(f"Failed to parse valid QA schema from response: {raw_text[:150]}...")
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.5 * attempt)

    raise RuntimeError(f"Ollama generation failed after {max_retries} attempts: {last_error}")


def load_or_create_chunks(
    chunks_path: Union[str, Path] = "src/data/processed/chunks.json",
    raw_data_dir: Union[str, Path] = "src/data/raw",
    force_rebuild: bool = False,
) -> List[Dict[str, Any]]:
    """
    Loads chunks from chunks_path. If missing or forced, runs chunker on raw_data_dir
    and saves the output to chunks_path.
    """
    chunks_file = Path(chunks_path)
    raw_dir = Path(raw_data_dir)

    # Alternate path checks
    if not chunks_file.exists() and not force_rebuild:
        alt_path = Path("data/processed/chunks.json")
        if alt_path.exists():
            chunks_file = alt_path

    if chunks_file.exists() and not force_rebuild:
        print(f"Loading existing chunks from '{chunks_file}'...")
        with open(chunks_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} chunks.")
        return data

    # Fallback to chunking raw docs
    if not raw_dir.exists():
        alt_raw = Path("data/raw")
        if alt_raw.exists():
            raw_dir = alt_raw
        else:
            raise FileNotFoundError(
                f"Neither chunks file '{chunks_file}' nor raw docs directory '{raw_dir}' were found."
            )

    print(f"Chunks file '{chunks_file}' not found. Ingesting and chunking docs from '{raw_dir}'...")
    from src.ingestion.chunker import chunk, load_docs

    docs = load_docs(str(raw_dir))
    if not docs:
        raise ValueError(f"No markdown documents found in '{raw_dir}'.")

    chunk_objects = chunk(docs)
    chunks_data = [c.model_dump() if hasattr(c, "model_dump") else c.dict() for c in chunk_objects]

    # Persist chunks for subsequent runs
    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(chunks_data)} chunks to '{chunks_file}'.")
    return chunks_data


def sample_stratified_chunks(
    chunks: List[Union[Dict[str, Any], ChunkModel]],
    total_samples: int = 40,
    min_char_len: int = 200,
    max_char_len: int = 3000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Samples chunks evenly across different source libraries with quality filtering
    and deterministic random seed.
    """
    random.seed(seed)

    # Convert all chunks to standard dicts
    dict_chunks: List[Dict[str, Any]] = []
    for c in chunks:
        if isinstance(c, dict):
            dict_chunks.append(c)
        elif hasattr(c, "model_dump"):
            dict_chunks.append(c.model_dump())
        elif hasattr(c, "dict"):
            dict_chunks.append(c.dict())

    # Quality filter: length and boilerplate exclusion
    ignored_keywords = ["license", "contributing", "changelog", "code_of_conduct", "_vale", "release-notes"]
    viable_chunks: List[Dict[str, Any]] = []

    for c in dict_chunks:
        content = c.get("content", "")
        file_path = c.get("file_path", "").lower()
        char_count = c.get("char_count", len(content))

        if not (min_char_len <= char_count <= max_char_len):
            continue

        if any(kw in file_path for kw in ignored_keywords):
            continue

        # Skip chunks that are just table of contents / list of links
        link_count = len(re.findall(r"\[.*?\]\(.*?\)", content))
        if link_count > 10 and len(content.split()) < 50:
            continue

        viable_chunks.append(c)

    if not viable_chunks:
        raise ValueError("No viable chunks left after applying quality filters.")

    # Group by source_lib
    lib_groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in viable_chunks:
        lib = c.get("source_lib", "general")
        lib_groups.setdefault(lib, []).append(c)

    num_libs = len(lib_groups)
    samples_per_lib = max(1, total_samples // num_libs)
    selected: List[Dict[str, Any]] = []

    for lib, group in lib_groups.items():
        k = min(len(group), samples_per_lib)
        selected.extend(random.sample(group, k))

    # Fill remaining quota from unselected viable chunks
    if len(selected) < total_samples:
        selected_ids = {c.get("chunk_id") for c in selected}
        remaining = [c for c in viable_chunks if c.get("chunk_id") not in selected_ids]
        needed = min(len(remaining), total_samples - len(selected))
        if needed > 0:
            selected.extend(random.sample(remaining, needed))

    random.shuffle(selected)
    return selected[:total_samples]


def generate_benchmark(
    chunks_path: str = "src/data/processed/chunks.json",
    raw_data_dir: str = "src/data/raw",
    output_path: str = "src/data/ground_truth.json",
    num_samples: int = 40,
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    api_key: Optional[str] = None,
    seed: int = 42,
    force_rebuild_chunks: bool = False,
) -> List[GroundTruthQA]:
    """
    Main benchmark generation orchestrator.
    """
    start_time = time.time()
    chunks = load_or_create_chunks(
        chunks_path=chunks_path,
        raw_data_dir=raw_data_dir,
        force_rebuild=force_rebuild_chunks,
    )

    print(f"Sampling {num_samples} candidate chunks across libraries (seed={seed})...")
    sampled_chunks = sample_stratified_chunks(chunks, total_samples=num_samples, seed=seed)

    print(f"Connecting to LLM [{model}] at {base_url}...")
    qa_dataset: List[GroundTruthQA] = []
    failed_count = 0
    lib_stats: Dict[str, int] = {}

    for idx, chunk_data in enumerate(tqdm(sampled_chunks, desc=f"Generating QA pairs via {model}"), start=1):
        source_lib = chunk_data.get("source_lib", "general")
        section_title = chunk_data.get("section_title", "Documentation")
        content = chunk_data.get("content", "")

        prompt = PROMPT_TEMPLATE.format(
            source_lib=source_lib,
            section_title=section_title,
            content=content,
        )

        try:
            result = query_ollama(
                prompt=prompt,
                base_url=base_url,
                model=model,
                api_key=api_key,
            )
            question = result["question"]
            answer = result["ground_truth_answer"]

            qa_item = GroundTruthQA(
                qa_id=f"qa-{idx:03d}",
                source_chunk_id=chunk_data["chunk_id"],
                source_lib=source_lib,
                file_path=chunk_data.get("file_path", ""),
                question=question,
                ground_truth_answer=answer,
            )
            qa_dataset.append(qa_item)
            lib_stats[source_lib] = lib_stats.get(source_lib, 0) + 1

        except Exception as e:
            print(f"\n[Warning] Skipped chunk {chunk_data.get('chunk_id')}: {e}")
            failed_count += 1

    # Save output benchmark JSON
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([item.model_dump() for item in qa_dataset], f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(" Benchmark Generation Complete!")
    print("=" * 60)
    print(f"Total Q&A pairs generated: {len(qa_dataset)} / {len(sampled_chunks)}")
    print(f"Failed / Skipped:          {failed_count}")
    print(f"Execution time:            {elapsed:.2f}s")
    print(f"Output saved to:           {out_file.resolve()}")
    print("\nBreakdown by Library:")
    for lib, count in sorted(lib_stats.items()):
        print(f"  - {lib:<15}: {count} pairs")
    print("=" * 60 + "\n")

    return qa_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic evaluation ground-truth dataset from documentation chunks."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=40,
        help="Total number of ground-truth QA pairs to generate (default: 40).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name (default: {DEFAULT_OLLAMA_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_OLLAMA_BASE_URL,
        help=f"Ollama Base URL (default: {DEFAULT_OLLAMA_BASE_URL}).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_OLLAMA_API_KEY,
        help="API Key for Ollama / cloud endpoint if required.",
    )
    parser.add_argument(
        "--chunks-path",
        type=str,
        default="src/data/processed/chunks.json",
        help="Path to processed chunks JSON file.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="src/data/raw",
        help="Path to raw markdown documentation directory.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="src/data/ground_truth.json",
        help="Path to output ground truth JSON file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--rebuild-chunks",
        action="store_true",
        help="Force re-chunking raw docs even if chunks.json already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generate_benchmark(
        chunks_path=args.chunks_path,
        raw_data_dir=args.raw_dir,
        output_path=args.output_path,
        num_samples=args.num_samples,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key or None,
        seed=args.seed,
        force_rebuild_chunks=args.rebuild_chunks,
    )


if __name__ == "__main__":
    main()