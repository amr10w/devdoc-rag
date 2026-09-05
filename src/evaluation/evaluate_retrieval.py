"""
src/evaluation/evaluate_retrieval.py

Benchmarks Vector, Lexical/BM25 Text, and Hybrid RRF retrieval strategies
against the ground-truth QA dataset. Calculates Hit Rate @ k (k=1, 3, 5),
Mean Reciprocal Rank (MRR), and average search latency.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from dotenv import find_dotenv, load_dotenv
from tqdm.auto import tqdm

from src.retrieval.search import DevDocSearcher, get_searcher

load_dotenv(find_dotenv())


def load_ground_truth(file_path: str | Path) -> list[dict[str, Any]]:
    """Loads ground-truth Q&A dataset from JSON."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def evaluate_query(
    search_func,
    query: str,
    target_chunk_id: str,
    source_lib: str | None = None,
    top_k: int = 5,
) -> tuple[dict[int, int], float, float, list[str]]:
    """
    Evaluates a single query against the retrieval function.

    Returns:
        (hit_at_k dict, reciprocal_rank, latency_ms, retrieved_chunk_ids)
    """
    start = time.perf_counter()
    results = search_func(query=query, k=top_k, source_lib=source_lib)
    latency_ms = (time.perf_counter() - start) * 1000.0

    retrieved_ids = [r.get("chunk_id", "") for r in results]

    hits = {1: 0, 3: 0, 5: 0}
    reciprocal_rank = 0.0

    for rank_idx, cid in enumerate(retrieved_ids):
        rank = rank_idx + 1
        if cid == target_chunk_id:
            for k in hits:
                if rank <= k:
                    hits[k] = 1
            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank

    return hits, reciprocal_rank, latency_ms, retrieved_ids


def run_retrieval_benchmark(
    ground_truth_path: str = "src/data/ground_truth.json",
    limit: int | None = None,
    filter_lib: str | None = None,
    output_path: str | None = "src/data/retrieval_evaluation_results.json",
) -> dict[str, Any]:
    """
    Runs the comprehensive retrieval evaluation comparing:
    1. Dense Vector Search
    2. Lexical / Keyword Search
    3. Hybrid RRF Search
    """
    qa_pairs = load_ground_truth(ground_truth_path)

    if filter_lib:
        qa_pairs = [q for q in qa_pairs if q.get("source_lib", "").lower() == filter_lib.lower()]

    if limit and limit > 0:
        qa_pairs = qa_pairs[:limit]

    total_queries = len(qa_pairs)
    if total_queries == 0:
        raise ValueError("No ground-truth queries found matching the criteria.")

    print("\n" + "=" * 65)
    print(" 🚀 Starting Retrieval Benchmark Evaluation")
    print("=" * 65)
    print(f"Total Test Queries:     {total_queries}")
    print(f"Library Filter:         {filter_lib or 'All Libraries'}")
    print(f"Ground Truth Dataset:   {ground_truth_path}")
    print("=" * 65 + "\n")

    searcher = get_searcher()

    methods = {
        "Dense Vector": searcher.vector_search,
        "Lexical Keyword": searcher.text_search,
        "Hybrid RRF": searcher.hybrid_search,
    }

    results_summary: dict[str, dict[str, Any]] = {}

    for method_name, search_func in methods.items():
        print(f"Evaluating: [{method_name}] ...")
        hit_1_list: list[int] = []
        hit_3_list: list[int] = []
        hit_5_list: list[int] = []
        rr_list: list[float] = []
        latency_list: list[float] = []

        for item in tqdm(qa_pairs, desc=f"Testing {method_name}"):
            query = item["question"]
            target_cid = item["source_chunk_id"]

            hits, rr, latency, _ = evaluate_query(
                search_func=search_func,
                query=query,
                target_chunk_id=target_cid,
                top_k=5,
            )

            hit_1_list.append(hits[1])
            hit_3_list.append(hits[3])
            hit_5_list.append(hits[5])
            rr_list.append(rr)
            latency_list.append(latency)

        hit_rate_1 = float(np.mean(hit_1_list))
        hit_rate_3 = float(np.mean(hit_3_list))
        hit_rate_5 = float(np.mean(hit_5_list))
        mrr = float(np.mean(rr_list))
        avg_latency = float(np.mean(latency_list))

        results_summary[method_name] = {
            "hit_rate_at_1": round(hit_rate_1, 4),
            "hit_rate_at_3": round(hit_rate_3, 4),
            "hit_rate_at_5": round(hit_rate_5, 4),
            "mrr": round(mrr, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "total_evaluated": total_queries,
        }

    # Generate Markdown Table
    md_table = [
        "| Retrieval Method | Hit Rate @ 1 | Hit Rate @ 3 | Hit Rate @ 5 | MRR | Avg Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, stats in results_summary.items():
        md_table.append(
            f"| **{name}** | {stats['hit_rate_at_1']:.2%} | {stats['hit_rate_at_3']:.2%} | "
            f"{stats['hit_rate_at_5']:.2%} | {stats['mrr']:.4f} | {stats['avg_latency_ms']:.1f} ms |"
        )

    markdown_output = "\n".join(md_table)

    print("\n" + "=" * 65)
    print(" 📊 Retrieval Evaluation Results Summary")
    print("=" * 65)
    print(markdown_output)
    print("=" * 65 + "\n")

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_queries": total_queries,
        "results": results_summary,
        "markdown_table": markdown_output,
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2)
        print(f"Results successfully saved to: {output_path}")

    return final_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DevDoc RAG Retrieval Strategies")
    parser.add_argument("--ground-truth", type=str, default="src/data/ground_truth.json", help="Path to ground-truth JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation queries")
    parser.add_argument("--filter-lib", type=str, default=None, help="Filter evaluation by source library")
    parser.add_argument("--output", type=str, default="src/data/retrieval_evaluation_results.json", help="Path to output JSON")
    args = parser.parse_args()

    run_retrieval_benchmark(
        ground_truth_path=args.ground_truth,
        limit=args.limit,
        filter_lib=args.filter_lib,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
