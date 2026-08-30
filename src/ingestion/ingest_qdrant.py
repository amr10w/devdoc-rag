"""
src/ingestion/ingest_qdrant.py

Ingests documentation chunks into Qdrant (Cloud or Local Embedded) using ONNX embeddings.
Sets up vector configurations, payload indices (full-text and keyword), and uploads points in batches.
"""

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import find_dotenv, load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm.auto import tqdm

from src.embeddings import ONNXEmbedder

# 1. Environment Loading: Check src/.env, root .env, or find_dotenv()
env_paths = [
    Path("src/.env"),
    Path(".env"),
    Path(__file__).resolve().parent.parent / ".env",
    Path(__file__).resolve().parent.parent.parent / ".env",
]
loaded = False
for p in env_paths:
    if p.exists():
        load_dotenv(dotenv_path=p, override=True)
        loaded = True
        break
if not loaded:
    load_dotenv(find_dotenv())

# Read Qdrant credentials with flexible case matching
DEFAULT_QDRANT_URL = (
    os.getenv("Qdrant_API_URL")
    or os.getenv("QDRANT_API_URL")
    or os.getenv("QDRANT_URL")
    or None
)
DEFAULT_QDRANT_API_KEY = (
    os.getenv("Qdrant_API_KEY")
    or os.getenv("QDRANT_API_KEY")
    or None
)
DEFAULT_STORAGE_PATH = os.getenv("QDRANT_STORAGE_PATH", "src/data/qdrant_storage")
DEFAULT_COLLECTION_NAME = "devdoc_chunks"


def get_qdrant_client(
    url: Optional[str] = DEFAULT_QDRANT_URL,
    api_key: Optional[str] = DEFAULT_QDRANT_API_KEY,
    storage_path: Optional[str] = DEFAULT_STORAGE_PATH,
    timeout: int = 60,
) -> QdrantClient:
    """
    Initializes QdrantClient.
    Connects to Qdrant Cloud/remote server if URL is provided,
    otherwise falls back to local embedded disk storage.
    """
    if url:
        clean_url = url.strip().rstrip("/")
        print(f"Connecting to remote Qdrant at: {clean_url}")
        return QdrantClient(url=clean_url, api_key=api_key, timeout=timeout)
    else:
        path = Path(storage_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"Using local embedded Qdrant storage at: '{path.resolve()}'")
        return QdrantClient(path=str(path))


def setup_collection(
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    vector_dim: int = 384,
    recreate: bool = False,
):
    """
    Creates or recreates the Qdrant collection with vector configuration
    and payload indices for fast filtering and BM25 full-text search.
    """
    collections = [c.name for c in client.get_collections().collections]

    if collection_name in collections:
        if recreate:
            print(f"Recreating collection '{collection_name}'...")
            client.delete_collection(collection_name)
        else:
            print(f"Collection '{collection_name}' already exists. Skipping creation.")
            return

    print(f"Creating collection '{collection_name}' (vector_size={vector_dim}, distance=COSINE)...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=vector_dim,
            distance=qmodels.Distance.COSINE,
        ),
    )

    # 1. Full-text index on content (for BM25 keyword search in Phase 4)
    print("Creating full-text payload index on 'content'...")
    client.create_payload_index(
        collection_name=collection_name,
        field_name="content",
        field_schema=qmodels.TextIndexParams(
            type="text",
            tokenizer=qmodels.TokenizerType.WORD,
            lowercase=True,
        ),
    )

    # 2. Keyword indices for metadata filtering
    print("Creating keyword payload indices on 'source_lib' and 'file_path'...")
    client.create_payload_index(
        collection_name=collection_name,
        field_name="source_lib",
        field_schema=qmodels.KeywordIndexParams(type="keyword"),
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="file_path",
        field_schema=qmodels.KeywordIndexParams(type="keyword"),
    )
    print("Collection setup complete.")


def load_chunks(chunks_path: Union[str, Path] = "src/data/processed/chunks.json") -> List[Dict[str, Any]]:
    """Loads processed chunks from JSON file or runs chunker if missing."""
    p = Path(chunks_path)
    if not p.exists():
        alt = Path("data/processed/chunks.json")
        if alt.exists():
            p = alt
        else:
            print(f"Chunks file not found at '{p}'. Ingesting raw docs...")
            from src.ingestion.chunker import chunk, load_docs
            docs = load_docs()
            chunk_objs = chunk(docs)
            chunks_data = [c.model_dump() if hasattr(c, "model_dump") else c.dict() for c in chunk_objs]
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            return chunks_data

    with open(p, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks


def ingest_chunks_to_qdrant(
    client: Optional[QdrantClient] = None,
    chunks_path: str = "src/data/processed/chunks.json",
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = 128,
    limit: Optional[int] = None,
    recreate: bool = False,
    embedder: Optional[ONNXEmbedder] = None,
):
    """
    Main ingestion function:
    1. Connects to Qdrant (Cloud or Embedded).
    2. Initializes ONNXEmbedder.
    3. Sets up collection and indices.
    4. Encodes chunks and upserts points in batches.
    5. Verifies indexing.
    """
    start_time = time.time()

    if client is None:
        client = get_qdrant_client()

    if embedder is None:
        print("Initializing ONNX Embedder (BAAI/bge-small-en-v1.5)...")
        embedder = ONNXEmbedder()

    # Setup collection
    setup_collection(
        client=client,
        collection_name=collection_name,
        vector_dim=embedder.dimension,
        recreate=recreate,
    )

    # Load chunks
    chunks = load_chunks(chunks_path)
    if limit and limit > 0:
        print(f"Limiting ingestion to first {limit} chunks (out of {len(chunks)} total).")
        chunks = chunks[:limit]
    else:
        print(f"Loaded {len(chunks)} chunks for ingestion.")

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    print(f"Ingesting into Qdrant collection '{collection_name}' in {total_batches} batches...")

    for i in tqdm(range(0, len(chunks), batch_size), total=total_batches, desc="Ingesting to Qdrant"):
        batch_chunks = chunks[i : i + batch_size]
        texts = [c.get("content", "") for c in batch_chunks]

        # 1. Generate dense ONNX embeddings
        vectors = embedder.embed_documents(texts, batch_size=batch_size)

        # 2. Build PointStruct list with deterministic UUIDs
        points = []
        for chunk_data, vector in zip(batch_chunks, vectors):
            chunk_id = chunk_data.get("chunk_id", "")
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

            point = qmodels.PointStruct(
                id=point_uuid,
                vector=vector,
                payload=chunk_data,
            )
            points.append(point)

        # 3. Upsert to Qdrant
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(" Qdrant Ingestion Complete!")
    print("=" * 60)
    print(f"Collection:            {collection_name}")
    print(f"Total Chunks Ingested: {len(chunks)}")
    print(f"Elapsed Time:          {elapsed:.2f}s")
    if len(chunks) > 0:
        print(f"Throughput:            {len(chunks) / elapsed:.1f} chunks/sec")
    print("=" * 60)

    # Verification: Test query
    verify_indexing(client, collection_name, embedder)


def verify_indexing(client: QdrantClient, collection_name: str, embedder: ONNXEmbedder):
    """Runs verification tests on the populated collection."""
    print("\nRunning Verification Test...")
    try:
        col_info = client.get_collection(collection_name)
        print(f"✓ Collection points count: {col_info.points_count}")

        test_query = "FastAPI dependency injection"
        query_vec = embedder.embed_query(test_query)

        search_res = client.query_points(
            collection_name=collection_name,
            query=query_vec,
            limit=2,
        )

        print(f"✓ Sample query ('{test_query}') returned {len(search_res.points)} results:")
        for idx, p in enumerate(search_res.points, start=1):
            title = p.payload.get("section_title", "N/A")
            lib = p.payload.get("source_lib", "N/A")
            score = getattr(p, "score", 0.0)
            print(f"   [{idx}] Score: {score:.4f} | Lib: {lib:<10} | Section: {title}")
        print(" Verification successful!\n")
    except Exception as e:
        print(f"⚠ Verification warning: {e}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest documentation chunks into Qdrant Cloud or Embedded storage using ONNX embeddings."
    )
    parser.add_argument(
        "--chunks-path",
        type=str,
        default="src/data/processed/chunks.json",
        help="Path to processed chunks JSON file.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION_NAME}).",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=DEFAULT_QDRANT_URL,
        help="Qdrant server / cloud URL (defaults to env var Qdrant_API_URL).",
    )
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=DEFAULT_QDRANT_API_KEY,
        help="Qdrant API key (defaults to env var Qdrant_API_KEY).",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default=DEFAULT_STORAGE_PATH,
        help="Local embedded storage path if no remote URL is used.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for ONNX embedding & Qdrant upsert (default: 128).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of chunks to ingest (useful for quick testing).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the Qdrant collection if it already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    client = get_qdrant_client(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        storage_path=args.storage_path,
    )
    ingest_chunks_to_qdrant(
        client=client,
        chunks_path=args.chunks_path,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
        limit=args.limit,
        recreate=args.recreate,
    )


if __name__ == "__main__":
    main()
