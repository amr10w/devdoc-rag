"""
src/ingestion/ingest_qdrant.py

Ingests documentation chunks into Qdrant (Cloud or Local Embedded).
Sets up vector configurations, payload indices (full-text and keyword), and uploads points in batches.
Embedder is dynamically selected based on the EMBEDDING_PROVIDER environment variable or CLI argument.
"""

import argparse
import hashlib
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

from src.embeddings.embedder import get_embedder

# 1. Environment Loading: Check src/.env, root .env, or find_dotenv()
load_dotenv(find_dotenv())

# Read Qdrant credentials with flexible fallback
DEFAULT_QDRANT_URL = os.getenv("QDRANT_API_URL", None)
DEFAULT_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
DEFAULT_STORAGE_PATH = os.getenv("QDRANT_STORAGE_PATH", "src/data/qdrant_storage")
DEFAULT_COLLECTION_NAME = "devdoc_chunks"

# Embedding and Vector Configuration
DEFAULT_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "onnx")
DEFAULT_VECTOR_DIMENSION = int(
    os.getenv("VECTOR_DIMENSION")
    or os.getenv("EMBEDDING_DIM")
    or os.getenv("VECTOR_DIM")
    or 384
)


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
    vector_dim: int = DEFAULT_VECTOR_DIMENSION,
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

    # 1. Full-text index on content (for BM25 keyword search)
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


def _point_id_for_chunk(chunk_data: Dict[str, Any]) -> str:
    """Derives a deterministic point UUID for a chunk.

    Falls back to hashing the chunk's content (plus file_path, if present)
    when `chunk_id` is missing, so that chunks without an explicit ID don't
    all collide into the same UUID and silently overwrite each other.
    """
    chunk_id = chunk_data.get("chunk_id")
    if chunk_id:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk_id)))

    fallback_key = f"{chunk_data.get('file_path', '')}::{chunk_data.get('content', '')}"
    digest = hashlib.sha256(fallback_key.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, digest))


def ingest_chunks_to_qdrant(
    client: Optional[QdrantClient] = None,
    chunks_path: str = "src/data/processed/chunks.json",
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = 128,
    limit: Optional[int] = None,
    offset: int = 0,
    recreate: bool = False,
    embedding_provider: Optional[str] = None,
    vector_dim: Optional[int] = None,
    embedder: Optional[Any] = None,
):
    """
    Main ingestion function:
    1. Connects to Qdrant (Cloud or Embedded).
    2. Initializes embedder based on provider or EMBEDDING_PROVIDER env var.
    3. Sets up collection and indices.
    4. Encodes chunks into vectors and upserts points in batches.
    5. Verifies indexing.
    """
    start_time = time.time()

    if client is None:
        client = get_qdrant_client()

    if embedder is None:
        provider = embedding_provider or DEFAULT_EMBEDDING_PROVIDER
        embedder = get_embedder(provider=provider)

    if vector_dim is None:
        vector_dim = getattr(embedder, "dimension", DEFAULT_VECTOR_DIMENSION)

    # Setup collection
    setup_collection(
        client=client,
        collection_name=collection_name,
        vector_dim=vector_dim,
        recreate=recreate,
    )

    # Load chunks
    chunks = load_chunks(chunks_path)
    total_loaded = len(chunks)
    if offset and offset > 0:
        print(f"Skipping first {offset} chunks (resuming from chunk index {offset}/{total_loaded}).")
        chunks = chunks[offset:]

    if limit and limit > 0:
        print(f"Limiting ingestion to first {limit} chunks (out of {len(chunks)} remaining).")
        chunks = chunks[:limit]
    else:
        print(f"Loaded {len(chunks)} chunks to ingest.")

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    initial_batch = (offset // batch_size) if offset > 0 else 0
    overall_total_batches = initial_batch + total_batches
    print(
        f"Ingesting into Qdrant collection '{collection_name}' in {total_batches} batches "
        f"(batches {initial_batch + 1} to {overall_total_batches}) using {type(embedder).__name__} (dim={vector_dim})..."
    )

    for i in tqdm(
        range(0, len(chunks), batch_size),
        total=overall_total_batches,
        initial=initial_batch,
        desc="Ingesting to Qdrant",
    ):
        batch_chunks = chunks[i : i + batch_size]
        texts = [c.get("content", "") for c in batch_chunks]

        # 1. Generate dense embeddings
        vectors = embedder.embed_documents(texts, batch_size=len(texts))

        # 2. Build PointStruct list with deterministic UUIDs
        points = []
        for chunk_data, vector in zip(batch_chunks, vectors):
            point_uuid = _point_id_for_chunk(chunk_data)

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
    print(f"Vector Dimension:      {vector_dim}")
    print(f"Embedder:              {type(embedder).__name__}")
    chunks_info = f"{len(chunks)}" if offset == 0 else f"{len(chunks)} (resumed from chunk {offset})"
    print(f"Total Chunks Ingested: {chunks_info}")
    print(f"Elapsed Time:          {elapsed:.2f}s")
    if len(chunks) > 0 and elapsed > 0:
        print(f"Throughput:            {len(chunks) / elapsed:.1f} chunks/sec")
    print("=" * 60)

    # Verification: Test query
    verify_indexing(client, collection_name, embedder)


def verify_indexing(client: QdrantClient, collection_name: str, embedder: Any):
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
        description="Ingest documentation chunks into Qdrant Cloud or Embedded storage."
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
        help="Qdrant server / cloud URL (defaults to env var QDRANT_API_URL; omit for local embedded storage).",
    )
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=DEFAULT_QDRANT_API_KEY,
        help="Qdrant API key (defaults to env var QDRANT_API_KEY).",
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
        help="Batch size for embedding & Qdrant upsert (default: 128).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of chunks to skip before starting ingestion (default: 0).",
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=None,
        help="Batch index to resume from (e.g. 104). Automatically calculates offset = start_batch * batch_size.",
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
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default=DEFAULT_EMBEDDING_PROVIDER,
        help=f"Embedding provider: 'onnx' or 'ollama' (default: '{DEFAULT_EMBEDDING_PROVIDER}', from env EMBEDDING_PROVIDER).",
    )
    parser.add_argument(
        "--vector-dim",
        type=int,
        default=None,
        help=f"Vector dimension for Qdrant collection (defaults to env VECTOR_DIMENSION or embedder dimension).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    offset = args.offset
    if args.start_batch is not None:
        offset = args.start_batch * args.batch_size

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
        offset=offset,
        recreate=args.recreate,
        embedding_provider=args.embedding_provider,
        vector_dim=args.vector_dim,
    )


if __name__ == "__main__":
    main()