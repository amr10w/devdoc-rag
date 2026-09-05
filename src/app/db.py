"""
src/app/db.py

SQLite logging and telemetry engine for DevDoc RAG.
Stores query logs, retrieved chunk metadata, generation latencies, and user feedback (thumbs up/down).
Includes lifespan auto-seeding of 35 diverse mock records when the database is empty.
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = os.getenv("RAG_LOGS_DB", "src/data/rag_logs.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Creates the tables if they don't exist."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_logs (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    rewritten_query TEXT,
                    source_lib TEXT,
                    retrieval_method TEXT,
                    retrieved_chunks TEXT,
                    response TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    char_count INTEGER NOT NULL,
                    word_count INTEGER NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_logs (
                    id TEXT PRIMARY KEY,
                    query_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    comment TEXT,
                    FOREIGN KEY (query_id) REFERENCES query_logs (id)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_ts ON query_logs(timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_qid ON feedback_logs(query_id);")
    finally:
        conn.close()


def log_query(
    query: str,
    response: str,
    latency_ms: float,
    retrieved_chunks: list[dict[str, Any]],
    source_lib: str | None = None,
    retrieval_method: str = "hybrid_rrf",
    rewritten_query: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Inserts a new query log entry and returns the generated log ID."""
    log_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    chunks_json = json.dumps(
        [
            {
                "chunk_id": c.get("chunk_id", ""),
                "score": c.get("score", 0.0),
                "source_lib": c.get("source_lib", ""),
                "file_path": c.get("file_path", ""),
                "section_title": c.get("section_title", ""),
            }
            for c in retrieved_chunks
        ]
    )

    char_count = len(response)
    word_count = len(response.split())

    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO query_logs (
                    id, timestamp, query, rewritten_query, source_lib,
                    retrieval_method, retrieved_chunks, response,
                    latency_ms, char_count, word_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    log_id,
                    ts,
                    query,
                    rewritten_query,
                    source_lib or "all",
                    retrieval_method,
                    chunks_json,
                    response,
                    latency_ms,
                    char_count,
                    word_count,
                ),
            )
    finally:
        conn.close()

    return log_id


def log_feedback(
    query_id: str,
    rating: int,
    comment: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Records user feedback (+1 or -1) associated with a query log ID."""
    feedback_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO feedback_logs (id, query_id, timestamp, rating, comment)
                VALUES (?, ?, ?, ?, ?);
                """,
                (feedback_id, query_id, ts, rating, comment or ""),
            )
    finally:
        conn.close()

    return feedback_id


def get_recent_logs(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Fetches recent query logs joined with any available feedback."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                q.id, q.timestamp, q.query, q.rewritten_query, q.source_lib,
                q.retrieval_method, q.retrieved_chunks, q.response,
                q.latency_ms, q.char_count, q.word_count,
                f.rating as feedback_rating, f.comment as feedback_comment
            FROM query_logs q
            LEFT JOIN feedback_logs f ON q.id = f.query_id
            ORDER BY q.timestamp DESC
            LIMIT ?;
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["retrieved_chunks"] = json.loads(d["retrieved_chunks"])
            except Exception:
                d["retrieved_chunks"] = []
            result.append(d)
        return result
    finally:
        conn.close()


def get_analytics_summary(db_path: str = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Computes aggregated metrics for the 5 dashboard figures."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()

        # 1. Total counts
        cursor.execute("SELECT COUNT(*) FROM query_logs;")
        total_queries = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedback_logs;")
        total_feedback = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedback_logs WHERE rating > 0;")
        positive_feedback = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedback_logs WHERE rating < 0;")
        negative_feedback = cursor.fetchone()[0]

        # 2. Latency percentiles & averages
        cursor.execute("SELECT AVG(latency_ms) FROM query_logs;")
        avg_latency = cursor.fetchone()[0] or 0.0

        cursor.execute("SELECT latency_ms FROM query_logs ORDER BY latency_ms ASC;")
        latencies = [row[0] for row in cursor.fetchall()]
        p50_lat = latencies[len(latencies) // 2] if latencies else 0.0
        p90_lat = latencies[int(len(latencies) * 0.9)] if latencies else 0.0
        p99_lat = latencies[int(len(latencies) * 0.99)] if latencies else 0.0

        # 3. Response Length metrics
        cursor.execute("SELECT AVG(char_count), AVG(word_count) FROM query_logs;")
        avg_chars, avg_words = cursor.fetchone()

        # 4. Library distribution
        cursor.execute("SELECT source_lib, COUNT(*) FROM query_logs GROUP BY source_lib;")
        lib_counts = {row[0]: row[1] for row in cursor.fetchall()}

        return {
            "total_queries": total_queries,
            "total_feedback": total_feedback,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "satisfaction_rate": (positive_feedback / total_feedback * 100.0) if total_feedback > 0 else 100.0,
            "avg_latency_ms": round(avg_latency, 2),
            "p50_latency_ms": round(p50_lat, 2),
            "p90_latency_ms": round(p90_lat, 2),
            "p99_latency_ms": round(p99_lat, 2),
            "avg_chars": round(avg_chars or 0, 1),
            "avg_words": round(avg_words or 0, 1),
            "lib_counts": lib_counts,
        }
    finally:
        conn.close()


def seed_mock_logs_if_empty(db_path: str = DEFAULT_DB_PATH, count: int = 35) -> bool:
    """
    Checks if query_logs is empty on startup.
    If 0 records, seeds count diverse realistic query & feedback entries (Rule 2).
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM query_logs;")
        existing_count = cursor.fetchone()[0]
        if existing_count > 0:
            return False

        print(f"Database empty. Seeding {count} mock telemetry logs for monitoring dashboard...")

        mock_topics = [
            ("fastapi", "How do I create a global dependency in FastAPI?", "Declare it in FastAPI(dependencies=[Depends(func)])."),
            ("fastapi", "What is the difference between path and query parameters?", "Path parameters are defined in the path string, query parameters are function args."),
            ("fastapi", "How do background tasks work in FastAPI?", "Use BackgroundTasks parameter and call tasks.add_task(func, *args)."),
            ("docker", "How to create a multi-stage Dockerfile for Python?", "Use multiple FROM statements to separate build stage from lightweight runtime."),
            ("docker", "What command inspects container network details?", "Run docker inspect <container_id> or docker network inspect <network_name>."),
            ("docker", "How do Docker volumes differ from bind mounts?", "Volumes are managed by Docker in /var/lib/docker, bind mounts map host paths."),
            ("pytorch", "How to move model and tensors to CUDA device?", "Use model.to(device) and tensor.to(device) where device='cuda'."),
            ("pytorch", "What does torch.no_grad() do during inference?", "Disables autograd engine gradient calculation to reduce memory consumption."),
            ("pydantic", "How to define optional fields in Pydantic V2?", "Use Optional[T] = None or T | None = None with default."),
            ("pydantic", "What is field_validator vs model_validator in Pydantic?", "field_validator validates a single attribute; model_validator validates entire model state."),
            ("qdrant", "How does Qdrant handle payload filtering?", "Qdrant executes payload filtering alongside HNSW graph traversal for fast vector filtering."),
            ("transformers", "How do I load a HuggingFace model in 4-bit precision?", "Use BitsAndBytesConfig(load_in_4bit=True) with AutoModelForCausalLM."),
        ]

        now = datetime.now(timezone.utc)
        random.seed(42)

        for i in range(count):
            topic = random.choice(mock_topics)
            lib, query_text, ans_snippet = topic
            days_ago = random.uniform(0.1, 7.0)
            entry_time = (now - timedelta(days=days_ago, hours=random.uniform(0, 12))).isoformat()
            latency = round(random.uniform(220.0, 1850.0), 2)
            log_id = str(uuid.uuid4())

            fake_chunks = [
                {
                    "chunk_id": f"{lib}/docs/section_{random.randint(1, 10)}.md#chunk-{random.randint(0, 5)}",
                    "score": round(random.uniform(0.72, 0.94), 4),
                    "source_lib": lib,
                    "file_path": f"{lib}/docs/section_{random.randint(1, 10)}.md",
                    "section_title": f"{lib.capitalize()} Guide",
                }
                for _ in range(3)
            ]

            full_response = f"Based on {lib} documentation:\n\n{ans_snippet}\n\n```python\n# Example for {lib}\ndef example():\n    pass\n```"
            char_cnt = len(full_response)
            word_cnt = len(full_response.split())

            cursor.execute(
                """
                INSERT INTO query_logs (
                    id, timestamp, query, rewritten_query, source_lib,
                    retrieval_method, retrieved_chunks, response,
                    latency_ms, char_count, word_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    log_id,
                    entry_time,
                    query_text,
                    f"Search documentation for: {query_text}",
                    lib,
                    "hybrid_rrf",
                    json.dumps(fake_chunks),
                    full_response,
                    latency,
                    char_cnt,
                    word_cnt,
                ),
            )

            # 70% of queries get feedback
            if random.random() < 0.70:
                rating = 1 if random.random() < 0.85 else -1
                comment = "Helpful code example!" if rating == 1 else "Could be more detailed."
                fb_time = (datetime.fromisoformat(entry_time) + timedelta(minutes=random.randint(1, 15))).isoformat()
                cursor.execute(
                    """
                    INSERT INTO feedback_logs (id, query_id, timestamp, rating, comment)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (str(uuid.uuid4()), log_id, fb_time, rating, comment),
                )

        conn.commit()
        print(f"Successfully seeded {count} mock telemetry records.")
        return True
    finally:
        conn.close()
