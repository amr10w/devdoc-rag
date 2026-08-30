from tqdm.auto import tqdm
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from src.models import ChunkModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_docs(data_dir= "src/data/raw")-> List[Dict[str, str]]:
    """
    Recursively scans all subfolders in data_dir (e.g., docker, fastapi, pytorch)
    and loads every markdown (.md, .mdx) file found at any directory depth.

    Returns:
        List of dicts: [
            {
                "file_path": "docker/get-started/overview.md",
                "source_lib": "docker",
                "content": "..."
            }, ...
        ]
    """

    base_path = Path(data_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Documentation directory '{data_dir}' not found.")

    docs = []
    md_files = list(base_path.rglob("*.md")) + list(base_path.rglob("*.mdx"))
    
    for file_path in tqdm(md_files):
        if any(part.startswith(".") for part in file_path.parts):
                continue
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()

            # Skip completely empty files
            if not content:
                continue

            rel_path = file_path.relative_to(base_path)
            rel_path_str = str(rel_path).replace("\\", "/")

            # Detect the library/tool folder name (e.g. 'docker', 'fastapi')
            source_lib = rel_path.parts[0] if len(rel_path.parts) > 1 else "general"

            docs.append({
                "file_path": rel_path_str,
                "source_lib": source_lib,
                "content": content
            })
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")

    print(f"Loaded {len(docs)} markdown files from '{base_path}'.")
    return docs

def to_dataframe(docs:list[Dict[str,str]])->pd.DataFrame:
    df = pd.DataFrame(docs)
    return df

def chunk(
    docs: List[Dict[str, str]],
    batch_size: int = 50,
    chunk_size: int = 3000,
    chunk_overlap: int = 100,
    min_chunk_len: int = 200
) -> List[ChunkModel]:
    """
    Chunks documents using LangChain's RecursiveCharacterTextSplitter and returns ChunkModel objects.
    """

    separators = [
        "\n# ",
        "\n## ",
        "\n### ",
        "\n#### ",
        "\n\n",
        "\n",
        " ",
        ""
    ]

    splitter= RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False

    )


    chunks: List[ChunkModel] = []

    total_batches = (len(docs) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(docs), batch_size), total=total_batches, desc="Chunking batches"):
        doc_batch = docs[i : i + batch_size]

        for doc in doc_batch:
            file_path = doc["file_path"]
            source_lib = doc["source_lib"]
            content = doc["content"]
            section_title = Path(file_path).stem.replace("-", " ").replace("_", " ").title()

            text_splits = splitter.split_text(content)

            for idx, text_block in enumerate(text_splits):
                clean_text = text_block.strip()
                if len(clean_text) < min_chunk_len:
                    continue

                # Generate deterministic hash for the chunk ID
                chunk_hash = hashlib.md5(f"{file_path}::{clean_text}".encode()).hexdigest()[:10]
                chunk_id = f"{file_path}#chunk-{idx}-{chunk_hash}"

                chunk_obj = ChunkModel(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    source_lib=source_lib,
                    section_title=section_title,
                    chunk_index=idx,
                    content=clean_text,
                    char_count=len(clean_text)
                )
                chunks.append(chunk_obj)

    return chunks


