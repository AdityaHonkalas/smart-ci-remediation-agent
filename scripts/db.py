#!/usr/bin/env python3
"""SQLite-backed vector store for CI log retrieval.

This module intentionally avoids third-party dependencies. It uses a hashed
bag-of-tokens embedding so the pipeline works offline, while keeping the same
document/upsert/search shape you can later swap for Chroma, FAISS, pgvector, or
remote embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "vector_store.sqlite"
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]{1,}")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class VectorSearchResult:
    document_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class LocalVectorDB:
    """Small local vector database backed by SQLite."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH, dimensions: int = 512) -> None:
        self.db_path = Path(db_path)
        self.dimensions = dimensions
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.connection.close()

    def clear_collection(self, collection: str) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM documents WHERE collection = ?", (collection,))

    def upsert_document(
        self,
        document_id: str,
        text: str,
        metadata: dict[str, Any],
        collection: str = "gha_logs",
    ) -> None:
        embedding = self.embed(text)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO documents (
                    id, collection, text, metadata_json, embedding_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    collection = excluded.collection,
                    text = excluded.text,
                    metadata_json = excluded.metadata_json,
                    embedding_json = excluded.embedding_json,
                    updated_at = excluded.updated_at
                """,
                (
                    document_id,
                    collection,
                    text,
                    json.dumps(metadata, sort_keys=True),
                    json.dumps(embedding, sort_keys=True),
                    utc_now(),
                ),
            )

    def upsert_documents(
        self,
        documents: Iterable[dict[str, Any]],
        collection: str = "gha_logs",
    ) -> int:
        count = 0
        for document in documents:
            self.upsert_document(
                document_id=document["document_id"],
                text=document["text"],
                metadata=document.get("metadata", {}),
                collection=collection,
            )
            count += 1
        return count

    def search(
        self,
        query: str,
        top_k: int = 5,
        collection: str = "gha_logs",
    ) -> list[VectorSearchResult]:
        query_embedding = self.embed(query)
        if not query_embedding:
            return []

        rows = self.connection.execute(
            "SELECT id, text, metadata_json, embedding_json FROM documents WHERE collection = ?",
            (collection,),
        ).fetchall()

        results: list[VectorSearchResult] = []
        for row in rows:
            embedding = json.loads(row["embedding_json"])
            score = cosine_similarity(query_embedding, embedding)
            if score <= 0:
                continue
            results.append(
                VectorSearchResult(
                    document_id=row["id"],
                    score=score,
                    text=row["text"],
                    metadata=json.loads(row["metadata_json"]),
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def embed(self, text: str) -> dict[str, float]:
        vector: dict[int, float] = {}
        for token in TOKEN_RE.findall(text.lower()):
            index = stable_hash(token) % self.dimensions
            vector[index] = vector.get(index, 0.0) + 1.0

        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return {}
        return {str(index): value / norm for index, value in sorted(vector.items())}

    def _init_schema(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    collection TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection)"
            )


def stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the local CI log vector store.")
    parser.add_argument("query", nargs="?", help="Search query.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite vector store path.")
    parser.add_argument("--collection", default="gha_logs", help="Collection name.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.query:
        print(f"Vector store path: {args.db}")
        print("Pass a query to search indexed log excerpts.")
        return 0

    db = LocalVectorDB(args.db)
    try:
        results = db.search(args.query, top_k=args.top_k, collection=args.collection)
    finally:
        db.close()

    for result in results:
        metadata = result.metadata
        title = " / ".join(
            str(value)
            for value in (
                metadata.get("workflow_name"),
                metadata.get("job_name"),
                metadata.get("error_type"),
                metadata.get("error_code"),
            )
            if value
        )
        print(f"{result.score:.3f} {result.document_id} {title}")
        print(result.text[:500].replace("\n", " "))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
