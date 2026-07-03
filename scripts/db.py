#!/usr/bin/env python3
"""Self-Learning Knowledge Base backed by ChromaDB for CI failure RCA.

Replaces the original SQLite-backed LocalVectorDB with a production-grade
ChromaDB vector store.  The knowledge base grows organically with every RCA
performed — no offline training phase is required.

=============================================================================
Entity / Schema Design
=============================================================================

Collection 1 — ci_failure_logs
────────────────────────────────
Purpose   : Stores chunked raw CI log text for semantic similarity retrieval.
Embedding : sentence-transformers/all-MiniLM-L6-v2  (384-dim, L2-normalised)

Document attributes (ChromaDB metadata per vector):
  document_id    TEXT  PK  Stable SHA-256 chunk / signal ID
  doc_kind       TEXT      'log_chunk' | 'error_context' | 'failure_block'
  repository     TEXT      e.g. 'kubernetes/kubernetes'
  run_id         TEXT      GitHub Actions run ID
  run_attempt    INT       Attempt number
  workflow_name  TEXT      Workflow display name
  job_name       TEXT      Job display name
  file_name      TEXT      Source log filename inside the zip archive
  start_line     INT       First log line in this chunk
  end_line       INT       Last log line in this chunk
  error_type     TEXT      e.g. 'dependency_error', 'permission_error'
  error_code     TEXT      e.g. 'EXIT_1', 'HTTP_403'
  severity       TEXT      'high' | 'medium' | 'low'
  status         TEXT      Run conclusion (e.g. 'failure')
  html_url       TEXT      GitHub Actions run URL
  commit_sha     TEXT      HEAD commit SHA
  branch         TEXT      Branch name
  updated_at     TEXT      ISO-8601 UTC timestamp of last upsert

Collection 2 — rca_knowledge_base
───────────────────────────────────
Purpose   : Stores (error_signature → RCA) pairs — the self-learning store.
Embedding : Error signature text embedded via the same MiniLM model.

Document attributes:
  document_id     TEXT  PK  error_signature string used as ChromaDB ID
  error_signature TEXT      Human-readable error signature
  error_type      TEXT      Classified error type
  failure_stage   TEXT      Pipeline stage where error occurred
  source          TEXT      'knowledge_base' | 'web_search' | 'model'
  generated_at    TEXT      ISO-8601 UTC timestamp of RCA generation

Key-value sidecar (rca_repository.json):
  Maps error_signature → full RCA payload including:
    rca_summary, error_type, confidence, failure_stage, source, evidence,
    generated_at, hit_count

Self-Learning Loop
───────────────────
  CI failure
      │
      ▼
  RCA Agent generates RCA
      │
      ▼
  kb.update(error_signature, rca_payload)      ← auto-called after every RCA
      │  ├── embeds signature (MiniLM 384-dim)
      │  ├── upserts into ChromaDB collection
      │  └── persists to JSON sidecar (atomic tmp-rename)
      ▼
  Next similar failure → kb.search() returns this RCA as high-similarity hit
  (no model call needed if similarity ≥ KB_CONFIDENCE_THRESHOLD = 0.70)
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
DEFAULT_CHROMA_PATH = Path(__file__).resolve().parents[1] / "data" / "chroma_store"
DEFAULT_RCA_KB_PATH = Path(__file__).resolve().parents[1] / "data" / "rca_knowledge_base"

LOG_COLLECTION = "ci_failure_logs"
RCA_COLLECTION = "rca_knowledge_base"

# Embedding model — matches pre-processing pipeline
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DIM = 384


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# Confidence labels written by heuristic / agent fallback paths
_CONFIDENCE_LABEL_MAP: dict[str, float] = {
    "low":    0.25,
    "medium": 0.50,
    "high":   0.85,
}


def _coerce_confidence(value: Any) -> float:
    """
    Coerce a confidence value to float.

    Handles three cases that appear in rca_repository.json entries:
      • Already a float / int  → cast to float directly.
      • A numeric string       → parse with float().
      • A label string         → map via _CONFIDENCE_LABEL_MAP
                                  ('low' → 0.25, 'medium' → 0.50, 'high' → 0.85).
    Returns 0.0 for any unrecognised value rather than raising.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        label = value.strip().lower()
        if label in _CONFIDENCE_LABEL_MAP:
            return _CONFIDENCE_LABEL_MAP[label]
        try:
            return float(label)
        except ValueError:
            logger.warning("Unrecognised confidence value %r — defaulting to 0.0", value)
            return 0.0
    return 0.0


def _coerce_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """
    ChromaDB only accepts str / int / float / bool metadata values.
    Convert everything else to str and drop None values.
    """
    safe: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VectorSearchResult:
    """Single result returned by a ChromaDB similarity search."""
    document_id: str
    score: float          # cosine similarity in [0, 1]  (1 = identical)
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RCAEntry:
    """A (error_signature, RCA) pair retrieved from the self-learning KB."""
    error_signature: str
    rca_summary:     str
    error_type:      str
    confidence:      float
    source:          str
    failure_stage:   str
    generated_at:    str
    similarity:      float   # cosine similarity to query in [0, 1]
    hit_count:       int = 0


# ---------------------------------------------------------------------------
# Lazy singleton embedder (sentence-transformers/all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------

class _Embedder:
    """
    Thin wrapper around sentence-transformers loaded on first use.

    Produces L2-normalised 384-dimensional vectors for cosine similarity.
    Raises RuntimeError with an actionable message when the library is absent.
    """

    _instance: "_Embedder | None" = None

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
            self._model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded: %s", EMBEDDING_MODEL)
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for ChromaDB embeddings. "
                "Install it with:  pip install sentence-transformers"
            ) from exc

    @classmethod
    def get(cls) -> "_Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return a list of L2-normalised 384-dim float vectors."""
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist() if hasattr(vecs, "tolist") else [v.tolist() for v in vecs]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# ChromaDB vector store — Collection: ci_failure_logs
# ---------------------------------------------------------------------------

class ChromaVectorDB:
    """
    ChromaDB-backed vector store for CI log chunks and error signals.

    Drop-in replacement for the original SQLite LocalVectorDB:
    same public interface — upsert_document / upsert_documents / search /
    clear_collection / close — so existing callers need no changes.

    Vector space: cosine, 384 dimensions (all-MiniLM-L6-v2).
    """

    def __init__(
        self,
        persist_dir: Path | str = DEFAULT_CHROMA_PATH,
        collection_name: str = LOG_COLLECTION,
    ) -> None:
        try:
            import chromadb  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is required. Install it with:  pip install chromadb"
            ) from exc

        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaVectorDB ready — collection=%s  path=%s  count=%d",
            collection_name, self._persist_dir, self._collection.count(),
        )

    # ------------------------------------------------------------------
    def close(self) -> None:
        """No-op: ChromaDB persists automatically. Kept for API compatibility."""

    def clear_collection(self, collection: str | None = None) -> None:
        name = collection or self._collection_name
        try:
            self._client.delete_collection(name)
        except Exception:  # noqa: BLE001
            pass
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    # ------------------------------------------------------------------
    def upsert_document(
        self,
        document_id: str,
        text: str,
        metadata: dict[str, Any],
        collection: str = LOG_COLLECTION,
    ) -> None:
        embedding = _Embedder.get().embed_one(text)
        col = self._get_collection(collection)
        col.upsert(
            ids=[document_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[_coerce_meta({**metadata, "updated_at": utc_now()})],
        )

    def upsert_documents(
        self,
        documents: Iterable[dict[str, Any]],
        collection: str = LOG_COLLECTION,
        batch_size: int = 64,
    ) -> int:
        col = self._get_collection(collection)
        embedder = _Embedder.get()

        ids:   list[str]           = []
        texts: list[str]           = []
        metas: list[dict[str, Any]] = []
        count = 0

        def _flush() -> None:
            if not ids:
                return
            vecs = embedder.embed(texts)
            col.upsert(ids=ids, documents=texts, embeddings=vecs, metadatas=metas)
            ids.clear(); texts.clear(); metas.clear()

        for doc in documents:
            ids.append(doc["document_id"])
            texts.append(doc["text"])
            metas.append(_coerce_meta({**doc.get("metadata", {}), "updated_at": utc_now()}))
            count += 1
            if len(ids) >= batch_size:
                _flush()
        _flush()
        return count

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        collection: str = LOG_COLLECTION,
        where: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        col = self._get_collection(collection)
        if col.count() == 0:
            return []

        query_vec = _Embedder.get().embed_one(query)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vec],
            "n_results":        min(top_k, col.count()),
            "include":          ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = col.query(**kwargs)

        output: list[VectorSearchResult] = []
        for doc_id, text, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            # ChromaDB cosine space: distance ∈ [0, 2]  →  similarity = 1 - dist/2
            similarity = max(0.0, 1.0 - dist / 2.0)
            output.append(VectorSearchResult(
                document_id=doc_id,
                score=round(similarity, 4),
                text=text,
                metadata=meta,
            ))

        output.sort(key=lambda r: r.score, reverse=True)
        return output

    # ------------------------------------------------------------------
    def _get_collection(self, name: str):  # type: ignore[return]
        if name == self._collection_name:
            return self._collection
        return self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )


# ---------------------------------------------------------------------------
# Self-Learning Knowledge Base — Collection: rca_knowledge_base
# ---------------------------------------------------------------------------

class SelfLearningKnowledgeBase:
    """
    Self-learning vector knowledge base for (error_signature → RCA) pairs.

    Architecture
    ─────────────
    Vector store  : ChromaDB collection 'rca_knowledge_base'
                    Stores error-signature embeddings for fast similarity search.
    Key-value sidecar : rca_repository.json
                    Maps each error_signature → full RCA payload dict.

    Self-Learning Loop
    ───────────────────
    1. RCA Agent generates a new RCA for an incoming CI failure.
    2. Agent calls  kb.update(error_signature, rca_payload)
    3. KB embeds the signature → upserts into ChromaDB.
    4. KB persists the full RCA to the JSON sidecar (atomic write).
    5. Next time a similar failure arrives:
         kb.search(error_signature) returns the historical RCA with a
         similarity score.  If score ≥ KB_CONFIDENCE_THRESHOLD (0.70) the
         RCA Agent uses it directly — no model call needed.

    Key Features
    ─────────────
    • No Offline Phase : KB starts empty, grows with every RCA generation.
    • Automatic Updates: Every RCA generation triggers kb.update().
    • Vector Similarity: Fast cosine-similarity retrieval over embeddings.
    • Key-Value Store  : Efficient (error_signature → full RCA) mapping.
    • Hit Tracking     : hit_count incremented on every retrieval.
    """

    def __init__(
        self,
        persist_dir: Path | str = DEFAULT_RCA_KB_PATH,
        confidence_threshold: float = 0.70,
    ) -> None:
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        try:
            import chromadb  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is required. Install it with:  pip install chromadb"
            ) from exc

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=RCA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._kv_path = persist_dir / "rca_repository.json"
        self._rca_repository: dict[str, dict[str, Any]] = self._load_kv()
        self.confidence_threshold = confidence_threshold

        logger.info(
            "SelfLearningKnowledgeBase ready — %d entries  threshold=%.2f",
            len(self._rca_repository), confidence_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        error_signature: str,
        top_k: int = 5,
    ) -> list[RCAEntry]:
        """
        Return the top-k most similar historical (error, RCA) pairs.

        Parameters
        ----------
        error_signature : str
            Human-readable or hashed error signature produced by the
            preprocessing pipeline (e.g. 'ImportError_module_not_found_sklearn').
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[RCAEntry]
            Ordered by descending cosine similarity.  Empty list when the KB
            has no entries yet.
        """
        if self._collection.count() == 0:
            return []

        query_vec = _Embedder.get().embed_one(error_signature)
        n = min(top_k, self._collection.count())
        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=n,
            include=["metadatas", "distances"],
        )

        entries: list[RCAEntry] = []
        for meta, dist in zip(result["metadatas"][0], result["distances"][0]):
            sig  = str(meta.get("error_signature", ""))
            sim  = max(0.0, 1.0 - dist / 2.0)
            rca  = self._rca_repository.get(sig)
            if not rca:
                continue
            entries.append(RCAEntry(
                error_signature=sig,
                rca_summary    =str(rca.get("rca_summary", "")),
                error_type     =str(rca.get("error_type", "")),
                confidence     =_coerce_confidence(rca.get("confidence", 0.0)),
                source         =str(rca.get("source", "knowledge_base")),
                failure_stage  =str(rca.get("failure_stage", "")),
                generated_at   =str(rca.get("generated_at", "")),
                similarity     =round(sim, 4),
                hit_count      =int(rca.get("hit_count", 0)),
            ))

        entries.sort(key=lambda e: e.similarity, reverse=True)

        # Increment hit counts for returned entries
        for entry in entries:
            stored = self._rca_repository.get(entry.error_signature)
            if stored:
                stored["hit_count"] = stored.get("hit_count", 0) + 1
        if entries:
            self._save_kv()

        return entries

    def update(
        self,
        error_signature: str,
        rca_payload: dict[str, Any],
    ) -> None:
        """
        Persist a new (error_signature → RCA) pair to the knowledge base.

        Called automatically by the RCA Agent after every successful RCA
        generation so the KB grows with usage.

        Parameters
        ----------
        error_signature : str
            Human-readable error signature from the preprocessing pipeline.
        rca_payload : dict
            RCA result dict.  Must contain at minimum:
            rca_summary, error_type, confidence, failure_stage, source.
            May also contain evidence and inline_fix_suggestions.
        """
        embedding = _Embedder.get().embed_one(error_signature)
        self._collection.upsert(
            ids=[error_signature],
            embeddings=[embedding],
            documents=[error_signature],
            metadatas=[_coerce_meta({
                "error_signature": error_signature,
                "error_type":      rca_payload.get("error_type", ""),
                "failure_stage":   rca_payload.get("failure_stage", ""),
                "source":          rca_payload.get("source", "knowledge_base"),
                "generated_at":    utc_now(),
            })],
        )
        # Preserve existing hit_count when overwriting an entry
        existing = self._rca_repository.get(error_signature, {})
        self._rca_repository[error_signature] = {
            "rca_summary":           rca_payload.get("rca_summary", ""),
            "error_type":            rca_payload.get("error_type", ""),
            "confidence":            rca_payload.get("confidence", 0.0),
            "failure_stage":         rca_payload.get("failure_stage", ""),
            "source":                rca_payload.get("source", "knowledge_base"),
            "evidence":              rca_payload.get("evidence", []),
            "inline_fix_suggestions":rca_payload.get("inline_fix_suggestions", []),
            "generated_at":          utc_now(),
            "hit_count":             existing.get("hit_count", 0),
        }
        self._save_kv()
        logger.debug("KB updated — signature=%s", error_signature)

    def best_confidence(self, entries: list[RCAEntry]) -> float:
        """Return the highest similarity score from a search result list."""
        return max((e.similarity for e in entries), default=0.0)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_kv(self) -> dict[str, dict[str, Any]]:
        if self._kv_path.exists():
            try:
                with self._kv_path.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load RCA repository from %s: %s", self._kv_path, exc)
        return {}

    def _save_kv(self) -> None:
        """Atomic write via tmp file → rename."""
        tmp = self._kv_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._rca_repository, fh, indent=2, sort_keys=True)
        tmp.replace(self._kv_path)


# ---------------------------------------------------------------------------
# Backwards-compatible alias
# ---------------------------------------------------------------------------

class LocalVectorDB(ChromaVectorDB):
    """
    Backwards-compatible shim so existing callers can continue to use:
        from db import LocalVectorDB

    Accepts the legacy ``db_path`` keyword argument and routes to
    ChromaVectorDB using the parent directory's chroma_store/.

    Any new code should use ChromaVectorDB directly.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        dimensions: int = VECTOR_DIM,
        **kwargs: Any,
    ) -> None:
        if db_path is None:
            persist_dir = DEFAULT_CHROMA_PATH
        else:
            # Legacy callers may pass e.g. data/vector_store.sqlite — route to
            # the sibling chroma_store/ directory instead.
            persist_dir = Path(db_path).parent / "chroma_store"
        super().__init__(persist_dir=persist_dir, **kwargs)


# ---------------------------------------------------------------------------
# CLI — query either collection from the command line
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the ChromaDB CI log vector store or RCA knowledge base."
    )
    parser.add_argument("query", nargs="?", help="Search query.")
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_CHROMA_PATH,
        help="ChromaDB persist directory (default: data/chroma_store).",
    )
    parser.add_argument(
        "--collection", default=LOG_COLLECTION,
        help=f"Collection name (default: {LOG_COLLECTION}).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    parser.add_argument(
        "--kb", action="store_true",
        help="Query the RCA knowledge base (data/rca_knowledge_base) instead.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ---- RCA knowledge-base mode ----
    if args.kb:
        kb = SelfLearningKnowledgeBase(persist_dir=DEFAULT_RCA_KB_PATH)
        if not args.query:
            print(f"RCA Knowledge Base : {DEFAULT_RCA_KB_PATH}")
            print(f"Entries            : {len(kb._rca_repository)}")
            print("Pass a query to search the knowledge base.")
            return 0
        results = kb.search(args.query, top_k=args.top_k)
        for entry in results:
            print(f"{entry.similarity:.3f}  [{entry.error_type}]  {entry.error_signature}")
            print(f"         {entry.rca_summary[:200]}")
            print()
        return 0

    # ---- Log vector store mode ----
    db = ChromaVectorDB(persist_dir=args.db, collection_name=args.collection)
    if not args.query:
        print(f"ChromaDB path : {args.db}")
        print(f"Collection    : {args.collection}")
        print(f"Documents     : {db._collection.count()}")
        print("Pass a query to search indexed log excerpts.")
        return 0

    results = db.search(args.query, top_k=args.top_k, collection=args.collection)
    for result in results:
        meta  = result.metadata
        title = " / ".join(
            str(v)
            for v in (
                meta.get("workflow_name"),
                meta.get("job_name"),
                meta.get("error_type"),
                meta.get("error_code"),
            )
            if v
        )
        print(f"{result.score:.3f}  {result.document_id}  {title}")
        print(result.text[:500].replace("\n", " "))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
