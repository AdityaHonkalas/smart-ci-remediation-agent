#!/usr/bin/env python3
"""
utility/feedback_utils.py
─────────────────────────
Human-in-the-Loop feedback store for CI fix validation.

Stores structured user judgements on generated RCA and fix recommendations
in a dedicated ``user_feedback`` ChromaDB collection that lives alongside the
existing ``rca_knowledge_base`` collection under ``data/rca_knowledge_base/``.

This is the authoritative source of human-validated ground truth, separate
from the proxy ``fix_success_rates.json`` used by the automated pipeline.

=============================================================================
FeedbackEntry Schema
=============================================================================

  workflow_id       str    GitHub Actions run_id (stored as string)
  repository        str    owner/repo  e.g. "spring-projects/spring-boot"
  error_type        str    from rca.error_type
  error_signature   str    from rca.error_signature (links back to KB entry)
  fix_rank          int    which fix (1–5) was evaluated
  fix_description   str    fix.description for context
  root_cause_correct str   "yes" | "no" | "partially"
  fix_correct        str   "yes" | "no" | "partially"
  resolved_issue     str   "yes" | "no" | "partially"
  confidence         float pre-computed fix.confidence_report.score
  rating             int   1–5
  feedback_comment   str   free text (embedded into ChromaDB for future search)
  model              str   LLM model name (from env MODEL_ID or rca.source)
  timestamp          str   ISO-8601 UTC
  accepted           bool  True if fix_correct in ("yes", "partially")
  validation_status  str   "Verified" | "Rejected" | "Partial"

=============================================================================
Collection: user_feedback (ChromaDB)
=============================================================================

  Embedding document : feedback_comment text (384-dim MiniLM)
  ChromaDB metadata  : all fields above except feedback_comment
  ChromaDB ID        : UUID4 (returned from submit_feedback)

Usage
─────
  from utility.feedback_utils import UserFeedbackStore, FeedbackEntry
  store = UserFeedbackStore()
  entry_id = store.submit_feedback(entry)
  entries  = store.list_all_feedback(limit=50)
  summary  = store.feedback_summary_by_error_type()
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

# Ensure scripts/ is on sys.path so db.py helpers are importable
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from db import _Embedder, _coerce_meta  # noqa: E402

logger = logging.getLogger(__name__)

_FEEDBACK_COLLECTION = "user_feedback"
_DEFAULT_KB_PATH = _ROOT / "data" / "rca_knowledge_base"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _derive_acceptance(fix_correct: str) -> tuple[bool, str]:
    """Return (accepted, validation_status) derived from fix_correct answer."""
    fc = (fix_correct or "").strip().lower()
    accepted = fc in ("yes", "partially")
    status_map = {"yes": "Verified", "no": "Rejected", "partially": "Partial"}
    validation_status = status_map.get(fc, "Unknown")
    return accepted, validation_status


# ---------------------------------------------------------------------------
# FeedbackEntry dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeedbackEntry:
    """Structured user judgement on a single fix recommendation."""

    # Identity
    workflow_id:        str   # GitHub Actions run_id as string
    repository:         str   # owner/repo
    error_type:         str   # from rca.error_type
    error_signature:    str   # from rca.error_signature

    # Which fix was evaluated
    fix_rank:           int   # 1–5
    fix_description:    str   # fix.description

    # Acceptance criteria (tristate: "yes" | "no" | "partially")
    root_cause_correct: str
    fix_correct:        str
    resolved_issue:     str

    # Scores
    confidence:         float  # fix.confidence_report.score
    rating:             int    # 1–5

    # Free text (this is what gets embedded in ChromaDB)
    feedback_comment:   str

    # Provenance
    model:              str    # LLM model name
    timestamp:          str    # ISO-8601 UTC

    # Derived fields — computed automatically; set defaults so callers can
    # skip them and call FeedbackEntry.derive() after construction.
    accepted:           bool = field(default=False)
    validation_status:  str  = field(default="Unknown")

    def derive(self) -> "FeedbackEntry":
        """Compute accepted + validation_status from fix_correct and return self."""
        self.accepted, self.validation_status = _derive_acceptance(self.fix_correct)
        return self

    @classmethod
    def from_meta(cls, meta: dict[str, Any], comment: str = "") -> "FeedbackEntry":
        """Reconstruct a FeedbackEntry from ChromaDB metadata + document text."""
        entry = cls(
            workflow_id        = str(meta.get("workflow_id", "")),
            repository         = str(meta.get("repository", "")),
            error_type         = str(meta.get("error_type", "")),
            error_signature    = str(meta.get("error_signature", "")),
            fix_rank           = int(meta.get("fix_rank", 0)),
            fix_description    = str(meta.get("fix_description", "")),
            root_cause_correct = str(meta.get("root_cause_correct", "")),
            fix_correct        = str(meta.get("fix_correct", "")),
            resolved_issue     = str(meta.get("resolved_issue", "")),
            confidence         = float(meta.get("confidence", 0.0)),
            rating             = int(meta.get("rating", 0)),
            feedback_comment   = comment,
            model              = str(meta.get("model", "")),
            timestamp          = str(meta.get("timestamp", "")),
            accepted           = bool(meta.get("accepted", False)),
            validation_status  = str(meta.get("validation_status", "Unknown")),
        )
        return entry


# ---------------------------------------------------------------------------
# UserFeedbackStore
# ---------------------------------------------------------------------------

class UserFeedbackStore:
    """
    ChromaDB-backed store for human feedback on fix recommendations.

    Collection : ``user_feedback``
    Path       : ``data/rca_knowledge_base/`` (shares PersistentClient with KB)
    Embedding  : ``feedback_comment`` text via all-MiniLM-L6-v2 (384-dim)
    """

    def __init__(self, persist_dir: Path | str = _DEFAULT_KB_PATH) -> None:
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
            name=_FEEDBACK_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "UserFeedbackStore ready — %d entries stored",
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def submit_feedback(self, entry: FeedbackEntry) -> str:
        """
        Persist a FeedbackEntry to ChromaDB.

        Derives ``accepted`` and ``validation_status`` automatically from
        ``fix_correct`` before storing, so callers don't need to set them.

        Returns
        -------
        str
            UUID4 entry ID (stable identifier for this feedback record).
        """
        # Ensure derived fields are up-to-date
        entry.accepted, entry.validation_status = _derive_acceptance(entry.fix_correct)

        entry_id = str(uuid.uuid4())
        comment  = entry.feedback_comment or ""

        # Embed the free-text comment; fall back to a zero-vector if empty
        if comment.strip():
            embedding = _Embedder.get().embed_one(comment)
        else:
            embedding = [0.0] * 384

        metadata = _coerce_meta({
            "workflow_id":        str(entry.workflow_id),
            "repository":         entry.repository,
            "error_type":         entry.error_type,
            "error_signature":    entry.error_signature,
            "fix_rank":           entry.fix_rank,
            "fix_description":    entry.fix_description[:400],  # cap for metadata size
            "root_cause_correct": entry.root_cause_correct,
            "fix_correct":        entry.fix_correct,
            "resolved_issue":     entry.resolved_issue,
            "confidence":         round(float(entry.confidence), 4),
            "rating":             int(entry.rating),
            "model":              entry.model,
            "timestamp":          entry.timestamp or _utc_now(),
            "accepted":           entry.accepted,
            "validation_status":  entry.validation_status,
        })

        self._collection.upsert(
            ids        = [entry_id],
            embeddings = [embedding],
            documents  = [comment],
            metadatas  = [metadata],
        )
        logger.info(
            "Feedback submitted — id=%s  workflow=%s  fix_rank=%d  accepted=%s",
            entry_id, entry.workflow_id, entry.fix_rank, entry.accepted,
        )
        return entry_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query_feedback(self, workflow_id: str) -> list[FeedbackEntry]:
        """
        Retrieve all feedback entries for a specific workflow run.

        Parameters
        ----------
        workflow_id : str
            GitHub Actions run_id (string).
        """
        try:
            result = self._collection.get(
                where     = {"workflow_id": {"$eq": str(workflow_id)}},
                include   = ["metadatas", "documents"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("query_feedback failed for workflow_id=%s: %s", workflow_id, exc)
            return []

        return self._to_entries(result)

    def list_all_feedback(self, limit: int = 100) -> list[FeedbackEntry]:
        """
        Return up to ``limit`` feedback entries, most recent first.

        ChromaDB does not support ORDER BY, so all entries are fetched and
        sorted in Python by ``timestamp`` descending.
        """
        try:
            result = self._collection.get(
                include = ["metadatas", "documents"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_all_feedback failed: %s", exc)
            return []

        entries = self._to_entries(result)
        # Sort newest first
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]

    # ------------------------------------------------------------------
    # Aggregation for evaluation dashboard
    # ------------------------------------------------------------------

    def feedback_summary_by_error_type(self) -> dict[str, Any]:
        """
        Return per-error-type aggregated feedback metrics.

        Returns
        -------
        dict with keys:
          total_count         int
          acceptance_rate     float  (accepted / total)
          avg_rating          float
          root_cause_correct_rate float  (yes + partially) / total
          resolved_issue_rate float  yes / total
          by_error_type       list[dict]  per-type breakdown
          rating_distribution dict   {1: n, 2: n, 3: n, 4: n, 5: n}
          correctness_distribution dict  {yes: n, partially: n, no: n}
        """
        entries = self.list_all_feedback(limit=10_000)
        total = len(entries)

        if total == 0:
            return {
                "total_count":              0,
                "acceptance_rate":          None,
                "avg_rating":               None,
                "root_cause_correct_rate":  None,
                "resolved_issue_rate":      None,
                "by_error_type":            [],
                "rating_distribution":      {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                "correctness_distribution": {"yes": 0, "partially": 0, "no": 0},
            }

        accepted_count = sum(1 for e in entries if e.accepted)
        ratings        = [e.rating for e in entries if 1 <= e.rating <= 5]
        rcc_count      = sum(
            1 for e in entries
            if e.root_cause_correct.lower() in ("yes", "partially")
        )
        ri_count       = sum(
            1 for e in entries
            if e.resolved_issue.lower() == "yes"
        )

        # Rating distribution
        rating_dist: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in ratings:
            if r in rating_dist:
                rating_dist[r] += 1

        # fix_correct correctness distribution
        correct_dist: dict[str, int] = {"yes": 0, "partially": 0, "no": 0}
        for e in entries:
            key = e.fix_correct.lower()
            if key in correct_dist:
                correct_dist[key] += 1

        # Per-error-type breakdown
        by_type: dict[str, dict] = defaultdict(
            lambda: {"total": 0, "accepted": 0, "ratings": [], "rcc": 0, "resolved": 0}
        )
        for e in entries:
            t = e.error_type or "unknown"
            by_type[t]["total"]    += 1
            by_type[t]["accepted"] += int(e.accepted)
            by_type[t]["rcc"]      += int(e.root_cause_correct.lower() in ("yes", "partially"))
            by_type[t]["resolved"] += int(e.resolved_issue.lower() == "yes")
            if 1 <= e.rating <= 5:
                by_type[t]["ratings"].append(e.rating)

        per_type_list = []
        for err_type, agg in sorted(by_type.items()):
            n = agg["total"]
            per_type_list.append({
                "error_type":              err_type,
                "total":                   n,
                "acceptance_rate":         round(agg["accepted"] / n, 4),
                "avg_rating":              round(mean(agg["ratings"]), 2) if agg["ratings"] else None,
                "root_cause_correct_rate": round(agg["rcc"] / n, 4),
                "resolved_issue_rate":     round(agg["resolved"] / n, 4),
            })

        return {
            "total_count":              total,
            "acceptance_rate":          round(accepted_count / total, 4),
            "avg_rating":               round(mean(ratings), 2) if ratings else None,
            "root_cause_correct_rate":  round(rcc_count / total, 4),
            "resolved_issue_rate":      round(ri_count / total, 4),
            "by_error_type":            per_type_list,
            "rating_distribution":      rating_dist,
            "correctness_distribution": correct_dist,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_entries(chroma_result: dict[str, Any]) -> list[FeedbackEntry]:
        """Convert a raw ChromaDB get() result into FeedbackEntry objects."""
        metadatas = chroma_result.get("metadatas") or []
        documents = chroma_result.get("documents") or []
        entries: list[FeedbackEntry] = []
        for meta, doc in zip(metadatas, documents):
            if not isinstance(meta, dict):
                continue
            entries.append(FeedbackEntry.from_meta(meta, comment=doc or ""))
        return entries
