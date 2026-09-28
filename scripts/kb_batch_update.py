#!/usr/bin/env python3
"""
scripts/kb_batch_update.py
──────────────────────────
Offline KB batch update runner.

Reads all RCA entries buffered to ``data/rca_staging_queue.json`` by the live
diagnosis pipeline (via ``queue_rca_for_kb_update``), upserts each one into the
ChromaDB ``SelfLearningKnowledgeBase``, then clears the processed entries.

Usage
─────
    # Run directly (standalone):
    python scripts/kb_batch_update.py

    # Or via the Flask admin endpoint:
    POST /api/admin/kb-update

Returns (as dict, printed as JSON when run standalone)
──────
{
    "processed": int,   — entries successfully upserted
    "skipped":   int,   — entries skipped (empty signature or already in KB)
    "errors":    list[str]   — per-entry error messages (non-fatal)
}
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR   = _SCRIPT_DIR.parent
_DATA_DIR   = _ROOT_DIR / "data"

for _p in (str(_ROOT_DIR), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env_loader import load_dotenv  # noqa: E402
load_dotenv(_ROOT_DIR / ".env")

from db import SelfLearningKnowledgeBase  # noqa: E402

STAGING_FILE = _DATA_DIR / "rca_staging_queue.json"
ARCHIVE_FILE = _DATA_DIR / "rca_staging_archive.json"
KB_DIR       = _DATA_DIR / "rca_knowledge_base"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_staging() -> list[dict[str, Any]]:
    """Load the current staging queue; return empty list if missing or corrupt."""
    if not STAGING_FILE.exists():
        return []
    try:
        with STAGING_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load staging queue: %s", exc)
        return []


def _save_staging(entries: list[dict[str, Any]]) -> None:
    """Atomically overwrite the staging queue file."""
    STAGING_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STAGING_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, default=str)
    tmp.replace(STAGING_FILE)


def _append_archive(entries: list[dict[str, Any]]) -> None:
    """Append successfully processed entries to the archive sidecar."""
    existing: list[dict[str, Any]] = []
    if ARCHIVE_FILE.exists():
        try:
            with ARCHIVE_FILE.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, OSError):
            existing = []
    existing.extend(entries)
    tmp = ARCHIVE_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, default=str)
    tmp.replace(ARCHIVE_FILE)


# ---------------------------------------------------------------------------
# Main batch function (callable from Flask endpoint or standalone)
# ---------------------------------------------------------------------------

def run_kb_batch_update(
    data_dir: Path = _DATA_DIR,
) -> dict[str, Any]:
    """
    Process the RCA staging queue and upsert entries into ChromaDB.

    Parameters
    ----------
    data_dir : Path
        Root data directory (default: ``data/``).

    Returns
    -------
    dict
        ``{processed, skipped, errors, timestamp}``
    """
    staging_file = data_dir / "rca_staging_queue.json"
    kb_dir       = data_dir / "rca_knowledge_base"

    # --- Load staging queue ---
    if not staging_file.exists():
        logger.info("KB batch update: staging queue is empty — nothing to process.")
        return {
            "processed": 0,
            "skipped":   0,
            "errors":    [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    try:
        with staging_file.open("r", encoding="utf-8") as fh:
            queue: list[dict[str, Any]] = json.load(fh)
        if not isinstance(queue, list):
            queue = []
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "processed": 0,
            "skipped":   0,
            "errors":    [f"Failed to load staging queue: {exc}"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    if not queue:
        return {
            "processed": 0,
            "skipped":   0,
            "errors":    [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # --- Open KB connection once for the whole batch ---
    try:
        kb = SelfLearningKnowledgeBase(persist_dir=kb_dir)
    except Exception as exc:  # noqa: BLE001
        return {
            "processed": 0,
            "skipped":   len(queue),
            "errors":    [f"Failed to open KB: {exc}"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    processed: int      = 0
    skipped:   int      = 0
    errors:    list[str] = []
    archived:  list[dict[str, Any]] = []

    for entry in queue:
        error_signature = str(entry.get("error_signature") or "").strip()
        rca             = entry.get("rca") or {}

        if not error_signature:
            skipped += 1
            logger.debug("Skipping staging entry with empty error_signature.")
            continue

        try:
            kb.update(error_signature, rca)
            processed += 1
            archived.append(entry)
            logger.info("KB updated from staging — signature=%s", error_signature[:80])
        except Exception as exc:  # noqa: BLE001
            msg = f"signature={error_signature[:60]}: {exc}"
            errors.append(msg)
            logger.warning("KB batch update failed for entry (%s)", msg)

    # --- Clear processed entries from queue ---
    unprocessed = [e for e in queue if e not in archived]
    try:
        tmp = staging_file.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(unprocessed, fh, indent=2, default=str)
        tmp.replace(staging_file)
    except OSError as exc:
        errors.append(f"Failed to clear staging queue: {exc}")

    # --- Archive successfully processed entries ---
    if archived:
        try:
            archive_file = data_dir / "rca_staging_archive.json"
            existing_archive: list[dict[str, Any]] = []
            if archive_file.exists():
                try:
                    with archive_file.open("r", encoding="utf-8") as fh:
                        existing_archive = json.load(fh)
                    if not isinstance(existing_archive, list):
                        existing_archive = []
                except (json.JSONDecodeError, OSError):
                    existing_archive = []
            existing_archive.extend(archived)
            tmp = archive_file.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(existing_archive, fh, indent=2, default=str)
            tmp.replace(archive_file)
        except OSError as exc:
            logger.warning("Failed to write KB archive (non-fatal): %s", exc)

    summary = {
        "processed": processed,
        "skipped":   skipped,
        "errors":    errors,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(
        "KB batch update complete — processed=%d  skipped=%d  errors=%d",
        processed, skipped, len(errors),
    )
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = run_kb_batch_update()
    print(json.dumps(result, indent=2))
    sys.exit(0 if not result["errors"] else 1)
