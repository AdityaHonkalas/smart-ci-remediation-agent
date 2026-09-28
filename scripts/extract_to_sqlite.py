#!/usr/bin/env python3
"""
scripts/extract_to_sqlite.py
─────────────────────────────
Extract the ``user_feedback`` ChromaDB collection from
``data/rca_knowledge_base/chroma.sqlite3`` into a flat SQLite table at
``output/analysis.db``.

The script queries the ChromaDB backing store **directly** using the
stdlib ``sqlite3`` module — no ``chromadb`` package import required.

Usage
─────
    python scripts/extract_to_sqlite.py
    python scripts/extract_to_sqlite.py --output output/analysis.db
    python scripts/extract_to_sqlite.py --chroma-kb path/to/chroma.sqlite3

ChromaDB internal schema (≥ 0.5)
─────────────────────────────────
    collections           id (TEXT PK), name
    segments              id (TEXT PK), scope, collection (FK → collections.id)
    embeddings            id (INTEGER PK), segment_id (FK → segments.id), embedding_id
    embedding_metadata    id (FK → embeddings.id), key, string_value,
                          int_value, float_value, bool_value

    The document text (feedback_comment) is stored as a metadata row with
    key = 'chroma:document' and the text in string_value.

output/analysis.db — user_feedback table
─────────────────────────────────────────
    chroma_id            TEXT   embedding_id UUID from ChromaDB
    workflow_id          TEXT
    repository           TEXT
    error_type           TEXT
    error_signature      TEXT
    fix_rank             INTEGER  (1–5)
    fix_description      TEXT
    root_cause_correct   TEXT   "yes" | "no" | "partially"
    fix_correct          TEXT   "yes" | "no" | "partially"
    resolved_issue       TEXT   "yes" | "no" | "partially"
    confidence           REAL   (0.0–1.0)
    rating               INTEGER (1–5)
    model                TEXT
    timestamp            TEXT   ISO-8601 UTC
    accepted             INTEGER (1 = accepted, 0 = not)
    validation_status    TEXT   "Verified" | "Rejected" | "Partial" | "Unknown"
    feedback_comment     TEXT
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COLLECTION_NAME = "user_feedback"

# Ordered list of columns for the output table (matches INSERT order)
_COLUMNS = [
    ("chroma_id",           "TEXT"),
    ("workflow_id",         "TEXT"),
    ("repository",          "TEXT"),
    ("error_type",          "TEXT"),
    ("error_signature",     "TEXT"),
    ("fix_rank",            "INTEGER"),
    ("fix_description",     "TEXT"),
    ("root_cause_correct",  "TEXT"),
    ("fix_correct",         "TEXT"),
    ("resolved_issue",      "TEXT"),
    ("confidence",          "REAL"),
    ("rating",              "INTEGER"),
    ("model",               "TEXT"),
    ("timestamp",           "TEXT"),
    ("accepted",            "INTEGER"),
    ("validation_status",   "TEXT"),
    ("feedback_comment",    "TEXT"),
]

# Metadata keys that hold integer values in the int_value column
_INT_KEYS = {"fix_rank", "rating"}

# Metadata keys that hold float values in the float_value column
_FLOAT_KEYS = {"confidence"}

# Metadata keys that hold boolean values in the bool_value column
_BOOL_KEYS = {"accepted"}

# The special key ChromaDB uses to store the embedded document text
_DOCUMENT_KEY = "chroma:document"


# ---------------------------------------------------------------------------
# ChromaDB reader
# ---------------------------------------------------------------------------

def _read_user_feedback(src_path: Path) -> list[dict]:
    """
    Read all documents from the ``user_feedback`` collection in a ChromaDB
    SQLite file and return them as a list of flat dicts.

    Steps
    -----
    1. Resolve the collection UUID for ``user_feedback``.
    2. Find its METADATA segment (the one that owns embedding_metadata rows).
    3. Fetch all embeddings for that segment.
    4. Pivot the EAV ``embedding_metadata`` rows into one dict per document,
       picking the right typed column (string/int/float/bool) per key.
    5. Extract ``feedback_comment`` from the ``chroma:document`` metadata key.
    """
    con = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # -- 1. Collection UUID ---------------------------------------------------
    cur.execute(
        "SELECT id FROM collections WHERE name = ?",
        (_COLLECTION_NAME,),
    )
    row = cur.fetchone()
    if row is None:
        con.close()
        print(
            f"[extract_to_sqlite] WARNING: collection '{_COLLECTION_NAME}' "
            f"not found in {src_path}. Table will be empty.",
            file=sys.stderr,
        )
        return []
    collection_id: str = row["id"]

    # -- 2. METADATA segment for this collection ------------------------------
    cur.execute(
        "SELECT id FROM segments WHERE collection = ? AND scope = 'METADATA'",
        (collection_id,),
    )
    seg_row = cur.fetchone()
    if seg_row is None:
        con.close()
        print(
            f"[extract_to_sqlite] WARNING: no METADATA segment for "
            f"collection '{_COLLECTION_NAME}'.",
            file=sys.stderr,
        )
        return []
    segment_id: str = seg_row["id"]

    # -- 3. All embeddings in this segment ------------------------------------
    cur.execute(
        "SELECT id, embedding_id FROM embeddings WHERE segment_id = ?",
        (segment_id,),
    )
    embeddings = cur.fetchall()  # list of (internal_id, user_uuid)

    # -- 4. Pivot EAV metadata rows -------------------------------------------
    # Build: internal_id → embedding_id mapping
    id_to_uuid: dict[int, str] = {row["id"]: row["embedding_id"] for row in embeddings}
    internal_ids = list(id_to_uuid.keys())

    if not internal_ids:
        con.close()
        return []

    # Fetch all metadata rows for these embeddings in one query
    placeholders = ",".join("?" * len(internal_ids))
    cur.execute(
        f"""
        SELECT id, key, string_value, int_value, float_value, bool_value
        FROM   embedding_metadata
        WHERE  id IN ({placeholders})
        """,
        internal_ids,
    )
    meta_rows = cur.fetchall()
    con.close()

    # Build per-document metadata dicts
    docs: dict[int, dict] = {iid: {} for iid in internal_ids}
    for row in meta_rows:
        iid  = row["id"]
        key  = row["key"]
        if key == _DOCUMENT_KEY:
            # Document text — stored under 'chroma:document'
            docs[iid]["feedback_comment"] = row["string_value"] or ""
        elif key in _INT_KEYS:
            docs[iid][key] = row["int_value"]
        elif key in _FLOAT_KEYS:
            docs[iid][key] = row["float_value"]
        elif key in _BOOL_KEYS:
            # ChromaDB stores bools as int 0/1 in bool_value
            raw = row["bool_value"]
            docs[iid][key] = int(raw) if raw is not None else 0
        else:
            docs[iid][key] = row["string_value"]

    # -- 5. Assemble output rows ----------------------------------------------
    result: list[dict] = []
    for iid, meta in docs.items():
        record: dict = {"chroma_id": id_to_uuid[iid]}
        for col_name, _ in _COLUMNS[1:]:   # skip chroma_id — already set
            record[col_name] = meta.get(col_name)
        result.append(record)

    return result


# ---------------------------------------------------------------------------
# SQLite writer
# ---------------------------------------------------------------------------

_CREATE_USER_FEEDBACK = """
CREATE TABLE user_feedback (
    chroma_id            TEXT    PRIMARY KEY,
    workflow_id          TEXT,
    repository           TEXT,
    error_type           TEXT,
    error_signature      TEXT,
    fix_rank             INTEGER,
    fix_description      TEXT,
    root_cause_correct   TEXT,
    fix_correct          TEXT,
    resolved_issue       TEXT,
    confidence           REAL,
    rating               INTEGER,
    model                TEXT,
    timestamp            TEXT,
    accepted             INTEGER,
    validation_status    TEXT,
    feedback_comment     TEXT
)
"""


def extract_user_feedback(src_path: Path, dst_conn: sqlite3.Connection) -> int:
    """
    Extract user_feedback from *src_path* (chroma.sqlite3) and write rows
    into the ``user_feedback`` table of *dst_conn*.

    Returns the number of rows inserted.
    """
    rows = _read_user_feedback(src_path)

    cur = dst_conn.cursor()
    cur.execute("DROP TABLE IF EXISTS user_feedback")
    cur.execute(_CREATE_USER_FEEDBACK)

    if not rows:
        dst_conn.commit()
        return 0

    col_names = [c[0] for c in _COLUMNS]
    placeholders = ",".join("?" * len(col_names))
    insert_sql = (
        f"INSERT INTO user_feedback ({', '.join(col_names)}) "
        f"VALUES ({placeholders})"
    )
    values = [tuple(row.get(c) for c in col_names) for row in rows]
    cur.executemany(insert_sql, values)
    dst_conn.commit()
    return len(values)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Extract the user_feedback ChromaDB collection into "
            "a flat SQLite table at output/analysis.db."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "output" / "analysis.db",
        metavar="PATH",
        help="Destination SQLite file (default: output/analysis.db)",
    )
    parser.add_argument(
        "--chroma-kb",
        type=Path,
        default=root / "data" / "rca_knowledge_base" / "chroma.sqlite3",
        metavar="PATH",
        help="Source ChromaDB SQLite file (default: data/rca_knowledge_base/chroma.sqlite3)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Validate source
    if not args.chroma_kb.exists():
        print(
            f"ERROR: ChromaDB source not found: {args.chroma_kb}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Open destination DB and extract
    dst_conn = sqlite3.connect(args.output)
    try:
        n = extract_user_feedback(args.chroma_kb, dst_conn)
    finally:
        dst_conn.close()

    # Summary
    col_w = max(len(c[0]) for c in _COLUMNS) + 2
    table_name = "user_feedback"
    print(f"{table_name:<{col_w}} -> {n} rows")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
