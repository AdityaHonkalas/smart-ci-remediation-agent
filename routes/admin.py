#!/usr/bin/env python3
"""
routes/admin.py
───────────────
Blueprint: admin / operational endpoints.

Routes
------
POST /api/admin/kb-update  — trigger offline KB batch update from staging queue
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify

admin_bp = Blueprint("admin", __name__)

_ROOT_DIR = Path(__file__).resolve().parents[1]


@admin_bp.post("/api/admin/kb-update")
def kb_update() -> Any:
    """
    Trigger the offline KB batch update process.

    Reads all buffered RCA entries from ``data/rca_staging_queue.json``,
    upserts them into ChromaDB, and clears the queue.

    Returns
    -------
    JSON: {status, processed, skipped, errors, timestamp}
    """
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "kb_batch_update",
            _ROOT_DIR / "scripts" / "kb_batch_update.py",
        )
        _mod = _ilu.module_from_spec(_spec)      # type: ignore[arg-type]
        _spec.loader.exec_module(_mod)            # type: ignore[union-attr]
        result = _mod.run_kb_batch_update()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    return jsonify({"status": "ok", **result})
