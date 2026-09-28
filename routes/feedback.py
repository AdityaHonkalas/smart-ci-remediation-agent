#!/usr/bin/env python3
"""
routes/feedback.py
──────────────────
Blueprint: human-in-the-loop feedback endpoint.

Routes
------
POST /api/user-feedback  — Record structured user judgement on a fix recommendation.
"""

from __future__ import annotations

import os
from typing import Any

from flask import Blueprint, jsonify, request

feedback_bp = Blueprint("feedback", __name__)


@feedback_bp.post("/api/user-feedback")
def user_feedback() -> Any:
    """
    Record a structured human feedback entry on a fix recommendation.

    Expected JSON body
    ------------------
    Required fields:
      workflow_id        : str   — GitHub Actions run_id (string)
      repository         : str   — "owner/repo"
      error_type         : str   — from rca.error_type
      error_signature    : str   — from rca.error_signature
      fix_rank           : int   — which fix (1–5) was evaluated
      root_cause_correct : str   — "yes" | "no" | "partially"
      fix_correct        : str   — "yes" | "no" | "partially"
      resolved_issue     : str   — "yes" | "no" | "partially"
      rating             : int   — 1–5

    Optional fields:
      fix_description    : str   — fix.description for context
      confidence         : float — fix.confidence_report.score (default 0.0)
      feedback_comment   : str   — free text (default "")
      model              : str   — LLM model name (default from MODEL_ID env)

    Responses
    ---------
    201  {"status": "recorded", "entry_id": "<uuid>"}
    400  {"error": "<validation message>"}
    500  {"error": "<exception message>"}
    """
    payload = request.get_json(silent=True) or {}

    # ── Required field validation ──────────────────────────────────────
    required = [
        "workflow_id", "repository", "error_type", "error_signature",
        "fix_rank", "root_cause_correct", "fix_correct", "resolved_issue",
        "rating",
    ]
    missing = [f for f in required if not payload.get(f) and payload.get(f) != 0]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    tristate_fields = ["root_cause_correct", "fix_correct", "resolved_issue"]
    valid_tristate = {"yes", "no", "partially"}
    for tf in tristate_fields:
        val = str(payload.get(tf, "")).strip().lower()
        if val not in valid_tristate:
            return jsonify({
                "error": f"Field '{tf}' must be one of: yes, no, partially. Got: {val!r}"
            }), 400

    rating = payload.get("rating")
    try:
        rating = int(rating)
        if not (1 <= rating <= 5):
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "Field 'rating' must be an integer between 1 and 5."}), 400

    fix_rank = payload.get("fix_rank")
    try:
        fix_rank = int(fix_rank)
    except (TypeError, ValueError):
        return jsonify({"error": "Field 'fix_rank' must be an integer."}), 400

    # ── Build FeedbackEntry ────────────────────────────────────────────
    try:
        from utility.feedback_utils import FeedbackEntry, UserFeedbackStore  # noqa: PLC0415
        from db import utc_now  # noqa: PLC0415

        fix_correct = str(payload["fix_correct"]).strip().lower()
        entry = FeedbackEntry(
            workflow_id        = str(payload["workflow_id"]),
            repository         = str(payload["repository"]),
            error_type         = str(payload["error_type"]),
            error_signature    = str(payload["error_signature"]),
            fix_rank           = fix_rank,
            fix_description    = str(payload.get("fix_description") or ""),
            root_cause_correct = str(payload["root_cause_correct"]).strip().lower(),
            fix_correct        = fix_correct,
            resolved_issue     = str(payload["resolved_issue"]).strip().lower(),
            confidence         = float(payload.get("confidence") or 0.0),
            rating             = rating,
            feedback_comment   = str(payload.get("feedback_comment") or ""),
            model              = str(
                payload.get("model") or os.getenv("MODEL_ID", "unknown")
            ),
            timestamp          = utc_now(),
        )
        # derive accepted + validation_status
        entry.derive()

        store    = UserFeedbackStore()
        entry_id = store.submit_feedback(entry)

    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "status":            "recorded",
        "entry_id":          entry_id,
        "workflow_id":       entry.workflow_id,
        "fix_rank":          entry.fix_rank,
        "accepted":          entry.accepted,
        "validation_status": entry.validation_status,
    }), 201
