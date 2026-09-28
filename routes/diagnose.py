#!/usr/bin/env python3
"""
routes/diagnose.py
──────────────────
Blueprint: main diagnosis routes (index page, /api/diagnose, /api/feedback).
Moved from app.py as part of the modular restructure.
"""

from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, render_template, request

diagnose_bp = Blueprint("diagnose", __name__)


@diagnose_bp.get("/")
def index() -> str:
    # model_id field removed from UI — model is configured via .env (MODEL_ID / API_URL)
    return render_template("index.html")


@diagnose_bp.post("/api/diagnose")
def diagnose() -> Any:
    """
    Trigger the full RCA pipeline for a GitHub Actions workflow run.

    Expected JSON body:
        run_url       : str  — GitHub Actions run URL (required)
        github_token  : str  — GitHub PAT (optional; falls back to GITHUB_TOKEN env)
        model_id      : str  — LLM model override (optional)
        use_model     : bool — use LLM for RCA (default true)
    """
    # Import lazily — agents/ is on sys.path from app.py setup
    from agents.diagnosis_agent import diagnose_workflow_run  # noqa: PLC0415

    payload = request.get_json(silent=True) or {}
    run_url = str(payload.get("run_url") or "").strip()
    if not run_url:
        return jsonify({"error": "GitHub Actions run URL is required."}), 400

    try:
        result = diagnose_workflow_run(
            run_url=run_url,
            token=str(payload.get("github_token") or "").strip() or None,
            use_model=bool(payload.get("use_model", True)),
            # model_id no longer sent from UI — model configured via .env
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


@diagnose_bp.post("/api/feedback")
def feedback() -> Any:
    """
    Record whether a recommended fix resolved the CI failure.

    Expected JSON body:
        fix_type   : str  — fix_type from the recommended fix (required)
        error_type : str  — error_type from the RCA report (required)
        success    : bool — did the fix resolve the failure? (required)
    """
    payload    = request.get_json(silent=True) or {}
    fix_type   = str(payload.get("fix_type")   or "").strip()
    error_type = str(payload.get("error_type") or "").strip()
    success    = payload.get("success")

    if not fix_type or not error_type or success is None:
        return jsonify({"error": "fix_type, error_type, and success are required."}), 400

    try:
        from utility.fix_utils import record_fix_outcome  # noqa: PLC0415
        record_fix_outcome(fix_type, error_type, bool(success))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"warning": f"Feedback recorded but KB update failed: {exc}"}), 207

    return jsonify({
        "status":     "recorded",
        "fix_type":   fix_type,
        "error_type": error_type,
        "success":    bool(success),
    })
