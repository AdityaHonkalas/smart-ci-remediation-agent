#!/usr/bin/env python3
"""
routes/validation.py
────────────────────
Blueprint: automated fix validation endpoints (Track B).

Routes
------
GET  /api/validate/status  — Probe whether Tier-2 live validation is enabled.
POST /api/validate         — Trigger live CI validation for selected fix candidates.

Dual-Gate Design
────────────────
Tier-2 live validation is protected by two gates:

  1. **Backend env gate** — TIER2_ENABLED must be set to "true" in .env (or the
     environment).  This is the admin safety switch that prevents any live clone /
     PR creation from happening unexpectedly.

  2. **User opt-in** — The request body must include ``live_mode: true``.  The UI
     surfaces a toggle that is disabled when the backend gate is off and interactive
     when it is on.

When both gates are open, the route launches a background thread running
``validate_fixes.run_live_validation(inputs)`` and returns 202 immediately.
When either gate is closed, it returns 200 with status "not_enabled".
"""

from __future__ import annotations

import os
import threading
from typing import Any

from flask import Blueprint, jsonify, request

validation_bp = Blueprint("validation", __name__)


def _tier2_enabled() -> bool:
    """Return True if TIER2_ENABLED env var is 'true' (read at request time)."""
    return os.getenv("TIER2_ENABLED", "false").strip().lower() == "true"


# ---------------------------------------------------------------------------
# GET /api/validate/status
# ---------------------------------------------------------------------------

@validation_bp.get("/api/validate/status")
def validate_status() -> Any:
    """
    Probe Tier-2 live validation availability.

    Called on page load by the UI to set the initial state of the
    "Enable Live CI Validation" toggle.

    Response
    --------
    200  {"tier2_enabled": bool, "message": str}
    """
    enabled = _tier2_enabled()
    return jsonify({
        "tier2_enabled": enabled,
        "message": (
            "Live validation is enabled. "
            "Toggle 'Enable Live CI Validation' and click Run to start."
            if enabled else
            "Live validation is disabled. "
            "Set TIER2_ENABLED=true in .env to enable it."
        ),
    })


# ---------------------------------------------------------------------------
# POST /api/validate
# ---------------------------------------------------------------------------

@validation_bp.post("/api/validate")
def validate() -> Any:
    """
    Trigger Tier-2 live CI validation for selected fix candidates.

    Expected JSON body
    ------------------
    Required fields:
      fork_repo       : str        — "owner/repo" of the user's fork
      branch          : str        — base branch (e.g. "main")
      github_username : str        — GitHub username
      error_signature : str        — KB error signature

    Optional fields:
      token           : str        — GitHub PAT (falls back to GITHUB_TOKEN env)
      fix_ranks       : list[int] | "all"  — which fixes to validate (default "all")
      fixes           : list[dict] — fix objects from RCA output
      workflow_id     : str        — original failed action run_id (string)
      live_mode       : bool       — user opt-in toggle (default false)

    Responses
    ---------
    200  {"status": "not_enabled", "message": str}
         — when TIER2_ENABLED=False OR live_mode=False
    202  {"status": "queued", "message": str, "run_id": str}
         — when both gates are open; background thread launched
    400  {"error": str}
         — missing required fields
    500  {"error": str}
         — import or threading failure
    """
    payload = request.get_json(silent=True) or {}

    # ── Required field validation ──────────────────────────────────────
    required = ["fork_repo", "branch", "github_username", "error_signature"]
    missing  = [f for f in required if not str(payload.get(f) or "").strip()]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    fork_repo       = str(payload["fork_repo"]).strip()
    branch          = str(payload["branch"]).strip()
    github_username = str(payload["github_username"]).strip()
    error_signature = str(payload["error_signature"]).strip()
    token           = str(payload.get("token") or os.getenv("GITHUB_TOKEN") or "").strip()
    fix_ranks       = payload.get("fix_ranks", "all")
    fixes           = payload.get("fixes") or []
    workflow_id     = str(payload.get("workflow_id") or "0").strip()
    live_mode       = bool(payload.get("live_mode", False))

    # ── Dual-gate check ───────────────────────────────────────────────
    if not _tier2_enabled() or not live_mode:
        reason = (
            "TIER2_ENABLED is not set in .env" if not _tier2_enabled()
            else "Live mode toggle is off"
        )
        return jsonify({
            "status":  "not_enabled",
            "message": (
                f"Live validation is not active ({reason}). "
                "Enable TIER2_ENABLED=true in .env and toggle 'Enable Live CI Validation' to proceed."
            ),
        })

    # ── Build inputs dict for the pipeline ───────────────────────────
    inputs: dict[str, Any] = {
        "fork_repo":       fork_repo,
        "branch":          branch,
        "github_username": github_username,
        "token":           token,
        "error_signature": error_signature,
        "fix_ranks":       fix_ranks,
        "fixes":           fixes,
        "workflow_id":     workflow_id,
    }

    # ── Launch background thread ──────────────────────────────────────
    try:
        from scripts.validate_fixes import run_live_validation  # noqa: PLC0415

        def _run() -> None:
            try:
                result = run_live_validation(inputs)
                import logging  # noqa: PLC0415
                logging.getLogger(__name__).info(
                    "Live validation completed — fork=%s  summary=%s",
                    fork_repo, result.get("summary"),
                )
            except Exception as exc:  # noqa: BLE001
                import logging  # noqa: PLC0415
                logging.getLogger(__name__).error(
                    "Live validation thread failed: %s", exc, exc_info=True
                )

        t = threading.Thread(target=_run, daemon=True, name=f"val-{workflow_id}")
        t.start()

    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to start validation: {exc}"}), 500

    return jsonify({
        "status":     "queued",
        "message":    (
            f"Live validation queued for {fork_repo} on branch '{branch}'. "
            "A PR will be raised automatically. Check the action run for results."
        ),
        "run_id":     workflow_id,
        "fork_repo":  fork_repo,
        "branch":     branch,
        "fix_ranks":  fix_ranks,
    }), 202
