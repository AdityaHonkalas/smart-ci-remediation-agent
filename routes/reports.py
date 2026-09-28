#!/usr/bin/env python3
"""
routes/reports.py
─────────────────
Blueprint: statistical analysis and evaluation metrics report pages.

Routes
------
GET  /stats                  — Statistical Analysis page (templates/stats.html)
GET  /api/stats/summary      — JSON data for all stats charts
GET  /evaluation             — Evaluation Metrics page (templates/evaluation.html)
GET  /api/evaluation/metrics — JSON data for all evaluation charts
"""

from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, render_template

reports_bp = Blueprint("reports", __name__)


# ---------------------------------------------------------------------------
# Statistical Analysis
# ---------------------------------------------------------------------------

@reports_bp.get("/stats")
def stats_page() -> str:
    return render_template("stats.html")


@reports_bp.get("/api/stats/summary")
def stats_summary() -> Any:
    """Return pre-aggregated data for all 10 statistical analysis charts."""
    try:
        from utility.stats_utils import build_stats_payload  # noqa: PLC0415
        payload = build_stats_payload()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500
    return jsonify(payload)


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

@reports_bp.get("/evaluation")
def evaluation_page() -> str:
    return render_template("evaluation.html")


@reports_bp.get("/api/evaluation/metrics")
def evaluation_metrics() -> Any:
    """Return pre-computed evaluation metric values and chart data."""
    try:
        from utility.evaluation_utils import build_evaluation_payload  # noqa: PLC0415
        payload = build_evaluation_payload()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500
    return jsonify(payload)
