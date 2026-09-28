#!/usr/bin/env python3
"""
utility/stats_utils.py
───────────────────────
Data aggregation helpers for the /stats Statistical Analysis page.

Primary data source: ChromaDB vector store (both collections).
  - ``rca_knowledge_base`` collection   → RCA-level stats
  - ``ci_failure_logs`` collection      → log-level severity / repository stats
Supplement:
  - ``data/fix_success_rates.json``       → fix success and source stats
  - ``data/rca_staging_archive.json``     → full RCA+fix records from batch runs
  - ``data/rca_knowledge_base/rca_repository.json`` → KB sidecar (hit counts etc.)

All functions return graceful empty datasets when the collections are empty
so Chart.js renders empty-axis charts rather than crashing.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env_loader import load_dotenv  # noqa: E402
load_dotenv(_ROOT / ".env")

from db import ChromaVectorDB, SelfLearningKnowledgeBase, LOG_COLLECTION, RCA_COLLECTION  # noqa: E402

_DATA_DIR      = _ROOT / "data"
FIX_RATES_FILE = _DATA_DIR / "fix_success_rates.json"
_ARCHIVE_FILE  = _DATA_DIR / "rca_staging_archive.json"
_SIDECAR_FILE  = _DATA_DIR / "rca_knowledge_base" / "rca_repository.json"


# ---------------------------------------------------------------------------
# Internal: ChromaDB collection accessors
# ---------------------------------------------------------------------------

def _get_rca_kb() -> SelfLearningKnowledgeBase | None:
    try:
        return SelfLearningKnowledgeBase(persist_dir=_DATA_DIR / "rca_knowledge_base")
    except Exception:  # noqa: BLE001
        return None


def _get_log_db() -> ChromaVectorDB | None:
    try:
        return ChromaVectorDB(
            persist_dir=str(_DATA_DIR / "chroma_store"),
            collection_name=LOG_COLLECTION,
        )
    except Exception:  # noqa: BLE001
        return None


def _fetch_all_rca_metadata() -> list[dict[str, Any]]:
    """Return all metadata dicts from the rca_knowledge_base ChromaDB collection."""
    kb = _get_rca_kb()
    if kb is None:
        return []
    try:
        result = kb._collection.get(include=["metadatas"])
        return result.get("metadatas") or []
    except Exception:  # noqa: BLE001
        return []


def _fetch_all_log_metadata() -> list[dict[str, Any]]:
    """Return all metadata dicts from the ci_failure_logs ChromaDB collection."""
    db = _get_log_db()
    if db is None:
        return []
    try:
        result = db._collection.get(include=["metadatas"])
        return result.get("metadatas") or []
    except Exception:  # noqa: BLE001
        return []


def _load_fix_rates() -> dict[str, Any]:
    if FIX_RATES_FILE.exists():
        try:
            with FIX_RATES_FILE.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _load_rca_archive() -> list[dict[str, Any]]:
    """Load all entries from the RCA staging archive (rca_staging_archive.json)."""
    if _ARCHIVE_FILE.exists():
        try:
            with _ARCHIVE_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _load_rca_sidecar() -> dict[str, Any]:
    """Load rca_repository.json KB sidecar (error_signature → full RCA payload)."""
    if _SIDECAR_FILE.exists():
        try:
            with _SIDECAR_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            pass
    return {}


_CONF_LABEL_MAP = {
    "very_high": 0.92, "high": 0.80, "medium": 0.55, "low": 0.30, "very_low": 0.15,
}


def load_archive_stats() -> dict[str, Any]:
    """
    Aggregate statistics from ``data/rca_staging_archive.json``.

    The archive holds the full RCA + fix output for every completed batch
    diagnosis run.  Each entry carries a per-entry ``confidence_report``
    (with 6 factor scores) and a ``fixes`` block that includes per-candidate
    scoring and a ``validation_summary``.

    Returns
    -------
    dict with keys:
        fix_applicability_by_type   : {labels, applicable, blocked, manual_review}
        rca_confidence_hist_archive : {labels, values}   — 10-bin histogram
        fix_source_archive          : {labels, values}   — llm/historical/template
        confidence_factors_avg      : {factor_names, avg_values}
        conf_vs_applicability       : {points: [{x, y, label}]}
        archive_data_points         : int
    """
    archive = _load_rca_archive()
    if not archive:
        return {
            "fix_applicability_by_type":   {"labels": [], "applicable": [], "blocked": [], "manual_review": []},
            "rca_confidence_hist_archive":  {"labels": [], "values": []},
            "fix_source_archive":           {"labels": [], "values": []},
            "confidence_factors_avg":       {"factor_names": [], "avg_values": []},
            "conf_vs_applicability":        {"points": []},
            "archive_data_points":          0,
        }

    # --- Per error_type: applicable / blocked / manual_review counts ---
    by_type: dict[str, dict] = defaultdict(lambda: {"applicable": 0, "blocked": 0, "manual_review": 0})
    # --- Confidence score histogram (10 bins 0.0–1.0) ---
    score_bins = [0] * 10
    # --- Fix source counts ---
    source_counter: Counter = Counter()
    # --- Confidence factor accumulators ---
    factor_names  = ["kb_similarity", "evidence_count", "error_block_clarity",
                     "signal_count", "source_quality", "fix_applicability"]
    factor_sums   = {f: 0.0 for f in factor_names}
    factor_counts = {f: 0 for f in factor_names}
    # --- Confidence vs applicability scatter points ---
    scatter_points: list[dict] = []

    for entry in archive:
        rca        = entry.get("rca") or {}
        error_type = str(rca.get("error_type") or "unknown")
        fixes_obj  = rca.get("fixes") or {}
        v_summary  = fixes_obj.get("validation_summary") or {}
        fixes_list = fixes_obj.get("fixes") or []
        conf_report = rca.get("confidence_report") or {}

        # Applicability per error_type
        applicable   = int(v_summary.get("applicable_count",    0))
        blocked      = int(v_summary.get("blocked_count",       0))
        manual_rev   = int(v_summary.get("manual_review_count", 0))
        by_type[error_type]["applicable"]    += applicable
        by_type[error_type]["blocked"]       += blocked
        by_type[error_type]["manual_review"] += manual_rev

        # RCA confidence score → histogram bin
        score = conf_report.get("score")
        if isinstance(score, (int, float)):
            score = float(min(max(score, 0.0), 1.0))
            bin_idx = min(int(score * 10), 9)
            score_bins[bin_idx] += 1

            # Scatter: x = confidence score, y = applicability rate
            total_cand = int(v_summary.get("total_validated") or len(fixes_list) or 1)
            app_rate   = round(applicable / total_cand, 4) if total_cand else 0.0
            scatter_points.append({"x": round(score, 4), "y": app_rate, "label": error_type})

        # Confidence factors
        factors = conf_report.get("factors") or {}
        for fname in factor_names:
            val = factors.get(fname)
            if isinstance(val, (int, float)):
                factor_sums[fname]   += float(val)
                factor_counts[fname] += 1

        # Fix source counts
        for fix in fixes_list:
            src = str(fix.get("source") or "unknown").lower()
            source_counter[src] += 1

    # Build applicability output sorted by error_type
    sorted_types = sorted(by_type.keys())
    app_labels    = sorted_types
    app_applicable   = [by_type[t]["applicable"]    for t in sorted_types]
    app_blocked      = [by_type[t]["blocked"]        for t in sorted_types]
    app_manual       = [by_type[t]["manual_review"]  for t in sorted_types]

    # Confidence histogram labels
    hist_labels = [f"{i*10}–{(i+1)*10}%" for i in range(10)]

    # Confidence factors averages
    factor_avgs = [
        round(factor_sums[f] / factor_counts[f], 4) if factor_counts[f] else 0.0
        for f in factor_names
    ]

    # Fix source: group llm variants under "llm"
    normalized_src: Counter = Counter()
    for src, cnt in source_counter.items():
        key = src if src in ("llm", "historical", "template") else "other"
        normalized_src[key] += cnt
    src_labels = list(normalized_src.keys())
    src_values = list(normalized_src.values())

    return {
        "fix_applicability_by_type": {
            "labels":       app_labels,
            "applicable":   app_applicable,
            "blocked":      app_blocked,
            "manual_review": app_manual,
        },
        "rca_confidence_hist_archive": {
            "labels": hist_labels,
            "values": score_bins,
        },
        "fix_source_archive": {
            "labels": src_labels,
            "values": src_values,
        },
        "confidence_factors_avg": {
            "factor_names": factor_names,
            "avg_values":   factor_avgs,
        },
        "conf_vs_applicability": {"points": scatter_points},
        "archive_data_points":   len(archive),
    }


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------

def load_rca_stats_from_chroma() -> dict[str, Any]:
    """
    Aggregate RCA-level statistics from the ``rca_knowledge_base`` ChromaDB collection.

    Returns
    -------
    dict with keys:
        error_type_counts  : {label: str, values: list[int]}
        source_counts      : {label: str, values: list[int]}
        confidence_histogram : {labels: list[str], values: list[int]}
        kb_growth          : {labels: list[str], values: list[int]}  — cumulative by date
        data_points        : int
    """
    metas = _fetch_all_rca_metadata()

    # --- Error type counts ---
    error_type_counter: Counter = Counter()
    source_counter: Counter     = Counter()
    confidence_values: list[float] = []
    dates: list[str]            = []

    for m in metas:
        if not isinstance(m, dict):
            continue
        et = str(m.get("error_type") or "unknown")
        error_type_counter[et] += 1

        src = str(m.get("source") or "unknown")
        source_counter[src] += 1

        raw_conf = m.get("confidence")
        if raw_conf is not None:
            try:
                conf_map = {"very_high": 0.92, "high": 0.80, "medium": 0.55,
                            "low": 0.30, "very_low": 0.15}
                if isinstance(raw_conf, str):
                    confidence_values.append(conf_map.get(raw_conf.strip().lower(), 0.5))
                else:
                    confidence_values.append(float(raw_conf))
            except (ValueError, TypeError):
                pass

        gen_at = str(m.get("generated_at") or "")
        if gen_at:
            dates.append(gen_at[:10])   # YYYY-MM-DD

    # --- Confidence histogram (10 bins: 0-0.1, 0.1-0.2, … 0.9-1.0) ---
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_labels = [f"{int(b*100)}–{int(bins[i+1]*100)}%" for i, b in enumerate(bins[:-1])]
    bin_counts = [0] * 10
    for v in confidence_values:
        idx = min(int(v * 10), 9)
        bin_counts[idx] += 1

    # --- KB growth curve (cumulative count by date) ---
    date_counter: Counter = Counter(dates)
    sorted_dates = sorted(date_counter)
    cumulative   = []
    running      = 0
    for d in sorted_dates:
        running += date_counter[d]
        cumulative.append(running)

    # Sort error types descending
    top_errors = error_type_counter.most_common(10)

    return {
        "error_type_counts": {
            "labels": [x[0] for x in top_errors],
            "values": [x[1] for x in top_errors],
        },
        "source_counts": {
            "labels": list(source_counter.keys()),
            "values": list(source_counter.values()),
        },
        "confidence_histogram": {
            "labels": bin_labels,
            "values": bin_counts,
        },
        "kb_growth": {
            "labels": sorted_dates,
            "values": cumulative,
        },
        "data_points": len(metas),
    }


def load_log_stats_from_chroma() -> dict[str, Any]:
    """
    Aggregate log-level statistics from the ``ci_failure_logs`` ChromaDB collection.

    Returns
    -------
    dict with keys:
        severity_counts    : {labels: list[str], values: list[int]}
        repo_freq          : {labels: list[str], values: list[int]}
        error_severity_matrix : {error_types: list[str], matrix: {sev: list[int]}}
        data_points        : int
    """
    metas = _fetch_all_log_metadata()

    severity_counter: Counter                  = Counter()
    repo_counter:     Counter                  = Counter()
    # matrix[error_type][severity] = count
    matrix: dict[str, Counter] = defaultdict(Counter)

    for m in metas:
        if not isinstance(m, dict):
            continue
        sev = str(m.get("severity") or "unknown").lower()
        severity_counter[sev] += 1

        repo = str(m.get("repository") or "unknown")
        repo_counter[repo] += 1

        et  = str(m.get("error_type") or "unknown")
        matrix[et][sev] += 1

    # Top 8 repos, top 8 error types for matrix
    top_repos   = repo_counter.most_common(8)
    top_errors  = sorted(matrix.keys(), key=lambda k: sum(matrix[k].values()), reverse=True)[:8]
    severities  = ["high", "medium", "low"]

    matrix_data: dict[str, list[int]] = {
        sev: [matrix[et][sev] for et in top_errors] for sev in severities
    }

    return {
        "severity_counts": {
            "labels": list(severity_counter.keys()),
            "values": list(severity_counter.values()),
        },
        "repo_freq": {
            "labels": [x[0] for x in top_repos],
            "values": [x[1] for x in top_repos],
        },
        "error_severity_matrix": {
            "error_types": top_errors,
            "matrix":      matrix_data,
        },
        "data_points": len(metas),
    }


def load_fix_stats() -> dict[str, Any]:
    """
    Aggregate fix-level statistics from ``data/fix_success_rates.json``.

    Returns
    -------
    dict with keys:
        fix_success_rate : {labels: list[str], success: list[int], total: list[int]}
        fix_score_dist   : {labels: list[str], values: list[float]}  — success ratio
        fix_source_dist  : {labels: list[str], series: dict}  — stacked by fix_type prefix
        data_points      : int
    """
    rates = _load_fix_rates()
    if not rates:
        return {
            "fix_success_rate": {"labels": [], "success": [], "total": []},
            "fix_score_dist":   {"labels": [], "values": []},
            "fix_source_dist":  {"labels": [], "series": {}},
            "data_points":      0,
        }

    labels:   list[str]   = []
    success:  list[int]   = []
    total_:   list[int]   = []
    ratios:   list[float] = []

    # Group by error_type (key = "fix_type:error_type")
    by_error: dict[str, dict] = defaultdict(lambda: {"success": 0, "total": 0})
    fix_types_seen: Counter   = Counter()

    for key, entry in rates.items():
        parts = key.split(":", 1)
        fix_t = parts[0] if len(parts) == 2 else "unknown"
        err_t = parts[1] if len(parts) == 2 else parts[0]
        s     = int(entry.get("success", 0))
        t     = int(entry.get("total",   0))
        by_error[err_t]["success"] += s
        by_error[err_t]["total"]   += t
        fix_types_seen[fix_t]      += t

    for err_t, agg in sorted(by_error.items()):
        labels.append(err_t)
        success.append(agg["success"])
        total_.append(agg["total"])
        ratios.append(round(agg["success"] / agg["total"], 4) if agg["total"] else 0.0)

    # Fix source distribution: fix_type totals as a simple bar
    fix_src_labels = list(fix_types_seen.keys())
    fix_src_values = list(fix_types_seen.values())

    return {
        "fix_success_rate": {"labels": labels, "success": success, "total": total_},
        "fix_score_dist":   {"labels": labels, "values": ratios},
        "fix_source_dist":  {
            "labels": fix_src_labels,
            "series": {fix_t: [fix_types_seen[fix_t]] for fix_t in fix_src_labels},
        },
        "data_points": len(rates),
    }


# ---------------------------------------------------------------------------
# Public: combined payload builder
# ---------------------------------------------------------------------------

def build_stats_payload() -> dict[str, Any]:
    """
    Build the complete stats payload for the ``GET /api/stats/summary`` endpoint.

    Merges RCA ChromaDB stats, log ChromaDB stats, fix file stats, and
    archive-derived stats into a single dict consumed by
    ``stats.html`` / ``static/js/charts.js``.
    """
    rca_stats     = load_rca_stats_from_chroma()
    log_stats     = load_log_stats_from_chroma()
    fix_stats     = load_fix_stats()
    archive_stats = load_archive_stats()

    total_data_points = (
        rca_stats["data_points"]
        + log_stats["data_points"]
        + fix_stats["data_points"]
        + archive_stats["archive_data_points"]
    )

    return {
        # --- Existing ChromaDB / KB charts ---
        "error_type_counts":      rca_stats["error_type_counts"],
        "confidence_histogram":   rca_stats["confidence_histogram"],
        "severity_counts":        log_stats["severity_counts"],
        "repo_freq":              log_stats["repo_freq"],
        "error_severity_matrix":  log_stats["error_severity_matrix"],
        "kb_growth":              rca_stats["kb_growth"],
        # --- Archive-powered charts (replace sparse fix/source charts) ---
        "fix_applicability_by_type":   archive_stats["fix_applicability_by_type"],
        "rca_confidence_hist_archive":  archive_stats["rca_confidence_hist_archive"],
        "fix_source_archive":           archive_stats["fix_source_archive"],
        # --- Kept for evaluation page & backward compat (may be sparse) ---
        "fix_success_rate":       fix_stats["fix_success_rate"],
        "fix_source_dist":        fix_stats["fix_source_dist"],
        "fix_score_dist":         fix_stats["fix_score_dist"],
        "rca_source_counts":      rca_stats["source_counts"],
        # Metadata
        "total_data_points":      total_data_points,
        "rca_data_points":        rca_stats["data_points"],
        "log_data_points":        log_stats["data_points"],
        "fix_data_points":        fix_stats["data_points"],
        "archive_data_points":    archive_stats["archive_data_points"],
    }
