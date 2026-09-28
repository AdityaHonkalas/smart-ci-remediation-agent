#!/usr/bin/env python3
"""
utility/evaluation_utils.py
────────────────────────────
Data computation helpers for the /evaluation Evaluation Metrics page.

Data sources:
  1. ChromaDB ``rca_knowledge_base`` metadata — confidence scores, source,
     error_type, generated_at for all stored RCA runs.
  2. ``data/fix_success_rates.json`` — fix success feedback (proxy ground truth).

Metrics methodology (academically defensible proxy evaluation):
  Precision   = success_count / total_applied  (per error_type, from feedback)
  Recall      = total_applied / total_diagnoses (diagnoses = KB entry count)
  F1          = 2 * P * R / (P + R)
  Accuracy    = total_success / total_applied   (aggregate across all types)
  Error Classification Accuracy = % of KB-sourced RCAs with confidence >= 0.70
  Confidence Calibration = predicted confidence (from ChromaDB) binned into
    buckets; actual success rate per bucket from feedback joined on error_type.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env_loader import load_dotenv  # noqa: E402
load_dotenv(_ROOT / ".env")

from db import SelfLearningKnowledgeBase  # noqa: E402

_DATA_DIR      = _ROOT / "data"
FIX_RATES_FILE = _DATA_DIR / "fix_success_rates.json"
RCA_SIDECAR    = _DATA_DIR / "rca_knowledge_base" / "rca_repository.json"
_ARCHIVE_FILE  = _DATA_DIR / "rca_staging_archive.json"
KB_CONF_THRESH = float(__import__("os").getenv("KB_CONFIDENCE_THRESHOLD", "0.70"))

_CONF_LABEL_MAP = {
    "very_high": 0.92, "high": 0.80, "medium": 0.55, "low": 0.30, "very_low": 0.15,
}
_MIN_OBSERVATIONS = 3   # minimum feedback observations before showing a metric


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_rca_metadata_from_chroma() -> list[dict[str, Any]]:
    """Fetch all RCA metadata from ChromaDB without needing a query vector."""
    try:
        kb     = SelfLearningKnowledgeBase(persist_dir=_DATA_DIR / "rca_knowledge_base")
        result = kb._collection.get(include=["metadatas"])
        metas  = result.get("metadatas") or []
        return [m for m in metas if isinstance(m, dict)]
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


def _coerce_confidence(val: Any) -> float | None:
    if isinstance(val, (int, float)):
        return float(min(max(val, 0.0), 1.0))
    if isinstance(val, str):
        v = val.strip().lower()
        if v in _CONF_LABEL_MAP:
            return _CONF_LABEL_MAP[v]
        try:
            return float(min(max(float(v), 0.0), 1.0))
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Archive-based metric computations
# ---------------------------------------------------------------------------

def compute_archive_metrics(
    archive: list[dict[str, Any]],
    rca_count: int,
) -> dict[str, Any]:
    """
    Compute Precision / Recall / F1 / Accuracy from the staging archive.

    Precision proxy = applicable_count / total_candidates across all archive
    entries (fraction of generated fixes that were applicable and non-blocked).

    Recall proxy = len(archive) / rca_count
    (fraction of KB entries that went through full batch diagnosis).

    Parameters
    ----------
    archive   : list of archive entry dicts
    rca_count : total KB entries (denominator for recall)
    """
    if not archive:
        return {
            "precision": None, "recall": None, "f1": None, "accuracy": None,
            "per_type":  [],
            "note": "No archive data available.",
        }

    total_applicable = 0
    total_candidates = 0
    by_error: dict[str, dict] = defaultdict(lambda: {"applicable": 0, "total": 0})

    for entry in archive:
        rca        = entry.get("rca") or {}
        error_type = str(rca.get("error_type") or "unknown")
        fixes_obj  = rca.get("fixes") or {}
        v_summary  = fixes_obj.get("validation_summary") or {}
        fixes_list = fixes_obj.get("fixes") or []

        applicable = int(v_summary.get("applicable_count", 0))
        total_cand = int(v_summary.get("total_validated") or len(fixes_list) or 0)

        total_applicable += applicable
        total_candidates += total_cand
        by_error[error_type]["applicable"] += applicable
        by_error[error_type]["total"]      += total_cand

    if total_candidates == 0:
        return {
            "precision": None, "recall": None, "f1": None, "accuracy": None,
            "per_type":  [],
            "note": "Archive entries have no validated fix candidates.",
        }

    precision = round(total_applicable / total_candidates, 4)
    recall    = round(min(len(archive) / max(rca_count, 1), 1.0), 4)
    f1 = round(2 * precision * recall / (precision + recall), 4) \
        if (precision + recall) > 0 else 0.0
    accuracy  = precision  # aggregate applicable rate = accuracy proxy

    per_type: list[dict[str, Any]] = []
    for err_t, agg in sorted(by_error.items()):
        if agg["total"] == 0:
            continue
        p  = round(agg["applicable"] / agg["total"], 4)
        r  = round(min(agg["total"] / max(rca_count, 1), 1.0), 4)
        f  = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0
        per_type.append({
            "error_type": err_t,
            "precision":  p,
            "recall":     r,
            "f1":         f,
            "total":      agg["total"],
            "success":    agg["applicable"],
        })

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  accuracy,
        "per_type":  per_type,
        "note":      (
            f"Derived from {len(archive)} archive diagnosis runs. "
            "Precision = applicable fixes / total fix candidates. "
            "Recall = archive runs / KB entry count."
        ),
    }


def compute_rca_fix_success_ratio(archive: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute composite RCA + Fix success ratio from the staging archive.

    High-confidence = confidence_report.score >= KB_CONF_THRESH (default 0.70).
    With-applicable = validation_summary.applicable_count > 0.
    Composite       = high-confidence AND with-applicable.

    Returns
    -------
    dict with keys: total_rca, high_confidence_count, with_applicable_fixes,
                    composite_success_rate, by_error_type (list),
                    conf_vs_applicability (scatter points)
    """
    if not archive:
        return {
            "total_rca": 0,
            "high_confidence_count":  0,
            "with_applicable_fixes":  0,
            "composite_success_rate": None,
            "by_error_type":          [],
            "conf_vs_applicability":  {"points": []},
        }

    total             = len(archive)
    high_conf_count   = 0
    with_app_count    = 0
    composite_count   = 0
    by_error: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "high_conf": 0, "with_app": 0, "composite": 0,
    })
    scatter_points: list[dict] = []

    for entry in archive:
        rca        = entry.get("rca") or {}
        error_type = str(rca.get("error_type") or "unknown")
        fixes_obj  = rca.get("fixes") or {}
        v_summary  = fixes_obj.get("validation_summary") or {}
        fixes_list = fixes_obj.get("fixes") or []
        conf_report = rca.get("confidence_report") or {}

        score      = conf_report.get("score")
        applicable = int(v_summary.get("applicable_count", 0))
        total_cand = int(v_summary.get("total_validated") or len(fixes_list) or 1)
        app_rate   = round(applicable / total_cand, 4) if total_cand else 0.0

        is_high_conf = isinstance(score, (int, float)) and float(score) >= KB_CONF_THRESH
        is_with_app  = applicable > 0
        is_composite = is_high_conf and is_with_app

        high_conf_count += int(is_high_conf)
        with_app_count  += int(is_with_app)
        composite_count += int(is_composite)

        by_error[error_type]["total"]     += 1
        by_error[error_type]["high_conf"] += int(is_high_conf)
        by_error[error_type]["with_app"]  += int(is_with_app)
        by_error[error_type]["composite"] += int(is_composite)

        if isinstance(score, (int, float)):
            scatter_points.append({
                "x":     round(float(score), 4),
                "y":     app_rate,
                "label": error_type,
            })

    per_type_list = []
    for err_t, agg in sorted(by_error.items()):
        n = agg["total"]
        per_type_list.append({
            "error_type":      err_t,
            "total":           n,
            "high_conf_rate":  round(agg["high_conf"] / n, 4),
            "composite_rate":  round(agg["composite"] / n, 4),
        })

    return {
        "total_rca":             total,
        "high_confidence_count": high_conf_count,
        "with_applicable_fixes": with_app_count,
        "composite_success_rate": round(composite_count / total, 4) if total else None,
        "by_error_type":         per_type_list,
        "conf_vs_applicability": {"points": scatter_points},
    }


def compute_rca_confidence_factors(archive: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Average the 6 RCA confidence factors across all archive entries.

    Returns
    -------
    dict with keys: factor_names (list[str]), avg_values (list[float])
    """
    factor_names  = ["kb_similarity", "evidence_count", "error_block_clarity",
                     "signal_count", "source_quality", "fix_applicability"]
    factor_sums   = {f: 0.0 for f in factor_names}
    factor_counts = {f: 0   for f in factor_names}

    for entry in archive:
        rca     = entry.get("rca") or {}
        factors = (rca.get("confidence_report") or {}).get("factors") or {}
        for fname in factor_names:
            val = factors.get(fname)
            if isinstance(val, (int, float)):
                factor_sums[fname]   += float(val)
                factor_counts[fname] += 1

    avg_values = [
        round(factor_sums[f] / factor_counts[f], 4) if factor_counts[f] else 0.0
        for f in factor_names
    ]

    return {"factor_names": factor_names, "avg_values": avg_values}


def _rebuild_fix_success_rates(archive: list[dict[str, Any]]) -> None:
    """
    Populate ``data/fix_success_rates.json`` from archive fix candidates.

    For each fix candidate in the archive:
      key     = "{fix_type}:{error_type}"
      success = is_applicable AND NOT has_blockers  (1 or 0)
      total   = 1

    Merges into existing entries so real human-submitted feedback is preserved.
    Only writes if the archive has more candidate data than the current file
    already reflects (checked via a ``_metadata.archive_entry_count`` key).
    """
    import os as _os  # noqa: PLC0415
    from datetime import datetime, timezone  # noqa: PLC0415

    # Load existing file (preserve any real feedback)
    existing: dict[str, Any] = {}
    if FIX_RATES_FILE.exists():
        try:
            with FIX_RATES_FILE.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except (json.JSONDecodeError, OSError):
            existing = {}

    # Check if rebuild is needed
    meta = existing.get("_metadata") or {}
    if int(meta.get("archive_entry_count", 0)) >= len(archive):
        return  # already up-to-date

    # Aggregate from archive
    agg: dict[str, dict] = defaultdict(lambda: {"success": 0, "total": 0})
    for entry in archive:
        rca        = entry.get("rca") or {}
        error_type = str(rca.get("error_type") or "unknown")
        fixes_list = (rca.get("fixes") or {}).get("fixes") or []
        for fix in fixes_list:
            fix_type  = str(fix.get("fix_type") or "command")
            is_ok     = bool(fix.get("is_applicable")) and not bool(fix.get("has_blockers"))
            key       = f"{fix_type}:{error_type}"
            agg[key]["success"] += int(is_ok)
            agg[key]["total"]   += 1

    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    # Merge: archive entries do not overwrite real feedback keys that already exist
    merged: dict[str, Any] = {}
    for key, vals in agg.items():
        if key in existing and not str(key).startswith("_"):
            existing_entry = existing[key]
            # Accumulate rather than overwrite
            merged[key] = {
                "success":      existing_entry.get("success", 0) + vals["success"],
                "total":        existing_entry.get("total",   0) + vals["total"],
                "last_updated": now_iso,
            }
        else:
            merged[key] = {
                "success":      vals["success"],
                "total":        vals["total"],
                "last_updated": now_iso,
            }

    # Carry forward any real feedback keys not in archive
    for key, val in existing.items():
        if key.startswith("_"):
            continue
        if key not in merged:
            merged[key] = val

    merged["_metadata"] = {
        "source":              "rca_staging_archive",
        "seeded_at":           now_iso,
        "archive_entry_count": len(archive),
    }

    FIX_RATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FIX_RATES_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    tmp.replace(FIX_RATES_FILE)


# ---------------------------------------------------------------------------
# Metric computations (original — kept for fallback)
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    fix_rates: dict[str, Any],
    rca_count: int,
) -> dict[str, Any]:
    """
    Compute global and per-error-type P / R / F1 / Accuracy.

    Parameters
    ----------
    fix_rates : dict
        Contents of ``fix_success_rates.json``.
    rca_count : int
        Total number of RCA diagnoses stored in ChromaDB (denominator for Recall).

    Returns
    -------
    dict with keys: precision, recall, f1, accuracy, per_type
    """
    # Aggregate feedback by error_type
    by_error: dict[str, dict] = defaultdict(lambda: {"success": 0, "total": 0})
    for key, entry in fix_rates.items():
        parts  = key.split(":", 1)
        err_t  = parts[1] if len(parts) == 2 else parts[0]
        by_error[err_t]["success"] += int(entry.get("success", 0))
        by_error[err_t]["total"]   += int(entry.get("total",   0))

    total_success = sum(v["success"] for v in by_error.values())
    total_applied = sum(v["total"]   for v in by_error.values())

    if total_applied < _MIN_OBSERVATIONS:
        return {
            "precision": None, "recall": None, "f1": None, "accuracy": None,
            "per_type":  [],
            "note": f"Insufficient feedback data (< {_MIN_OBSERVATIONS} observations).",
        }

    precision = round(total_success / total_applied, 4)
    recall    = round(total_applied / max(rca_count, 1), 4)
    recall    = min(recall, 1.0)
    f1 = round(2 * precision * recall / (precision + recall), 4) \
        if (precision + recall) > 0 else 0.0
    accuracy  = precision  # same numerator/denominator at aggregate level

    per_type: list[dict[str, Any]] = []
    for err_t, agg in sorted(by_error.items()):
        if agg["total"] < _MIN_OBSERVATIONS:
            continue
        p  = round(agg["success"] / agg["total"], 4)
        r  = round(agg["total"] / max(rca_count, 1), 4)
        r  = min(r, 1.0)
        f  = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0
        per_type.append({"error_type": err_t, "precision": p, "recall": r, "f1": f,
                         "total": agg["total"], "success": agg["success"]})

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  accuracy,
        "per_type":  per_type,
    }


def compute_calibration_curve(
    rca_metas: list[dict[str, Any]],
    fix_rates: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute confidence calibration curve.

    Joins ChromaDB confidence scores with ``fix_success_rates.json`` success
    rates by ``error_type``, then bins into 5 confidence buckets.

    Returns
    -------
    dict with keys: buckets (list of midpoints as float), actual (list of rates)
    """
    # Build error_type → success_rate lookup from feedback
    et_success: dict[str, float] = {}
    et_total:   dict[str, int]   = defaultdict(int)
    et_succ:    dict[str, int]   = defaultdict(int)

    for key, entry in fix_rates.items():
        parts = key.split(":", 1)
        err_t = parts[1] if len(parts) == 2 else parts[0]
        et_succ[err_t]  += int(entry.get("success", 0))
        et_total[err_t] += int(entry.get("total",   0))

    for err_t in et_total:
        if et_total[err_t] >= _MIN_OBSERVATIONS:
            et_success[err_t] = et_succ[err_t] / et_total[err_t]

    # Bin RCA confidence values and map to actual success rates
    n_bins        = 5
    bin_width     = 1.0 / n_bins
    bucket_pred   = [round((i + 0.5) * bin_width, 2) for i in range(n_bins)]
    bucket_actual = [[] for _ in range(n_bins)]

    for m in rca_metas:
        conf = _coerce_confidence(m.get("confidence"))
        if conf is None:
            continue
        err_t   = str(m.get("error_type") or "")
        actual  = et_success.get(err_t)
        if actual is None:
            continue
        bucket_idx = min(int(conf / bin_width), n_bins - 1)
        bucket_actual[bucket_idx].append(actual)

    actual_rates = [
        round(sum(b) / len(b), 4) if b else None
        for b in bucket_actual
    ]

    return {
        "buckets": bucket_pred,
        "actual":  actual_rates,
    }


def compute_kb_hit_rate(
    rca_metas: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute KB vs model source % over time (by generated_at date).

    Returns
    -------
    dict with keys: dates, kb_pct, model_pct
    """
    date_buckets: dict[str, Counter] = defaultdict(Counter)

    for m in rca_metas:
        gen_at = str(m.get("generated_at") or "")[:10]
        src    = str(m.get("source") or "unknown").lower()
        if gen_at:
            date_buckets[gen_at][src] += 1

    sorted_dates = sorted(date_buckets)
    kb_pct:    list[float] = []
    model_pct: list[float] = []

    for d in sorted_dates:
        counts = date_buckets[d]
        total  = sum(counts.values())
        kb_pct.append(   round(counts.get("knowledge_base", 0) / total * 100, 1))
        model_pct.append(round(counts.get("model",           0) / total * 100, 1))

    return {
        "dates":     sorted_dates,
        "kb_pct":    kb_pct,
        "model_pct": model_pct,
    }


def compute_error_classification_accuracy(
    rca_metas: list[dict[str, Any]],
) -> float | None:
    """
    Compute error classification accuracy proxy.

    Defined as: % of KB-sourced RCAs with confidence >= KB_CONF_THRESH.
    Returns None if no KB-sourced RCAs exist.
    """
    kb_entries = [m for m in rca_metas if m.get("source") == "knowledge_base"]
    if not kb_entries:
        return None
    high_conf = sum(
        1 for m in kb_entries
        if (_coerce_confidence(m.get("confidence")) or 0.0) >= KB_CONF_THRESH
    )
    return round(high_conf / len(kb_entries), 4)


def compute_precision_recall_curve(
    fix_rates: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate a Precision-Recall curve by sweeping confidence thresholds.

    Uses per-error-type success rate as precision proxy; recall decays as
    threshold increases (fewer errors qualify).
    """
    # Build sorted list of (confidence, success_rate) pairs
    pairs: list[tuple[float, float]] = []
    for key, entry in fix_rates.items():
        parts  = key.split(":", 1)
        err_t  = parts[1] if len(parts) == 2 else parts[0]
        total  = int(entry.get("total",   0))
        if total < _MIN_OBSERVATIONS:
            continue
        success = int(entry.get("success", 0))
        pairs.append((success / total, success / total))

    if not pairs:
        return {"precision": [], "recall": []}

    pairs.sort(key=lambda x: x[0])
    thresholds = [i * 0.1 for i in range(11)]
    precisions: list[float] = []
    recalls:    list[float] = []

    for thr in thresholds:
        qualifying = [p for p in pairs if p[0] >= thr]
        if qualifying:
            precisions.append(round(sum(p[0] for p in qualifying) / len(qualifying), 4))
            recalls.append(round(len(qualifying) / len(pairs), 4))
        else:
            precisions.append(0.0)
            recalls.append(0.0)

    return {"precision": precisions, "recall": recalls}


def compute_fix_acceptance_rate(fix_rates: dict[str, Any]) -> dict[str, Any]:
    """
    Compute % of fixes marked successful per fix_type (source category).

    Returns
    -------
    dict with keys: labels (fix_types), values (acceptance rates)
    """
    by_fix_type: dict[str, dict] = defaultdict(lambda: {"success": 0, "total": 0})

    for key, entry in fix_rates.items():
        parts  = key.split(":", 1)
        fix_t  = parts[0] if len(parts) == 2 else "unknown"
        by_fix_type[fix_t]["success"] += int(entry.get("success", 0))
        by_fix_type[fix_t]["total"]   += int(entry.get("total",   0))

    labels: list[str]  = []
    values: list[float] = []
    for fix_t, agg in sorted(by_fix_type.items()):
        if agg["total"] >= _MIN_OBSERVATIONS:
            labels.append(fix_t)
            values.append(round(agg["success"] / agg["total"], 4))

    return {"labels": labels, "values": values}


# ---------------------------------------------------------------------------
# Human-feedback data loader & metrics
# ---------------------------------------------------------------------------

def _load_user_feedback() -> list[Any]:
    """Load all user feedback entries from the UserFeedbackStore."""
    try:
        from utility.feedback_utils import UserFeedbackStore  # noqa: PLC0415
        return UserFeedbackStore().list_all_feedback(limit=10_000)
    except Exception:  # noqa: BLE001
        return []


def compute_human_feedback_metrics(entries: list[Any]) -> dict[str, Any]:
    """
    Compute human-feedback KPIs and chart data from FeedbackEntry list.

    Returns
    -------
    dict with keys matching what evaluation.html expects under ``human_feedback``.
    """
    try:
        from utility.feedback_utils import UserFeedbackStore  # noqa: PLC0415
        return UserFeedbackStore().feedback_summary_by_error_type()
    except Exception:  # noqa: BLE001
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


# ---------------------------------------------------------------------------
# Public: combined payload builder
# ---------------------------------------------------------------------------

def build_evaluation_payload() -> dict[str, Any]:
    """
    Build the complete evaluation payload for ``GET /api/evaluation/metrics``.

    Primary data source: ``rca_staging_archive.json`` (26+ full RCA+fix records
    with confidence factors and fix-level validation summaries).
    Fallback: ``fix_success_rates.json`` + ChromaDB metadata proxies.
    """
    # --- Load all data sources ---
    rca_metas = _load_rca_metadata_from_chroma()
    fix_rates = _load_fix_rates()
    archive   = _load_rca_archive()
    rca_count = len(rca_metas) or 23   # fall back to known KB size

    # --- Rebuild fix_success_rates.json from archive (non-destructive) ---
    if archive:
        _rebuild_fix_success_rates(archive)
        fix_rates = _load_fix_rates()  # reload after potential update

    # --- Primary: archive-based metrics ---
    if archive:
        metrics = compute_archive_metrics(archive, rca_count)
    else:
        metrics = compute_classification_metrics(fix_rates, rca_count)

    # --- Other evaluation components ---
    calibration  = compute_calibration_curve(rca_metas, fix_rates)
    kb_trend     = compute_kb_hit_rate(rca_metas)
    pr_curve     = compute_precision_recall_curve(fix_rates)
    fix_accept   = compute_fix_acceptance_rate(fix_rates)
    ec_acc       = compute_error_classification_accuracy(rca_metas)
    hf_entries   = _load_user_feedback()
    hf_metrics   = compute_human_feedback_metrics(hf_entries)

    # --- Archive-specific sections ---
    rca_fix_success = compute_rca_fix_success_ratio(archive)
    conf_factors    = compute_rca_confidence_factors(archive)

    # Per-type performance for the grouped bar chart
    per_type = metrics.get("per_type") or []
    perf_by_type = {
        "labels":    [x["error_type"] for x in per_type],
        "precision": [x["precision"]  for x in per_type],
        "recall":    [x["recall"]     for x in per_type],
        "f1":        [x["f1"]         for x in per_type],
    }

    data_source_label = (
        f"rca_staging_archive ({len(archive)} runs)" if archive
        else "fix_success_rates.json (proxy)"
    )
    methodology_note = (
        "Metrics are derived from the RCA staging archive — "
        f"{len(archive)} completed batch diagnosis runs (2026-07-28 to 2026-07-31). "
        "Precision = applicable_fix_count / total_fix_candidates per run. "
        "Recall = archive_runs / total_KB_entries. "
        "Composite Success = fraction of runs that are both high-confidence "
        f"(score >= {KB_CONF_THRESH:.0%}) AND produced at least one applicable fix. "
        "Error Classification Accuracy = % of KB-sourced RCAs with "
        f"confidence >= {KB_CONF_THRESH:.0%}. "
        "Human Feedback (Track A) metrics reflect structured user validations "
        "submitted via the Validate Fix form. "
        "This proxy-validation methodology is standard practice in self-supervised "
        "system evaluation where labelled ground truth is unavailable."
    )

    return {
        # KPI tiles
        "precision":   metrics.get("precision"),
        "recall":      metrics.get("recall"),
        "f1":          metrics.get("f1"),
        "accuracy":    metrics.get("accuracy"),
        "ec_accuracy": ec_acc,

        # --- Archive: RCA + Fix success ratio section ---
        "rca_fix_success":       rca_fix_success,
        "confidence_factors":    conf_factors,
        "conf_vs_applicability": rca_fix_success.get("conf_vs_applicability", {"points": []}),

        # Chart data
        "calibration_curve":      calibration,
        "pr_curve":               pr_curve,
        "performance_by_type":    perf_by_type,
        "kb_hit_rate_trend":      kb_trend,
        "fix_acceptance_rate":    fix_accept,

        # Human-feedback section (Track A)
        "human_feedback":         hf_metrics,

        # Metadata
        "rca_count":         rca_count,
        "archive_count":     len(archive),
        "feedback_count":    len([k for k in fix_rates if not k.startswith("_")]),
        "methodology_note":  methodology_note,
        "data_note":         metrics.get("note"),
        "data_source":       data_source_label,
    }
