#!/usr/bin/env python3
"""
utility/confidence_validator.py
────────────────────────────────
Lightweight explainable confidence validator for RCA, fix recommendations,
and web search results.

Replaces the ad-hoc label mapping ("low"→0.25, "medium"→0.50, "high"→0.85)
with a calibrated, multi-factor score that shows which signals contributed to
the confidence and why.  No heavy ML dependencies — all scoring is algebraic.

Output contract
───────────────
All ``validate_*`` methods return a ``ConfidenceReport`` dataclass:
{
    "score":       float          — calibrated confidence in [0, 1]
    "label":       str            — "very_high"|"high"|"medium"|"low"|"very_low"
    "explanation": str            — human-readable summary of top contributing factors
    "factors":     dict[str,float]  — per-factor scores (each in [0, 1])
}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceReport:
    """Explainable confidence result returned by ConfidenceValidator methods."""
    score:       float
    label:       str
    explanation: str
    factors:     dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score":       round(self.score, 4),
            "label":       self.label,
            "explanation": self.explanation,
            "factors":     {k: round(v, 4) for k, v in self.factors.items()},
        }


# ---------------------------------------------------------------------------
# Label thresholds
# ---------------------------------------------------------------------------

_LABEL_THRESHOLDS = [
    (0.85, "very_high"),
    (0.70, "high"),
    (0.50, "medium"),
    (0.30, "low"),
    (0.00, "very_low"),
]

def _score_to_label(score: float) -> str:
    for threshold, label in _LABEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "very_low"


# ---------------------------------------------------------------------------
# Source quality map
# ---------------------------------------------------------------------------

_SOURCE_QUALITY: dict[str, float] = {
    "knowledge_base":     0.90,
    "model":              0.80,
    "heuristic_fallback": 0.50,
    "web_search":         0.60,
}

_COERCE_CONFIDENCE: dict[str, float] = {
    "very_high": 0.90,
    "high":      0.80,
    "medium":    0.55,
    "low":       0.30,
    "very_low":  0.15,
}


def _coerce(value: Any) -> float:
    """Coerce a confidence value (string label or float) to float in [0,1]."""
    if isinstance(value, (int, float)):
        return float(min(max(value, 0.0), 1.0))
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _COERCE_CONFIDENCE:
            return _COERCE_CONFIDENCE[v]
        try:
            return float(min(max(float(v), 0.0), 1.0))
        except ValueError:
            pass
    return 0.50  # neutral prior


def _top_factors(factors: dict[str, float], n: int = 2) -> list[str]:
    """Return names of the top-N factors by score."""
    return [k for k, _ in sorted(factors.items(), key=lambda x: x[1], reverse=True)[:n]]


# ---------------------------------------------------------------------------
# ConfidenceValidator
# ---------------------------------------------------------------------------

class ConfidenceValidator:
    """
    Computes explainable confidence scores for RCA, fix recommendations,
    and web search results.

    Usage
    -----
    validator = ConfidenceValidator()

    rca_report  = validator.validate_rca(rca_result, signals, kb_hits)
    fix_report  = validator.validate_fix(fix_candidate_dict)
    web_report  = validator.validate_web(search_results)
    """

    # ------------------------------------------------------------------
    # RCA confidence
    # ------------------------------------------------------------------

    def validate_rca(
        self,
        rca_result:  dict[str, Any],
        signals:     list[Any],
        kb_hits:     list[dict[str, Any]],
    ) -> ConfidenceReport:
        """
        Validate RCA confidence using 6 factors.

        Factors and weights:
          kb_similarity      30%  — best cosine similarity from KB search
          evidence_count     15%  — number of evidence items (capped at 5)
          error_block_clarity 15% — primary_error has file + line info
          signal_count       15%  — number of extracted failure signals (capped at 10)
          source_quality     15%  — model > knowledge_base > heuristic_fallback
          fix_applicability  10%  — ratio of applicable fixes

        Parameters
        ----------
        rca_result : dict
            The normalised RCA dict from ``normalize_rca_report()``.
        signals : list
            Failure signals extracted by the preprocessing pipeline.
        kb_hits : list[dict]
            Top-k KB search results (each with a ``similarity`` key).
        """
        # Factor 1: KB similarity
        kb_sim = max(
            (float(e.get("similarity", 0.0)) for e in kb_hits),
            default=0.0,
        )

        # Factor 2: Evidence count (0 → 0.0, 5+ → 1.0)
        evidence = rca_result.get("evidence") or []
        evidence_score = min(len(evidence) / 5.0, 1.0)

        # Factor 3: Error block clarity (file + line present)
        fl = rca_result.get("failure_location") or {}
        has_file = bool(fl.get("file") or rca_result.get("file_path"))
        has_line = bool(fl.get("line") or rca_result.get("line_number"))
        clarity_score = 0.5 * int(has_file) + 0.5 * int(has_line)

        # Factor 4: Signal count (0 → 0.0, 10+ → 1.0)
        signal_score = min(len(signals) / 10.0, 1.0)

        # Factor 5: Source quality
        source = str(rca_result.get("source") or "").strip().lower()
        source_score = _SOURCE_QUALITY.get(source, 0.55)

        # Factor 6: Fix applicability ratio
        fixes_payload = rca_result.get("fixes") or {}
        all_fixes     = fixes_payload.get("fixes") or []
        if all_fixes:
            applicable = sum(1 for f in all_fixes if f.get("is_applicable", True))
            fix_score  = applicable / len(all_fixes)
        else:
            fix_score = 0.5  # neutral — no fixes generated yet

        factors = {
            "kb_similarity":       kb_sim,
            "evidence_count":      evidence_score,
            "error_block_clarity": clarity_score,
            "signal_count":        signal_score,
            "source_quality":      source_score,
            "fix_applicability":   fix_score,
        }

        score = (
            0.30 * kb_sim
            + 0.15 * evidence_score
            + 0.15 * clarity_score
            + 0.15 * signal_score
            + 0.15 * source_score
            + 0.10 * fix_score
        )
        score = round(min(max(score, 0.0), 1.0), 4)
        label = _score_to_label(score)

        top = _top_factors(factors)
        explanation = (
            f"Confidence {label} ({score:.0%}). "
            f"Top contributors: {top[0].replace('_', ' ')}"
            + (f" and {top[1].replace('_', ' ')}" if len(top) > 1 else "")
            + f". Source: {source or 'unknown'}."
        )

        return ConfidenceReport(
            score=score,
            label=label,
            explanation=explanation,
            factors=factors,
        )

    # ------------------------------------------------------------------
    # Fix recommendation confidence
    # ------------------------------------------------------------------

    def validate_fix(
        self,
        fix: dict[str, Any],
    ) -> ConfidenceReport:
        """
        Validate confidence for a single fix recommendation.

        Uses the fix's own 5-dimension ranking scores plus a source-quality bonus.

        Parameters
        ----------
        fix : dict
            A single fix candidate dict (from FixRecommendationAgent output).
        """
        rca_sim   = float(fix.get("rca_similarity",       0.0))
        err_sim   = float(fix.get("error_similarity",     0.0))
        success   = float(fix.get("success_rate",         0.5))
        spec      = float(fix.get("specificity",          0.5))
        ease      = float(fix.get("implementation_ease",  0.6))

        # Source quality bonus (same map as RCA)
        source       = str(fix.get("source") or "llm").strip().lower()
        src_bonus    = _SOURCE_QUALITY.get(source, 0.65)

        # Blockers penalty
        has_blockers = bool(fix.get("has_blockers", False))
        blocker_pen  = 0.20 if has_blockers else 0.0

        factors = {
            "rca_similarity":      rca_sim,
            "error_similarity":    err_sim,
            "success_rate":        success,
            "specificity":         spec,
            "implementation_ease": ease,
            "source_quality":      src_bonus,
        }

        # Weighted score (matches fix_utils.py WEIGHTS with source_quality overlay)
        raw = (
            0.30 * rca_sim
            + 0.25 * err_sim
            + 0.20 * success
            + 0.15 * spec
            + 0.05 * ease
            + 0.05 * src_bonus
        )
        score = round(min(max(raw - blocker_pen, 0.0), 1.0), 4)
        label = _score_to_label(score)

        top = _top_factors(factors)
        explanation = (
            f"Fix confidence {label} ({score:.0%}). "
            f"Strongest signals: {top[0].replace('_', ' ')}"
            + (f", {top[1].replace('_', ' ')}" if len(top) > 1 else "")
            + ("  [has blockers]" if has_blockers else "")
            + f". Source: {source}."
        )

        return ConfidenceReport(
            score=score,
            label=label,
            explanation=explanation,
            factors=factors,
        )

    # ------------------------------------------------------------------
    # Web search confidence
    # ------------------------------------------------------------------

    def validate_web(
        self,
        search_results: list[dict[str, Any]],
    ) -> ConfidenceReport:
        """
        Validate confidence for a web search result set.

        Based on the mean relevance score of the top-3 results and the
        diversity of result types (issue, pr, documentation).

        Parameters
        ----------
        search_results : list[dict]
            List of source dicts from ``light_search()`` → ``quick_references``.
        """
        if not search_results:
            return ConfidenceReport(
                score=0.0,
                label="very_low",
                explanation="No web search results available.",
                factors={"result_count": 0.0, "mean_relevance": 0.0, "type_diversity": 0.0},
            )

        relevances = [
            float(r.get("relevance_score", 0.0)) for r in search_results
        ]
        top3  = sorted(relevances, reverse=True)[:3]
        mean_rel = sum(top3) / len(top3)

        # Type diversity: how many distinct result types are present?
        types     = {str(r.get("type", "")) for r in search_results}
        diversity = min(len(types) / 3.0, 1.0)  # 3 types → max diversity

        result_count_score = min(len(search_results) / 5.0, 1.0)

        factors = {
            "mean_relevance":   mean_rel,
            "type_diversity":   diversity,
            "result_count":     result_count_score,
        }

        score = round(
            0.60 * mean_rel + 0.25 * diversity + 0.15 * result_count_score,
            4,
        )
        label = _score_to_label(score)

        type_names = ", ".join(sorted(types)) if types else "none"
        explanation = (
            f"Web search confidence {label} ({score:.0%}). "
            f"{len(search_results)} result(s) across types: {type_names}. "
            f"Mean relevance of top-3: {mean_rel:.0%}."
        )

        return ConfidenceReport(
            score=score,
            label=label,
            explanation=explanation,
            factors=factors,
        )
