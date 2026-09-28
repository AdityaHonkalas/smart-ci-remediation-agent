#!/usr/bin/env python3
"""
utility/fix_utils.py
─────────────────────
Shared helpers for the Fix Recommendation Agent (fix-recommendor.py).

• Fix candidate dataclass
• Weighted similarity ranker (35/25/20/15/5)
• Fix applicability validator
• Template-based fix library keyed on error_type
• Historical success-rate tracker (persisted in data/fix_success_rates.json)
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
DEFAULT_SUCCESS_RATE_FILE = (
    Path(__file__).resolve().parents[1] / "data" / "fix_success_rates.json"
)

# ---------------------------------------------------------------------------
# Fix candidate dataclass
# ---------------------------------------------------------------------------

@dataclass
class FixCandidate:
    """A single fix recommendation candidate before ranking."""
    description:      str
    fix_type:         str          # 'code_change'|'config_change'|'dependency_update'|'command'|'template'
    code_snippet:     str = ""
    file_path:        str = ""
    command:          str = ""
    source:           str = "llm" # 'llm'|'historical'|'template'|'web'
    raw_similarity:   float = 0.0  # pre-ranking similarity score
    # Ranking dimensions (filled by ranker)
    rca_similarity:   float = 0.0
    error_similarity: float = 0.0
    success_rate:     float = 0.0
    specificity:      float = 0.0
    implementation_ease: float = 0.0
    # Validation
    is_applicable:          bool = True
    has_blockers:           bool = False
    requires_manual_review: bool = False
    warnings:               list[str] = field(default_factory=list)
    side_effects:           list[str] = field(default_factory=list)
    estimated_effort:       str = "medium"
    # Final
    rank:             int  = 0
    final_score:      float = 0.0


# ---------------------------------------------------------------------------
# Ranking weights (per architecture spec)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "rca_similarity":      0.35,
    "error_similarity":    0.25,
    "success_rate":        0.20,
    "specificity":         0.15,
    "implementation_ease": 0.05,
}


def weighted_score(fix: FixCandidate) -> float:
    return (
        WEIGHTS["rca_similarity"]      * fix.rca_similarity
        + WEIGHTS["error_similarity"]    * fix.error_similarity
        + WEIGHTS["success_rate"]        * fix.success_rate
        + WEIGHTS["specificity"]         * fix.specificity
        + WEIGHTS["implementation_ease"] * fix.implementation_ease
    )


# ---------------------------------------------------------------------------
# Semantic similarity helpers (cosine over token overlap — no heavy deps)
# ---------------------------------------------------------------------------

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text.lower()))


def token_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    intersection = len(ta & tb)
    union        = len(ta | tb)
    return round(intersection / union, 4) if union else 0.0


def compute_rca_similarity(fix: FixCandidate, rca_summary: str) -> float:
    return token_similarity(
        fix.description + " " + fix.code_snippet, rca_summary
    )


def compute_error_similarity(fix: FixCandidate, error_message: str) -> float:
    return token_similarity(
        fix.description + " " + fix.code_snippet, error_message
    )


def compute_specificity(fix: FixCandidate) -> float:
    """Score: template=0.4, web=0.6, historical=0.8, llm=0.7.  +0.2 if file_path set."""
    base = {"template": 0.4, "web": 0.6, "historical": 0.8, "llm": 0.7}.get(fix.source, 0.5)
    if fix.file_path:
        base = min(base + 0.2, 1.0)
    return round(base, 4)


def compute_implementation_ease(fix: FixCandidate) -> float:
    effort_map = {"low": 1.0, "medium": 0.6, "high": 0.3}
    return effort_map.get(fix.estimated_effort, 0.6)


# ---------------------------------------------------------------------------
# Historical success-rate tracker
# ---------------------------------------------------------------------------

def _load_rates(path: Path = DEFAULT_SUCCESS_RATE_FILE) -> dict[str, Any]:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_rates(rates: dict[str, Any], path: Path = DEFAULT_SUCCESS_RATE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(rates, fh, indent=2, sort_keys=True)
    tmp.replace(path)


def get_success_rate(fix_type: str, error_type: str) -> float:
    """Return historical success rate for (fix_type, error_type) pair in [0, 1]."""
    key = f"{fix_type}:{error_type}"
    rates = _load_rates()
    entry = rates.get(key, {})
    total = entry.get("total", 0)
    success = entry.get("success", 0)
    return round(success / total, 4) if total >= 3 else 0.5  # prior = 0.5


def record_fix_outcome(
    fix_type: str,
    error_type: str,
    success: bool,
    path: Path = DEFAULT_SUCCESS_RATE_FILE,
) -> None:
    """Record a fix application outcome to update future success-rate priors."""
    key = f"{fix_type}:{error_type}"
    with _LOCK:
        rates = _load_rates(path)
        entry = rates.setdefault(key, {"success": 0, "total": 0})
        entry["total"] += 1
        if success:
            entry["success"] += 1
        entry["last_updated"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        _save_rates(rates, path)


# ---------------------------------------------------------------------------
# Template-based fix library
# ---------------------------------------------------------------------------

TEMPLATE_FIXES: dict[str, list[dict[str, str]]] = {
    "dependency_error": [
        {"description": "Pin the failing dependency to the last known-good version in requirements.txt / package.json.",
         "fix_type": "dependency_update", "estimated_effort": "low"},
        {"description": "Delete and regenerate the lockfile (pip-compile, npm ci, go mod tidy).",
         "fix_type": "command", "command": "pip-compile requirements.in && pip install -r requirements.txt",
         "estimated_effort": "low"},
    ],
    "test_failure": [
        {"description": "Update the failing assertion to match the new expected output.",
         "fix_type": "code_change", "estimated_effort": "medium"},
        {"description": "Mark the test as xfail or skip until the underlying issue is resolved.",
         "fix_type": "code_change", "estimated_effort": "low"},
    ],
    "permission_error": [
        {"description": "Add `permissions: contents: read` (or the required scope) to the workflow job.",
         "fix_type": "config_change", "code_snippet": "permissions:\n  contents: read\n  packages: write",
         "estimated_effort": "low"},
        {"description": "Rotate or re-add the failing secret/token in repository Settings → Secrets.",
         "fix_type": "config_change", "estimated_effort": "low"},
    ],
    "build_error": [
        {"description": "Fix the syntax/compile error in the referenced source file.",
         "fix_type": "code_change", "estimated_effort": "medium"},
        {"description": "Upgrade or downgrade the compiler/SDK to the version used in the last successful build.",
         "fix_type": "dependency_update", "estimated_effort": "medium"},
    ],
    "timeout": [
        {"description": "Add `timeout-minutes:` to the failing job or step.",
         "fix_type": "config_change", "code_snippet": "- name: Slow step\n  timeout-minutes: 30",
         "estimated_effort": "low"},
        {"description": "Cache external dependency downloads to reduce step duration.",
         "fix_type": "config_change", "estimated_effort": "low"},
    ],
    "network_error": [
        {"description": "Add retry logic around the failing network call with exponential backoff.",
         "fix_type": "code_change", "estimated_effort": "medium"},
        {"description": "Check DNS / proxy settings in the runner environment.",
         "fix_type": "config_change", "estimated_effort": "medium"},
    ],
    "configuration_error": [
        {"description": "Validate and correct the workflow YAML or environment variable reference.",
         "fix_type": "config_change", "estimated_effort": "low"},
    ],
    "container_error": [
        {"description": "Rebuild the Docker image with the corrected Dockerfile.",
         "fix_type": "command", "command": "docker build --no-cache -t myimage:latest .",
         "estimated_effort": "medium"},
        {"description": "Verify registry credentials are current in repository secrets.",
         "fix_type": "config_change", "estimated_effort": "low"},
    ],
    "kubernetes_error": [
        {"description": "Correct the Kubernetes manifest resource limits / image tag.",
         "fix_type": "config_change", "estimated_effort": "medium"},
        {"description": "Check cluster RBAC and service account permissions.",
         "fix_type": "config_change", "estimated_effort": "medium"},
    ],
    "resource_error": [
        {"description": "Move the job to a larger runner or add a disk cleanup step.",
         "fix_type": "config_change", "estimated_effort": "low"},
    ],
    "process_exit": [
        {"description": "Inspect the failing command's exit code and add `|| true` / proper error handling if expected.",
         "fix_type": "config_change", "estimated_effort": "low"},
    ],
}


def get_template_fixes(
    error_type: str,
    error_code: str = "",
) -> list[FixCandidate]:
    """Return template FixCandidate objects for the given error_type."""
    templates = TEMPLATE_FIXES.get(error_type, [])
    candidates: list[FixCandidate] = []
    for tmpl in templates:
        candidates.append(FixCandidate(
            description=tmpl.get("description", ""),
            fix_type=tmpl.get("fix_type", "config_change"),
            code_snippet=tmpl.get("code_snippet", ""),
            command=tmpl.get("command", ""),
            source="template",
            estimated_effort=tmpl.get("estimated_effort", "medium"),
            success_rate=get_success_rate(tmpl.get("fix_type", "config_change"), error_type),
        ))
    return candidates
