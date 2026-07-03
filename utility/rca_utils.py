#!/usr/bin/env python3
"""
utility/rca_utils.py
─────────────────────
Shared helpers consumed by the RCA Agent (diagnosis_agent.py).

• Text truncation, line-range formatting
• Evidence line scoring (uses SPECIFIC_ERROR_HINTS from patterns_utils)
• Error detail inference helpers
• Signal / failure-block compact serialisers
• JSON response parser
"""

from __future__ import annotations

import re
from typing import Any

from utility.patterns_utils import SPECIFIC_ERROR_HINTS, CLASSIFICATION_SIGNALS


# ---------------------------------------------------------------------------
# Sets derived from classification_signals (built lazily on first access)
# ---------------------------------------------------------------------------

def _build_sets() -> tuple[set[str], set[str], set[str]]:
    """Derive GENERIC / LESS_SPECIFIC error type sets from loaded patterns."""
    generic      = {"", "unknown_error", "generic_error"}
    less_specific = generic | {"test_failure", "process_exit"}
    generic_codes = {"", "UNKNOWN", "UNKNOWN_ERROR", "TEST_FAILURE", "GENERIC_ERROR"}
    return generic, less_specific, generic_codes


GENERIC_ERROR_TYPES, LESS_SPECIFIC_ERROR_TYPES, GENERIC_ERROR_CODES = _build_sets()


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def truncate_text(value: Any, limit: int = 900) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def line_range(start_line: Any, end_line: Any) -> str:
    if start_line and end_line and start_line != end_line:
        return f"{start_line}-{end_line}"
    return str(start_line) if start_line else ""


def as_text_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


# ---------------------------------------------------------------------------
# Evidence line scoring
# ---------------------------------------------------------------------------

def parse_numbered_line(line: str) -> tuple[int | None, str]:
    match = re.match(r"^\s*(?P<line>\d+):\s*(?P<message>.*)$", line)
    if not match:
        return None, line.strip()
    return int(match.group("line")), match.group("message").strip()


def best_evidence_line(text: str) -> tuple[int | None, str]:
    best_num: int | None = None
    best_msg = ""
    best_score = -1
    for raw_line in str(text or "").splitlines():
        ln, message = parse_numbered_line(raw_line)
        if not message:
            continue
        lowered = message.lower()
        score = 0
        if "##[error]" in lowered or "::error" in lowered or "error:" in lowered:
            score += 10
        if SPECIFIC_ERROR_HINTS and SPECIFIC_ERROR_HINTS.search(message):
            score += 30
        if re.search(r"\b[45]\d{2}\b", message):
            score += 20
        if score > best_score:
            best_score = score
            best_num = ln
            best_msg = message
    if best_msg:
        return best_num, best_msg
    return None, truncate_text(text, 600)


# ---------------------------------------------------------------------------
# Compact serialisers
# ---------------------------------------------------------------------------

def compact_retrieved_context(
    retrieved_context: list[dict[str, Any]], limit: int = 3
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in retrieved_context[:limit]:
        meta = item.get("metadata") or {}
        compact.append({
            "score":        item.get("score"),
            "failure_type": meta.get("failure_type") or meta.get("error_type"),
            "error_code":   meta.get("error_code"),
            "severity":     meta.get("severity"),
            "location": {
                "job":        meta.get("job_name"),
                "log_file":   meta.get("file_name"),
                "line":       meta.get("line_number"),
                "line_range": line_range(meta.get("start_line"), meta.get("end_line")),
            },
            "text": truncate_text(item.get("text"), 900),
        })
    return compact


def compact_evidence(
    primary: dict[str, Any] | None,
    extra_messages: list[str] | None = None,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    if primary:
        message = truncate_text(primary.get("message"), 700)
        if message:
            evidence.append({"line": primary.get("location", {}).get("line"), "message": message})
            seen.add(message)
    for msg in extra_messages or []:
        msg = truncate_text(msg, 700)
        if msg and msg not in seen:
            evidence.append({"line": None, "message": msg})
            seen.add(msg)
        if len(evidence) >= 3:
            break
    return evidence


# ---------------------------------------------------------------------------
# JSON response parser
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> dict[str, Any] | None:
    import json
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end   = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start: end + 1])
            except json.JSONDecodeError:
                return None
    return None
