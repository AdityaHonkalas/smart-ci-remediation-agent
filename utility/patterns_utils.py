#!/usr/bin/env python3
"""
utility/patterns_utils.py
──────────────────────────
Centralised loader and writer for ./data/failure_signal_patterns.json.

Responsibilities
─────────────────
• Load the pattern file at start-up and expose compiled regex objects
  so every agent works from the same single source of truth.
• Provide ``update_patterns()`` to add new error/block patterns discovered
  at runtime (e.g. from web-search results or RCA feedback).
• Hot-reload: call ``reload()`` to refresh in-process state after an
  external edit without restarting the process.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

# Default path — can be overridden with the PATTERNS_FILE env variable
DEFAULT_PATTERNS_FILE = Path(__file__).resolve().parents[1] / "data" / "failure_signal_patterns.json"


# ---------------------------------------------------------------------------
# Internal state (module-level singletons, refreshed by reload())
# ---------------------------------------------------------------------------

_raw: dict[str, Any] = {}

# Compiled sequences  — list[tuple[name, Pattern]]
FAILURE_BLOCK_PATTERNS: list[tuple[str, re.Pattern[str]]] = []
ERROR_PATTERNS:         list[tuple[str, re.Pattern[str]]] = []
FAILURE_BLOCK_END_RE:   re.Pattern[str] | None = None

# Plain token dicts
CLASSIFICATION_SIGNALS: dict[str, tuple[str, ...]] = {}
ERROR_CODE_SIGNALS:     dict[str, tuple[str, ...]] = {}
SEVERITY_SIGNALS:       dict[str, Any] = {}
NOISE_PREFIXES:         tuple[str, ...] = ()

# Compiled hint pattern for fast evidence scoring
SPECIFIC_ERROR_HINTS:   re.Pattern[str] | None = None


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _compile_flags(flag_names: list[str] | None) -> int:
    flags = 0
    for name in flag_names or []:
        try:
            flags |= getattr(re, str(name).upper())
        except AttributeError as exc:
            raise ValueError(f"Unsupported regex flag in patterns file: {name}") from exc
    return flags


def _compile_named(entries: list[dict[str, Any]]) -> list[tuple[str, re.Pattern[str]]]:
    result: list[tuple[str, re.Pattern[str]]] = []
    for entry in entries:
        pat = re.compile(str(entry["pattern"]), _compile_flags(entry.get("flags")))
        result.append((str(entry["name"]), pat))
    return result


def _build_specific_hints(signals: dict[str, Any]) -> re.Pattern[str]:
    """Derive SPECIFIC_ERROR_HINTS from classification_signals in the JSON."""
    high_priority = (
        "permission_error", "network_error", "timeout",
        "resource_error", "dependency_error", "build_error",
    )
    tokens: list[str] = []
    for category in high_priority:
        for token in signals.get(category, []):
            escaped = re.escape(str(token))
            tokens.append(escaped)
    # Always keep HTTP error code hint
    tokens.extend([r"40[13]", r"403", r"401", r"forbidden", r"unauthorized"])
    pattern = r"\b(?:" + "|".join(dict.fromkeys(tokens)) + r")\b"
    return re.compile(pattern, re.I)


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _apply(raw: dict[str, Any]) -> None:
    """Push loaded raw data into the module-level compiled objects."""
    global _raw
    global FAILURE_BLOCK_PATTERNS, ERROR_PATTERNS, FAILURE_BLOCK_END_RE
    global CLASSIFICATION_SIGNALS, ERROR_CODE_SIGNALS, SEVERITY_SIGNALS
    global NOISE_PREFIXES, SPECIFIC_ERROR_HINTS

    _raw = raw
    FAILURE_BLOCK_PATTERNS = _compile_named(raw.get("failure_block_patterns", []))
    ERROR_PATTERNS         = _compile_named(raw.get("error_patterns", []))

    end_entry = raw.get("failure_block_end_pattern")
    if end_entry:
        FAILURE_BLOCK_END_RE = re.compile(
            str(end_entry["pattern"]), _compile_flags(end_entry.get("flags"))
        )

    CLASSIFICATION_SIGNALS = {
        str(k): tuple(str(t).lower() for t in v)
        for k, v in raw.get("classification_signals", {}).items()
    }
    ERROR_CODE_SIGNALS = {
        str(k): tuple(str(t).lower() for t in v)
        for k, v in raw.get("error_code_signals", {}).items()
    }
    SEVERITY_SIGNALS = raw.get("severity_signals", {})
    NOISE_PREFIXES   = tuple(str(p).lower() for p in raw.get("noise_prefixes", []))
    SPECIFIC_ERROR_HINTS = _build_specific_hints(raw.get("classification_signals", {}))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(path: Path | str | None = None) -> None:
    """Load (or reload) patterns from *path* into the module globals."""
    import os
    resolved = Path(path or os.getenv("PATTERNS_FILE") or DEFAULT_PATTERNS_FILE)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing failure pattern config: {resolved}")
    with _LOCK:
        raw = _load(resolved)
        _apply(raw)
    logger.debug("Patterns loaded from %s (%d error patterns, %d block patterns)",
                 resolved, len(ERROR_PATTERNS), len(FAILURE_BLOCK_PATTERNS))


def reload() -> None:
    """Hot-reload patterns from the default file path."""
    load()


def get_raw() -> dict[str, Any]:
    """Return a deep copy of the current raw JSON payload."""
    import copy
    return copy.deepcopy(_raw)


def update_patterns(
    new_error_patterns:        list[dict[str, Any]] | None = None,
    new_block_patterns:        list[dict[str, Any]] | None = None,
    new_classification_signals:dict[str, list[str]] | None = None,
    path: Path | str | None = None,
) -> None:
    """
    Merge new patterns into the JSON file and reload in-process state.

    This is the hook called by the RCA Agent and Web Search Agent when a
    novel error signature is discovered that should be persisted for future
    runs.

    Parameters
    ----------
    new_error_patterns : list of pattern dicts
        Each dict must have at least ``name`` and ``pattern`` keys.
    new_block_patterns : list of pattern dicts
        Same schema as error_patterns entries.
    new_classification_signals : dict[error_type, list[token]]
        Additional classification tokens to merge into existing categories.
    path : Path or str, optional
        Override the default patterns file path.
    """
    import os
    resolved = Path(path or os.getenv("PATTERNS_FILE") or DEFAULT_PATTERNS_FILE)

    with _LOCK:
        raw = _load(resolved) if resolved.exists() else {}
        raw.setdefault("schema_version", 1)
        raw.setdefault("error_patterns", [])
        raw.setdefault("failure_block_patterns", [])
        raw.setdefault("classification_signals", {})

        existing_error_names = {e["name"] for e in raw["error_patterns"]}
        existing_block_names = {e["name"] for e in raw["failure_block_patterns"]}

        changed = False

        for ep in new_error_patterns or []:
            if ep.get("name") not in existing_error_names:
                ep.setdefault("added_at", datetime.now(timezone.utc).replace(microsecond=0).isoformat())
                raw["error_patterns"].append(ep)
                existing_error_names.add(ep["name"])
                changed = True

        for bp in new_block_patterns or []:
            if bp.get("name") not in existing_block_names:
                bp.setdefault("added_at", datetime.now(timezone.utc).replace(microsecond=0).isoformat())
                raw["failure_block_patterns"].append(bp)
                existing_block_names.add(bp["name"])
                changed = True

        for error_type, tokens in (new_classification_signals or {}).items():
            existing = raw["classification_signals"].setdefault(error_type, [])
            for token in tokens:
                if token not in existing:
                    existing.append(token)
                    changed = True

        if changed:
            raw["last_updated"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            tmp = resolved.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(raw, fh, indent=2, sort_keys=True)
                fh.write("\n")
            tmp.replace(resolved)
            _apply(raw)
            logger.info("Patterns file updated: %d error patterns, %d block patterns",
                        len(raw["error_patterns"]), len(raw["failure_block_patterns"]))
        else:
            logger.debug("No new patterns to add.")


# ---------------------------------------------------------------------------
# Auto-load on import
# ---------------------------------------------------------------------------
try:
    load()
except FileNotFoundError as _e:
    logger.warning("Patterns file not found at startup (%s). Call load(path) explicitly.", _e)
