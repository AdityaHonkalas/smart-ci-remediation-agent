#!/usr/bin/env python3
"""
utility/web_search_utils.py
────────────────────────────
Shared HTTP/retry utilities for the Web Search Agent.

• Exponential-backoff HTTP GET with configurable retries
• GitHub API helper (issues, PRs, code search)
• URL-safe query builder
• Result deduplication by URL
• Simple relevance scorer based on token overlap
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT  = 20   # seconds per request
_DEFAULT_RETRIES  = 3
_DEFAULT_BACKOFF  = 1.5  # exponential base


# ---------------------------------------------------------------------------
# HTTP helpers with retry / backoff
# ---------------------------------------------------------------------------

def http_get_json(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
    retries: int = _DEFAULT_RETRIES,
    backoff: float = _DEFAULT_BACKOFF,
) -> Any:
    """
    Fetch *url* and return the parsed JSON payload.

    Retries on transient HTTP errors (429, 500-599) with exponential backoff.
    Raises ``RuntimeError`` on final failure.
    """
    _headers = {"Accept": "application/json", "User-Agent": "smart-ci-remediation-agent/1.0"}
    if headers:
        _headers.update(headers)

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=_headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 404:
                return None
            if exc.code in {403, 429} or exc.code >= 500:
                wait = backoff ** attempt
                logger.warning("HTTP %d on %s — retrying in %.1fs (attempt %d/%d)",
                               exc.code, url, wait, attempt, retries)
                time.sleep(wait)
            else:
                raise RuntimeError(f"HTTP {exc.code} fetching {url}") from exc
        except OSError as exc:
            last_exc = exc
            wait = backoff ** attempt
            logger.warning("Network error on %s — retrying in %.1fs (%s)", url, wait, exc)
            time.sleep(wait)

    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def github_api_headers(token: str | None) -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def search_github_issues(
    query: str,
    repository: str | None = None,
    state: str = "all",
    max_results: int = 5,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Search GitHub issues (and PRs) via the Search API."""
    q = query
    if repository:
        q = f"repo:{repository} {q}"
    q += " in:title,body"
    params = urllib.parse.urlencode({"q": q, "state": state, "per_page": min(max_results, 10)})
    url = f"https://api.github.com/search/issues?{params}"
    try:
        data = http_get_json(url, headers=github_api_headers(token))
    except RuntimeError as exc:
        logger.warning("GitHub issue search failed: %s", exc)
        return []
    if not data or "items" not in data:
        return []
    return [
        {
            "type":             "github_issue" if not item.get("pull_request") else "github_pr",
            "title":            item.get("title", ""),
            "url":              item.get("html_url", ""),
            "snippet":          (item.get("body") or "")[:500],
            "relevance_score":  0.0,   # filled by ranker
            "date":             item.get("updated_at", ""),
            "metadata": {
                "repository":   item.get("repository_url", "").replace("https://api.github.com/repos/", ""),
                "status":       item.get("state", ""),
                "number":       item.get("number"),
            },
        }
        for item in data["items"][:max_results]
    ]


def search_github_code(
    query: str,
    repository: str | None = None,
    max_results: int = 5,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Search GitHub code for relevant fix snippets."""
    q = query
    if repository:
        q = f"repo:{repository} {q}"
    params = urllib.parse.urlencode({"q": q, "per_page": min(max_results, 10)})
    url = f"https://api.github.com/search/code?{params}"
    try:
        data = http_get_json(url, headers=github_api_headers(token))
    except RuntimeError as exc:
        logger.warning("GitHub code search failed: %s", exc)
        return []
    if not data or "items" not in data:
        return []
    return [
        {
            "type":            "github_code",
            "title":           item.get("path", ""),
            "url":             item.get("html_url", ""),
            "snippet":         "",
            "relevance_score": 0.0,
            "date":            "",
            "metadata": {
                "repository":  item.get("repository", {}).get("full_name", ""),
                "status":      None,
            },
        }
        for item in data["items"][:max_results]
    ]


# ---------------------------------------------------------------------------
# Relevance scorer
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text.lower()))


def score_relevance(
    result: dict[str, Any],
    error_signature: str,
    error_message: str,
) -> float:
    """
    Simple token-overlap relevance score in [0, 1].

    Weights: title 50%, snippet 30%, url-path 20%.
    """
    sig_tokens = _tokenise(error_signature + " " + error_message)
    if not sig_tokens:
        return 0.0

    def _overlap(text: str, weight: float) -> float:
        tokens = _tokenise(text)
        if not tokens:
            return 0.0
        return weight * len(sig_tokens & tokens) / len(sig_tokens | tokens)

    score = (
        _overlap(result.get("title", ""), 0.5)
        + _overlap(result.get("snippet", ""), 0.3)
        + _overlap(result.get("url", ""), 0.2)
    )
    return round(min(score, 1.0), 4)


def deduplicate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for r in results:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(r)
    return unique
