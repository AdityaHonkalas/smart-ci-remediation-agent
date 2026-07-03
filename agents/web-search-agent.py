#!/usr/bin/env python3
"""
agents/web-search-agent.py
───────────────────────────
Web Search Agent — Agent 2 of the Smart CI Remediation Agent system.

Architecture (REVISED_ARCHITECTURE_PLAN.md §3.2 / §4.2)
─────────────────────────────────────────────────────────
Purpose : Provide fallback RCA evidence for errors not found in the KB.
Triggered by: RCA Agent when KB confidence < KB_CONFIDENCE_THRESHOLD (0.70).

Search Sources (in priority order):
  1. GitHub Issues  — repository-specific issues matching the error signature
  2. GitHub PRs     — merged fixes and discussions
  3. GitHub Code    — fix snippets from related repos
  4. Official Docs  — cached documentation scrape (GitHub Actions, language docs)
  5. Stack Overflow — optional, rate-limited

Search Strategy:
  1. Extract error signature tokens from the failure context
  2. Build targeted queries for each source
  3. Apply exponential-backoff HTTP retries
  4. Rank results by weighted relevance (title 50%, snippet 30%, url 20%)
  5. Deduplicate by URL and return top-K (default 5) with confidence score

Input contract
──────────────
{
    "error_signature" : str   — human-readable error signature
    "error_message"   : str   — primary error message text
    "error_type"      : str   — classified error type
    "repository"      : str   — GitHub repository (owner/name)
    "context"         : str   — surrounding log context (optional)
    "language"        : str   — programming language (optional)
    "framework"       : str   — framework name (optional)
}

Output contract
───────────────
{
    "sources"       : list[SourceResult]   — ranked search results
    "confidence"    : float                — overall confidence in [0, 1]
    "total_results" : int                  — total raw results before ranking
    "search_time"   : float                — wall-clock seconds
    "error"         : str | None           — non-fatal error message if any source failed
}
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add project root and utility/ to path
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "utility"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env_loader import load_dotenv  # noqa: E402
from utility.web_search_utils import (  # noqa: E402
    search_github_issues,
    search_github_code,
    score_relevance,
    deduplicate_results,
)

load_dotenv(_ROOT / ".env")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MAX_RESULTS_PER_SOURCE = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
DEFAULT_TOP_K                  = int(os.getenv("WEB_SEARCH_TOP_K", "5"))
MIN_RELEVANCE_SCORE            = float(os.getenv("WEB_SEARCH_MIN_SCORE", "0.05"))

# Official documentation base URLs keyed on language / framework
_DOC_URLS: dict[str, str] = {
    "python":      "https://docs.python.org/3/",
    "node":        "https://nodejs.org/en/docs/",
    "go":          "https://pkg.go.dev/",
    "java":        "https://docs.oracle.com/en/java/",
    "docker":      "https://docs.docker.com/",
    "kubernetes":  "https://kubernetes.io/docs/",
    "github_actions": "https://docs.github.com/en/actions/",
}


# ---------------------------------------------------------------------------
# Result dataclass (plain dict for JSON-serialisability)
# ---------------------------------------------------------------------------

def _make_source(
    source_type: str,
    title: str,
    url: str,
    snippet: str,
    relevance_score: float = 0.0,
    date: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type":            source_type,
        "title":           title,
        "url":             url,
        "snippet":         snippet[:500],
        "relevance_score": round(relevance_score, 4),
        "date":            date,
        "metadata":        metadata or {},
    }


# ---------------------------------------------------------------------------
# WebSearchAgent
# ---------------------------------------------------------------------------

class WebSearchAgent:
    """
    Multi-source web search agent for CI/CD failure RCA evidence.

    Searches GitHub Issues, PRs, code, and (optionally) official docs to
    provide context for errors not resolved by the self-learning KB.

    Parameters
    ----------
    github_token : str, optional
        GitHub personal access token.  Set in .env as GITHUB_TOKEN.
    max_results_per_source : int
        Maximum raw results fetched from each individual source.
    top_k : int
        Number of results to return after deduplication and ranking.
    """

    def __init__(
        self,
        github_token: str | None = None,
        max_results_per_source: int = DEFAULT_MAX_RESULTS_PER_SOURCE,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.github_token          = github_token or os.getenv("GITHUB_TOKEN")
        self.max_results           = max_results_per_source
        self.top_k                 = top_k
        self._errors: list[str]    = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def search(
        self,
        error_signature: str,
        error_message:   str,
        error_type:      str = "",
        repository:      str = "",
        context:         str = "",
        language:        str = "",
        framework:       str = "",
    ) -> dict[str, Any]:
        """
        Perform multi-source web search for RCA evidence.

        Parameters
        ----------
        error_signature : str
            Human-readable error signature (e.g.
            ``dependency_error_module_not_found_sklearn``).
        error_message : str
            Raw primary error message from the CI log.
        error_type : str
            Classified error type (e.g. ``dependency_error``).
        repository : str
            GitHub repository in ``owner/name`` format.
        context : str
            Surrounding log context for enriching the query.
        language : str
            Primary programming language.
        framework : str
            Framework name if applicable.

        Returns
        -------
        dict
            Structured search result with ``sources``, ``confidence``,
            ``total_results``, ``search_time``, and ``error`` fields.
        """
        t_start = time.monotonic()
        self._errors = []
        all_results: list[dict[str, Any]] = []

        # Build search query from signature + message tokens
        query = self._build_query(error_signature, error_message, context)

        # ---- Source 1: Repository-specific GitHub Issues ----
        if repository:
            repo_issues = self._safe(
                "github_issues_repo",
                lambda: search_github_issues(
                    query=query,
                    repository=repository,
                    state="all",
                    max_results=self.max_results,
                    token=self.github_token,
                ),
            )
            all_results.extend(repo_issues)

        # ---- Source 2: General GitHub Issues ----
        gen_issues = self._safe(
            "github_issues_general",
            lambda: search_github_issues(
                query=query,
                max_results=self.max_results,
                token=self.github_token,
            ),
        )
        all_results.extend(gen_issues)

        # ---- Source 3: GitHub PRs (closed = merged fixes) ----
        pr_results = self._safe(
            "github_prs",
            lambda: search_github_issues(
                query=f"fix {query}",
                repository=repository or None,
                state="closed",
                max_results=self.max_results,
                token=self.github_token,
            ),
        )
        all_results.extend(pr_results)

        # ---- Source 4: GitHub Code Search ----
        code_results = self._safe(
            "github_code",
            lambda: search_github_code(
                query=query,
                repository=repository or None,
                max_results=self.max_results,
                token=self.github_token,
            ),
        )
        all_results.extend(code_results)

        # ---- Source 5: Documentation links (static, no network call) ----
        doc_results = self._generate_doc_links(
            error_signature=error_signature,
            error_type=error_type,
            language=language,
            framework=framework,
        )
        all_results.extend(doc_results)

        total_results = len(all_results)

        # ---- Deduplicate ----
        unique = deduplicate_results(all_results)

        # ---- Score relevance ----
        for result in unique:
            result["relevance_score"] = score_relevance(
                result, error_signature, error_message
            )

        # ---- Filter low-relevance results ----
        filtered = [r for r in unique if r["relevance_score"] >= MIN_RELEVANCE_SCORE]

        # ---- Sort by relevance then recency ----
        filtered.sort(key=lambda r: (r["relevance_score"], r.get("date", "")), reverse=True)

        ranked = filtered[: self.top_k]
        confidence = self._compute_confidence(ranked)
        elapsed    = round(time.monotonic() - t_start, 3)

        logger.info(
            "WebSearchAgent: query=%r  total=%d  ranked=%d  confidence=%.3f  time=%.2fs",
            query[:80], total_results, len(ranked), confidence, elapsed,
        )

        return {
            "sources":       ranked,
            "confidence":    confidence,
            "total_results": total_results,
            "search_time":   elapsed,
            "error":         "; ".join(self._errors) if self._errors else None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_query(
        error_signature: str,
        error_message:   str,
        context:         str = "",
    ) -> str:
        """
        Assemble a compact, high-signal search query.

        Takes the top meaningful tokens from the error signature and message
        to avoid overly broad or overly narrow queries.
        """
        import re
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", error_signature + " " + error_message)
        # Prefer specific tokens; drop super-common words
        stop = {"error", "failed", "failure", "the", "and", "for", "with", "not", "from"}
        unique_tokens = list(dict.fromkeys(t for t in tokens if t.lower() not in stop))
        # Take up to 6 most unique-looking tokens
        query_tokens = unique_tokens[:6]
        return " ".join(query_tokens) if query_tokens else (error_message[:120] or error_signature[:80])

    def _safe(
        self,
        source_name: str,
        fn: Any,
    ) -> list[dict[str, Any]]:
        """Execute *fn* and catch all exceptions, logging them as non-fatal."""
        try:
            return fn() or []
        except Exception as exc:  # noqa: BLE001
            msg = f"{source_name}: {exc}"
            self._errors.append(msg)
            logger.warning("Web search source failed (%s)", msg)
            return []

    @staticmethod
    def _generate_doc_links(
        error_signature: str,
        error_type: str,
        language: str,
        framework: str,
    ) -> list[dict[str, Any]]:
        """
        Generate static documentation pointers without a network call.

        These are ranked low but give the model a starting URL for official docs.
        """
        results: list[dict[str, Any]] = []

        # GitHub Actions docs for workflow / configuration errors
        if error_type in {"configuration_error", "permission_error", "process_exit"}:
            results.append(_make_source(
                source_type="documentation",
                title="GitHub Actions: Workflow syntax reference",
                url="https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions",
                snippet="Complete workflow syntax, permissions, secrets, expressions, and context reference.",
                relevance_score=0.3,
            ))

        # Permissions docs
        if error_type == "permission_error":
            results.append(_make_source(
                source_type="documentation",
                title="GitHub Actions: Automatic token authentication",
                url="https://docs.github.com/en/actions/security-guides/automatic-token-authentication",
                snippet="GITHUB_TOKEN scopes, permissions key, and troubleshooting access-denied failures.",
                relevance_score=0.4,
            ))

        # Language / framework official docs
        for key, doc_url in _DOC_URLS.items():
            if key in (language.lower(), framework.lower(), error_type.lower()):
                results.append(_make_source(
                    source_type="documentation",
                    title=f"Official {key.title()} documentation",
                    url=doc_url,
                    snippet=f"Official reference documentation for {key}.",
                    relevance_score=0.25,
                ))

        return results

    @staticmethod
    def _compute_confidence(ranked: list[dict[str, Any]]) -> float:
        """
        Compute an overall confidence score for the search result set.

        Based on the mean relevance of the top-3 results.
        """
        if not ranked:
            return 0.0
        top3 = sorted(ranked, key=lambda r: r.get("relevance_score", 0), reverse=True)[:3]
        return round(sum(r.get("relevance_score", 0) for r in top3) / len(top3), 4)


# ---------------------------------------------------------------------------
# CLI entry point for standalone testing
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse, json
    parser = argparse.ArgumentParser(description="Test the Web Search Agent standalone.")
    parser.add_argument("error_signature", help="Error signature to search for.")
    parser.add_argument("--repo",    default="", help="GitHub repository (owner/name).")
    parser.add_argument("--msg",     default="", help="Primary error message.")
    parser.add_argument("--type",    default="", help="Error type.")
    parser.add_argument("--top-k",   type=int, default=5)
    args = parser.parse_args()

    agent = WebSearchAgent(top_k=args.top_k)
    result = agent.search(
        error_signature=args.error_signature,
        error_message=args.msg,
        error_type=args.type,
        repository=args.repo,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
