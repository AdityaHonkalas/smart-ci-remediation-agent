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

Light Search Mode (always triggered with every RCA)
────────────────────────────────────────────────────
A lightweight, inexpensive companion search that runs unconditionally alongside
every RCA.  It collects quick reference links from three sources only:
  1. Official documentation  — static URL hints, no network call
  2. GitHub Issues           — ≤3 results from the repository-specific index
  3. GitHub PRs              — ≤2 closed PRs that match the error query
Results are attached to the RCA output as ``quick_references`` and are purely
informational — they do NOT influence the RCA reasoning or prompt.

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
import re
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

# Deep search toggle — set to True to re-enable WebSearchAgent.search() in the pipeline.
# The search() method is fully preserved below; only the pipeline call is gated.
DEEP_SEARCH_ENABLED: bool = False

# Official documentation base URLs keyed on language / framework / topic
_DOC_URLS: dict[str, dict[str, str]] = {
    "python": {
        "base":   "https://docs.python.org/3/",
        "errors": "https://docs.python.org/3/library/exceptions.html",
        "pip":    "https://pip.pypa.io/en/stable/",
    },
    "java": {
        "base":    "https://docs.oracle.com/en/java/",
        "maven":   "https://maven.apache.org/guides/",
        "gradle":  "https://docs.gradle.org/current/userguide/",
    },
    "go": {
        "base":    "https://pkg.go.dev/",
        "modules": "https://go.dev/ref/mod",
    },
    "node": {
        "base":  "https://nodejs.org/api/",
        "npm":   "https://docs.npmjs.com/",
    },
    "javascript": {
        "base": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/",
        "npm":  "https://docs.npmjs.com/",
    },
    "sql": {
        "postgres": "https://www.postgresql.org/docs/current/",
        "mysql":    "https://dev.mysql.com/doc/",
        "sqlite":   "https://www.sqlite.org/docs.html",
    },
    "kubernetes": {
        "base":   "https://kubernetes.io/docs/",
        "kubectl": "https://kubernetes.io/docs/reference/kubectl/",
    },
    "shell": {
        "base":    "https://www.gnu.org/software/bash/manual/",
        "linux":   "https://man7.org/linux/man-pages/",
        "coreutils": "https://www.gnu.org/software/coreutils/manual/",
    },
    "network": {
        "curl":  "https://curl.se/docs/",
        "http":  "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status",
        "dns":   "https://www.iana.org/domains/root/db",
    },
    "docker": {
        "base":       "https://docs.docker.com/",
        "compose":    "https://docs.docker.com/compose/",
        "dockerfile": "https://docs.docker.com/engine/reference/builder/",
    },
    "github_actions": {
        "base":       "https://docs.github.com/en/actions/",
        "syntax":     "https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions",
        "contexts":   "https://docs.github.com/en/actions/learn-github-actions/contexts",
        "security":   "https://docs.github.com/en/actions/security-guides/automatic-token-authentication",
    },
}

# Language / error-type detection hints for automatic doc-link selection
_LANGUAGE_HINTS: dict[str, list[str]] = {
    "python":     ["python", "pip", "importerror", "modulenotfounderror", "syntaxerror",
                   "typeerror", "valueerror", "indentation", "traceback", "django", "flask"],
    "java":       ["java", "maven", "gradle", "nullpointerexception", "classnotfound",
                   "compilationerror", "spring", "junit"],
    "go":         ["go", "golang", "gomod", "panic", "goroutine", "gotest"],
    "node":       ["node", "nodejs", "npm", "yarn", "require", "esmodule"],
    "javascript": ["javascript", "js", "typescript", "webpack", "babel", "eslint"],
    "sql":        ["sql", "postgres", "mysql", "sqlite", "migration", "query", "constraint"],
    "kubernetes": ["kubernetes", "kubectl", "k8s", "pod", "deployment", "namespace",
                   "configmap", "secret", "ingress", "crashloopbackoff"],
    "shell":      ["bash", "sh", "shell", "chmod", "permission denied", "command not found",
                   "exit code", "segfault", "linux"],
    "network":    ["connection refused", "timeout", "dns", "curl", "http", "ssl", "tls",
                   "certificate", "network"],
    "docker":     ["docker", "dockerfile", "container", "image", "registry", "compose"],
    "github_actions": ["workflow", "action", "runner", "checkout", "artifact",
                       "github_token", "permission", "yaml"],
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
# Module-level helpers: language detection + 5-dimension web result scorer
# ---------------------------------------------------------------------------

def detect_language_from_error(error_type: str, error_message: str) -> str:
    """
    Infer the primary programming language / platform from error context.

    Returns the matching key from ``_LANGUAGE_HINTS``, or ``""`` if unknown.
    Used by ``light_search`` when the caller does not supply an explicit language.
    """
    combined = (error_type + " " + error_message).lower()
    for lang_key, hints in _LANGUAGE_HINTS.items():
        if any(h in combined for h in hints):
            return lang_key
    return ""


def score_web_result_5d(
    result: dict[str, Any],
    error_signature: str,
    error_message:   str,
    rca_summary:     str = "",
    error_type:      str = "",
) -> float:
    """
    Compute a weighted 5-dimension relevance score for a single web result.

    Dimensions and weights (matching the fix-ranking formula):
      rca_similarity      35%  — token overlap between RCA summary and result title/snippet
      error_similarity    25%  — token overlap between error signature and result title
      historical_success  20%  — success rate for (fix_type:error_type) from KB feedback
      specificity         15%  — 0.90 repo-specific issue/PR, 0.70 lang doc, 0.50 generic doc
      implementation_ease  5%  — 0.90 official docs, 0.70 closed PR, 0.50 open issue

    Returns float in [0, 1].
    """
    title   = str(result.get("title",   "") or "")
    snippet = str(result.get("snippet", "") or "")
    rtype   = str(result.get("type",    "") or "")
    meta    = result.get("metadata", {}) or {}

    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text.lower()))

    def _jaccard(a: str, b: str) -> float:
        ta, tb = _tokens(a), _tokens(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    # 1. RCA similarity — how well does the result relate to the RCA reasoning?
    rca_sim = _jaccard(rca_summary, title + " " + snippet) if rca_summary else 0.0

    # 2. Error similarity — how directly does result match the error signature?
    err_sim = _jaccard(error_signature + " " + error_message, title)

    # 3. Historical success — attempt to look up (fix_type:error_type) success rate
    try:
        from utility.fix_utils import get_success_rate  # lazy import; avoids circular
        # Map result type to a rough fix_type for KB lookup
        _ft_map = {"pr": "code_change", "issue": "config_change",
                   "documentation": "config_change", "code": "code_change"}
        inferred_fix_type = _ft_map.get(rtype, "code_change")
        hist_success = get_success_rate(inferred_fix_type, error_type) if error_type else 0.5
    except Exception:  # noqa: BLE001
        hist_success = 0.5  # neutral prior

    # 4. Specificity — repo-specific issues/PRs are most specific
    is_repo_specific = bool(meta.get("repository"))
    if rtype in {"issue", "pr"} and is_repo_specific:
        specificity = 0.90
    elif rtype in {"issue", "pr"}:
        specificity = 0.70
    elif rtype == "documentation":
        specificity = 0.70
    else:
        specificity = 0.50

    # 5. Implementation ease — official docs are easiest to act on immediately
    if rtype == "documentation":
        impl_ease = 0.90
    elif rtype == "pr" and result.get("metadata", {}).get("state") == "closed":
        impl_ease = 0.70
    else:
        impl_ease = 0.50

    score = (
        0.35 * rca_sim
        + 0.25 * err_sim
        + 0.20 * hist_success
        + 0.15 * specificity
        + 0.05 * impl_ease
    )
    return round(min(max(score, 0.0), 1.0), 4)


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
    # Light search — always-on quick reference mode
    # ------------------------------------------------------------------

    def light_search(
        self,
        error_signature: str,
        error_message:   str,
        error_type:      str = "",
        repository:      str = "",
        language:        str = "",
        framework:       str = "",
        rca_summary:     str = "",
    ) -> dict[str, Any]:
        """
        Lightweight, always-triggered search that gathers quick reference links.

        Searches only three inexpensive sources:
          1. Official documentation  (static, no network call)
          2. GitHub Issues           (≤3 repo-scoped results)
          3. GitHub PRs / closed issues (≤2 results)

        Each result is ranked using the 5-dimension weighted formula:
          RCA similarity (35%) + error similarity (25%) + historical success (20%)
          + specificity (15%) + implementation ease (5%)

        Results are purely informational — attached to the output as
        ``quick_references`` and NOT injected into the RCA prompt.

        Returns
        -------
        dict with keys:
            quick_references : list[dict]   — deduplicated, 5-dimension ranked links
            search_time      : float        — wall-clock seconds
            error            : str | None   — non-fatal error string if any source failed
        """
        t_start = time.monotonic()
        self._errors = []
        refs: list[dict[str, Any]] = []

        query = self._build_query(error_signature, error_message)

        # Auto-detect language from error context if not supplied
        detected_lang = language or detect_language_from_error(error_type, error_message)

        # Source 1 — Official documentation (static, free)
        refs.extend(self._generate_doc_links(
            error_signature=error_signature,
            error_type=error_type,
            language=detected_lang,
            framework=framework,
        ))

        # Source 2 — GitHub Issues (repo-scoped, max 3)
        if repository:
            refs.extend(self._safe(
                "light_github_issues",
                lambda: search_github_issues(
                    query=query,
                    repository=repository,
                    state="all",
                    max_results=3,
                    token=self.github_token,
                ),
            ))

        # Source 3 — GitHub PRs / closed issues (max 2)
        refs.extend(self._safe(
            "light_github_prs",
            lambda: search_github_issues(
                query=f"fix {query}",
                repository=repository or None,
                state="closed",
                max_results=2,
                token=self.github_token,
            ),
        ))

        # Apply 5-dimension relevance scoring, deduplicate, sort, cap at top-5
        unique = deduplicate_results(refs)
        for r in unique:
            r["relevance_score"] = score_web_result_5d(
                result=r,
                error_signature=error_signature,
                error_message=error_message,
                rca_summary=rca_summary,
                error_type=error_type,
            )
            r["confidence"] = round(r["relevance_score"], 4)
        unique.sort(key=lambda r: r["relevance_score"], reverse=True)
        top = unique[:5]

        elapsed = round(time.monotonic() - t_start, 3)

        # Attach aggregate confidence report for the quick_references set
        web_confidence: dict[str, Any] = {}
        try:
            from utility.confidence_validator import ConfidenceValidator  # noqa: PLC0415
            _cv = ConfidenceValidator()
            _cr = _cv.validate_web(top)
            web_confidence = _cr.to_dict()
        except Exception:  # noqa: BLE001
            pass

        logger.info(
            "WebSearchAgent.light_search: query=%r  refs=%d  confidence=%.3f  time=%.2fs",
            query[:80], len(top),
            web_confidence.get("score", 0.0), elapsed,
        )
        return {
            "quick_references":       top,
            "search_time":            elapsed,
            "web_search_confidence":  web_confidence,
            "error":                  "; ".join(self._errors) if self._errors else None,
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

        Covers GitHub Actions, all supported languages (Python, Java, Go, Node.js,
        JavaScript, SQL, Kubernetes, Shell/Linux, Network, Docker) and framework-
        specific sub-pages.  Results are returned with base relevance 0.3–0.5.
        """
        results: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        def _add(title: str, url: str, snippet: str, score: float = 0.35) -> None:
            if url not in seen_urls:
                seen_urls.add(url)
                results.append(_make_source(
                    source_type="documentation",
                    title=title,
                    url=url,
                    snippet=snippet,
                    relevance_score=score,
                ))

        # --- Always add GitHub Actions base doc for CI errors ---
        if error_type in {"configuration_error", "permission_error", "process_exit",
                          "workflow_error", "runner_error"}:
            ga = _DOC_URLS["github_actions"]
            _add("GitHub Actions: Workflow syntax reference", ga["syntax"],
                 "Complete workflow syntax, permissions, secrets, expressions, and context reference.",
                 score=0.45)
        if error_type == "permission_error":
            ga = _DOC_URLS["github_actions"]
            _add("GitHub Actions: Automatic token authentication", ga["security"],
                 "GITHUB_TOKEN scopes, permissions key, and troubleshooting access-denied failures.",
                 score=0.50)

        # --- Language / framework docs (matched by language, framework, or error_type) ---
        candidates = {language.lower(), framework.lower(), error_type.lower()} - {"", "unknown"}

        # Also match via error_type keyword hints
        for lang_key, hints in _LANGUAGE_HINTS.items():
            combined = (error_signature + " " + error_type + " " + language).lower()
            if any(h in combined for h in hints):
                candidates.add(lang_key)

        for lang_key in candidates:
            if lang_key not in _DOC_URLS:
                continue
            urls_for_lang = _DOC_URLS[lang_key]
            # Add primary (base) URL
            primary_url = urls_for_lang.get("base") or next(iter(urls_for_lang.values()))
            _add(
                title=f"Official {lang_key.replace('_', ' ').title()} documentation",
                url=primary_url,
                snippet=f"Official reference documentation for {lang_key.replace('_', ' ')}.",
                score=0.35,
            )
            # Add one sub-page if relevant to the error_type
            if lang_key == "python" and "import" in error_type.lower():
                _add("Python: Built-in Exceptions", urls_for_lang["errors"],
                     "Python's built-in exception hierarchy and import-related errors.", score=0.40)
            elif lang_key in {"node", "javascript"} and "depend" in error_type.lower():
                _add("npm documentation", urls_for_lang["npm"],
                     "npm package installation, versioning, and troubleshooting.", score=0.40)
            elif lang_key == "kubernetes" and error_type.lower() in {
                    "container_error", "permission_error", "configuration_error"}:
                _add("kubectl reference", urls_for_lang["kubectl"],
                     "kubectl command reference for debugging pods and deployments.", score=0.40)
            elif lang_key == "shell" and "permission" in error_type.lower():
                _add("Linux man-pages", urls_for_lang["linux"],
                     "Linux command reference, chmod, file permissions, and shell built-ins.",
                     score=0.40)

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
