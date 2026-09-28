#!/usr/bin/env python3
"""
agents/fix-recommendor.py
──────────────────────────
Fix Recommendation Agent — Agent 3 of the Smart CI Remediation Agent system.

Architecture (REVISED_ARCHITECTURE_PLAN.md §3.3 / §4.3)
──────────────────────────────────────────────────────────
Purpose : Generate, rank, validate, and return the top-K fix recommendations
          for a detected CI failure root cause.

3-step process:

  Step 1 — Fix Generation
    Candidates are generated from four sources:
    a) LLM-based generation from the RCA analysis (same Ollama model as RCA Agent)
    b) Historical fix retrieval from the self-learning KB
    c) Template-based fixes for known error patterns
    d) Web search fix extraction from GitHub Issues / PRs

  Step 2 — Similarity-Based Ranking
    Each candidate is scored by a 5-dimension weighted function:
      • RCA Similarity       35% — how well the fix addresses the root cause
      • Error Similarity     25% — semantic match to the error context
      • Historical Success   20% — past effectiveness for similar failures
      • Specificity          15% — targeted vs. generic fix
      • Implementation Ease   5% — effort / complexity of the change

  Step 3 — Validation & Output
    Each top-K candidate is validated for:
      • Applicability to the current failure context
      • Potential side effects and dependency conflicts
      • Security implications
      • Dependency compatibility
    Returns top-K ranked fixes with implementation instructions.

LLM Model
─────────
Uses the same Ollama LLMClientInterface as the RCA Agent (default: llama3.1:8b),
ensuring consistent model behaviour across the diagnosis and remediation pipeline.
Override via MODEL_ID or API_URL environment variables.

Input contract
──────────────
{
    "rca"         : dict    — RCA result from RCA Agent
    "error_block" : dict    — primary enriched ErrorBlock
    "web_context" : list    — optional web search results
    "metadata"    : dict    — repository / environment metadata
    "top_k"       : int     — number of fixes to return (default 5)
}

Output contract
───────────────
{
    "fixes"              : list[RankedFix]
    "total_candidates"   : int
    "generation_time"    : float
    "generation_errors"  : dict   — per-source error details
    "validation_summary" : dict   — counts of applicable/blocked/manual_review
}

Each RankedFix:
{
    "rank"                 : int
    "description"          : str
    "fix_type"             : str
    "code_snippet"         : str
    "file_path"            : str
    "command"              : str
    "source"               : str   — 'llm'|'historical'|'template'|'web'
    "rca_similarity"       : float
    "error_similarity"     : float
    "success_rate"         : float
    "specificity"          : float
    "implementation_ease"  : float
    "final_score"          : float
    "is_applicable"        : bool
    "has_blockers"         : bool
    "requires_manual_review": bool
    "warnings"             : list[str]
    "side_effects"         : list[str]
    "estimated_effort"     : str
}
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, replace as dc_replace
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "utility"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env_loader import load_dotenv          # noqa: E402
from db import SelfLearningKnowledgeBase   # noqa: E402
from utility.fix_utils import (            # noqa: E402
    FixCandidate,
    WEIGHTS,
    weighted_score,
    compute_rca_similarity,
    compute_error_similarity,
    compute_specificity,
    compute_implementation_ease,
    get_success_rate,
    get_template_fixes,
    record_fix_outcome,
)

load_dotenv(_ROOT / ".env")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model settings — mirrors RCA Agent (LLMClientInterface / Ollama)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID  = os.getenv("MODEL_ID", "llama3.1:8b")
DEFAULT_API_URL   = os.getenv("API_URL",  "http://localhost:11434/api/chat")
DEFAULT_TOP_K     = int(os.getenv("FIX_TOP_K", "5"))
DATA_DIR          = _ROOT / "data"

# Request / generation limits
_LLM_TIMEOUT_SECONDS  = int(os.getenv("FIX_LLM_TIMEOUT", "300"))
_LLM_NUM_PREDICT      = int(os.getenv("FIX_NUM_PREDICT", "1024"))

_FIX_SYSTEM_PROMPT = (
    "You are a CI/CD fix recommendation agent. "
    "Given an RCA report and error context, generate concrete, actionable fix "
    "recommendations. Return ONLY a JSON array where each element has: "
    "description (str), fix_type (str: code_change|config_change|dependency_update|command), "
    "code_snippet (str, optional), file_path (str, optional), command (str, optional), "
    "estimated_effort (str: low|medium|high). "
    "Be specific — reference actual files, packages, config keys, and commands. "
    "Do NOT wrap the array in any outer object."
)

# Dependency keywords that hint at compatibility concerns
_DEPENDENCY_RE = re.compile(
    r"\b(?:pip install|npm install|yarn add|go get|apt-get install|brew install|"
    r"requirements\.txt|package\.json|go\.mod|Gemfile|Pipfile|pom\.xml|"
    r"build\.gradle|setup\.py|pyproject\.toml)\b",
    re.I,
)

# Patterns that always warrant manual review regardless of other checks
_MANUAL_REVIEW_RE = re.compile(
    r"\b(?:chmod 777|curl.*\|.*sh|wget.*\|.*sh|eval\s*\(|exec\s*\(|"
    r"os\.system\s*\(|subprocess\.run.*shell\s*=\s*True|"
    r"DROP TABLE|DELETE FROM|TRUNCATE|rm\s+-rf\s+/|"
    r":\s*\(\)\s*\{[^}]*\}\s*;|fork\s+bomb)\b",
    re.I,
)


# ---------------------------------------------------------------------------
# LLM Fix Generator  (uses the same Ollama transport as the RCA Agent)
# ---------------------------------------------------------------------------

class LLMFixGenerator:
    """
    Generate fix candidates by calling the same Ollama endpoint used by the
    RCA Agent (LLMClientInterface).

    The model and API URL are resolved from the same environment variables
    (MODEL_ID, API_URL) so that both agents stay in sync.
    """

    def __init__(
        self,
        model_id:    str = DEFAULT_MODEL_ID,
        api_url:     str = DEFAULT_API_URL,
        num_predict: int = _LLM_NUM_PREDICT,
        temperature: float = 0.2,
        top_p:       float = 0.9,
        top_k:       int   = 40,
        repeat_penalty: float = 1.1,
        timeout:     int = _LLM_TIMEOUT_SECONDS,
    ) -> None:
        self.model_id      = model_id
        self.api_url       = api_url
        self.num_predict   = num_predict
        self.temperature   = temperature
        self.top_p         = top_p
        self.top_k_model   = top_k
        self.repeat_penalty = repeat_penalty
        self.timeout       = timeout

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def generate(
        self,
        rca_summary:   str,
        error_type:    str,
        error_message: str,
        error_code:    str = "",
        stack_trace:   list[str] | None = None,
        repository:    str = "",
        language:      str = "",
    ) -> tuple[list[FixCandidate], str | None]:
        """
        Ask the LLM to generate fix candidates from the provided RCA.

        Returns
        -------
        (candidates, error_message)
            ``candidates`` is the (possibly empty) list of parsed fixes.
            ``error_message`` is None on success, or a descriptive string on
            any failure (API error, timeout, parse failure).
        """
        prompt = self._build_prompt(
            rca_summary, error_type, error_message,
            error_code, stack_trace, repository, language,
        )
        try:
            raw = self._call_ollama(prompt)
        except TimeoutError as exc:
            msg = f"LLM request timed out after {self.timeout}s: {exc}"
            logger.warning("LLMFixGenerator: %s", msg)
            return [], msg
        except ConnectionError as exc:
            msg = f"LLM API connection error (is Ollama running at {self.api_url}?): {exc}"
            logger.warning("LLMFixGenerator: %s", msg)
            return [], msg
        except RuntimeError as exc:
            msg = f"LLM API request failed: {exc}"
            logger.warning("LLMFixGenerator: %s", msg)
            return [], msg
        except Exception as exc:           # noqa: BLE001 — catch-all with full context
            msg = f"LLM fix generation unexpected error: {type(exc).__name__}: {exc}"
            logger.warning("LLMFixGenerator: %s", msg)
            return [], msg

        candidates, parse_error = self._parse_response(raw, error_type)
        if parse_error:
            logger.warning("LLMFixGenerator parse error: %s", parse_error)
            return [], parse_error
        return candidates, None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        rca_summary:   str,
        error_type:    str,
        error_message: str,
        error_code:    str,
        stack_trace:   list[str] | None,
        repository:    str,
        language:      str,
    ) -> str:
        payload = {
            "rca_summary":   rca_summary,
            "error_type":    error_type,
            "error_message": error_message,
            "error_code":    error_code,
            "stack_trace":   (stack_trace or [])[:10],
            "repository":    repository,
            "language":      language,
        }
        system_block = f"SYSTEM:\n{_FIX_SYSTEM_PROMPT}\n\n"
        user_block = (
            "USER:\nGenerate fix recommendations for the following CI/CD failure RCA.\n"
            "Return a JSON array of fix objects. Each fix must have: "
            "description, fix_type, code_snippet (if applicable), "
            "file_path (if applicable), command (if applicable), estimated_effort.\n\n"
            f"{json.dumps(payload, indent=2)}"
        )
        return system_block + user_block

    def _call_ollama(self, prompt: str) -> str:
        """
        POST the prompt to the Ollama /api/chat endpoint.

        Raises
        ------
        TimeoutError
            When the HTTP request exceeds ``self.timeout`` seconds.
        ConnectionError
            When the server cannot be reached.
        RuntimeError
            On HTTP 4xx / 5xx responses.
        """
        import requests  # type: ignore[import-not-found]
        from requests.exceptions import Timeout, ConnectionError as ReqConnError

        body = {
            "model": self.model_id,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "num_predict":    self.num_predict,
                "temperature":    self.temperature,
                "top_p":          self.top_p,
                "top_k":          self.top_k_model,
                "repeat_penalty": self.repeat_penalty,
            },
        }

        logger.debug(
            "LLMFixGenerator: POST %s  model=%s  num_predict=%d",
            self.api_url, self.model_id, self.num_predict,
        )

        try:
            response = requests.post(
                url=self.api_url,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
        except Timeout as exc:
            raise TimeoutError(str(exc)) from exc
        except ReqConnError as exc:
            raise ConnectionError(str(exc)) from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP {response.status_code}: {response.text[:200]}"
            )

        payload = response.json()
        msg = payload.get("message", {})
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()
        # Fallback: return raw JSON
        return response.text

    @staticmethod
    def _parse_response(
        text: str,
        error_type: str,
    ) -> tuple[list[FixCandidate], str | None]:
        """
        Extract a JSON array from the model response.

        Returns (candidates, error_message).  error_message is None on success.
        """
        match = re.search(r"\[.*?\]", text, re.S)
        if not match:
            return [], f"No JSON array found in LLM response (length={len(text)})"
        try:
            items = json.loads(match.group())
        except json.JSONDecodeError as exc:
            return [], f"JSON parse error in LLM response: {exc}"

        if not isinstance(items, list):
            return [], "LLM response JSON array is not a list"

        candidates: list[FixCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            desc = str(item.get("description", "")).strip()
            if not desc:
                continue
            candidates.append(FixCandidate(
                description=desc,
                fix_type=str(item.get("fix_type", "config_change")),
                code_snippet=str(item.get("code_snippet") or ""),
                file_path=str(item.get("file_path") or ""),
                command=str(item.get("command") or ""),
                source="llm",
                estimated_effort=str(item.get("estimated_effort", "medium")),
                success_rate=get_success_rate(
                    str(item.get("fix_type", "config_change")), error_type
                ),
            ))
        return candidates, None


# ---------------------------------------------------------------------------
# Historical Fix Retriever (from KB)
# ---------------------------------------------------------------------------

class HistoricalFixRetriever:
    """Retrieve previously successful fix descriptions from the self-learning KB."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.kb = SelfLearningKnowledgeBase(
            persist_dir=data_dir / "rca_knowledge_base"
        )

    def retrieve(
        self,
        error_signature: str,
        rca_summary: str,
        top_k: int = 10,
    ) -> tuple[list[FixCandidate], str | None]:
        """
        Query KB for similar RCA entries and reuse their inline_fix_suggestions.

        Returns
        -------
        (candidates, error_message)
            ``error_message`` is None on success, or a descriptive string on failure.
        """
        try:
            entries = self.kb.search(error_signature, top_k=top_k)
        except Exception as exc:           # noqa: BLE001
            msg = f"KB search failed for signature={error_signature!r}: {type(exc).__name__}: {exc}"
            logger.warning("HistoricalFixRetriever: %s", msg)
            return [], msg

        candidates: list[FixCandidate] = []
        for entry in entries:
            if entry.similarity < 0.5:
                continue
            try:
                rca_stored = self.kb._rca_repository.get(entry.error_signature, {})
            except Exception as exc:       # noqa: BLE001
                logger.debug("KB repository lookup failed for %s: %s", entry.error_signature, exc)
                continue
            for fix in rca_stored.get("inline_fix_suggestions", []) or []:
                if not isinstance(fix, dict):
                    continue
                desc = str(fix.get("suggested_change") or fix.get("description") or "").strip()
                if not desc:
                    continue
                candidates.append(FixCandidate(
                    description=desc,
                    fix_type="config_change",
                    code_snippet="",
                    file_path=str(fix.get("target") or ""),
                    source="historical",
                    estimated_effort="medium",
                    raw_similarity=entry.similarity,
                    success_rate=entry.confidence if entry.confidence > 0 else 0.6,
                ))
        return candidates, None


# ---------------------------------------------------------------------------
# Web Fix Extractor
# ---------------------------------------------------------------------------

def extract_fixes_from_web(
    web_context: list[dict[str, Any]],
    error_type:  str,
) -> tuple[list[FixCandidate], str | None]:
    """
    Extract actionable fix hints from Web Search Agent results.

    Parameters
    ----------
    web_context : list[dict]
        Ranked source results from the WebSearchAgent.
    error_type : str
        Classified error type for success-rate lookup.

    Returns
    -------
    (candidates, error_message)
        ``error_message`` is None on success, or a description of any
        non-fatal issue encountered while processing the web results.
    """
    if not web_context:
        return [], None

    fix_keywords = re.compile(
        r"\b(?:fix|solution|workaround|resolved by|change|update|pin|add|remove|set|replace)\b",
        re.I,
    )
    candidates: list[FixCandidate] = []
    malformed:  list[int] = []

    for idx, src in enumerate(web_context):
        if not isinstance(src, dict):
            malformed.append(idx)
            logger.debug("Web context item[%d] is not a dict — skipped", idx)
            continue
        try:
            snippet = str(src.get("snippet") or "")
            title   = str(src.get("title")   or "")
            url     = str(src.get("url")     or "")
            if not fix_keywords.search(snippet) and not fix_keywords.search(title):
                continue
            description = f"{title}: {snippet[:250]}"
            if url:
                description += f" [source: {url}]"
            candidates.append(FixCandidate(
                description=description,
                fix_type="config_change",
                source="web",
                estimated_effort="medium",
                raw_similarity=float(src.get("relevance_score", 0.1)),
                success_rate=get_success_rate("web", error_type),
            ))
        except Exception as exc:           # noqa: BLE001
            logger.warning("Web context item[%d] processing error: %s", idx, exc)
            malformed.append(idx)

    error_msg: str | None = None
    if malformed:
        error_msg = f"web_extraction: {len(malformed)} malformed source item(s) at indices {malformed}"

    return candidates, error_msg


# ---------------------------------------------------------------------------
# Fix Validator
# ---------------------------------------------------------------------------

class FixValidator:
    """
    Validate whether a fix is applicable and flag side-effects, blockers, and
    fixes that require manual review before deployment.

    Validation checks
    ─────────────────
    1. Applicability  — fix type is coherent with the error type
    2. Side effects   — high-effort or broad-scope changes are flagged
    3. Dependency compatibility — fixes that install/update packages are annotated
    4. Security implications — unsafe shell/eval/chmod patterns set has_blockers
    5. Manual review  — patterns that MUST be reviewed by a human (e.g. rm -rf /)
    """

    _SECURITY_RISK_RE = re.compile(
        r"\b(?:eval\s*\(|exec\s*\(|shell\s*=\s*True|os\.system\s*\(|"
        r"subprocess\.run.*shell\s*=\s*True|chmod 777|"
        r"curl[^;\n]*\|\s*(?:ba)?sh|wget[^;\n]*\|\s*(?:ba)?sh)\b",
        re.I,
    )

    def validate(
        self,
        fix:        FixCandidate,
        error_type: str,
        language:   str = "",
        metadata:   dict[str, Any] | None = None,
    ) -> FixCandidate:
        """
        Validate *fix* and return an updated candidate with all flags set.

        Parameters
        ----------
        fix : FixCandidate
            Candidate to validate (not mutated — a new instance is returned).
        error_type : str
            Classified error type from the RCA.
        language : str
            Primary programming language (used for dependency checks).
        metadata : dict, optional
            Repository / environment context (used for future extension).
        """
        warnings:      list[str] = list(fix.warnings)
        side_effects:  list[str] = list(fix.side_effects)
        has_blockers          = fix.has_blockers
        is_applicable         = fix.is_applicable
        requires_manual_review = getattr(fix, "requires_manual_review", False)

        text_to_check = (fix.code_snippet + " " + fix.command).strip()

        # ── 1. Applicability ──────────────────────────────────────────────
        if not fix.description.strip() and not fix.code_snippet.strip() and not fix.command.strip():
            is_applicable = False
            warnings.append("Fix has no actionable content — skipped.")

        if error_type == "test_failure" and fix.fix_type == "dependency_update":
            warnings.append(
                "Dependency update may not resolve a test logic failure — verify root cause first."
            )
            is_applicable = False

        if error_type == "build_error" and fix.fix_type == "config_change":
            # Config changes can sometimes fix build errors (e.g. env var) but flag for review
            warnings.append(
                "Config change applied to a build error — confirm the root cause is not a code defect."
            )

        # ── 2. Side effects ───────────────────────────────────────────────
        if fix.estimated_effort == "high":
            side_effects.append(
                "High-effort change — may affect multiple components; plan and test in a branch first."
            )

        if fix.fix_type == "dependency_update":
            side_effects.append(
                "Dependency version change may introduce or resolve other transitive dependencies."
            )

        if fix.fix_type == "code_change" and not fix.file_path:
            side_effects.append(
                "Code change has no target file specified — verify the correct file before applying."
            )

        # ── 3. Dependency compatibility ───────────────────────────────────
        if _DEPENDENCY_RE.search(text_to_check):
            side_effects.append(
                "Fix installs or updates a package — run the full test suite and check lockfile diff."
            )
            # If pinning to a specific version, note potential conflicts
            if re.search(r"==\d|@\d|\^\d|~\d|>=\d", text_to_check):
                warnings.append(
                    "Version constraint detected — ensure it is compatible with all other direct dependencies."
                )

        # ── 4. Security implications ──────────────────────────────────────
        if self._SECURITY_RISK_RE.search(text_to_check):
            warnings.append(
                "Contains a potentially unsafe shell/eval pattern — review carefully before applying."
            )
            has_blockers = True

        if "777" in text_to_check:
            warnings.append(
                "chmod 777 grants world-write permission — use the minimum required permission instead."
            )
            has_blockers = True

        if re.search(r"\$\{?\w+\}?(?:\s*\||\s*&&|\s*;)", text_to_check):
            warnings.append(
                "Unquoted variable expansion or shell chaining detected — risk of injection; quote all variables."
            )

        # ── 5. Manual review flag ─────────────────────────────────────────
        if _MANUAL_REVIEW_RE.search(text_to_check):
            requires_manual_review = True
            warnings.append(
                "MANUAL REVIEW REQUIRED: fix contains a destructive or high-risk command."
            )
            has_blockers = True

        if has_blockers and not requires_manual_review:
            # Any blocker that hasn't been explicitly flagged should still be manually checked
            requires_manual_review = True
            warnings.append(
                "MANUAL REVIEW REQUIRED: fix has one or more blocking issues (see warnings above)."
            )

        return dc_replace(
            fix,
            warnings=warnings,
            side_effects=side_effects,
            has_blockers=has_blockers,
            is_applicable=is_applicable,
            requires_manual_review=requires_manual_review,
        )


# ---------------------------------------------------------------------------
# Similarity-Based Ranker (Step 2)
# ---------------------------------------------------------------------------

class SimilarityRanker:
    """
    Rank fix candidates using the 5-dimension weighted scoring function.

    Ranking weights (per architecture spec):
      RCA Similarity      35%
      Error Similarity    25%
      Historical Success  20%
      Specificity         15%
      Implementation Ease  5%
    """

    def rank(
        self,
        candidates:    list[FixCandidate],
        rca_summary:   str,
        error_message: str,
    ) -> list[FixCandidate]:
        """Score each candidate and return a list sorted by final_score descending."""
        scored: list[FixCandidate] = []
        for fix in candidates:
            rca_sim   = compute_rca_similarity(fix, rca_summary)
            err_sim   = compute_error_similarity(fix, error_message)
            suc_rate  = fix.success_rate or get_success_rate(fix.fix_type, "")
            spec      = compute_specificity(fix)
            ease      = compute_implementation_ease(fix)

            updated = dc_replace(
                fix,
                rca_similarity      = rca_sim,
                error_similarity    = err_sim,
                success_rate        = suc_rate,
                specificity         = spec,
                implementation_ease = ease,
            )
            final = weighted_score(updated)
            scored.append(dc_replace(updated, final_score=round(final, 4)))

        scored.sort(key=lambda f: f.final_score, reverse=True)
        return scored


# ---------------------------------------------------------------------------
# Fix Recommendation Agent (main class)
# ---------------------------------------------------------------------------

class FixRecommendationAgent:
    """
    Orchestrates fix generation, ranking, validation, and output.

    Parameters
    ----------
    model_id : str, optional
        Ollama model identifier.  Defaults to the MODEL_ID env var, or
        ``llama3.1:8b`` — the same default as the RCA Agent.
    top_k : int
        Number of validated fixes to return.
    data_dir : Path
        Project data directory for KB access.
    """

    def __init__(
        self,
        model_id: str  = DEFAULT_MODEL_ID,
        top_k:    int  = DEFAULT_TOP_K,
        data_dir: Path = DATA_DIR,
    ) -> None:
        self.llm_generator = LLMFixGenerator(model_id=model_id)
        self.kb_retriever  = HistoricalFixRetriever(data_dir=data_dir)
        self.ranker        = SimilarityRanker()
        self.validator     = FixValidator()
        self.top_k         = top_k

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def recommend(
        self,
        rca:         dict[str, Any],
        error_block: dict[str, Any],
        web_context: list[dict[str, Any]] | None = None,
        metadata:    dict[str, Any] | None = None,
        top_k:       int | None = None,
    ) -> dict[str, Any]:
        """
        Generate and rank fix recommendations for the provided RCA.

        Parameters
        ----------
        rca : dict
            RCA result from the RCA Agent.  Must contain at minimum:
            rca_summary, error_type (or failure_type), error_code, error_signature.
        error_block : dict
            Primary enriched ErrorBlock dict from the preprocessing pipeline.
        web_context : list[dict], optional
            Ranked source results from the Web Search Agent.
        metadata : dict, optional
            Repository / environment metadata.
        top_k : int, optional
            Override the instance-level top_k.

        Returns
        -------
        dict
            Keys: fixes, total_candidates, generation_time,
                  generation_errors, validation_summary.
        """
        k       = top_k if top_k is not None else self.top_k
        t_start = time.monotonic()
        meta    = metadata or {}

        # ── Extract fields from RCA and error block ───────────────────────
        rca_summary   = str(rca.get("rca_summary", ""))
        error_type    = str(rca.get("error_type") or rca.get("failure_type") or "")
        error_message = str(error_block.get("error_message", ""))
        error_code    = str(rca.get("error_code", ""))
        error_sig     = str(rca.get("error_signature", ""))
        stack_trace   = error_block.get("stack_trace", [])
        repository    = str(meta.get("repository", ""))
        language      = str(
            meta.get("environment", {}).get("language", "")
            if isinstance(meta.get("environment"), dict) else ""
        )

        candidates: list[FixCandidate] = []
        generation_errors: dict[str, str | None] = {}

        # ── Source a: LLM generation ──────────────────────────────────────
        llm_fixes, llm_err = self.llm_generator.generate(
            rca_summary=rca_summary,
            error_type=error_type,
            error_message=error_message,
            error_code=error_code,
            stack_trace=stack_trace,
            repository=repository,
            language=language,
        )
        candidates.extend(llm_fixes)
        generation_errors["llm"] = llm_err
        logger.debug("LLM fixes: %d candidates  error=%s", len(llm_fixes), llm_err)

        # ── Source b: Historical KB ───────────────────────────────────────
        kb_fixes, kb_err = self.kb_retriever.retrieve(
            error_signature=error_sig or error_type,
            rca_summary=rca_summary,
            top_k=10,
        )
        candidates.extend(kb_fixes)
        generation_errors["kb"] = kb_err
        logger.debug("KB fixes: %d candidates  error=%s", len(kb_fixes), kb_err)

        # ── Source c: Templates ───────────────────────────────────────────
        template_fixes = get_template_fixes(error_type, error_code)
        candidates.extend(template_fixes)
        generation_errors["template"] = None
        logger.debug("Template fixes: %d candidates", len(template_fixes))

        # ── Source d: Web search ──────────────────────────────────────────
        web_fixes, web_err = extract_fixes_from_web(web_context or [], error_type)
        candidates.extend(web_fixes)
        generation_errors["web"] = web_err
        logger.debug("Web fixes: %d candidates  error=%s", len(web_fixes), web_err)

        total_candidates = len(candidates)

        # ── Deduplicate by description fingerprint ────────────────────────
        seen:   set[str]            = set()
        unique: list[FixCandidate]  = []
        for fix in candidates:
            key = (fix.description[:80] + fix.code_snippet[:40]).lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(fix)

        # ── Step 2: Similarity-based ranking ─────────────────────────────
        ranked = self.ranker.rank(unique, rca_summary, error_message)

        # ── Step 3: Validate top candidates ──────────────────────────────
        validated:      list[FixCandidate] = []
        manual_review:  list[FixCandidate] = []
        blocked:        list[FixCandidate] = []

        for fix in ranked[:k * 2]:      # validate 2× budget, keep best k
            v_fix = self.validator.validate(fix, error_type, language, meta)

            if v_fix.requires_manual_review:
                manual_review.append(v_fix)
            elif v_fix.has_blockers:
                blocked.append(v_fix)
            elif v_fix.is_applicable:
                validated.append(v_fix)
                if len(validated) >= k:
                    break

        # Fallback: if all candidates have blockers, promote manual-review
        # fixes to the output (clearly flagged) so the operator has something
        # to work with rather than an empty list.
        if not validated:
            if manual_review:
                logger.warning(
                    "All fix candidates require manual review — returning flagged fixes."
                )
                validated = manual_review[:k]
            elif blocked:
                logger.warning(
                    "All fix candidates are blocked — returning blocked fixes with warnings."
                )
                validated = [dc_replace(f, is_applicable=True) for f in blocked[:k]]
            else:
                # Absolute last resort: promote top-ranked as-is
                validated = [dc_replace(f, is_applicable=True) for f in ranked[:k]]

        # Assign final ranks
        top_fixes = [dc_replace(f, rank=i + 1) for i, f in enumerate(validated[:k])]

        elapsed = round(time.monotonic() - t_start, 3)
        logger.info(
            "FixRecommendationAgent: %d candidates → %d ranked → %d returned  "
            "(manual_review=%d blocked=%d)  time=%.2fs",
            total_candidates, len(ranked), len(top_fixes),
            len(manual_review), len(blocked), elapsed,
        )

        validation_summary = {
            "total_validated":      len(top_fixes),
            "manual_review_count":  len(manual_review),
            "blocked_count":        len(blocked),
            "applicable_count":     sum(1 for f in top_fixes if f.is_applicable),
        }

        # Attach per-fix explainable confidence reports
        fix_dicts = [self._fix_to_dict(f) for f in top_fixes]
        try:
            import sys as _sys
            from pathlib import Path as _Path
            _root = _Path(__file__).resolve().parents[1]
            if str(_root / "utility") not in _sys.path:
                _sys.path.insert(0, str(_root / "utility"))
            from utility.confidence_validator import ConfidenceValidator  # noqa: PLC0415
            _cv = ConfidenceValidator()
            for fix_dict in fix_dicts:
                _cr = _cv.validate_fix(fix_dict)
                fix_dict["confidence_report"] = _cr.to_dict()
        except Exception as _cv_exc:  # noqa: BLE001
            logger.debug("Fix confidence validation skipped (non-fatal): %s", _cv_exc)

        return {
            "fixes":              fix_dicts,
            "total_candidates":   total_candidates,
            "generation_time":    elapsed,
            "generation_errors":  {k: v for k, v in generation_errors.items() if v is not None},
            "validation_summary": validation_summary,
        }

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_to_dict(fix: FixCandidate) -> dict[str, Any]:
        """Serialise a FixCandidate to a clean dict for JSON output."""
        d = asdict(fix)
        d.pop("raw_similarity", None)   # internal field; not in output contract
        return d

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        fix_type:   str,
        error_type: str,
        success:    bool,
    ) -> None:
        """Record whether a recommended fix was successful to update success-rate priors."""
        record_fix_outcome(fix_type, error_type, success)


# ---------------------------------------------------------------------------
# CLI entry point for standalone testing
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Test the Fix Recommendation Agent standalone.")
    parser.add_argument("--rca-summary",   default="Dependency resolution failure for scikit-learn.")
    parser.add_argument("--error-type",    default="dependency_error")
    parser.add_argument("--error-message", default="ModuleNotFoundError: No module named 'sklearn'")
    parser.add_argument("--error-code",    default="EXIT_1")
    parser.add_argument("--repo",          default="scikit-learn/scikit-learn")
    parser.add_argument("--top-k",         type=int, default=5)
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Ollama model ID (default: %(default)s)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

    agent = FixRecommendationAgent(model_id=args.model_id, top_k=args.top_k)
    result = agent.recommend(
        rca={
            "rca_summary":     args.rca_summary,
            "error_type":      args.error_type,
            "error_code":      args.error_code,
            "error_signature": "dependency_error_module_not_found",
        },
        error_block={
            "error_message": args.error_message,
            "error_type":    args.error_type,
            "stack_trace":   [],
        },
        metadata={"repository": args.repo},
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
