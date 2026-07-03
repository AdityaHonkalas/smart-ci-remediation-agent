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
    a) LLM-based generation from the RCA analysis (Qwen-2.5-14B-Instruct)
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
    Returns top-K ranked fixes with implementation instructions.

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
    "fixes"             : list[RankedFix]
    "total_candidates"  : int
    "generation_time"   : float
}

Each RankedFix:
{
    "rank"             : int
    "description"      : str
    "fix_type"         : str
    "code_snippet"     : str
    "file_path"        : str
    "command"          : str
    "source"           : str   — 'llm'|'historical'|'template'|'web'
    "rca_similarity"   : float
    "error_similarity" : float
    "success_rate"     : float
    "specificity"      : float
    "implementation_ease": float
    "final_score"      : float
    "is_applicable"    : bool
    "has_blockers"     : bool
    "warnings"         : list[str]
    "side_effects"     : list[str]
    "estimated_effort" : str
}
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Add project root / utility / scripts to path
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "utility"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env_loader import load_dotenv  # noqa: E402
from db import SelfLearningKnowledgeBase  # noqa: E402
from utility.fix_utils import (   # noqa: E402
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
# Model settings (same Qwen client as RCA Agent)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_ALIASES: dict[str, str] = {
    "qwen-13b-instruct":        DEFAULT_MODEL_ID,
    "qwen2.5-13b-instruct":     DEFAULT_MODEL_ID,
    "qwen2.5-14b-instruct":     DEFAULT_MODEL_ID,
    "qwen/qwen2.5-14b-instruct":DEFAULT_MODEL_ID,
}
DEFAULT_TOP_K = int(os.getenv("FIX_TOP_K", "5"))
DATA_DIR      = _ROOT / "data"

_FIX_SYSTEM_PROMPT = (
    "You are a CI/CD fix recommendation agent. "
    "Given an RCA report and error context, generate concrete, actionable fix "
    "recommendations. Return a JSON array where each element has: "
    "description (str), fix_type (str: code_change|config_change|dependency_update|command), "
    "code_snippet (str, optional), file_path (str, optional), command (str, optional), "
    "estimated_effort (str: low|medium|high). "
    "Be specific — reference actual files, packages, config keys, and commands."
)


# ---------------------------------------------------------------------------
# LLM Fix Generator
# ---------------------------------------------------------------------------

class LLMFixGenerator:
    """
    Generate fix candidates using Qwen-2.5-14B-Instruct via the HF Inference API
    or a local transformers pipeline.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        max_new_tokens: int = 1024,
    ) -> None:
        self.model_id       = MODEL_ALIASES.get(model_id.lower(), model_id)
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        rca_summary:   str,
        error_type:    str,
        error_message: str,
        error_code:    str = "",
        stack_trace:   list[str] | None = None,
        repository:    str = "",
        language:      str = "",
    ) -> list[FixCandidate]:
        """
        Ask the model to generate fix candidates for the given RCA.

        Returns a list of FixCandidate objects; falls back to an empty list
        on any model error.
        """
        prompt = self._build_prompt(
            rca_summary, error_type, error_message, error_code, stack_trace, repository, language
        )
        try:
            raw = self._call_model(prompt)
            return self._parse_response(raw, error_type)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM fix generation failed: %s", exc)
            return []

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
        return (
            "Generate fix recommendations for the following CI/CD failure RCA.\n"
            "Return a JSON array of fix objects. Each fix must have: "
            "description, fix_type, code_snippet (if applicable), "
            "file_path (if applicable), command (if applicable), estimated_effort.\n\n"
            f"{json.dumps(payload, indent=2)}"
        )

    def _call_model(self, prompt: str) -> str:
        provider = (os.getenv("MODEL_PROVIDER") or "").strip().lower()
        if provider == "local":
            return self._call_local(prompt)
        return self._call_hf_api(prompt)

    def _call_hf_api(self, prompt: str) -> str:
        api_key = os.getenv("HF_API_KEY") or os.getenv("MODEL_API_KEY")
        if not api_key:
            raise RuntimeError("HF_API_KEY required for fix LLM generation.")
        url = os.getenv("HF_API_URL") or f"https://api-inference.huggingface.co/models/{self.model_id}"
        body = json.dumps({
            "inputs": {
                "messages": [
                    {"role": "system", "content": _FIX_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ]
            },
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature":    0.2,
                "top_p":          0.9,
                "do_sample":      False,
                "return_full_text": False,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HF API {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc
        if isinstance(payload, list) and payload:
            first = payload[0]
            gt = first.get("generated_text")
            if isinstance(gt, list):
                for msg in reversed(gt):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        return str(msg.get("content", "")).strip()
            return str(gt or "").strip()
        return json.dumps(payload)

    def _call_local(self, prompt: str) -> str:
        try:
            from transformers import pipeline  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("transformers + torch required for local inference") from exc
        pipe = pipeline(
            "text-generation", model=self.model_id,
            device_map="auto", torch_dtype=torch.float16,
        )
        messages = [
            {"role": "system", "content": _FIX_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        result = pipe(messages, max_new_tokens=self.max_new_tokens, do_sample=False, return_full_text=False)
        if isinstance(result, list) and result:
            first = result[0]
            gt = first.get("generated_text")
            if isinstance(gt, list):
                for msg in reversed(gt):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        return str(msg.get("content", "")).strip()
            return str(gt or "").strip()
        return str(result)

    @staticmethod
    def _parse_response(text: str, error_type: str) -> list[FixCandidate]:
        """Parse JSON array from model response into FixCandidate objects."""
        import re
        # Extract JSON array
        match = re.search(r"\[.*\]", text, re.S)
        if not match:
            return []
        try:
            items = json.loads(match.group())
        except json.JSONDecodeError:
            return []
        candidates: list[FixCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            candidates.append(FixCandidate(
                description=str(item.get("description", "")),
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
        return candidates


# ---------------------------------------------------------------------------
# Historical Fix Retriever (from KB)
# ---------------------------------------------------------------------------

class HistoricalFixRetriever:
    """Retrieve previously successful fix descriptions from the KB."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.kb = SelfLearningKnowledgeBase(
            persist_dir=data_dir / "rca_knowledge_base"
        )

    def retrieve(
        self,
        error_signature: str,
        rca_summary: str,
        top_k: int = 10,
    ) -> list[FixCandidate]:
        """Query KB for similar RCA entries and reuse their inline_fix_suggestions."""
        entries = self.kb.search(error_signature, top_k=top_k)
        candidates: list[FixCandidate] = []
        for entry in entries:
            if entry.similarity < 0.5:
                continue
            # The KB stores RCA dicts; inline_fix_suggestions may be present
            rca_stored = self.kb._rca_repository.get(entry.error_signature, {})
            for fix in rca_stored.get("inline_fix_suggestions", []) or []:
                if not isinstance(fix, dict):
                    continue
                desc = str(fix.get("suggested_change") or fix.get("description") or "")
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
        return candidates


# ---------------------------------------------------------------------------
# Web Fix Extractor
# ---------------------------------------------------------------------------

def extract_fixes_from_web(
    web_context: list[dict[str, Any]],
    error_type: str,
) -> list[FixCandidate]:
    """
    Extract actionable fix hints from Web Search Agent results.

    Looks for fix-like snippets in the ``snippet`` field of each source.
    """
    import re
    fix_keywords = re.compile(
        r"\b(?:fix|solution|workaround|resolved by|change|update|pin|add|remove|set|replace)\b",
        re.I,
    )
    candidates: list[FixCandidate] = []
    for src in web_context or []:
        snippet = str(src.get("snippet") or "")
        title   = str(src.get("title") or "")
        url     = str(src.get("url") or "")
        if not fix_keywords.search(snippet) and not fix_keywords.search(title):
            continue
        # Include source URL in the description for model traceability
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
    return candidates


# ---------------------------------------------------------------------------
# Fix Validator
# ---------------------------------------------------------------------------

class FixValidator:
    """
    Validate whether a fix is applicable and flag side-effects or blockers.
    """

    _SECURITY_RISK_RE = __import__("re").compile(
        r"\b(?:eval|exec|shell=True|os\.system|subprocess\.run.*shell=True|"
        r"chmod 777|curl.*\|.*sh|wget.*\|.*sh)\b",
        __import__("re").I,
    )

    def validate(
        self,
        fix: FixCandidate,
        error_type: str,
        language:   str = "",
    ) -> FixCandidate:
        """
        Validate *fix* in-place and return the updated candidate.

        Checks:
        - Applicability: fix type matches error type
        - Security: flag risky patterns in code_snippet / command
        - Side effects: flag high-effort changes
        """
        warnings:     list[str] = list(fix.warnings)
        side_effects: list[str] = list(fix.side_effects)
        has_blockers  = fix.has_blockers
        is_applicable = fix.is_applicable

        # Security check
        text_to_check = fix.code_snippet + " " + fix.command
        if self._SECURITY_RISK_RE.search(text_to_check):
            warnings.append("Contains potentially unsafe shell/eval pattern — review before applying.")
            has_blockers = True

        # Effort warning
        if fix.estimated_effort == "high":
            side_effects.append("High-effort change — may affect multiple components.")

        # Type applicability
        if error_type == "test_failure" and fix.fix_type == "dependency_update":
            warnings.append("Dependency update may not resolve a test logic failure — verify root cause.")

        if error_type == "permission_error" and "777" in fix.code_snippet:
            warnings.append("chmod 777 is a security risk; prefer specific permission grants.")
            has_blockers = True

        # Empty fix is not applicable
        if not fix.description.strip() and not fix.code_snippet.strip() and not fix.command.strip():
            is_applicable = False

        from dataclasses import replace as dc_replace
        return dc_replace(
            fix,
            warnings=warnings,
            side_effects=side_effects,
            has_blockers=has_blockers,
            is_applicable=is_applicable,
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
        """
        Score each candidate and return a list sorted by final_score descending.
        """
        from dataclasses import replace as dc_replace
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
        Override the default Qwen model ID for LLM fix generation.
    top_k : int
        Number of validated fixes to return.
    data_dir : Path
        Project data directory for KB access.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        top_k:    int = DEFAULT_TOP_K,
        data_dir: Path = DATA_DIR,
    ) -> None:
        self.llm_generator   = LLMFixGenerator(model_id=model_id)
        self.kb_retriever    = HistoricalFixRetriever(data_dir=data_dir)
        self.ranker          = SimilarityRanker()
        self.validator       = FixValidator()
        self.top_k           = top_k

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
            RCA result from the RCA Agent (must have at minimum rca_summary,
            error_type, error_code, error_signature).
        error_block : dict
            Primary enriched ErrorBlock dict from the preprocessing pipeline.
        web_context : list[dict], optional
            Web search results from the Web Search Agent.
        metadata : dict, optional
            Repository / environment metadata.
        top_k : int, optional
            Override the instance-level top_k.

        Returns
        -------
        dict with keys: fixes, total_candidates, generation_time
        """
        k = top_k if top_k is not None else self.top_k
        t_start = time.monotonic()
        meta = metadata or {}

        rca_summary   = str(rca.get("rca_summary", ""))
        error_type    = str(rca.get("error_type") or rca.get("failure_type") or "")
        error_message = str(error_block.get("error_message", ""))
        error_code    = str(rca.get("error_code", ""))
        error_sig     = str(rca.get("error_signature", ""))
        stack_trace   = error_block.get("stack_trace", [])
        repository    = str(meta.get("repository", ""))
        language      = str(meta.get("environment", {}).get("language", "") if isinstance(meta.get("environment"), dict) else "")

        candidates: list[FixCandidate] = []

        # ---- Source a: LLM generation ----
        llm_fixes = self.llm_generator.generate(
            rca_summary=rca_summary,
            error_type=error_type,
            error_message=error_message,
            error_code=error_code,
            stack_trace=stack_trace,
            repository=repository,
            language=language,
        )
        candidates.extend(llm_fixes)
        logger.debug("LLM fixes: %d candidates", len(llm_fixes))

        # ---- Source b: Historical KB ----
        kb_fixes = self.kb_retriever.retrieve(
            error_signature=error_sig or error_type,
            rca_summary=rca_summary,
            top_k=10,
        )
        candidates.extend(kb_fixes)
        logger.debug("KB fixes: %d candidates", len(kb_fixes))

        # ---- Source c: Templates ----
        template_fixes = get_template_fixes(error_type, error_code)
        candidates.extend(template_fixes)
        logger.debug("Template fixes: %d candidates", len(template_fixes))

        # ---- Source d: Web search ----
        web_fixes = extract_fixes_from_web(web_context or [], error_type)
        candidates.extend(web_fixes)
        logger.debug("Web fixes: %d candidates", len(web_fixes))

        total_candidates = len(candidates)

        # ---- Deduplicate by description fingerprint ----
        seen: set[str] = set()
        unique: list[FixCandidate] = []
        for fix in candidates:
            key = (fix.description[:80] + fix.code_snippet[:40]).lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(fix)

        # ---- Step 2: Similarity-based ranking ----
        ranked = self.ranker.rank(unique, rca_summary, error_message)

        # ---- Step 3: Validate top candidates ----
        validated: list[FixCandidate] = []
        for fix in ranked[:k * 2]:  # validate 2x candidates, keep best k
            v_fix = self.validator.validate(fix, error_type, language)
            if not v_fix.has_blockers and v_fix.is_applicable:
                validated.append(v_fix)
                if len(validated) >= k:
                    break

        # If all have blockers, include top-k even with warnings
        if not validated:
            from dataclasses import replace as dc_replace
            validated = [dc_replace(f, is_applicable=True) for f in ranked[:k]]

        # Assign final ranks
        from dataclasses import replace as dc_replace
        top_fixes = [dc_replace(f, rank=i + 1) for i, f in enumerate(validated[:k])]

        elapsed = round(time.monotonic() - t_start, 3)
        logger.info(
            "FixRecommendationAgent: %d candidates → %d ranked → %d returned (%.2fs)",
            total_candidates, len(ranked), len(top_fixes), elapsed,
        )

        return {
            "fixes": [self._fix_to_dict(f) for f in top_fixes],
            "total_candidates": total_candidates,
            "generation_time":  elapsed,
        }

    @staticmethod
    def _fix_to_dict(fix: FixCandidate) -> dict[str, Any]:
        """Serialise a FixCandidate to a clean dict for JSON output."""
        d = asdict(fix)
        # Remove internal fields not part of the output contract
        d.pop("raw_similarity", None)
        return d

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
    parser.add_argument("--rca-summary",  default="Dependency resolution failure for scikit-learn.")
    parser.add_argument("--error-type",   default="dependency_error")
    parser.add_argument("--error-message",default="ModuleNotFoundError: No module named 'sklearn'")
    parser.add_argument("--error-code",   default="EXIT_1")
    parser.add_argument("--repo",         default="scikit-learn/scikit-learn")
    parser.add_argument("--top-k",        type=int, default=5)
    args = parser.parse_args()

    agent = FixRecommendationAgent(top_k=args.top_k)
    result = agent.recommend(
        rca={
            "rca_summary": args.rca_summary,
            "error_type":  args.error_type,
            "error_code":  args.error_code,
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
