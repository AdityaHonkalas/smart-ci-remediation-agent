#!/usr/bin/env python3
"""
scripts/run_rca_batch.py
────────────────────────
Standalone batch runner: extracts the last N failed GitHub Actions run URLs
for a list of repositories, runs the full RCA pipeline for each one, and
triggers a ChromaDB knowledge-base update after every successful diagnosis.

Usage
─────
    # Basic — process last 10 failed runs for each repo in repos.txt
    python scripts/run_rca_batch.py

    # Custom repo file, fetch last 5 failed runs, skip LLM inference
    python scripts/run_rca_batch.py --repos-file my_repos.txt --last-n 5 --no-model

    # Dry-run: only fetch & save failed_runs.json, skip RCA
    python scripts/run_rca_batch.py --last-n 20 --dry-run

    # Explicit token and output directory
    python scripts/run_rca_batch.py --token ghp_xxx --output-dir results/

CLI Flags
─────────
    --repos-file     Path to plain-text file with owner/repo entries (default: repos.txt)
    --last-n         Number of most-recent failed runs to fetch per repo (default: 10)
    --output-dir     Directory where failed_runs.json is written (default: output/)
    --token          GitHub personal access token (default: $GITHUB_ACCESS_TOKEN)
    --no-model       Disable LLM inference; produce heuristic-only RCA
    --sleep-seconds  Delay in seconds between consecutive RCA runs (default: 2.0)
    --dry-run        Fetch & save failed run URLs only; skip the RCA pipeline

Output
──────
    output/failed_runs.json   — array of {repo, run_id, run_url, workflow, created_at}
    logs/rca_batch_runner.log — rotating log file (5 MB × 3 backups)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — must happen before any local imports
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/
_ROOT_DIR   = _SCRIPT_DIR.parent                       # project root
_AGENTS_DIR = _ROOT_DIR / "agents"

for _p in (str(_ROOT_DIR), str(_SCRIPT_DIR), str(_AGENTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env_loader import load_dotenv  # noqa: E402
load_dotenv(_ROOT_DIR / ".env")

# ---------------------------------------------------------------------------
# Logging — stdout + rotating file
# ---------------------------------------------------------------------------

def _configure_logging() -> logging.Logger:
    log_dir = _ROOT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "rca_batch_runner.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)


logger = _configure_logging()

# ---------------------------------------------------------------------------
# Lazy-load heavy modules (mirrors diagnosis_agent.py pattern)
# ---------------------------------------------------------------------------

def _load_script_module(module_name: str, path: Path) -> Any:
    """Load a Python file as a module by absolute path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load {module_name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _lazy_load_modules() -> tuple[Any, Any, Any]:
    """Load log_collector, diagnosis_agent, and kb_batch_update modules."""
    logger.info("Loading pipeline modules …")
    log_collector   = _load_script_module(
        "smart_ci_log_collector", _SCRIPT_DIR / "log-collector.py"
    )
    diagnosis_agent = _load_script_module(
        "smart_ci_diagnosis_agent", _AGENTS_DIR / "diagnosis_agent.py"
    )
    kb_batch_update = _load_script_module(
        "smart_ci_kb_batch_update", _SCRIPT_DIR / "kb_batch_update.py"
    )
    logger.info("Pipeline modules loaded.")
    return log_collector, diagnosis_agent, kb_batch_update


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _positive_int(value: str) -> int:
    """argparse type validator — rejects zero and negatives."""
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"--last-n must be ≥ 1, got {value}")
    return ivalue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch RCA runner: fetch failed GitHub Actions runs and diagnose them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repos-file",
        default="repos.txt",
        help="Plain-text file with owner/repo entries, one per line.",
    )
    parser.add_argument(
        "--last-n",
        type=_positive_int,
        default=10,
        dest="last_n",
        metavar="N",
        help="Number of most-recent failed runs to fetch per repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        dest="output_dir",
        help="Directory where failed_runs.json is written.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("GITHUB_ACCESS_TOKEN"),
        help="GitHub personal access token.",
    )
    parser.add_argument(
        "--no-model",
        action="store_false",
        dest="use_model",
        help="Disable LLM inference; produce heuristic-only RCA.",
    )
    parser.set_defaults(use_model=True)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=2.0,
        dest="sleep_seconds",
        help="Delay in seconds between consecutive RCA pipeline runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Fetch and save failed run URLs only; skip the RCA pipeline.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sub-Task 2 helpers — repo reader & failed run URL extractor
# ---------------------------------------------------------------------------

def read_repos(path: Path, log_collector: Any) -> list[str]:
    """
    Read owner/repo entries from *path*.

    Blank lines and lines beginning with ``#`` are ignored.
    Each entry is normalised via ``log_collector.normalize_repo()``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Repos file not found: {path}")

    repos: list[str] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                repos.append(log_collector.normalize_repo(line))
            except ValueError as exc:
                logger.warning("Skipping invalid repo entry %r: %s", line, exc)
                skipped += 1

    logger.info(
        "Repos file %s: %d valid repo(s) loaded, %d skipped.",
        path, len(repos), skipped,
    )
    return repos


def fetch_last_n_failed_runs(
    client: Any,
    repo: str,
    last_n: int,
    log_collector: Any,
) -> list[dict[str, Any]]:
    """
    Return the last *last_n* failed workflow runs for *repo*.

    Delegates to ``log_collector.list_failed_runs()`` which already applies
    FAILURE_CONCLUSIONS filtering and stops at *limit*.
    """
    return log_collector.list_failed_runs(
        client=client,
        repo=repo,
        limit=last_n,
        per_page=50,
        max_pages=10,
        branch=None,
        workflow=None,
        include_cancelled=False,
    )


def save_failed_runs(runs: list[dict[str, Any]], output_path: Path) -> None:
    """Atomically write the failed-run records to *output_path* as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(runs, fh, indent=2, default=str)
    tmp.replace(output_path)
    logger.info("Saved %d failed run record(s) → %s", len(runs), output_path)


# ---------------------------------------------------------------------------
# Sub-Task 3 helpers — RCA pipeline runner
# ---------------------------------------------------------------------------

def run_rca_for_url(
    run_url: str,
    token: str | None,
    use_model: bool,
    diagnosis_agent: Any,
) -> dict[str, Any] | None:
    """
    Execute the full 8-step RCA pipeline for *run_url*.

    Returns the result dict on success; logs the error and returns ``None``
    on any exception so the batch loop can continue.
    """
    try:
        return diagnosis_agent.diagnose_workflow_run(
            run_url=run_url,
            token=token,
            use_model=use_model,
        )
    except SystemExit as exc:
        # diagnose_workflow_run calls exit(1) on empty URL — guard just in case
        logger.error("RCA pipeline called exit(%s) for %s", exc.code, run_url)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("RCA pipeline raised an unexpected error for %s: %s", run_url, exc)
        return None


# ---------------------------------------------------------------------------
# Sub-Task 4 helpers — KB update trigger
# ---------------------------------------------------------------------------

def trigger_kb_update(kb_batch_update: Any) -> dict[str, Any]:
    """
    Run the offline KB batch update to persist any staged RCA entries.

    Returns the summary dict ``{processed, skipped, errors, timestamp}``.
    Logs a warning on exception without re-raising so the batch loop is
    not interrupted.
    """
    try:
        summary = kb_batch_update.run_kb_batch_update(data_dir=_ROOT_DIR / "data")
        logger.info(
            "KB update: processed=%d  skipped=%d  errors=%d",
            summary.get("processed", 0),
            summary.get("skipped",   0),
            len(summary.get("errors", [])),
        )
        if summary.get("errors"):
            for err in summary["errors"]:
                logger.warning("KB update error: %s", err)
        return summary
    except Exception as exc:  # noqa: BLE001
        logger.warning("KB batch update raised an exception (non-fatal): %s", exc)
        return {"processed": 0, "skipped": 0, "errors": [str(exc)]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:  # noqa: C901 — accepted complexity for a batch orchestrator
    args = parse_args()

    repos_path  = Path(args.repos_file)
    output_path = args.output_dir / "failed_runs.json"

    # ------------------------------------------------------------------
    # Validate repos file exists before loading heavy modules
    # ------------------------------------------------------------------
    if not repos_path.exists():
        logger.error("Repos file not found: %s", repos_path)
        return 1

    # ------------------------------------------------------------------
    # Load modules
    # ------------------------------------------------------------------
    log_collector, diagnosis_agent, kb_batch_update = _lazy_load_modules()

    # ------------------------------------------------------------------
    # Sub-Task 2 — Read repos & extract failed run URLs
    # ------------------------------------------------------------------
    try:
        repos = read_repos(repos_path, log_collector)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if not repos:
        logger.error("No valid repositories found in %s — aborting.", repos_path)
        return 1

    github_client = log_collector.GitHubActionsClient(token=args.token)

    all_failed_runs: list[dict[str, Any]] = []
    repo_run_counts: dict[str, int] = {}

    logger.info(
        "═══ Phase 1: Fetching failed run URLs — %d repo(s), last %d run(s) each ═══",
        len(repos), args.last_n,
    )

    for repo in repos:
        logger.info("→ %s: fetching last %d failed run(s) …", repo, args.last_n)
        try:
            runs = fetch_last_n_failed_runs(github_client, repo, args.last_n, log_collector)
        except RuntimeError as exc:
            logger.warning("  %s: GitHub API error — %s (skipping)", repo, exc)
            repo_run_counts[repo] = 0
            continue

        if not runs:
            logger.info("  %s: 0 failed runs found — skipping.", repo)
            repo_run_counts[repo] = 0
            continue

        for run in runs:
            all_failed_runs.append({
                "repo":       repo,
                "run_id":     run.get("id"),
                "run_url":    run.get("html_url", ""),
                "workflow":   run.get("name", ""),
                "created_at": run.get("created_at", ""),
                "conclusion": run.get("conclusion", ""),
            })

        repo_run_counts[repo] = len(runs)
        logger.info(
            "  %s: %d failed run(s) found (last %d requested).",
            repo, len(runs), args.last_n,
        )

    # Save the collected URLs regardless of dry-run
    if all_failed_runs:
        save_failed_runs(all_failed_runs, output_path)
    else:
        logger.warning("No failed runs found across all repositories — nothing to save.")
        return 0

    total_runs = len(all_failed_runs)
    logger.info(
        "Phase 1 complete: %d failed run(s) across %d repo(s) → %s",
        total_runs, len(repos), output_path,
    )

    # ------------------------------------------------------------------
    # Dry-run exit
    # ------------------------------------------------------------------
    if args.dry_run:
        logger.info("--dry-run set: skipping RCA pipeline.")
        return 0

    # ------------------------------------------------------------------
    # Sub-Tasks 3 & 4 — RCA pipeline + KB update per run
    # ------------------------------------------------------------------
    logger.info(
        "═══ Phase 2: Running RCA pipeline — %d run(s) total ═══", total_runs
    )

    succeeded  = 0
    failed     = 0
    kb_processed_total = 0
    kb_skipped_total   = 0
    kb_errors_total    = 0

    for idx, run_record in enumerate(all_failed_runs, start=1):
        repo    = run_record["repo"]
        run_id  = run_record["run_id"]
        run_url = run_record["run_url"]

        if not run_url:
            logger.warning("[%d/%d] %s run %s: missing html_url — skipping.", idx, total_runs, repo, run_id)
            failed += 1
            continue

        logger.info(
            "[%d/%d] Starting RCA — %s  run_id=%s  url=%s",
            idx, total_runs, repo, run_id, run_url,
        )

        result = run_rca_for_url(run_url, args.token, args.use_model, diagnosis_agent)

        if result is None:
            logger.error("[%d/%d] ✗ RCA FAILED — %s run %s", idx, total_runs, repo, run_id)
            failed += 1
        else:
            rca             = result.get("rca", {})
            failure_type    = rca.get("failure_type", "unknown")
            confidence      = rca.get("confidence", "unknown")
            gen_time        = result.get("generation_time", 0.0)
            error_signature = result.get("error_signature", "")

            logger.info(
                "[%d/%d] ✓ RCA complete — %s run %s  "
                "failure_type=%s  confidence=%s  generation_time=%.1fs  signature=%s",
                idx, total_runs, repo, run_id,
                failure_type, confidence, gen_time,
                error_signature[:60] if error_signature else "—",
            )
            succeeded += 1

            # Sub-Task 4 — trigger KB update immediately after successful RCA
            kb_summary = trigger_kb_update(kb_batch_update)
            kb_processed_total += kb_summary.get("processed", 0)
            kb_skipped_total   += kb_summary.get("skipped",   0)
            kb_errors_total    += len(kb_summary.get("errors", []))

        # Polite delay between runs (skip after the last one)
        if idx < total_runs and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("═" * 66)
    logger.info("BATCH SUMMARY")
    logger.info("  Repositories    : %d", len(repos))
    logger.info("  Failed runs found  : %d", total_runs)
    logger.info("  RCA succeeded   : %d", succeeded)
    logger.info("  RCA failed      : %d", failed)
    logger.info("  KB entries added: %d  (skipped=%d  errors=%d)",
                kb_processed_total, kb_skipped_total, kb_errors_total)
    logger.info("  Output file     : %s", output_path)
    logger.info("═" * 66)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
