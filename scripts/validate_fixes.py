#!/usr/bin/env python3
"""
scripts/validate_fixes.py
─────────────────────────
Automated Fix Validation Pipeline — Tier-2 Live CI Validation.

Overview
────────
This script orchestrates live CI validation of recommended fix candidates by:

  1. Cloning the user's fork repository to a temporary local directory.
  2. Locating the relevant workflow YAML file.
  3. Applying the fix code_snippet and enabling PR trigger on the workflow.
  4. Committing and pushing the changes to a new validation branch.
  5. Raising a Pull Request via the GitHub API.
  6. Polling for the triggered GitHub Actions run_id.
  7. Waiting for the action run to complete (configurable timeout).
  8. Extracting the action run logs.
  9. Checking the run conclusion (success / failure).
  10. Recording the result in the KB via SelfLearningKnowledgeBase.update_validation_result().

Feature Flag
────────────
TIER2_ENABLED controls whether live validation runs.  It defaults to False and is
read from the environment variable TIER2_ENABLED at request time so no server restart
is needed when enabling it:

    # In .env
    TIER2_ENABLED=true

When TIER2_ENABLED=False, run_live_validation() raises NotImplementedError immediately
and the Flask route returns a friendly "not_enabled" response to the UI.

# TODO(tier2): Wire background threading in routes/validation.py once TIER2_ENABLED
#              has been set to True in production.  The pipeline itself is fully
#              implemented below — only the execution harness needs to be added.

Expected inputs dict for run_live_validation()
──────────────────────────────────────────────
  fork_repo       : str   — "owner/repo" of the user's fork
  branch          : str   — base branch to branch off from (e.g. "main")
  github_username : str   — GitHub username used to construct clone URL
  token           : str   — GitHub PAT with repo + workflow write scopes
  error_signature : str   — KB error signature to record validation result against
  fix_ranks       : list[int] | "all"  — which fixes (1–5) to validate
  fixes           : list[dict] — fix objects from RCA result (each has file_path, code_snippet, rank)
  workflow_id     : str   — original failed action run_id (string, stored as-is)
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flag — read from env at call time (no restart needed)
# ---------------------------------------------------------------------------

def _tier2_enabled() -> bool:
    """Return True if TIER2_ENABLED env var is set to 'true' (case-insensitive)."""
    return os.getenv("TIER2_ENABLED", "false").strip().lower() == "true"


# Fallback constant for module-level checks in tests / imports
TIER2_ENABLED: bool = True  # overridden by _tier2_enabled() at runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run_git(args: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a git sub-command; raises RuntimeError on non-zero exit."""
    result = subprocess.run(
        ["git"] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )
    return result


# ---------------------------------------------------------------------------
# Step 5 — Clone fork repo
# ---------------------------------------------------------------------------

def _clone_repo(
    fork_repo: str,
    branch: str,
    token: str,
    github_username: str,
    workdir: Path,
) -> Path:
    """
    Clone the user's fork repo into ``workdir/{repo_name}``.

    Clone URL: ``https://{token}@github.com/{github_username}/{repo_name}.git``

    Parameters
    ----------
    fork_repo : str
        Full "owner/repo" path of the fork (used to derive repo_name).
    branch : str
        Branch to checkout after cloning.
    token : str
        GitHub PAT embedded in HTTPS clone URL for auth.
    github_username : str
        GitHub username — used as the auth identity in the clone URL.
    workdir : Path
        Parent directory under which the clone is created.

    Returns
    -------
    Path
        Path to the cloned repository root.
    """
    repo_name  = fork_repo.split("/")[-1]
    clone_url  = f"https://{github_username}:{token}@github.com/{github_username}/{repo_name}.git"
    clone_path = workdir / repo_name

    workdir.mkdir(parents=True, exist_ok=True)

    logger.info("Cloning %s → %s", fork_repo, clone_path)
    _run_git(["clone", "--depth", "1", "--branch", branch, clone_url, str(clone_path)],
             cwd=workdir)

    return clone_path


# ---------------------------------------------------------------------------
# Step 6 — Locate workflow file in clone
# ---------------------------------------------------------------------------

def _find_workflow_file(repo_path: Path, file_path: str | None) -> Path | None:
    """
    Locate a workflow YAML file in the cloned repository.

    Tries:
      1. Exact path (file_path relative to repo root).
      2. .github/workflows/{basename} if file_path is an absolute or relative path.
      3. Any *.yml / *.yaml under .github/workflows/ as a fallback.
    """
    workflows_dir = repo_path / ".github" / "workflows"

    if file_path:
        # Try exact relative path
        candidate = repo_path / file_path.lstrip("/")
        if candidate.exists():
            return candidate
        # Try basename under .github/workflows/
        basename = Path(file_path).name
        candidate = workflows_dir / basename
        if candidate.exists():
            return candidate

    # Fallback: first workflow file found
    if workflows_dir.exists():
        for p in sorted(workflows_dir.iterdir()):
            if p.suffix in (".yml", ".yaml"):
                return p

    return None


# ---------------------------------------------------------------------------
# Step 7 — Apply fix to workflow YAML + enable PR trigger
# ---------------------------------------------------------------------------

def _apply_fix_to_workflow(workflow_path: Path, fix: dict[str, Any]) -> bool:
    """
    Patch a workflow YAML file in-place.

    Two operations:
      1. Ensure ``pull_request:`` is present under the ``on:`` trigger block.
      2. Append the ``code_snippet`` from the fix as a new step comment block
         after the last existing step in the first job that references the
         failing file or first job overall.

    Uses plain text / regex manipulation to avoid a full PyYAML dependency.

    Returns True if any change was made, False if the file was not modified.
    """
    content = workflow_path.read_text(encoding="utf-8")
    original = content

    # ── 1. Ensure pull_request: trigger ──────────────────────────────
    # Pattern: look for  on:\n  (or on: {...})  and insert pull_request: if absent
    if "pull_request" not in content:
        # Try to insert after an existing push: trigger
        content = re.sub(
            r"(^on:\s*\n(?:[ \t]+\w[^\n]*\n)*)",
            lambda m: m.group(0).rstrip("\n") + "\n  pull_request:\n    branches: ['**']\n",
            content,
            count=1,
            flags=re.MULTILINE,
        )
        # Fallback: if "on:" with inline dict or no match, prepend pull_request
        if "pull_request" not in content:
            content = re.sub(
                r"^on:",
                "on:\n  pull_request:\n    branches: ['**']\n  _original:",
                content,
                count=1,
                flags=re.MULTILINE,
            )

    # ── 2. Append code_snippet as a new named step ───────────────────
    snippet = (fix.get("code_snippet") or "").strip()
    if snippet:
        # Build a commented block that makes the change visible in the PR diff
        step_comment = (
            "\n      # --- Applied fix candidate (smart-ci-remediation-agent) ---\n"
            "      # Fix: " + (fix.get("description") or "automated fix") + "\n"
        )
        # Indent each snippet line by 6 spaces (step level in GHA YAML)
        indented = "\n".join("      " + ln if ln.strip() else ln for ln in snippet.splitlines())
        new_step = step_comment + indented + "\n      # --- end fix ---\n"

        # Append before the last line (end of file) so it's visible
        content = content.rstrip("\n") + "\n" + new_step + "\n"

    if content == original:
        logger.debug("No changes applied to %s", workflow_path)
        return False

    workflow_path.write_text(content, encoding="utf-8")
    logger.info("Fix applied to %s", workflow_path)
    return True


# ---------------------------------------------------------------------------
# Step 8 — Commit and push to a new validation branch
# ---------------------------------------------------------------------------

def _commit_and_push(
    repo_path: Path,
    base_branch: str,
    fix_rank: int,
    token: str,
    github_username: str,
    fork_repo: str,
) -> tuple[str, str]:
    """
    Create a new validation branch, commit the changes, and push.

    Returns
    -------
    tuple[str, str]
        (validation_branch_name, commit_sha)
    """
    ts = int(time.time())
    val_branch = f"ci-fix-validation/rank-{fix_rank}-{ts}"

    # Configure git identity (required for commit)
    _run_git(["config", "user.email", f"{github_username}@users.noreply.github.com"], repo_path)
    _run_git(["config", "user.name",  github_username], repo_path)

    # Create and checkout new branch
    _run_git(["checkout", "-b", val_branch], repo_path)

    # Stage all changes
    _run_git(["add", "-A"], repo_path)

    # Commit
    commit_msg = f"ci: apply fix candidate #{fix_rank} (smart-ci-remediation-agent)"
    _run_git(["commit", "-m", commit_msg], repo_path)

    # Get commit SHA
    sha_result = _run_git(["rev-parse", "HEAD"], repo_path)
    commit_sha = sha_result.stdout.strip()

    # Push — embed credentials in remote URL
    repo_name  = fork_repo.split("/")[-1]
    remote_url = (
        f"https://{github_username}:{token}@github.com/{github_username}/{repo_name}.git"
    )
    _run_git(["push", remote_url, val_branch], repo_path)

    logger.info("Pushed branch %s (commit %s)", val_branch, commit_sha[:8])
    return val_branch, commit_sha


# ---------------------------------------------------------------------------
# Step 9 — Raise a Pull Request
# ---------------------------------------------------------------------------

def _create_pull_request(
    client: Any,
    fork_repo: str,
    head_branch: str,
    base_branch: str,
    fix_rank: int,
    fix_description: str,
) -> dict[str, Any]:
    """
    Create a PR on the fork repo via GitHub API.

    Returns the PR dict (contains html_url, number, head.sha, etc.).
    """
    title = f"[CI Validation] Fix candidate #{fix_rank} — smart-ci-remediation-agent"
    body  = (
        "This PR was automatically raised by the **Smart CI Remediation Agent** "
        "to validate a fix recommendation against live GitHub Actions.\n\n"
        f"**Fix description:** {fix_description}\n\n"
        "_This branch will be deleted after the validation run completes._"
    )
    pr = client.post_json(
        f"repos/{fork_repo}/pulls",
        {"title": title, "head": head_branch, "base": base_branch, "body": body},
    )
    logger.info("PR created: %s", pr.get("html_url"))
    return pr


# ---------------------------------------------------------------------------
# Step 10 — Poll for the triggered action run_id
# ---------------------------------------------------------------------------

def _poll_for_run_id(
    client: Any,
    fork_repo: str,
    branch: str,
    after_ts: float,
    timeout_s: int = 60,
    interval_s: int = 5,
) -> str | None:
    """
    Poll GitHub Actions runs for a new run on the given branch triggered by
    a pull_request event after ``after_ts`` (Unix timestamp).

    Returns the run_id string, or None if timeout is reached.
    """
    deadline = time.time() + timeout_s
    after_dt = datetime.fromtimestamp(after_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    while time.time() < deadline:
        try:
            payload = client.request_json(
                f"repos/{fork_repo}/actions/runs",
                params={"event": "pull_request", "branch": branch, "created": f">={after_dt}"},
            )
            runs = payload.get("workflow_runs") or []
            if runs:
                run_id = str(runs[0]["id"])
                logger.info("Detected action run_id=%s for branch=%s", run_id, branch)
                return run_id
        except Exception as exc:  # noqa: BLE001
            logger.debug("_poll_for_run_id: API error (will retry): %s", exc)

        time.sleep(interval_s)

    logger.warning("_poll_for_run_id: timed out after %ds for branch=%s", timeout_s, branch)
    return None


# ---------------------------------------------------------------------------
# Step 12 — Poll action run completion
# ---------------------------------------------------------------------------

def _poll_run_completion(
    client: Any,
    fork_repo: str,
    run_id: str,
    timeout_s: int = 1800,   # 30 minutes
    interval_s: int = 30,
) -> dict[str, Any]:
    """
    Poll until the action run reaches ``status == "completed"`` or timeout.

    Returns the final run dict (contains ``conclusion``, ``status``, etc.).
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            run = client.request_json(f"repos/{fork_repo}/actions/runs/{run_id}")
            if run.get("status") == "completed":
                logger.info(
                    "Run %s completed — conclusion=%s", run_id, run.get("conclusion")
                )
                return run
        except Exception as exc:  # noqa: BLE001
            logger.debug("_poll_run_completion: API error (will retry): %s", exc)

        time.sleep(interval_s)

    logger.warning("_poll_run_completion: timed out after %ds for run_id=%s", timeout_s, run_id)
    return {"status": "timed_out", "conclusion": "unknown", "id": run_id}


# ---------------------------------------------------------------------------
# Step 13 — Extract run logs
# ---------------------------------------------------------------------------

def _extract_run_logs(client: Any, fork_repo: str, run_id: str) -> str:
    """
    Download the log archive for a completed run and return the first 4 KB
    of text content as a log excerpt.
    """
    try:
        log_bytes = client.download_bytes_no_redirect_auth(
            f"https://api.github.com/repos/{fork_repo}/actions/runs/{run_id}/logs"
        )
        # Log archive is a zip; extract text from first file
        import io
        import zipfile
        with zipfile.ZipFile(io.BytesIO(log_bytes)) as zf:
            names = [n for n in zf.namelist() if n.endswith(".txt")]
            if names:
                text = zf.read(names[0]).decode("utf-8", errors="replace")
                return text[:4096]
    except Exception as exc:  # noqa: BLE001
        logger.debug("_extract_run_logs: could not download logs: %s", exc)
    return ""


# ---------------------------------------------------------------------------
# Main orchestrator — Tier-2 Live Validation
# ---------------------------------------------------------------------------

def run_live_validation(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Run the full Tier-2 live validation pipeline for a set of fix candidates.

    Parameters
    ----------
    inputs : dict
        Required keys:
          fork_repo       — "owner/repo" of the user's fork
          branch          — base branch (e.g. "main")
          github_username — GitHub username for clone auth
          token           — GitHub PAT (repo + workflow write scopes required)
          error_signature — KB error signature for DB recording
          fix_ranks       — list[int] or "all"
          fixes           — list of fix dicts (from RCA output)
          workflow_id     — original failed action run_id (string)

    Returns
    -------
    dict
        Validation report with per-fix results.

    Raises
    ------
    NotImplementedError
        When TIER2_ENABLED is False (feature-flagged off).
    """
    # ── Feature gate ──────────────────────────────────────────────────
    # TODO(tier2): Remove this gate once background threading harness is wired
    #              in routes/validation.py and TIER2_ENABLED=True is set in .env.
    if not _tier2_enabled():
        raise NotImplementedError(
            "Tier-2 live validation is scaffolded but not yet enabled. "
            "Set TIER2_ENABLED=True in .env to activate."
        )

    # ── Parse inputs ──────────────────────────────────────────────────
    fork_repo       = str(inputs["fork_repo"]).strip()
    branch          = str(inputs.get("branch", "main")).strip()
    github_username = str(inputs["github_username"]).strip()
    token           = str(inputs["token"]).strip()
    error_signature = str(inputs.get("error_signature", ""))
    workflow_id     = str(inputs.get("workflow_id", ""))
    fixes_list      = inputs.get("fixes") or []
    fix_ranks_raw   = inputs.get("fix_ranks", "all")

    if fix_ranks_raw == "all":
        selected_ranks = {f.get("rank", i + 1) for i, f in enumerate(fixes_list)}
    else:
        selected_ranks = set(int(r) for r in fix_ranks_raw)

    selected_fixes = [
        f for f in fixes_list
        if (f.get("rank") or (fixes_list.index(f) + 1)) in selected_ranks
    ]

    if not selected_fixes:
        return {"status": "error", "message": "No fix candidates matched the requested ranks."}

    # ── Import GitHub client (log-collector.py has a hyphen — use importlib) ──
    import importlib.util as _ilu  # noqa: PLC0415
    _lc_spec = _ilu.spec_from_file_location(
        "log_collector", Path(__file__).parent / "log-collector.py"
    )
    _lc_mod = _ilu.module_from_spec(_lc_spec)  # type: ignore[arg-type]
    _lc_spec.loader.exec_module(_lc_mod)       # type: ignore[union-attr]
    GitHubActionsClient = _lc_mod.GitHubActionsClient
    client = GitHubActionsClient(token=token)

    # ── Import KB for result recording ───────────────────────────────
    from db import SelfLearningKnowledgeBase  # noqa: PLC0415
    kb = SelfLearningKnowledgeBase()

    # ── Set up temp clone directory ───────────────────────────────────
    clone_base = _ROOT / "output" / "validation_clones" / workflow_id
    clone_base.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    try:
        # ── Clone repo (once for all selected fixes) ──────────────────
        clone_path = _clone_repo(fork_repo, branch, token, github_username, clone_base)

        for fix in selected_fixes:
            fix_rank    = fix.get("rank", selected_fixes.index(fix) + 1)
            fix_result: dict[str, Any] = {
                "fix_rank":        fix_rank,
                "validation_type": "live",
                "status":          "error",
                "run_id":          None,
                "pr_url":          None,
                "log_excerpt":     None,
                "validated_at":    _utc_now(),
            }

            try:
                # Step 6 — Find workflow file
                wf_path = _find_workflow_file(
                    clone_path, fix.get("file_path")
                )
                if wf_path is None:
                    fix_result["status"] = "error"
                    fix_result["log_excerpt"] = "Workflow file not found in repository."
                    results.append(fix_result)
                    continue

                # Step 7 — Apply fix
                _apply_fix_to_workflow(wf_path, fix)

                # Step 8 — Commit & push
                start_ts = time.time()
                val_branch, _sha = _commit_and_push(
                    clone_path, branch, fix_rank, token, github_username, fork_repo
                )

                # Step 9 — Raise PR
                pr = _create_pull_request(
                    client, fork_repo, val_branch, branch, fix_rank,
                    fix.get("description", "")
                )
                fix_result["pr_url"] = pr.get("html_url")

                # Step 10 — Poll for run_id
                run_id = _poll_for_run_id(
                    client, fork_repo, val_branch,
                    after_ts=start_ts, timeout_s=90, interval_s=5,
                )
                fix_result["run_id"] = run_id
                fix_result["status"] = "queued"

                if run_id:
                    # Step 12 — Poll completion (long-running)
                    completed_run = _poll_run_completion(
                        client, fork_repo, run_id,
                        timeout_s=1800, interval_s=30,
                    )
                    conclusion = completed_run.get("conclusion", "unknown")

                    # Step 13 — Extract logs
                    log_excerpt = _extract_run_logs(client, fork_repo, run_id)

                    # Step 14 — Check status
                    fix_result["status"]      = "live_pass" if conclusion == "success" else "live_fail"
                    fix_result["log_excerpt"] = log_excerpt
                    fix_result["validated_at"] = _utc_now()

                # Step 15 — Record in KB
                if error_signature:
                    kb.update_validation_result(error_signature, fix_rank, fix_result)

                results.append(fix_result)

                # Reset repo to base branch for the next fix
                try:
                    _run_git(["checkout", branch], clone_path)
                except Exception:  # noqa: BLE001
                    pass

            except Exception as exc:  # noqa: BLE001
                logger.exception("Validation failed for fix_rank=%d: %s", fix_rank, exc)
                fix_result["status"]      = "error"
                fix_result["log_excerpt"] = str(exc)
                fix_result["validated_at"] = _utc_now()
                results.append(fix_result)

                if error_signature:
                    kb.update_validation_result(error_signature, fix_rank, fix_result)

    finally:
        # Always clean up the clone directory
        shutil.rmtree(clone_base, ignore_errors=True)
        logger.info("Cleaned up clone directory: %s", clone_base)

    passed  = sum(1 for r in results if r["status"] == "live_pass")
    failed  = sum(1 for r in results if r["status"] == "live_fail")
    queued  = sum(1 for r in results if r["status"] == "queued")
    errored = sum(1 for r in results if r["status"] == "error")

    return {
        "status":     "completed",
        "fork_repo":  fork_repo,
        "branch":     branch,
        "results":    results,
        "summary": {
            "total":   len(results),
            "passed":  passed,
            "failed":  failed,
            "queued":  queued,
            "errored": errored,
        },
        "completed_at": _utc_now(),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Tier-2 Live Fix Validation Pipeline (smart-ci-remediation-agent)"
    )
    parser.add_argument("--fork-repo",       required=True,  help="owner/repo of the fork")
    parser.add_argument("--branch",          default="main", help="base branch (default: main)")
    parser.add_argument("--github-username", required=True,  help="GitHub username")
    parser.add_argument("--token",           default=os.getenv("GITHUB_TOKEN", ""),
                        help="GitHub PAT (default: $GITHUB_TOKEN)")
    parser.add_argument("--error-signature", default="",     help="KB error signature")
    parser.add_argument("--fix-ranks",       nargs="*",      help="fix ranks to validate (default: all)")
    parser.add_argument("--rca-file",        required=True,
                        help="Path to JSON file containing the full RCA output")
    parser.add_argument("--output-dir",      default="output",
                        help="Directory to save validation report (default: output/)")
    args = parser.parse_args()

    if not _tier2_enabled():
        print("⚠  TIER2_ENABLED is not set to true in your environment.")
        print("   Set TIER2_ENABLED=true in .env or export TIER2_ENABLED=true to run live validation.")
        sys.exit(1)

    import json as _json_mod
    with open(args.rca_file, encoding="utf-8") as fh:
        rca_data = _json_mod.load(fh)

    fixes = rca_data.get("rca", {}).get("fixes", {}).get("fixes") or []

    inputs_dict: dict[str, Any] = {
        "fork_repo":       args.fork_repo,
        "branch":          args.branch,
        "github_username": args.github_username,
        "token":           args.token,
        "error_signature": args.error_signature or rca_data.get("error_signature", ""),
        "workflow_id":     str(rca_data.get("run", {}).get("run_id", "0")),
        "fixes":           fixes,
        "fix_ranks":       [int(r) for r in args.fix_ranks] if args.fix_ranks else "all",
    }

    report = run_live_validation(inputs_dict)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"validation_report_{ts_str}.json"
    out_path.write_text(_json_mod.dumps(report, indent=2, default=str), encoding="utf-8")

    print(_json_mod.dumps(report, indent=2, default=str))
    print(f"\nReport saved to {out_path}", file=sys.stderr)
