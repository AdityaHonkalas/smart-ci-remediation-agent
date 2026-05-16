#!/usr/bin/env python3
"""Collect failed GitHub Actions workflow logs.

The collector downloads the GitHub-provided log archives for failed workflow
runs and writes an index file with enough metadata for later RCA stages.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPO = "kubernetes/kubernetes"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FAILURE_CONCLUSIONS = {"failure", "timed_out", "startup_failure"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_repo(repo: str) -> str:
    """Accept owner/repo or a GitHub URL and return owner/repo."""
    repo = repo.strip().rstrip("/")
    if repo.startswith("http://") or repo.startswith("https://"):
        parsed = urllib.parse.urlparse(repo)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError(f"Cannot infer owner/repo from URL: {repo}")
        return f"{parts[0]}/{parts[1]}"

    parts = repo.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("Repository must be an owner/repo string or GitHub URL")
    return repo


def safe_repo_name(repo: str) -> str:
    return repo.replace("/", "_").replace(".", "_")


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp_path.replace(path)


@dataclass(frozen=True)
class DownloadResult:
    zip_path: Path | None
    size_bytes: int
    error: str | None = None


class GitHubActionsClient:
    def __init__(
        self,
        token: str | None = None,
        api_url: str = "https://api.github.com",
        timeout_seconds: int = 45,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "smart-ci-remediation-agent",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def request_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self._build_url(path, params)
        request = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API error {exc.code} for {url}: {body}") from exc

    def paginated_items(
        self,
        path: str,
        item_key: str,
        params: dict[str, Any] | None = None,
        max_pages: int = 10,
    ) -> Iterable[dict[str, Any]]:
        params = dict(params or {})
        per_page = min(int(params.get("per_page", 50)), 100)
        params["per_page"] = per_page

        for page in range(1, max_pages + 1):
            params["page"] = page
            payload = self.request_json(path, params=params)
            items = payload.get(item_key, [])
            if not items:
                break
            yield from items
            if len(items) < per_page:
                break

    def download_bytes(self, url: str) -> bytes:
        request = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub log download error {exc.code}: {body}") from exc

    def _build_url(self, path: str, params: dict[str, Any] | None = None) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            url = path
        else:
            url = f"{self.api_url}/{path.lstrip('/')}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        return url


def list_failed_runs(
    client: GitHubActionsClient,
    repo: str,
    limit: int,
    per_page: int,
    max_pages: int,
    branch: str | None,
    workflow: str | None,
    include_cancelled: bool,
) -> list[dict[str, Any]]:
    conclusions = set(FAILURE_CONCLUSIONS)
    if include_cancelled:
        conclusions.add("cancelled")

    if workflow:
        encoded_workflow = urllib.parse.quote(workflow, safe="")
        path = f"/repos/{repo}/actions/workflows/{encoded_workflow}/runs"
    else:
        path = f"/repos/{repo}/actions/runs"

    params: dict[str, Any] = {"status": "completed", "per_page": per_page}
    if branch:
        params["branch"] = branch

    runs: list[dict[str, Any]] = []
    for run in client.paginated_items(path, "workflow_runs", params=params, max_pages=max_pages):
        if run.get("conclusion") in conclusions:
            runs.append(run)
            if len(runs) >= limit:
                break

    return runs


def list_jobs_for_run(
    client: GitHubActionsClient,
    repo: str,
    run_id: int,
    max_pages: int,
) -> list[dict[str, Any]]:
    path = f"/repos/{repo}/actions/runs/{run_id}/jobs"
    jobs: list[dict[str, Any]] = []
    for job in client.paginated_items(
        path,
        "jobs",
        params={"per_page": 100, "filter": "latest"},
        max_pages=max_pages,
    ):
        jobs.append(
            {
                "id": job.get("id"),
                "name": job.get("name"),
                "status": job.get("status"),
                "conclusion": job.get("conclusion"),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "html_url": job.get("html_url"),
                "steps": [
                    {
                        "name": step.get("name"),
                        "status": step.get("status"),
                        "conclusion": step.get("conclusion"),
                        "number": step.get("number"),
                        "started_at": step.get("started_at"),
                        "completed_at": step.get("completed_at"),
                    }
                    for step in job.get("steps", [])
                    if step.get("conclusion") not in (None, "success", "skipped")
                ],
            }
        )
    return jobs


def download_run_zip(
    client: GitHubActionsClient,
    repo: str,
    run: dict[str, Any],
    output_dir: Path,
    overwrite: bool,
) -> DownloadResult:
    run_id = run["id"]
    attempt = run.get("run_attempt", 1)
    filename = f"{safe_repo_name(repo)}_run-{run_id}_attempt-{attempt}.zip"
    zip_path = output_dir / filename

    if zip_path.exists() and not overwrite:
        return DownloadResult(zip_path=zip_path, size_bytes=zip_path.stat().st_size)

    try:
        archive_bytes = client.download_bytes(run["logs_url"])
    except Exception as exc:  # noqa: BLE001 - store the per-run failure in index.json.
        return DownloadResult(zip_path=None, size_bytes=0, error=str(exc))

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path.write_bytes(archive_bytes)
    return DownloadResult(zip_path=zip_path, size_bytes=len(archive_bytes))


def compact_run_metadata(
    repo: str,
    run: dict[str, Any],
    jobs: list[dict[str, Any]],
    download: DownloadResult,
    data_dir: Path,
) -> dict[str, Any]:
    head_commit = run.get("head_commit") or {}
    zip_path = None
    if download.zip_path:
        zip_path = download.zip_path.resolve().relative_to(data_dir.resolve()).as_posix()

    return {
        "repository": repo,
        "run_id": run.get("id"),
        "run_attempt": run.get("run_attempt"),
        "run_number": run.get("run_number"),
        "workflow_id": run.get("workflow_id"),
        "workflow_name": run.get("name"),
        "event": run.get("event"),
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "branch": run.get("head_branch"),
        "commit_sha": run.get("head_sha"),
        "commit_message": head_commit.get("message"),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "html_url": run.get("html_url"),
        "logs_url": run.get("logs_url"),
        "jobs_url": run.get("jobs_url"),
        "jobs": jobs,
        "zip_path": zip_path,
        "zip_size_bytes": download.size_bytes,
        "download_error": download.error,
        "collected_at": utc_now(),
    }


def upsert_index(index_path: Path, repo: str, run_records: list[dict[str, Any]]) -> None:
    index = load_json(
        index_path,
        {
            "schema_version": 1,
            "description": "GitHub Actions failed workflow log archive index.",
            "repositories": {},
            "runs": [],
        },
    )

    existing: dict[tuple[str, int, int], dict[str, Any]] = {}
    for record in index.get("runs", []):
        key = (
            record.get("repository", ""),
            int(record.get("run_id") or 0),
            int(record.get("run_attempt") or 0),
        )
        existing[key] = record

    for record in run_records:
        key = (
            record["repository"],
            int(record["run_id"] or 0),
            int(record.get("run_attempt") or 0),
        )
        existing[key] = record

    runs = sorted(
        existing.values(),
        key=lambda item: (item.get("created_at") or "", item.get("run_id") or 0),
        reverse=True,
    )
    index["runs"] = runs
    index["updated_at"] = utc_now()
    index["repositories"][repo] = {
        "last_collected_at": utc_now(),
        "run_count": sum(1 for item in runs if item.get("repository") == repo),
    }

    write_json(index_path, index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect failed GitHub Actions logs.")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="GitHub owner/repo or repository URL. Defaults to kubernetes/kubernetes.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Maximum failed runs to download.")
    parser.add_argument("--per-page", type=int, default=50, help="GitHub API page size.")
    parser.add_argument("--max-pages", type=int, default=10, help="Maximum workflow run pages to scan.")
    parser.add_argument("--job-pages", type=int, default=5, help="Maximum job pages to scan per run.")
    parser.add_argument("--branch", help="Optional branch filter.")
    parser.add_argument(
        "--workflow",
        help="Optional workflow id, file name, or workflow name accepted by the GitHub API.",
    )
    parser.add_argument(
        "--include-cancelled",
        action="store_true",
        help="Also collect cancelled runs as failure-like examples.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory where logs and index.json are stored.",
    )
    parser.add_argument("--index-name", default="index.json", help="Index file name under data-dir.")
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN"), help="GitHub token.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing zip files.")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Delay between run downloads to be gentle with the API.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = normalize_repo(args.repo)
    data_dir = args.data_dir.resolve()
    log_dir = data_dir / "logs" / safe_repo_name(repo)
    index_path = data_dir / args.index_name

    client = GitHubActionsClient(token=args.token)
    print(f"Scanning failed GitHub Actions runs for {repo}...")
    runs = list_failed_runs(
        client=client,
        repo=repo,
        limit=args.limit,
        per_page=args.per_page,
        max_pages=args.max_pages,
        branch=args.branch,
        workflow=args.workflow,
        include_cancelled=args.include_cancelled,
    )

    if not runs:
        print("No failed workflow runs found with the supplied filters.")
        return 0

    records: list[dict[str, Any]] = []
    for ordinal, run in enumerate(runs, start=1):
        run_id = run["id"]
        workflow_name = run.get("name") or "<unknown workflow>"
        print(f"[{ordinal}/{len(runs)}] Downloading run {run_id} ({workflow_name})")

        jobs = list_jobs_for_run(client, repo=repo, run_id=run_id, max_pages=args.job_pages)
        download = download_run_zip(client, repo=repo, run=run, output_dir=log_dir, overwrite=args.overwrite)
        records.append(compact_run_metadata(repo, run, jobs, download, data_dir))

        if download.error:
            print(f"  log download failed: {download.error}", file=sys.stderr)
        elif download.zip_path:
            print(f"  stored {download.zip_path} ({download.size_bytes} bytes)")

        if args.sleep_seconds and ordinal < len(runs):
            time.sleep(args.sleep_seconds)

    upsert_index(index_path, repo=repo, run_records=records)
    downloaded = sum(1 for record in records if record.get("zip_path"))
    print(f"Indexed {len(records)} runs, {downloaded} zip archives available.")
    print(f"Metadata index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
