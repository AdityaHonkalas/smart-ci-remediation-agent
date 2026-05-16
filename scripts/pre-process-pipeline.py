#!/usr/bin/env python3
"""Pre-process collected GitHub Actions logs for CI failure RCA.

Pipeline outputs:
- preprocessed_logs.jsonl: cleaned failure excerpts
- error_signals.jsonl: extracted error signals and weak labels
- knowledge_graph.json: run/job/error/status/type/code graph
- training_dataset.jsonl: labelled input -> output examples
- vector_store.sqlite: local vector store for semantic retrieval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from db import LocalVectorDB


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ISO_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s*")
BRACKET_TS_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\]\s*")
PLAIN_TS_RE = re.compile(r"^\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+")
GROUP_RE = re.compile(r"^(?:##\[group\]|::group::)(?P<title>.*)$")
END_GROUP_RE = re.compile(r"^(?:##\[endgroup\]|::endgroup::)\s*$")
GITHUB_COMMAND_RE = re.compile(r"^(?:##\[(?:debug|command|section|notice|warning)\]|::(?:debug|notice)\b)", re.I)
DEBUG_RE = re.compile(r"^(?:debug\b|trace\b|verbose\b|##\[debug\]|::debug\b)", re.I)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
LEADING_STEP_RE = re.compile(r"^\s*(?:Run|shell:|env:)\s+", re.I)


ERROR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("github_annotation_error", re.compile(r"::error\b.*?(?:::|message=)(?P<message>.+)$", re.I)),
    ("process_exit_code", re.compile(r"\bprocess completed with exit code (?P<exit_code>\d+)\b", re.I)),
    ("exit_status", re.compile(r"\b(?:exit status|exited with code|exit code)\s+(?P<exit_code>\d+)\b", re.I)),
    ("http_status", re.compile(r"\b(?:HTTP|status(?: code)?)\s*(?P<http_status>[45]\d{2})\b", re.I)),
    ("errno", re.compile(r"\b(?P<errno>E(?:ACCES|PERM|NOENT|CONNRESET|CONNREFUSED|TIMEDOUT|HOSTUNREACH|ADDRINUSE|PIPE|INVAL|IO|EXIST|NOTDIR|ISDIR))\b")),
    ("go_test_failure", re.compile(r"^(?:--- FAIL:|FAIL\b|panic:|fatal error:)", re.I)),
    ("python_failure", re.compile(r"(?:Traceback \(most recent call last\):|AssertionError|pytest(?:.+)failed|FAILED\s+.+\.py)", re.I)),
    ("javascript_failure", re.compile(r"(?:npm ERR!|yarn (?:run )?v?\d*.*error|pnpm ERR!|Jest.*failed|TypeError:|ReferenceError:)", re.I)),
    ("docker_failure", re.compile(r"(?:docker: Error response from daemon|Cannot connect to the Docker daemon|denied: requested access|failed to solve:|image pull)", re.I)),
    ("kubernetes_failure", re.compile(r"(?:Error from server \((?P<k8s_reason>[^)]+)\)|CrashLoopBackOff|ImagePullBackOff|CreateContainerConfigError|CreateContainerError|OOMKilled|DeadlineExceeded|Back-off restarting failed container)", re.I)),
    ("timeout", re.compile(r"\b(?:timed out|timeout|deadline exceeded|context deadline exceeded)\b", re.I)),
    ("generic_error", re.compile(r"\b(?:ERROR|FATAL|Exception|failed|failure|cannot|unable to)\b", re.I)),
]


@dataclass(frozen=True)
class ErrorSignal:
    signal_id: str
    run_id: str
    repository: str
    workflow_name: str
    job_name: str
    file_name: str
    line_number: int
    section: str
    status: str
    error_type: str
    error_code: str
    pattern_name: str
    severity: str
    signal_line: str
    context: str
    fingerprint: str


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp_path.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True))
            fh.write("\n")
            count += 1
    return count


def decode_log(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def clean_line(line: str) -> str:
    line = line.rstrip("\r\n")
    line = ANSI_RE.sub("", line)
    line = CONTROL_CHAR_RE.sub("", line)
    line = ISO_TS_RE.sub("", line)
    line = BRACKET_TS_RE.sub("", line)
    line = PLAIN_TS_RE.sub("", line)
    line = LEADING_STEP_RE.sub("", line)
    return line.strip()


def is_noise(line: str) -> bool:
    if not line:
        return True
    if DEBUG_RE.search(line) or GITHUB_COMMAND_RE.search(line):
        return True
    lowered = line.lower()
    noise_prefixes = (
        "requested labels:",
        "job is waiting for",
        "current runner version:",
        "runner name:",
        "runner group name:",
        "operating system",
        "runner image",
        "prepare workflow directory",
        "prepare all required actions",
        "complete job name:",
        "cleanup.",
    )
    return lowered.startswith(noise_prefixes)


def section_by_line(raw_lines: list[str]) -> dict[int, str]:
    sections: dict[int, str] = {}
    current = "root"
    stack: list[str] = []

    for line_number, raw_line in enumerate(raw_lines, start=1):
        cleaned = ANSI_RE.sub("", raw_line.strip())
        group_match = GROUP_RE.match(cleaned)
        if group_match:
            title = group_match.group("title").strip() or "group"
            stack.append(current)
            current = title
        elif END_GROUP_RE.match(cleaned):
            current = stack.pop() if stack else "root"
        sections[line_number] = current

    return sections


def cleaned_lines(raw_text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = clean_line(raw_line)
        if not is_noise(line):
            lines.append((line_number, line))
    return lines


def infer_job_name(member_name: str, run_metadata: dict[str, Any]) -> str:
    stem = Path(member_name).stem
    stem = re.sub(r"^\d+[_\s-]+", "", stem).strip()
    normalized_stem = normalize_label(stem)

    for job in run_metadata.get("jobs", []):
        job_name = job.get("name") or ""
        if normalize_label(job_name) == normalized_stem:
            return job_name

    return stem or member_name


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def extract_signals(
    lines: list[tuple[int, str]],
    sections: dict[int, str],
    run_metadata: dict[str, Any],
    member_name: str,
    context_lines: int,
    max_signals: int,
) -> list[ErrorSignal]:
    signals: list[ErrorSignal] = []
    job_name = infer_job_name(member_name, run_metadata)
    status = run_metadata.get("conclusion") or run_metadata.get("status") or "unknown"

    for index, (line_number, line) in enumerate(lines):
        match_payload = first_error_match(line)
        if not match_payload:
            continue

        pattern_name, match = match_payload
        error_type = classify_error(line, pattern_name)
        error_code = extract_error_code(line, match, error_type)
        severity = classify_severity(line, error_type, error_code)
        context = context_excerpt(lines, index=index, radius=context_lines)
        fingerprint = fingerprint_error(line)
        signal_id = stable_id(
            "|".join(
                [
                    str(run_metadata.get("run_id")),
                    str(run_metadata.get("run_attempt")),
                    member_name,
                    str(line_number),
                    fingerprint,
                ]
            )
        )

        signals.append(
            ErrorSignal(
                signal_id=signal_id,
                run_id=str(run_metadata.get("run_id")),
                repository=run_metadata.get("repository", "unknown"),
                workflow_name=run_metadata.get("workflow_name") or "unknown",
                job_name=job_name,
                file_name=member_name,
                line_number=line_number,
                section=sections.get(line_number, "root"),
                status=status,
                error_type=error_type,
                error_code=error_code,
                pattern_name=pattern_name,
                severity=severity,
                signal_line=line,
                context=context,
                fingerprint=fingerprint,
            )
        )

        if len(signals) >= max_signals:
            break

    return signals


def first_error_match(line: str) -> tuple[str, re.Match[str]] | None:
    for pattern_name, pattern in ERROR_PATTERNS:
        match = pattern.search(line)
        if match:
            return pattern_name, match
    return None


def classify_error(line: str, pattern_name: str) -> str:
    lowered = line.lower()
    if pattern_name == "kubernetes_failure" or any(token in lowered for token in ("crashloopbackoff", "imagepullbackoff", "error from server", "pod ", "kubectl")):
        return "kubernetes_error"
    if pattern_name == "docker_failure" or "docker" in lowered or "container" in lowered:
        return "container_error"
    if "permission denied" in lowered or "forbidden" in lowered or "unauthorized" in lowered or "eacces" in lowered or "eperm" in lowered:
        return "permission_error"
    if "timed out" in lowered or "timeout" in lowered or "deadline exceeded" in lowered or "etimedout" in lowered:
        return "timeout"
    if any(token in lowered for token in ("connection refused", "connection reset", "temporary failure", "tls handshake", "no route to host", "could not resolve", "econnreset", "econnrefused")):
        return "network_error"
    if any(token in lowered for token in ("oomkilled", "out of memory", "no space left", "disk quota", "cannot allocate memory")):
        return "resource_error"
    if pattern_name in {"go_test_failure", "python_failure"} or any(token in lowered for token in ("assertionerror", "--- fail:", "test failed", "failed tests", "expected", "actual")):
        return "test_failure"
    if pattern_name == "javascript_failure" or any(token in lowered for token in ("module not found", "cannot find module", "npm err!", "go: module", "no matching distribution")):
        return "dependency_error"
    if any(token in lowered for token in ("syntax error", "compilation failed", "build failed", "make:", "compiler")):
        return "build_error"
    if any(token in lowered for token in ("yaml", "invalid configuration", "missing required", "secret not found", "unknown flag")):
        return "configuration_error"
    if pattern_name in {"process_exit_code", "exit_status"}:
        return "process_exit"
    return "unknown_error"


def extract_error_code(line: str, match: re.Match[str], error_type: str) -> str:
    groups = match.groupdict()
    if groups.get("exit_code"):
        return f"EXIT_{groups['exit_code']}"
    if groups.get("http_status"):
        return f"HTTP_{groups['http_status']}"
    if groups.get("errno"):
        return groups["errno"]
    if groups.get("k8s_reason"):
        reason = re.sub(r"[^A-Za-z0-9]+", "_", groups["k8s_reason"]).strip("_").upper()
        return f"K8S_{reason}"

    lowered = line.lower()
    if "deadline exceeded" in lowered or "timed out" in lowered or "timeout" in lowered:
        return "TIMEOUT"
    if "oomkilled" in lowered or "out of memory" in lowered:
        return "OOM"
    if "permission denied" in lowered:
        return "PERMISSION_DENIED"
    if "connection refused" in lowered:
        return "CONNECTION_REFUSED"
    if "connection reset" in lowered:
        return "CONNECTION_RESET"
    return error_type.upper()


def classify_severity(line: str, error_type: str, error_code: str) -> str:
    lowered = line.lower()
    if any(token in lowered for token in ("fatal", "panic", "traceback", "oomkilled")):
        return "high"
    if error_code.startswith(("EXIT_", "HTTP_5")) or error_type in {"timeout", "resource_error"}:
        return "medium"
    return "low"


def context_excerpt(lines: list[tuple[int, str]], index: int, radius: int) -> str:
    start = max(index - radius, 0)
    end = min(index + radius + 1, len(lines))
    return "\n".join(f"{line_number}: {line}" for line_number, line in lines[start:end])


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def fingerprint_error(line: str) -> str:
    normalized = line.lower()
    normalized = re.sub(r"https?://\S+", "<url>", normalized)
    normalized = re.sub(r"\b[0-9a-f]{7,40}\b", "<sha>", normalized)
    normalized = re.sub(r"\b\d+\b", "<num>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return stable_id(normalized)


def process_zip(
    zip_path: Path,
    run_metadata: dict[str, Any],
    context_lines: int,
    max_signals_per_log: int,
    chunk_lines: int,
    chunk_overlap: int,
) -> tuple[list[ErrorSignal], list[dict[str, Any]]]:
    signals: list[ErrorSignal] = []
    documents: list[dict[str, Any]] = []

    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        print(f"Skipping invalid zip {zip_path}: {exc}", file=sys.stderr)
        return signals, documents

    with archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            raw = archive.read(member)
            text = decode_log(raw)
            raw_lines = text.splitlines()
            parsed_sections = section_by_line(raw_lines)
            lines = cleaned_lines(text)
            documents.extend(
                build_chunk_documents(
                    lines=lines,
                    run_metadata=run_metadata,
                    member_name=member.filename,
                    chunk_lines=chunk_lines,
                    chunk_overlap=chunk_overlap,
                )
            )
            member_signals = extract_signals(
                lines=lines,
                sections=parsed_sections,
                run_metadata=run_metadata,
                member_name=member.filename,
                context_lines=context_lines,
                max_signals=max_signals_per_log,
            )
            signals.extend(member_signals)

            for signal in member_signals:
                metadata = signal_metadata(signal, run_metadata)
                metadata["doc_kind"] = "error_context"
                metadata["signal_id"] = signal.signal_id
                documents.append(
                    {
                        "document_id": signal.signal_id,
                        "text": signal.context,
                        "metadata": metadata,
                    }
                )

    return signals, documents


def build_chunk_documents(
    lines: list[tuple[int, str]],
    run_metadata: dict[str, Any],
    member_name: str,
    chunk_lines: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    if not lines:
        return []

    chunk_lines = max(1, chunk_lines)
    chunk_overlap = max(0, min(chunk_overlap, chunk_lines - 1))
    step = chunk_lines - chunk_overlap
    documents: list[dict[str, Any]] = []
    job_name = infer_job_name(member_name, run_metadata)

    for chunk_number, start in enumerate(range(0, len(lines), step), start=1):
        chunk = lines[start : start + chunk_lines]
        if not chunk:
            continue
        start_line = chunk[0][0]
        end_line = chunk[-1][0]
        text = "\n".join(f"{line_number}: {line}" for line_number, line in chunk)
        document_id = stable_id(
            "|".join(
                [
                    str(run_metadata.get("repository")),
                    str(run_metadata.get("run_id")),
                    str(run_metadata.get("run_attempt")),
                    member_name,
                    "chunk",
                    str(chunk_number),
                    str(start_line),
                    str(end_line),
                ]
            )
        )
        documents.append(
            {
                "document_id": document_id,
                "text": text,
                "metadata": {
                    "doc_kind": "log_chunk",
                    "repository": run_metadata.get("repository"),
                    "run_id": str(run_metadata.get("run_id")),
                    "run_attempt": run_metadata.get("run_attempt"),
                    "workflow_name": run_metadata.get("workflow_name") or "unknown",
                    "job_name": job_name,
                    "file_name": member_name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "status": run_metadata.get("conclusion") or run_metadata.get("status") or "unknown",
                    "html_url": run_metadata.get("html_url"),
                    "commit_sha": run_metadata.get("commit_sha"),
                    "branch": run_metadata.get("branch"),
                },
            }
        )

    return documents


def signal_metadata(signal: ErrorSignal, run_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "repository": signal.repository,
        "run_id": signal.run_id,
        "run_attempt": run_metadata.get("run_attempt"),
        "workflow_name": signal.workflow_name,
        "job_name": signal.job_name,
        "file_name": signal.file_name,
        "line_number": signal.line_number,
        "section": signal.section,
        "status": signal.status,
        "error_type": signal.error_type,
        "error_code": signal.error_code,
        "severity": signal.severity,
        "html_url": run_metadata.get("html_url"),
        "commit_sha": run_metadata.get("commit_sha"),
        "branch": run_metadata.get("branch"),
    }


def build_knowledge_graph(signals: list[ErrorSignal]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []

    def add_node(node_id: str, node_type: str, **properties: Any) -> None:
        nodes.setdefault(node_id, {"id": node_id, "type": node_type, "properties": {}})
        nodes[node_id]["properties"].update({k: v for k, v in properties.items() if v is not None})

    def add_edge(source: str, relation: str, target: str) -> None:
        edges.append({"source": source, "relation": relation, "target": target})

    for signal in signals:
        run_id = f"run:{signal.repository}:{signal.run_id}"
        job_id = f"job:{signal.repository}:{signal.run_id}:{stable_id(signal.job_name)}"
        signal_id = f"signal:{signal.signal_id}"
        status_id = f"status:{signal.status}"
        type_id = f"error_type:{signal.error_type}"
        code_id = f"error_code:{signal.error_code}"

        add_node(run_id, "workflow_run", repository=signal.repository, run_id=signal.run_id, workflow_name=signal.workflow_name)
        add_node(job_id, "job", name=signal.job_name, file_name=signal.file_name)
        add_node(signal_id, "error_signal", line_number=signal.line_number, section=signal.section, severity=signal.severity, fingerprint=signal.fingerprint, signal_line=signal.signal_line)
        add_node(status_id, "status", name=signal.status)
        add_node(type_id, "error_type", name=signal.error_type)
        add_node(code_id, "error_code", code=signal.error_code)

        add_edge(run_id, "has_job", job_id)
        add_edge(run_id, "has_status", status_id)
        add_edge(job_id, "emits", signal_id)
        add_edge(job_id, "has_status", status_id)
        add_edge(signal_id, "classified_as", type_id)
        add_edge(signal_id, "has_code", code_id)

    unique_edges = [dict(item) for item in {tuple(edge.items()) for edge in edges}]
    unique_edges.sort(key=lambda item: (item["source"], item["relation"], item["target"]))

    return {
        "schema_version": 1,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(unique_edges),
            "signal_count": len(signals),
        },
        "nodes": sorted(nodes.values(), key=lambda item: item["id"]),
        "edges": unique_edges,
    }


def training_example(signal: ErrorSignal, run_metadata_lookup: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    run_metadata = run_metadata_lookup.get((signal.repository, signal.run_id), {})
    input_text = "\n".join(
        [
            f"Repository: {signal.repository}",
            f"Workflow: {signal.workflow_name}",
            f"Job: {signal.job_name}",
            f"Run status: {signal.status}",
            f"Branch: {run_metadata.get('branch') or 'unknown'}",
            f"Commit: {run_metadata.get('commit_sha') or 'unknown'}",
            "Log excerpt:",
            signal.context,
        ]
    )

    return {
        "id": signal.signal_id,
        "task": "ci_failure_root_cause_analysis",
        "label_source": "heuristic_regex",
        "input": input_text,
        "output": {
            "root_cause_category": signal.error_type,
            "error_code": signal.error_code,
            "status": signal.status,
            "summary": summarize_failure(signal),
            "evidence": [signal.signal_line],
            "recommended_next_steps": recommended_steps(signal.error_type),
        },
    }


def summarize_failure(signal: ErrorSignal) -> str:
    summaries = {
        "kubernetes_error": "The failure appears to come from a Kubernetes API, pod, or container orchestration state.",
        "container_error": "The failure appears related to container runtime, image build, or image pull behavior.",
        "permission_error": "The failure appears to be caused by missing permissions or denied access.",
        "timeout": "The failure appears to be caused by an operation exceeding its time limit.",
        "network_error": "The failure appears to be caused by network connectivity or remote service availability.",
        "resource_error": "The failure appears to be caused by memory, disk, or resource exhaustion.",
        "test_failure": "The failure appears to be caused by failing tests or assertions.",
        "dependency_error": "The failure appears to be caused by missing, incompatible, or unavailable dependencies.",
        "build_error": "The failure appears to be caused by compilation or build command errors.",
        "configuration_error": "The failure appears to be caused by invalid or missing configuration.",
        "process_exit": "A workflow command exited with a non-zero status.",
    }
    return summaries.get(signal.error_type, "The failure contains an error signal that needs manual triage.")


def recommended_steps(error_type: str) -> list[str]:
    steps = {
        "kubernetes_error": [
            "Inspect kubectl events and pod/container status around the failed step.",
            "Check cluster availability, image pull status, and resource quotas.",
        ],
        "container_error": [
            "Inspect image build output and registry access.",
            "Retry image pulls and verify Docker/container runtime health.",
        ],
        "permission_error": [
            "Verify repository secrets, GitHub token scopes, and cloud/Kubernetes RBAC.",
            "Confirm the failing command has access to the referenced path or resource.",
        ],
        "timeout": [
            "Check for slow external dependencies and recent latency spikes.",
            "Increase timeout only after confirming the operation is expected to take longer.",
        ],
        "network_error": [
            "Check DNS, proxy, TLS, and remote service availability.",
            "Retry the job to determine whether the failure is transient.",
        ],
        "resource_error": [
            "Inspect runner disk, memory, and process limits.",
            "Reduce parallelism or request a larger runner for the failing job.",
        ],
        "test_failure": [
            "Open the failing test case and compare expected versus actual output.",
            "Check recent commits that touched the failing package or fixture.",
        ],
        "dependency_error": [
            "Verify lockfiles, package registries, module versions, and caches.",
            "Rebuild dependency caches if stale artifacts are suspected.",
        ],
        "build_error": [
            "Inspect compiler output immediately above the failure line.",
            "Check recent source or build configuration changes.",
        ],
        "configuration_error": [
            "Validate workflow YAML, environment variables, and referenced secrets.",
            "Compare the failing run configuration with the last successful run.",
        ],
    }
    return steps.get(error_type, ["Inspect the extracted error line and surrounding log context."])


def resolve_zip_path(data_dir: Path, run_metadata: dict[str, Any]) -> Path | None:
    zip_path = run_metadata.get("zip_path")
    if not zip_path:
        return None
    path = Path(zip_path)
    if not path.is_absolute():
        path = data_dir / path
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-process collected GitHub Actions logs.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Data directory.")
    parser.add_argument("--index", type=Path, help="Path to index.json. Defaults to data-dir/index.json.")
    parser.add_argument("--context-lines", type=int, default=8, help="Lines of context around each error signal.")
    parser.add_argument("--max-signals-per-log", type=int, default=100, help="Max error signals extracted per log file.")
    parser.add_argument("--chunk-lines", type=int, default=120, help="Cleaned log lines per retrieval chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=20, help="Overlapping lines between retrieval chunks.")
    parser.add_argument("--preprocessed-output", default="preprocessed_logs.jsonl")
    parser.add_argument("--signals-output", default="error_signals.jsonl")
    parser.add_argument("--graph-output", default="knowledge_graph.json")
    parser.add_argument("--dataset-output", default="training_dataset.jsonl")
    parser.add_argument("--vector-db", type=Path, help="SQLite vector DB path. Defaults to data-dir/vector_store.sqlite.")
    parser.add_argument("--skip-vector-db", action="store_true", help="Do not build the local vector DB.")
    parser.add_argument("--append-vectors", action="store_true", help="Append to vector DB instead of replacing gha_logs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    index_path = args.index or (data_dir / "index.json")
    vector_db_path = args.vector_db or (data_dir / "vector_store.sqlite")

    if not index_path.exists():
        print(f"Missing index: {index_path}", file=sys.stderr)
        print("Run scripts/log-collector.py first.", file=sys.stderr)
        return 1

    index = load_json(index_path)
    run_records = index.get("runs", [])
    run_lookup = {
        (str(record.get("repository")), str(record.get("run_id"))): record
        for record in run_records
    }

    all_signals: list[ErrorSignal] = []
    vector_documents: list[dict[str, Any]] = []
    skipped = 0

    for run_metadata in run_records:
        zip_path = resolve_zip_path(data_dir, run_metadata)
        if not zip_path or not zip_path.exists():
            skipped += 1
            continue

        signals, documents = process_zip(
            zip_path=zip_path,
            run_metadata=run_metadata,
            context_lines=args.context_lines,
            max_signals_per_log=args.max_signals_per_log,
            chunk_lines=args.chunk_lines,
            chunk_overlap=args.chunk_overlap,
        )
        all_signals.extend(signals)
        vector_documents.extend(documents)

    preprocessed_records = vector_documents
    signal_records = [asdict(signal) for signal in all_signals]
    training_records = [training_example(signal, run_lookup) for signal in all_signals]
    knowledge_graph = build_knowledge_graph(all_signals)

    preprocessed_count = write_jsonl(data_dir / args.preprocessed_output, preprocessed_records)
    signal_count = write_jsonl(data_dir / args.signals_output, signal_records)
    dataset_count = write_jsonl(data_dir / args.dataset_output, training_records)
    write_json(data_dir / args.graph_output, knowledge_graph)

    vector_count = 0
    if not args.skip_vector_db:
        vector_db = LocalVectorDB(vector_db_path)
        try:
            if not args.append_vectors:
                vector_db.clear_collection("gha_logs")
            vector_count = vector_db.upsert_documents(vector_documents, collection="gha_logs")
        finally:
            vector_db.close()

    print(f"Processed runs: {len(run_records)}; skipped missing zips: {skipped}")
    print(f"Error signals: {signal_count}")
    print(f"Preprocessed documents: {preprocessed_count}")
    print(f"Training examples: {dataset_count}")
    print(f"Knowledge graph: {data_dir / args.graph_output}")
    if not args.skip_vector_db:
        print(f"Vector documents indexed: {vector_count}")
        print(f"Vector DB: {vector_db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
