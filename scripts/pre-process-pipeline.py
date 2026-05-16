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
import csv
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from db import LocalVectorDB


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_KAGGLE_DATASET_SLUG = "mirzayasirabdullah07/cicd-pipeline-failure-logs-dataset-for-aiops"
DEFAULT_KAGGLE_DATASET_URL = f"https://www.kaggle.com/datasets/{DEFAULT_KAGGLE_DATASET_SLUG}"

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
FAILURE_BLOCK_END_RE = re.compile(
    r"^(?:Post job cleanup|Complete job|Finishing:|##\[endgroup\]|::endgroup::|Cleaning up|Uploading artifacts)",
    re.I,
)


FAILURE_BLOCK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("github_error_block", re.compile(r"(?:##\[error\]|::error\b)", re.I)),
    ("traceback_block", re.compile(r"Traceback \(most recent call last\):", re.I)),
    ("test_failure_block", re.compile(r"^(?:--- FAIL:|FAIL:|FAILED\s+|FAIL\b|Error Trace:)", re.I)),
    ("process_exit_block", re.compile(r"\bprocess completed with exit code \d+\b", re.I)),
    ("exception_block", re.compile(r"\b(?:Exception|Error|AssertionError|TypeError|ReferenceError):\s+", re.I)),
    ("build_failure_block", re.compile(r"(?:make: \*\*\*|compilation failed|build failed|failed to compile)", re.I)),
    ("dependency_failure_block", re.compile(r"(?:npm ERR!|pip install .* failed|could not find a version|module not found|go: .*: module)", re.I)),
    ("container_failure_block", re.compile(r"(?:failed to solve:|Dockerfile parse error|Cannot connect to the Docker daemon|ImagePullBackOff|CrashLoopBackOff)", re.I)),
    ("timeout_block", re.compile(r"\b(?:timed out|timeout|deadline exceeded|context deadline exceeded)\b", re.I)),
]


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


@dataclass(frozen=True)
class FailureBlock:
    block_id: str
    run_id: str
    repository: str
    workflow_name: str
    job_name: str
    file_name: str
    start_line: int
    end_line: int
    failure_stage: str
    failure_type: str
    error_code: str
    error_message: str
    severity: str
    matched_pattern: str
    text: str
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


def extract_failure_blocks(
    lines: list[tuple[int, str]],
    sections: dict[int, str],
    run_metadata: dict[str, Any],
    member_name: str,
    context_lines: int,
    max_blocks: int,
) -> list[FailureBlock]:
    blocks: list[FailureBlock] = []
    seen_fingerprints: set[str] = set()
    job_name = infer_job_name(member_name, run_metadata)

    for index, (line_number, line) in enumerate(lines):
        block_match = first_failure_block_match(line)
        if not block_match:
            continue

        matched_pattern, _ = block_match
        start_index = max(index - max(1, context_lines // 2), 0)
        end_index = find_failure_block_end(lines, index, context_lines)
        block_lines = lines[start_index:end_index]
        if not block_lines:
            continue

        text = "\n".join(f"{number}: {content}" for number, content in block_lines)
        fingerprint = fingerprint_error(text)
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)

        signal_match = first_error_match(line)
        error_type = classify_error(line, signal_match[0] if signal_match else matched_pattern)
        error_code = error_code_for_line(line, signal_match, error_type)
        severity = classify_severity(line, error_type, error_code)
        block_id = stable_id(
            "|".join(
                [
                    str(run_metadata.get("run_id")),
                    str(run_metadata.get("run_attempt")),
                    member_name,
                    str(block_lines[0][0]),
                    str(block_lines[-1][0]),
                    fingerprint,
                ]
            )
        )

        blocks.append(
            FailureBlock(
                block_id=block_id,
                run_id=str(run_metadata.get("run_id")),
                repository=run_metadata.get("repository", "unknown"),
                workflow_name=run_metadata.get("workflow_name") or "unknown",
                job_name=job_name,
                file_name=member_name,
                start_line=block_lines[0][0],
                end_line=block_lines[-1][0],
                failure_stage=sections.get(line_number, "root"),
                failure_type=error_type,
                error_code=error_code,
                error_message=line,
                severity=severity,
                matched_pattern=matched_pattern,
                text=text,
                fingerprint=fingerprint,
            )
        )

        if len(blocks) >= max_blocks:
            break

    return blocks


def first_failure_block_match(line: str) -> tuple[str, re.Match[str]] | None:
    for pattern_name, pattern in FAILURE_BLOCK_PATTERNS:
        match = pattern.search(line)
        if match:
            return pattern_name, match
    return None


def find_failure_block_end(
    lines: list[tuple[int, str]],
    start_index: int,
    context_lines: int,
) -> int:
    max_end = min(len(lines), start_index + max(context_lines * 3, 12))
    for index in range(start_index + 1, max_end):
        if FAILURE_BLOCK_END_RE.search(lines[index][1]):
            return index
        if index > start_index + 2 and first_failure_block_match(lines[index][1]):
            return index
    return max_end


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


def error_code_for_line(
    line: str,
    match_payload: tuple[str, re.Match[str]] | None,
    error_type: str,
) -> str:
    if match_payload:
        return extract_error_code(line, match_payload[1], error_type)

    lowered = line.lower()
    exit_code_match = re.search(r"\b(?:exit code|exit status)\s+(\d+)\b", lowered)
    if exit_code_match:
        return f"EXIT_{exit_code_match.group(1)}"
    http_status_match = re.search(r"\b([45]\d{2})\b", lowered)
    if http_status_match and any(token in lowered for token in ("http", "status", "response")):
        return f"HTTP_{http_status_match.group(1)}"
    errno_match = re.search(r"\b(E[A-Z0-9_]+)\b", line)
    if errno_match:
        return errno_match.group(1)
    return extract_error_code(line, re.match(r".*", line) or re.search(r".*", line), error_type)


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
) -> tuple[list[ErrorSignal], list[dict[str, Any]], list[FailureBlock]]:
    signals: list[ErrorSignal] = []
    documents: list[dict[str, Any]] = []
    failure_blocks: list[FailureBlock] = []

    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        print(f"Skipping invalid zip {zip_path}: {exc}", file=sys.stderr)
        return signals, documents, failure_blocks

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

            member_blocks = extract_failure_blocks(
                lines=lines,
                sections=parsed_sections,
                run_metadata=run_metadata,
                member_name=member.filename,
                context_lines=context_lines,
                max_blocks=max_signals_per_log,
            )
            failure_blocks.extend(member_blocks)

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

            for block in member_blocks:
                metadata = failure_block_metadata(block, run_metadata)
                documents.append(
                    {
                        "document_id": block.block_id,
                        "text": block.text,
                        "metadata": metadata,
                    }
                )

    return signals, documents, failure_blocks


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


def failure_block_metadata(block: FailureBlock, run_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_kind": "failure_block",
        "repository": block.repository,
        "run_id": block.run_id,
        "run_attempt": run_metadata.get("run_attempt"),
        "workflow_name": block.workflow_name,
        "job_name": block.job_name,
        "file_name": block.file_name,
        "start_line": block.start_line,
        "end_line": block.end_line,
        "failure_stage": block.failure_stage,
        "failure_type": block.failure_type,
        "error_type": block.failure_type,
        "error_code": block.error_code,
        "severity": block.severity,
        "matched_pattern": block.matched_pattern,
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
        stage_id = f"failure_stage:{stable_id(signal.section)}"

        add_node(run_id, "workflow_run", repository=signal.repository, run_id=signal.run_id, workflow_name=signal.workflow_name)
        add_node(job_id, "job", name=signal.job_name, file_name=signal.file_name)
        add_node(signal_id, "error_signal", line_number=signal.line_number, section=signal.section, severity=signal.severity, fingerprint=signal.fingerprint, signal_line=signal.signal_line)
        add_node(status_id, "status", name=signal.status)
        add_node(type_id, "error_type", name=signal.error_type)
        add_node(code_id, "error_code", code=signal.error_code)
        add_node(stage_id, "failure_stage", name=signal.section)

        add_edge(run_id, "has_job", job_id)
        add_edge(run_id, "has_status", status_id)
        add_edge(job_id, "emits", signal_id)
        add_edge(job_id, "has_status", status_id)
        add_edge(signal_id, "classified_as", type_id)
        add_edge(signal_id, "has_code", code_id)
        add_edge(signal_id, "observed_in_stage", stage_id)

    unique_edges = [dict(item) for item in {tuple(edge.items()) for edge in edges}]
    unique_edges.sort(key=lambda item: (item["source"], item["relation"], item["target"]))

    return {
        "schema_version": 1,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(unique_edges),
            "signal_count": len(signals),
        },
        "filter_terms": {
            "error_codes": sorted({signal.error_code for signal in signals if signal.error_code}),
            "error_types": sorted({signal.error_type for signal in signals if signal.error_type}),
            "failure_stages": sorted({signal.section for signal in signals if signal.section}),
            "failure_block_patterns": [name for name, _ in FAILURE_BLOCK_PATTERNS],
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
        "failure_stage": signal.section,
        "failure_type": signal.error_type,
        "error_code": signal.error_code,
        "error_message": signal.signal_line,
        "severity": signal.severity,
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


KAGGLE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "failure_stage": ("failure_stage", "stage_name", "stage", "pipeline_stage", "job_stage", "step_name"),
    "failure_type": ("failure_type", "error_type", "root_cause", "root_cause_category", "category", "label"),
    "error_code": ("error_code", "exit_code", "status_code", "http_status", "error_id", "code"),
    "error_message": ("error_message", "message", "log_message", "log", "raw_log", "details", "failure_message"),
    "severity": ("severity", "level", "log_level", "priority"),
    "pipeline_id": ("pipeline_id", "build_id", "run_id", "workflow_run_id", "execution_id"),
    "job_name": ("job_name", "job", "task_name", "task", "workflow_name"),
    "status": ("status", "conclusion", "result", "outcome"),
    "repository": ("repository", "repo", "project", "service", "application"),
}


def canonical_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def first_present(row: dict[str, Any], aliases: tuple[str, ...], default: str = "") -> str:
    canonical_row = {canonical_column(str(key)): value for key, value in row.items()}
    for alias in aliases:
        value = canonical_row.get(canonical_column(alias))
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def normalize_error_label(value: str, default: str = "unknown_error") -> str:
    value = value.strip().lower()
    if not value:
        return default
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    if not value.endswith(("error", "failure", "timeout")) and value in {
        "permission",
        "network",
        "dependency",
        "configuration",
        "resource",
        "container",
        "kubernetes",
        "build",
        "test",
    }:
        value = f"{value}_error" if value != "test" else "test_failure"
    return value or default


def normalize_error_code(value: str, message: str, error_type: str) -> str:
    value = value.strip()
    if not value:
        return error_code_for_line(message, first_error_match(message), error_type)
    if value.isdigit():
        if any(token in message.lower() for token in ("http", "status", "response")):
            return f"HTTP_{value}"
        return f"EXIT_{value}"
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()


def normalize_severity(value: str, message: str, error_type: str, error_code: str) -> str:
    value = value.strip().lower()
    if value in {"critical", "fatal"}:
        return "high"
    if value in {"high", "medium", "low"}:
        return value
    if value in {"warn", "warning"}:
        return "medium"
    return classify_severity(message, error_type, error_code)


def iter_dataset_files(dataset_path: Path) -> Iterable[Path]:
    if dataset_path.is_file():
        yield dataset_path
        return

    for suffix in ("*.csv", "*.jsonl", "*.ndjson", "*.json"):
        yield from sorted(dataset_path.rglob(suffix))


def read_dataset_file(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            yield from csv.DictReader(fh)
        return

    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        yield payload
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
        elif isinstance(payload, dict):
            records = payload.get("records") or payload.get("data") or payload.get("rows")
            if isinstance(records, list):
                for item in records:
                    if isinstance(item, dict):
                        yield item
            else:
                yield payload


def kaggle_record_to_signal(row: dict[str, Any], source_file: Path, row_number: int) -> ErrorSignal | None:
    message = clean_line(first_present(row, KAGGLE_COLUMN_ALIASES["error_message"]))
    if not message:
        return None

    supplied_type = first_present(row, KAGGLE_COLUMN_ALIASES["failure_type"])
    inferred_type = classify_error(message, first_error_match(message)[0] if first_error_match(message) else "generic_error")
    error_type = normalize_error_label(supplied_type, default=inferred_type)
    error_code = normalize_error_code(first_present(row, KAGGLE_COLUMN_ALIASES["error_code"]), message, error_type)
    severity = normalize_severity(first_present(row, KAGGLE_COLUMN_ALIASES["severity"]), message, error_type, error_code)
    stage = first_present(row, KAGGLE_COLUMN_ALIASES["failure_stage"], default="unknown_stage")
    pipeline_id = first_present(row, KAGGLE_COLUMN_ALIASES["pipeline_id"], default=f"row-{row_number}")
    job_name = first_present(row, KAGGLE_COLUMN_ALIASES["job_name"], default=stage)
    status = first_present(row, KAGGLE_COLUMN_ALIASES["status"], default="failure")
    repository = first_present(
        row,
        KAGGLE_COLUMN_ALIASES["repository"],
        default=f"kaggle/{DEFAULT_KAGGLE_DATASET_SLUG}",
    )
    fingerprint = fingerprint_error(message)
    signal_id = stable_id("|".join([str(source_file), str(row_number), pipeline_id, fingerprint]))

    return ErrorSignal(
        signal_id=signal_id,
        run_id=pipeline_id,
        repository=repository,
        workflow_name="kaggle_cicd_failure_dataset",
        job_name=job_name or "unknown_job",
        file_name=source_file.name,
        line_number=row_number,
        section=stage or "unknown_stage",
        status=status or "failure",
        error_type=error_type,
        error_code=error_code,
        pattern_name="kaggle_dataset",
        severity=severity,
        signal_line=message,
        context=message,
        fingerprint=fingerprint,
    )


def preprocess_kaggle_dataset(dataset_path: Path, max_records: int = 0) -> list[ErrorSignal]:
    signals: list[ErrorSignal] = []
    for file_path in iter_dataset_files(dataset_path):
        for row_number, row in enumerate(read_dataset_file(file_path), start=1):
            signal = kaggle_record_to_signal(row, file_path, row_number)
            if signal:
                signals.append(signal)
            if max_records and len(signals) >= max_records:
                return signals
    return signals


def download_kaggle_dataset(dataset_slug: str, data_dir: Path) -> Path:
    target_dir = data_dir / "kaggle" / dataset_slug.split("/")[-1]
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub  # type: ignore[import-not-found]

        return Path(kagglehub.dataset_download(dataset_slug))
    except ImportError:
        command = [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_slug,
            "-p",
            str(target_dir),
            "--unzip",
        ]
        subprocess.run(command, check=True)
        return target_dir


def resolve_kaggle_dataset_path(args: argparse.Namespace, data_dir: Path) -> Path | None:
    if args.kaggle_dataset_path:
        return args.kaggle_dataset_path.resolve()

    default_path = data_dir / "kaggle" / args.kaggle_dataset_slug.split("/")[-1]
    if default_path.exists():
        return default_path

    if args.download_kaggle:
        return download_kaggle_dataset(args.kaggle_dataset_slug, data_dir)

    return None


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
    parser.add_argument("--blocks-output", default="failure_blocks.jsonl")
    parser.add_argument("--graph-output", default="knowledge_graph.json")
    parser.add_argument("--dataset-output", default="training_dataset.jsonl")
    parser.add_argument("--vector-db", type=Path, help="SQLite vector DB path. Defaults to data-dir/vector_store.sqlite.")
    parser.add_argument("--skip-vector-db", action="store_true", help="Do not build the local vector DB.")
    parser.add_argument("--append-vectors", action="store_true", help="Append to vector DB instead of replacing gha_logs.")
    parser.add_argument(
        "--kaggle-dataset-path",
        type=Path,
        help="Local file or directory for the Kaggle CI/CD failure dataset.",
    )
    parser.add_argument(
        "--kaggle-dataset-slug",
        default=DEFAULT_KAGGLE_DATASET_SLUG,
        help=f"Kaggle dataset slug. Defaults to {DEFAULT_KAGGLE_DATASET_SLUG}.",
    )
    parser.add_argument(
        "--download-kaggle",
        action="store_true",
        help="Download the Kaggle dataset with kagglehub or the Kaggle CLI before preprocessing.",
    )
    parser.add_argument(
        "--max-kaggle-records",
        type=int,
        default=0,
        help="Maximum Kaggle records to import. Use 0 for all records.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    index_path = args.index or (data_dir / "index.json")
    vector_db_path = args.vector_db or (data_dir / "vector_store.sqlite")
    kaggle_path = resolve_kaggle_dataset_path(args, data_dir)

    if not index_path.exists() and not kaggle_path:
        print(f"Missing index: {index_path}", file=sys.stderr)
        print("Run scripts/log-collector.py first or pass --kaggle-dataset-path / --download-kaggle.", file=sys.stderr)
        return 1

    index = load_json(index_path) if index_path.exists() else {"runs": []}
    run_records = index.get("runs", [])
    run_lookup = {
        (str(record.get("repository")), str(record.get("run_id"))): record
        for record in run_records
    }

    all_signals: list[ErrorSignal] = []
    all_blocks: list[FailureBlock] = []
    vector_documents: list[dict[str, Any]] = []
    skipped = 0

    for run_metadata in run_records:
        zip_path = resolve_zip_path(data_dir, run_metadata)
        if not zip_path or not zip_path.exists():
            skipped += 1
            continue

        signals, documents, failure_blocks = process_zip(
            zip_path=zip_path,
            run_metadata=run_metadata,
            context_lines=args.context_lines,
            max_signals_per_log=args.max_signals_per_log,
            chunk_lines=args.chunk_lines,
            chunk_overlap=args.chunk_overlap,
        )
        all_signals.extend(signals)
        all_blocks.extend(failure_blocks)
        vector_documents.extend(documents)

    if kaggle_path:
        if not kaggle_path.exists():
            print(f"Kaggle dataset path does not exist: {kaggle_path}", file=sys.stderr)
        else:
            kaggle_signals = preprocess_kaggle_dataset(kaggle_path, max_records=args.max_kaggle_records)
            all_signals.extend(kaggle_signals)
            for signal in kaggle_signals:
                metadata = signal_metadata(signal, {})
                metadata["doc_kind"] = "kaggle_error_record"
                metadata["dataset_url"] = DEFAULT_KAGGLE_DATASET_URL
                vector_documents.append(
                    {
                        "document_id": f"kaggle_{signal.signal_id}",
                        "text": signal.context,
                        "metadata": metadata,
                    }
                )

    preprocessed_records = vector_documents
    signal_records = [asdict(signal) for signal in all_signals]
    block_records = [asdict(block) for block in all_blocks]
    training_records = [training_example(signal, run_lookup) for signal in all_signals]
    knowledge_graph = build_knowledge_graph(all_signals)

    preprocessed_count = write_jsonl(data_dir / args.preprocessed_output, preprocessed_records)
    signal_count = write_jsonl(data_dir / args.signals_output, signal_records)
    block_count = write_jsonl(data_dir / args.blocks_output, block_records)
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
    if kaggle_path:
        print(f"Kaggle dataset: {kaggle_path}")
    print(f"Error signals: {signal_count}")
    print(f"Failure blocks: {block_count}")
    print(f"Preprocessed documents: {preprocessed_count}")
    print(f"Training examples: {dataset_count}")
    print(f"Knowledge graph: {data_dir / args.graph_output}")
    if not args.skip_vector_db:
        print(f"Vector documents indexed: {vector_count}")
        print(f"Vector DB: {vector_db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
