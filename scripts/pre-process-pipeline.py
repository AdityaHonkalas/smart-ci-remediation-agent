#!/usr/bin/env python3
"""Pre-process collected GitHub Actions logs for CI failure RCA.

Redesigned pipeline:
  Step 1  Log Normalization          — clean ANSI, timestamps, control chars
  Step 2  Workflow-aware Segmentation — segment log into workflow → job → step → command
                                        hierarchy using marker patterns from config
  Step 3  Failure Event Construction — group related error lines, stack traces, and
                                        metadata into FailureEvents using start/end markers;
                                        context preserved from failure-start to failure-end
                                        (no fixed window truncation)
  Step 4  Hierarchical Severity Score — score blocks via curated failure taxonomy tiers
                                        (critical/high/medium/low) rather than flat weights
  Step 5  Root-cause Validation      — filter terminal status noise (exit codes, "job
                                        failed" wrappers) so only causal errors remain
  Step 6  Context Enrichment         — CI/repo metadata, language, framework, impact
  Step 7  Error Signature            — SHA-256 hash + human-readable label
  Step 8  Semantic Deduplication     — compare new errors against ChromaDB KB; skip
                                        near-duplicates above similarity threshold
  Step 9  Vector Embedding           — sentence-transformers/all-MiniLM-L6-v2 (384-dim)

Context window policy:
  Every failure event preserves the complete log context from the line where a
  failure-start marker is detected up to (and including) the line where the
  matching failure-end marker is detected.  If no end marker is found within
  the causal window defined in failure_signal_patterns.json, the block closes at
  the next unrelated section boundary.  No fixed ±N-line truncation is applied.

Pipeline outputs:
- preprocessed_logs.jsonl   : cleaned failure excerpts
- error_signals.jsonl       : extracted error signals and weak labels
- failure_blocks.jsonl      : failure block records
- knowledge_graph.json      : run/job/error/status/type/code graph
- training_dataset.jsonl    : labelled input -> output examples
- chroma_store/             : ChromaDB vector store for semantic retrieval
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import ssl
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from db import ChromaVectorDB, _Embedder as _SentenceEmbedder
from env_loader import load_dotenv

logger = logging.getLogger(__name__)


load_dotenv()


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_KAGGLE_DATASET_SLUG = "mirzayasirabdullah07/cicd-pipeline-failure-logs-dataset-for-aiops"
DEFAULT_KAGGLE_DATASET_URL = f"https://www.kaggle.com/datasets/{DEFAULT_KAGGLE_DATASET_SLUG}"
DEFAULT_HUGGINGFACE_DATASET_ID = "Snaseem2026/devops-incident-response"
DEFAULT_HUGGINGFACE_DATASET_URL = f"https://huggingface.co/datasets/{DEFAULT_HUGGINGFACE_DATASET_ID}"
DEFAULT_HUGGINGFACE_SPLITS = ("train", "validation", "test")
DEFAULT_PATTERN_CONFIG_PATH = DEFAULT_DATA_DIR / "failure_signal_patterns.json"


def compile_flags(flag_names: Iterable[str] | None) -> int:
    flags = 0
    for name in flag_names or []:
        try:
            flags |= getattr(re, str(name).upper())
        except AttributeError as exc:
            raise ValueError(f"Unsupported regex flag in {DEFAULT_PATTERN_CONFIG_PATH}: {name}") from exc
    return flags


def load_failure_pattern_config(path: Path = DEFAULT_PATTERN_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing failure pattern config: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def compile_pattern_entry(entry: dict[str, Any]) -> re.Pattern[str]:
    return re.compile(str(entry["pattern"]), compile_flags(entry.get("flags")))


def compile_named_patterns(config: dict[str, Any], key: str) -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for entry in config.get(key, []):
        patterns.append((str(entry["name"]), compile_pattern_entry(entry)))
    if not patterns:
        raise ValueError(f"No patterns configured for {key} in {DEFAULT_PATTERN_CONFIG_PATH}")
    return patterns


PATTERN_CONFIG = load_failure_pattern_config()

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

# ---------------------------------------------------------------------------
# Compiled pattern sets from failure_signal_patterns.json
# ---------------------------------------------------------------------------
FAILURE_BLOCK_END_RE = compile_pattern_entry(PATTERN_CONFIG["failure_block_end_pattern"])
FAILURE_BLOCK_PATTERNS = compile_named_patterns(PATTERN_CONFIG, "failure_block_patterns")
ERROR_PATTERNS = compile_named_patterns(PATTERN_CONFIG, "error_patterns")
NOISE_PREFIXES = tuple(str(value).lower() for value in PATTERN_CONFIG.get("noise_prefixes", []))
CLASSIFICATION_SIGNALS = {
    str(key): tuple(str(token).lower() for token in values)
    for key, values in PATTERN_CONFIG.get("classification_signals", {}).items()
}
ERROR_CODE_SIGNALS = {
    str(key): tuple(str(token).lower() for token in values)
    for key, values in PATTERN_CONFIG.get("error_code_signals", {}).items()
}
SEVERITY_SIGNALS = PATTERN_CONFIG.get("severity_signals", {})

# Failure start/end markers (new – used by FailureEventBuilder)
_FAILURE_START_MARKERS: list[tuple[str, re.Pattern[str], str, bool]] = []
for _entry in PATTERN_CONFIG.get("failure_block_start_markers", []):
    _FAILURE_START_MARKERS.append((
        str(_entry["name"]),
        compile_pattern_entry(_entry),
        str(_entry.get("severity_tier", "medium")),
        bool(_entry.get("is_terminal", False)),
    ))

_FAILURE_END_MARKERS: list[tuple[str, re.Pattern[str], bool, bool]] = []
for _entry in PATTERN_CONFIG.get("failure_block_end_markers", []):
    _FAILURE_END_MARKERS.append((
        str(_entry["name"]),
        compile_pattern_entry(_entry),
        bool(_entry.get("terminates_all", False)),
        bool(_entry.get("is_cascade_noise", False)),
    ))

# Terminal noise patterns – lines that describe cascade/wrap, not root cause
_TERMINAL_NOISE_PATTERNS: list[re.Pattern[str]] = [
    compile_pattern_entry(e) for e in PATTERN_CONFIG.get("terminal_noise_patterns", [])
]

# Workflow segmentation markers
_WF_MARKERS = PATTERN_CONFIG.get("workflow_step_markers", {})
_STEP_START_RE = compile_pattern_entry(_WF_MARKERS["step_start"])   if "step_start"  in _WF_MARKERS else None
_STEP_END_RE   = compile_pattern_entry(_WF_MARKERS["step_end"])     if "step_end"    in _WF_MARKERS else None
_JOB_START_RE  = compile_pattern_entry(_WF_MARKERS["job_start"])    if "job_start"   in _WF_MARKERS else None
_CMD_RUN_RE    = compile_pattern_entry(_WF_MARKERS["command_run"])  if "command_run" in _WF_MARKERS else None

# Causal link patterns
_CAUSAL_LINK_PATTERNS: list[tuple[str, re.Pattern[str]]] = []
for _cname, _centry in PATTERN_CONFIG.get("causal_link_patterns", {}).items():
    _CAUSAL_LINK_PATTERNS.append((_cname, compile_pattern_entry(_centry)))

# Stack trace patterns per language
_STACK_TRACE_LANG_RES: dict[str, re.Pattern[str]] = {
    lang: re.compile(spec["frame_pattern"], compile_flags(spec.get("flags")))
    for lang, spec in PATTERN_CONFIG.get("stack_trace_patterns", {}).items()
}

# Failure taxonomy – tiers drive hierarchical severity scoring
_FAILURE_TAXONOMY: dict[str, Any] = PATTERN_CONFIG.get("failure_taxonomy", {})
_TAXONOMY_TIERS: dict[str, dict[str, Any]] = _FAILURE_TAXONOMY.get("tiers", {})
_ROOT_CAUSE_TYPES_TAXONOMY: frozenset[str] = frozenset(_FAILURE_TAXONOMY.get("root_cause_types", []))
_CASCADING_TYPES_TAXONOMY: frozenset[str]  = frozenset(_FAILURE_TAXONOMY.get("cascading_types", []))
_TERMINAL_NOISE_TYPES: frozenset[str]      = frozenset(_FAILURE_TAXONOMY.get("terminal_noise_types", []))

# Semantic similarity threshold for deduplication
_SEMANTIC_SIM_THRESHOLD: float = float(PATTERN_CONFIG.get("semantic_similarity_threshold", 0.85))
# Max causal window (lines) for failure block boundary detection
_CAUSAL_WINDOW_LINES: int = int(PATTERN_CONFIG.get("causal_window_lines", 15))
# Max stack trace depth
_MAX_STACK_DEPTH: int = int(PATTERN_CONFIG.get("max_stack_trace_depth", 50))

DATASET_FILE_SUFFIXES = {".csv", ".jsonl", ".ndjson", ".json"}


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
    root_cause: str = ""
    remediation_steps: tuple[str, ...] = field(default_factory=tuple)
    source_url: str = ""


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
    with path.open("r", encoding="utf-8-sig") as fh:
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
    line = line.lstrip("\ufeff")
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
    return lowered.startswith(NOISE_PREFIXES)


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


def has_signal(line: str, signal_name: str) -> bool:
    return any(token in line for token in CLASSIFICATION_SIGNALS.get(signal_name, ()))


def classify_error(line: str, pattern_name: str) -> str:
    lowered = line.lower()
    if pattern_name == "kubernetes_failure" or has_signal(lowered, "kubernetes_error"):
        return "kubernetes_error"
    if pattern_name == "docker_failure" or has_signal(lowered, "container_error"):
        return "container_error"
    if has_signal(lowered, "permission_error"):
        return "permission_error"
    if has_signal(lowered, "timeout"):
        return "timeout"
    if has_signal(lowered, "network_error"):
        return "network_error"
    if has_signal(lowered, "resource_error"):
        return "resource_error"
    if pattern_name in {"go_test_failure", "python_failure"} or has_signal(lowered, "test_failure"):
        return "test_failure"
    if pattern_name == "javascript_failure" or has_signal(lowered, "dependency_error"):
        return "dependency_error"
    if has_signal(lowered, "build_error"):
        return "build_error"
    if has_signal(lowered, "configuration_error"):
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
    for error_code, tokens in ERROR_CODE_SIGNALS.items():
        if any(token in lowered for token in tokens):
            return error_code
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
    high_tokens = tuple(str(token).lower() for token in SEVERITY_SIGNALS.get("high_tokens", []))
    medium_types = set(str(token) for token in SEVERITY_SIGNALS.get("medium_error_types", []))
    medium_prefixes = tuple(str(token) for token in SEVERITY_SIGNALS.get("medium_error_code_prefixes", []))
    if any(token in lowered for token in high_tokens):
        return "high"
    if error_code.startswith(medium_prefixes) or error_type in medium_types:
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
        "source_url": signal.source_url,
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
            "error_patterns": [name for name, _ in ERROR_PATTERNS],
            "failure_block_patterns": [name for name, _ in FAILURE_BLOCK_PATTERNS],
            "pattern_config": str(DEFAULT_PATTERN_CONFIG_PATH),
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
            f"Source: {signal.source_url or run_metadata.get('html_url') or 'unknown'}",
            "Log excerpt:",
            signal.context,
        ]
    )
    remediation_steps = list(signal.remediation_steps) if signal.remediation_steps else recommended_steps(signal.error_type)

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
            "summary": signal.root_cause or summarize_failure(signal),
            "evidence": [signal.signal_line],
            "recommended_next_steps": remediation_steps,
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
            return text_value(value).strip()
    return default


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(text_value(item) for item in value.values() if text_value(item))
    if isinstance(value, list):
        return " | ".join(text_value(item) for item in value if text_value(item))
    return str(value)


def list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [text_value(item).strip() for item in value if text_value(item).strip()]
    text = text_value(value).strip()
    return [text] if text else []


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


def is_huggingface_devops_incident(row: dict[str, Any]) -> bool:
    canonical_keys = {canonical_column(str(key)) for key in row}
    return {"incident_id", "root_cause", "symptoms"}.issubset(canonical_keys)


def huggingface_record_to_signal(row: dict[str, Any], source_file: Path, row_number: int) -> ErrorSignal | None:
    symptoms = list_value(row.get("symptoms"))
    title = text_value(row.get("title")).strip()
    description = text_value(row.get("description")).strip()
    root_cause = text_value(row.get("root_cause")).strip()
    resolution_steps = tuple(list_value(row.get("resolution_steps")))
    incident_id = text_value(row.get("incident_id")).strip() or f"row-{row_number}"
    severity_value = text_value(row.get("severity")).strip()
    category = text_value(row.get("category")).strip()
    environment = text_value(row.get("environment")).strip()
    tags = list_value(row.get("tags"))
    related_technologies = list_value(row.get("related_technologies"))

    message = symptoms[0] if symptoms else description or title
    if not message:
        return None

    conversation = [
        text_value(item.get("message") if isinstance(item, dict) else item).strip()
        for item in row.get("troubleshooting_conversation", [])
        if text_value(item.get("message") if isinstance(item, dict) else item).strip()
    ]
    context_parts = [
        f"Incident: {title}" if title else "",
        f"Environment: {environment}" if environment else "",
        f"Description: {description}" if description else "",
        "Symptoms: " + " | ".join(symptoms) if symptoms else "",
        "Troubleshooting: " + " | ".join(conversation[:8]) if conversation else "",
        f"Root cause: {root_cause}" if root_cause else "",
        "Resolution steps: " + " | ".join(resolution_steps) if resolution_steps else "",
        "Tags: " + ", ".join(tags) if tags else "",
        "Technologies: " + ", ".join(related_technologies) if related_technologies else "",
    ]
    context = "\n".join(part for part in context_parts if part)
    match_payload = first_error_match(context) or first_error_match(message)
    pattern_name = match_payload[0] if match_payload else "generic_error"
    inferred_type = classify_error(context, pattern_name)
    supplied_type = category if normalize_label(category) not in {"cicd", "cicdincident"} else ""
    error_type = normalize_error_label(supplied_type, default=inferred_type)
    if error_type in {"unknown_error", "cicd"}:
        error_type = inferred_type if inferred_type != "unknown_error" else "process_exit"
    error_code = normalize_error_code("", context, error_type)
    severity = normalize_severity(severity_value, context, error_type, error_code)
    stage = environment or category or "devops_incident"
    fingerprint = fingerprint_error(context or message)
    signal_id = stable_id("|".join([DEFAULT_HUGGINGFACE_DATASET_ID, str(source_file), incident_id, fingerprint]))

    return ErrorSignal(
        signal_id=signal_id,
        run_id=incident_id,
        repository=f"huggingface/{DEFAULT_HUGGINGFACE_DATASET_ID}",
        workflow_name="huggingface_devops_incident_response",
        job_name=category or "devops",
        file_name=source_file.name,
        line_number=row_number,
        section=stage,
        status="failure",
        error_type=error_type,
        error_code=error_code,
        pattern_name="huggingface_devops_incident",
        severity=severity,
        signal_line=clean_line(message),
        context=context,
        fingerprint=fingerprint,
        root_cause=root_cause,
        remediation_steps=resolution_steps,
        source_url=DEFAULT_HUGGINGFACE_DATASET_URL,
    )


def iter_dataset_files(dataset_path: Path) -> Iterable[Path]:
    if dataset_path.is_file():
        if dataset_path.suffix.lower() in DATASET_FILE_SUFFIXES:
            yield dataset_path
        return

    for suffix in ("*.csv", "*.jsonl", "*.ndjson", "*.json"):
        yield from sorted(dataset_path.rglob(suffix))


def read_csv_rows_from_text(text_stream: Any) -> Iterable[dict[str, Any]]:
    yield from csv.DictReader(text_stream)


def read_json_rows_from_text(text: str) -> Iterable[dict[str, Any]]:
    payload = json.loads(text)
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


def read_dataset_file(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            yield from read_csv_rows_from_text(fh)
        return

    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        yield payload
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as fh:
            yield from read_json_rows_from_text(fh.read())
        return


def kaggle_record_to_signal(row: dict[str, Any], source_file: Path, row_number: int) -> ErrorSignal | None:
    if is_huggingface_devops_incident(row):
        return huggingface_record_to_signal(row, source_file, row_number)

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
        source_url=DEFAULT_KAGGLE_DATASET_URL,
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
    if any(path.stat().st_size > 0 for path in iter_dataset_files(target_dir)):
        return target_dir

    download_errors: list[str] = []
    try:
        import kagglehub  # type: ignore[import-not-found]

        downloaded_path = Path(kagglehub.dataset_download(dataset_slug))
        if downloaded_path.resolve() == target_dir.resolve():
            return target_dir

        for source_path in downloaded_path.rglob("*"):
            if source_path.is_dir():
                continue
            relative_path = source_path.relative_to(downloaded_path)
            destination_path = target_dir / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
        return target_dir
    except ImportError as exc:
        download_errors.append(f"kagglehub is not installed: {exc}")
    except Exception as exc:  # noqa: BLE001 - try the Kaggle CLI fallback before failing.
        download_errors.append(f"kagglehub failed: {exc}")

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
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode == 0:
        return target_dir

    cli_output = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
    download_errors.append(f"kaggle CLI failed: {cli_output or f'exit code {completed.returncode}'}")
    raise RuntimeError("Unable to download Kaggle dataset. " + " | ".join(download_errors))


def safe_dataset_name(dataset_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_id).strip("_")


def fetch_url_text(url: str) -> str:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "smart-ci-remediation-agent"})
        context = ssl._create_unverified_context() if os.getenv("ALLOW_INSECURE_DATASET_DOWNLOAD") == "1" else None
        with urllib.request.urlopen(request, timeout=60, context=context) as response:
            return response.read().decode("utf-8")
    except Exception as urllib_exc:  # noqa: BLE001 - Windows cert store fallback below.
        if os.name != "nt":
            raise RuntimeError(f"Failed to fetch {url}: {urllib_exc}") from urllib_exc

        escaped_url = url.replace("'", "''")
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "$ProgressPreference='SilentlyContinue'; "
                f"(Invoke-WebRequest -UseBasicParsing -Uri '{escaped_url}').Content"
            ),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout
        raise RuntimeError(
            f"Failed to fetch {url}: {urllib_exc}; PowerShell fallback failed: {completed.stderr.strip()}"
        ) from urllib_exc


def download_huggingface_dataset(dataset_id: str, data_dir: Path, splits: tuple[str, ...]) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", dataset_id):
        raise ValueError("Hugging Face dataset id must look like namespace/name")

    target_dir = data_dir / "huggingface" / safe_dataset_name(dataset_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / "rows.jsonl"
    if jsonl_file_has_records(output_path):
        return target_dir

    total_rows = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for split in splits:
            query = urllib.parse.urlencode(
                {
                    "dataset": dataset_id,
                    "config": "default",
                    "split": split,
                    "offset": 0,
                    "length": 100,
                }
            )
            url = f"https://datasets-server.huggingface.co/rows?{query}"
            payload = json.loads(fetch_url_text(url))
            for item in payload.get("rows", []):
                row = item.get("row")
                if isinstance(row, dict):
                    row["_hf_split"] = split
                    row["_hf_dataset_id"] = dataset_id
                    fh.write(json.dumps(row, sort_keys=True))
                    fh.write("\n")
                    total_rows += 1

    if total_rows == 0:
        raise RuntimeError(f"No rows downloaded from Hugging Face dataset {dataset_id}")
    return target_dir


def jsonl_file_has_records(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8-sig") as fh:
        return any(line.strip() for line in fh)


def resolve_kaggle_dataset_path(args: argparse.Namespace, data_dir: Path) -> Path | None:
    if args.kaggle_dataset_path:
        return args.kaggle_dataset_path.resolve()

    default_path = data_dir / "kaggle" / args.kaggle_dataset_slug.split("/")[-1]
    if any(path.stat().st_size > 0 for path in iter_dataset_files(default_path)):
        return default_path

    if args.download_kaggle:
        return download_kaggle_dataset(args.kaggle_dataset_slug, data_dir)

    return None


def resolve_huggingface_dataset_path(args: argparse.Namespace, data_dir: Path) -> Path | None:
    if args.huggingface_dataset_path:
        return args.huggingface_dataset_path.resolve()

    default_path = data_dir / "huggingface" / safe_dataset_name(args.huggingface_dataset_id)
    if any(path.stat().st_size > 0 for path in iter_dataset_files(default_path)):
        return default_path

    if args.download_huggingface:
        splits = tuple(split.strip() for split in args.huggingface_splits.split(",") if split.strip())
        return download_huggingface_dataset(args.huggingface_dataset_id, data_dir, splits)

    return None


def resolve_zip_path(data_dir: Path, run_metadata: dict[str, Any]) -> Path | None:
    zip_path = run_metadata.get("zip_path")
    if not zip_path:
        return None
    path = Path(zip_path)
    if not path.is_absolute():
        path = data_dir / path
    return path



# ===========================================================================
# Redesigned Architecture - Data Processing Pipeline Components
# ===========================================================================

# ---------------------------------------------------------------------------
# Step 2 data structures
# ---------------------------------------------------------------------------

@dataclass
class WorkflowLocation:
    """Hierarchical location within a CI workflow log."""
    workflow: str = ""      # workflow name (from run metadata)
    job: str = ""           # job name (inferred from file / group header)
    step: str = ""          # step name (from ::group:: / ##[group] marker)
    command: str = ""       # command being executed (from ##[command] / $ marker)


@dataclass
class ErrorBlock:
    """
    A single extracted failure event with complete context.

    Context policy (change #9):
      context_before / context_after are NOT limited to a fixed +-N window.
      They hold every log line from the failure-start marker up to the error
      line (before) and from the error line to the failure-end marker (after).
      This guarantees no information is truncated across a failure block.

    rca_weight -- hierarchical score [0-100] assigned by HierarchicalSeverityScorer.
    The block with the highest weight is the root cause / primary error.
    """
    error_type: str                     # e.g. 'java_exception', 'python_exception'
    error_message: str                  # Primary error message text
    error_code: str                     # Error code if present, else ''
    stack_trace: list[str]              # Stack trace lines (may be empty)
    context_before: list[str]           # All lines from block-start to error line
    context_after: list[str]            # All lines from error line to block-end
    file_path: str                      # File where error occurred ('' if unknown)
    line_number: int | None             # Log line number (None if unknown)
    timestamp: str                      # ISO-8601 UTC timestamp of extraction
    # Workflow location (change #1)
    location: WorkflowLocation = field(default_factory=WorkflowLocation)
    # Start/end marker names that bounded this block (change #8)
    start_marker: str = ""
    end_marker: str = ""
    start_line_number: int = 0          # absolute log line where block started
    end_line_number: int = 0            # absolute log line where block ended
    # Enrichment fields (Step 6)
    repository: str = ""
    language: str = ""
    framework: str = ""
    ci_environment: dict[str, Any] = field(default_factory=dict)
    dependencies: list[dict[str, str]] = field(default_factory=list)
    severity: str = "low"               # 'critical' | 'high' | 'medium' | 'low'
    impact_scope: str = "unknown"       # 'build' | 'test' | 'deployment'
    rca_weight: int = 0                 # [0-100] root-cause weight; highest = primary
    is_terminal_noise: bool = False     # True -> filtered out before RCA


@dataclass
class ErrorSignatureResult:
    """Step 7 output: deterministic error signature for deduplication."""
    hash: str             # SHA-256 hash of normalised error components
    readable: str         # e.g. 'java_exception_nullpointer_main'
    components: dict[str, str] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Step 9 output: 384-dimensional L2-normalised embeddings."""
    error_embedding: list[float]      # Shape: (384,)
    context_embedding: list[float]    # Shape: (384,)
    combined_embedding: list[float]   # Shape: (384,)


@dataclass
class ProcessedLog:
    """
    Complete output of the PreprocessingPipeline for a single CI log.
    Passed directly to the RCA Agent as structured context input.
    """
    normalized_log: str
    workflow_segments: list[dict[str, Any]]   # Step 2: segmented hierarchy
    error_blocks: list[ErrorBlock]             # Step 3-5: validated failure events
    primary_error: ErrorBlock | None           # highest rca_weight block
    error_signature: ErrorSignatureResult | None
    embeddings: EmbeddingResult | None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step 9: Text Embedder
# ---------------------------------------------------------------------------

class TextEmbedder:
    """
    Transformer-based text embedding using sentence-transformers/all-MiniLM-L6-v2.
    Produces L2-normalised 384-dimensional vectors for cosine similarity search.
    Falls back to a deterministic zero vector when the library is unavailable.
    """

    ZERO_VEC: list[float] = [0.0] * 384

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model
        self._ready = False
        try:
            self._embedder = _SentenceEmbedder.get()
            self._ready = True
        except RuntimeError:
            logger.warning(
                "sentence-transformers unavailable -- zero-vector fallback enabled. "
                "Run: pip install sentence-transformers"
            )

    def embed(self, text: str) -> list[float]:
        if not self._ready:
            return self.ZERO_VEC[:]
        return self._embedder.embed_one(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self._ready:
            return [self.ZERO_VEC[:] for _ in texts]
        return self._embedder.embed(texts)


# ---------------------------------------------------------------------------
# Step 1: Log Parser / Normaliser
# ---------------------------------------------------------------------------

class LogParser:
    """Step 1 -- Log Normalization: strip ANSI, timestamps, deduplicate."""

    def normalize(self, raw_log: str) -> str:
        lines = raw_log.splitlines()
        cleaned: list[str] = []
        prev: str | None = None
        for raw_line in lines:
            line = clean_line(raw_line)
            if is_noise(line):
                continue
            if line == prev:
                continue
            cleaned.append(line)
            prev = line
        return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Step 2: Workflow-aware Segmentation
# ---------------------------------------------------------------------------

@dataclass
class WorkflowSegment:
    """A named section of the log within the workflow -> job -> step -> command tree."""
    location: WorkflowLocation
    start_line: int
    end_line: int
    lines: list[tuple[int, str]]          # (log_line_number, cleaned_text)


class WorkflowSegmenter:
    """
    Step 2 -- Workflow-aware Segmentation.

    Parses the normalised log and segments it into a hierarchy:
      workflow -> job -> step -> command

    Markers come from workflow_step_markers in failure_signal_patterns.json.
    The returned list of WorkflowSegment objects is consumed by FailureEventBuilder
    to attach precise location context to every failure event.
    """

    def segment(
        self,
        normalized_log: str,
        workflow_name: str,
        job_name: str,
    ) -> list[WorkflowSegment]:
        lines_raw = normalized_log.splitlines()
        segments: list[WorkflowSegment] = []

        current_step = ""
        current_cmd  = ""
        step_start_idx = 0
        step_lines: list[tuple[int, str]] = []

        for idx, raw_line in enumerate(lines_raw):
            line_no = idx + 1
            line = raw_line

            if _JOB_START_RE and _JOB_START_RE.search(line):
                job_name = line

            if _STEP_START_RE:
                m = _STEP_START_RE.match(line)
                if m:
                    if step_lines:
                        segments.append(WorkflowSegment(
                            location=WorkflowLocation(
                                workflow=workflow_name, job=job_name,
                                step=current_step, command=current_cmd,
                            ),
                            start_line=step_start_idx + 1,
                            end_line=line_no - 1,
                            lines=step_lines,
                        ))
                    current_step = m.group("step_name").strip() if "step_name" in m.groupdict() else line
                    current_cmd  = ""
                    step_start_idx = idx
                    step_lines = []
                    continue

            if _STEP_END_RE and _STEP_END_RE.match(line):
                if step_lines:
                    segments.append(WorkflowSegment(
                        location=WorkflowLocation(
                            workflow=workflow_name, job=job_name,
                            step=current_step, command=current_cmd,
                        ),
                        start_line=step_start_idx + 1,
                        end_line=line_no,
                        lines=step_lines,
                    ))
                current_step = ""
                current_cmd  = ""
                step_start_idx = idx
                step_lines = []
                continue

            if _CMD_RUN_RE and _CMD_RUN_RE.search(line):
                current_cmd = line

            step_lines.append((line_no, line))

        if step_lines:
            segments.append(WorkflowSegment(
                location=WorkflowLocation(
                    workflow=workflow_name, job=job_name,
                    step=current_step, command=current_cmd,
                ),
                start_line=step_start_idx + 1,
                end_line=len(lines_raw),
                lines=step_lines,
            ))

        return segments


# ---------------------------------------------------------------------------
# Step 3: Failure Event Construction
# ---------------------------------------------------------------------------

_FILE_PATH_RE = re.compile(r'(?:File "?([^",\n]+)"?|in ([^\s\n]+))', re.I)
_LINE_NUM_RE  = re.compile(r"line (\d+)", re.I)


def _detect_language_from_stack(lines: list[str]) -> str:
    """Identify the stack-trace language from a list of trace lines."""
    for lang, pat in _STACK_TRACE_LANG_RES.items():
        if any(pat.search(l) for l in lines):
            return lang
    return ""


def _collect_stack_trace(
    all_lines: list[tuple[int, str]],
    start_idx: int,
) -> list[str]:
    """
    Collect consecutive stack-trace lines beginning at start_idx.
    Uses per-language patterns from stack_trace_patterns and caps depth
    at _MAX_STACK_DEPTH lines (from failure_signal_patterns.json).
    """
    stack: list[str] = []
    generic_re = re.compile(
        r"(?:Traceback|^\s+at |^\s+File |^\s+in |"
        r"^\s+from |Caused by:|^\tat [a-zA-Z]|\.\.\. \d+ more)",
        re.M,
    )
    for _, line in all_lines[start_idx: start_idx + _MAX_STACK_DEPTH]:
        is_frame = generic_re.search(line) or any(
            pat.search(line) for pat in _STACK_TRACE_LANG_RES.values()
        )
        if is_frame:
            stack.append(line)
        elif stack and line.strip():
            stack.append(line)
            break
    return stack


class FailureEventBuilder:
    """
    Step 3 -- Failure Event Construction.

    For each failure-start marker found in a segment's lines:
    1. Record the start marker name and line.
    2. Scan forward until a failure-end marker or causal-window limit is reached.
    3. Collect ALL lines between start and end as context (no fixed +-N window).
    4. Detect and attach stack traces (language-aware).
    5. Extract file paths and line numbers from the error line.
    6. Return one ErrorBlock per distinct failure event, before scoring.
    """

    def build(
        self,
        segments: list[WorkflowSegment],
        now_ts: str,
    ) -> list[ErrorBlock]:
        blocks: list[ErrorBlock] = []
        seen_fingerprints: set[str] = set()

        for segment in segments:
            seg_lines = segment.lines
            seg_len   = len(seg_lines)

            for idx, (line_no, line) in enumerate(seg_lines):
                # --- Match failure start marker (change #8) ---
                start_name, _start_tier = self._match_start(line)
                if start_name is None:
                    continue

                fp = fingerprint_error(line)
                if fp in seen_fingerprints:
                    continue
                seen_fingerprints.add(fp)

                # --- Find failure block end (change #8) ---
                end_idx, end_name = self._find_end(seg_lines, idx, seg_len)

                # --- Collect full context (change #9: no fixed window) ---
                block_lines    = seg_lines[idx: end_idx]
                context_before = [l for _, l in seg_lines[max(0, idx - 5): idx]]
                context_after  = [l for _, l in block_lines[1:]]

                # --- Language-aware stack trace collection ---
                stack_trace = _collect_stack_trace(seg_lines, idx)

                # --- File path / line number ---
                file_path = ""
                ln: int | None = None
                fp_m = _FILE_PATH_RE.search(line)
                if fp_m:
                    file_path = (fp_m.group(1) or fp_m.group(2) or "").strip()
                ln_m = _LINE_NUM_RE.search(line)
                if ln_m:
                    ln = int(ln_m.group(1))

                # --- Language-specific error classification (change #7) ---
                match_payload = first_error_match(line)
                pattern_name  = match_payload[0] if match_payload else start_name
                error_type    = classify_error(line, pattern_name)
                error_code    = (
                    extract_error_code(line, match_payload[1], error_type)
                    if match_payload else error_code_for_line(line, None, error_type)
                )

                blocks.append(ErrorBlock(
                    error_type=error_type,
                    error_message=line,
                    error_code=error_code,
                    stack_trace=stack_trace,
                    context_before=context_before,
                    context_after=context_after,
                    file_path=file_path,
                    line_number=ln or line_no,
                    timestamp=now_ts,
                    location=segment.location,
                    start_marker=start_name,
                    end_marker=end_name,
                    start_line_number=line_no,
                    end_line_number=seg_lines[end_idx - 1][0] if end_idx > idx else line_no,
                    rca_weight=0,
                ))

        return blocks

    # ------------------------------------------------------------------
    def _match_start(self, line: str) -> tuple[str | None, str]:
        """Return (marker_name, severity_tier) if line matches a start marker."""
        for name, pat, tier, _is_terminal in _FAILURE_START_MARKERS:
            if pat.search(line):
                return name, tier
        # Fallback: also honour legacy failure_block_patterns
        for name, pat in FAILURE_BLOCK_PATTERNS:
            if pat.search(line):
                return name, "medium"
        return None, ""

    def _find_end(
        self,
        lines: list[tuple[int, str]],
        start_idx: int,
        seg_len: int,
    ) -> tuple[int, str]:
        """
        Scan forward from start_idx to find the failure block end.
        Returns (end_idx, end_marker_name) where end_idx is exclusive.
        """
        max_end = min(seg_len, start_idx + _CAUSAL_WINDOW_LINES * 2)
        for i in range(start_idx + 1, max_end):
            _, line = lines[i]
            for name, pat, _terminates_all, _is_cascade in _FAILURE_END_MARKERS:
                if pat.search(line):
                    return i, name
            if FAILURE_BLOCK_END_RE.search(line):
                return i, "legacy_end"
            if _STEP_START_RE and _STEP_START_RE.match(line):
                return i, "step_boundary"
        return max_end, "causal_window_limit"


# ---------------------------------------------------------------------------
# Step 4: Hierarchical Severity Scorer  (change #3)
# ---------------------------------------------------------------------------

# Severity rank for comparisons
_SEVERITY_RANK: dict[str, int] = {"critical": 4, "high": 3, "medium": 2, "low": 1}

# Tier -> base score (from taxonomy)
_TIER_BASE_SCORE: dict[str, int] = {
    "critical": 100,
    "high":     75,
    "medium":   50,
    "low":      25,
}


def _tier_for_error_type(error_type: str) -> str:
    """Look up the failure taxonomy tier for a given error type."""
    for tier_name, tier_data in _TAXONOMY_TIERS.items():
        if error_type in tier_data.get("error_types", []):
            return tier_name
    return "low"


def _tier_for_start_marker(start_marker: str) -> str:
    """Map a failure-start marker name back to its severity_tier."""
    for name, _pat, tier, _t in _FAILURE_START_MARKERS:
        if name == start_marker:
            return tier
    return "low"


class HierarchicalSeverityScorer:
    """
    Step 4 -- Hierarchical Severity Scoring.

    Scores are driven by the failure taxonomy tiers defined in
    failure_taxonomy.tiers rather than flat regex weight constants (change #3).

    Scoring dimensions
    ------------------
    Taxonomy tier    : base score from tiers: critical=100, high=75, medium=50, low=25
    Root-cause type  : taxonomy root_cause_types earn +20
    Cascading type   : taxonomy cascading_types lose -10
    Stack trace      : language-aware stack frames present -> +15
    File path        : error localised to a known file -> +10
    Error code       : non-generic concrete code -> +10
    Causal link      : "Caused by:" / "due to" lines in context -> +10
    Position bonus   : earlier in log -> up to +15 (decreases linearly)
    """

    _GENERIC_CODES: frozenset[str] = frozenset({
        "", "UNKNOWN", "UNKNOWN_ERROR", "TEST_FAILURE", "GENERIC_ERROR", "PROCESS_EXIT",
    })

    def score(
        self,
        block: "ErrorBlock",
        position_index: int,
        total_blocks: int,
    ) -> int:
        # 1. Base: taxonomy tier (prefer marker tier over error-type tier for
        #    cases where the marker has an explicit tier assignment)
        marker_tier    = _tier_for_start_marker(block.start_marker)
        type_tier      = _tier_for_error_type(block.error_type)
        effective_tier = (
            marker_tier
            if _SEVERITY_RANK.get(marker_tier, 0) >= _SEVERITY_RANK.get(type_tier, 0)
            else type_tier
        )
        base_score = _TIER_BASE_SCORE.get(effective_tier, 25)
        score = base_score // 2

        # 2. Root-cause vs cascading
        if block.error_type in _ROOT_CAUSE_TYPES_TAXONOMY:
            score += 20
        elif block.error_type in _CASCADING_TYPES_TAXONOMY:
            score -= 10

        # 3. Stack trace (language-aware frames)
        if block.stack_trace:
            score += 15

        # 4. File path localisation
        if block.file_path:
            score += 10

        # 5. Concrete error code
        if block.error_code and block.error_code.upper() not in self._GENERIC_CODES:
            score += 10

        # 6. Causal link in context
        all_ctx = "\n".join(block.context_before + [block.error_message] + block.context_after)
        if any(pat.search(all_ctx) for _, pat in _CAUSAL_LINK_PATTERNS):
            score += 10

        # 7. Position bonus (earlier = higher)
        if total_blocks > 1:
            score += round(15 * (1 - position_index / (total_blocks - 1)))
        else:
            score += 15

        return max(0, min(100, score))

    def classify_severity(self, block: "ErrorBlock") -> str:
        """Return the severity label from the taxonomy for this block's error type."""
        tier = _tier_for_error_type(block.error_type)
        if tier == "low" and block.start_marker:
            tier = _tier_for_start_marker(block.start_marker)
        # Also respect legacy signal-based classification; take the higher
        sig_severity = classify_severity(block.error_message, block.error_type, block.error_code)
        if _SEVERITY_RANK.get(sig_severity, 0) > _SEVERITY_RANK.get(tier, 0):
            return sig_severity
        return tier


# ---------------------------------------------------------------------------
# Step 5: Root-cause Validator  (change #5)
# ---------------------------------------------------------------------------

class RootCauseValidator:
    """
    Step 5 -- Root-cause Validation.

    Filters out ErrorBlock objects that are terminal-status noise rather than
    actionable root causes.  A block is marked as terminal noise when:
    - Its error_type is in the taxonomy's terminal_noise_types, OR
    - Its start_marker has is_terminal=True in failure_block_start_markers, OR
    - The error_message matches any terminal_noise_patterns entry.

    Blocks flagged is_terminal_noise=True are excluded so the RCA model never
    sees cascade wrappers (exit codes, "job failed", etc.) as the primary error.
    """

    def validate(self, blocks: list[ErrorBlock]) -> list[ErrorBlock]:
        """Return only non-terminal-noise blocks, flagging each block."""
        from dataclasses import replace as _dc_replace
        flagged: list[ErrorBlock] = []
        for blk in blocks:
            is_noise_flag = self._is_terminal_noise(blk)
            flagged.append(_dc_replace(blk, is_terminal_noise=is_noise_flag))

        kept = [b for b in flagged if not b.is_terminal_noise]
        return kept if kept else flagged[:1]

    def _is_terminal_noise(self, block: ErrorBlock) -> bool:
        if block.error_type in _TERMINAL_NOISE_TYPES:
            return True
        for name, _pat, _tier, is_terminal in _FAILURE_START_MARKERS:
            if name == block.start_marker and is_terminal:
                return True
        for pat in _TERMINAL_NOISE_PATTERNS:
            if pat.search(block.error_message):
                return True
        return False


# ---------------------------------------------------------------------------
# Step 6: Context Enricher  (change #6 – adapted for new ErrorBlock shape)
# ---------------------------------------------------------------------------

class ContextEnricher:
    """
    Step 6 -- Context Enrichment.

    Integrates CI/repository metadata into each ErrorBlock, detects language
    from stack traces, and computes severity/impact using the taxonomy scorer.
    """

    _IMPACT_MAP: dict[str, str] = {
        "build_error":         "build",
        "dependency_error":    "build",
        "java_exception":      "build",
        "python_exception":    "build",
        "shell_error":         "build",
        "test_failure":        "test",
        "configuration_error": "deployment",
        "kubernetes_error":    "deployment",
        "container_error":     "deployment",
        "permission_error":    "deployment",
        "timeout":             "deployment",
        "network_error":       "deployment",
        "resource_error":      "deployment",
        "javascript_error":    "test",
        "typescript_error":    "build",
        "ruby_error":          "test",
    }

    def __init__(self) -> None:
        self._scorer = HierarchicalSeverityScorer()

    def enrich(
        self,
        error_blocks: list[ErrorBlock],
        metadata: dict[str, Any],
    ) -> list[ErrorBlock]:
        from dataclasses import replace as _dc_replace
        enriched: list[ErrorBlock] = []
        env = metadata.get("environment", {})
        for block in error_blocks:
            severity     = self._scorer.classify_severity(block)
            new_weight   = self._scorer.score(block, 0, 1)
            final_weight = max(block.rca_weight, new_weight)

            existing_env = dict(block.ci_environment)
            existing_env.update({
                "os":       env.get("os", ""),
                "runner":   env.get("runner", ""),
                "workflow": metadata.get("workflow_name", ""),
                "job":      metadata.get("job_name", ""),
            })

            block = _dc_replace(
                block,
                repository=metadata.get("repository", ""),
                language=env.get("language", "") or _detect_language_from_stack(block.stack_trace),
                framework=env.get("framework", ""),
                ci_environment=existing_env,
                dependencies=metadata.get("dependencies", []),
                severity=severity,
                impact_scope=self._IMPACT_MAP.get(block.error_type, "unknown"),
                rca_weight=final_weight,
            )
            enriched.append(block)

        enriched.sort(key=lambda b: b.rca_weight, reverse=True)
        return enriched


# ---------------------------------------------------------------------------
# Step 7: Error Signature Generator
# ---------------------------------------------------------------------------

class ErrorSignatureGenerator:
    """
    Step 7 -- Error Signature Generation.

    Produces a deterministic SHA-256 hash and human-readable label for each
    primary error block.  Used as the key in the self-learning KB.
    Signature now includes workflow/job/step location components.
    """

    _VARIABLE_RE = re.compile(
        r"(?:0x[0-9a-f]+|[0-9a-f]{8,}|\b\d+\b|/[^\s:]+|\"[^\"]+\")",
        re.I,
    )

    def generate(self, block: ErrorBlock) -> ErrorSignatureResult:
        normalised_msg = self._normalise(block.error_message)
        normalised_loc = self._normalise(block.file_path or "")
        raw = f"{block.error_type}|{normalised_msg}|{normalised_loc}"
        sig_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        key_tokens = [
            t for t in re.split(r"[^a-z0-9]+", normalised_msg.lower())
            if len(t) > 3 and not t.isdigit()
        ][:4]
        readable = "_".join([block.error_type] + key_tokens) or sig_hash[:16]

        return ErrorSignatureResult(
            hash=sig_hash,
            readable=readable,
            components={
                "error_type":     block.error_type,
                "error_location": block.file_path or "",
                "error_pattern":  normalised_msg[:120],
                "workflow":       block.location.workflow,
                "job":            block.location.job,
                "step":           block.location.step,
            },
        )

    def _normalise(self, text: str) -> str:
        text = self._VARIABLE_RE.sub("<var>", text)
        return re.sub(r"\s+", " ", text).strip().lower()


# ---------------------------------------------------------------------------
# Compatibility shim: score_error_block for the batch extract_failure_blocks path
# ---------------------------------------------------------------------------

def score_error_block(
    error_type: str,
    error_code: str,
    severity: str,
    stack_trace: list[str],
    file_path: str,
    position_index: int,
    total_blocks: int,
) -> int:
    """
    Thin compatibility wrapper: delegates to HierarchicalSeverityScorer.
    Used by the batch extract_failure_blocks / legacy call-sites.
    """
    _scorer = HierarchicalSeverityScorer()
    from dataclasses import replace as _dc_replace
    dummy = ErrorBlock(
        error_type=error_type,
        error_message="",
        error_code=error_code,
        stack_trace=stack_trace,
        context_before=[],
        context_after=[],
        file_path=file_path,
        line_number=None,
        timestamp="",
        severity=severity,
        rca_weight=0,
    )
    return _scorer.score(dummy, position_index, total_blocks)


# ---------------------------------------------------------------------------
# Step 8: Semantic Deduplication  (change #4)
# ---------------------------------------------------------------------------

class SemanticDeduplicator:
    """
    Step 8 -- Semantic Deduplication against ChromaDB knowledge base.

    Before a new failure event reaches the RCA model, this step queries the
    ChromaDB vector store to check for semantically near-identical errors.
    Blocks whose cosine similarity to an existing KB entry exceeds
    _SEMANTIC_SIM_THRESHOLD (from failure_signal_patterns.json, default 0.85)
    are flagged _kb_duplicate=True in ci_environment so downstream consumers
    can skip full RCA and reuse cached results.

    If no vector_db is provided, deduplication is a no-op.
    """

    def __init__(self, vector_db: Any | None = None) -> None:
        self._db = vector_db
        self._embedder = TextEmbedder()

    def deduplicate(
        self,
        blocks: list[ErrorBlock],
        collection: str = "ci_failure_logs",
    ) -> list[ErrorBlock]:
        if not self._db:
            return blocks

        from dataclasses import replace as _dc_replace
        deduped: list[ErrorBlock] = []
        for blk in blocks:
            query_text = blk.error_message
            if blk.stack_trace:
                query_text = "\n".join([blk.error_message] + blk.stack_trace[:5])
            try:
                results = self._db.search(
                    query=query_text,
                    n_results=1,
                    collection=collection,
                )
            except Exception:  # noqa: BLE001
                deduped.append(blk)
                continue

            is_dup = False
            if results:
                top_score = results[0].score if hasattr(results[0], "score") else 0.0
                is_dup = float(top_score) >= _SEMANTIC_SIM_THRESHOLD

            env = dict(blk.ci_environment)
            env["_kb_duplicate"] = is_dup
            deduped.append(_dc_replace(blk, ci_environment=env))

        return deduped


# ---------------------------------------------------------------------------
# Main PreprocessingPipeline orchestrator
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """
    Real-time preprocessing pipeline for CI failure logs.

    Orchestrates the redesigned nine-step pipeline:

    Step 1: Log Normalization          -> clean, deduplicated log text
    Step 2: Workflow-aware Segmentation -> workflow -> job -> step -> command tree (change #1)
    Step 3: Failure Event Construction  -> FailureEventBuilder groups error lines,
                                           stack traces, and metadata using start/end
                                           markers (change #8); context = full block (change #9)
    Step 4: Hierarchical Severity Score -> HierarchicalSeverityScorer using taxonomy
                                           tiers (change #3)
    Step 5: Root-cause Validation       -> RootCauseValidator filters terminal noise (change #5)
    Step 6: Context Enrichment          -> CI/repo metadata, language detection
    Step 7: Error Signature             -> deterministic SHA-256 + readable label
    Step 8: Semantic Deduplication      -> ChromaDB KB similarity check (change #4)
    Step 9: Vector Embedding            -> 384-dim L2-normalised embeddings

    Returns a ProcessedLog ready for the RCA Agent.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        context_window: int = 10,     # kept for API compatibility; not used as fixed window
        vector_db: Any | None = None,
    ) -> None:
        self.log_parser           = LogParser()
        self.segmenter            = WorkflowSegmenter()
        self.event_builder        = FailureEventBuilder()
        self.severity_scorer      = HierarchicalSeverityScorer()
        self.root_cause_validator = RootCauseValidator()
        self.context_enricher     = ContextEnricher()
        self.signature_generator  = ErrorSignatureGenerator()
        self.deduplicator         = SemanticDeduplicator(vector_db=vector_db)
        self.embedder             = TextEmbedder(model=embedding_model)

    # ------------------------------------------------------------------
    def process(self, raw_log: str, metadata: dict[str, Any]) -> ProcessedLog:
        """
        Transform a raw CI failure log into a fully structured ProcessedLog.

        Parameters
        ----------
        raw_log : str
            Raw text from a GitHub Actions log file.
        metadata : dict
            Run/job metadata: repository, run_id, workflow_name, job_name,
            branch, commit_sha.  Optionally an 'environment' sub-dict with
            os/runner/language/framework.
        """
        from datetime import datetime, timezone
        now_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

        # Step 1 -- Log Normalization
        normalized = self.log_parser.normalize(raw_log)

        # Step 2 -- Workflow-aware Segmentation
        segments = self.segmenter.segment(
            normalized,
            workflow_name=metadata.get("workflow_name", ""),
            job_name=metadata.get("job_name", ""),
        )
        segment_dicts = [
            {
                "workflow":   s.location.workflow,
                "job":        s.location.job,
                "step":       s.location.step,
                "command":    s.location.command,
                "start_line": s.start_line,
                "end_line":   s.end_line,
                "line_count": len(s.lines),
            }
            for s in segments
        ]

        # Step 3 -- Failure Event Construction
        raw_blocks = self.event_builder.build(segments, now_ts)

        # Step 4 -- Hierarchical Severity Scoring
        total = len(raw_blocks)
        from dataclasses import replace as _dc_replace
        scored_blocks: list[ErrorBlock] = []
        for pos, blk in enumerate(raw_blocks):
            weight = self.severity_scorer.score(blk, pos, total)
            scored_blocks.append(_dc_replace(blk, rca_weight=weight))

        # Step 5 -- Root-cause Validation
        validated_blocks = self.root_cause_validator.validate(scored_blocks)

        # Step 6 -- Context Enrichment
        enriched_blocks = self.context_enricher.enrich(validated_blocks, metadata)

        # Step 7 -- Error Signature Generation
        primary_error = self._identify_primary(enriched_blocks)
        error_signature = (
            self.signature_generator.generate(primary_error) if primary_error else None
        )

        # Step 8 -- Semantic Deduplication
        enriched_blocks = self.deduplicator.deduplicate(enriched_blocks)

        # Step 9 -- Vector Embedding
        embeddings = self._generate_embeddings(primary_error) if primary_error else None

        return ProcessedLog(
            normalized_log=normalized,
            workflow_segments=segment_dicts,
            error_blocks=enriched_blocks,
            primary_error=primary_error,
            error_signature=error_signature,
            embeddings=embeddings,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def process_to_rca_input(
        self,
        raw_log: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Process a log and return the structured RCA input dict for the RCA Agent.

        Keys returned:
          vectorized_error/context/combined  -- 384-dim L2-normalised embeddings
          error_signature / hash / components
          error_blocks    -- list sorted by rca_weight descending (root cause first)
          primary_error   -- highest-weight block with full block context and location
          workflow_segments -- workflow hierarchy context
          metadata        -- CI/run context
          model_tuning_params -- Ollama generation parameters
        """
        processed = self.process(raw_log, metadata)

        primary_dict: dict[str, Any] = {}
        if processed.primary_error:
            pb = processed.primary_error
            primary_dict = {
                "error_type":       pb.error_type,
                "error_message":    pb.error_message,
                "error_code":       pb.error_code,
                "stack_trace":      pb.stack_trace,
                # Full failure-block context (change #9: start-to-end, no truncation)
                "context_before":   pb.context_before,
                "context_after":    pb.context_after,
                "file_path":        pb.file_path,
                "line_number":      pb.line_number,
                "severity":         pb.severity,
                "impact_scope":     pb.impact_scope,
                "repository":       pb.repository,
                "language":         pb.language,
                "framework":        pb.framework,
                "ci_environment":   pb.ci_environment,
                "rca_weight":       pb.rca_weight,
                # Workflow location (change #1)
                "workflow_location": {
                    "workflow": pb.location.workflow,
                    "job":      pb.location.job,
                    "step":     pb.location.step,
                    "command":  pb.location.command,
                },
                # Failure block boundaries (change #8)
                "start_marker":       pb.start_marker,
                "end_marker":         pb.end_marker,
                "start_line_number":  pb.start_line_number,
                "end_line_number":    pb.end_line_number,
                "is_terminal_noise":  pb.is_terminal_noise,
            }

        error_blocks_list = [
            {
                "error_type":        b.error_type,
                "error_message":     b.error_message,
                "error_code":        b.error_code,
                "severity":          b.severity,
                "impact_scope":      b.impact_scope,
                "file_path":         b.file_path,
                "line_number":       b.line_number,
                "rca_weight":        b.rca_weight,
                "context_before":    b.context_before,
                "context_after":     b.context_after,
                "stack_trace":       b.stack_trace,
                "start_marker":      b.start_marker,
                "end_marker":        b.end_marker,
                "is_terminal_noise": b.is_terminal_noise,
                "workflow_location": {
                    "workflow": b.location.workflow,
                    "job":      b.location.job,
                    "step":     b.location.step,
                    "command":  b.location.command,
                },
            }
            for b in sorted(
                processed.error_blocks or [],
                key=lambda b: b.rca_weight,
                reverse=True,
            )
        ]

        sig = processed.error_signature
        emb = processed.embeddings

        return {
            "vectorized_error":           emb.error_embedding    if emb else [],
            "vectorized_context":         emb.context_embedding  if emb else [],
            "vectorized_combined":        emb.combined_embedding if emb else [],
            "error_signature":            sig.readable    if sig else "",
            "error_signature_hash":       sig.hash        if sig else "",
            "error_signature_components": sig.components  if sig else {},
            "error_blocks":        error_blocks_list,
            "primary_error":       primary_dict,
            "workflow_segments":   processed.workflow_segments,
            "metadata":            metadata,
            # "model_tuning_params": {
            #     "num_predict":    1024,
            #     "temperature":    0.1,
            #     "top_p":          0.85,
            #     "top_k":          40,
            #     "repeat_penalty": 1.1,
            # },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _identify_primary(self, blocks: list[ErrorBlock]) -> ErrorBlock | None:
        """Return the highest rca_weight non-terminal-noise block."""
        if not blocks:
            return None
        candidates = [b for b in blocks if not b.is_terminal_noise] or blocks
        return max(candidates, key=lambda b: b.rca_weight)

    def _generate_embeddings(self, primary_error: ErrorBlock) -> EmbeddingResult:
        """Generate three 384-dim embeddings (error / context / combined)."""
        error_text = primary_error.error_message
        # Full failure-block context (change #9: no truncation)
        context_text = "\n".join(
            primary_error.context_before
            + [primary_error.error_message]
            + primary_error.context_after
        )
        if primary_error.stack_trace:
            context_text = "\n".join(primary_error.stack_trace) + "\n" + context_text
        combined_text = f"{error_text} {context_text}"

        vecs = self.embedder.embed_batch([error_text, context_text, combined_text])
        return EmbeddingResult(
            error_embedding=vecs[0],
            context_embedding=vecs[1],
            combined_embedding=vecs[2],
        )


# ---------------------------------------------------------------------------
# datetime import (needed by PreprocessingPipeline.process)
# ---------------------------------------------------------------------------
from datetime import datetime, timezone  # noqa: E402


# ---------------------------------------------------------------------------
# Backward-compat: ErrorBlockExtractor shim for the batch extract_signals path
# ---------------------------------------------------------------------------

_STACK_TRACE_RE = re.compile(
    r"(?:Traceback|at |^\s+File |^\s+in |Error:|Exception:)", re.M
)


class ErrorBlockExtractor:
    """
    Backward-compatible extractor used by the batch extract_signals path.

    The real-time pipeline (PreprocessingPipeline) uses FailureEventBuilder.
    This class bridges legacy call-sites without breaking them.
    """

    def __init__(self, context_window: int = 10) -> None:
        self._builder   = FailureEventBuilder()
        self._segmenter = WorkflowSegmenter()
        self._scorer    = HierarchicalSeverityScorer()

    def extract(self, normalized_log: str) -> list[ErrorBlock]:
        from datetime import datetime, timezone
        from dataclasses import replace as _dc_replace
        now_ts     = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        segments   = self._segmenter.segment(normalized_log, "", "")
        raw_blocks = self._builder.build(segments, now_ts)
        total      = len(raw_blocks)
        scored: list[ErrorBlock] = []
        for pos, blk in enumerate(raw_blocks):
            weight = self._scorer.score(blk, pos, total)
            scored.append(_dc_replace(blk, rca_weight=weight))
        scored.sort(key=lambda b: b.rca_weight, reverse=True)
        return scored


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
    parser.add_argument("--vector-db", type=Path, help="ChromaDB persist directory. Defaults to data-dir/chroma_store.")
    parser.add_argument("--skip-vector-db", action="store_true", help="Do not build the ChromaDB vector store.")
    parser.add_argument("--append-vectors", action="store_true", help="Append to vector store instead of replacing ci_failure_logs.")
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
    parser.add_argument(
        "--huggingface-dataset-path",
        type=Path,
        help="Local file or directory for a Hugging Face DevOps incident dataset export.",
    )
    parser.add_argument(
        "--huggingface-dataset-id",
        default=DEFAULT_HUGGINGFACE_DATASET_ID,
        help=f"Hugging Face dataset id. Defaults to {DEFAULT_HUGGINGFACE_DATASET_ID}.",
    )
    parser.add_argument(
        "--download-huggingface",
        action="store_true",
        help="Download a Hugging Face DevOps incident dataset through the datasets-server rows API.",
    )
    parser.add_argument(
        "--huggingface-splits",
        default=",".join(DEFAULT_HUGGINGFACE_SPLITS),
        help="Comma-separated Hugging Face splits to download.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    index_path = args.index or (data_dir / "index.json")
    vector_db_path = args.vector_db or (data_dir / "chroma_store")
    kaggle_path = resolve_kaggle_dataset_path(args, data_dir)
    huggingface_path = resolve_huggingface_dataset_path(args, data_dir)

    if not index_path.exists() and not kaggle_path and not huggingface_path:
        print(f"Missing index: {index_path}", file=sys.stderr)
        print(
            "Run scripts/log-collector.py first, pass --kaggle-dataset-path / --download-kaggle, "
            "or pass --huggingface-dataset-path / --download-huggingface.",
            file=sys.stderr,
        )
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

    if huggingface_path:
        if not huggingface_path.exists():
            print(f"Hugging Face dataset path does not exist: {huggingface_path}", file=sys.stderr)
        else:
            huggingface_signals = preprocess_kaggle_dataset(huggingface_path, max_records=args.max_kaggle_records)
            all_signals.extend(huggingface_signals)
            for signal in huggingface_signals:
                metadata = signal_metadata(signal, {})
                metadata["doc_kind"] = "huggingface_devops_incident"
                metadata["dataset_url"] = DEFAULT_HUGGINGFACE_DATASET_URL
                vector_documents.append(
                    {
                        "document_id": f"huggingface_{signal.signal_id}",
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
        vector_db = ChromaVectorDB(persist_dir=vector_db_path)
        try:
            if not args.append_vectors:
                vector_db.clear_collection("ci_failure_logs")
            vector_count = vector_db.upsert_documents(vector_documents, collection="ci_failure_logs")
        finally:
            vector_db.close()

    print(f"Processed runs: {len(run_records)}; skipped missing zips: {skipped}")
    if kaggle_path:
        print(f"Kaggle dataset: {kaggle_path}")
    if huggingface_path:
        print(f"Hugging Face dataset: {huggingface_path}")
    print(f"Error signals: {signal_count}")
    print(f"Failure blocks: {block_count}")
    print(f"Preprocessed documents: {preprocessed_count}")
    print(f"Training examples: {dataset_count}")
    print(f"Knowledge graph: {data_dir / args.graph_output}")
    if not args.skip_vector_db:
        print(f"Vector documents indexed: {vector_count}")
        print(f"ChromaDB store: {vector_db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
