#!/usr/bin/env python3
"""RCA orchestration for GitHub Actions CI failures."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
DATA_DIR = ROOT_DIR / "data"
DEFAULT_MODEL_ID = "anthropic.claude-sonnet-4-20250514-v1:0"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from env_loader import load_dotenv  # noqa: E402
from db import LocalVectorDB  # noqa: E402


load_dotenv(ROOT_DIR / ".env")


def load_script_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


log_collector = load_script_module("smart_ci_log_collector", SCRIPTS_DIR / "log-collector.py")
preprocess = load_script_module("smart_ci_preprocess", SCRIPTS_DIR / "pre-process-pipeline.py")


DEFAULT_ANTHROPIC_MODEL_ID = "claude-sonnet-4-20250514"
GENERIC_ERROR_TYPES = {"", "unknown_error", "generic_error"}
LESS_SPECIFIC_ERROR_TYPES = GENERIC_ERROR_TYPES | {"test_failure", "process_exit"}
GENERIC_ERROR_CODES = {"", "UNKNOWN", "UNKNOWN_ERROR", "TEST_FAILURE", "GENERIC_ERROR"}
SPECIFIC_ERROR_HINTS = re.compile(
    r"\b(?:403|401|forbidden|unauthorized|permission denied|access denied|token|write access|"
    r"timeout|timed out|connection refused|connection reset|no space left|out of memory|"
    r"module not found|cannot find module|failed to compile|syntax error)\b",
    re.I,
)
BEDROCK_MODEL_ALIASES = {
    "anthropic.claude-sonnet-4": DEFAULT_MODEL_ID,
    "claude-sonnet-4": DEFAULT_MODEL_ID,
}
ANTHROPIC_MODEL_ALIASES = {
    "anthropic.claude-sonnet-4": DEFAULT_ANTHROPIC_MODEL_ID,
    "claude-sonnet-4": DEFAULT_ANTHROPIC_MODEL_ID,
    DEFAULT_MODEL_ID: DEFAULT_ANTHROPIC_MODEL_ID,
}


class ClaudeAnthropicClient:
    def __init__(
        self,
        model_id: str = DEFAULT_ANTHROPIC_MODEL_ID,
        api_key: str | None = None,
        api_url: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        self.model_id = ANTHROPIC_MODEL_ALIASES.get(model_id, model_id)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("MODEL_API_KEY")
        self.api_url = api_url or os.getenv("ANTHROPIC_API_URL") or "https://api.anthropic.com/v1/messages"
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY or MODEL_API_KEY is required for Anthropic API inference.")

        body = json.dumps(
            {
                "model": self.model_id,
                "max_tokens": self.max_tokens,
                "system": (
                    "You are a CI/CD failure diagnosis agent. Return strict JSON with "
                    "rca_summary, failure_stage, failure_type, error_code, severity, "
                    "failure_location, evidence, remediation_steps, inline_fix_suggestions, "
                    "verification_commands, and confidence. Do not include raw log metadata "
                    "or retrieved_context."
                ),
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.api_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic API error {exc.code}: {error_body}") from exc

        content = payload.get("content", [])
        if content and isinstance(content, list):
            return "\n".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
        return json.dumps(payload, sort_keys=True)


class ClaudeBedrockClient:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        region_name: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        self.model_id = BEDROCK_MODEL_ALIASES.get(model_id, model_id)
        self.region_name = region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        import boto3  # type: ignore[import-not-found]

        client = boto3.client("bedrock-runtime", region_name=self.region_name)
        response = client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "system": (
                        "You are a CI/CD failure diagnosis agent. Return strict JSON with "
                        "rca_summary, failure_stage, failure_type, error_code, severity, "
                        "failure_location, evidence, remediation_steps, inline_fix_suggestions, "
                        "verification_commands, and confidence. Do not include raw log metadata "
                        "or retrieved_context."
                    ),
                    "messages": [{"role": "user", "content": prompt}],
                }
            ),
        )
        payload = json.loads(response["body"].read())
        content = payload.get("content", [])
        if content and isinstance(content, list):
            return "\n".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
        return json.dumps(payload, sort_keys=True)


class ClaudeRcaClient:
    def __init__(self, model_id: str, max_tokens: int = 2048) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        provider = (os.getenv("MODEL_PROVIDER") or "").strip().lower()
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("MODEL_API_KEY")
        if provider == "bedrock" or (not api_key and provider != "anthropic"):
            return ClaudeBedrockClient(self.model_id, max_tokens=self.max_tokens).complete(prompt)
        return ClaudeAnthropicClient(self.model_id, api_key=api_key, max_tokens=self.max_tokens).complete(prompt)


def collect_run_from_url(run_url: str, token: str | None = None, data_dir: Path = DATA_DIR) -> dict[str, Any]:
    run_ref = log_collector.parse_workflow_run_url(run_url)
    client = log_collector.GitHubActionsClient(token=token or os.getenv("GITHUB_TOKEN"))
    run = log_collector.get_workflow_run(client, run_ref.repository, run_ref.run_id)
    jobs = log_collector.list_jobs_for_run(client, run_ref.repository, run_ref.run_id, max_pages=5)
    log_dir = data_dir / "logs" / log_collector.safe_repo_name(run_ref.repository)
    download = log_collector.download_run_zip(client, run_ref.repository, run, log_dir, overwrite=True)
    record = log_collector.compact_run_metadata(run_ref.repository, run, jobs, download, data_dir)
    record = log_collector.write_run_metadata_sidecar(data_dir, record)
    log_collector.upsert_index(data_dir / "index.json", run_ref.repository, [record])
    return record


def preprocess_run_record(
    record: dict[str, Any],
    data_dir: Path = DATA_DIR,
    vector_db_path: Path | None = None,
) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    zip_path = preprocess.resolve_zip_path(data_dir, record)
    if not zip_path or not zip_path.exists():
        raise FileNotFoundError(f"Missing collected zip for run {record.get('run_id')}: {zip_path}")

    signals, documents, failure_blocks = preprocess.process_zip(
        zip_path=zip_path,
        run_metadata=record,
        context_lines=8,
        max_signals_per_log=100,
        chunk_lines=120,
        chunk_overlap=20,
    )
    db_path = vector_db_path or (data_dir / "vector_store.sqlite")
    vector_db = LocalVectorDB(db_path)
    try:
        vector_db.upsert_documents(documents, collection="gha_logs")
    finally:
        vector_db.close()
    return signals, failure_blocks, documents


def retrieve_similar_context(query: str, top_k: int = 5, data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    vector_db = LocalVectorDB(data_dir / "vector_store.sqlite")
    try:
        results = vector_db.search(query, top_k=top_k, collection="gha_logs")
    finally:
        vector_db.close()
    return [
        {
            "document_id": result.document_id,
            "score": round(result.score, 4),
            "text": result.text,
            "metadata": result.metadata,
        }
        for result in results
    ]


def truncate_text(value: Any, limit: int = 900) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def line_range(start_line: Any, end_line: Any) -> str:
    if start_line and end_line and start_line != end_line:
        return f"{start_line}-{end_line}"
    if start_line:
        return str(start_line)
    return ""


def parse_numbered_line(line: str) -> tuple[int | None, str]:
    match = re.match(r"^\s*(?P<line>\d+):\s*(?P<message>.*)$", line)
    if not match:
        return None, line.strip()
    return int(match.group("line")), match.group("message").strip()


def best_evidence_line(text: str) -> tuple[int | None, str]:
    best_line_number: int | None = None
    best_message = ""
    best_score = -1

    for raw_line in str(text or "").splitlines():
        line_number, message = parse_numbered_line(raw_line)
        if not message:
            continue
        lowered = message.lower()
        score = 0
        if "##[error]" in lowered or "::error" in lowered or "error:" in lowered:
            score += 10
        if SPECIFIC_ERROR_HINTS.search(message):
            score += 30
        if re.search(r"\b[45]\d{2}\b", message):
            score += 20
        if score > best_score:
            best_score = score
            best_line_number = line_number
            best_message = message

    if best_message:
        return best_line_number, best_message
    return None, truncate_text(text, 600)


def infer_error_details(text: str, error_type: str, error_code: str, severity: str) -> tuple[str, str, str]:
    match_payload = preprocess.first_error_match(text)
    inferred_type = preprocess.classify_error(text, match_payload[0] if match_payload else "generic_error")
    selected_type = error_type or "unknown_error"
    if inferred_type not in GENERIC_ERROR_TYPES and (
        selected_type in LESS_SPECIFIC_ERROR_TYPES or inferred_type != "test_failure"
    ):
        selected_type = inferred_type

    selected_code = error_code or "UNKNOWN"
    inferred_code = preprocess.error_code_for_line(text, match_payload, selected_type)
    http_status_match = re.search(r"\b([45]\d{2})\b", text)
    if http_status_match and any(token in text.lower() for token in ("http", "status", "response")):
        inferred_code = f"HTTP_{http_status_match.group(1)}"
    if inferred_code and (selected_code in GENERIC_ERROR_CODES or selected_type != error_type):
        selected_code = inferred_code

    selected_severity = severity or "low"
    inferred_severity = preprocess.classify_severity(text, selected_type, selected_code)
    if inferred_severity:
        selected_severity = inferred_severity
    if selected_severity == "low" and selected_type in {"permission_error", "network_error", "timeout", "resource_error"}:
        selected_severity = "medium"
    return selected_type, selected_code, selected_severity


def compact_retrieved_context(retrieved_context: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in retrieved_context[:limit]:
        metadata = item.get("metadata") or {}
        compact.append(
            {
                "score": item.get("score"),
                "failure_type": metadata.get("failure_type") or metadata.get("error_type"),
                "error_code": metadata.get("error_code"),
                "severity": metadata.get("severity"),
                "location": {
                    "job": metadata.get("job_name"),
                    "log_file": metadata.get("file_name"),
                    "line": metadata.get("line_number"),
                    "line_range": line_range(metadata.get("start_line"), metadata.get("end_line")),
                },
                "text": truncate_text(item.get("text"), 900),
            }
        )
    return compact


def compact_signal(signal: Any) -> dict[str, Any]:
    return {
        "stage": signal.section,
        "failure_type": signal.error_type,
        "error_code": signal.error_code,
        "severity": signal.severity,
        "location": {
            "job": signal.job_name,
            "log_file": signal.file_name,
            "line": signal.line_number,
        },
        "message": truncate_text(signal.signal_line, 500),
        "context": truncate_text(signal.context, 900),
    }


def compact_failure_block(block: Any) -> dict[str, Any]:
    line_number, message = best_evidence_line(block.text)
    return {
        "stage": block.failure_stage,
        "failure_type": block.failure_type,
        "error_code": block.error_code,
        "severity": block.severity,
        "location": {
            "job": block.job_name,
            "log_file": block.file_name,
            "line": line_number or block.start_line,
            "line_range": line_range(block.start_line, block.end_line),
        },
        "message": truncate_text(message or block.error_message, 500),
        "context": truncate_text(block.text, 900),
    }


def build_prompt(
    run_url: str,
    record: dict[str, Any],
    signals: list[Any],
    failure_blocks: list[Any],
    retrieved_context: list[dict[str, Any]],
) -> str:
    signal_payload = [compact_signal(signal) for signal in signals[:12]]
    block_payload = [compact_failure_block(block) for block in failure_blocks[:8]]
    payload = {
        "run_url": run_url,
        "workflow": {
            "repository": record.get("repository"),
            "run_id": record.get("run_id"),
            "workflow_name": record.get("workflow_name"),
            "status": record.get("status"),
            "conclusion": record.get("conclusion"),
            "branch": record.get("branch"),
            "commit_sha": record.get("commit_sha"),
        },
        "error_signals": signal_payload,
        "failure_blocks": block_payload,
        "retrieved_similar_context": compact_retrieved_context(retrieved_context),
    }
    return (
        "Diagnose this CI/CD failure from the supplied evidence. "
        "Prefer the first concrete failure over later cascading errors. "
        "Return only compact JSON with rca_summary, failure_stage, failure_type, "
        "error_code, severity, failure_location, evidence, remediation_steps, "
        "inline_fix_suggestions, verification_commands, and confidence. "
        "inline_fix_suggestions must be an array of objects with target, "
        "suggested_change, and rationale fields. Make each suggestion small enough "
        "to apply directly in a workflow, dependency file, test, application code, "
        "or runtime configuration. Do not include retrieved_context or raw metadata.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def parse_json_response(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def failure_summary(error_type: str, error_code: str, evidence_message: str) -> str:
    if error_type == "permission_error" and ("403" in error_code or "403" in evidence_message):
        return (
            "The workflow was denied access while executing the failing step. "
            "The evidence shows an HTTP 403/Forbidden response, so the token or account "
            "does not have the required repository permissions."
        )
    if error_type == "permission_error":
        return "The workflow failed because a required token, secret, path, or remote resource was not accessible."
    return preprocess.summarize_failure(SimpleNamespace(error_type=error_type))


def source_candidate_from_signal(signal: Any) -> dict[str, Any]:
    error_type, error_code, severity = infer_error_details(
        signal.context or signal.signal_line,
        signal.error_type,
        signal.error_code,
        signal.severity,
    )
    _, evidence_message = best_evidence_line(signal.context or signal.signal_line)
    return {
        "kind": "signal",
        "stage": signal.section,
        "failure_type": error_type,
        "error_code": error_code,
        "severity": severity,
        "location": {
            "job": signal.job_name,
            "log_file": signal.file_name,
            "line": signal.line_number,
            "line_range": "",
        },
        "message": evidence_message or signal.signal_line,
        "context": signal.context,
    }


def source_candidate_from_block(block: Any) -> dict[str, Any]:
    line_number, evidence_message = best_evidence_line(block.text or block.error_message)
    error_type, error_code, severity = infer_error_details(
        block.text or block.error_message,
        block.failure_type,
        block.error_code,
        block.severity,
    )
    return {
        "kind": "failure_block",
        "stage": block.failure_stage,
        "failure_type": error_type,
        "error_code": error_code,
        "severity": severity,
        "location": {
            "job": block.job_name,
            "log_file": block.file_name,
            "line": line_number or block.start_line,
            "line_range": line_range(block.start_line, block.end_line),
        },
        "message": evidence_message or block.error_message,
        "context": block.text,
    }


def candidate_score(candidate: dict[str, Any]) -> int:
    text = f"{candidate.get('message', '')}\n{candidate.get('context', '')}"
    failure_type = str(candidate.get("failure_type") or "")
    error_code = str(candidate.get("error_code") or "")
    score = 0
    if candidate.get("kind") == "failure_block":
        score += 15
    if failure_type not in GENERIC_ERROR_TYPES:
        score += 20
    if failure_type not in LESS_SPECIFIC_ERROR_TYPES:
        score += 35
    if error_code not in GENERIC_ERROR_CODES:
        score += 25
    if SPECIFIC_ERROR_HINTS.search(text):
        score += 35
    if re.search(r"\b[45]\d{2}\b", text):
        score += 20
    return score


def select_primary_failure(signals: list[Any], failure_blocks: list[Any]) -> dict[str, Any] | None:
    candidates = [source_candidate_from_block(block) for block in failure_blocks]
    candidates.extend(source_candidate_from_signal(signal) for signal in signals)
    if not candidates:
        return None
    return max(candidates, key=candidate_score)


def compact_evidence(primary: dict[str, Any] | None, extra_messages: list[str] | None = None) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()

    if primary:
        location = primary.get("location") or {}
        message = truncate_text(primary.get("message"), 700)
        if message:
            evidence.append(
                {
                    "line": location.get("line"),
                    "message": message,
                }
            )
            seen.add(message)

    for message in extra_messages or []:
        message = truncate_text(message, 700)
        if message and message not in seen:
            evidence.append({"line": None, "message": message})
            seen.add(message)
        if len(evidence) >= 3:
            break
    return evidence


def model_value(model_response: dict[str, Any] | None, *keys: str) -> Any:
    if not model_response:
        return None
    for key in keys:
        value = model_response.get(key)
        if value:
            return value
    return None


def evidence_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("message") or item.get("text") or item.get("line") or "").strip()
    return str(item).strip()


def as_text_list(value: Any) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_inline_fix_suggestions(value: Any) -> list[dict[str, str]]:
    suggestions: list[dict[str, str]] = []
    items = value if isinstance(value, list | tuple) else [value]

    for item in items:
        if isinstance(item, dict):
            target = str(
                item.get("target")
                or item.get("file")
                or item.get("path")
                or item.get("location")
                or item.get("area")
                or ""
            ).strip()
            suggested_change = str(
                item.get("suggested_change")
                or item.get("fix")
                or item.get("suggestion")
                or item.get("change")
                or item.get("patch")
                or ""
            ).strip()
            rationale = str(item.get("rationale") or item.get("reason") or item.get("why") or "").strip()
        else:
            target = ""
            suggested_change = str(item or "").strip()
            rationale = ""

        if not suggested_change:
            continue

        suggestion = {"suggested_change": suggested_change}
        if target:
            suggestion["target"] = target
        if rationale:
            suggestion["rationale"] = rationale
        suggestions.append(suggestion)

    return suggestions


def fallback_inline_fix_suggestions(
    failure_type: str,
    error_code: str,
    primary: dict[str, Any] | None,
) -> list[dict[str, str]]:
    location = primary.get("location") if primary else {}
    location_text = ""
    if isinstance(location, dict):
        location_parts = [
            str(location.get("job") or "").strip(),
            str(location.get("log_file") or "").strip(),
            str(location.get("line") or "").strip(),
        ]
        location_text = " / ".join(part for part in location_parts if part)

    targets = {
        "kubernetes_error": "Kubernetes manifest, cluster RBAC, image, or runner cluster context",
        "container_error": "Dockerfile, image build step, registry credentials, or container runtime settings",
        "permission_error": "Workflow permissions, repository secrets, cloud IAM, or Kubernetes RBAC",
        "timeout": "Failing workflow step timeout, retry policy, or external dependency call",
        "network_error": "Runner network, proxy, DNS, TLS, or remote service configuration",
        "resource_error": "Runner size, disk cleanup step, memory usage, or job parallelism",
        "test_failure": "Failing test, fixture, assertion, or implementation touched by the test",
        "dependency_error": "Dependency manifest, lockfile, registry configuration, or cache restore step",
        "build_error": "Build source file, compiler configuration, or build command",
        "configuration_error": "Workflow YAML, environment variable, referenced path, or secret",
        "process_exit": "Workflow step command that exited with a non-zero status",
    }
    changes = {
        "kubernetes_error": "Correct the Kubernetes resource, image, namespace, or RBAC setting referenced by the failure evidence.",
        "container_error": "Fix the image build/pull configuration and confirm the registry credentials used by the failing step.",
        "permission_error": "Grant the missing least-privilege access or update the referenced secret/token used by the failing command.",
        "timeout": "Add retry/backoff for the slow operation or increase the step timeout after confirming the longer runtime is expected.",
        "network_error": "Validate proxy, DNS, TLS, and remote endpoint settings for the runner, then add retry/backoff around the remote call.",
        "resource_error": "Reduce parallelism, clean up disk-heavy artifacts before the failing step, or move the job to a larger runner.",
        "test_failure": "Update the failing assertion, fixture, or implementation after confirming the intended behavior.",
        "dependency_error": "Pin, update, or restore the failing dependency in the manifest/lockfile and rebuild dependency caches.",
        "build_error": "Fix the compiler/build error in the referenced source or configuration and rerun the build command locally.",
        "configuration_error": "Correct the missing or invalid workflow input, environment variable, path, or secret referenced by the evidence.",
        "process_exit": "Make the failing command handle the expected condition or correct the argument/configuration that caused the exit.",
    }

    rationale = f"Generated from the classified {failure_type or 'unknown'} failure"
    if error_code:
        rationale += f" with error code {error_code}"
    if location_text:
        rationale += f" near {location_text}"
    rationale += "."

    return [
        {
            "target": targets.get(failure_type, location_text or "Failing workflow step or referenced source/configuration"),
            "suggested_change": changes.get(
                failure_type,
                "Apply the smallest workflow, code, dependency, or environment change indicated by the failure evidence.",
            ),
            "rationale": rationale,
        }
    ]


def normalize_rca_report(
    model_response: dict[str, Any] | None,
    signals: list[Any],
    failure_blocks: list[Any],
    model_error: str | None = None,
) -> dict[str, Any]:
    primary = select_primary_failure(signals, failure_blocks)
    fallback_type = primary.get("failure_type") if primary else "unknown_error"
    fallback_code = primary.get("error_code") if primary else "UNKNOWN"
    fallback_severity = primary.get("severity") if primary else "low"
    evidence_message = str(primary.get("message") if primary else "")

    model_failure_type = str(model_value(model_response, "failure_type", "error_type") or "")
    failure_type = model_failure_type if model_failure_type not in LESS_SPECIFIC_ERROR_TYPES else fallback_type
    if not failure_type:
        failure_type = fallback_type

    model_error_code = str(model_value(model_response, "error_code") or "")
    error_code = model_error_code if model_error_code not in GENERIC_ERROR_CODES else fallback_code
    if not error_code:
        error_code = fallback_code

    severity = str(model_value(model_response, "severity") or fallback_severity or "low")
    summary = str(model_value(model_response, "rca_summary", "root_cause", "summary") or "").strip()
    if not summary:
        summary = failure_summary(failure_type, error_code, evidence_message)

    model_evidence = model_value(model_response, "evidence")
    extra_messages: list[str] = []
    if isinstance(model_evidence, list):
        extra_messages = [text for item in model_evidence if (text := evidence_text(item))]
    elif isinstance(model_evidence, str):
        extra_messages = [model_evidence]
    confidence = str(model_value(model_response, "confidence") or ("medium" if primary else "low"))
    model_location = model_value(model_response, "failure_location", "location")
    failure_location = primary.get("location") if primary else (model_location if isinstance(model_location, dict) else {})
    remediation_steps = as_text_list(model_value(model_response, "remediation_steps")) or preprocess.recommended_steps(failure_type)
    inline_fix_suggestions = normalize_inline_fix_suggestions(
        model_value(model_response, "inline_fix_suggestions", "inline_fixes", "fix_suggestions", "suggested_fixes")
    )
    if not inline_fix_suggestions:
        inline_fix_suggestions = fallback_inline_fix_suggestions(failure_type, error_code, primary)
    verification_commands = as_text_list(model_value(model_response, "verification_commands")) or [
        "Re-run the failed workflow after applying the remediation."
    ]

    report = {
        "rca_summary": summary,
        "failure_stage": str(model_value(model_response, "failure_stage") or (primary.get("stage") if primary else "unknown_stage")),
        "failure_type": failure_type,
        "error_type": failure_type,
        "error_code": error_code,
        "severity": severity,
        "failure_location": failure_location,
        "evidence": compact_evidence(primary, extra_messages),
        "remediation_steps": remediation_steps,
        "inline_fix_suggestions": inline_fix_suggestions,
        "verification_commands": verification_commands,
        "confidence": confidence,
        "diagnosis_mode": "model" if model_response else "heuristic_fallback",
    }
    if model_error and not model_response:
        report["diagnosis_note"] = "Model inference was unavailable, so a heuristic RCA was generated."
    return report


def heuristic_rca(
    signals: list[Any],
    failure_blocks: list[Any],
    retrieved_context: list[dict[str, Any]],
    model_error: str | None = None,
) -> dict[str, Any]:
    _ = retrieved_context
    return normalize_rca_report(None, signals, failure_blocks, model_error=model_error)


def diagnose_workflow_run(
    run_url: str,
    token: str | None = None,
    use_model: bool = True,
    model_id: str | None = None,
) -> dict[str, Any]:
    record = collect_run_from_url(run_url, token=token)
    signals, failure_blocks, _ = preprocess_run_record(record)
    query_parts = [signal.signal_line for signal in signals[:5]]
    query_parts.extend(block.error_message for block in failure_blocks[:3])
    query = "\n".join(query_parts) or run_url
    retrieved_context = retrieve_similar_context(query, top_k=5)

    model_response: dict[str, Any] | None = None
    model_error = None
    if use_model:
        try:
            prompt = build_prompt(run_url, record, signals, failure_blocks, retrieved_context)
            default_model_id = os.getenv("ANTHROPIC_MODEL_ID") or os.getenv("BEDROCK_MODEL_ID") or DEFAULT_MODEL_ID
            text = ClaudeRcaClient(model_id=model_id or default_model_id).complete(prompt)
            model_response = parse_json_response(text) or {"raw_model_response": text}
        except Exception as exc:  # noqa: BLE001 - surface model setup/runtime failures in the UI.
            model_error = str(exc)

    rca = normalize_rca_report(model_response, signals, failure_blocks, model_error=model_error)
    return {
        "run": {
            "repository": record.get("repository"),
            "run_id": record.get("run_id"),
            "workflow_name": record.get("workflow_name"),
            "html_url": record.get("html_url"),
        },
        "signal_count": len(signals),
        "failure_block_count": len(failure_blocks),
        "rca": rca,
    }
