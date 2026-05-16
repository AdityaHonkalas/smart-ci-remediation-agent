#!/usr/bin/env python3
"""RCA orchestration for GitHub Actions CI failures."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
DATA_DIR = ROOT_DIR / "data"
DEFAULT_MODEL_ID = "anthropic.claude-sonnet-4-20250514-v1:0"
MODEL_ALIASES = {
    "anthropic.claude-sonnet-4": DEFAULT_MODEL_ID,
    "claude-sonnet-4": DEFAULT_MODEL_ID,
}

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from db import LocalVectorDB  # noqa: E402


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


class ClaudeBedrockClient:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        region_name: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        self.model_id = MODEL_ALIASES.get(model_id, model_id)
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
                        "root_cause, failure_stage, failure_type, error_code, severity, "
                        "evidence, remediation_steps, verification_commands, and confidence."
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


def build_prompt(
    run_url: str,
    record: dict[str, Any],
    signals: list[Any],
    failure_blocks: list[Any],
    retrieved_context: list[dict[str, Any]],
) -> str:
    signal_payload = [preprocess.asdict(signal) for signal in signals[:12]]
    block_payload = [preprocess.asdict(block) for block in failure_blocks[:8]]
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
        "retrieved_similar_context": retrieved_context,
    }
    return (
        "Diagnose this CI/CD failure from the supplied evidence. "
        "Prefer the first concrete failure over later cascading errors. "
        "Return only JSON.\n\n"
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


def heuristic_rca(
    signals: list[Any],
    failure_blocks: list[Any],
    retrieved_context: list[dict[str, Any]],
    model_error: str | None = None,
) -> dict[str, Any]:
    primary = signals[0] if signals else None
    primary_block = failure_blocks[0] if failure_blocks else None
    evidence = []
    if primary:
        evidence.append(primary.signal_line)
    if primary_block and primary_block.text not in evidence:
        evidence.append(primary_block.text[:1200])

    failure_type = primary.error_type if primary else (primary_block.failure_type if primary_block else "unknown_error")
    error_code = primary.error_code if primary else (primary_block.error_code if primary_block else "UNKNOWN")
    severity = primary.severity if primary else (primary_block.severity if primary_block else "low")

    return {
        "root_cause": preprocess.summarize_failure(primary) if primary else "No strong failure signal was extracted from the logs.",
        "failure_stage": primary.section if primary else (primary_block.failure_stage if primary_block else "unknown_stage"),
        "failure_type": failure_type,
        "error_code": error_code,
        "severity": severity,
        "evidence": evidence[:5],
        "remediation_steps": preprocess.recommended_steps(failure_type),
        "verification_commands": ["Re-run the failed workflow after applying the remediation."],
        "confidence": "medium" if primary else "low",
        "retrieved_context": retrieved_context[:3],
        "model_error": model_error,
    }


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
            text = ClaudeBedrockClient(model_id=model_id or os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)).complete(prompt)
            model_response = parse_json_response(text) or {"raw_model_response": text}
        except Exception as exc:  # noqa: BLE001 - surface model setup/runtime failures in the UI.
            model_error = str(exc)

    rca = model_response or heuristic_rca(signals, failure_blocks, retrieved_context, model_error=model_error)
    return {
        "run": {
            "repository": record.get("repository"),
            "run_id": record.get("run_id"),
            "workflow_name": record.get("workflow_name"),
            "html_url": record.get("html_url"),
            "zip_path": record.get("zip_path"),
            "metadata_path": record.get("metadata_path"),
        },
        "signal_count": len(signals),
        "failure_block_count": len(failure_blocks),
        "rca": rca,
    }
