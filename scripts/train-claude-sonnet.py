#!/usr/bin/env python3
"""Prepare CI failure RCA training artifacts for Claude Sonnet on Bedrock.

Claude Sonnet 4 is used as the RCA model through prompting/RAG. Amazon Bedrock
model customization does not list Claude Sonnet 4 as a fine-tunable base model,
so this script creates supervised message examples and a reusable prompt pack.
If you pass a fine-tunable Bedrock model id, the same script can start a Bedrock
customization job.
"""

from __future__ import annotations

import argparse
import os
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from env_loader import load_dotenv


load_dotenv()

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_CLAUDE_SONNET_4_BEDROCK_ID = "anthropic.claude-sonnet-4-20250514-v1:0"
SONNET_4_ALIASES = {
    "anthropic.claude-sonnet-4",
    "claude-sonnet-4",
    DEFAULT_CLAUDE_SONNET_4_BEDROCK_ID,
}

BEDROCK_FINE_TUNABLE_MODELS = {
    "anthropic.claude-3-haiku-20240307-v1:0:200k",
    "amazon.nova-2-lite-v1:0:256k",
    "amazon.nova-lite-v1:0:300k",
    "amazon.nova-micro-v1:0:128k",
}

SYSTEM_PROMPT = """You are a CI/CD failure diagnosis agent. Given workflow metadata, logs, extracted failure blocks, and retrieved similar incidents, return compact JSON with root_cause, failure_stage, failure_type, error_code, severity, evidence, remediation_steps, inline_fix_suggestions, verification_commands, and confidence. inline_fix_suggestions must be an array of objects with target, suggested_change, and rationale fields. Ground every claim in the supplied evidence."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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


def validate_training_records(records: list[dict[str, Any]]) -> dict[str, int]:
    if not records:
        raise ValueError("Training dataset is empty after preprocessing.")

    missing_failure_type = sum(1 for record in records if not record.get("failure_type"))
    missing_error_code = sum(1 for record in records if not record.get("error_code"))
    missing_error_message = sum(1 for record in records if not record.get("error_message"))
    if missing_failure_type == len(records) or missing_error_message == len(records):
        raise ValueError(
            "Training records do not contain usable failure/error columns. "
            "Check the Kaggle column aliases in scripts/pre-process-pipeline.py."
        )

    return {
        "records": len(records),
        "missing_failure_type": missing_failure_type,
        "missing_error_code": missing_error_code,
        "missing_error_message": missing_error_message,
    }


def normalize_model_id(model_id: str) -> str:
    if model_id in SONNET_4_ALIASES:
        return DEFAULT_CLAUDE_SONNET_4_BEDROCK_ID
    return model_id


def as_text_list(value: Any) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def inline_fix_suggestions_for_record(record: dict[str, Any], remediation_steps: list[str]) -> list[dict[str, str]]:
    target = (
        record.get("failure_stage")
        or record.get("failure_type")
        or record.get("output", {}).get("root_cause_category")
        or "failing workflow step"
    )
    error_message = str(record.get("error_message") or "").strip()
    suggested_change = remediation_steps[0] if remediation_steps else (
        "Apply the smallest workflow, code, dependency, or environment change indicated by the failure evidence."
    )
    rationale = "Derived from the labeled CI failure evidence."
    if error_message:
        rationale = f"Evidence: {error_message[:220]}"

    return [
        {
            "target": str(target),
            "suggested_change": suggested_change,
            "rationale": rationale,
        }
    ]


def claude_message_example(record: dict[str, Any]) -> dict[str, Any]:
    remediation_steps = as_text_list(record.get("output", {}).get("recommended_next_steps", []))
    expected = {
        "root_cause": record.get("output", {}).get("summary", ""),
        "failure_stage": record.get("failure_stage", "unknown_stage"),
        "failure_type": record.get("failure_type", record.get("output", {}).get("root_cause_category", "unknown_error")),
        "error_code": record.get("error_code", record.get("output", {}).get("error_code", "UNKNOWN")),
        "severity": record.get("severity", "low"),
        "evidence": record.get("output", {}).get("evidence", []),
        "remediation_steps": remediation_steps,
        "inline_fix_suggestions": record.get("output", {}).get("inline_fix_suggestions")
        or inline_fix_suggestions_for_record(record, remediation_steps),
        "verification_commands": [],
        "confidence": "medium",
    }
    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": record.get("input", "")}]},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(expected, sort_keys=True)}]},
        ],
        "metadata": {
            "source_id": record.get("id"),
            "task": record.get("task", "ci_failure_root_cause_analysis"),
            "label_source": record.get("label_source", "unknown"),
        },
    }


def build_prompt_pack(records: list[dict[str, Any]], model_id: str, few_shot_count: int) -> dict[str, Any]:
    few_shots = [claude_message_example(record) for record in records[:few_shot_count]]
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "base_model": normalize_model_id(model_id),
        "system_prompt": SYSTEM_PROMPT,
        "few_shot_examples": few_shots,
        "output_schema": {
            "root_cause": "string",
            "failure_stage": "string",
            "failure_type": "string",
            "error_code": "string",
            "severity": "low|medium|high",
            "evidence": ["string"],
            "remediation_steps": ["string"],
            "inline_fix_suggestions": [
                {
                    "target": "string",
                    "suggested_change": "string",
                    "rationale": "string",
                }
            ],
            "verification_commands": ["string"],
            "confidence": "low|medium|high",
        },
    }


def build_training_manifest(
    args: argparse.Namespace,
    dataset_path: Path,
    model_id: str,
    validation: dict[str, int],
    messages_count: int,
) -> dict[str, Any]:
    graph_path = args.data_dir.resolve() / args.graph_output
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "base_model": model_id,
        "dataset_path": str(dataset_path),
        "training_iterations": args.epoch_count,
        "records": validation,
        "message_examples": messages_count,
        "knowledge_graph_path": str(graph_path),
        "knowledge_graph_populated": graph_path.exists(),
        "model_training_mode": "bedrock_customization" if args.start_bedrock_job else "claude_sonnet_prompt_rag_artifacts",
        "note": (
            "Claude Sonnet 4 is prepared through prompt/RAG artifacts in this project. "
            "When --start-bedrock-job is used with a fine-tunable Bedrock model, epochCount "
            "is set from training_iterations."
        ),
    }


def start_bedrock_customization_job(args: argparse.Namespace, model_id: str) -> dict[str, Any]:
    if model_id not in BEDROCK_FINE_TUNABLE_MODELS:
        supported = ", ".join(sorted(BEDROCK_FINE_TUNABLE_MODELS))
        raise ValueError(
            f"{model_id} is not in the Bedrock fine-tuning support list used by this project. "
            f"Prepare prompt/RAG artifacts for Claude Sonnet 4, or choose one of: {supported}"
        )

    missing = [
        name
        for name in ("training_s3_uri", "output_s3_uri", "role_arn", "custom_model_name", "job_name")
        if not getattr(args, name)
    ]
    if missing:
        raise ValueError(f"Missing required Bedrock customization arguments: {', '.join(missing)}")

    import boto3  # type: ignore[import-not-found]

    client = boto3.client("bedrock", region_name=args.region)
    return client.create_model_customization_job(
        jobName=args.job_name,
        customModelName=args.custom_model_name,
        roleArn=args.role_arn,
        baseModelIdentifier=model_id,
        hyperParameters={
            "epochCount": str(args.epoch_count),
            "batchSize": str(args.batch_size),
            "learningRateMultiplier": str(args.learning_rate_multiplier),
        },
        trainingDataConfig={"s3Uri": args.training_s3_uri},
        outputDataConfig={"s3Uri": args.output_s3_uri},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Claude Sonnet RCA training artifacts.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset", type=Path, help="Training dataset JSONL. Defaults to data-dir/training_dataset.jsonl.")
    parser.add_argument("--base-model", default=os.getenv("BEDROCK_MODEL_ID") or os.getenv("ANTHROPIC_MODEL_ID") or "anthropic.claude-sonnet-4")
    parser.add_argument("--messages-output", default="claude_training_messages.jsonl")
    parser.add_argument("--prompt-pack-output", default="claude_prompt_pack.json")
    parser.add_argument("--training-manifest-output", default="claude_training_manifest.json")
    parser.add_argument("--few-shot-count", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--graph-output", default="knowledge_graph.json")
    parser.add_argument("--start-bedrock-job", action="store_true")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--training-s3-uri")
    parser.add_argument("--output-s3-uri")
    parser.add_argument("--role-arn")
    parser.add_argument("--custom-model-name")
    parser.add_argument("--job-name")
    parser.add_argument("--epoch-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate-multiplier", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    args.data_dir = data_dir
    dataset_path = args.dataset or (data_dir / "training_dataset.jsonl")
    dataset_path = dataset_path.resolve()
    try:
        if not dataset_path.exists():
            print(f"Missing training dataset: {dataset_path}", file=sys.stderr)
            print("Run scripts/pre-process-pipeline.py first.", file=sys.stderr)
            return 1

        records = read_jsonl(dataset_path)
        if args.max_records:
            records = records[: args.max_records]
        validation = validate_training_records(records)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    model_id = normalize_model_id(args.base_model)
    message_records = [claude_message_example(record) for record in records]
    messages_count = write_jsonl(data_dir / args.messages_output, message_records)
    write_json(data_dir / args.prompt_pack_output, build_prompt_pack(records, model_id, args.few_shot_count))
    write_json(
        data_dir / args.training_manifest_output,
        build_training_manifest(args, dataset_path, model_id, validation, messages_count),
    )

    print(f"Claude message examples: {messages_count}")
    print(f"Training iterations: {args.epoch_count}")
    print(f"Messages JSONL: {data_dir / args.messages_output}")
    print(f"Prompt pack: {data_dir / args.prompt_pack_output}")
    print(f"Training manifest: {data_dir / args.training_manifest_output}")
    print(f"Knowledge graph: {data_dir / args.graph_output}")

    if args.start_bedrock_job:
        response = start_bedrock_customization_job(args, model_id)
        print(json.dumps(response, indent=2, sort_keys=True, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
