#!/usr/bin/env python3
"""RCA Agent for GitHub Actions CI failures — revised architecture.

Revised RCA pipeline (per REVISED_ARCHITECTURE_PLAN.md §4.1):
  Step 1  Error Extraction       — extract & classify error blocks from the raw log
  Step 2  Data Pre-processing    — PreprocessingPipeline produces structured context
  Step 3  KB Search              — query ChromaDB SelfLearningKnowledgeBase
  Step 4  Decision               — use KB if confidence ≥ threshold, else heuristic RCA
  Step 5  RCA Generation         — Qwen-2.5-13B-Instruct with vectorized CI error,
                                   contextual parameters, and model-tuning parameters
  Step 6  KB Update              — persist new (error_signature → RCA) pair

Model: Qwen/Qwen2.5-14B-Instruct  (via HuggingFace Inference API or local pipeline)

RCA input contract fed to model
────────────────────────────────
{
    "vectorized_error"   : list[float]   — 384-dim L2-normalised error embedding
    "vectorized_context" : list[float]   — 384-dim context embedding
    "error_signature"    : str           — human-readable error signature
    "primary_error"      : dict          — enriched ErrorBlock dict
    "error_blocks"       : list[dict]    — all extracted error blocks
    "kb_context"         : list[dict]    — top-k historical RCA entries from KB
    "retrieved_context"  : list[dict]    — top-k similar log chunks from ChromaDB
    "metadata"           : dict          — CI run / repository metadata
    "model_tuning_params": dict          — temperature, top_p, max_new_tokens …
}

RCA output contract (structured JSON)
───────────────────────────────────────
{
    "rca_summary"              : str
    "failure_stage"            : str
    "failure_type"             : str
    "error_type"               : str
    "error_code"               : str
    "severity"                 : str
    "failure_location"         : dict
    "evidence"                 : list[dict]
    "remediation_steps"        : list[str]
    "inline_fix_suggestions"   : list[dict]
    "verification_commands"    : list[str]
    "confidence"               : str | float
    "source"                   : str   — 'knowledge_base' | 'model' | 'heuristic_fallback'
    "diagnosis_mode"           : str
}
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
DATA_DIR = ROOT_DIR / "data"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from env_loader import load_dotenv          # noqa: E402
from db import ChromaVectorDB, SelfLearningKnowledgeBase, LOG_COLLECTION  # noqa: E402

load_dotenv(ROOT_DIR / ".env")

# ---------------------------------------------------------------------------
# Lazy-load heavy script modules (log-collector, pre-process-pipeline)
# ---------------------------------------------------------------------------

def _load_script_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


log_collector = _load_script_module("smart_ci_log_collector", SCRIPTS_DIR / "log-collector.py")
preprocess    = _load_script_module("smart_ci_preprocess",    SCRIPTS_DIR / "pre-process-pipeline.py")

# ---------------------------------------------------------------------------
# Model constants & helpers
# ---------------------------------------------------------------------------

# Primary: Qwen2.5-14B-Instruct via HuggingFace Inference API
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

DEFAULT_MODEL_INFERENCE_PROVIDER = "featherless-ai"

# Inference provider mapper for selected model
MODEL_INFERENCE_PROVIDER: dict[str, str] = {
    "qwen-13b-instruct":           "featherless-ai",
    "qwen2.5-13b-instruct":        "featherless-ai",
    "qwen2.5-14b-instruct":        "featherless-ai",
    "qwen/qwen2.5-14b-instruct":   "featherless-ai"
}

# Considering KB confidence threshold upto 70%
KB_CONFIDENCE_THRESHOLD = float(os.getenv("KB_CONFIDENCE_THRESHOLD", "0.70"))

# Modification: Use the failure/ error patterns from the ./data/failure_signal_patterns.json
GENERIC_ERROR_TYPES   = {"", "unknown_error", "generic_error"}
LESS_SPECIFIC_ERROR_TYPES = GENERIC_ERROR_TYPES | {"test_failure", "process_exit"}
GENERIC_ERROR_CODES   = {"", "UNKNOWN", "UNKNOWN_ERROR", "TEST_FAILURE", "GENERIC_ERROR"}
SPECIFIC_ERROR_HINTS  = re.compile(
    r"\b(?:403|401|forbidden|unauthorized|permission denied|access denied|token|write access|"
    r"timeout|timed out|connection refused|connection reset|no space left|out of memory|"
    r"module not found|cannot find module|failed to compile|syntax error)\b",
    re.I,
)

# System prompt injected into every Qwen completion request
_SYSTEM_PROMPT = (
    "You are a CI/CD failure root-cause analysis (RCA) agent. "
    "Return ONLY a compact JSON object with these keys: "
    "rca_summary, failure_stage, failure_type, error_code, severity, "
    "failure_location, evidence, remediation_steps, inline_fix_suggestions, "
    "verification_commands, and confidence. "
    "inline_fix_suggestions must be an array of objects each containing "
    "target, suggested_change, and rationale. "
    "Do NOT include raw log text, embeddings, or retrieved_context in the output."
)

# ---------------------------------------------------------------------------
# Qwen inference clients
# ---------------------------------------------------------------------------

class QwenHFClient:
    """
    Call the HuggingFace Inference API for Qwen2.5-14B-Instruct.

    Requires:
        HF_API_KEY  (or MODEL_API_KEY)  in environment / .env
        HF_API_URL  override optional (defaults to HF Inference API endpoint)
    """

    HF_DEFAULT_API_URL = "https://api-inference.huggingface.co/models"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        model_inference_provider: str  = DEFAULT_MODEL_INFERENCE_PROVIDER,
        api_key: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> None:
        self.model_id = (
            os.getenv("MODEL_ID")
            or DEFAULT_MODEL_ID
        )
        self.model_inference_provider = MODEL_INFERENCE_PROVIDER.get(self.model_id.lower(), DEFAULT_MODEL_INFERENCE_PROVIDER)

        self.api_key = os.getenv("HF_API_KEY")
    
        self.api_url = (
            os.getenv("HF_API_URL")
            or f"{self.HF_DEFAULT_API_URL}/{self.model_id}"
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def complete(self, prompt: str, tuning: dict[str, Any] | None = None) -> str:
        """
        Send a chat-completion request to the HF Inference API.

        Parameters
        ----------
        prompt : str
            The user-turn message (already contains the serialised RCA payload).
        tuning : dict, optional
            Per-call model tuning overrides (temperature, top_p, max_new_tokens,
            repetition_penalty).  Merged over instance defaults.

        Returns
        -------
        str
            Raw model text response.
        """
        if not self.api_key:
            raise RuntimeError(
                "HF_API_KEY or MODEL_API_KEY is required for HuggingFace Inference API."
            )

        params = {
            "max_new_tokens":    tuning.get("max_new_tokens",    self.max_new_tokens)    if tuning else self.max_new_tokens,
            "temperature":       tuning.get("temperature",       self.temperature)       if tuning else self.temperature,
            "top_p":             tuning.get("top_p",             self.top_p)             if tuning else self.top_p,
            "repetition_penalty":tuning.get("repetition_penalty",self.repetition_penalty) if tuning else self.repetition_penalty,
            "do_sample":         False,
            "return_full_text":  False,
        }

        if os.getenv("HF_API_URL"):
            body = json.dumps(
                {
                    "model": f"{self.model_id}:{self.model_inference_provider}",
                    "inputs": {
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ]
                    },
                    "parameters": params,
                }
            ).encode("utf-8")
        else: 
            # Chat-template payload (messages format for Qwen instruct models)
            body = json.dumps(
                {
                    "inputs": {
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ]
                    },
                    "parameters": params,
                }
            ).encode("utf-8")

        request = urllib.request.Request(
            self.api_url,
            data=body,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"HuggingFace API error {exc.code}: {error_body}"
            ) from exc

        # HF Inference API returns list[{generated_text: ...}]
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                # chat response: {"generated_text": [{"role":…, "content": …}]}
                gt = first.get("generated_text")
                if isinstance(gt, list):
                    for msg in reversed(gt):
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            return str(msg.get("content", "")).strip()
                return str(gt or "").strip()
        if isinstance(payload, dict):
            return str(payload.get("generated_text", json.dumps(payload))).strip()
        return json.dumps(payload)


class QwenLocalClient:
    """
    Call a locally loaded Qwen2.5-14B-Instruct via HuggingFace transformers.

    Use when MODEL_PROVIDER=local in .env.  Requires:
        pip install transformers torch accelerate
    and enough VRAM/RAM for the chosen quantisation.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        load_in_4bit: bool = False,
    ) -> None:
        self.model_id = MODEL_ALIASES.get(model_id.lower(), model_id)
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.load_in_4bit = load_in_4bit
        self._pipeline = None   # lazy-loaded on first call

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch  # type: ignore[import-not-found]
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load(self) -> None:
        try:
            from transformers import pipeline, BitsAndBytesConfig  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for local inference. "
                "Run: pip install transformers torch accelerate"
            ) from exc

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "task":  "text-generation",
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self._pipeline = pipeline(**kwargs)
        logger.info("Loaded local model: %s on %s", self.model_id, self.device)

    def complete(self, prompt: str, tuning: dict[str, Any] | None = None) -> str:
        if self._pipeline is None:
            self._load()

        messages = [
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {"role": "user",    "content": prompt},
        ]
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens":    (tuning or {}).get("max_new_tokens",    self.max_new_tokens),
            "temperature":       (tuning or {}).get("temperature",       self.temperature),
            "top_p":             (tuning or {}).get("top_p",             self.top_p),
            "repetition_penalty":(tuning or {}).get("repetition_penalty",self.repetition_penalty),
            "do_sample":         False,
            "return_full_text":  False,
        }
        result = self._pipeline(messages, **gen_kwargs)
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                gt = first.get("generated_text")
                if isinstance(gt, list):
                    for msg in reversed(gt):
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            return str(msg.get("content", "")).strip()
                return str(gt or "").strip()
        return str(result)


class QwenRcaClient:
    """
    Provider-routing wrapper for Qwen inference.

    Reads MODEL_PROVIDER from environment:
      'local'  → QwenLocalClient   (transformers pipeline)
      anything else → QwenHFClient  (HuggingFace Inference API, default)
    """

    def __init__(self, model_id: str, max_new_tokens: int = 1024) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

    def complete(self, prompt: str, tuning: dict[str, Any] | None = None) -> str:
        provider = (os.getenv("MODEL_PROVIDER") or "").strip().lower()
        if provider == "local":
            load_4bit = os.getenv("LOAD_IN_4BIT", "").lower() in {"1", "true", "yes"}
            return QwenLocalClient(
                self.model_id,
                max_new_tokens=self.max_new_tokens,
                load_in_4bit=load_4bit,
            ).complete(prompt, tuning=tuning)
        return QwenHFClient(
            self.model_id,
            max_new_tokens=self.max_new_tokens,
        ).complete(prompt, tuning=tuning)


# ---------------------------------------------------------------------------
# Log collection helpers (unchanged interface)
# ---------------------------------------------------------------------------

def collect_run_from_url(
    run_url: str,
    token: str | None = None,
    data_dir: Path = DATA_DIR,
) -> dict[str, Any]:
    run_ref  = log_collector.parse_workflow_run_url(run_url)
    client   = log_collector.GitHubActionsClient(token=token or os.getenv("GITHUB_TOKEN"))
    run      = log_collector.get_workflow_run(client, run_ref.repository, run_ref.run_id)
    jobs     = log_collector.list_jobs_for_run(client, run_ref.repository, run_ref.run_id, max_pages=5)
    log_dir  = data_dir / "logs" / log_collector.safe_repo_name(run_ref.repository)
    download = log_collector.download_run_zip(client, run_ref.repository, run, log_dir, overwrite=True)
    record   = log_collector.compact_run_metadata(run_ref.repository, run, jobs, download, data_dir)
    record   = log_collector.write_run_metadata_sidecar(data_dir, record)
    log_collector.upsert_index(data_dir / "index.json", run_ref.repository, [record])
    return record


# ---------------------------------------------------------------------------
# Step 1 & 2: Error extraction + pre-processing pipeline
# ---------------------------------------------------------------------------

def preprocess_run_record(
    record: dict[str, Any],
    data_dir: Path = DATA_DIR,
    chroma_path: Path | None = None,
) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    """
    Run the data pre-processing pipeline on a collected run zip.

    Returns (signals, failure_blocks, documents) — same interface as before,
    now indexing into ChromaDB instead of SQLite.
    """
    zip_path = preprocess.resolve_zip_path(data_dir, record)
    if not zip_path or not zip_path.exists():
        raise FileNotFoundError(
            f"Missing collected zip for run {record.get('run_id')}: {zip_path}"
        )

    signals, documents, failure_blocks = preprocess.process_zip(
        zip_path=zip_path,
        run_metadata=record,
        context_lines=8,
        max_signals_per_log=100,
        chunk_lines=120,
        chunk_overlap=20,
    )

    db_path = chroma_path or (data_dir / "chroma_store")
    vector_db = ChromaVectorDB(persist_dir=db_path, collection_name=LOG_COLLECTION)
    try:
        vector_db.upsert_documents(documents, collection=LOG_COLLECTION)
    finally:
        vector_db.close()

    return signals, failure_blocks, documents


def extract_processed_context(
    record: dict[str, Any],
    signals: list[Any],
    failure_blocks: list[Any],
) -> dict[str, Any]:
    """
    Step 2 — Use PreprocessingPipeline to produce the structured RCA input dict.

    Assembles a representative log text from the top error signals and failure
    blocks, then runs it through the revised PreprocessingPipeline (5-step),
    which returns the vectorized error, context embeddings, error signature,
    enriched error blocks, and model tuning parameters.
    """
    pipeline = preprocess.PreprocessingPipeline()

    # Build a representative raw log text from top signals/blocks
    log_parts: list[str] = []
    for signal in signals[:10]:
        log_parts.append(signal.context or signal.signal_line)
    for block in failure_blocks[:5]:
        log_parts.append(block.text or block.error_message)

    raw_log = "\n\n".join(log_parts) or "No log content available."

    metadata = {
        "repository":    record.get("repository", ""),
        "run_id":        str(record.get("run_id", "")),
        "workflow_name": record.get("workflow_name", ""),
        "job_name":      "",  # enriched per-signal, not at run level
        "branch":        record.get("branch", ""),
        "commit_sha":    record.get("commit_sha", ""),
        "html_url":      record.get("html_url", ""),
        "trigger":       record.get("event", ""),
        "environment": {
            "os":        "",
            "runner":    "",
            "language":  "",
            "framework": "",
        },
    }

    return pipeline.process_to_rca_input(raw_log, metadata)


# ---------------------------------------------------------------------------
# Step 3: KB search + vector store retrieval
# ---------------------------------------------------------------------------

def retrieve_kb_context(
    error_signature: str,
    top_k: int = 5,
    data_dir: Path = DATA_DIR,
) -> list[dict[str, Any]]:
    """Query the self-learning knowledge base for similar historical RCA entries."""
    kb = SelfLearningKnowledgeBase(persist_dir=data_dir / "rca_knowledge_base")
    entries = kb.search(error_signature, top_k=top_k)
    return [
        {
            "error_signature": e.error_signature,
            "rca_summary":     e.rca_summary,
            "error_type":      e.error_type,
            "confidence":      e.confidence,
            "failure_stage":   e.failure_stage,
            "similarity":      e.similarity,
            "hit_count":       e.hit_count,
        }
        for e in entries
    ]


def retrieve_similar_context(
    query: str,
    top_k: int = 5,
    data_dir: Path = DATA_DIR,
) -> list[dict[str, Any]]:
    """Query ChromaDB for the top-k most similar historical log chunks."""
    vector_db = ChromaVectorDB(
        persist_dir=data_dir / "chroma_store",
        collection_name=LOG_COLLECTION,
    )
    try:
        results = vector_db.search(query, top_k=top_k, collection=LOG_COLLECTION)
    finally:
        vector_db.close()
    return [
        {
            "document_id": r.document_id,
            "score":       round(r.score, 4),
            "text":        r.text,
            "metadata":    r.metadata,
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# KB self-learning update (Step 6)
# ---------------------------------------------------------------------------

def update_knowledge_base(
    error_signature: str,
    rca: dict[str, Any],
    data_dir: Path = DATA_DIR,
) -> None:
    """
    Persist a new (error_signature → RCA) pair to the self-learning KB.

    Called automatically after every successful RCA generation.
    """
    if not error_signature:
        return
    try:
        kb = SelfLearningKnowledgeBase(persist_dir=data_dir / "rca_knowledge_base")
        kb.update(error_signature, rca)
        logger.info("KB updated — signature=%s", error_signature)
    except Exception as exc:  # noqa: BLE001
        logger.warning("KB update failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def truncate_text(value: Any, limit: int = 900) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def line_range(start_line: Any, end_line: Any) -> str:
    if start_line and end_line and start_line != end_line:
        return f"{start_line}-{end_line}"
    return str(start_line) if start_line else ""


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


def infer_error_details(
    text: str, error_type: str, error_code: str, severity: str
) -> tuple[str, str, str]:
    match_payload = preprocess.first_error_match(text)
    inferred_type = preprocess.classify_error(
        text, match_payload[0] if match_payload else "generic_error"
    )
    selected_type = error_type or "unknown_error"
    if inferred_type not in GENERIC_ERROR_TYPES and (
        selected_type in LESS_SPECIFIC_ERROR_TYPES or inferred_type != "test_failure"
    ):
        selected_type = inferred_type

    selected_code = error_code or "UNKNOWN"
    inferred_code = preprocess.error_code_for_line(text, match_payload, selected_type)
    http_match = re.search(r"\b([45]\d{2})\b", text)
    if http_match and any(t in text.lower() for t in ("http", "status", "response")):
        inferred_code = f"HTTP_{http_match.group(1)}"
    if inferred_code and (
        selected_code in GENERIC_ERROR_CODES or selected_type != error_type
    ):
        selected_code = inferred_code

    selected_severity = severity or "low"
    inferred_severity = preprocess.classify_severity(text, selected_type, selected_code)
    if inferred_severity:
        selected_severity = inferred_severity
    if selected_severity == "low" and selected_type in {
        "permission_error", "network_error", "timeout", "resource_error"
    }:
        selected_severity = "medium"
    return selected_type, selected_code, selected_severity


def compact_retrieved_context(
    retrieved_context: list[dict[str, Any]], limit: int = 3
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in retrieved_context[:limit]:
        meta = item.get("metadata") or {}
        compact.append(
            {
                "score":        item.get("score"),
                "failure_type": meta.get("failure_type") or meta.get("error_type"),
                "error_code":   meta.get("error_code"),
                "severity":     meta.get("severity"),
                "location": {
                    "job":        meta.get("job_name"),
                    "log_file":   meta.get("file_name"),
                    "line":       meta.get("line_number"),
                    "line_range": line_range(meta.get("start_line"), meta.get("end_line")),
                },
                "text": truncate_text(item.get("text"), 900),
            }
        )
    return compact


def compact_signal(signal: Any) -> dict[str, Any]:
    return {
        "stage":        signal.section,
        "failure_type": signal.error_type,
        "error_code":   signal.error_code,
        "severity":     signal.severity,
        "location": {
            "job":      signal.job_name,
            "log_file": signal.file_name,
            "line":     signal.line_number,
        },
        "message": truncate_text(signal.signal_line, 500),
        "context": truncate_text(signal.context, 900),
    }


def compact_failure_block(block: Any) -> dict[str, Any]:
    line_number, message = best_evidence_line(block.text)
    return {
        "stage":        block.failure_stage,
        "failure_type": block.failure_type,
        "error_code":   block.error_code,
        "severity":     block.severity,
        "location": {
            "job":        block.job_name,
            "log_file":   block.file_name,
            "line":       line_number or block.start_line,
            "line_range": line_range(block.start_line, block.end_line),
        },
        "message": truncate_text(message or block.error_message, 500),
        "context": truncate_text(block.text, 900),
    }


# ---------------------------------------------------------------------------
# Step 5: Build the structured prompt / RCA model input
# ---------------------------------------------------------------------------

def build_rca_prompt(
    run_url: str,
    record: dict[str, Any],
    signals: list[Any],
    failure_blocks: list[Any],
    retrieved_context: list[dict[str, Any]],
    kb_context: list[dict[str, Any]],
    processed_ctx: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """
    Assemble the full RCA model input payload.

    Returns
    -------
    prompt : str
        Human-readable text prompt sent to the model.
    input_payload : dict
        Full structured payload (vectorized_error, contextual params, tuning params)
        — logged / stored for auditability.
    """
    signal_payload  = [compact_signal(s)       for s in signals[:12]]
    block_payload   = [compact_failure_block(b) for b in failure_blocks[:8]]

    # Contextual parameters (CI/run metadata + error signals)
    contextual_params: dict[str, Any] = {
        "run_url":   run_url,
        "workflow": {
            "repository":    record.get("repository"),
            "run_id":        record.get("run_id"),
            "workflow_name": record.get("workflow_name"),
            "status":        record.get("status"),
            "conclusion":    record.get("conclusion"),
            "branch":        record.get("branch"),
            "commit_sha":    record.get("commit_sha"),
        },
        "error_signals":              signal_payload,
        "failure_blocks":             block_payload,
        "retrieved_similar_context":  compact_retrieved_context(retrieved_context),
        "kb_historical_context":      kb_context[:3],
        "primary_error":              processed_ctx.get("primary_error", {}),
        "error_signature":            processed_ctx.get("error_signature", ""),
        "error_blocks":               processed_ctx.get("error_blocks", [])[:5],
    }

    # Model tuning parameters (Qwen-2.5-14B-Instruct defaults)
    model_tuning_params: dict[str, Any] = processed_ctx.get("model_tuning_params", {
        "temperature":        0.1,
        "top_p":              0.9,
        "max_new_tokens":     1024,
        "repetition_penalty": 1.1,
        "do_sample":          False,
    })

    # Full structured input (for storage/audit)
    input_payload: dict[str, Any] = {
        "vectorized_error":           processed_ctx.get("vectorized_error", []),
        "vectorized_context":         processed_ctx.get("vectorized_context", []),
        "vectorized_combined":        processed_ctx.get("vectorized_combined", []),
        "error_signature":            processed_ctx.get("error_signature", ""),
        "error_signature_hash":       processed_ctx.get("error_signature_hash", ""),
        "contextual_params":          contextual_params,
        "model_tuning_params":        model_tuning_params,
    }

    # Human-readable prompt (embeddings omitted — too large for text prompt)
    prompt = (
        "Diagnose this CI/CD failure from the supplied evidence. "
        "Prefer the first concrete failure over later cascading errors. "
        "If kb_historical_context contains similar failures with high similarity "
        "scores, use them as strong evidence for your RCA. "
        "Return only compact JSON with rca_summary, failure_stage, failure_type, "
        "error_code, severity, failure_location, evidence, remediation_steps, "
        "inline_fix_suggestions, verification_commands, and confidence. "
        "inline_fix_suggestions must be an array of objects with target, "
        "suggested_change, and rationale fields. Make each suggestion small enough "
        "to apply directly in a workflow, dependency file, test, or runtime "
        "configuration. Do not include retrieved_context or raw metadata.\n\n"
        f"{json.dumps(contextual_params, indent=2, sort_keys=True)}"
    )

    return prompt, input_payload


# ---------------------------------------------------------------------------
# Response parsing & normalisation
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end   = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start: end + 1])
            except json.JSONDecodeError:
                return None
    return None


def source_candidate_from_signal(signal: Any) -> dict[str, Any]:
    error_type, error_code, severity = infer_error_details(
        signal.context or signal.signal_line,
        signal.error_type, signal.error_code, signal.severity,
    )
    _, evidence_message = best_evidence_line(signal.context or signal.signal_line)
    return {
        "kind":         "signal",
        "stage":        signal.section,
        "failure_type": error_type,
        "error_code":   error_code,
        "severity":     severity,
        "location": {
            "job":        signal.job_name,
            "log_file":   signal.file_name,
            "line":       signal.line_number,
            "line_range": "",
        },
        "message": evidence_message or signal.signal_line,
        "context": signal.context,
    }


def source_candidate_from_block(block: Any) -> dict[str, Any]:
    line_number, evidence_message = best_evidence_line(block.text or block.error_message)
    error_type, error_code, severity = infer_error_details(
        block.text or block.error_message,
        block.failure_type, block.error_code, block.severity,
    )
    return {
        "kind":         "failure_block",
        "stage":        block.failure_stage,
        "failure_type": error_type,
        "error_code":   error_code,
        "severity":     severity,
        "location": {
            "job":        block.job_name,
            "log_file":   block.file_name,
            "line":       line_number or block.start_line,
            "line_range": line_range(block.start_line, block.end_line),
        },
        "message": evidence_message or block.error_message,
        "context": block.text,
    }


def candidate_score(candidate: dict[str, Any]) -> int:
    text         = f"{candidate.get('message', '')}\n{candidate.get('context', '')}"
    failure_type = str(candidate.get("failure_type") or "")
    error_code   = str(candidate.get("error_code")   or "")
    score = 0
    if candidate.get("kind") == "failure_block":    score += 15
    if failure_type not in GENERIC_ERROR_TYPES:      score += 20
    if failure_type not in LESS_SPECIFIC_ERROR_TYPES: score += 35
    if error_code not in GENERIC_ERROR_CODES:        score += 25
    if SPECIFIC_ERROR_HINTS.search(text):            score += 35
    if re.search(r"\b[45]\d{2}\b", text):            score += 20
    return score


def select_primary_failure(
    signals: list[Any], failure_blocks: list[Any]
) -> dict[str, Any] | None:
    candidates = [source_candidate_from_block(b) for b in failure_blocks]
    candidates.extend(source_candidate_from_signal(s) for s in signals)
    if not candidates:
        return None
    return max(candidates, key=candidate_score)


def compact_evidence(
    primary: dict[str, Any] | None,
    extra_messages: list[str] | None = None,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    if primary:
        message = truncate_text(primary.get("message"), 700)
        if message:
            evidence.append({"line": primary.get("location", {}).get("line"), "message": message})
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


# Modification: May need to move this helper utility along with the fallback_inline_fix_suggestions() function in fix-recommendor.py   
def normalize_inline_fix_suggestions(value: Any) -> list[dict[str, str]]:
    suggestions: list[dict[str, str]] = []
    items = value if isinstance(value, list | tuple) else [value]
    for item in items:
        if isinstance(item, dict):
            target = str(item.get("target") or item.get("file") or item.get("path") or item.get("location") or item.get("area") or "").strip()
            suggested_change = str(item.get("suggested_change") or item.get("fix") or item.get("suggestion") or item.get("change") or item.get("patch") or "").strip()
            rationale = str(item.get("rationale") or item.get("reason") or item.get("why") or "").strip()
        else:
            target = ""
            suggested_change = str(item or "").strip()
            rationale = ""
        if not suggested_change:
            continue
        suggestion: dict[str, str] = {"suggested_change": suggested_change}
        if target:
            suggestion["target"] = target
        if rationale:
            suggestion["rationale"] = rationale
        suggestions.append(suggestion)
    return suggestions


# Modification: Move this to the ./agents/fix-recommendor.py agent as part of fix generation 
def fallback_inline_fix_suggestions(
    failure_type: str,
    error_code: str,
    primary: dict[str, Any] | None,
) -> list[dict[str, str]]:
    location = primary.get("location") if primary else {}
    location_text = ""
    if isinstance(location, dict):
        parts = [str(location.get("job") or ""), str(location.get("log_file") or ""), str(location.get("line") or "")]
        location_text = " / ".join(p for p in parts if p)

    targets = {
        "kubernetes_error":   "Kubernetes manifest, cluster RBAC, image, or runner cluster context",
        "container_error":    "Dockerfile, image build step, registry credentials, or container runtime settings",
        "permission_error":   "Workflow permissions, repository secrets, cloud IAM, or Kubernetes RBAC",
        "timeout":            "Failing workflow step timeout, retry policy, or external dependency call",
        "network_error":      "Runner network, proxy, DNS, TLS, or remote service configuration",
        "resource_error":     "Runner size, disk cleanup step, memory usage, or job parallelism",
        "test_failure":       "Failing test, fixture, assertion, or implementation touched by the test",
        "dependency_error":   "Dependency manifest, lockfile, registry configuration, or cache restore step",
        "build_error":        "Build source file, compiler configuration, or build command",
        "configuration_error":"Workflow YAML, environment variable, referenced path, or secret",
        "process_exit":       "Workflow step command that exited with a non-zero status",
    }
    changes = {
        "kubernetes_error":   "Correct the Kubernetes resource, image, namespace, or RBAC setting referenced by the failure evidence.",
        "container_error":    "Fix the image build/pull configuration and confirm the registry credentials used by the failing step.",
        "permission_error":   "Grant the missing least-privilege access or update the referenced secret/token used by the failing command.",
        "timeout":            "Add retry/backoff for the slow operation or increase the step timeout after confirming the longer runtime is expected.",
        "network_error":      "Validate proxy, DNS, TLS, and remote endpoint settings for the runner, then add retry/backoff around the remote call.",
        "resource_error":     "Reduce parallelism, clean up disk-heavy artifacts before the failing step, or move the job to a larger runner.",
        "test_failure":       "Update the failing assertion, fixture, or implementation after confirming the intended behavior.",
        "dependency_error":   "Pin, update, or restore the failing dependency in the manifest/lockfile and rebuild dependency caches.",
        "build_error":        "Fix the compiler/build error in the referenced source or configuration and rerun the build command locally.",
        "configuration_error":"Correct the missing or invalid workflow input, environment variable, path, or secret referenced by the evidence.",
        "process_exit":       "Make the failing command handle the expected condition or correct the argument/configuration that caused the exit.",
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
            "suggested_change": changes.get(failure_type, "Apply the smallest workflow, code, dependency, or environment change indicated by the failure evidence."),
            "rationale": rationale,
        }
    ]


def failure_summary(error_type: str, error_code: str, evidence_message: str) -> str:
    if error_type == "permission_error" and ("403" in error_code or "403" in evidence_message):
        return (
            "The workflow was denied access while executing the failing step. "
            "The evidence shows an HTTP 403/Forbidden response, so the token or "
            "account does not have the required repository permissions."
        )
    if error_type == "permission_error":
        return "The workflow failed because a required token, secret, path, or remote resource was not accessible."
    return preprocess.summarize_failure(SimpleNamespace(error_type=error_type))


# ---------------------------------------------------------------------------
# Step 5: Formatted RCA output normalisation
# ---------------------------------------------------------------------------

def normalize_rca_report(
    model_response: dict[str, Any] | None,
    signals: list[Any],
    failure_blocks: list[Any],
    model_error: str | None = None,
    kb_context: list[dict[str, Any]] | None = None,
    processed_ctx: dict[str, Any] | None = None,
    source: str = "model",
) -> dict[str, Any]:
    """
    Produce the final formatted RCA output.

    Merges model response with heuristic fallbacks.  If the model returned
    nothing, the heuristic RCA is generated from the extracted signals and
    failure blocks alone.
    """
    primary          = select_primary_failure(signals, failure_blocks)
    fallback_type    = primary.get("failure_type") if primary else "unknown_error"
    fallback_code    = primary.get("error_code")   if primary else "UNKNOWN"
    fallback_severity= primary.get("severity")     if primary else "low"
    evidence_message = str(primary.get("message")  if primary else "")

    model_failure_type = str(model_value(model_response, "failure_type", "error_type") or "")
    failure_type = model_failure_type if model_failure_type not in LESS_SPECIFIC_ERROR_TYPES else fallback_type
    if not failure_type:
        failure_type = fallback_type

    model_error_code = str(model_value(model_response, "error_code") or "")
    error_code = model_error_code if model_error_code not in GENERIC_ERROR_CODES else fallback_code
    if not error_code:
        error_code = fallback_code

    severity = str(model_value(model_response, "severity") or fallback_severity or "low")
    summary  = str(model_value(model_response, "rca_summary", "root_cause", "summary") or "").strip()
    if not summary:
        # Use KB historical context as summary seed if confidence is high
        if kb_context:
            best = max(kb_context, key=lambda e: e.get("similarity", 0), default=None)
            if best and float(best.get("similarity", 0)) >= KB_CONFIDENCE_THRESHOLD:
                summary = best.get("rca_summary", "")
                source  = "knowledge_base"
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
    remediation_steps = (
        as_text_list(model_value(model_response, "remediation_steps"))
        or preprocess.recommended_steps(failure_type)
    )
    inline_fix_suggestions = normalize_inline_fix_suggestions(
        model_value(model_response, "inline_fix_suggestions", "inline_fixes", "fix_suggestions", "suggested_fixes")
    )
    if not inline_fix_suggestions:
        inline_fix_suggestions = fallback_inline_fix_suggestions(failure_type, error_code, primary)
    verification_commands = (
        as_text_list(model_value(model_response, "verification_commands"))
        or ["Re-run the failed workflow after applying the remediation."]
    )

    # Error signature for KB update
    error_signature = ""
    if processed_ctx:
        error_signature = processed_ctx.get("error_signature", "")

    report: dict[str, Any] = {
        "rca_summary":           summary,
        "failure_stage":         str(model_value(model_response, "failure_stage") or (primary.get("stage") if primary else "unknown_stage")),
        "failure_type":          failure_type,
        "error_type":            failure_type,
        "error_code":            error_code,
        "severity":              severity,
        "failure_location":      failure_location,
        "evidence":              compact_evidence(primary, extra_messages),
        "remediation_steps":     remediation_steps,
        "inline_fix_suggestions":inline_fix_suggestions,
        "verification_commands": verification_commands,
        "confidence":            confidence,
        "source":                source,
        "error_signature":       error_signature,
        "diagnosis_mode":        "model" if model_response else "heuristic_fallback",
    }
    if model_error and not model_response:
        report["diagnosis_note"] = "Model inference was unavailable; a heuristic RCA was generated."
    return report


def heuristic_rca(
    signals: list[Any],
    failure_blocks: list[Any],
    retrieved_context: list[dict[str, Any]],
    kb_context: list[dict[str, Any]] | None = None,
    processed_ctx: dict[str, Any] | None = None,
    model_error: str | None = None,
) -> dict[str, Any]:
    _ = retrieved_context
    return normalize_rca_report(
        None, signals, failure_blocks,
        model_error=model_error,
        kb_context=kb_context,
        processed_ctx=processed_ctx,
        source="heuristic_fallback",
    )


# ---------------------------------------------------------------------------
# Main orchestration entry point
# ---------------------------------------------------------------------------

def diagnose_workflow_run(
    run_url: str,
    token: str | None = None,
    use_model: bool = True,
    model_id: str | None = None,
) -> dict[str, Any]:
    """
    End-to-end RCA pipeline for a GitHub Actions workflow run.

    Revised 6-step flow
    ───────────────────
    1. Log collection     — download run zip, extract metadata
    2. Pre-processing     — PreprocessingPipeline (normalise → extract →
                            enrich → signature → embed)
    3. KB search          — query ChromaDB SelfLearningKnowledgeBase
    4. Vector retrieval   — retrieve similar log chunks from ChromaDB
    5. RCA generation     — Qwen-2.5-14B-Instruct with full structured input
    6. KB update          — persist (error_signature → RCA) pair

    Parameters
    ----------
    run_url : str
        GitHub Actions workflow run URL.
    token : str, optional
        GitHub personal access token.
    use_model : bool
        Set False to skip model inference and return heuristic RCA only.
    model_id : str, optional
        Override the default Qwen model ID.

    Returns
    -------
    dict
        Full diagnostic result with run metadata and structured RCA report.
    """

    # Check the input params are empty
    if len(run_url) == 0 :
        print("Job url is missing ... Terminating...")
        exit(1)
    elif token == None:
        # Load the token from the .env file
        token = os.getenv("GITHUB_ACCESS_TOKEN")

    
    print("\n------- Received inputs --------\n")
    print("1. Run URL: ", run_url)
    print("\n2. Git PAT: ", token)
    
    # Step 1 — Log collection
    record = collect_run_from_url(run_url, token=token)

    # Step 2 — Data pre-processing pipeline
    signals, failure_blocks, _ = preprocess_run_record(record)
    processed_ctx = extract_processed_context(record, signals, failure_blocks)

    # Step 3 — KB search (self-learning knowledge base)
    error_signature = processed_ctx.get("error_signature", "")
    kb_context: list[dict[str, Any]] = []
    if error_signature:
        kb_context = retrieve_kb_context(error_signature, top_k=5)

    # Step 4 — Vector store retrieval (similar historical log chunks)
    query_parts = [s.signal_line for s in signals[:5]]
    query_parts.extend(b.error_message for b in failure_blocks[:3])
    query = "\n".join(query_parts) or run_url
    retrieved_context = retrieve_similar_context(query, top_k=5)

    # Step 5 — RCA generation
    model_response: dict[str, Any] | None = None
    model_error:  str | None = None
    source = "model"

    if use_model:
        try:
            resolved_model = (
                model_id
                or os.getenv("QWEN_MODEL_ID")
                or os.getenv("MODEL_ID")
                or DEFAULT_MODEL_ID
            )

            print("\n ----- Model details -----")
            print("model_id: ",resolved_model)
            print("\n")
            # resolved_model = MODEL_ALIASES.get(resolved_model.lower(), resolved_model)

            prompt, _input_payload = build_rca_prompt(
                run_url, record, signals, failure_blocks,
                retrieved_context, kb_context, processed_ctx,
            )

            # Use the QwenHFClient to trigger the request to the HF model
            tuning = processed_ctx.get("model_tuning_params")
            text = QwenRcaClient(model_id=resolved_model).complete(prompt, tuning=tuning)
            model_response = parse_json_response(text) or {"raw_model_response": text}
        except Exception as exc:  # noqa: BLE001
            model_error = str(exc)
            logger.warning("Model inference failed (using heuristic fallback): %s", exc)

    rca = normalize_rca_report(
        model_response, signals, failure_blocks,
        model_error=model_error,
        kb_context=kb_context,
        processed_ctx=processed_ctx,
        source=source if model_response else "heuristic_fallback",
    )

    # Step 6 — KB self-learning update
    if error_signature:
        update_knowledge_base(error_signature, rca)

    return {
        "run": {
            "repository":    record.get("repository"),
            "run_id":        record.get("run_id"),
            "workflow_name": record.get("workflow_name"),
            "html_url":      record.get("html_url"),
        },
        "signal_count":       len(signals),
        "failure_block_count":len(failure_blocks),
        "error_signature":    error_signature,
        "kb_hits":            len(kb_context),
        "rca":                rca,
    }

# Test the RCA workflow run in the main
def main():
    # invoke the diagnose_run_workflow() for functionality checking
    rca_output = diagnose_workflow_run(run_url="https://github.com/AdityaHonkalas/scikit-learn/actions/runs/28492128605", model_id="Qwen/Qwen2.5-7B-Instruct")

    print(rca_output)
    
if __name__  == "__main__":
    main()