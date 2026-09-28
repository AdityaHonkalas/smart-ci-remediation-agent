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

{
    "rca_summary"              : str
    "failure_stage"            : str
    "failure_type"             : str
    "error_type"               : str
    "error_code"               : str
    "severity"                 : str
    "failure_location"         : dict
    "evidence"                 : list[dict]
    "verification_commands"    : list[str]
    "confidence"               : str | float
    "source"                   : str   — 'knowledge_base' | 'model' | 'heuristic_fallback'
    "diagnosis_mode"           : str
    "fixes"                    : dict  — output from FixRecommendationAgent (fixes, total_candidates, …)
}
"""

from __future__ import annotations

from ctypes import Array
import importlib.util
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
# SimpleNamespace removed — was only used by the deleted failure_summary() helper
from typing import Any
import requests

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


log_collector   = _load_script_module("smart_ci_log_collector",   SCRIPTS_DIR / "log-collector.py")
preprocess      = _load_script_module("smart_ci_preprocess",       SCRIPTS_DIR / "pre-process-pipeline.py")
fix_recommendor = _load_script_module("smart_ci_fix_recommendor",  Path(__file__).resolve().parent / "fix-recommendor.py")

# Web Search Agent — imported here so the agents/ package root is on sys.path
_AGENTS_DIR = Path(__file__).resolve().parent
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))
web_search_agent_module = _load_script_module(
    "smart_ci_web_search_agent", _AGENTS_DIR / "web-search-agent.py"
)

# ---------------------------------------------------------------------------
# Model constants & helpers
# ---------------------------------------------------------------------------

'''
 Will follow the hybrid approach for using model for RCA where,
  
 1. For local development will use llama3.1:8b model
 2. For final evaluation will use Claude Sonnet/ Qwen2.5-8B model
'''
# Primary: Qwen2.5-14B-Instruct via HuggingFace Inference API
DEFAULT_MODEL_ID = "llama3.1:8b"
QWEN_MODEL = "qwen3.5:9b"


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
    "You are an expert CI/CD failure root-cause analysis (RCA) engineer with "
    "deep knowledge of GitHub Actions, build systems, dependency managers, "
    "container runtimes, and cloud infrastructure.\n\n"
    "Your task is to analyse the provided GitHub Actions failure evidence and "
    "produce a thorough, actionable diagnostic report.\n\n"
    "Before writing the JSON, reason through these steps:\n"
    "  1. Identify which error in error_signals / failure_blocks is the TRUE ROOT CAUSE "
    "(not a cascading side-effect). Root causes have the highest rca_weight / "
    "candidate_score and appear earliest in the workflow execution.\n"
    "  2. Trace the full causal chain: what triggered the root error, what downstream "
    "failures it caused, and why those are symptoms not causes.\n"
    "  3. Identify the exact file, line, configuration key, or environment setting "
    "that is the point of failure.\n"
    "  4. Determine the minimum remediation action required to fix the root cause.\n"
    "  5. Assess confidence based on the quality and specificity of the available evidence.\n\n"
    "Output format: return a single JSON object with EXACTLY these keys:\n\n"
    "  rca_summary  — string (REQUIRED, minimum 4 complete sentences):\n"
    "                   Sentence 1: What failed and in which workflow job/step.\n"
    "                   Sentence 2: The root cause — WHY it failed (not just what).\n"
    "                   Sentence 3: The causal chain — how the root error propagated "
    "to any downstream failures.\n"
    "                   Sentence 4+: What the key evidence shows and what needs to change.\n"
    "                   Do NOT write a one-line label. "
    "Do NOT copy raw log lines verbatim — synthesise them.\n\n"
    "  failure_stage  — string: the workflow stage/job where root failure occurred\n"
    "  failure_type   — string: e.g. dependency_error, permission_error, build_error, "
    "test_failure, configuration_error, network_error, container_error\n"
    "  error_code     — string: specific exit code, HTTP status, or error identifier\n"
    "  severity       — string: 'critical' | 'high' | 'medium' | 'low'\n"
    "  failure_location — object: {job, log_file, line, line_range}\n"
    "  evidence       — array of 3-6 objects: "
    "{line: int_or_null, message: str, significance: str}. "
    "Each 'significance' must explain WHY this line proves the root cause.\n"
    "  verification_commands — array of strings: actionable shell commands specific "
    "to this error (not generic placeholders)\n"
    "  confidence     — string: 'high' | 'medium' | 'low'\n\n"
    "Rules:\n"
    "  - rca_summary MUST be at least 4 complete sentences.\n"
    "  - evidence MUST reference specific lines/messages from error_signals or failure_blocks.\n"
    "  - Do NOT include raw embeddings, retrieved_context blobs, or vectorized fields.\n"
    "  - If kb_historical_context has similarity >= 0.70, reference it in rca_summary.\n"
    "  - Output ONLY the JSON object — no prose, no markdown fences, nothing before or after."
)


class LLMClientInterface:
    """
    Generic LLM interface for local Ollama-served models.

    Required pre-requisite:
      1. Install Ollama (https://ollama.com) and the Python requests package.
      2. Pull the target model:  ollama pull llama3.1:8b

    Llama 3.1 8B — fixed model architecture (not tuneable at inference time)
    ─────────────────────────────────────────────────────────────────────────
    Architecture type  : Decoder-only transformer (Meta Llama 3.1)
    Transformer layers : 32
    Hidden size        : 4 096
    Attention heads    : 32  (query), 8  (key/value — grouped-query attention)
    Feed-forward dim   : 14 336
    Activation function: SwiGLU  (Swish-gated linear unit — a smooth gating
                         variant of ReLU used in all Llama 3.x models)
    Context length     : 128 000 tokens
    Vocabulary size    : 128 256 tokens
    Parameters         : ~8.03 B

    Tuneable inference parameters (Ollama option names)
    ────────────────────────────────────────────────────
    num_predict    (int,   default 1024)  — max tokens the model will generate
    temperature    (float, default 0.1 ) — sampling temperature; 0 = greedy/deterministic
    top_p          (float, default 0.85) — nucleus sampling probability mass
    top_k          (int,   default 40  ) — sample from the top-k highest-prob tokens
    repeat_penalty (float, default 1.1 ) — penalise recently used tokens to reduce
                                           repetition (1.0 = no penalty)

    Default LLM options:
        DEFAULT_MODEL_ID = "llama3.1:8b"
        DEFAULT_API_URL  = "http://localhost:11434/api/chat"
    """

    DEFAULT_MODEL_ID = "llama3.1:8b"
    DEFAULT_API_URL  = "http://localhost:11434/api/chat"

    def __init__(
        self,
        model_id: str = "gemma4:latest",
        api_url: str = DEFAULT_API_URL,
        api_key: str | None = None,
        num_predict: int = 3000,
        temperature: float = 0.25,
        top_k: int = 50,
        top_p: float = 0.90,
        repeat_penalty: float = 1.15,
    ) -> None:
        self.model_id = (
            "gemma4:latest"
            or os.getenv("MODEL_ID")
            or self.DEFAULT_MODEL_ID
        )

        # Use the ollama API key to access ollama models
        self.api_key = os.getenv("OLLAMA_API_KEY")

        # Use the ollama api endpoint
        self.api_url = (
            os.getenv("OLLAMA_API_URL")
            or self.DEFAULT_API_URL
        )

        self.num_predict    = num_predict
        self.temperature    = temperature
        self.top_p          = top_p
        self.top_k          = top_k
        self.repeat_penalty = repeat_penalty

    def model_api_request(self, prompt: str, tuning: dict[str, Any] | None = None) -> str:
        """
        Send an API request to the LLM model via the Ollama /api/chat endpoint.

        Parameters
        ----------
        prompt : str
            The user-turn message (already contains the serialised RCA payload).
        tuning : dict, optional
            Per-call model tuning overrides.  Accepted keys (Ollama option names):
              num_predict, temperature, top_p, top_k, repeat_penalty.
            Legacy HuggingFace key names are also accepted as fallbacks:
              max_new_tokens  → num_predict
              repetition_penalty → repeat_penalty

        Returns
        -------
        str
            Raw model text response.
        """
        tuning = tuning or {}
        params = {
            # Accept both Ollama-native keys and legacy HuggingFace key names
            "num_predict":    tuning.get("num_predict",
                              tuning.get("max_new_tokens",    self.num_predict)),
            "temperature":    tuning.get("temperature",       self.temperature),
            "top_p":          tuning.get("top_p",             self.top_p),
            "top_k":          tuning.get("top_k",             self.top_k),
            "repeat_penalty": tuning.get("repeat_penalty",
                              tuning.get("repetition_penalty", self.repeat_penalty)),
        }

        print("\n-------- Tuning params ----------\n")
        print(params)
        print("\n-------------\n")
        
        body = {
            "model": self.model_id,
            "messages": [
                # System prompt establishes the RCA agent persona and output contract.
                # Ollama /api/chat supports the system role for llama3.1:8b natively.
                # Previously _SYSTEM_PROMPT was defined but never sent — fixed here.
                {
                    "role":    "system",
                    "content": _SYSTEM_PROMPT,
                },
                {
                    "role":    "user",
                    "content": prompt,
                },
            ],
            "stream": False,
            "options": params,
        }

        # request = urllib.request.Request(
        #     self.api_url,
        #     data=body,
        #     headers={
        #         "Content-Type":  "application/json",
        #     },
        #     method="POST",
        # )

        print("\n======== Request object =============")
        print(self.api_url)
        print(self.model_id)
        print("\n===============================\n")
        
        # try:
        #     with urllib.request.urlopen(request, timeout=300) as resp:
        #         print("\n---------- Waiting for response ---------------\n")
        #         if resp:
        #             print("Response ready: ", resp)
        #         else:
        #             print("Reponse not ready...")
        #         print("-------------------------------------------------\n")
        #         payload = json.loads(resp.read().decode("utf-8"))
        # except urllib.error.HTTPError as exc:
        #     error_body = exc.read().decode("utf-8", errors="replace")
        #     raise RuntimeError(
        #         f"API error {exc.code}: {error_body}"
        #     ) from exc
        # except urllib.error.URLError as exc:
        #     raise RuntimeError(
        #         f"Cannot reach Ollama at {self.api_url}: {exc.reason}"
        #     ) from exc

        # Replacing the above block with simple POST request

        try:
            response = requests.post(
                url=self.api_url,
                json=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}", 
                    "Content-Type": "application/json" 
                },
                timeout=300
            )
        except Exception as e:
            print("\n------- Exception Occurred --------")
            print("Model request failed due to: ",e)
            print("\n")
            raise

        
        response.raise_for_status()


        return response.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Qwen inference clients
# ---------------------------------------------------------------------------
# class QwenHFClient:
#     """
#     Call the HuggingFace Inference API for Qwen2.5-14B-Instruct.

#     Requires:
#         HF_API_KEY  (or MODEL_API_KEY)  in environment / .env
#         HF_API_URL  override optional (defaults to HF Inference API endpoint)
#     """

#     HF_DEFAULT_API_URL = "https://api-inference.huggingface.co/models"

#     def __init__(
#         self,
#         model_id: str = DEFAULT_MODEL_ID,
#         api_key: str | None = None,
#         max_new_tokens: int = 1024,
#         temperature: float = 0.1,
#         top_p: float = 0.9,
#         repetition_penalty: float = 1.1,
#     ) -> None:
#         self.model_id = (
#             model_id
#             or os.getenv("MODEL_ID")
#             or DEFAULT_MODEL_ID
#         )

#         self.api_key = os.getenv("HF_API_KEY")
    
#         self.api_url = (
#             os.getenv("API_URL")
#             or f"{self.HF_DEFAULT_API_URL}/{self.model_id}"
#         )
#         self.max_new_tokens = max_new_tokens
#         self.temperature = temperature
#         self.top_p = top_p
#         self.repetition_penalty = repetition_penalty

#     def complete(self, prompt: str, tuning: dict[str, Any] | None = None) -> str:
#         """
#         Send a chat-completion request to the HF Inference API.

#         Parameters
#         ----------
#         prompt : str
#             The user-turn message (already contains the serialised RCA payload).
#         tuning : dict, optional
#             Per-call model tuning overrides (temperature, top_p, max_new_tokens,
#             repetition_penalty).  Merged over instance defaults.

#         Returns
#         -------
#         str
#             Raw model text response.
#         """
#         if not self.api_key:
#             raise RuntimeError(
#                 "HF_API_KEY or MODEL_API_KEY is required for HuggingFace Inference API."
#             )

#         params = {
#             "max_new_tokens":    tuning.get("max_new_tokens",    self.max_new_tokens)    if tuning else self.max_new_tokens,
#             "temperature":       tuning.get("temperature",       self.temperature)       if tuning else self.temperature,
#             "top_p":             tuning.get("top_p",             self.top_p)             if tuning else self.top_p,
#             "repetition_penalty":tuning.get("repetition_penalty",self.repetition_penalty) if tuning else self.repetition_penalty,
#             "do_sample":         False,
#             "return_full_text":  False,
#         }

        
#         body = json.dumps(
#             {
#                 "model": f"{self.model_id}:fastest",
#                 "messages": [
#                     {
#                         "role": "user",
#                         "content": prompt
#                     },
#                 ],
#                 "max_new_tokens":    tuning.get("max_new_tokens",    self.max_new_tokens)    if tuning else self.max_new_tokens,
#                 "temperature":       tuning.get("temperature",       self.temperature)       if tuning else self.temperature
#             }
#         ).encode("utf-8")

#         request = urllib.request.Request(
#             self.api_url,
#             data=body,
#             headers={
#                 "Content-Type":  "application/json",
#                 "Authorization": f"Bearer {self.api_key}",
#             },
#             method="POST",
#         )

#         print("\n-------- Request params ---------")
#         print(self.api_url)
#         print(self.api_key)

#         try:
#             with urllib.request.urlopen(request, timeout=120) as resp:
#                 payload = json.loads(resp.read().decode("utf-8"))
#         except urllib.error.HTTPError as exc:
#             error_body = exc.read().decode("utf-8", errors="replace")
#             raise RuntimeError(
#                 f"HuggingFace API error {exc.code}: {error_body}"
#             ) from exc

#         # HF Inference API returns list[{generated_text: ...}]
#         if isinstance(payload, list) and payload:
#             first = payload[0]
#             if isinstance(first, dict):
#                 # chat response: {"generated_text": [{"role":…, "content": …}]}
#                 gt = first.get("generated_text")
#                 if isinstance(gt, list):
#                     for msg in reversed(gt):
#                         if isinstance(msg, dict) and msg.get("role") == "assistant":
#                             return str(msg.get("content", "")).strip()
#                 return str(gt or "").strip()
#         if isinstance(payload, dict):
#             return str(payload.get("generated_text", json.dumps(payload))).strip()
#         return json.dumps(payload)


# class QwenLocalClient:
#     """
#     Call a locally loaded Qwen2.5-14B-Instruct via HuggingFace transformers.

#     Use when MODEL_PROVIDER=local in .env.  Requires:
#         pip install transformers torch accelerate
#     and enough VRAM/RAM for the chosen quantisation.
#     """

#     def __init__(
#         self,
#         model_id: str = DEFAULT_MODEL_ID,
#         device: str | None = None,
#         max_new_tokens: int = 1024,
#         temperature: float = 0.1,
#         top_p: float = 0.9,
#         repetition_penalty: float = 1.1,
#         load_in_4bit: bool = False,
#     ) -> None:
#         self.model_id = MODEL_ALIASES.get(model_id.lower(), model_id)
#         self.device = device or ("cuda" if self._cuda_available() else "cpu")
#         self.max_new_tokens = max_new_tokens
#         self.temperature = temperature
#         self.top_p = top_p
#         self.repetition_penalty = repetition_penalty
#         self.load_in_4bit = load_in_4bit
#         self._pipeline = None   # lazy-loaded on first call

#     @staticmethod
#     def _cuda_available() -> bool:
#         try:
#             import torch  # type: ignore[import-not-found]
#             return torch.cuda.is_available()
#         except ImportError:
#             return False

#     def _load(self) -> None:
#         try:
#             from transformers import pipeline, BitsAndBytesConfig  # type: ignore[import-not-found]
#             import torch  # type: ignore[import-not-found]
#         except ImportError as exc:
#             raise RuntimeError(
#                 "transformers and torch are required for local inference. "
#                 "Run: pip install transformers torch accelerate"
#             ) from exc

#         kwargs: dict[str, Any] = {
#             "model": self.model_id,
#             "task":  "text-generation",
#             "device_map": "auto",
#             "torch_dtype": torch.float16,
#         }
#         if self.load_in_4bit:
#             kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

#         self._pipeline = pipeline(**kwargs)
#         logger.info("Loaded local model: %s on %s", self.model_id, self.device)

#     def complete(self, prompt: str, tuning: dict[str, Any] | None = None) -> str:
#         if self._pipeline is None:
#             self._load()

#         messages = [
#             {"role": "system",  "content": _SYSTEM_PROMPT},
#             {"role": "user",    "content": prompt},
#         ]
#         gen_kwargs: dict[str, Any] = {
#             "max_new_tokens":    (tuning or {}).get("max_new_tokens",    self.max_new_tokens),
#             "temperature":       (tuning or {}).get("temperature",       self.temperature),
#             "top_p":             (tuning or {}).get("top_p",             self.top_p),
#             "repetition_penalty":(tuning or {}).get("repetition_penalty",self.repetition_penalty),
#             "do_sample":         False,
#             "return_full_text":  False,
#         }
#         result = self._pipeline(messages, **gen_kwargs)
#         if isinstance(result, list) and result:
#             first = result[0]
#             if isinstance(first, dict):
#                 gt = first.get("generated_text")
#                 if isinstance(gt, list):
#                     for msg in reversed(gt):
#                         if isinstance(msg, dict) and msg.get("role") == "assistant":
#                             return str(msg.get("content", "")).strip()
#                 return str(gt or "").strip()
#         return str(result)


# class QwenRcaClient:
#     """
#     Provider-routing wrapper for Qwen inference.

#     Reads MODEL_PROVIDER from environment:
#       'local'  → QwenLocalClient   (transformers pipeline)
#       anything else → QwenHFClient  (HuggingFace Inference API, default)
#     """

#     def __init__(self, model_id: str, max_new_tokens: int = 1024) -> None:
#         self.model_id = model_id
#         self.max_new_tokens = max_new_tokens

#     def complete(self, prompt: str, tuning: dict[str, Any] | None = None) -> str:
#         provider = (os.getenv("MODEL_PROVIDER") or "").strip().lower()
#         if provider == "local":
#             load_4bit = os.getenv("LOAD_IN_4BIT", "").lower() in {"1", "true", "yes"}
#             return QwenLocalClient(
#                 self.model_id,
#                 max_new_tokens=self.max_new_tokens,
#                 load_in_4bit=load_4bit,
#             ).complete(prompt, tuning=tuning)
#         return QwenHFClient(
#             self.model_id,
#             max_new_tokens=self.max_new_tokens,
#         ).complete(prompt, tuning=tuning)


# ---------------------------------------------------------------------------
# Log collection helpers (unchanged interface)
# ---------------------------------------------------------------------------

def collect_run_from_url(
    run_url: str,
    token: str | None = None,
    data_dir: Path = DATA_DIR,
) -> dict[str, Any]:

    print("\n---------- Step-1: Collect log from the URL --------------")
    run_ref  = log_collector.parse_workflow_run_url(run_url)
    client   = log_collector.GitHubActionsClient(token=token or os.getenv("GITHUB_TOKEN"))
    run      = log_collector.get_workflow_run(client, run_ref.repository, run_ref.run_id)
    jobs     = log_collector.list_jobs_for_run(client, run_ref.repository, run_ref.run_id, max_pages=5)
    log_dir  = data_dir / "logs" / log_collector.safe_repo_name(run_ref.repository)
    download = log_collector.download_run_zip(client, run_ref.repository, run, log_dir, overwrite=True)

    # Fetch the dedicated log for the failed job so the pre-processing pipeline
    # can target error extraction at the job that actually failed rather than
    # scanning every job log in the run-level archive.
    failed_job_download = log_collector.download_failed_job_log(
        client, run_ref.repository, jobs, log_dir, overwrite=True
    )
    if failed_job_download.error:
        print(f"[warn] Failed job log download: {failed_job_download.error}")
    else:
        print(f"[info] Failed job log fetched: {failed_job_download.zip_path}")

    record = log_collector.compact_run_metadata(
        run_ref.repository, run, jobs, download, data_dir,
        failed_job_download=failed_job_download,
    )
    record  = log_collector.write_run_metadata_sidecar(data_dir, record)
    log_collector.upsert_index(data_dir / "index.json", run_ref.repository, [record])
    print("Failed action run log", record)
    print("\n-------------------------------------------------------\n")
    return record


# ---------------------------------------------------------------------------
# Step 1 & 2: Error extraction + pre-processing pipeline
# ---------------------------------------------------------------------------

def _resolve_failed_job_zip_path(data_dir: Path, record: dict[str, Any]) -> Path | None:
    """Return the absolute path of the failed-job zip if it was collected."""
    rel = record.get("failed_job_zip_path")
    if not rel:
        return None
    path = Path(rel)
    if not path.is_absolute():
        path = data_dir / path
    return path if path.exists() else None


def preprocess_run_record(
    record: dict[str, Any],
    data_dir: Path = DATA_DIR,
    chroma_path: Path | None = None,
) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    """
    Run the data pre-processing pipeline on a collected run zip.

    When a failed-job zip is present in the record (collected by
    :func:`collect_run_from_url`) the pipeline processes that targeted log
    first to produce high-fidelity error signals for the failed job.  The
    full run-level zip is then processed to supply broad context documents
    for the vector store.  If no failed-job zip is available the pipeline
    falls back to processing only the run-level zip.

    Returns (signals, failure_blocks, documents) — same interface as before,
    now indexing into ChromaDB instead of SQLite.
    """
    run_zip_path = preprocess.resolve_zip_path(data_dir, record)
    failed_job_zip_path = _resolve_failed_job_zip_path(data_dir, record)

    # We need at least one zip to work with.
    if not failed_job_zip_path and (not run_zip_path or not run_zip_path.exists()):
        raise FileNotFoundError(
            f"Missing collected zip for run {record.get('run_id')}: {run_zip_path}"
        )

    signals:        list[Any]          = []
    failure_blocks: list[Any]          = []
    documents:      list[dict[str, Any]] = []

    # --- Primary: failed-job log (targeted, high signal-to-noise) ---
    if failed_job_zip_path:
        print(f"[info] Processing failed job log: {failed_job_zip_path}")
        job_signals, job_docs, job_blocks = preprocess.process_zip(
            zip_path=failed_job_zip_path,
            run_metadata=record,
            context_lines=8,
            max_signals_per_log=100,
            chunk_lines=120,
            chunk_overlap=20,
        )
        signals.extend(job_signals)
        failure_blocks.extend(job_blocks)
        documents.extend(job_docs)

    # --- Secondary: full run-level archive (broad context for vector store) ---
    if run_zip_path and run_zip_path.exists():
        print(f"[info] Processing run-level log archive: {run_zip_path}")
        run_signals, run_docs, run_blocks = preprocess.process_zip(
            zip_path=run_zip_path,
            run_metadata=record,
            context_lines=8,
            max_signals_per_log=100,
            chunk_lines=120,
            chunk_overlap=20,
        )
        # Merge: if the failed-job zip already produced signals, keep those as
        # the authoritative source and only append the run-level docs for
        # vector-store context (skip duplicate signals / blocks).
        if not signals:
            signals.extend(run_signals)
            failure_blocks.extend(run_blocks)
        documents.extend(run_docs)

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

def queue_rca_for_kb_update(
    error_signature: str,
    rca: dict[str, Any],
    data_dir: Path = DATA_DIR,
) -> None:
    """
    Buffer a new (error_signature → RCA) pair to the offline KB staging queue.

    Replaces the former inline ``update_knowledge_base`` call so the live RCA
    pipeline is not blocked by ChromaDB writes.  The staging file is consumed
    by ``scripts/kb_batch_update.py`` (offline) or via ``POST /api/admin/kb-update``.
    """
    if not error_signature:
        return
    import threading

    staging_path = data_dir / "rca_staging_queue.json"
    entry = {
        "error_signature": error_signature,
        "rca":             rca,
        "queued_at":       __import__("datetime").datetime.utcnow().replace(
                               microsecond=0).isoformat() + "Z",
    }
    _staging_lock = getattr(queue_rca_for_kb_update, "_lock", None)
    if _staging_lock is None:
        _staging_lock = threading.Lock()
        queue_rca_for_kb_update._lock = _staging_lock  # type: ignore[attr-defined]

    with _staging_lock:
        try:
            staging_path.parent.mkdir(parents=True, exist_ok=True)
            existing: list[dict] = []
            if staging_path.exists():
                try:
                    with staging_path.open("r", encoding="utf-8") as fh:
                        existing = json.load(fh)
                    if not isinstance(existing, list):
                        existing = []
                except (json.JSONDecodeError, OSError):
                    existing = []
            existing.append(entry)
            tmp = staging_path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(existing, fh, indent=2, default=str)
            tmp.replace(staging_path)
            logger.info("RCA queued for KB batch update — signature=%s", error_signature)
        except Exception as exc:  # noqa: BLE001
            logger.warning("KB staging queue write failed (non-fatal): %s", exc)


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
        "rca_weight":   getattr(signal, "rca_weight", 0),
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
        "rca_weight":   getattr(block, "rca_weight", 0),
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

    Signals and failure blocks are sorted by their candidate_score (which
    includes severity) before capping — so the top-N fed to the model are
    always the highest-quality evidence items.  KB context entries are sorted
    by similarity descending so the most relevant historical RCA appears first.

    Returns
    -------
    prompt : str
        Human-readable text prompt sent to the model.
    input_payload : dict
        Full structured payload (vectorized_error, contextual params, tuning params)
        — logged / stored for auditability.
    """
    # Sort by candidate_score before capping — highest-quality evidence first
    sorted_signals = sorted(
        signals,
        key=lambda s: candidate_score(source_candidate_from_signal(s)),
        reverse=True,
    )
    sorted_blocks = sorted(
        failure_blocks,
        key=lambda b: candidate_score(source_candidate_from_block(b)),
        reverse=True,
    )
    # Sort KB context by similarity descending so best historical match is first
    sorted_kb = sorted(
        kb_context,
        key=lambda e: float(e.get("similarity") or 0),
        reverse=True,
    )

    signal_payload = [compact_signal(s)         for s in sorted_signals[:12]]
    block_payload  = [compact_failure_block(b)   for b in sorted_blocks[:8]]

    # error_blocks from preprocessing are already sorted by rca_weight; take top-5
    sorted_error_blocks = sorted(
        processed_ctx.get("error_blocks", []),
        key=lambda b: int(b.get("rca_weight", 0)) if isinstance(b, dict) else 0,
        reverse=True,
    )

    # Contextual parameters (CI/run metadata + error signals)
    contextual_params: dict[str, Any] = {
        "run_url": run_url,
        "workflow": {
            "repository":    record.get("repository"),
            "run_id":        record.get("run_id"),
            "workflow_name": record.get("workflow_name"),
            "status":        record.get("status"),
            "conclusion":    record.get("conclusion"),
            "branch":        record.get("branch"),
            "commit_sha":    record.get("commit_sha"),
        },
        # Signals ranked by candidate_score — highest scoring first
        "error_signals":   signal_payload,
        # Failure blocks ranked by candidate_score — root cause first
        "failure_blocks":  block_payload,
        "retrieved_similar_context": compact_retrieved_context(retrieved_context),
        # KB entries ranked by similarity — most relevant historical RCA first
        "kb_historical_context":     sorted_kb[:3],
        "primary_error":    processed_ctx.get("primary_error", {}),
        "error_signature":  processed_ctx.get("error_signature", ""),
        # Preprocessor error_blocks already sorted by rca_weight
        "error_blocks":     sorted_error_blocks[:5],
    }

    # Model tuning parameters — Ollama-native key names for llama3.1:8b.
    # See rca-enhancement-plan.md §Parameter Tuning for full rationale.
    raw_tuning: dict[str, Any] = processed_ctx.get("model_tuning_params", {})
    model_tuning_params: dict[str, Any] = {
        "num_predict":    raw_tuning.get("num_predict",
                          raw_tuning.get("max_new_tokens",     3000)),
        "temperature":    raw_tuning.get("temperature",        0.25),
        "top_p":          raw_tuning.get("top_p",              0.90),
        "top_k":          raw_tuning.get("top_k",              50),
        "repeat_penalty": raw_tuning.get("repeat_penalty",
                          raw_tuning.get("repetition_penalty", 1.15)),
    }

    # Full structured input (for storage/audit)
    input_payload: dict[str, Any] = {
        "vectorized_error":    processed_ctx.get("vectorized_error",    []),
        "vectorized_context":  processed_ctx.get("vectorized_context",  []),
        "vectorized_combined": processed_ctx.get("vectorized_combined", []),
        "error_signature":     processed_ctx.get("error_signature",     ""),
        "error_signature_hash":processed_ctx.get("error_signature_hash",""),
        "contextual_params":   contextual_params,
        "model_tuning_params": model_tuning_params,
    }

    # ------------------------------------------------------------------
    # Structured user-turn prompt (ST-3: chain-of-thought scaffolding)
    # ------------------------------------------------------------------

    # Section 1 — Primary error anchor (highest rca_weight = most causal)
    primary_error_raw = processed_ctx.get("primary_error") or {}
    if primary_error_raw and isinstance(primary_error_raw, dict):
        _pe_parts = [
            f"  error_type   : {primary_error_raw.get('error_type') or primary_error_raw.get('failure_type', 'unknown')}",
            f"  error_code   : {primary_error_raw.get('error_code', 'N/A')}",
            f"  severity     : {primary_error_raw.get('severity', 'unknown')}",
            f"  message      : {str(primary_error_raw.get('error_message') or primary_error_raw.get('message', ''))[:300]}",
        ]
        _loc = primary_error_raw.get("location") or {}
        if isinstance(_loc, dict) and (_loc.get("job") or _loc.get("log_file")):
            _pe_parts.append(
                f"  location     : job={_loc.get('job','?')}  "
                f"file={_loc.get('log_file','?')}  line={_loc.get('line','?')}"
            )
        primary_error_section = (
            "## PRIMARY ERROR (highest rca_weight — focus your root-cause reasoning here)\n"
            + "\n".join(_pe_parts)
        )
    else:
        primary_error_section = (
            "## PRIMARY ERROR\n"
            "  (No isolated primary error block — use the top entry in error_signals "
            "and failure_blocks below as the root cause anchor.)"
        )

    # Section 2 — KB historical match (show explicitly if high similarity)
    kb_section = ""
    if sorted_kb:
        best_kb = sorted_kb[0]
        best_sim = float(best_kb.get("similarity") or 0.0)
        if best_sim >= 0.50:
            kb_section = (
                f"\n## HISTORICAL KB MATCH (similarity={best_sim:.2f})\n"
                f"A similar failure was previously diagnosed:\n"
                f"  error_signature : {best_kb.get('error_signature', 'N/A')}\n"
                f"  previous_rca    : {str(best_kb.get('rca_summary', ''))[:400]}\n"
                f"  source          : {best_kb.get('source', 'N/A')}\n"
                + (
                    "Use this as strong supporting evidence — the same root cause likely applies.\n"
                    if best_sim >= KB_CONFIDENCE_THRESHOLD else
                    "Use this as weak supporting context only — similarity is below the high-confidence threshold.\n"
                )
            )

    # Section 3 — Evidence ordering guidance
    evidence_guidance = (
        "\n## EVIDENCE ORDERING\n"
        "error_signals and failure_blocks are sorted by candidate_score (highest first).\n"
        "The FIRST entry in each list is the most likely root cause.\n"
        "Later entries are likely cascading failures triggered by the root cause.\n"
        "Anchor your rca_summary on the first entry; use later entries only to describe "
        "the downstream impact."
    )

    # Section 4 — Full evidence JSON
    evidence_section = (
        "\n## FULL EVIDENCE\n"
        + json.dumps(contextual_params, indent=2, sort_keys=True)
    )

    # Section 5 — Output instruction
    output_instruction = (
        "\n## OUTPUT\n"
        "Return ONLY the JSON object specified in the system prompt.\n"
        "rca_summary must be at least 4 complete sentences — do not truncate it.\n"
        "No text before or after the JSON."
    )

    prompt = "\n".join([
        "# CI/CD FAILURE DIAGNOSIS REQUEST",
        "",
        primary_error_section,
        kb_section,
        evidence_guidance,
        evidence_section,
        output_instruction,
    ])

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


# Severity → points mapping used by candidate_score
_CANDIDATE_SEVERITY_POINTS: dict[str, int] = {
    "critical": 40,
    "high":     30,
    "medium":   20,
    "low":      10,
}


def candidate_score(candidate: dict[str, Any]) -> int:
    """
    Score a candidate error/block dict on [0, ∞) for primary-failure selection.

    Dimensions
    ----------
    Severity (critical/high/medium/low)  : 10–40 pts
    Source is a failure_block            : +15
    failure_type not generic             : +20
    failure_type not less-specific       : +35
    error_code not generic               : +25
    Matches SPECIFIC_ERROR_HINTS regex   : +35
    Contains HTTP 4xx/5xx status code    : +20
    rca_weight from preprocessor (0-100) : added directly (same scale)
    """
    text         = f"{candidate.get('message', '')}\n{candidate.get('context', '')}"
    failure_type = str(candidate.get("failure_type") or "")
    error_code   = str(candidate.get("error_code")   or "")
    severity     = str(candidate.get("severity")     or "low")
    rca_weight   = int(candidate.get("rca_weight")   or 0)

    # Severity base — replaces the missing severity contribution from before
    score = _CANDIDATE_SEVERITY_POINTS.get(severity, 10)

    if candidate.get("kind") == "failure_block":       score += 15
    if failure_type not in GENERIC_ERROR_TYPES:         score += 20
    if failure_type not in LESS_SPECIFIC_ERROR_TYPES:   score += 35
    if error_code not in GENERIC_ERROR_CODES:           score += 25
    if SPECIFIC_ERROR_HINTS.search(text):               score += 35
    if re.search(r"\b[45]\d{2}\b", text):               score += 20
    # Incorporate the preprocessor's rca_weight directly
    score += rca_weight
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


def build_heuristic_summary(
    error_type: str,
    error_code: str,
    evidence_message: str,
    primary: dict[str, Any] | None = None,
) -> str:
    """
    Build a multi-sentence heuristic RCA summary when the model is unavailable
    or returns an empty rca_summary.

    Produces a minimum 3-sentence narrative combining error type, evidence
    message, location, and investigation direction — suitable for display
    in the UI without LLM inference.
    """
    error_type     = str(error_type     or "unknown_error").strip()
    error_code     = str(error_code     or "").strip()
    evidence_message = str(evidence_message or "").strip()

    # Location context from primary candidate
    location_text = ""
    stage_text    = ""
    if primary and isinstance(primary, dict):
        loc = primary.get("location") or {}
        if isinstance(loc, dict):
            parts = [
                p for p in [
                    str(loc.get("job")      or ""),
                    str(loc.get("log_file") or ""),
                    (f"line {loc.get('line')}" if loc.get("line") else ""),
                ]
                if p
            ]
            location_text = " / ".join(parts)
        stage_text = str(primary.get("stage") or "")

    code_clause  = f" (exit code: {error_code})" if error_code else ""
    loc_clause   = f" in {location_text}"         if location_text else ""
    stage_clause = f" during the {stage_text} stage" if stage_text else ""
    msg_clause   = f'The primary evidence is: "{evidence_message[:200]}".' if evidence_message else ""

    # Per-error-type 3-sentence templates
    _templates: dict[str, str] = {
        "dependency_error": (
            f"The workflow failed{stage_clause} due to a dependency resolution error{code_clause}{loc_clause}. "
            f"A required package or library could not be installed, downloaded, or resolved from the registry — "
            f"this is the root cause of the build failure. "
            f"{msg_clause} "
            f"To investigate: check the package manifest and lockfile for version conflicts or missing entries, "
            f"verify registry access and network connectivity, and try clearing the dependency cache before retrying."
        ),
        "permission_error": (
            f"The workflow was denied access{stage_clause}{loc_clause}{code_clause}. "
            f"A required GitHub token, repository secret, cloud IAM permission, or file-system path "
            f"was not accessible to the runner at the point of failure. "
            f"{msg_clause} "
            f"To investigate: verify the GITHUB_TOKEN scopes and workflow `permissions` key, "
            f"confirm all referenced secrets are configured for this repository, "
            f"and check any cloud credentials used by the failing step."
        ),
        "build_error": (
            f"The workflow failed{stage_clause} with a build or compilation error{code_clause}{loc_clause}. "
            f"The build tool encountered a syntax error, type error, or unresolvable import "
            f"that prevented successful compilation or bundling. "
            f"{msg_clause} "
            f"To investigate: reproduce the build locally with the same tool version, "
            f"check recent commits to the referenced file or configuration, "
            f"and review compiler/linter output for the exact error location."
        ),
        "test_failure": (
            f"The workflow failed{stage_clause} because one or more automated tests did not pass{code_clause}{loc_clause}. "
            f"A test assertion failed or the test runner exited with a non-zero status, "
            f"indicating a regression or environment mismatch introduced by a recent change. "
            f"{msg_clause} "
            f"To investigate: run the failing test suite locally, review recent commits "
            f"for changes to the tested component, and check for flaky test patterns or "
            f"environment-specific setup issues."
        ),
        "configuration_error": (
            f"The workflow failed{stage_clause} due to a configuration error{code_clause}{loc_clause}. "
            f"An invalid, missing, or malformed workflow YAML key, environment variable, "
            f"or referenced file caused the runner to abort before completing the job. "
            f"{msg_clause} "
            f"To investigate: validate the workflow YAML with the GitHub Actions schema, "
            f"check that all environment variables and secrets referenced in the failing step exist, "
            f"and verify any path or file references are correct for the repository structure."
        ),
        "network_error": (
            f"The workflow failed{stage_clause} due to a network connectivity error{code_clause}{loc_clause}. "
            f"The runner could not reach a remote endpoint — this may be caused by a DNS failure, "
            f"TLS certificate issue, proxy misconfiguration, or a transient service outage. "
            f"{msg_clause} "
            f"To investigate: check the target endpoint's availability, verify any proxy or "
            f"egress rules for the runner, and consider adding retry logic with exponential backoff "
            f"around the failing network call."
        ),
        "container_error": (
            f"The workflow failed{stage_clause} due to a container image or runtime error{code_clause}{loc_clause}. "
            f"The Docker image could not be built, pulled, or started — this is often caused by "
            f"an invalid Dockerfile, missing base image, or invalid registry credentials. "
            f"{msg_clause} "
            f"To investigate: verify the image tag and registry credentials, "
            f"try building the image locally with the same Dockerfile, "
            f"and check the container runtime logs for the specific failure reason."
        ),
        "timeout": (
            f"The workflow failed{stage_clause} because a step exceeded its time limit{code_clause}{loc_clause}. "
            f"The job or a specific step took longer than the configured timeout, "
            f"which may indicate a slow external dependency, an infinite loop, or under-provisioned runner. "
            f"{msg_clause} "
            f"To investigate: check the step's `timeout-minutes` configuration, "
            f"profile the slow operation to identify the bottleneck, "
            f"and consider caching heavy dependencies or splitting the job into smaller steps."
        ),
    }

    if error_type in _templates:
        return _templates[error_type].strip()

    # HTTP 403 / permission shortcut (most common case worth calling out explicitly)
    if "403" in error_code or "403" in evidence_message or "permission" in error_type.lower():
        return _templates["permission_error"].strip()

    # Generic fallback — combine available signals into a narrative
    generic_intro = (
        f"The workflow failed{stage_clause}{code_clause}{loc_clause}. "
    )
    generic_cause = (
        f"The error type is classified as '{error_type}', "
        + (f"which occurred with error code '{error_code}'. " if error_code else "and no specific exit code was captured. ")
    )
    generic_evidence = msg_clause or "No primary error message was captured from the log."
    generic_action = (
        "To investigate: review the full job log for the first non-zero exit or exception, "
        "apply the most recent changes against a clean environment, "
        "and check the workflow YAML for any misconfigured steps or missing secrets."
    )
    return f"{generic_intro}{generic_cause}{generic_evidence} {generic_action}".strip()


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
            summary = build_heuristic_summary(
                error_type=str(failure_type or ""),
                error_code=str(error_code or ""),
                evidence_message=evidence_message,
                primary=primary,
            )

    model_evidence = model_value(model_response, "evidence")
    extra_messages: list[str] = []
    if isinstance(model_evidence, list):
        extra_messages = [text for item in model_evidence if (text := evidence_text(item))]
    elif isinstance(model_evidence, str):
        extra_messages = [model_evidence]

    confidence = str(model_value(model_response, "confidence") or ("medium" if primary else "low"))
    model_location = model_value(model_response, "failure_location", "location")
    failure_location = primary.get("location") if primary else (model_location if isinstance(model_location, dict) else {})
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
        "verification_commands": verification_commands,
        "confidence":            confidence,
        "source":                source,
        "error_signature":       error_signature,
        "diagnosis_mode":        "model" if model_response else "heuristic_fallback",
    }
    if model_error and not model_response:
        report["diagnosis_note"] = "Model inference was unavailable; a heuristic RCA was generated."

    # Attach explainable confidence report (backward-compatible — alongside existing 'confidence' field)
    try:
        from utility.confidence_validator import ConfidenceValidator  # lazy import
        _cv = ConfidenceValidator()
        _kb_hits = [{"similarity": e.get("similarity", 0.0)} for e in (kb_context or [])]
        _cr = _cv.validate_rca(report, signals, _kb_hits)
        report["confidence_report"] = _cr.to_dict()
    except Exception as _cv_exc:  # noqa: BLE001
        logger.debug("Confidence validation skipped (non-fatal): %s", _cv_exc)

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

    Revised 8-step flow
    ───────────────────
    1. Log collection     — download run zip, extract metadata
    2. Pre-processing     — PreprocessingPipeline (normalise → extract →
                            enrich → signature → embed)
    3. KB search          — query ChromaDB SelfLearningKnowledgeBase
    4. Vector retrieval   — retrieve similar log chunks from ChromaDB
    5. RCA generation     — LLM with full structured input
    6. Light web search   — always runs; gathers quick reference links (docs,
                            GitHub issues, GitHub PRs) that enrich the output
                            without influencing the RCA prompt.
                            Deep search is retained in web-search-agent.py
                            but disabled (DEEP_SEARCH_ENABLED=False).
    7. Fix generation     — FixRecommendationAgent with web context
    8. KB staging queue   — buffer (error_signature → RCA) for offline
                            batch update via kb_batch_update.py

    Parameters
    ----------
    run_url : str
        GitHub Actions workflow run URL.
    token : str, optional
        GitHub personal access token.
    use_model : bool
        Set False to skip model inference and return heuristic RCA only.
    model_id : str, optional
        Override the default model ID.

    Returns
    -------
    dict
        Full diagnostic result with run metadata, structured RCA report,
        fix recommendations, web search references, and generation time.
    """
    t_pipeline_start = time.monotonic()

    # Validate inputs
    if not run_url:
        print("Job url is missing ... Terminating...")
        exit(1)
    if token is None:
        token = os.getenv("GITHUB_ACCESS_TOKEN")

    print("\n------- Received inputs --------\n")
    print("1. Run URL: ", run_url)
    print("\n2. Git PAT: ", token)

    # Step 1 — Log collection
    record = collect_run_from_url(run_url, token=token)

    # Step 2 — Data pre-processing pipeline
    signals, failure_blocks, _ = preprocess_run_record(record)
    processed_ctx = extract_processed_context(record, signals, failure_blocks)

    print("----------------- Step-2: Pre-processing pipeline: Failure blocks  -----------------------\n")
    print(failure_blocks)
    print("\n---------------------------------------------------------------------------\n")

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
            prompt, _input_payload = build_rca_prompt(
                run_url, record, signals, failure_blocks,
                retrieved_context, kb_context, processed_ctx,
            )

            #tuning = processed_ctx.get("model_tuning_params")
            text = LLMClientInterface().model_api_request(prompt)
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

    # Shared search inputs used by both light and deep search
    _primary_failure = select_primary_failure(signals, failure_blocks) or {}
    _error_msg  = str(_primary_failure.get("message", ""))
    _error_type = rca.get("failure_type", "")
    _repo       = record.get("repository", "")
    _meta_env   = processed_ctx.get("metadata", {}).get("environment", {})
    _language   = str(_meta_env.get("language",  "") if isinstance(_meta_env, dict) else "")
    _framework  = str(_meta_env.get("framework", "") if isinstance(_meta_env, dict) else "")

    _ws_agent = web_search_agent_module.WebSearchAgent(
        github_token=token or os.getenv("GITHUB_TOKEN"),
        top_k=5,
    )

    # Step 6 — Light web search (always runs regardless of KB confidence)
    # Results enriched with 5-dimension relevance score; rca_summary passed
    # so the scorer can compute rca_similarity.
    _rca_summary = rca.get("rca_summary", "")
    quick_references: dict[str, Any] = {"quick_references": [], "search_time": 0.0, "error": None}
    try:
        quick_references = _ws_agent.light_search(
            error_signature=error_signature or _error_type,
            error_message=_error_msg,
            error_type=_error_type,
            repository=_repo,
            language=_language,
            framework=_framework,
            rca_summary=_rca_summary,
        )
        logger.info(
            "WebSearchAgent.light_search: %d quick references",
            len(quick_references.get("quick_references", [])),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Light web search failed (non-fatal): %s", exc)
        quick_references["error"] = str(exc)

    # Step 7 — Deep search is DISABLED (DEEP_SEARCH_ENABLED=False in web-search-agent.py).
    # WebSearchAgent.search() is preserved in the module for future use but not called here.
    # quick_references from the light search (Step 6) serve as web context for fix generation.
    web_search_result: dict[str, Any] = {
        "sources":       [],
        "confidence":    0.0,
        "total_results": 0,
        "search_time":   0.0,
        "error":         None,
        "note":          "Deep search disabled; see quick_references for light-search results.",
    }

    # Fix generation via FixRecommendationAgent
    primary_error_block: dict[str, Any] = {}
    raw_error_blocks = processed_ctx.get("error_blocks", [])
    if raw_error_blocks and isinstance(raw_error_blocks[0], dict):
        primary_error_block = raw_error_blocks[0]
    elif failure_blocks:
        fb = failure_blocks[0]
        primary_error_block = {
            "error_message": getattr(fb, "error_message", ""),
            "stack_trace":   getattr(fb, "stack_trace",   []),
        }

    metadata_for_fix: dict[str, Any] = {
        "repository":  record.get("repository", ""),
        "environment": processed_ctx.get("metadata", {}).get("environment", {}),
    }

    fixes: dict[str, Any] = {}
    try:
        fix_agent = fix_recommendor.FixRecommendationAgent()
        # Pass quick_references as web context for fix generation (deep search disabled)
        _light_refs = quick_references.get("quick_references", [])
        fixes = fix_agent.recommend(
            rca=rca,
            error_block=primary_error_block,
            web_context=_light_refs,
            metadata=metadata_for_fix,
        )
        logger.info(
            "FixRecommendationAgent returned %d fix(es)",
            len(fixes.get("fixes", [])),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fix generation failed (non-fatal): %s", exc)
        fixes = {"fixes": [], "generation_errors": {"agent": str(exc)}}

    # Attach fixes to the RCA report
    rca["fixes"] = fixes

    # Step 8 — Queue RCA for offline KB batch update
    # Deep search is disabled (DEEP_SEARCH_ENABLED=False in web-search-agent.py)
    # but code is retained there for future use.
    if error_signature:
        queue_rca_for_kb_update(error_signature, rca)

    generation_time_seconds = round(time.monotonic() - t_pipeline_start, 2)

    return {
        "run": {
            "repository":    record.get("repository"),
            "run_id":        record.get("run_id"),
            "workflow_name": record.get("workflow_name"),
            "html_url":      record.get("html_url"),
        },
        "signal_count":         len(signals),
        "failure_block_count":  len(failure_blocks),
        "error_signature":      error_signature,
        "kb_hits":              len(kb_context),
        "quick_references":     quick_references,
        "web_search":           web_search_result,
        "generation_time":      generation_time_seconds,
        "rca":                  rca,
    }

# Test the RCA workflow run in the main
def main():
    # invoke the diagnose_run_workflow() for functionality checking
    rca_output = diagnose_workflow_run(run_url="https://github.com/pytest-dev/pytest/actions/runs/30181683620")

    print(rca_output)
    
if __name__  == "__main__":
    main()