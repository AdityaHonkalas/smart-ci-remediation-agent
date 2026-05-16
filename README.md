## Smart CI Remediation Agent

An AI-assisted approach for debugging CI pipeline failures, starting with
GitHub Actions workflow runs.

### Problem Statement

CI/CD failures are often noisy, repetitive, and expensive to debug. A single
failed GitHub Actions run can contain logs from many jobs and steps, while the
actual root cause may be hidden in a small failure block such as a dependency
resolution error, test assertion, permission issue, timeout, or environment
configuration mismatch.

The **Smart CI Remediation Agent** is designed to reduce this investigation
time. It collects workflow logs, extracts the most relevant error signals,
retrieves similar historical failures, and generates a root cause analysis
(RCA) with actionable remediation steps. The goal is to help engineers move
from "the pipeline failed" to "this failed because of X, in Y stage, and here
is the likely fix" with minimal manual log digging.

### Workflow

#### Model training and preparation workflow

The training workflow converts raw CI/CD logs and curated datasets into
structured artifacts that can be used for RCA prompting, retrieval, and model
evaluation.

1. Collect GitHub Actions run logs as zip archives:

```powershell
$env:GITHUB_TOKEN = "<optional-token>"
python scripts/log-collector.py --repo https://github.com/kubernetes/kubernetes --limit 10
python scripts/log-collector.py --run-url https://github.com/OWNER/REPO/actions/runs/RUN_ID
```

The collector stores zip files under `data/logs/kubernetes_kubernetes/` and
updates `data/index.json` plus `data/metadata/...json` with workflow, job,
commit, status, and archive metadata.

2. Pre-process logs, extract error signals, build the graph, create training
examples, and populate the local vector store:

```powershell
python scripts/pre-process-pipeline.py
python scripts/pre-process-pipeline.py --download-kaggle
```

Generated artifacts:

- `data/preprocessed_logs.jsonl`
- `data/error_signals.jsonl`
- `data/failure_blocks.jsonl`
- `data/knowledge_graph.json`
- `data/training_dataset.jsonl`
- `data/vector_store.sqlite`

The Kaggle CI/CD failure dataset is supported through
`mirzayasirabdullah07/cicd-pipeline-failure-logs-dataset-for-aiops`. You can
download it with `--download-kaggle` or pass an existing local CSV/JSON/JSONL
file or directory with `--kaggle-dataset-path`.

On networks with TLS inspection, set `ALLOW_INSECURE_DATASET_DOWNLOAD=1` only
for public dataset downloads when normal certificate verification fails.

Failure block regexes, error patterns, error-code signals, and classification
signals live in `data/failure_signal_patterns.json`.

3. Query indexed log excerpts to validate retrieval quality:

```powershell
python scripts/db.py "process completed with exit code 1 kubelet test failure" --top-k 5
```

The vector store is a SQLite-backed local implementation with deterministic
hashed embeddings, so the proof of concept runs without external services.

4. Prepare Claude Sonnet RCA training artifacts:

```powershell
python scripts/train-claude-sonnet.py
```

This reads the existing `data/training_dataset.jsonl` and writes
`data/claude_training_messages.jsonl`, `data/claude_prompt_pack.json`, and
`data/claude_training_manifest.json`. Run the pre-processing pipeline first if
the training dataset is missing. The default training iteration count is 5.
Claude Sonnet 4 is intended for RCA inference through prompt/RAG artifacts; the
script guards Bedrock fine-tuning because Sonnet 4 is not listed as a
fine-tunable base model in the current Bedrock support table.

Local secrets and model settings are loaded from `.env`:

```powershell
ANTHROPIC_API_KEY=<model-api-key>
ANTHROPIC_MODEL_ID=anthropic.claude-sonnet-4
BEDROCK_MODEL_ID=anthropic.claude-sonnet-4
GITHUB_TOKEN=<github-token>
```

#### User RCA and inline fix suggestion workflow

The user-facing workflow starts from a failed GitHub Actions run URL and returns
an RCA report with remediation guidance.

1. Start the Flask UI:

```powershell
python app.py
```

Open `http://127.0.0.1:5000`, paste a GitHub Actions run URL, and submit it.

2. The Flask app sends the run URL to the diagnosis agent.
3. The agent calls the GitHub REST API, downloads the workflow log archive, and
   stores the run metadata under `data/`.
4. The pre-processing pipeline extracts failure blocks, error signals, job and
   step context, and searchable log chunks for the current run.
5. The vector store retrieves similar historical log excerpts from previously
   indexed workflow failures.
6. Claude Sonnet receives the run metadata, extracted failure block, retrieved
   context, and prompt pack, then returns structured RCA JSON.
7. The UI displays the RCA summary, failure stage, failure type, evidence,
   remediation steps, verification commands, and inline fix suggestions that the
   engineer can apply to the workflow, test code, dependency configuration, or
   runtime environment.

### Installation Guide

Use the following steps to install and run the Flask app locally.

1. Create and activate a virtual environment:

```powershell
cd smart-ci-remediation-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Create or update `.env` with the credentials and model settings required for
   your run:

```powershell
ANTHROPIC_API_KEY=<model-api-key>
ANTHROPIC_MODEL_ID=anthropic.claude-sonnet-4
BEDROCK_MODEL_ID=anthropic.claude-sonnet-4
MODEL_PROVIDER=anthropic
GITHUB_TOKEN=<github-token>
```

Use `MODEL_PROVIDER=bedrock` if you want to call Claude through Amazon Bedrock.
For Bedrock, also configure your AWS credentials and region in the normal AWS
environment variables or credential files.

4. Optional: prepare local datasets and retrieval artifacts before launching the
   UI:

```powershell
python scripts/pre-process-pipeline.py
python scripts/train-claude-sonnet.py
```

5. Start the Flask app:

```powershell
python app.py
```

6. Open the app in your browser:

```text
http://127.0.0.1:5000
```
