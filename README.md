## Smart CI Remediation Tool

An AI approach for debugging CI pipeline failures, starting with GitHub Actions
workflow runs.

### Pipeline

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

3. Query indexed log excerpts:

```powershell
python scripts/db.py "process completed with exit code 1 kubelet test failure" --top-k 5
```

The vector store is a SQLite-backed local implementation with deterministic
hashed embeddings, so the proof of concept runs without external services.

4. Prepare Claude Sonnet RCA training artifacts:

```powershell
python scripts/train-claude-sonnet.py
```

This writes `data/claude_training_messages.jsonl` and
`data/claude_prompt_pack.json`. Claude Sonnet 4 is intended for RCA inference
through Bedrock and RAG; the script guards Bedrock fine-tuning because Sonnet 4
is not listed as a fine-tunable base model in the current Bedrock support table.

5. Start the Flask UI:

```powershell
python app.py
```

Open `http://127.0.0.1:5000`, paste a GitHub Actions run URL, and submit it.
