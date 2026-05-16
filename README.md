## Smart CI Remediation Tool

An AI approach for debugging CI pipeline failures, starting with GitHub Actions
workflow runs.

### Pipeline

1. Collect failed GitHub Actions run logs as zip archives:

```powershell
$env:GITHUB_TOKEN = "<optional-token>"
python scripts/log-collector.py --repo https://github.com/kubernetes/kubernetes --limit 10
```

The collector stores zip files under `data/logs/kubernetes_kubernetes/` and
updates `data/index.json` with workflow, job, commit, status, and archive
metadata.

2. Pre-process logs, extract error signals, build the graph, create training
examples, and populate the local vector store:

```powershell
python scripts/pre-process-pipeline.py
```

Generated artifacts:

- `data/preprocessed_logs.jsonl`
- `data/error_signals.jsonl`
- `data/knowledge_graph.json`
- `data/training_dataset.jsonl`
- `data/vector_store.sqlite`

3. Query indexed log excerpts:

```powershell
python scripts/db.py "process completed with exit code 1 kubelet test failure" --top-k 5
```

The vector store is a SQLite-backed local implementation with deterministic
hashed embeddings, so the proof of concept runs without external services.
