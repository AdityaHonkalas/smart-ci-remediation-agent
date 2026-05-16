#!/usr/bin/env python3
"""Flask UI for CI/CD failure RCA."""

from __future__ import annotations

import os
from typing import Any

from flask import Flask, jsonify, render_template_string, request

from agents.diagnosis_agent import diagnose_workflow_run


app = Flask(__name__)


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Smart CI Remediation</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #17202a;
      --muted: #5d6b78;
      --line: #d7dee6;
      --panel: #ffffff;
      --page: #f4f7fb;
      --accent: #087f8c;
      --accent-dark: #075f68;
      --danger: #b42318;
      --ok: #146c43;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--page);
    }
    header {
      padding: 24px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    h1 {
      margin: 0;
      font-size: clamp(24px, 4vw, 38px);
      font-weight: 760;
      letter-spacing: 0;
    }
    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 24px auto 48px;
      display: grid;
      grid-template-columns: minmax(320px, 440px) 1fr;
      gap: 20px;
      align-items: start;
    }
    section, form {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    form {
      padding: 18px;
      display: grid;
      gap: 14px;
    }
    label {
      display: grid;
      gap: 7px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }
    input[type="url"], input[type="password"], input[type="text"] {
      width: 100%;
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 11px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }
    .row {
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 32px;
      color: var(--muted);
      font-size: 14px;
    }
    button {
      min-height: 42px;
      border: 0;
      border-radius: 6px;
      padding: 10px 14px;
      font: inherit;
      font-weight: 760;
      color: #fff;
      background: var(--accent);
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled { opacity: .62; cursor: wait; }
    .result {
      min-height: 420px;
      padding: 0;
      overflow: hidden;
    }
    .toolbar {
      min-height: 54px;
      padding: 14px 16px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      border-bottom: 1px solid var(--line);
    }
    .status {
      color: var(--muted);
      font-size: 14px;
      overflow-wrap: anywhere;
    }
    .content {
      padding: 16px;
      display: grid;
      gap: 16px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 74px;
      background: #fbfcfe;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .metric strong {
      display: block;
      margin-top: 6px;
      font-size: 16px;
      overflow-wrap: anywhere;
    }
    pre {
      margin: 0;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #101820;
      color: #e8f0f7;
      overflow: auto;
      max-height: 520px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 13px;
      line-height: 1.45;
    }
    .error { color: var(--danger); }
    .ok { color: var(--ok); }
    @media (max-width: 860px) {
      header { padding: 20px 16px 14px; }
      main { grid-template-columns: 1fr; width: calc(100vw - 24px); margin-top: 16px; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Smart CI Remediation</h1>
  </header>
  <main>
    <form id="diagnosis-form">
      <label>
        GitHub Actions Run URL
        <input name="run_url" type="url" required placeholder="https://github.com/OWNER/REPO/actions/runs/123456789">
      </label>
      <label>
        GitHub Token
        <input name="github_token" type="password" placeholder="Uses GITHUB_TOKEN when blank">
      </label>
      <label>
        Bedrock Model ID
        <input name="model_id" type="text" value="{{ model_id }}">
      </label>
      <label class="row">
        <input name="use_model" type="checkbox" checked>
        Use Claude Sonnet through Bedrock
      </label>
      <button id="submit" type="submit">Diagnose Failure</button>
    </form>
    <section class="result">
      <div class="toolbar">
        <strong>RCA Output</strong>
        <span id="status" class="status">Idle</span>
      </div>
      <div id="content" class="content">
        <div class="grid">
          <div class="metric"><span>Run</span><strong>-</strong></div>
          <div class="metric"><span>Signals</span><strong>-</strong></div>
          <div class="metric"><span>Blocks</span><strong>-</strong></div>
        </div>
        <pre id="output">{}</pre>
      </div>
    </section>
  </main>
  <script>
    const form = document.getElementById('diagnosis-form');
    const submit = document.getElementById('submit');
    const statusEl = document.getElementById('status');
    const output = document.getElementById('output');
    const metrics = document.querySelectorAll('.metric strong');

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      submit.disabled = true;
      statusEl.textContent = 'Collecting logs and diagnosing...';
      statusEl.className = 'status';
      output.textContent = '{}';

      const formData = new FormData(form);
      const payload = {
        run_url: formData.get('run_url'),
        github_token: formData.get('github_token'),
        model_id: formData.get('model_id'),
        use_model: formData.get('use_model') === 'on'
      };

      try {
        const response = await fetch('/api/diagnose', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || 'Diagnosis failed');
        }
        metrics[0].textContent = data.run?.run_id || '-';
        metrics[1].textContent = data.signal_count ?? '-';
        metrics[2].textContent = data.failure_block_count ?? '-';
        output.textContent = JSON.stringify(data.rca, null, 2);
        statusEl.textContent = 'Complete';
        statusEl.className = 'status ok';
      } catch (error) {
        statusEl.textContent = error.message;
        statusEl.className = 'status error';
      } finally {
        submit.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@app.get("/")
def index() -> str:
    return render_template_string(PAGE, model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4"))


@app.post("/api/diagnose")
def diagnose() -> Any:
    payload = request.get_json(silent=True) or {}
    run_url = str(payload.get("run_url") or "").strip()
    if not run_url:
        return jsonify({"error": "GitHub Actions run URL is required."}), 400

    try:
        result = diagnose_workflow_run(
            run_url=run_url,
            token=str(payload.get("github_token") or "").strip() or None,
            use_model=bool(payload.get("use_model", True)),
            model_id=str(payload.get("model_id") or "").strip() or None,
        )
    except Exception as exc:  # noqa: BLE001 - return actionable UI errors.
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)


if __name__ == "__main__":
    app.run(host=os.getenv("FLASK_HOST", "127.0.0.1"), port=int(os.getenv("FLASK_PORT", "5000")), debug=True)
