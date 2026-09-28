#!/usr/bin/env python3
"""
Flask application — Smart CI Remediation Agent.

Thin factory that sets up sys.path, loads environment, and registers blueprints.
All route logic lives in the routes/ package:
  routes/diagnose.py   — / (index), /api/diagnose, /api/feedback
  routes/admin.py      — /api/admin/kb-update
  routes/reports.py    — /stats, /evaluation, /api/stats/summary, /api/evaluation/metrics
  routes/feedback.py   — /api/user-feedback
  routes/validation.py — /api/validate/status, /api/validate
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify

# ---------------------------------------------------------------------------
# Path setup — must happen before any project imports
# ---------------------------------------------------------------------------
ROOT_DIR    = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
UTILITY_DIR = ROOT_DIR / "utility"
ROUTES_DIR  = ROOT_DIR / "routes"

for _p in (str(ROOT_DIR), str(SCRIPTS_DIR), str(UTILITY_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env_loader import load_dotenv  # noqa: E402

load_dotenv(ROOT_DIR / ".env")

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = Flask(__name__)

_APP_START = time.time()

# Register blueprints
from routes.diagnose  import diagnose_bp   # noqa: E402
from routes.admin     import admin_bp      # noqa: E402
from routes.reports   import reports_bp    # noqa: E402
from routes.feedback  import feedback_bp   # noqa: E402
from routes.validation import validation_bp  # noqa: E402

app.register_blueprint(diagnose_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(reports_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(validation_bp)


# ---------------------------------------------------------------------------
# API: health check (kept in app.py — not route-specific)
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> Any:
    """Liveness / readiness probe."""
    return jsonify(
        {
            "status": "ok",
            "uptime_seconds": round(time.time() - _APP_START, 1),
            "model_id": (
                os.getenv("ANTHROPIC_MODEL_ID")
                or os.getenv("BEDROCK_MODEL_ID")
                or "anthropic.claude-sonnet-4"
            ),
        }
    )


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=True,
    )
