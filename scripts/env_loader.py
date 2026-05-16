#!/usr/bin/env python3
"""Small .env loader for local scripts.

It intentionally supports the common KEY=VALUE shape only; existing process
environment values win unless override=True is passed.
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_dotenv(path: Path | str = DEFAULT_ENV_PATH, override: bool = False) -> dict[str, str]:
    env_path = Path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    with env_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue

            loaded[key] = value
            if override or key not in os.environ:
                os.environ[key] = value

    return loaded
