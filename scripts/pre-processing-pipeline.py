#!/usr/bin/env python3
"""Compatibility wrapper for pre-process-pipeline.py."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("pre-process-pipeline.py")), run_name="__main__")
