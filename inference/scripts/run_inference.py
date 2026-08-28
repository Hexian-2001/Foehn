#!/usr/bin/env python
"""Thin CLI entry point for model-agnostic forecast inference.

Adds the package's ``src/`` directory to sys.path, then delegates to
``inference.cli.main``. All logic lives in ``src/inference/``; keep this stub.

Run from anywhere::

    python scripts/run_inference.py --model graphcast --init-time 2026-08-27T00
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
