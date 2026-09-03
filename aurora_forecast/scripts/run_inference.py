#!/usr/bin/env python
"""Thin CLI entry point for Aurora inference.

Adds the project's `src/` directory to sys.path, then delegates to the real
pipeline in `aurora_forecast.inference`. All logic and configuration live in
`src/aurora_forecast/`; keep this file a stub.

Run from anywhere::

    python scripts/run_inference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root is one level up from this file:
#   <root>/scripts/run_inference.py  ->  parents[1] == <root>
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aurora_forecast.inference import main  # noqa: E402

if __name__ == "__main__":
    main()
