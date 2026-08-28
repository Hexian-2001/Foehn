#!/usr/bin/env python
"""Thin entry point for the ECMWF open-data downloader.

Examples:
    # Forecast initialized at 2026-08-27 00 UTC (downloads T-6h and T cycles + static):
    python scripts/download_fc0.py --date 2026-08-27 --time 00

    # Re-download even if files already exist:
    python scripts/download_fc0.py --date 2026-08-27 --time 00 --force

    # Latest available analysis cycle (for the unattended 24/7 auto-run):
    python scripts/download_fc0.py --latest
"""

import sys
from pathlib import Path

# Make the src/ package importable without a prior `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opendata_download.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
