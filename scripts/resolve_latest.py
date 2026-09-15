#!/usr/bin/env python3
"""Print one fully available ECMWF cycle for the dual-model orchestrator."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_SRC = REPO_ROOT / "opendata_download" / "src"
sys.path.insert(0, str(DOWNLOAD_SRC))

from opendata_download import config  # noqa: E402
from opendata_download.client import OpenDataClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=config.SOURCE)
    args = parser.parse_args()
    latest = OpenDataClient(source=args.source).latest(
        {
            "type": "fc",
            "step": config.ANALYSIS_STEP,
            "levtype": "sfc",
            "param": config.SURFACE_PARAMS[0],
        }
    )
    if not isinstance(latest, dt.datetime):
        raise RuntimeError(f"unexpected latest() result: {latest!r}")
    print(f"{latest:%Y-%m-%d %H}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
