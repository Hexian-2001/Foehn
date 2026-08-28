# -*- coding: utf-8 -*-
"""Central configuration for the ECMWF open-data downloader.

This package is intentionally MODEL-AGNOSTIC: it downloads raw ECMWF GRIB
fields by their short name (``2t``, ``z``, ``q``, ...) and does NOT know about
any model's variable renaming. That mapping belongs to the downstream
"processing" stage, so other forecast models can reuse this downloader as-is.

All paths derive from this file's location (portable, like the sibling
``weathernext_forecast``), and the data root is overridable for server
deployment via the ``OPENDATA_DATA_ROOT`` environment variable or the
``--data-root`` CLI flag.
"""

from __future__ import annotations

import os
from pathlib import Path

# Package root: <root>/src/opendata_download/config.py -> parents[2] == <root>
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Source.
# ---------------------------------------------------------------------------
# The ecmwf.opendata Client already defaults to model="ifs", resol="0p25"
# (0.25 deg), so we only pin the source here. Default to the Google Cloud
# mirror: direct links to data.ecmwf.int are cut at ~16KB chunk boundaries
# from some networks (retry cannot fix a deterministic truncation), and the
# AWS mirror rate-limits anonymous requests with 503 Slow Down.
# Valid: "ecmwf" | "aws" | "azure" | "google".
SOURCE = "google"

# ---------------------------------------------------------------------------
# What to download — ECMWF GRIB short names, NOT model variable names.
# ---------------------------------------------------------------------------
# Pressure levels: the "WeatherBench 13" set. It matches both the levels
# GraphCast_operational consumes and the levels ECMWF open data publishes on
# pressure levels.
PRESSURE_LEVELS: tuple[int, ...] = (
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000,
)

# Dynamic surface fields (levtype="sfc").
SURFACE_PARAMS: tuple[str, ...] = ("2t", "msl", "10u", "10v")

# Dynamic pressure-level fields (levtype="pl").
PRESSURE_PARAMS: tuple[str, ...] = ("z", "t", "u", "v", "w", "q")

# Static, time-invariant fields (levtype="sfc"): surface geopotential and the
# land-sea mask. Downloaded once per deployment, not per cycle.
#   z   -> geopotential_at_surface  (m^2/s^2)
#   lsm -> land_sea_mask            (0/1)
STATIC_PARAMS: tuple[str, ...] = ("z", "lsm")

# Forecast step 0 = the analysis (fc0): the "initial condition", not a forecast.
ANALYSIS_STEP: int = 0

# IFS analysis cycles are run 4x/day at these UTC hours.
CYCLE_HOURS: tuple[int, ...] = (0, 6, 12, 18)

# ---------------------------------------------------------------------------
# Output layout.
# ---------------------------------------------------------------------------
# Raw downloaded fields are MODEL-AGNOSTIC and shared across forecast models,
# so they live OUTSIDE any single model project — in a shared data store that
# is a sibling of the code folders (opendata_download, weathernext_forecast,
# ...). Model-specific "processed" files stay inside each model's own project
# (e.g. weathernext_forecast/data/processed).
#
# Default:  <forecast_models>/data/raw/ifs/<YYYY-MM-DD>/<HH>/...
# Override on the server (bigger/faster disk) via OPENDATA_DATA_ROOT or
# --data-root.
SHARED_DATA_ROOT: Path = PACKAGE_ROOT.parent / "data"
DATA_ROOT: Path = Path(os.environ.get("OPENDATA_DATA_ROOT", SHARED_DATA_ROOT))
