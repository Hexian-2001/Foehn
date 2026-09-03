# =============================================================================
# Central configuration for the aurora_forecast project.
# =============================================================================
# Mirrors `weathernext_forecast/config.py` conventions: every path derives from
# THIS file's location, so the project is portable. This package is
# MODEL-SPECIFIC — it knows Aurora's variable names, units, levels and file
# layout (the download stage in `opendata_download` stays model-agnostic).
# =============================================================================

from __future__ import annotations

import os
from pathlib import Path

# Project root: <root>/src/aurora_forecast/config.py -> parents[2] == <root>.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The upstream microsoft/aurora fork. A separate git repository (read-only
# dependency): we `import aurora` from here, never edit it day-to-day.
UPSTREAM_DIR = PROJECT_ROOT / "upstream" / "aurora"

# Binary artifacts — large, kept OUTSIDE version control.
WEIGHTS_DIR = PROJECT_ROOT / "model_weights" / "model"
CHECKPOINT_NAME = "aurora-0.25-finetuned.ckpt"      # IFS HRES T0, 0.25 deg
STATIC_PICKLE_NAME = "aurora-0.25-static.pickle"     # {z, slt, lsm} (721, 1440)

# External results tree — DECOUPLED, one level above the project:
#   <results>/<model>/<variant>/<init>Z/{predictions,visualizations}/
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", str(PROJECT_ROOT.parent / "results")))

# Results-tree identity (stable, independent of the checkpoint filename).
MODEL_FAMILY = "aurora"
MODEL_VARIANT = "0.25-finetuned"

# Forecast horizon. Aurora's base time-step is 6 h; 40 steps = 240 h (10 days),
# matching the GraphCast operational horizon for a shared downstream.
STEP_HOURS = 6
HISTORY_STEPS = 2                     # T-6h and T conditioning window
FORECAST_STEPS = int(os.environ.get("AURORA_FORECAST_STEPS", "40"))

# Crop region for the saved prediction ("china" ~0.4 GB vs "global" ~13 GB).
# Falls back to the GraphCast env var so a single top-level runner can set one.
PREDICT_REGION = os.environ.get(
    "AURORA_PREDICT_REGION",
    os.environ.get("WEATHERNEXT_PREDICT_REGION", "china"),
)

# Raw GRIB root (produced by the shared `opendata_download` package).
_DATA_ROOT = Path(os.environ.get("OPENDATA_DATA_ROOT", PROJECT_ROOT.parent / "data"))
RAW_IFS_ROOT: Path = _DATA_ROOT / "raw" / "ifs"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

# The prebuilt input .nc filename (produced by `aurora_forecast.adapter`). Set via
# AURORA_INPUT_FILENAME so the realtime pipeline can point the Slurm job at the
# exact file it just processed, without editing this file.
INPUT_FILENAME = os.environ.get("AURORA_INPUT_FILENAME", "")

# ---------------------------------------------------------------------------
# Variable mapping. Keys are the names cfgrib actually emits (probed from the
# raw open-data GRIB, 2026-08-27); values are Aurora's variable names.
# ---------------------------------------------------------------------------
SURFACE_MAP = {
    "t2m": "2t",
    "u10": "10u",
    "v10": "10v",
    "msl": "msl",
}
# `w` (vertical velocity) is emitted by the shared downloader but Aurora does
# NOT consume it — simply not mapped, so it is dropped.
PRESSURE_MAP = {
    "z": "z",
    "t": "t",
    "u": "u",
    "v": "v",
    "q": "q",
}

COORD_RENAME = {"latitude": "lat", "longitude": "lon", "isobaricInhPa": "level"}
DROP_COORDS = ("time", "step", "valid_time", "heightAboveGround", "meanSea", "surface")

# Pressure levels — must equal Aurora's 13-level set (same as GraphCast).
PRESSURE_LEVELS: tuple[int, ...] = (
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000,
)

# Open-data lon starts at 180 deg (180..359.75, 0..179.75); ERA5/Aurora start at
# 0. Rolling half the axis (720 of 1440) realigns to ascending [0, 360).
LON_ROLL: int = 720

# Aurora native variable names -> unified names shared with GraphCast, so the
# SAME visualizer and consumers read both models' predictions.
UNIFIED_MAP = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}
