# =============================================================================
# Central configuration for the weathernext_forecast project.
# =============================================================================
# Every path in the project is derived from THIS file's location, never from a
# hardcoded absolute path. That makes the whole project directory portable: you
# can copy it to another machine or pack it onto a drive (e.g. when leaving a
# job) and everything still runs unchanged.
# =============================================================================

from __future__ import annotations

import os
from pathlib import Path

# Project root. This file lives at:
#   <root>/src/weathernext_forecast/config.py
# so the root is two levels up:
#   parents[0] = <root>/src/weathernext_forecast   (this package)
#   parents[1] = <root>/src
#   parents[2] = <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The upstream fork of google-deepmind/weathernext. This is a SEPARATE git
# repository (it carries its own `.git` and tracks origin=your-fork /
# upstream=deepmind). Treat it as a read-only dependency: we `import weathernext`
# from here, never edit it for day-to-day work. It is git-ignored by this
# project (see .gitignore).
UPSTREAM_DIR = PROJECT_ROOT / "upstream" / "weathernext"

# Binary artifacts — large and/or regenerable, kept OUTSIDE version control
# (git-ignored). They live inside the project so it stays self-contained.
WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"     # model checkpoints (.npz)
STATS_DIR = PROJECT_ROOT / "models" / "stats"         # normalization stats (.nc)
DATA_DIR = PROJECT_ROOT / "data" / "processed"        # standard-format .nc inputs
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"          # files exactly as downloaded
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"        # model output (.nc)

# External results tree — DECOUPLED from the package, one level above the project.
# Predictions + visualizations for every model land here, organized by model:
#   <results>/<model>/<variant>/<init>Z/{predictions,visualizations}/
# Overridable via $RESULTS_ROOT (the realtime pipeline sets it explicitly).
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", str(PROJECT_ROOT.parent / "results")))


# -----------------------------------------------------------------------------
# Model & data selection — edit this block, nothing else.
# -----------------------------------------------------------------------------

# The checkpoint to load (one of the three pretrained GraphCast models, local
# clean name). Put its .npz into WEIGHTS_DIR:
#   GraphCast.npz              -> 0.25 deg, 37 levels, ERA5   (needs big GPU/TPU)
#   GraphCast_small.npz        -> 1.00 deg, 13 levels, ERA5   (runs on CPU, slow)
#   GraphCast_operational.npz  -> 0.25 deg, 13 levels, HRES-fc0 (no precip input)
MODEL_FILENAME = "GraphCast_operational.npz"

# Results-tree identity for the saved predictions (see prediction_store).
# Kept explicit rather than parsed from MODEL_FILENAME so the tree stays stable
# as checkpoints change.
MODEL_FAMILY = "graphcast"
MODEL_VARIANT = "operational"

# Crop region for the saved prediction. "china" writes only the China box
# (~0.4 GB) instead of the full-global ~13 GB. Set "global" to keep everything.
PREDICT_REGION = os.environ.get("WEATHERNEXT_PREDICT_REGION", "china")

# The input weather file in DATA_DIR, already in standard format:
#   dims   : (batch, time, lat, lon)            surface / static variables
#            (batch, time, lat, lon, level)     pressure-level variables
#   coords : time (timedelta, 6-hour steps), datetime (absolute),
#            lat, lon, level (hPa)
# Its resolution and level count MUST match the chosen model.
# (Produced by the stage-2 `data_processing` package from open-data IFS fc0.)
# Overridable via WEATHERNEXT_INPUT_FILENAME so the realtime pipeline can point
# at whatever date it just downloaded/processed without editing this file.
INPUT_FILENAME = os.environ.get(
    "WEATHERNEXT_INPUT_FILENAME",
    "source-ifs_date-2026-08-27_res-0.25_levels-13_steps-40.nc",
)

# Forecast lead times to predict, as an xarray label slice in 6-hour steps.
# The steps-40 input file has 42 timesteps = 2 for input + 40 for target, so
# this spans the full GraphCast 10-day horizon.
TARGET_LEAD_TIMES = slice("6h", "240h")
