"""Configuration for converting ECMWF open-data GRIB -> GraphCast input .nc.

This package is MODEL-SPECIFIC: unlike ``opendata_download`` (which is
model-agnostic), it knows GraphCast's variable names, units, levels and file
layout. It is still a separate package so the download / process / infer stages
stay decoupled and independently runnable.
"""

from __future__ import annotations

import os
from pathlib import Path

# Package root: <root>/src/data_processing/config.py -> parents[2] == <root>
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------
# Raw GRIB is produced by the sibling `opendata_download` package into a shared,
# model-agnostic store:  <forecast_models>/data/raw/ifs/<date>/<HH>/...
_DATA_ROOT = Path(os.environ.get("OPENDATA_DATA_ROOT", PACKAGE_ROOT.parent / "data"))
RAW_IFS_ROOT: Path = _DATA_ROOT / "raw" / "ifs"

# Processed .nc is model-specific (GraphCast naming/units), so it goes into the
# weathernext_forecast project's processed dir (its config.DATA_DIR).
PROCESSED_ROOT: Path = Path(
    os.environ.get(
        "PROCESSED_ROOT",
        PACKAGE_ROOT.parent / "weathernext_forecast" / "data" / "processed",
    )
)

# ---------------------------------------------------------------------------
# Variable mapping. Keys are the names cfgrib actually emits (CF names for the
# surface fields, GRIB short names for the pressure/static fields), values are
# GraphCast's variable names.
# ---------------------------------------------------------------------------
SURFACE_MAP = {
    "t2m": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
}
PRESSURE_MAP = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "q": "specific_humidity",
}
STATIC_MAP = {
    "z": "geopotential_at_surface",
    "lsm": "land_sea_mask",
}

# ---------------------------------------------------------------------------
# Coordinate handling. cfgrib emits `latitude`/`longitude`/`isobaricInhPa` and a
# few GRIB-only scalar coords we drop (they encode level-type metadata, not the
# physical axes GraphCast needs).
# ---------------------------------------------------------------------------
COORD_RENAME = {"latitude": "lat", "longitude": "lon", "isobaricInhPa": "level"}
DROP_COORDS = ("time", "step", "valid_time", "heightAboveGround", "meanSea", "surface")

# Pressure levels — must equal GraphCast_operational's WeatherBench-13 set.
PRESSURE_LEVELS: tuple[int, ...] = (
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000,
)

# Open-data lon starts at 180 deg (180..359.75, 0..179.75); ERA5 starts at 0.
# Rolling half the axis (720 of 1440) realigns them.
LON_ROLL: int = 720

# ---------------------------------------------------------------------------
# Forecast horizon / time axis.
# ---------------------------------------------------------------------------
STEP_HOURS: int = 6
INPUT_STEPS: int = 2            # T-6h and T (the 12h conditioning window)
FORECAST_HOURS: int = 240       # GraphCast max horizon (10 days)
TARGET_STEPS: int = FORECAST_HOURS // STEP_HOURS  # 40

# Lead hours along the output time axis: -6, 0, 6, ..., 240.
LEAD_HOURS: tuple[int, ...] = tuple(range(-STEP_HOURS, FORECAST_HOURS + 1, STEP_HOURS))

# ---------------------------------------------------------------------------
# Output filename (matches weathernext_forecast's INPUT_FILENAME convention).
# ---------------------------------------------------------------------------
RESOLUTION = 0.25
FILENAME_TPL = "source-ifs_date-{date}_res-{res}_levels-{nlevels}_steps-{nsteps}.nc"
