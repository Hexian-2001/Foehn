"""Extract point wind series from a model prediction .nc and build the 72-hour
1-hourly platform arrays.

Model output contract (``unified-forecast-1``):
  - 40 steps x 6h (lead 6..240 h), 0.25 deg China crop (lat 15..55 desc, lon 70..140 asc).
  - ``10m_u/v_component_of_wind``  -> 10 m wind components.
  - ``u/v_component_of_wind`` @ level=1000 -> 100 m wind proxy (project convention).

The platform wants lead 1..72 at 1 h. Leads 6..72 land directly on model steps;
leads 1..5 are filled by linear interpolation between the lead-0 analysis field
(IFS fc0, read from the stage-2 input .nc) and the lead-6 model step. Without
the analysis file, ``np.interp``'s left-edge clamp yields zero-order hold
(leads 1..5 = lead-6 value) — a safe but coarser fallback.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from .config import RESULTS_ROOT, PROCESSED_IFS_ROOT, LEAD_TIME_COUNT
from .assets_map import Catalog

logger = logging.getLogger("push_to_platform.extract")

# 6h -> 1h anchors for the 72 h horizon: lead 0 (analysis) + leads 6..72.
_ANCHOR_HOURS = np.arange(0, 73, 6, dtype=float)      # [0, 6, ..., 72]
_LEAD_HOURS = np.arange(1, 73, dtype=float)            # [1, ..., 72]


# ── File discovery ──

def find_prediction_nc(model: str, variant: str, date: str) -> Path:
    """Locate the prediction .nc for a model/variant at ``date``T00Z."""
    pattern = str(RESULTS_ROOT / model / variant / f"{date}T00Z" / "predictions" / "*.nc")
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no prediction .nc for {model}/{variant} @ {date}T00Z: {pattern}")
    return Path(hits[0])


def find_analysis_nc(date: str) -> Path | None:
    """Locate the stage-2 IFS input .nc for ``date`` (lead-0 analysis), or None."""
    pattern = str(PROCESSED_IFS_ROOT / f"source-ifs_date-{date}_res-0.25_*.nc")
    hits = sorted(glob.glob(pattern))
    if not hits:
        logger.warning("no analysis input .nc for %s -> zero-order hold for leads 1..5", date)
        return None
    return Path(hits[0])


# ── Point series ──

def _nearest(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    return ds.sel(lat=lat, lon=lon, method="nearest")


def _wind_direction_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological direction (degrees FROM which wind blows, 0=N, 90=E)."""
    return (180.0 + np.degrees(np.arctan2(u, v))) % 360.0


def _interp_component(v0: float, v6_72: np.ndarray) -> np.ndarray:
    """Interpolate one u/v component from 6h anchors to 1h leads 1..72.

    ``v0`` is the lead-0 analysis value; ``v6_72`` are the model step values at
    leads 6..72 (12 values). Linear in the interior; left edge is clamped, so a
    missing analysis (v0 = first model value) yields zero-order hold for 1..5.
    """
    full = np.concatenate([[v0], v6_72])
    return np.interp(_LEAD_HOURS, _ANCHOR_HOURS, full)


def wind_point_series(
    pred: xr.Dataset,
    analysis: xr.Dataset | None,
    lat: float,
    lon: float,
) -> dict[str, list[float | None]]:
    """Return 72-item ``{wind_speed_100m_ms, wind_speed_10m_ms, wind_direction_deg}``.

    ``pred`` is the model prediction (already opened); ``analysis`` is the
    lead-0 input .nc (or None). Everything is extracted at the nearest 0.25 deg
    grid point to (lat, lon).
    """
    pp = _nearest(pred, lat, lon)

    u10 = pp["10m_u_component_of_wind"].values[:12]        # leads 6..72
    v10 = pp["10m_v_component_of_wind"].values[:12]
    u100 = pp["u_component_of_wind"].sel(level=1000).values[:12]
    v100 = pp["v_component_of_wind"].sel(level=1000).values[:12]

    if analysis is not None:
        # lead 0 = IFS fc0; time coord is timedelta64[s] with 0 at the analysis step.
        lead0_idx = int(np.argmin(np.abs(analysis.time.values)))
        ap = _nearest(analysis.isel(batch=0, time=lead0_idx), lat, lon)
        u10_0 = float(ap["10m_u_component_of_wind"])
        v10_0 = float(ap["10m_v_component_of_wind"])
        u100_0 = float(ap["u_component_of_wind"].sel(level=1000))
        v100_0 = float(ap["v_component_of_wind"].sel(level=1000))
    else:
        u10_0, v10_0 = float(u10[0]), float(v10[0])
        u100_0, v100_0 = float(u100[0]), float(v100[0])

    u10_1h = _interp_component(u10_0, u10)
    v10_1h = _interp_component(v10_0, v10)
    u100_1h = _interp_component(u100_0, u100)
    v100_1h = _interp_component(v100_0, v100)

    spd100 = np.sqrt(u100_1h ** 2 + v100_1h ** 2)
    spd10 = np.sqrt(u10_1h ** 2 + v10_1h ** 2)
    direction = _wind_direction_deg(u10_1h, v10_1h)

    return {
        "wind_speed_100m_ms": [round(float(x), 3) for x in spd100],
        "wind_speed_10m_ms": [round(float(x), 3) for x in spd10],
        "wind_direction_deg": [round(float(x), 2) for x in direction],
    }


# ── Site/turbine assembly ──

def build_wind_sites(
    pred_path: Path,
    analysis_path: Path | None,
    catalog: Catalog,
) -> dict[str, dict]:
    """Build ``{site_id: {farm: {...}, turbines: {asset_id: {...}}}}`` for all 8 wind sites."""
    pred = xr.open_dataset(pred_path)
    analysis = xr.open_dataset(analysis_path) if analysis_path else None
    try:
        sites: dict[str, dict] = {}
        for site in catalog.wind_sites:
            site_id = site["site_id"]
            slat, slon = catalog.site_centroid(site_id)
            turbines = {
                t["asset_id"]: wind_point_series(pred, analysis, t["lat"], t["lon"])
                for t in catalog.turbines_by_site[site_id]
            }
            sites[site_id] = {
                "farm": wind_point_series(pred, analysis, slat, slon),
                "turbines": turbines,
            }
        return sites
    finally:
        pred.close()
        if analysis is not None:
            analysis.close()


def _count_values(sites: dict) -> int:
    """Total turbine series (for logging); each wind site also adds one farm series."""
    return sum(len(s["turbines"]) for s in sites.values())
