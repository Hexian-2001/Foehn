"""Extract point wind series from a model prediction .nc and build the 72-hour
1-hourly platform arrays.

Model output contract (``unified-forecast-1``):
  - 40 steps x 6h (lead 6..240 h), absolute ``time`` + integer ``lead_time``
    coords, 0.25 deg China crop (lat 55..15 desc, lon 70..140 asc).
  - ``10m_u/v_component_of_wind``  -> 10 m wind components.
  - ``u/v_component_of_wind`` @ level=1000 -> 100 m wind proxy (project convention).

The platform now issues every forecast at a fixed 16:00Z start (``FORECAST_ISSUANCE_UTC``,
= next-day Beijing 00:00) and wants lead 0..71 at 1 h, where lead 0 is that
16:00Z instant. We compute the model lead offset of the issuance relative to the
prediction's ``init_time``, then linearly interpolate the 6h model steps onto the
72 one-hourly points. In the daily flow the offset is always >= 6 h (the cycle is
same-day, the issuance is next-day), so the window lands on/between model steps
and no lead-0 analysis anchor is required.
"""

from __future__ import annotations

import glob
import logging
import re
from pathlib import Path

import numpy as np
import xarray as xr

from .config import RESULTS_ROOT, LEAD_TIME_COUNT
from .assets_map import Catalog

logger = logging.getLogger("push_to_platform.extract")

_INIT_DIR_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2})Z")


def _to_dt64(iso: str) -> np.datetime64:
    """Parse an ISO-8601 UTC instant into a tz-naive ``datetime64[s]``.

    ``Z`` is stripped first because numpy has no timezone representation and
    otherwise emits a warning on every parse; all timestamps here are UTC.
    """
    return np.datetime64(iso.replace("Z", ""), "s")


# ── File discovery ──

def find_prediction_nc(model: str, variant: str, target_start_iso: str) -> Path:
    """Locate the newest prediction .nc whose init time is <= ``target_start_iso``.

    Predictions live under ``results/<model>/<variant>/<init>Z/predictions/*.nc``.
    We pick the latest init cycle that is not later than the target issuance, so a
    stale cycle from a previous day never produces a forecast for a time before
    its own analysis.
    """
    base = RESULTS_ROOT / model / variant
    target64 = _to_dt64(target_start_iso)
    best: tuple[np.datetime64, Path] | None = None
    for d in sorted(base.glob("*T*Z")):
        m = _INIT_DIR_RE.fullmatch(d.name)
        if not m:
            continue
        init64 = np.datetime64(m.group(1) + ":00:00", "s")
        if init64 > target64:
            continue
        preds = sorted(d.glob("predictions/*.nc"))
        if not preds:
            continue
        if best is None or init64 > best[0]:
            best = (init64, preds[-1])
    if best is None:
        raise FileNotFoundError(
            f"no prediction .nc for {model}/{variant} with init <= {target_start_iso}: {base}"
        )
    logger.info("prediction init=%s file=%s", np.datetime_as_string(best[0], unit="h"), best[1].name)
    return best[1]


# ── Point series ──

def _nearest(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    return ds.sel(lat=lat, lon=lon, method="nearest")


def _wind_direction_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological direction (degrees FROM which wind blows, 0=N, 90=E)."""
    return (180.0 + np.degrees(np.arctan2(u, v))) % 360.0


def _init_datetime64(pred: xr.Dataset) -> np.datetime64:
    """Recover the init instant (UTC) of a prediction, from attrs then coords."""
    attr = pred.attrs.get("init_time")
    if attr:
        return np.datetime64(str(attr).replace("Z", ""), "s")
    init = pred["init_time"].values
    return init.astype("datetime64[s]").reshape(-1)[0]


def wind_point_series(
    pred: xr.Dataset,
    lat: float,
    lon: float,
    target_start_iso: str,
) -> dict[str, list[float | None]]:
    """Return 72-item ``{wind_speed_100m_ms, wind_speed_10m_ms, wind_direction_deg}``.

    Values are extracted at the nearest 0.25 deg grid point to (lat, lon) and
    interpolated from the 6h model steps onto the 72 one-hourly leads whose lead 0
    is ``target_start_iso`` (``YYYY-MM-DDTHH:MM:SS``, UTC).
    """
    pp = _nearest(pred, lat, lon)
    init64 = _init_datetime64(pred)
    target64 = _to_dt64(target_start_iso)
    lead_start = int((target64 - init64) / np.timedelta64(1, "h"))
    target_leads = np.arange(lead_start, lead_start + LEAD_TIME_COUNT, dtype=float)

    model_leads = pred["lead_time"].values.astype(float)  # [6, 12, ..., 240]
    if lead_start < model_leads[0] or lead_start + LEAD_TIME_COUNT - 1 > model_leads[-1]:
        raise ValueError(
            f"target window leads [{lead_start}..{lead_start + LEAD_TIME_COUNT - 1}] "
            f"fall outside model steps [{model_leads[0]}..{model_leads[-1]}]; "
            f"pick a prediction init earlier than {target_start_iso}"
        )

    u10 = pp["10m_u_component_of_wind"].values.astype(float)
    v10 = pp["10m_v_component_of_wind"].values.astype(float)
    u100 = pp["u_component_of_wind"].sel(level=1000).values.astype(float)
    v100 = pp["v_component_of_wind"].sel(level=1000).values.astype(float)

    u10_1h = np.interp(target_leads, model_leads, u10)
    v10_1h = np.interp(target_leads, model_leads, v10)
    u100_1h = np.interp(target_leads, model_leads, u100)
    v100_1h = np.interp(target_leads, model_leads, v100)

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
    target_start_iso: str,
    catalog: Catalog,
) -> dict[str, dict]:
    """Build ``{site_id: {farm: {...}, turbines: {asset_id: {...}}}}`` for all 8 wind sites."""
    pred = xr.open_dataset(pred_path)
    try:
        sites: dict[str, dict] = {}
        for site in catalog.wind_sites:
            site_id = site["site_id"]
            slat, slon = catalog.site_centroid(site_id)
            turbines = {
                t["asset_id"]: wind_point_series(pred, t["lat"], t["lon"], target_start_iso)
                for t in catalog.turbines_by_site[site_id]
            }
            sites[site_id] = {
                "farm": wind_point_series(pred, slat, slon, target_start_iso),
                "turbines": turbines,
            }
        return sites
    finally:
        pred.close()
