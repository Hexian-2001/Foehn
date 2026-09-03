# =============================================================================
# GRIB -> Aurora input .nc adapter.
# =============================================================================
# The work here is *format alignment*: open-data IFS fields already use the same
# SI units as the ERA5 data Aurora was trained on (verified by probing the raw
# GRIB), so no numerical conversion is needed — only renaming, coordinate
# normalization (roll lon -180..180 -> 0..360, sort levels ascending), and laying
# out the conditioning window (T-6h, T).
#
# The output is written in Aurora's own ``Batch.to_netcdf`` layout (``surf_*`` /
# ``static_*`` / ``atmos_*`` prefixed variables over ``batch``/``history``/
# ``latitude``/``longitude``/``level``), so the GPU stage reads it with the
# upstream ``Batch.from_netcdf`` unchanged. This stage is CPU-only (no torch).
# =============================================================================

from __future__ import annotations

import datetime as dt
import logging
import pickle
from pathlib import Path

import cfgrib
import numpy as np
import xarray as xr

from aurora_forecast import config

logger = logging.getLogger(__name__)


def _read_grib(path: Path) -> xr.Dataset:
    """Read one GRIB file, merging all its hypercubes into a single Dataset."""
    datasets = cfgrib.open_datasets(str(path))
    return xr.merge(datasets, compat="override")


def _normalize(ds: xr.Dataset, var_map: dict[str, str]) -> xr.Dataset:
    """Rename coords/vars, drop GRIB-only coords, realign grid and levels.

    ``var_map`` is authoritative: any data variable not mapped (e.g. ``w``, which
    Aurora does not consume) is dropped.
    """
    coord_rename = {k: v for k, v in config.COORD_RENAME.items() if k in ds.variables}
    ds = ds.rename(coord_rename)
    ds = ds.rename({k: v for k, v in var_map.items() if k in ds.data_vars})
    ds = ds.drop_vars([c for c in config.DROP_COORDS if c in ds.coords])
    ds = ds.drop_vars([v for v in ds.data_vars if v not in var_map.values()])

    # cfgrib emits lon in [-180, 180); ERA5/Aurora use [0, 360). Normalize the
    # range, then roll half the axis (720 of 1440) so it is ascending from 0.
    ds = ds.assign_coords(lon=(ds.lon + 360) % 360)
    ds = ds.roll(lon=config.LON_ROLL, roll_coords=True)

    if "level" in ds.dims:
        ds = ds.sortby("level", ascending=True)
        ds = ds.assign_coords(level=ds.level.astype(int))
    return ds


def read_cycle(date: dt.date, hour: int, raw_root: Path) -> tuple[xr.Dataset, xr.Dataset]:
    """Surface + pressure fields of one analysis cycle, as two Datasets.

    Kept separate because they carry different dims: surface is ``(lat, lon)``,
    pressure is ``(level, lat, lon)``. ``w`` (vertical velocity) is dropped —
    Aurora does not consume it.
    """
    day = date.strftime("%Y-%m-%d")
    cycle_dir = raw_root / day / f"{hour:02d}"
    sfc = _normalize(_read_grib(cycle_dir / "sfc_fc0.grib2"), config.SURFACE_MAP)
    pl = _normalize(_read_grib(cycle_dir / "pl_fc0.grib2"), config.PRESSURE_MAP)
    return sfc, pl


def load_static(static_path: Path) -> dict[str, np.ndarray]:
    """Load the static fields ``{z, slt, lsm}`` from the Aurora pickle.

    The pickle already stores them as ``(721, 1440)`` float32 with latitude
    strictly decreasing (row 0 = 90 N) and longitude ascending [0, 360) — the
    same orientation as the normalized GRIB, so no flip is required.
    """
    with open(static_path, "rb") as f:
        static = pickle.load(f)
    out: dict[str, np.ndarray] = {}
    for key in ("z", "slt", "lsm"):
        if key not in static:
            raise KeyError(f"static pickle missing {key!r}; got {sorted(static)}")
        arr = np.ascontiguousarray(static[key], dtype=np.float32)
        if arr.shape != (721, 1440):
            raise ValueError(f"static {key!r} has shape {arr.shape}, expected (721, 1440)")
        out[key] = arr
    return out


def _filename(date: dt.date, hour: int) -> str:
    return f"{date.strftime('%Y-%m-%d')}T{hour:02d}Z_aurora_input.nc"


def build_input(
    date: dt.date,
    hour: int,
    raw_root: Path | None = None,
    static_path: Path | None = None,
    out_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Build the Aurora input .nc for a forecast initialized at ``date``/``hour``.

    Idempotent: an existing output is skipped unless ``force``. Returns the path
    of the (newly written or already present) output file.
    """
    raw_root = raw_root or config.RAW_IFS_ROOT
    static_path = static_path or config.WEIGHTS_DIR / config.STATIC_PICKLE_NAME
    out_dir = out_dir or config.PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / _filename(date, hour)
    if path.exists() and not force:
        logger.info("already present, skipping %s", path)
        return path

    if not static_path.exists():
        raise FileNotFoundError(f"static pickle not found: {static_path}")

    t = dt.datetime.combine(date, dt.time(hour))
    prev = t - dt.timedelta(hours=config.STEP_HOURS)

    sfc_prev, pl_prev = read_cycle(prev.date(), prev.hour, raw_root)
    sfc_cur, pl_cur = read_cycle(date, hour, raw_root)

    # Conditioning window along a fresh "history" dim: [T-6h, T].
    sfc = xr.concat([sfc_prev, sfc_cur], dim="history")
    pl = xr.concat([pl_prev, pl_cur], dim="history")

    lat = sfc.lat.values  # decreasing 90 -> -90 (already Aurora-compatible)
    lon = sfc.lon.values  # ascending [0, 360)
    levels = np.asarray(config.PRESSURE_LEVELS, dtype=int)

    static = load_static(static_path)

    surf_vars = {
        f"surf_{k}": (("batch", "history", "latitude", "longitude"),
                      sfc[k].values[None, ...])
        for k in ("2t", "10u", "10v", "msl")
    }
    atmos_vars = {
        f"atmos_{k}": (("batch", "history", "level", "latitude", "longitude"),
                       pl[k].values[None, ...])
        for k in ("z", "u", "v", "t", "q")
    }
    static_vars = {
        f"static_{k}": (("latitude", "longitude"), static[k])
        for k in ("z", "slt", "lsm")
    }

    out = xr.Dataset(
        data_vars={**surf_vars, **atmos_vars, **static_vars},
        coords={
            "latitude": ("latitude", lat),
            "longitude": ("longitude", lon),
            "level": ("level", levels),
            # metadata.time is per-batch-element: one value for the single batch.
            "time": ("batch", np.atleast_1d(np.datetime64(t))),
            "rollout_step": 0,
        },
    )

    encoding = {v: {"zlib": True, "complevel": 4, "shuffle": True} for v in out.data_vars}
    out.to_netcdf(path, engine="netcdf4", encoding=encoding)
    logger.info("wrote %s", path)
    return path
