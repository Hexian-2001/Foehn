"""Convert raw open-data GRIB into a GraphCast input .nc.

The work here is *format alignment*, not physics: open-data IFS fields already
use the same SI units as the ERA5 data GraphCast was trained on (verified by
probing the raw GRIB), so no numerical conversion is needed — only renaming,
coordinate normalization (roll lon 180->0, sort levels ascending), and laying
out the full forecast time axis with the future steps NaN-filled.

The model reads only the first 2 steps (T-6h, T) as its conditioning window; the
remaining steps are a shape/coordinate template plus the ``datetime`` axis used
by the downstream pipeline to derive solar/time forcings.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import cfgrib
import numpy as np
import xarray as xr

from data_processing import config

logger = logging.getLogger(__name__)


def _read_grib(path: Path) -> xr.Dataset:
    """Read one GRIB file, merging all its hypercubes into a single Dataset."""
    datasets = cfgrib.open_datasets(str(path))
    return xr.merge(datasets, compat="override")


def _normalize(ds: xr.Dataset, var_map: dict[str, str]) -> xr.Dataset:
    """Rename coords/vars, drop GRIB-only coords, realign grid and levels."""
    coord_rename = {k: v for k, v in config.COORD_RENAME.items() if k in ds.variables}
    ds = ds.rename(coord_rename)
    ds = ds.rename({k: v for k, v in var_map.items() if k in ds.data_vars})
    ds = ds.drop_vars([c for c in config.DROP_COORDS if c in ds.coords])

    # cfgrib emits lon in [-180, 180); GraphCast/ERA5 use [0, 360). Normalize
    # the range, then roll half the axis so it is ascending from 0 (the solar
    # forcing in the downstream pipeline depends on these VALUES, not just the
    # ordering).
    ds = ds.assign_coords(lon=(ds.lon + 360) % 360)
    ds = ds.roll(lon=config.LON_ROLL, roll_coords=True)

    if "level" in ds.dims:
        ds = ds.sortby("level", ascending=True)
        # Levels are exact integers (50..1000); keep them as int so the
        # downstream `sel(level=[50, 100, ...])` matches exactly.
        ds = ds.assign_coords(level=ds.level.astype(int))
    return ds


def read_cycle(date: dt.date, hour: int, raw_root: Path) -> xr.Dataset:
    """Physical fields (surface + pressure) of one analysis cycle, no time dim."""
    day = date.strftime("%Y-%m-%d")
    cycle_dir = raw_root / day / f"{hour:02d}"
    sfc = _normalize(_read_grib(cycle_dir / "sfc_fc0.grib2"), config.SURFACE_MAP)
    pl = _normalize(_read_grib(cycle_dir / "pl_fc0.grib2"), config.PRESSURE_MAP)
    return xr.merge([sfc, pl], compat="override")


def read_static(raw_root: Path) -> xr.Dataset:
    """Time-invariant static fields (geopotential_at_surface, land_sea_mask)."""
    return _normalize(_read_grib(raw_root / "static" / "static.grib2"), config.STATIC_MAP)


def _filename(date: dt.date) -> str:
    return config.FILENAME_TPL.format(
        date=date.strftime("%Y-%m-%d"),
        res=str(config.RESOLUTION),
        nlevels=len(config.PRESSURE_LEVELS),
        nsteps=config.TARGET_STEPS,
    )


def build_input(
    date: dt.date,
    hour: int,
    raw_root: Path | None = None,
    out_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Build the GraphCast input .nc for a forecast initialized at `date`/`hour`.

    Idempotent: an existing output is skipped unless ``force``. Returns the path
    of the (newly written or already present) output file.
    """
    raw_root = raw_root or config.RAW_IFS_ROOT
    out_dir = out_dir or config.PROCESSED_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / _filename(date)
    if path.exists() and not force:
        logger.info("already present, skipping %s", path)
        return path

    t = dt.datetime.combine(date, dt.time(hour))
    prev = t - dt.timedelta(hours=config.STEP_HOURS)

    ds_prev = read_cycle(prev.date(), prev.hour, raw_root)
    ds_cur = read_cycle(date, hour, raw_root)

    # Full time axis: lead hours relative to init time T (timedelta), 42 steps.
    time = np.asarray(config.LEAD_HOURS).astype("timedelta64[h]")
    init_dt = np.datetime64(t)
    datetime = init_dt + time  # absolute datetime64, (time,)

    # Physical variables: 2 real steps, future steps become NaN templates.
    phys = xr.concat([ds_prev, ds_cur], dim="time")
    phys = phys.assign_coords(time=time[:2])
    phys = phys.reindex(time=time)

    # Static fields (no time dim) + batch dim.
    out = xr.merge([phys, read_static(raw_root)], compat="override")
    out = out.expand_dims(batch=[0])

    # datetime must be a (batch, time) coord — the downstream pipeline relies on
    # it (inference.py reads example_batch["datetime"].isel(time=-1)).
    out = out.assign_coords(datetime=(("batch", "time"), datetime[None, :]))

    # The operational model predicts `total_precipitation_6hr` but does not take
    # it as input (HRES-fc0 carries no precip). Stage 3 selects it as part of the
    # target template, so it must exist as a variable; NaN is fine because the
    # template is overwritten with NaN at inference time anyway.
    out = out.assign(total_precipitation_6hr=out["2m_temperature"] * np.nan)

    out = out.transpose("batch", "time", "level", "lat", "lon", missing_dims="ignore")

    # netCDF-4 (HDF5) engine + zlib: the future steps are all NaN and the real
    # data is smooth, so compression shrinks a ~14 GB logical file to a few
    # hundred MB. The scipy writer (netCDF-3) caps a variable at 2 GB, which a
    # single 0.25-deg pressure variable exceeds — hence the explicit engine.
    encoding = {
        v: {"zlib": True, "complevel": 4, "shuffle": True} for v in out.data_vars
    }
    out.to_netcdf(path, engine="netcdf4", encoding=encoding)
    logger.info("wrote %s", path)
    return path
