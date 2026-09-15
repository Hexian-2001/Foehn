"""Normalize, crop, annotate and atomically persist forecast datasets."""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import numpy as np
import xarray as xr

INIT_RE = re.compile(r"predictions_(\d{4}-\d{2}-\d{2}T\d{2})_")
UNIFIED_RE = re.compile(r"_IC(\d{4}-\d{2}-\d{2}T\d{2})_")

REGIONS = {
    "china": {"lat": (15.0, 55.0), "lon": (70.0, 140.0)},
    "global": {"lat": None, "lon": None},
}


def parse_init_time(path: Path) -> np.datetime64:
    """Recover the initialization time from a legacy or unified filename."""
    for pattern in (INIT_RE, UNIFIED_RE):
        match = pattern.search(path.stem)
        if match:
            return np.datetime64(match.group(1))
    raise ValueError(f"cannot parse init time from filename: {path.name}")


def _validate(ds: xr.Dataset) -> None:
    missing = {"time", "lat", "lon"}.difference(ds.coords)
    if missing:
        raise ValueError(f"prediction dataset is missing coordinates: {sorted(missing)}")
    empty = [name for name in ("time", "lat", "lon") if ds.sizes.get(name, 0) == 0]
    if empty:
        raise ValueError(f"prediction dataset has empty dimensions: {empty}")
    if not ds.data_vars:
        raise ValueError("prediction dataset has no data variables")


def crop(ds: xr.Dataset, coord: str, lo: float, hi: float) -> xr.Dataset:
    """Subset a monotonic coordinate to ``[lo, hi]`` in either direction."""
    values = ds[coord].values
    if values[0] > values[-1]:
        return ds.sel({coord: slice(hi, lo)})
    return ds.sel({coord: slice(lo, hi)})


def unify(ds: xr.Dataset, init: np.datetime64) -> xr.Dataset:
    """Remove singleton batch and convert lead times to valid datetimes."""
    _validate(ds)
    if "batch" in ds.dims:
        if ds.sizes["batch"] != 1:
            raise ValueError(f"expected one batch, got {ds.sizes['batch']}")
        ds = ds.isel(batch=0)
    if "batch" in ds.coords:
        ds = ds.drop_vars("batch")

    init_s = init.astype("datetime64[s]")
    raw_time = ds["time"].values
    if np.issubdtype(raw_time.dtype, np.datetime64):
        valid = raw_time.astype("datetime64[s]")
        lead_h = (valid - init_s).astype("timedelta64[h]").astype(int)
    elif np.issubdtype(raw_time.dtype, np.timedelta64):
        valid = (init_s + raw_time.astype("timedelta64[s]")).astype("datetime64[s]")
        lead_h = raw_time.astype("timedelta64[h]").astype(int)
    else:
        raise TypeError(f"time must be datetime64 or timedelta64, got {raw_time.dtype}")

    return ds.assign_coords(
        time=valid,
        init_time=init_s,
        lead_time=("time", lead_h),
    )


def annotate(
    ds: xr.Dataset,
    *,
    model: str,
    variant: str,
    init: np.datetime64,
    region: str,
    lat_box,
    lon_box,
    source: str,
) -> xr.Dataset:
    """Attach the stable metadata contract consumed by downstream tooling."""
    init_text = np.datetime_as_string(init.astype("datetime64[s]"), unit="s") + "Z"
    steps = ds.sizes.get("time", 0)
    lead = ds["lead_time"].values
    step_h = int(np.median(np.diff(lead))) if lead.size > 1 else 0
    horizon_h = int(np.max(lead)) if lead.size else 0
    resolution = (
        float(abs(np.median(np.diff(ds["lon"].values)))) if ds.sizes["lon"] > 1 else 0.0
    )
    attrs = {
        "model": model,
        "variant": variant,
        "init_time": init_text,
        "resolution_deg": resolution,
        "steps": steps,
        "step_h": step_h,
        "horizon_h": horizon_h,
        "region": region,
        "source_file": source,
        "convention": "unified-forecast-1",
    }
    if lat_box is not None:
        attrs["lat_min"], attrs["lat_max"] = map(float, lat_box)
    if lon_box is not None:
        attrs["lon_min"], attrs["lon_max"] = map(float, lon_box)
    ds.attrs = attrs
    return ds


def output_name(
    ds: xr.Dataset, *, model: str, variant: str, init: np.datetime64, region: str
) -> str:
    init_text = np.datetime_as_string(init.astype("datetime64[s]"), unit="h")
    return (
        f"{model}_{variant}_IC{init_text}_STEPS{ds.sizes['time']}_"
        f"{int(ds.attrs['horizon_h'])}h_{ds.attrs['resolution_deg']:g}deg_{region}.nc"
    )


def save_unified(
    ds: xr.Dataset,
    *,
    model: str,
    variant: str,
    init: np.datetime64,
    region: str = "china",
    out_root,
    source: str = "",
    dry_run: bool = False,
    complevel: int = 4,
) -> Path:
    """Write a canonical prediction using an atomic same-directory rename."""
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r} (expected one of {sorted(REGIONS)})")
    region_cfg = REGIONS[region]
    out = ds
    if region_cfg["lat"] is not None:
        out = crop(out, "lat", *region_cfg["lat"])
    if region_cfg["lon"] is not None:
        out = crop(out, "lon", *region_cfg["lon"])
    out = annotate(
        unify(out, init),
        model=model,
        variant=variant,
        init=init,
        region=region,
        lat_box=region_cfg["lat"],
        lon_box=region_cfg["lon"],
        source=source,
    )

    init_dir = np.datetime_as_string(init.astype("datetime64[s]"), unit="m")[:13] + "Z"
    path = Path(out_root) / model / variant / init_dir / "predictions"
    out_path = path / output_name(out, model=model, variant=variant, init=init, region=region)
    if dry_run:
        return out_path

    path.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": complevel} for name in out.data_vars}
    temp_path = path / f".{out_path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.nc"
    try:
        out.to_netcdf(temp_path, encoding=encoding)
        os.replace(temp_path, out_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return out_path
