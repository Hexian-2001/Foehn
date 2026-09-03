# =============================================================================
# Model-agnostic prediction store: unify + crop + save into the results tree.
# =============================================================================
# One shared, decoupled contract for EVERY forecast model (GraphCast / WeatherNext
# and whatever ships next). It turns a raw model output Dataset into a
# self-describing, CF-friendly prediction and writes it into the EXTERNAL,
# model-organized results tree:
#
#     <results>/<model>/<variant>/<init>Z/predictions/
#         <model>_<variant>_IC<init>_STEPS<n>_<horizon>h_<res>deg_<region>.nc
#
# Unified schema (model-agnostic) — the contract downstream tooling
# (visualize.py, consumers) is written against:
#
#   * the singleton `batch` dim is dropped
#   * `time` becomes the ACTUAL valid datetime (init + lead), not a timedelta
#   * scalar coord `init_time` records the run's initialization time
#   * `lead_time` (hours since init) is kept as an auxiliary coord
#   * global attrs document model / variant / init / resolution / steps / horizon
#     / region bounds / source
#
# Used by:
#   * `scripts/crop_region.py`  — crop + relocate an already-saved global .nc
#   * `weathernext_forecast.inference` — save the China region directly, never
#     materializing the full-global ~13 GB file.
# =============================================================================

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import xarray as xr

INIT_RE = re.compile(r"predictions_(\d{4}-\d{2}-\d{2}T\d{2})_")
UNIFIED_RE = re.compile(r"_IC(\d{4}-\d{2}-\d{2}T\d{2})_")

# Named regions (lat_min, lat_max, lon_min, lon_max). The China box is the country
# plus a generous weather buffer (upstream systems included). `global` = no crop.
REGIONS = {
    "china": dict(lat=(15.0, 55.0), lon=(70.0, 140.0)),
    "global": dict(lat=None, lon=None),
}


def parse_init_time(path: Path) -> np.datetime64:
    """Recover the init time from a filename (legacy or unified naming)."""
    for rx in (INIT_RE, UNIFIED_RE):
        m = rx.search(path.stem)
        if m:
            return np.datetime64(m.group(1))  # "2026-08-27T00" -> datetime64
    raise ValueError(f"cannot parse init time from filename: {path.name}")


def crop(ds: xr.Dataset, coord: str, lo: float, hi: float) -> xr.Dataset:
    """Subset a monotonic coordinate to [lo, hi], regardless of its direction."""
    c = ds[coord].values
    if c[0] > c[-1]:  # descending (e.g. lat 90 -> -90)
        return ds.sel({coord: slice(hi, lo)})
    return ds.sel({coord: slice(lo, hi)})


def unify(ds: xr.Dataset, init: np.datetime64) -> xr.Dataset:
    """Normalize a raw prediction into the model-agnostic schema (no batch,
    absolute time)."""
    if "batch" in ds.dims:
        ds = ds.isel(batch=0)
    if "batch" in ds.coords:
        ds = ds.drop_vars("batch")  # drop the leftover scalar coord from isel

    init_s = init.astype("datetime64[s]")
    valid = (init_s + ds["time"].values.astype("timedelta64[s]")).astype("datetime64[s]")
    lead_h = ds["time"].values.astype("timedelta64[h]").astype(int)

    ds = ds.assign_coords(time=valid)
    ds = ds.assign_coords(init_time=init_s)
    ds = ds.assign_coords(lead_time=("time", lead_h))
    return ds


def annotate(ds: xr.Dataset, *, model: str, variant: str, init: np.datetime64,
             region: str, lat_box, lon_box, source: str) -> xr.Dataset:
    """Attach the machine-readable metadata that makes the file self-describing."""
    init_s = np.datetime_as_string(init.astype("datetime64[s]"), unit="s") + "Z"
    n_steps = ds.sizes.get("time", 0)
    lead = ds["lead_time"].values if "lead_time" in ds.coords else np.zeros(0)
    step_h = int(np.max(lead)) // max(n_steps - 1, 1) if n_steps > 1 else 0
    horizon_h = int(np.max(lead)) if lead.size else 0
    resolution = float(np.median(np.diff(ds["lon"].values))) if ds.sizes["lon"] > 1 else 0.0

    attrs = {
        "model": model,
        "variant": variant,
        "init_time": init_s,
        "resolution_deg": resolution,
        "steps": n_steps,
        "step_h": step_h,
        "horizon_h": horizon_h,
        "region": region,
        "source_file": source,
        "convention": "unified-forecast-1",
    }
    if lat_box is not None:
        attrs["lat_min"], attrs["lat_max"] = float(lat_box[0]), float(lat_box[1])
    if lon_box is not None:
        attrs["lon_min"], attrs["lon_max"] = float(lon_box[0]), float(lon_box[1])
    ds.attrs = attrs
    return ds


def output_name(ds: xr.Dataset, *, model: str, variant: str, init: np.datetime64,
                region: str) -> str:
    """Build the canonical filename from the (already-annotated) dataset."""
    n_steps = ds.sizes["time"]
    horizon_h = int(ds.attrs["horizon_h"])
    res = ds.attrs["resolution_deg"]
    init_s = np.datetime_as_string(init.astype("datetime64[s]"), unit="h")
    return (f"{model}_{variant}_IC{init_s}_STEPS{n_steps}_"
            f"{horizon_h}h_{res:g}deg_{region}.nc")


def save_unified(ds: xr.Dataset, *, model: str, variant: str, init: np.datetime64,
                 region: str = "china", out_root, source: str = "",
                 dry_run: bool = False, complevel: int = 4) -> Path:
    """Crop, unify, annotate and write `ds` into the results tree; return the path.

    Does NOT mutate or close `ds` — it works on derived copies.
    """
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r} (expected one of {sorted(REGIONS)})")
    region_cfg = REGIONS[region]
    lat_box = region_cfg["lat"]
    lon_box = region_cfg["lon"]

    out = ds
    if lat_box is not None:
        out = crop(out, "lat", *lat_box)
    if lon_box is not None:
        out = crop(out, "lon", *lon_box)

    out = unify(out, init)
    out = annotate(out, model=model, variant=variant, init=init, region=region,
                   lat_box=lat_box, lon_box=lon_box, source=source)

    init_dir = np.datetime_as_string(init.astype("datetime64[s]"), unit="m")[:13] + "Z"
    name = output_name(out, model=model, variant=variant, init=init, region=region)
    out_path = Path(out_root) / model / variant / init_dir / "predictions" / name

    if dry_run:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": complevel} for v in out.data_vars}
    out.to_netcdf(out_path, encoding=enc)
    return out_path
