#!/usr/bin/env python3
# =============================================================================
# Prediction visualizer for the WeatherNext 1 Graph / GraphCast operational run.
# =============================================================================
# Reads a predictions .nc (produced by `scripts/run_inference.py`) and renders
# an industrial-grade figure set:
#
#   1. Animated GIFs        — 2-m temperature, and MSLP + 10-m wind, over the
#                             full 10-day (40 x 6 h) forecast horizon.
#   2. Static 2-D maps      — a curated set of lead times for each variable
#                             (2-m temp, 10-m wind, ~100-m/1000 hPa wind, MSLP,
#                             6-h precip, 500 hPa geopotential).
#   3. Overview grid        — one figure of 2-m temperature at many lead times.
#   4. 1-D time series      — at a chosen lat/lon: 2-m temp, 10-m wind, MSLP,
#                             precip over the horizon.
#
# Coastlines come from Cartopy when available and fall back to a plain
# plate-carree grid otherwise. Output goes to <project>/visualizations/<stem>/.
#
# Usage:
#   python scripts/visualize.py --predictions predictions/predictions_....nc
#   python scripts/visualize.py --predictions ...nc --lat 31.2 --lon 121.5
#   python scripts/visualize.py --predictions ...nc --lead-hours 0 72 240
#   python scripts/visualize.py --predictions ...nc --no-gif --no-maps --series
# =============================================================================

from __future__ import annotations

import argparse
import dataclasses
import io
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no $DISPLAY on Setonix
import matplotlib.pyplot as plt
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except Exception:  # pragma: no cover - depends on the env, not the code
    HAS_CARTOPY = False

import xarray as xr

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

# Project root (this file lives at <root>/scripts/visualize.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INIT_RE = re.compile(r"predictions_(\d{4}-\d{2}-\d{2}T\d{2})_")


# -----------------------------------------------------------------------------
# Field definitions
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class Field:
    """How to load and render one 2-D scalar field from the predictions .nc."""

    name: str                     # output basename
    label: str                    # plot title
    unit: str                     # colorbar label
    cmap: str                     # matplotlib colormap
    vmin: float | None            # fixed color floor (None -> percentile)
    vmax: float | None            # fixed color ceiling (None -> percentile)
    src: str | None = None        # source variable (defaults to name)
    level: float | None = None    # select this pressure level for 3-D vars
    u: str | None = None          # if set, field is wind speed sqrt(u^2+v^2)
    v: str | None = None


# The canonical set of maps. 10m wind uses the surface u/v; "~100 m wind" is
# approximated by the 1000 hPa pressure level (geopotential height ~100 m).
FIELDS = [
    Field("2m_temperature", "2-m temperature", "K", "RdBu_r", 215.0, 315.0),
    Field("mean_sea_level_pressure", "Mean sea-level pressure", "hPa", "RdBu_r", 970.0, 1040.0),
    Field("total_precipitation_6hr", "6-h total precipitation", "mm", "YlGnBu", 0.0, 50.0),
    Field("wind_speed_10m", "10-m wind speed", "m/s", "turbo", 0.0, 30.0,
          u="10m_u_component_of_wind", v="10m_v_component_of_wind"),
    Field("wind_speed_1000hPa", "1000 hPa wind speed (≈100 m)", "m/s", "turbo", 0.0, 40.0,
          u="u_component_of_wind", v="v_component_of_wind", level=1000.0),
    Field("geopotential_500hPa", "500 hPa geopotential", "m$^2$/s$^2$", "viridis", None, None,
          src="geopotential", level=500.0),
]

# Which fields get a GIF (pcolormesh-only). MSLP+wind is a special overlay GIF.
GIF_FIELDS = ["2m_temperature", "wind_speed_10m"]

# Static maps at these lead hours by default (every 24 h + the final step).
DEFAULT_LEAD_HOURS = [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def parse_init_time(path: Path) -> np.datetime64:
    m = INIT_RE.search(path.stem)
    if not m:
        # Fall back: assume the file is unnamed data and use a placeholder.
        return np.datetime64("NaT")
    return np.datetime64(m.group(1).replace("T", "T") + ":00")


def lead_hours(ds: xr.Dataset) -> np.ndarray:
    return ds["time"].values.astype("timedelta64[h]").astype(int)


def load_field(ds: xr.Dataset, f: Field) -> xr.DataArray:
    """Return the field as a (time, lat, lon) DataArray (batch dropped)."""
    if f.u is not None:
        u = ds[f.u].isel(batch=0)
        v = ds[f.v].isel(batch=0)
        if f.level is not None:
            u = u.sel(level=f.level)
            v = v.sel(level=f.level)
        data = np.sqrt(u**2 + v**2)
    else:
        src = f.src or f.name
        data = ds[src].isel(batch=0)
        if "level" in data.dims:
            if f.level is None:
                raise ValueError(f"field {f.name} needs a level")
            data = data.sel(level=f.level)
    return data.rename(f.name)


def color_limits(f: Field, data: np.ndarray) -> tuple[float, float]:
    vmin = f.vmin if f.vmin is not None else float(np.nanpercentile(data, 2))
    vmax = f.vmax if f.vmax is not None else float(np.nanpercentile(data, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = 0.0, 1.0
    return vmin, vmax


def new_map_ax():
    """Build a global plate-carree axis, with coastlines if cartopy is present."""
    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(12, 6.2), subplot_kw={"projection": proj})
        ax.set_global()
        ax.coastlines(resolution="110m", linewidth=0.6, color="0.25")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.6, color="0.4")
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, color="0.7")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}
        return fig, ax, proj
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    return fig, ax, None


def pcolormesh(ax, lon, lat, data, vmin, vmax, cmap, proj):
    kw = dict(vmin=vmin, vmax=vmax, cmap=cmap, shading="nearest")
    if HAS_CARTOPY:
        return ax.pcolormesh(lon, lat, data, transform=proj, **kw)
    return ax.pcolormesh(lon, lat, data, **kw)


def lead_label(lead_h: int) -> str:
    d, h = divmod(lead_h, 24)
    if d and not h:
        return f"T+{d}d"
    if d:
        return f"T+{d}d{h:02d}h"
    return f"T+{h}h"


def fig_to_pil(fig) -> "Image.Image":
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, facecolor="white")
    buf.seek(0)
    return Image.open(buf).copy()


# -----------------------------------------------------------------------------
# 1. Animated GIFs
# -----------------------------------------------------------------------------
def animate_pcolormesh(ds: xr.Dataset, f: Field, out_gif: Path, init_label: str, fps: int):
    lon = ds["lon"].values
    lat = ds["lat"].values
    data = load_field(ds, f).values
    vmin, vmax = color_limits(f, data)
    hours = lead_hours(ds)

    frames = []
    for i, lead in enumerate(hours):
        fig, ax, proj = new_map_ax()
        mesh = pcolormesh(ax, lon, lat, data[i], vmin, vmax, f.cmap, proj)
        ax.set_title(f"{f.label}  [{f.unit}]   init {init_label}   {lead_label(lead)}",
                     fontsize=12)
        fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.045, pad=0.07,
                     label=f.unit)
        frames.append(fig_to_pil(fig))
        plt.close(fig)

    if Image is None:
        # No Pillow: write frames as PNGs and bail on the GIF.
        out_gif.with_suffix(".png")
        raise SystemExit("Pillow not available; cannot build GIF")

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=max(40, int(1000 / fps)), loop=0, optimize=False)
    print(f"  gif: {out_gif}  ({len(frames)} frames, {fps} fps)", flush=True)


def animate_mslp_wind(ds: xr.Dataset, out_gif: Path, init_label: str, fps: int):
    """MSLP as filled field + contours, overlaid with 10-m wind speed."""
    lon = ds["lon"].values
    lat = ds["lat"].values
    mslp = ds["mean_sea_level_pressure"].isel(batch=0).values  # hPa
    wspd = np.sqrt(
        ds["10m_u_component_of_wind"].isel(batch=0).values ** 2
        + ds["10m_v_component_of_wind"].isel(batch=0).values ** 2
    )
    hours = lead_hours(ds)
    vmin, vmax = 0.0, 30.0  # m/s

    frames = []
    for i, lead in enumerate(hours):
        fig, ax, proj = new_map_ax()
        mesh = pcolormesh(ax, lon, lat, wspd[i], vmin, vmax, "turbo", proj)
        # MSLP contours every 4 hPa on top.
        levels = np.arange(950, 1051, 4.0)
        if HAS_CARTOPY:
            cs = ax.contour(lon, lat, mslp[i], levels=levels, colors="k",
                            linewidths=0.6, transform=proj)
        else:
            cs = ax.contour(lon, lat, mslp[i], levels=levels, colors="k", linewidths=0.6)
        ax.clabel(cs, fmt="%d", fontsize=5, inline=True, inline_spacing=2)
        ax.set_title(f"MSLP (contours) + 10-m wind   init {init_label}   {lead_label(lead)}",
                     fontsize=12)
        fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.045, pad=0.07,
                     label="10-m wind speed [m/s]")
        frames.append(fig_to_pil(fig))
        plt.close(fig)

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=max(40, int(1000 / fps)), loop=0, optimize=False)
    print(f"  gif: {out_gif}  ({len(frames)} frames, {fps} fps)", flush=True)


# -----------------------------------------------------------------------------
# 2. Static 2-D maps
# -----------------------------------------------------------------------------
def static_maps(ds: xr.Dataset, f: Field, out_dir: Path, lead_list, init_label: str):
    lon = ds["lon"].values
    lat = ds["lat"].values
    data = load_field(ds, f).values
    print(f"  [{f.name}] shape={data.shape} min={float(np.nanmin(data)):.4g} "
          f"max={float(np.nanmax(data)):.4g} nan={int(np.isnan(data).sum())}/{data.size}",
          flush=True)
    vmin, vmax = color_limits(f, data)
    hours = lead_hours(ds)

    out_dir = out_dir / f.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for lead in lead_list:
        idx = int(np.argmin(np.abs(hours - lead)))
        if abs(hours[idx] - lead) > 6:
            print(f"  skip {f.name} lead {lead}h (no matching step)", flush=True)
            continue
        fig, ax, proj = new_map_ax()
        mesh = pcolormesh(ax, lon, lat, data[idx], vmin, vmax, f.cmap, proj)
        ax.set_title(f"{f.label}  [{f.unit}]   init {init_label}   {lead_label(int(hours[idx]))}",
                     fontsize=12)
        fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.045, pad=0.07,
                     label=f.unit)
        fp = out_dir / f"{f.name}_T+{int(hours[idx]):03d}h.png"
        fig.savefig(fp, dpi=100, facecolor="white")
        plt.close(fig)
    print(f"  maps: {out_dir}  ({len(lead_list)} lead times)", flush=True)


def overview_grid(ds: xr.Dataset, out_path: Path, init_label: str):
    """One figure: 2-m temperature at a grid of lead times."""
    f = next(x for x in FIELDS if x.name == "2m_temperature")
    lon = ds["lon"].values
    lat = ds["lat"].values
    data = load_field(ds, f).values
    vmin, vmax = color_limits(f, data)
    hours = lead_hours(ds)
    leads = [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240]
    leads = [l for l in leads if l <= int(hours[-1])]

    ncols = 4
    nrows = int(np.ceil(len(leads) / ncols))
    proj = ccrs.PlateCarree() if HAS_CARTOPY else None
    fig = plt.figure(figsize=(ncols * 4.2, nrows * 2.4))
    for k, lead in enumerate(leads):
        idx = int(np.argmin(np.abs(hours - lead)))
        if HAS_CARTOPY:
            ax = fig.add_subplot(nrows, ncols, k + 1, projection=proj)
            ax.set_global()
            ax.coastlines(resolution="110m", linewidth=0.4, color="0.25")
        else:
            ax = fig.add_subplot(nrows, ncols, k + 1)
            ax.set_aspect("equal")
        pcolormesh(ax, lon, lat, data[idx], vmin, vmax, f.cmap, proj)
        ax.set_title(lead_label(int(hours[idx])), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"2-m temperature [K]  —  init {init_label}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"  overview: {out_path}", flush=True)


# -----------------------------------------------------------------------------
# 3. 1-D time series at a location
# -----------------------------------------------------------------------------
def time_series(ds: xr.Dataset, out_path: Path, lat, lon, init_label: str):
    # lon may be given as -180..180; the grid is 0..360.
    if lon < 0:
        lon += 360.0
    p = ds.sel(lat=lat, lon=lon, method="nearest").isel(batch=0)
    hours = lead_hours(ds)
    t2m = p["2m_temperature"].values
    mslp = p["mean_sea_level_pressure"].values
    ws10 = np.sqrt(p["10m_u_component_of_wind"].values ** 2
                   + p["10m_v_component_of_wind"].values ** 2)
    prcp = p["total_precipitation_6hr"].values
    loc = f"({float(p['lat'].values):.2f}°, {float(p['lon'].values) % 360:.2f}°)"

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(hours, t2m, "o-", lw=1.8, color="#c0392b")
    axes[0].set_ylabel("2-m temperature [K]")
    axes[1].plot(hours, ws10, "o-", lw=1.8, color="#1f6fb2")
    axes[1].set_ylabel("10-m wind speed [m/s]")
    axes[2].plot(hours, mslp, "o-", lw=1.8, color="#2c3e50")
    axes[2].set_ylabel("MSLP [hPa]")
    axes[2].set_xlabel("Lead time [h]")
    ax2 = axes[1].twinx()
    ax2.bar(hours, prcp, width=5, color="#27ae60", alpha=0.45, label="6-h precip")
    ax2.set_ylabel("6-h precip [mm]", color="#27ae60")
    ax2.set_ylim(0, max(1.0, float(np.nanmax(prcp)) * 1.25))
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Weather at {loc}  —  init {init_label}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  series: {out_path}", flush=True)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Visualize GraphCast predictions.")
    ap.add_argument("--predictions", required=True, type=Path, help="path to predictions .nc")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output root (default: <project>/visualizations/<stem>)")
    ap.add_argument("--fields", nargs="*", default=None,
                    help="subset of field names to render (default: all)")
    ap.add_argument("--lead-hours", nargs="*", type=int, default=None,
                    help="lead times for static maps (default: every 24h)")
    ap.add_argument("--lat", type=float, default=31.23, help="time-series latitude")
    ap.add_argument("--lon", type=float, default=121.47, help="time-series longitude")
    ap.add_argument("--fps", type=int, default=6, help="GIF frames per second")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-maps", action="store_true")
    ap.add_argument("--no-series", action="store_true")
    args = ap.parse_args()

    if not args.predictions.exists():
        raise SystemExit(f"predictions file not found: {args.predictions}")

    ds = xr.open_dataset(args.predictions, engine="netcdf4")
    init = parse_init_time(args.predictions)
    init_label = (
        str(init).replace("T", " ").replace(":00", "Z", 1)
        if str(init) != "NaT" else "?"
    )

    out_root = args.out_dir or (PROJECT_ROOT / "visualizations" / args.predictions.stem)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {out_root}", flush=True)

    # Field selection.
    by_name = {f.name: f for f in FIELDS}
    if args.fields:
        fields = [by_name[n] for n in args.fields]
    else:
        fields = FIELDS

    lead_list = args.lead_hours or DEFAULT_LEAD_HOURS

    # 1. GIFs
    if not args.no_gif and Image is not None:
        for name in GIF_FIELDS:
            if name not in by_name or by_name[name] not in fields:
                continue
            animate_pcolormesh(ds, by_name[name], out_root / "gif" / f"anim_{name}.gif",
                               init_label, args.fps)
        if "mean_sea_level_pressure" in [f.name for f in fields]:
            animate_mslp_wind(ds, out_root / "gif" / "anim_mslp_10mwind.gif",
                              init_label, args.fps)
    elif not args.no_gif and Image is None:
        print("Pillow not installed; skipping GIFs", flush=True)

    # 2. Static maps + overview
    if not args.no_maps:
        for f in fields:
            static_maps(ds, f, out_root / "maps", lead_list, init_label)
        overview_grid(ds, out_root / "overview" / "overview_2m_temperature.png", init_label)

    # 3. Time series
    if not args.no_series:
        time_series(ds, out_root / "series" / "timeseries.png", args.lat, args.lon, init_label)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
