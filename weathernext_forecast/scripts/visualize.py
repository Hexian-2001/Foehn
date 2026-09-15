#!/usr/bin/env python3
# =============================================================================
# Forecast visualizer — publication-grade (top-journal) figure set.
# =============================================================================
# Reads a GraphCast/WeatherNext predictions .nc and renders a compact, decoupled
# figure set into an EXTERNAL results tree (not inside the model project):
#
#     <results>/<model>/<variant>/<init>Z/visualizations/
#         series/   timeseries_<city>_<lat>N_<lon>E.png   (one 2x2 figure per city)
#         gif/      anim_2m_temperature.gif | anim_wind_10m.gif
#                   anim_wind_100m.gif | anim_mslp_wind10m.gif
#         overview/ 2m_temperature_40steps.png | wind_10m_40steps.png
#                   wind_100m_40steps.png   (small-multiples, all 40 lead times)
#
# Conventions (meteorological, colorblind-safe, non-rainbow):
#   * 2-m temperature  K  -> degC          (RdBu_r, diverging)
#   * wind speed       m/s                 (viridis, sequential, floor at 0)
#   * MSLP             Pa  -> hPa          (RdBu_r)
#   * 6-h precip       m   -> mm           (YlGnBu, sequential, floor at 0)
#   * three cities use Okabe-Ito hues (CVD-safe): Beijing/SH/GZ = orange/blue/green
#   * time axes show ACTUAL timestamps (init + lead), never bare lead-hours
#
# Accepts BOTH the legacy GraphCast file (batch dim + timedelta `time`) and the
# unified results file (no batch, absolute `time` + `init_time`/`lead_time`
# coords). Actual valid times are reconstructed from init + lead when the file
# carries only a timedelta coordinate.
#
# Usage:
#   python scripts/visualize.py --predictions predictions/predictions_....nc
#   python scripts/visualize.py --predictions ... --model graphcast --variant operational
#   python scripts/visualize.py --predictions ... --no-gif --no-overview
# =============================================================================

from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no $DISPLAY on Setonix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


# -----------------------------------------------------------------------------
# Paths & run identity
# -----------------------------------------------------------------------------
# script lives at <repo>/weathernext_forecast/scripts/visualize.py
REPO_ROOT = Path(__file__).resolve().parents[2]  # shared repo root (Foehn/ etc.)

INIT_RE = re.compile(r"predictions_(\d{4}-\d{2}-\d{2}T\d{2})_")
UNIFIED_RE = re.compile(r"_IC(\d{4}-\d{2}-\d{2}T\d{2})_")

# Okabe-Ito categorical hues (colorblind-safe), one per city.
CITIES = [
    ("Beijing", 39.90, 116.40, "#D55E00", "o"),
    ("Shanghai", 31.23, 121.47, "#0072B2", "s"),
    ("Guangzhou", 23.13, 113.26, "#009E73", "^"),
]

# Full, human-readable model names for figure titles. Top-journal figures identify
# the source model explicitly (not just a short slug), so every title carries the
# model's full name + variant instead of `graphcast` / `aurora` directory names.
MODEL_DISPLAY_NAMES = {
    ("graphcast", "operational"): "GraphCast (WeatherNext 1 Graph, operational 0.25°)",
    ("aurora", "0.25-finetuned"): "Aurora 0.25° fine-tuned (IFS HRES T0)",
}


def model_display_name(model: str, variant: str) -> str:
    """Return the full display name for a (model, variant) pair, with a safe fallback."""
    key = (str(model).strip().lower(), str(variant).strip().lower())
    if key in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[key]
    variant = str(variant).strip()
    return f"{str(model).strip().title()}{' ' + variant if variant else ''}".strip()


# -----------------------------------------------------------------------------
# Field definitions
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class Field:
    name: str            # output basename
    label: str           # plot title / legend label
    unit: str            # colorbar / axis label
    cmap: str            # matplotlib colormap
    src: str | None = None      # source variable (defaults to name)
    level: int | None = None    # select this pressure level for 3-D vars
    u: str | None = None        # if set, field = sqrt(u^2 + v^2) at level
    v: str | None = None
    scale: float = 1.0          # multiplicative unit conversion
    offset: float = 0.0         # additive unit conversion (K -> degC)
    vmin_zero: bool = False     # clamp color floor at 0 (wind, precip)
    symmetric: bool = False     # diverging map centered on the field median


FIELDS = [
    Field("2m_temperature", "2-m temperature", "°C", "RdBu_r",
          offset=-273.15, symmetric=True),
    Field("wind_10m", "10-m wind speed", "m/s", "viridis",
          u="10m_u_component_of_wind", v="10m_v_component_of_wind", vmin_zero=True),
    Field("wind_100m", "100-m wind speed (1000 hPa)", "m/s", "viridis",
          u="u_component_of_wind", v="v_component_of_wind", level=1000, vmin_zero=True),
    Field("mean_sea_level_pressure", "Mean sea-level pressure", "hPa", "RdBu_r",
          scale=0.01),
    Field("total_precipitation_6hr", "6-h total precipitation", "mm", "YlGnBu",
          scale=1000.0, vmin_zero=True),
]

BY_NAME = {f.name: f for f in FIELDS}


def field_available(ds: xr.Dataset, f: Field) -> bool:
    """Whether every source variable a field needs is present in `ds`.

    Lets the visualizer run unchanged on models that predict a different variable
    set (e.g. Aurora has no ``total_precipitation_6hr``): missing fields are
    skipped with a note rather than raising.
    """
    if f.u is not None:
        # Derived field (e.g. wind speed from u/v components): only the
        # component variables are needed, not the field's output name.
        needed = {f.u, f.v}
    else:
        needed = {f.src or f.name}
    return needed <= set(ds.variables)

# The condensed overview covers these three (the user's focus: wind + temperature).
OVERVIEW_FIELDS = ["2m_temperature", "wind_10m", "wind_100m"]
GIF_FIELDS = ["2m_temperature", "wind_10m", "wind_100m"]


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------
def parse_init_time(path: Path) -> np.datetime64:
    for rx in (INIT_RE, UNIFIED_RE):
        m = rx.search(path.stem)
        if m:
            return np.datetime64(m.group(1))  # "2026-08-27T00" -> datetime64
    raise SystemExit(f"cannot parse init time from filename: {path.name}")


def valid_times(ds: xr.Dataset, init: np.datetime64):
    """Actual valid timestamps (init + lead) and lead hours.

    Handles both schemas: the legacy GraphCast file carries a `time` (timedelta)
    coord, while the unified results file carries an absolute `time` (datetime).
    """
    t = ds["time"]
    init_s = init.astype("datetime64[s]")
    if np.issubdtype(t.dtype, np.timedelta64):
        lead = t.values.astype("timedelta64[h]").astype(int)
        valid = (init_s + t.values).astype("datetime64[m]")
        return valid, lead
    # unified: `time` is already the absolute valid time
    lead = (t.values.astype("datetime64[s]") - init_s).astype("timedelta64[h]").astype(int)
    valid = t.values.astype("datetime64[m]")
    return valid, lead


def _drop_batch(da):
    """Remove the singleton batch dim/coord if present (legacy schema)."""
    if "batch" in da.dims:
        da = da.isel(batch=0)
    if "batch" in da.coords:
        da = da.drop_vars("batch")
    return da


def load_field(ds: xr.Dataset, f: Field) -> np.ndarray:
    """Return the field as a (time, lat, lon) float32 array, batch dropped."""
    if f.u is not None:
        u = _drop_batch(ds[f.u])
        v = _drop_batch(ds[f.v])
        if f.level is not None:
            u = u.sel(level=f.level)
            v = v.sel(level=f.level)
        data = np.sqrt(u.values.astype(np.float32) ** 2 + v.values.astype(np.float32) ** 2)
    else:
        src = f.src or f.name
        da = _drop_batch(ds[src])
        if f.level is not None:
            da = da.sel(level=f.level)
        data = da.values.astype(np.float32)
    if f.scale != 1.0:
        data = data * f.scale
    if f.offset != 0.0:
        data = data + f.offset
    return data


def color_limits(f: Field, data: np.ndarray) -> tuple[float, float]:
    """Robust [vmin, vmax] honoring the field's color semantics.

    * ``vmin_zero``: floor at 0 (wind speed, precipitation — non-negative).
    * ``symmetric``: diverging map centered on the field median, so the map's
      neutral midpoint is meaningful (blue = cool, red = warm) instead of the
      washed-out floating center a raw percentile span produces.
    * otherwise: 2nd--98th percentile (robust to outliers).
    """
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    if f.vmin_zero:
        vmin = 0.0
        vmax = float(np.nanpercentile(finite, 98))
    elif f.symmetric:
        mid = float(np.nanmedian(finite))
        span = float(np.nanpercentile(finite, 98) - np.nanpercentile(finite, 2))
        vmin, vmax = mid - span / 2.0, mid + span / 2.0
    else:
        vmin = float(np.nanpercentile(finite, 2))
        vmax = float(np.nanpercentile(finite, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = 0.0, 1.0
    return vmin, vmax


def region_aspect(lon, lat) -> float:
    """PlateCarree display aspect (width/height) of a lon/lat grid.

    The China operational crop spans 70° lon x 40° lat -> aspect 1.75. Matching
    each overview panel to this aspect (instead of an arbitrary cell shape) lets
    every map fill its panel edge-to-edge with no distortion and no letterboxing
    whitespace between subfigures.
    """
    dlon = float(np.ptp(np.asarray(lon, dtype=float)))
    dlat = float(np.ptp(np.asarray(lat, dtype=float)))
    if dlat <= 0:
        return 1.0
    return max(dlon / dlat, 0.5)


# -----------------------------------------------------------------------------
# Map axes
# -----------------------------------------------------------------------------
def set_data_extent(ax, lon, lat, proj=None):
    """Fit a map axis to the data grid instead of shrinking regional data.

    Prediction files are commonly cropped before visualisation (the operational
    China files cover 70--140 E, 15--55 N).  Calling ``set_global`` for those
    files makes the useful field occupy only a few percent of every panel.
    Global files still use Cartopy's global extent.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = lon[np.isfinite(lon)]
    lat = lat[np.isfinite(lat)]
    if lon.size == 0 or lat.size == 0:
        return

    west, east = float(lon.min()), float(lon.max())
    south, north = float(lat.min()), float(lat.max())
    is_global = (east - west) >= 350.0 and (north - south) >= 170.0

    if HAS_CARTOPY:
        if is_global:
            ax.set_global()
        else:
            # Include half a grid cell so the boundary pixels are not clipped.
            unique_lon = np.unique(lon)
            unique_lat = np.unique(lat)
            dx = (float(np.median(np.abs(np.diff(unique_lon))))
                  if unique_lon.size > 1 else 0.0)
            dy = (float(np.median(np.abs(np.diff(unique_lat))))
                  if unique_lat.size > 1 else 0.0)
            ax.set_extent([west - dx / 2, east + dx / 2,
                           south - dy / 2, north + dy / 2], crs=proj)
    else:
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)


def new_global_ax(figsize=(8.9, 5.0), gridlines=True, border=True,
                  lon=None, lat=None):
    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
        if lon is None or lat is None:
            ax.set_global()
        else:
            set_data_extent(ax, lon, lat, proj)
        ax.coastlines(resolution="110m", linewidth=0.5, color="0.28")
        if border:
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5,
                           edgecolor="0.45")
        if gridlines:
            ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4, color="0.65")
        # Tight frame: title strip above, colorbar strip below, map fills the rest
        # (drop default subplot margins; no gridline labels in a moving animation).
        # Reserve enough room for a two-line title (field + model) and the
        # horizontal colorbar with its tick labels.
        fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.15)
        return fig, ax, proj
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    return fig, ax, None


def pcolormesh(ax, lon, lat, data, vmin, vmax, cmap, proj, shading="nearest"):
    kw = dict(vmin=vmin, vmax=vmax, cmap=cmap, shading=shading)
    if HAS_CARTOPY:
        return ax.pcolormesh(lon, lat, data, transform=proj, **kw)
    return ax.pcolormesh(lon, lat, data, **kw)


def fmt_valid(dt64) -> str:
    return np.datetime_as_string(dt64, unit="m").replace("T", " ") + "Z"


def fmt_init(init: np.datetime64) -> str:
    return np.datetime_as_string(init, unit="m").replace("T", " ")


# -----------------------------------------------------------------------------
# 1. Time series — one figure per city, actual timestamps on the x-axis
# -----------------------------------------------------------------------------
def fmt_latlon(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}°{ns}, {abs(lon):.2f}°{ew}"


def fmt_latlon_file(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}{ns}_{abs(lon):.2f}{ew}"


def _write_series_csv(csv_path: Path, valid, cols: dict) -> None:
    """Write the per-city time series as a CSV alongside the figure.

    ``cols`` maps column name -> 1-D array of values (or ``None`` when the field
    is not predicted by this model). Missing fields become empty cells so the
    schema stays stable across GraphCast (has precipitation) and Aurora (no
    precipitation).
    """
    names = list(cols.keys())
    valid_str = [np.datetime_as_string(v, unit="m").replace("T", " ") + "Z" for v in valid]
    rows = []
    for i in range(len(valid_str)):
        cells = [valid_str[i]]
        for name in names:
            arr = cols[name]
            cells.append("" if arr is None else f"{float(arr[i]):.4f}")
        rows.append(cells)
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["valid_time_utc"] + names)
        w.writerows(rows)
    print(f"  series csv: {csv_path}", flush=True)


def time_series_city(ds: xr.Dataset, init, valid, city, out_path: Path, model_label: str):
    name, lat, lon, color, marker = city
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.6), sharex=True)

    # (variable key, y-axis label, unit offset, unit scale, source vars, csv column)
    panels = [
        ("2m_temperature", "2-m temperature [°C]", -273.15, 1.0,
         ("2m_temperature",), "2m_temperature_degC"),
        ("wind_10m", "10-m wind speed [m/s]", 0.0, 1.0,
         ("10m_u_component_of_wind", "10m_v_component_of_wind"), "10m_wind_speed_m_s"),
        ("mean_sea_level_pressure", "MSLP [hPa]", 0.0, 0.01,
         ("mean_sea_level_pressure",), "mslp_hPa"),
        ("total_precipitation_6hr", "6-h precipitation [mm]", 0.0, 1000.0,
         ("total_precipitation_6hr",), "total_precipitation_6hr_mm"),
    ]
    x = mdates.date2num(valid.astype("datetime64[s]").astype(object).tolist())

    csv_cols: dict[str, np.ndarray | None] = {}
    p = _drop_batch(ds.sel(lat=lat, lon=lon, method="nearest"))
    for ax, (key, ylabel, off, sc, required, colname) in zip(axes.flat, panels):
        if not all(var in ds for var in required):
            model = str(ds.attrs.get("model", "This model")).strip().title()
            variant = str(ds.attrs.get("variant", "")).strip()
            detail = f"{model} {variant}".strip()
            if key == "total_precipitation_6hr" and model.lower() == "aurora":
                message = f"Not predicted by {detail}\nNo precipitation variable in this output"
            else:
                message = "Not available in this prediction file"
            ax.text(0.5, 0.52, message, transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="0.35",
                    linespacing=1.5)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_facecolor("#f7f7f7")
            ax.tick_params(left=False, labelleft=False)
            for sp in ax.spines.values():
                sp.set_color("0.82")
            csv_cols[colname] = None
            continue  
        if key == "wind_10m":
            y = np.sqrt(p["10m_u_component_of_wind"].values.astype(np.float32) ** 2
                        + p["10m_v_component_of_wind"].values.astype(np.float32) ** 2)
        else:
            y = p[key].values.astype(np.float32)
        y = y * sc + off    
        if key == "total_precipitation_6hr":
            # Neural forecasts can contain tiny negative numerical artefacts;
            # precipitation is non-negative and is clearer as interval bars.
            y = np.maximum(y, 0.0)
            width = 0.8 * float(np.median(np.diff(x))) if len(x) > 1 else 0.2
            ax.bar(x, y, width=width, color=color, alpha=0.78, edgecolor="none")
            ax.set_ylim(bottom=0.0)
        else:
            ax.plot(x, y, marker=marker, ms=4, lw=1.8, color=color)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        csv_cols[colname] = y

    for ax in axes[-1, :]:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.set_xlabel("Valid time (UTC)", fontsize=10)
        ax.tick_params(labelsize=9)

    # A deterministic layout is important when one panel contains only an
    # availability note. ``tight_layout`` can treat that annotation as an
    # oversized artist and push the suptitle outside the saved canvas.
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.84,
                        wspace=0.14, hspace=0.08)
    fig.suptitle(f"{name}  ({fmt_latlon(lat, lon)})", fontsize=14, weight="bold", y=0.985)
    fig.text(0.5, 0.915, f"{model_label}  ·  init {fmt_init(init)} UTC",
             ha="center", va="top", fontsize=10.5, color="0.25")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    _write_series_csv(out_path.with_suffix(".csv"), valid, csv_cols)
    print(f"  series: {out_path}", flush=True)


# -----------------------------------------------------------------------------
# 2. Overview — small multiples of all 40 lead times, one figure per field
# -----------------------------------------------------------------------------
def overview(ds: xr.Dataset, init, lead, field_name: str, out_path: Path, model_label: str):
    f = BY_NAME[field_name]
    lon = ds["lon"].values
    lat = ds["lat"].values
    data = load_field(ds, f)
    vmin, vmax = color_limits(f, data)

    n = data.shape[0]
    ncols = 8
    nrows = int(np.ceil(n / ncols))
    proj = ccrs.PlateCarree() if HAS_CARTOPY else None

    # Match each panel's aspect to the data region so the maps are undistorted
    # and fill their cells edge-to-edge (this is what removes the large gaps
    # between subfigures). Full-resolution data (no stride) for crisp panels.
    ar = region_aspect(lon, lat)
    cell_h = 1.28
    figsize = (ncols * cell_h * ar, nrows * cell_h)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        subplot_kw={"projection": proj} if HAS_CARTOPY else {},
        squeeze=False,
        gridspec_kw={"wspace": 0.04, "hspace": 0.05},
    )
    for i, ax in enumerate(axes.flat):
        if i < n:
            mesh = pcolormesh(ax, lon, lat, data[i], vmin, vmax, f.cmap, proj)
            # Lead-time tag inside the panel (top-left), so panels pack tightly
            # with no per-panel title strip forcing vertical gaps.
            ax.annotate(f"T+{lead[i]:03d}h", xy=(0.02, 0.98),
                        xycoords="axes fraction", va="top", ha="left",
                        fontsize=7.5, color="0.15",
                        bbox=dict(boxstyle="square,pad=0.12", fc="white",
                                  ec="0.7", lw=0.4, alpha=0.85))
        ax.set_xticks([])
        ax.set_yticks([])
        if HAS_CARTOPY:
            set_data_extent(ax, lon, lat, proj)
            ax.coastlines(resolution="110m", linewidth=0.25, color="0.35")
        else:
            ax.set_aspect("equal")

    # Title block above the grid; horizontal colorbar in its own band below.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05)
    cbar_ax = fig.add_axes([0.30, 0.02, 0.40, 0.016])
    cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(f.unit, fontsize=9, labelpad=2)
    fig.suptitle(f"{f.label} [{f.unit}]", fontsize=13, weight="bold", y=0.985)
    fig.text(0.5, 0.925, f"{model_label}  ·  init {fmt_init(init)} UTC  ·  {n} steps",
             ha="center", va="top", fontsize=10, color="0.25")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  overview: {out_path}", flush=True)


# -----------------------------------------------------------------------------
# 3. Animated GIF — standard operational animation
# -----------------------------------------------------------------------------
def _fig_to_pil(fig) -> "Image.Image":
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    buf.seek(0)
    return Image.open(buf).copy()


def animate_field(ds: xr.Dataset, init, valid, field_name: str, out_gif: Path, fps: int,
                  model_label: str):
    f = BY_NAME[field_name]
    lon = ds["lon"].values
    lat = ds["lat"].values
    data = load_field(ds, f)
    vmin, vmax = color_limits(f, data)
    frames = []
    for i in range(data.shape[0]):
        fig, ax, proj = new_global_ax(lon=lon, lat=lat)
        mesh = pcolormesh(ax, lon, lat, data[i], vmin, vmax, f.cmap, proj)
        ax.set_title(f"{f.label}  [{f.unit}]   Valid: {fmt_valid(valid[i])}\n{model_label}",
                     fontsize=11, pad=8)
        ax.annotate(f"INIT {fmt_init(init)} UTC", xy=(0.005, 0.015),
                    xycoords="axes fraction", fontsize=8, color="0.25")
        cbar_ax = fig.add_axes([0.35, 0.065, 0.30, 0.025])
        cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8, pad=1)
        frames.append(_fig_to_pil(fig))
        plt.close(fig)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=max(40, int(1000 / fps)), loop=0, optimize=False)
    print(f"  gif: {out_gif}  ({len(frames)} frames, {fps} fps)", flush=True)


def animate_mslp_wind(ds: xr.Dataset, init, valid, out_gif: Path, fps: int, model_label: str):
    lon = ds["lon"].values
    lat = ds["lat"].values
    mslp = _drop_batch(ds["mean_sea_level_pressure"]).values.astype(np.float32) * 0.01
    wspd = np.sqrt(_drop_batch(ds["10m_u_component_of_wind"]).values.astype(np.float32) ** 2
                   + _drop_batch(ds["10m_v_component_of_wind"]).values.astype(np.float32) ** 2)
    vmax = float(np.nanpercentile(wspd, 98))
    frames = []
    for i in range(mslp.shape[0]):
        fig, ax, proj = new_global_ax(lon=lon, lat=lat)
        mesh = pcolormesh(ax, lon, lat, wspd[i], 0.0, vmax, "viridis", proj)
        levels = np.arange(950, 1051, 4.0)
        kw = dict(levels=levels, colors="k", linewidths=0.55)
        if HAS_CARTOPY:
            cs = ax.contour(lon, lat, mslp[i], transform=proj, **kw)
        else:
            cs = ax.contour(lon, lat, mslp[i], **kw)
        ax.clabel(cs, fmt="%d", fontsize=5, inline=True, inline_spacing=2)
        ax.set_title(f"MSLP [hPa, contours] + 10-m wind [m/s, shading]   "
                     f"Valid: {fmt_valid(valid[i])}\n{model_label}",
                     fontsize=11, pad=8)
        ax.annotate(f"INIT {fmt_init(init)} UTC", xy=(0.005, 0.015),
                    xycoords="axes fraction", fontsize=8, color="0.25")
        cbar_ax = fig.add_axes([0.35, 0.065, 0.30, 0.025])
        cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8, pad=1)
        frames.append(_fig_to_pil(fig))
        plt.close(fig)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=max(40, int(1000 / fps)), loop=0, optimize=False)
    print(f"  gif: {out_gif}  ({len(frames)} frames, {fps} fps)", flush=True)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Publication-grade forecast visualizer.")
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--model", default=None,
                    help="model family (default: prediction-file metadata, then graphcast)")
    ap.add_argument("--variant", default=None,
                    help="model variant (default: prediction-file metadata, then operational)")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="results root (default: <repo>/results or $RESULTS_ROOT)")
    ap.add_argument("--fps", type=int, default=6, help="GIF frames per second")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-overview", action="store_true")
    ap.add_argument("--no-series", action="store_true")
    args = ap.parse_args()

    if not args.predictions.exists():
        raise SystemExit(f"predictions file not found: {args.predictions}")

    if Image is None:
        print("Pillow not installed; GIFs disabled", flush=True)

    import os
    out_root = args.out_root or Path(
        os.environ.get("RESULTS_ROOT", str(REPO_ROOT / "results"))
    )

    ds = xr.open_dataset(args.predictions, engine="netcdf4")
    # Prefer the self-describing init_time coord (unified schema); fall back to
    # the init time encoded in the filename (legacy GraphCast schema).
    if "init_time" in ds.coords:
        init = np.datetime64(ds["init_time"].values.astype("datetime64[s]")[()])
    else:
        init = parse_init_time(args.predictions)
    init_dir = np.datetime_as_string(init.astype("datetime64[s]"), unit="m")[:13] + "Z"

    model = args.model or str(ds.attrs.get("model", "graphcast"))
    variant = args.variant or str(ds.attrs.get("variant", "operational"))
    model_label = model_display_name(model, variant)
    run_dir = out_root / model / variant / init_dir / "visualizations"
    print(f"Output root: {run_dir}", flush=True)

    valid, lead = valid_times(ds, init)
    print(f"init={fmt_init(init)}Z  steps={len(lead)}  horizon={lead[-1]}h", flush=True)

    if not args.no_series:
        for city in CITIES:
            name, lat, lon, _, _ = city
            fname = f"timeseries_{name}_{fmt_latlon_file(lat, lon)}.png"
            time_series_city(ds, init, valid, city, run_dir / "series" / fname, model_label)

    if not args.no_overview:
        for name in OVERVIEW_FIELDS:
            if not field_available(ds, BY_NAME[name]):
                print(f"  overview: skip {name} (not in prediction)", flush=True)
                continue
            overview(ds, init, lead, name, run_dir / "overview" / f"{name}_40steps.png",
                     model_label)

    if not args.no_gif and Image is not None:
        for name in GIF_FIELDS:
            if not field_available(ds, BY_NAME[name]):
                print(f"  gif: skip {name} (not in prediction)", flush=True)
                continue
            animate_field(ds, init, valid, name, run_dir / "gif" / f"anim_{name}.gif",
                          args.fps, model_label)
        if all(n in ds for n in ("mean_sea_level_pressure", "10m_u_component_of_wind",
                                 "10m_v_component_of_wind")):
            animate_mslp_wind(ds, init, valid, run_dir / "gif" / "anim_mslp_wind10m.gif",
                              args.fps, model_label)
        else:
            print("  gif: skip anim_mslp_wind10m (not in prediction)", flush=True)

    ds.close()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
