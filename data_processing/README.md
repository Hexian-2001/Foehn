# data_processing — stage 2 (GRIB → GraphCast input)

Convert ECMWF open-data IFS analysis fields (`fc0`, 0.25°) into the standard
netCDF input consumed by the GraphCast inference pipeline.

This is the middle stage of a strictly decoupled 3-stage pipeline:

```
stage 1 (opendata_download)   →  raw/ifs/<date>/<HH>/{sfc,pl}_fc0.grib2 + static/
stage 2 (data_processing)     →  <processed>/source-ifs_date-<date>_res-0.25_levels-13_steps-40.nc
stage 3 (weathernext_forecast)→  predictions/*.nc
```

Each stage reads files and writes files — no shared in-memory state — so they
run independently and on their own schedule.

## What it does (and deliberately does not)

- **Renames** GRIB fields to GraphCast variable names (`t2m`→`2m_temperature`,
  `z`→`geopotential`, `q`→`specific_humidity`, …).
- **Realigns the grid**: open-data lon is `[-180, 180)`; GraphCast/ERA5 use
  `[0, 360)`. Normalised and rolled so `lon` is ascending `0…359.75`.
- **Sorts levels** ascending (`50…1000 hPa`, kept as int so downstream
  `sel(level=...)` matches exactly).
- **Lays out the forecast time axis**: 42 steps = 2 real input steps (T−6h, T)
  + 40 NaN-filled target steps (T+6h … T+240h). The model reads only the first
  two; the rest is a shape/coordinate template plus the `datetime` axis used to
  derive solar/time forcings.
- **No numerical unit conversion** — open-data IFS fields already use the same
  SI units as the ERA5 data GraphCast was trained on (verified by probing the
  raw GRIB).

Static fields (`geopotential_at_surface`, `land_sea_mask`) are read once from
`static/static.grib2` and included with no time dimension.

`total_precipitation_6hr` is added as an all-NaN variable. The operational model
predicts precipitation but does not take it as input (HRES-fc0 has none); stage 3
still selects it as part of the target template, so it must be present — NaN is
fine because the template is overwritten with NaN at inference time anyway.

## Usage

```powershell
# one-off
python scripts/prepare_fc0.py --date 2026-08-27 --time 00

# the T−6h cycle is read automatically; both must exist under raw/ifs
python scripts/prepare_fc0.py --date 2026-08-27 --time 00 --force
```

Options: `--date` / `--time` (00/06/12/18 UTC), `--raw-root`,
`--out-dir`, `--force`, `--log-level`.

Output is written to `weathernext_forecast/data/processed/` by default (the
stage-3 `DATA_DIR`), or wherever `PROCESSED_ROOT` / `--out-dir` points.

## Notes

- Output uses the **netCDF-4 (HDF5) engine with zlib** — a single 0.25° pressure
  variable exceeds the netCDF-3 2 GB/variable limit, and the NaN-heavy future
  steps compress a ~14 GB logical file down to ~250 MB.
- Requires `netcdf4`, `cfgrib`, `eccodes`, `xarray`, `numpy`, `pandas`.
