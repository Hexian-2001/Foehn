#!/usr/bin/env python3
"""Local (CPU, no torch/netcdf4) sanity check for the Aurora adapter."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from aurora_forecast import config, adapter  # noqa: E402

import datetime as dt  # noqa: E402

sfc, pl = adapter.read_cycle(dt.date(2026, 8, 27), 0, config.RAW_IFS_ROOT)

print("=== SURFACE ===")
print("  vars:", sorted(sfc.data_vars))
print("  dims:", dict(sfc.sizes))
lat = sfc.lat.values
lon = sfc.lon.values
print(f"  lat: n={lat.size} [{lat[0]:.2f} ... {lat[-1]:.2f}]",
      "decreasing" if lat[1] < lat[0] else "INCREASING(!)")
print(f"  lon: n={lon.size} [{lon[0]:.2f} ... {lon[-1]:.2f}]",
      "increasing [0,360)" if (lon[0] == 0.0 and lon[-1] < 360 and lon[1] > lon[0]) else "BAD(!)")

print("=== PRESSURE ===")
print("  vars:", sorted(pl.data_vars))
print("  dims:", dict(pl.sizes))
lvl = pl.level.values
print(f"  level: {lvl.tolist()}",
      "ascending" if lvl[0] < lvl[-1] else "DESCENDING(!)")

print("=== STATIC ===")
static = adapter.load_static(config.WEIGHTS_DIR / config.STATIC_PICKLE_NAME)
for k, v in static.items():
    print(f"  {k}: shape={v.shape} dtype={v.dtype} "
          f"range=[{v.min():.3g}, {v.max():.3g}]")

# Orientation cross-check: static row 0 must be 90 N, matching GRIB row 0.
assert lat[0] == 90.0 and lat[-1] == -90.0, "lat orientation wrong"
assert (lvl == list(config.PRESSURE_LEVELS)).all(), "level set mismatch"
assert sorted(sfc.data_vars) == ["10u", "10v", "2t", "msl"], "surf vars wrong"
assert sorted(pl.data_vars) == ["q", "t", "u", "v", "z"], "atmos vars wrong (w dropped?)"
print("\nALL ADAPTER CHECKS PASSED")
