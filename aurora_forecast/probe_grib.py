#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe the raw ECMWF open-data GRIB to lock the Aurora adapter contract.

Reads the same `sfc_fc0.grib2` / `pl_fc0.grib2` the opendata_download package
writes, and reports the cfgrib-emitted variable names, lat/lon direction and
range, level values, and grid shape. Run once on a new data source; the adapter
assumptions are encoded from its output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cfgrib
import xarray as xr

RAW = Path(r"D:\mingyang_tech_work\forecast_models\data\raw\ifs")

for fname in ("sfc_fc0.grib2", "pl_fc0.grib2"):
    path = RAW / "2026-08-27" / "00" / fname
    print("=" * 70)
    print(path)
    try:
        dss = cfgrib.open_datasets(str(path))
        print(f"  open_datasets -> {len(dss)} hypercube(s)")
        ds = xr.merge(dss, compat="override")
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: {e!r}")
        continue

    print("  data_vars:", sorted(ds.data_vars))
    print("  dims     :", dict(ds.sizes))
    for c in ("latitude", "longitude", "level", "isobaricInhPa"):
        if c in ds.coords:
            v = ds[c].values
            head = v[:3].tolist() if v.size else []
            tail = v[-3:].tolist() if v.size else []
            direction = ""
            if v.size > 1:
                d = v[1] - v[0]
                direction = " increasing" if d > 0 else " decreasing"
            print(f"  coord {c}: n={v.size} [{head} ... {tail}]{direction}")
    print("  coords   :", sorted(ds.coords))
