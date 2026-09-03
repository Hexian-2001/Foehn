#!/usr/bin/env python3
# =============================================================================
# Prediction store CLI: unify + crop + relocate an already-saved model .nc.
# =============================================================================
# Thin wrapper over `weathernext_forecast.prediction_store`. It reads a raw
# model prediction (GraphCast/WeatherNext .nc), normalizes it into the
# model-agnostic schema, crops it to a lat/lon box (default: China), and writes
# it into the EXTERNAL, model-organized results tree:
#
#     <results>/<model>/<variant>/<init>Z/predictions/
#         <model>_<variant>_IC<init>_STEPS<n>_<horizon>h_<res>deg_<region>.nc
#
# The full-global ~13 GB file is normally deleted after a successful crop
# (--delete-global); the region file is the durable artifact.
#
# Usage:
#   python scripts/crop_region.py --predictions predictions/predictions_....nc \
#       --model graphcast --variant operational --region china --delete-global
#   python scripts/crop_region.py --predictions ... --lat 15 55 --lon 70 140
# =============================================================================

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the package importable: <repo>/weathernext_forecast/src.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import xarray as xr

from weathernext_forecast import prediction_store as store

# Script lives at <repo>/weathernext_forecast/scripts/crop_region.py -> shared repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description="Unify + crop a model prediction into the results tree.")
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--model", default="graphcast", help="model family (results subdir)")
    ap.add_argument("--variant", default="operational", help="model variant")
    ap.add_argument("--region", default="china", choices=list(store.REGIONS),
                    help="named region to crop to")
    ap.add_argument("--lat", nargs=2, type=float, metavar=("MIN", "MAX"), default=None,
                    help="override region latitude bounds (deg N)")
    ap.add_argument("--lon", nargs=2, type=float, metavar=("MIN", "MAX"), default=None,
                    help="override region longitude bounds (deg E)")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="results root (default: <repo>/results or $RESULTS_ROOT)")
    ap.add_argument("--delete-global", action="store_true",
                    help="delete the source global file after a successful crop")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the plan and output path without writing")
    args = ap.parse_args()

    if not args.predictions.exists():
        raise SystemExit(f"predictions file not found: {args.predictions}")

    out_root = args.out_root or Path(os.environ.get("RESULTS_ROOT", str(REPO_ROOT / "results")))
    init = store.parse_init_time(args.predictions)

    # Resolve the crop box (explicit --lat/--lon override the named region).
    region_cfg = store.REGIONS[args.region]
    lat_box = tuple(args.lat) if args.lat is not None else region_cfg["lat"]
    lon_box = tuple(args.lon) if args.lon is not None else region_cfg["lon"]

    print(f"source:      {args.predictions}", flush=True)
    print(f"region:      {args.region}  lat={lat_box} lon={lon_box}", flush=True)

    ds = xr.open_dataset(args.predictions, engine="netcdf4")

    # Apply a custom box by patching the region config in-place for this call.
    region = args.region
    if args.lat is not None or args.lon is not None:
        region = "__custom__"
        store.REGIONS["__custom__"] = dict(lat=lat_box, lon=lon_box)

    out_path = store.save_unified(
        ds, model=args.model, variant=args.variant, init=init, region=region,
        out_root=out_root, source=args.predictions.name, dry_run=args.dry_run,
    )
    ds.close()

    if args.dry_run:
        print(f"[dry-run] would write: {out_path}", flush=True)
        return

    print(f"wrote:       {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)", flush=True)

    if args.delete_global:
        args.predictions.unlink()
        print(f"deleted:     {args.predictions}", flush=True)


if __name__ == "__main__":
    main()
