#!/usr/bin/env python3
# =============================================================================
# End-to-end realtime forecast pipeline.
# =============================================================================
# One command drives the whole production loop:
#
#     download latest ECMWF open-data analysis
#       -> process GRIB into the GraphCast input .nc
#       -> submit GPU inference to Slurm (and wait)
#       -> render the visualization figure set
#
# Stages 1-2 (download + process) are light CPU work and run on the login node.
# Stage 3 (inference) is the only GPU step and runs inside a Slurm job via
# scripts/run_inference.sbatch. Stage 4 (visualization) is CPU and runs on the
# login node after inference finishes.
#
# Run through scripts/realtime.sh (which sets the conda env PATH), e.g.
#     ./scripts/realtime.sh                    # latest cycle, full pipeline
#     ./scripts/realtime.sh --no-submit        # download + process only
#     ./scripts/realtime.sh --date 2026-08-31 --time 00
#     ./scripts/realtime.sh --partition gpu --walltime 08:00:00
# =============================================================================

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths: this file lives at <project>/scripts/realtime.py. Its sibling packages
# (opendata_download, data_processing) live one level up, under a shared repo
# root (locally `forecast_models/`, on Pawsey `Foehn/`).
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]   # weathernext_forecast
REPO_ROOT = PROJECT_DIR.parent                       # shared repo root

for _pkg in ("opendata_download", "data_processing"):
    _src = REPO_ROOT / _pkg / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from opendata_download import config as dl_config  # noqa: E402
from opendata_download import downloader          # noqa: E402
from opendata_download.client import OpenDataClient  # noqa: E402
from data_processing import ingest                # noqa: E402

SBATCH_SCRIPT = PROJECT_DIR / "scripts" / "run_inference.sbatch"
VIZ_SCRIPT = PROJECT_DIR / "scripts" / "visualize.py"


def resolve_latest(client: OpenDataClient) -> tuple[dt.date, int]:
    """Newest fully-available analysis cycle (date, hour)."""
    latest = client.latest(
        {"type": "fc", "step": dl_config.ANALYSIS_STEP, "levtype": "sfc",
         "param": dl_config.SURFACE_PARAMS[0]}
    )
    if isinstance(latest, dt.datetime):
        return latest.date(), latest.hour
    raise RuntimeError(f"unexpected latest() result: {latest!r}")


def stage_download(client, date, hour, data_root, force, include_static) -> None:
    print(f"[1/4] downloading cycle {date} {hour:02d}Z", flush=True)
    paths = downloader.download_init(
        client, date=date, hour=hour, data_root=data_root,
        force=force, include_static=include_static,
    )
    for label, path in paths.items():
        print(f"      {label:<10} -> {path}", flush=True)


def stage_process(date, hour, raw_root, out_dir, force) -> str:
    print("[2/4] processing GRIB -> GraphCast input .nc", flush=True)
    path = ingest.build_input(date, hour, raw_root=raw_root, out_dir=out_dir, force=force)
    print(f"      input file: {path.name}", flush=True)
    return path.name


def stage_submit(input_name: str, date, hour, args) -> None:
    print("[3/4] submitting GPU inference to Slurm", flush=True)
    log_dir = PROJECT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "sbatch",
        "--account", args.account,
        "--partition", args.partition,
        "--nodes", str(args.nodes),
        # Setonix allocation-pack model: each --gres=gpu:N pack auto-allocates
        # 8 CPU cores + ~29.44 GiB RAM. `--gpus-per-node` is deprecated/buggy.
        "--gres", f"gpu:{args.gpus_per_node}",
        "--time", args.walltime,
        "--job-name", f"gc_{date:%Y%m%d}_{hour:02d}",
        "--output", str(log_dir / f"infer_{date:%Y%m%d}_{hour:02d}.%j.out"),
        "--chdir", str(PROJECT_DIR),
        "--export", f"ALL,WEATHERNEXT_INPUT_FILENAME={input_name},CONDA_ENV={args.conda_env}",
    ]
    if not args.no_wait:
        cmd.append("--wait")
    cmd.append(str(SBATCH_SCRIPT))

    print("      " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def stage_visualize(args) -> None:
    print("[4/4] rendering visualizations", flush=True)
    # Predictions now land in the EXTERNAL results tree (unified + region-cropped):
    #   <repo>/results/<model>/<variant>/<init>Z/predictions/*.nc
    # — not in <project>/predictions. Find the newest one there.
    preds = sorted(
        (REPO_ROOT / "results").glob("**/predictions/*.nc"),
        key=lambda p: p.stat().st_mtime,
    )
    if not preds:
        raise SystemExit("no results/**/predictions/*.nc found to visualize")
    pred = preds[-1]
    cmd = [sys.executable, str(VIZ_SCRIPT), "--predictions", str(pred)]
    print("      " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end realtime GraphCast forecast.")
    g = ap.add_argument_group("cycle selection")
    g.add_argument("--date", help="init date YYYY-MM-DD (UTC)")
    g.add_argument("--time", help="init hour 00/06/12/18 (UTC)")
    g.add_argument("--latest", action="store_true",
                   help="use the newest available cycle (default when no --date/--time)")
    g.add_argument("--source", default=dl_config.SOURCE, help="ecmwf | aws | azure | google")

    g = ap.add_argument_group("download / process")
    g.add_argument("--data-root", type=Path, default=dl_config.DATA_ROOT,
                   help="raw-data root (default: OPENDATA_DATA_ROOT or <repo>/data)")
    g.add_argument("--force", action="store_true", help="re-download / re-process")
    g.add_argument("--no-static", action="store_true", help="skip static fields")

    g = ap.add_argument_group("inference (Slurm)")
    g.add_argument("--account", default="pawsey0115-gpu")
    g.add_argument("--partition", default="gpu")
    g.add_argument("--nodes", type=int, default=1)
    g.add_argument("--gpus-per-node", type=int, default=3,
                   help="GPUs requested (maps to --gres=gpu:N; 3 = ~88 GiB host RAM for 0.25-deg operational)")
    g.add_argument("--walltime", default="04:00:00", help="Slurm walltime HH:MM:SS")
    g.add_argument("--conda-env",
                   default="/scratch/pawsey0115/hwang4/miniconda3/envs/infer-gpu")
    g.add_argument("--no-submit", action="store_true", help="download + process only")
    g.add_argument("--no-wait", action="store_true", help="submit and return (no --wait)")

    g = ap.add_argument_group("visualization")
    g.add_argument("--no-visualize", action="store_true",
                   help="skip rendering (visualize.py draws the 3 fixed cities: Beijing/Shanghai/Guangzhou)")

    args = ap.parse_args()

    client = OpenDataClient(source=args.source)

    # Resolve the cycle.
    if args.date and args.time:
        date = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
        hour = int(args.time)
    elif args.latest or (not args.date and not args.time):
        date, hour = resolve_latest(client)
        print(f"latest available cycle: {date} {hour:02d}Z", flush=True)
    else:
        ap.error("provide both --date and --time, or --latest")

    raw_root = args.data_root / "raw" / "ifs"
    out_dir = PROJECT_DIR / "data" / "processed"

    stage_download(client, date, hour, args.data_root, args.force, not args.no_static)
    input_name = stage_process(date, hour, raw_root, out_dir, args.force)

    if args.no_submit:
        print(f"done (--no-submit). input ready: {input_name}", flush=True)
        return

    stage_submit(input_name, date, hour, args)

    if not args.no_visualize:
        stage_visualize(args)

    print("realtime pipeline complete.", flush=True)


if __name__ == "__main__":
    main()
