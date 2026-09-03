# =============================================================================
# Aurora GPU inference: input .nc -> autoregressive rollout -> unified store.
# =============================================================================
# Runs the 0.25-deg finetuned Aurora (IFS HRES T0) on a prebuilt input `.nc`,
# rolls out the forecast 6 h at a time, renames the outputs to the shared unified
# variable names, and hands the result to the model-agnostic prediction store so
# it lands in the same results tree / schema as GraphCast.
#
# Decoupling note: this module owns ONLY Aurora specifics (its variable names,
# grid, checkpoint). The unified storage contract is imported from the shared
# `prediction_store` — which is model-agnostic by design and knows nothing about
# either model's internals.
# =============================================================================

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from aurora_forecast import config

logger = logging.getLogger(__name__)

# Upstream Aurora package (read-only dependency; `import aurora`).
if str(config.UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(config.UPSTREAM_DIR))
from aurora import Aurora, Batch, rollout  # noqa: E402

# Shared model-agnostic prediction store (single source of truth for the unified
# results-tree schema). Lives in the weathernext package by historical accident;
# it has no dependency on that package's internals (only numpy/xarray).
_WEATHERNEXT_SRC = config.PROJECT_ROOT.parent / "weathernext_forecast" / "src"
if str(_WEATHERNEXT_SRC) not in sys.path:
    sys.path.insert(0, str(_WEATHERNEXT_SRC))
from weathernext_forecast import prediction_store  # noqa: E402


def load_model(ckpt_path: Path, device: str) -> Aurora:
    """Instantiate the finetuned 0.25-deg model and load its checkpoint."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    model = Aurora()  # defaults == aurora-0.25-finetuned (use_lora=True, patch 4)
    model = model.to(device)
    model.load_checkpoint_local(str(ckpt_path))
    model.eval()
    return model


def _to_unified(preds: list[Batch]) -> xr.Dataset:
    """Stack all rollout steps into one unified-name Dataset.

    The `time` coordinate is lead time in hours since init (timedelta64), which is
    exactly what ``prediction_store.save_unified`` expects. Static variables are
    excluded: they are conditioning inputs, not predictions.
    """
    last = preds[-1]
    lat = last.metadata.lat.cpu().numpy()
    lon = last.metadata.lon.cpu().numpy()
    level = np.asarray(last.metadata.atmos_levels, dtype=int)
    lead = np.arange(1, len(preds) + 1) * config.STEP_HOURS
    time = lead.astype("timedelta64[h]")

    # Stack each (1, 1, ...) prediction into a (time, ...) array.
    surf = {
        k: np.stack([p.surf_vars[k][0, 0].cpu().numpy() for p in preds])
        for k in last.surf_vars
    }
    atmos = {
        k: np.stack([p.atmos_vars[k][0, 0].cpu().numpy() for p in preds])
        for k in last.atmos_vars
    }

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for k, v in surf.items():
        data_vars[config.UNIFIED_MAP[k]] = (("time", "lat", "lon"), v)
    for k, v in atmos.items():
        data_vars[config.UNIFIED_MAP[k]] = (("time", "level", "lat", "lon"), v)

    return xr.Dataset(
        data_vars=data_vars,
        coords={"time": time, "lat": lat, "lon": lon, "level": level},
    )


def run_inference(
    input_nc: Path,
    ckpt_path: Path,
    steps: int,
    device: str = "cuda",
    region: str = "china",
    out_root: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Run the full inference pipeline and return the written prediction path."""
    if not input_nc.exists():
        raise FileNotFoundError(f"input .nc not found: {input_nc}")

    out_root = out_root or config.RESULTS_ROOT
    model = load_model(ckpt_path, device)

    batch = Batch.from_netcdf(input_nc)
    init = batch.metadata.time[0]  # reference/init time (T0)

    with torch.no_grad():
        preds = list(rollout(model, batch, steps=steps))

    ds = _to_unified(preds)
    return prediction_store.save_unified(
        ds,
        model=config.MODEL_FAMILY,
        variant=config.MODEL_VARIANT,
        init=np.datetime64(init),
        region=region,
        out_root=out_root,
        source=input_nc.name,
        dry_run=dry_run,
    )


def _resolve_input(arg_input: Path | None) -> Path:
    """Input .nc from --input, else the AURORA_INPUT_FILENAME env var."""
    if arg_input is not None:
        return arg_input
    if config.INPUT_FILENAME:
        return config.PROCESSED_DIR / config.INPUT_FILENAME
    raise SystemExit("no --input and AURORA_INPUT_FILENAME is unset")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aurora 0.25 finetuned real-time inference")
    ap.add_argument("--input", type=Path, default=None,
                    help="prebuilt input .nc (default: AURORA_INPUT_FILENAME under processed/)")
    ap.add_argument("--ckpt", type=Path, default=config.WEIGHTS_DIR / config.CHECKPOINT_NAME)
    ap.add_argument("--steps", type=int, default=config.FORECAST_STEPS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--region", default=config.PREDICT_REGION, choices=("china", "global"))
    ap.add_argument("--out-root", type=Path, default=config.RESULTS_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = run_inference(
        _resolve_input(args.input), args.ckpt, args.steps, args.device, args.region,
        args.out_root, args.dry_run,
    )
    logger.info("prediction written: %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
