# =============================================================================
# Command-line interface for model-agnostic forecast inference.
# =============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from inference import config
from inference import inputs as inputs_mod
from inference import registry, saver


def _parse_init_time(text: str) -> np.datetime64:
    """Parse a CLI init time into an hour-granular datetime64.

    Accepts ``2026-08-27T00``, ``2026-08-27 00``, or ``2026-08-27`` (the latter
    defaults to hour 00). Normalizes to UTC-hour precision.
    """
    text = text.strip()
    if "T" not in text and " " not in text:
        text += "T00"
    text = text.replace(" ", "T")
    try:
        return np.datetime64(text, "h")
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"cannot parse init time {text!r}: use YYYY-MM-DDTHH"
        ) from e


def _parse_steps(text: str) -> int:
    try:
        n = int(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid step count: {text!r}") from e
    if n < 1:
        raise argparse.ArgumentTypeError("steps must be >= 1")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_inference",
        description=(
            "Run a forecast model on a processed input .nc and save predictions. "
            "Model-agnostic: the model is chosen by name from the registry."
        ),
    )
    p.add_argument(
        "--model",
        default="graphcast",
        help="model name from the registry (see --list-models; default: graphcast)",
    )
    p.add_argument(
        "--init-time",
        type=_parse_init_time,
        default=None,
        help="forecast IC time as YYYY-MM-DDTHH; default = latest available input",
    )
    p.add_argument(
        "--steps",
        type=_parse_steps,
        default=None,
        help="number of forecast steps; default = the model's default_steps",
    )
    p.add_argument(
        "--resolution",
        default=None,
        help="timestep between forecast steps, e.g. '6h' or '1h'; "
             "default = the model's native resolution",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="directory of processed input .nc files (default: config.DATA_DIR)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="directory for predictions (default: config.PREDICTIONS_DIR)",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="list registered models and exit",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_models:
        print("Registered models:")
        for name, spec in sorted(registry.MODELS.items()):
            print(f"  {name:<18} {spec.description}")
        return 0

    spec = registry.get_spec(args.model)

    # Resolution defaults to the model's native timestep; steps to its default.
    resolution = args.resolution or spec.resolution
    steps = args.steps if args.steps is not None else spec.default_steps

    data_dir = Path(args.data_dir) if args.data_dir else config.DATA_DIR
    out_dir = Path(args.out_dir) if args.out_dir else config.PREDICTIONS_DIR

    # Pick the input file, defaulting to the latest available.
    input_path = inputs_mod.resolve_input(
        spec.name, spec.input_pattern, data_dir, args.init_time
    )

    # Build the target lead-time slice from resolution + steps. This is where a
    # future 1h model is supported: same code, just resolution="1h".
    lead = pd.Timedelta(resolution)
    target_lead_times = slice(lead, lead * steps)

    print(f"model      : {spec.name}  ({spec.resolution} native)")
    print(f"input      : {input_path}")
    print(f"resolution : {resolution}")
    print(f"steps      : {steps}  ->  lead {lead} .. {lead * steps}")

    runner = registry.load_runner(spec)
    predictions, init_time = runner.run(
        spec, input_path=input_path, target_lead_times=target_lead_times
    )

    out_path = saver.save_predictions(
        predictions,
        model_name=spec.name,
        init_time=init_time,
        resolution=resolution,
        steps=steps,
        out_dir=out_dir,
    )
    print(f"saved      : {out_path}")
    print("Prediction variables:", list(predictions.data_vars))
    return 0


if __name__ == "__main__":
    sys.exit(main())
