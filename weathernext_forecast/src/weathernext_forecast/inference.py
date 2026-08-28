# ============================================================================= 
# GraphCast inference pipeline (WeatherNext 1 Graph).
# ============================================================================= 
# Runs a pretrained GraphCast model end-to-end on local files. The pipeline is:
#
#     weights (.npz)  +  stats (.nc)  +  input weather data (.nc)
#          |                 |                   |
#          | checkpoint.load |                   | xarray.load_dataset
#          v                 v                   v
#   params/model_config   mean/stddev        example_batch
#   /task_config          /diffs_stddev          |
#          |                                     |
#          |        +----------------------------+
#          |        |  data_utils.extract_inputs_targets_forcings
#          v        v
#      construct_wrapped_graphcast()       inputs / targets / forcings
#          |
#          +--> hk.transform_with_state --> jax.jit --> rollout.chunked_prediction
#                                                          |
#                                                          v
#                                                predictions (saved to .nc)
#
# This module holds the pipeline logic only; all paths and knobs live in
# `config.py`. The thin CLI entry point is `scripts/run_inference.py`.
# =============================================================================

from __future__ import annotations

import dataclasses
import functools
import sys

import haiku as hk
import jax
import numpy as np
import xarray

from weathernext_forecast import config

# Make the upstream `weathernext` package importable. The fork lives outside the
# installed packages at config.UPSTREAM_DIR; adding it to sys.path lets us
# `import weathernext` without a `pip install`. (Alternatively `pip install -e`
# the fork once and delete this block.)
if str(config.UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(config.UPSTREAM_DIR))

from weathernext.weathernext1_graph import graphcast  # noqa: E402
from weathernext.utils import (                        # noqa: E402
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    normalization,
    rollout,
)


def build_wrapped_predictor(model_config, task_config, stats):
    """Assemble the GraphCast predictor by wrapping the raw model four times.

    Each wrapper adds one responsibility, applied inside-out:

        GraphCast                       : raw GNN (encoder-processor-decoder)
        casting.Bfloat16Cast            : run activations in bfloat16 to save
                                          memory (params stay float32)
        normalization.InputsAndResiduals: normalize inputs to ~zero-mean unit
                                          variance; predict *normalized
                                          residuals* (target - last input
                                          frame); un-normalize and add the
                                          residual back on output
        autoregressive.Predictor        : unroll the one-step model into a
                                          multi-step forecast (hk.scan)

    Args:
        model_config: graphcast.ModelConfig from the checkpoint.
        task_config:  graphcast.TaskConfig from the checkpoint.
        stats:        dict with keys 'diffs_stddev_by_level', 'mean_by_level',
                      'stddev_by_level' (xarray.Dataset each).

    Returns:
        A single Predictor exposing the full wrapped pipeline.
    """
    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
        mean_by_level=stats["mean_by_level"],
        stddev_by_level=stats["stddev_by_level"],
    )
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor


def validate_input_matches_model(model_config, task_config, inputs):
    """Fail fast with a clear message if the data does not match the model."""
    resolution = 360.0 / inputs.sizes["lon"]
    if model_config.resolution not in (0.0, resolution):
        raise ValueError(
            f"Resolution mismatch: model expects {model_config.resolution} deg, "
            f"but input data is {resolution} deg ({inputs.sizes['lon']} lon "
            f"points). Choose a dataset with matching resolution."
        )
    n_levels = inputs.sizes.get("level", len(task_config.pressure_levels))
    if n_levels != len(task_config.pressure_levels):
        raise ValueError(
            f"Level mismatch: model expects {len(task_config.pressure_levels)} "
            f"levels, but input data has {n_levels}. Choose matching data."
        )


def main():
    # ---- 1. Load the checkpoint (weights + architecture + task spec). ----
    # A GraphCast checkpoint is a single custom `np.savez` file bundling the
    # trained weights, the architecture config, and the task config together.
    ckpt_path = config.WEIGHTS_DIR / config.MODEL_FILENAME
    with open(ckpt_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    
    params = ckpt.params
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    print(f"[1/6] Loaded checkpoint: {config.MODEL_FILENAME}")
    print(f"      {ckpt.description}")

    # ---- 2. Load the normalization statistics. ----
    # The model operates on zero-mean/unit-variance data and predicts residuals
    # (target minus last input frame), so THREE statistic files are needed.
    stats = {
        "mean_by_level": xarray.load_dataset(
            config.STATS_DIR / "mean_by_level.nc").compute(),
        "stddev_by_level": xarray.load_dataset(
            config.STATS_DIR / "stddev_by_level.nc").compute(),
        "diffs_stddev_by_level": xarray.load_dataset(
            config.STATS_DIR / "diffs_stddev_by_level.nc").compute(),
    }
    print("[2/6] Loaded normalization statistics (3 files)")

    # ---- 3. Load the input weather data. ----
    input_path = config.DATA_DIR / config.INPUT_FILENAME
    example_batch = xarray.load_dataset(input_path).compute()
    print(f"[3/6] Loaded input data: {config.INPUT_FILENAME}")
    print(f"      {example_batch.dims.mapping}")

    # ---- 4. Extract inputs / targets / forcings. ----
    # This is the "tokenizer" equivalent: it selects the right levels, computes
    # any missing derived variables (TISR solar radiation, year/day progress
    # sin/cos), and slices out:
    #   inputs   : the last 12h (2 x 6h steps) as the conditioning window
    #   targets  : the requested lead times (the shape is the template)
    #   forcings : solar radiation + time progress AT the target lead times
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=config.TARGET_LEAD_TIMES,
        **dataclasses.asdict(task_config),
    )
    print(f"[4/6] Extracted: inputs {dict(inputs.sizes)}, "
          f"targets {dict(targets.sizes)}, forcings {dict(forcings.sizes)}")

    validate_input_matches_model(model_config, task_config, inputs)

    # Forecast reference time = datetime at lead time 0 (the initialization
    # time). Extraction shifted the time axis so that the final file timestep
    # became the last target lead time (`targets.time[-1]`); subtracting that
    # horizon from the last datetime recovers the reference time. `.isel` +
    # `.ravel()[0]` yield a *scalar* datetime64 (datetime is a (batch, time)
    # coord) — passing an array here would str() into `['...' '...']` and
    # produce an illegal Windows filename.
    ref_dt = (example_batch["datetime"].isel(time=-1).values.ravel()[0]
              - targets.time.values[-1])
    ref_time = np.datetime_as_string(ref_dt, unit="h")

    # ---- 5. Assemble the predictor and jit-compile the forward pass. ----

    # Wrap the stateful predictor into a pure functional form with explicit
    # params/state/rng — the shape JAX + jit require.
    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = build_wrapped_predictor(model_config, task_config, stats)
        return predictor(
            inputs, targets_template=targets_template, forcings=forcings)

    # Configs and params are bound via functools.partial (rather than captured
    # by closure) so JAX correctly re-compiles if they change. State is always
    # empty for GraphCast, so `drop_state` returns only the predictions.
    def with_configs(fn):
        return functools.partial(
            fn, model_config=model_config, task_config=task_config)

    def with_params(fn):
        return functools.partial(fn, params=params, state={})

    def drop_state(fn):
        # run_forward.apply returns (predictions, state); keep predictions only.
        return lambda **kw: fn(**kw)[0]

    run_forward_jitted = drop_state(
        with_params(jax.jit(with_configs(run_forward.apply))))
    print("[5/6] Predictor assembled and jit-compiled")

    # ---- 6. Run the autoregressive rollout. ----
    # `rollout.chunked_prediction` iterates the jitted one-step model in a
    # PYTHON loop (low memory, ideal for inference). The targets template is
    # filled with NaN because only its shape/coords matter — values are ignored.
    # First call triggers JAX compilation and may take a few minutes.
    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets * np.nan,   # template only; values unused
        forcings=forcings,
    )

    # ---- 7. Save the result. ----
    # `predictions` has the same shape as `targets` and contains the target
    # variables already un-normalized into real physical units (float32).
    out_name = (f"predictions_{ref_time}_"
                f"{config.MODEL_FILENAME.removesuffix('.npz')}.nc")
    out_path = config.PREDICTIONS_DIR / out_name
    predictions.to_netcdf(out_path)
    print(f"[6/6] Saved predictions to: {out_path}")
    print(f"      {dict(predictions.sizes)}")
    print("\nPrediction variables:", list(predictions.data_vars))


if __name__ == "__main__":
    main()
