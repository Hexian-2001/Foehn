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

# tqdm is optional: it gives a per-lead-time progress bar over the 40-step
# autoregressive rollout. If it isn't installed the run still works, just
# without the bar (the [N/6] step markers still show coarse progress).
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - depends on the env, not the code
    tqdm = None

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


def _progress(iterable, *, total, desc):
    """Wrap `iterable` in a tqdm bar if available, else pass through unchanged."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="step")


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
    # ---- 0. Report the JAX device topology (flushed so the slurm log shows it
    # immediately). On Setonix MI250X this confirms three things: (a) `import
    # jax` did not deadlock on a cold GPU, (b) how many GCDs JAX sees, (c) which
    # device this run will target. If the log stops here, the hang is in import /
    # GPU init, i.e. the rocminfo warm-up is missing.
    print(f"[0/6] JAX backend={jax.default_backend()} "
          f"devices={jax.devices()}", flush=True)

    # ---- 1. Load the checkpoint (weights + architecture + task spec). ----
    # A GraphCast checkpoint is a single custom `np.savez` file bundling the
    # trained weights, the architecture config, and the task config together.
    ckpt_path = config.WEIGHTS_DIR / config.MODEL_FILENAME
    with open(ckpt_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    
    params = ckpt.params
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    print(f"[1/6] Loaded checkpoint: {config.MODEL_FILENAME}", flush=True)
    print(f"      {ckpt.description}", flush=True)

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
    print("[2/6] Loaded normalization statistics (3 files)", flush=True)

    # ---- 3. Load the input weather data. ----
    input_path = config.DATA_DIR / config.INPUT_FILENAME
    example_batch = xarray.load_dataset(input_path).compute()
    print(f"[3/6] Loaded input data: {config.INPUT_FILENAME}", flush=True)
    print(f"      {example_batch.dims.mapping}", flush=True)

    # ---- 4. Extract inputs / targets / forcings. ----
    # This is the "tokenizer" equivalent: it selects the right levels, computes
    # any missing derived variables (TISR solar radiation, year/day progress
    # sin/cos), and slices out:
    #   inputs   : the last 12h (2 x 6h steps) as the conditioning window
    #   targets  : the requested lead times (the shape is the template)
    #   forcings : solar radiation + time progress AT the target lead times
    #
    # The solar forcing (`toa_incident_solar_radiation`) is computed with
    # `jax.jit` and would normally target the GPU. For the operational model
    # (0.25°, 40 target steps) that is ~40 jitted kernels of ~1.5 GB each, and
    # on a Setonix node whose GPU is cold or contended by other users' jobs the
    # eager/jitted GPU init here can stall in an idle wait — a second hang site
    # distinct from the `import jax` deadlock. Forcings are a one-time CPU-bound
    # computation, so pin them to the CPU device; the heavy model rollout in
    # step 5 still runs on the GPU.
    with jax.default_device(jax.devices("cpu")[0]):
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            example_batch,
            target_lead_times=config.TARGET_LEAD_TIMES,
            **dataclasses.asdict(task_config),
        )
    print(f"[4/6] Extracted: inputs {dict(inputs.sizes)}, "
          f"targets {dict(targets.sizes)}, forcings {dict(forcings.sizes)}",
          flush=True)

    validate_input_matches_model(model_config, task_config, inputs)

    # Fail fast with a *clear* message if the requested lead times ran past the
    # end of the input file (extraction then silently yields an empty time dim,
    # which surfaces far downstream as an opaque weight-shape error).
    if inputs.sizes.get("time", 0) == 0:
        raise ValueError(
            "Requested forecast horizon exceeds the input file: the input has "
            f"only {example_batch.sizes.get('time', 0)} timesteps, so the "
            f"{task_config.input_duration} conditioning window left zero input "
            "steps. Reduce TARGET_LEAD_TIMES or use a file with more steps."
        )

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
    # This print fires BEFORE the first real GPU execution of the big GNN. Its
    # jit-compilation of a mesh_size=6 / latent=512 / 16-msg-step graph can take
    # 5-20 min on ROCm — so a long gap here is normal, not a hang. The hang is
    # only confirmed if it stays here past the job's time limit with the GPU
    # idle (check `rocm-smi`). See the sbatch script for the preflight that
    # isolates import from compile.
    print("[5/6] Predictor assembled and jit-compiled", flush=True)

    # ---- 6. Run the autoregressive rollout. ----
    # `chunked_prediction_generator` yields one 6h lead-time step at a time in a
    # Python loop (low memory, ideal for inference). Wrapping it in a tqdm bar
    # makes progress visible step-by-step; the targets template is filled with
    # NaN because only its shape/coords matter — values are ignored.
    # NOTE: the FIRST step triggers JAX compilation (5-20 min on ROCm), so the
    # bar sits at 0/N for a while — that is compile time, not a hang.
    num_target_steps = targets.sizes["time"]
    chunks = []
    for chunk in _progress(
        rollout.chunked_prediction_generator(
            run_forward_jitted,
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=targets * np.nan,   # template only; values unused
            num_steps_per_chunk=1,
            forcings=forcings,
        ),
        total=num_target_steps,
        desc="Rollout (6h/step)",
    ):
        chunks.append(jax.device_get(chunk))
        del chunk
    predictions = xarray.concat(chunks, dim="time")

    # ---- 7. Save the result. ----
    # `predictions` has the same shape as `targets` and contains the target
    # variables already un-normalized into real physical units (float32).
    out_name = (f"predictions_{ref_time}_"
                f"{config.MODEL_FILENAME.removesuffix('.npz')}.nc")
    out_path = config.PREDICTIONS_DIR / out_name
    predictions.to_netcdf(out_path)
    print(f"[6/6] Saved predictions to: {out_path}", flush=True)
    print(f"      {dict(predictions.sizes)}", flush=True)
    print("\nPrediction variables:", list(predictions.data_vars), flush=True)


if __name__ == "__main__":
    main()
