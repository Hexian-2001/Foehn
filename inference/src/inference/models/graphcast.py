# =============================================================================
# GraphCast model runner (WeatherNext 1 Graph).
# =============================================================================
# The ONLY model-specific code in this package. It knows how to load a GraphCast
# checkpoint, run the autoregressive rollout, and un-normalize the result back
# to physical units — mirroring `weathernext_forecast.inference` but taking its
# inputs from a ModelSpec / CLI instead of a global config, and returning the
# predictions (plus the IC time) rather than saving anything itself.

from __future__ import annotations

import dataclasses
import functools
import sys
from pathlib import Path

import haiku as hk
import jax
import numpy as np
import xarray as xr

from inference.models.base import ModelRunner, ModelSpec


def _build_wrapped_predictor(model_config, task_config, stats):
    """Assemble the GraphCast predictor (GraphCast -> bfloat16 -> normalize
    residuals -> autoregressive unroll)."""
    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
        mean_by_level=stats["mean_by_level"],
        stddev_by_level=stats["stddev_by_level"],
    )
    return autoregressive.Predictor(predictor, gradient_checkpointing=True)


def _validate_input_matches_model(model_config, task_config, inputs):
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


class GraphCastRunner(ModelRunner):
    """Run a pretrained GraphCast checkpoint on a local .nc input file."""

    def run(
        self,
        spec: ModelSpec,
        *,
        input_path: Path,
        target_lead_times: slice,
    ) -> tuple[xr.Dataset, np.datetime64]:
        # Make the upstream `weathernext` fork importable (it lives outside the
        # installed packages at spec.upstream_dir).
        if spec.upstream_dir is not None and str(spec.upstream_dir) not in sys.path:
            sys.path.insert(0, str(spec.upstream_dir))

        global graphcast, autoregressive, casting, checkpoint, data_utils, normalization, rollout
        from weathernext.weathernext1_graph import graphcast  # noqa: E402
        from weathernext.utils import (  # noqa: E402
            autoregressive,
            casting,
            checkpoint,
            data_utils,
            normalization,
            rollout,
        )

        # ---- 1. Load the checkpoint (weights + architecture + task spec). ----
        with open(spec.checkpoint, "rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)

        model_config = ckpt.model_config
        task_config = ckpt.task_config
        print(f"[1/5] Loaded checkpoint: {spec.checkpoint.name}")

        # ---- 2. Load the normalization statistics. ----
        stats = {
            "mean_by_level": xr.load_dataset(spec.stats_dir / "mean_by_level.nc").compute(),
            "stddev_by_level": xr.load_dataset(spec.stats_dir / "stddev_by_level.nc").compute(),
            "diffs_stddev_by_level": xr.load_dataset(spec.stats_dir / "diffs_stddev_by_level.nc").compute(),
        }
        print("[2/5] Loaded normalization statistics (3 files)")

        # ---- 3. Load the input weather data. ----
        example_batch = xr.load_dataset(input_path).compute()
        print(f"[3/5] Loaded input data: {input_path.name}")

        # ---- 4. Extract inputs / targets / forcings. ----
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            example_batch,
            target_lead_times=target_lead_times,
            **dataclasses.asdict(task_config),
        )
        _validate_input_matches_model(model_config, task_config, inputs)

        # Fail fast with a *clear* message if the requested lead times ran past
        # the end of the input file. When that happens extraction silently
        # produces an empty input window (time dim 0), which otherwise surfaces
        # far downstream as an opaque "retrieved shape (186, 512) does not match
        # [10, 512]" weight error.
        if inputs.sizes.get("time", 0) == 0:
            available = example_batch.sizes.get("time", 0)
            raise ValueError(
                "Requested forecast horizon exceeds the input file: the input "
                f"has only {available} timesteps, so the {task_config.input_duration} "
                "conditioning window left zero input steps. Reduce --steps (or use "
                "a file with more timesteps); this model's input files carry "
                f"{spec.default_steps} target steps by default."
            )

        # IC time = datetime at lead time 0. Extraction shifted the time axis so
        # the last file timestep became the last target lead time; subtracting
        # that horizon from the last datetime recovers lead time 0.
        ref_dt = (
            example_batch["datetime"].isel(time=-1).values.ravel()[0]
            - targets.time.values[-1]
        )
        print(
            f"[4/5] Extracted: inputs {dict(inputs.sizes)}, "
            f"targets {dict(targets.sizes)}, forcings {dict(forcings.sizes)}"
        )

        # ---- 5. Assemble, jit, and run the autoregressive rollout. ----
        @hk.transform_with_state
        def run_forward(model_config, task_config, inputs, targets_template, forcings):
            predictor = _build_wrapped_predictor(model_config, task_config, stats)
            return predictor(inputs, targets_template=targets_template, forcings=forcings)

        def with_configs(fn):
            return functools.partial(fn, model_config=model_config, task_config=task_config)

        def with_params(fn):
            return functools.partial(fn, params=ckpt.params, state={})

        def drop_state(fn):
            return lambda **kw: fn(**kw)[0]

        run_forward_jitted = drop_state(
            with_params(jax.jit(with_configs(run_forward.apply)))
        )

        predictions = rollout.chunked_prediction(
            run_forward_jitted,
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=targets * np.nan,  # template only; values unused
            forcings=forcings,
        )
        print(f"[5/5] Rollout complete: {dict(predictions.sizes)}")

        return predictions, np.datetime64(ref_dt, "h")
