# =============================================================================
# Result saving — deliberately decoupled from any model.
# =============================================================================
# This module knows nothing about GraphCast (or any model). It receives an
# xarray.Dataset of predictions plus a few metadata strings and writes a netCDF
# file. Swapping models, or changing the output layout, never touches a runner.

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def save_predictions(
    predictions: xr.Dataset,
    *,
    model_name: str,
    init_time: np.datetime64,
    resolution: str,
    steps: int,
    out_dir: Path,
) -> Path:
    """Write predictions to ``out_dir`` and return the output path.

    The filename encodes the model, IC time, resolution, and step count, so the
    forecast horizon is recoverable from the name alone::

        predictions_2026-08-27T00_graphcast_res-6h_steps-40.nc

    meaning IC 2026-08-27T00, 6-hour steps, 40 steps -> valid through
    2026-09-06T00. ``resolution`` is the raw string (``"6h"``, ``"1h"``) with
    any filesystem-unsafe characters replaced by ``_``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_time = np.datetime_as_string(np.datetime64(init_time, "h"), unit="h")
    # numpy renders "T" + "00:00" for a round hour; tighten to "T00".
    ref_time = ref_time.replace("T00:00", "T00")
    res = "".join(c if c.isalnum() else "_" for c in resolution)
    out_name = (
        f"predictions_{ref_time}_{model_name}_res-{res}_steps-{steps}.nc"
    )
    out_path = out_dir / out_name
    predictions.to_netcdf(out_path)
    return out_path
