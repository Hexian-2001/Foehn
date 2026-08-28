# =============================================================================
# Input discovery: map a model + IC time onto a concrete .nc input file.
# =============================================================================

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def _init_time_of(path: Path) -> np.datetime64:
    """Read the forecast IC time embedded in an input .nc file.

    The IC is the datetime at forecast lead time 0 (the last *input* step).
    Stage-2 files lay their ``time`` axis out with a ``time == 0`` element at
    exactly that step, so we select the step nearest lead time 0 — robust to
    floating-point noise and independent of the filename. The absolute datetime
    there is the initialization time.
    """
    with xr.open_dataset(path, decode_timedelta=True) as ds:
        time = np.asarray(ds["time"].values, dtype="timedelta64[ns]")
        idx = int(np.argmin(np.abs(time)))
        return np.datetime64(ds["datetime"].values.ravel()[idx])


def resolve_input(
    spec_name: str,
    input_pattern: str,
    data_dir: Path,
    init_time: np.datetime64 | None,
) -> Path:
    """Pick the input file for ``init_time``, defaulting to the latest.

    If ``init_time`` is None, the newest file (by mtime) matching
    ``input_pattern`` is returned. Otherwise the file whose embedded IC time
    matches ``init_time`` is returned. Both look only at filenames/mtimes, so
    this is fast even with many files present.
    """
    candidates = sorted(data_dir.glob(input_pattern))
    if not candidates:
        raise FileNotFoundError(
            f"no input files matching {input_pattern!r} in {data_dir}"
        )

    if init_time is None:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    target = np.datetime64(init_time, "h")
    for p in candidates:
        if np.datetime64(_init_time_of(p), "h") == target:
            return p

    raise FileNotFoundError(
        f"no input file with IC time {target} for model '{spec_name}' "
        f"(found files: {[p.name for p in candidates]})"
    )
