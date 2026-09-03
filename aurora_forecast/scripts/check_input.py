#!/usr/bin/env python
"""CPU-only sanity check: does the Aurora input .nc load as a valid Batch?

Run on the login node (no GPU needed) inside the aurora-gpu env to confirm the
adapter's output is readable by the upstream ``Batch.from_netcdf`` and has the
shapes the model expects. This is exactly the first step the GPU job performs.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "upstream" / "aurora"))

from aurora import Batch  # noqa: E402

input_nc = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    PROJECT_ROOT / "data" / "processed" / "2026-09-01T18Z_aurora_input.nc"
)

b = Batch.from_netcdf(input_nc)
print("surf_vars  ", {k: tuple(v.shape) for k, v in b.surf_vars.items()})
print("static_vars", {k: tuple(v.shape) for k, v in b.static_vars.items()})
print("atmos_vars ", {k: tuple(v.shape) for k, v in b.atmos_vars.items()})
print("metadata.time        ", list(b.metadata.time))
print("metadata.lat shape   ", tuple(b.metadata.lat.shape))
print("metadata.lon shape   ", tuple(b.metadata.lon.shape))
print("metadata.atmos_levels", list(b.metadata.atmos_levels))
print("metadata.rollout_step ", getattr(b.metadata, "rollout_step", None))
print("CHECK_OK")
