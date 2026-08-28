# =============================================================================
# Central configuration for the inference package.
# =============================================================================
# Every path is derived from THIS file's location so the package stays portable.
# The layout is:
#
#   <forecast_models>/inference/src/inference/config.py
#     parents[0] = .../inference/src/inference   (this package)
#     parents[1] = .../inference/src
#     parents[2] = .../inference                  (the inference project)
#     parents[3] = .../forecast_models            (the shared workspace root)
# =============================================================================

from __future__ import annotations

import os
from pathlib import Path

# The shared workspace root that also holds `weathernext_forecast`,
# `data_processing`, and `opendata_download`.
FORECAST_ROOT = Path(__file__).resolve().parents[3]

# Where processed .nc inputs live (stage-2 `data_processing` writes here by
# default). Overridable via env var so the runner can point at another disk.
DATA_DIR = Path(
    os.environ.get(
        "INFERENCE_DATA_DIR",
        FORECAST_ROOT / "weathernext_forecast" / "data" / "processed",
    )
)

# Where predictions are written. Kept at the workspace root (NOT inside any
# model package) to reflect that result-saving is decoupled from the model.
PREDICTIONS_DIR = Path(
    os.environ.get("INFERENCE_PREDICTIONS_DIR", FORECAST_ROOT / "predictions")
)
