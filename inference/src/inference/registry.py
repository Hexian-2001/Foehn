# =============================================================================
# Model registry: the single table mapping a CLI model name to its ModelSpec.
# =============================================================================
# To add a future model you only touch THIS file (plus write its runner class):
# append one entry. The CLI, input discovery, and saving all stay unchanged.

from __future__ import annotations

import importlib
from pathlib import Path

from inference import config
from inference.models.base import ModelRunner, ModelSpec

_WN = config.FORECAST_ROOT / "weathernext_forecast"
_WEIGHTS = _WN / "models" / "weights"
_STATS = _WN / "models" / "stats"
_UPSTREAM = _WN / "upstream" / "weathernext"

MODELS: dict[str, ModelSpec] = {
    "graphcast": ModelSpec(
        name="graphcast",
        description=(
            "GraphCast operational, 0.25 deg / 13 levels / HRES-fc0 "
            "(no precipitation input)"
        ),
        runner="inference.models.graphcast:GraphCastRunner",
        checkpoint=_WEIGHTS / "GraphCast_operational.npz",
        stats_dir=_STATS,
        upstream_dir=_UPSTREAM,
        resolution="6h",
        default_steps=40,
        input_pattern="source-ifs_*.nc",
    ),
    "graphcast_small": ModelSpec(
        name="graphcast_small",
        description=(
            "GraphCast small, 1.0 deg / 13 levels / ERA5 "
            "(CPU-friendly; smoke test)"
        ),
        runner="inference.models.graphcast:GraphCastRunner",
        checkpoint=_WEIGHTS / "GraphCast_small.npz",
        stats_dir=_STATS,
        upstream_dir=_UPSTREAM,
        resolution="6h",
        default_steps=4,
        input_pattern="source-era5_*.nc",
    ),
}


def get_spec(name: str) -> ModelSpec:
    """Return the ModelSpec for a model name, or raise a helpful error."""
    try:
        return MODELS[name]
    except KeyError:
        known = ", ".join(sorted(MODELS))
        raise KeyError(f"unknown model '{name}' (available: {known})") from None


def load_runner(spec: ModelSpec) -> ModelRunner:
    """Resolve a spec's ``runner`` dotted path and instantiate the class."""
    module_name, _, cls_name = spec.runner.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)()
