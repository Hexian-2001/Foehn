# =============================================================================
# The model-runner interface: the single abstraction every model must satisfy.
# =============================================================================

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import xarray as xr


@dataclass(frozen=True)
class ModelSpec:
    """Everything the orchestrator needs to know about one model.

    This is a plain data record — no model code imported here — so the registry
    can list every model without touching heavy dependencies (jax, haiku, ...).
    The heavy import happens only when ``runner`` is resolved and called.
    """

    name: str                 # CLI key, e.g. "graphcast"
    description: str           # one-line summary for --list-models
    runner: str                # dotted path "module.sub:Class" of a ModelRunner
    checkpoint: Path           # weights file
    stats_dir: Path            # directory holding the normalization stats
    upstream_dir: Path | None  # extra import path the runner needs (or None)
    resolution: str            # native timestep, e.g. "6h" (future: "1h")
    default_steps: int         # default number of forecast steps
    input_pattern: str         # glob matching its input .nc files
    extra: dict[str, Any] = field(default_factory=dict)


class ModelRunner(abc.ABC):
    """Runs one model and returns predictions.

    The contract is deliberately small and model-agnostic:

    * ``run`` receives the input file path and a ``target_lead_times`` slice,
      runs the model, and returns an ``xarray.Dataset`` of *un-normalized,
      physical-unit* predictions (same variables/shape as the targets).

    Saving, file naming, and CLI are all handled *outside* the runner, by
    ``inference.saver`` — a runner never writes its own output file.
    """

    @abc.abstractmethod
    def run(
        self,
        spec: ModelSpec,
        *,
        input_path: Path,
        target_lead_times: slice,
    ) -> xr.Dataset:
        """Run the model and return the predictions Dataset."""
        raise NotImplementedError
