# inference — stage 3, model-agnostic forecast inference

Run a forecast model on a processed input `.nc` and save predictions. This
package is the *orchestration layer*: it is decoupled from any specific model,
and its result-saving is decoupled from the model that produced the results.

```
stage 2 (data_processing)  ->  source-ifs_..._steps-40.nc  (input)
stage 3 (inference)        ->  predictions_<IC>_<model>.nc  (output)
```

## Design

| module                    | responsibility                                        | model-specific? |
|---------------------------|-------------------------------------------------------|-----------------|
| `inference.cli`           | arg parsing, orchestration, lead-time slice           | no              |
| `inference.registry`      | name → `ModelSpec` table (model names, paths, defaults)| no (data only)  |
| `inference.inputs`        | locate the input file / read its IC time              | no              |
| `inference.saver`         | write predictions to netCDF                           | no              |
| `inference.models.base`   | `ModelRunner` interface + `ModelSpec` dataclass       | no              |
| `inference.models.graphcast` | load + run a GraphCast checkpoint                 | **yes**         |

Adding a future model (e.g. one with a 1-hour timestep) means writing one new
`ModelRunner` subclass and adding one `ModelSpec` entry in `registry.py` — the
CLI, input discovery, and saving do not change. The timestep is already
abstracted: GraphCast uses `resolution="6h"` today, but the same `--steps` /
`--resolution` machinery drives any timestep.

## Usage

```powershell
# list registered models
python scripts/run_inference.py --list-models

# run graphcast at the latest available IC, full 40 steps (10-day, 6h steps)
python scripts/run_inference.py --model graphcast

# a specific IC time
python scripts/run_inference.py --model graphcast --init-time 2026-08-27T00

# fewer steps, or an explicit resolution (for future 1h models)
python scripts/run_inference.py --model graphcast --steps 10
python scripts/run_inference.py --model <future_model> --resolution 1h --steps 24
```

Options: `--model`, `--init-time` (default = latest input), `--steps`
(default = the model's `default_steps`), `--resolution` (default = the model's
native timestep), `--data-dir`, `--out-dir`, `--list-models`.

## Environment

- **Stage 3 (this package's graphcast runner)** must run in the environment
  with `jax`, `haiku`, `scipy`, `xarray`, `numpy` — the `mymet` env on this
  machine. The upstream `weathernext` fork is found via each `ModelSpec`'s
  `upstream_dir`.
- Input files come from stage 2; output goes to `../predictions/` by default
  (`INFERENCE_PREDICTIONS_DIR` or `--out-dir` overrides).
