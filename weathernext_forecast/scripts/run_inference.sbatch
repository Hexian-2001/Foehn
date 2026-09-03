#!/bin/bash
#SBATCH --job-name=gc_oper
# =============================================================================
# GPU inference for GraphCast operational (WeatherNext 1 Graph).
#
# Resource flags (--account, --partition, --nodes, --gpus-per-node, --time,
# --output, --chdir, --export) are passed on the `sbatch` command line by
# scripts/realtime.py — do NOT edit them here. This file only sets up the
# runtime environment and launches the inference.
#
# Expected to run with --chdir=<weathernext_forecast> so `scripts/run_inference.py`
# resolves relative to the project root.
# =============================================================================

set -euo pipefail

module load rocm/6.4.1

# The inference conda env (jax 0.4.35 + rocm plugin + deps). The base env on
# Setonix is broken, so activate via PATH export, NOT `conda activate`.
CONDA_ENV="${CONDA_ENV:-/scratch/pawsey0115/hwang4/miniconda3/envs/infer-gpu}"
export PATH="$CONDA_ENV/bin:$PATH"

# JAX preallocation off (jax default-device placement handles memory); keep the
# persistent XLA compile cache so the 5-20 min ROCm jit-compile is amortized.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax_compilation_cache}"

# WEATHERNEXT_INPUT_FILENAME is inherited from the sbatch --export (set by
# realtime.py) so this run points at the exact .nc just downloaded/processed.

# Warm the ROCm context so `import jax`/GPU init doesn't race a cold device.
echo "[warmup] rocminfo"
rocminfo >/dev/null 2>&1 || true
echo "[warmup] done"

# Fail fast with the device topology visible in the log (backend + GCD count).
python -u -c "import jax; print('[preflight] backend', jax.default_backend(), jax.devices())"

echo "[run] inference"
python -u scripts/run_inference.py
