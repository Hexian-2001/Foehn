#!/bin/bash
# =============================================================================
# End-to-end realtime forecast on ONE compute node:
#   download (idempotent, cached) -> process GRIB -> input .nc
#   -> GPU inference (3 GCDs) -> predictions .nc
#   -> visualize (GIF + static maps + time series)
#
# Runs entirely off the login node, so it is immune to login-node contention.
# Submit once; monitor logs/e2e_<jobid>.log.
# =============================================================================
#SBATCH --job-name=gc_e2e
#SBATCH --account=pawsey0115-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --time=02:00:00
#SBATCH --output=logs/e2e_%j.log
#SBATCH --error=logs/e2e_%j.err
#SBATCH --chdir=/scratch/pawsey0115/hwang4/projects/Foehn/weathernext_forecast
#SBATCH --export=ALL,DATE=2026-08-30,HOUR=18

set -euo pipefail

module load rocm/6.4.1

CONDA_ENV="${CONDA_ENV:-/scratch/pawsey0115/hwang4/miniconda3/envs/infer-gpu}"
export PATH="$CONDA_ENV/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax_compilation_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
export PYTHONUNBUFFERED=1

DATE="${DATE:-2026-08-30}"
HOUR="${HOUR:-18}"

# Warm ROCm before anything imports jax (defeats the cold-GCD deadlock).
echo "[warmup] rocminfo"
rocminfo >/dev/null 2>&1 || true
echo "[warmup] done"

# ---- 1. download (idempotent, cached) + process GRIB -> input .nc ----------
echo "=== [1/3] process ($DATE ${HOUR}Z) ==="
python -u scripts/realtime.py --no-submit --date "$DATE" --time "$HOUR"

# ---- 2. GPU inference (3 GCDs) ---------------------------------------------
echo "=== [2/3] inference ==="
INPUT_NC=$(ls -t data/processed/source-ifs_date-${DATE}_res-*.nc | head -1)
echo "input: $INPUT_NC"
export WEATHERNEXT_INPUT_FILENAME="$(basename "$INPUT_NC")"
python -u -c "import jax; print('[preflight] backend', jax.default_backend(), jax.devices())"
python -u scripts/run_inference.py

# ---- 3. visualization ------------------------------------------------------
echo "=== [3/3] visualize ==="
# Predictions land in the external results tree (../results/<model>/<variant>/<init>Z/predictions/).
PRED=$(find ../results -path '*/predictions/*.nc' -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
echo "predictions: $PRED"
python -u scripts/visualize.py --predictions "$PRED"

echo "=== e2e complete ==="
