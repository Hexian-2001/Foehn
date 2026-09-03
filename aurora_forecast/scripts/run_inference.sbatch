#!/bin/bash
#SBATCH --job-name=aurora_fin
# =============================================================================
# GPU inference for Microsoft Aurora 0.25 finetuned (IFS HRES T0).
#
# Resource flags (--account, --partition, --nodes, --gpus-per-node, --time,
# --output, --chdir, --export) are passed on the `sbatch` command line by
# scripts/realtime.py — do NOT edit them here. This file only sets up the
# runtime environment and launches the inference.
#
# Aurora is PyTorch (NOT jax) and ~1 GPU is ample — far lighter than the
# 3-GPU GraphCast operational run.
# =============================================================================

set -euo pipefail

module load rocm/6.4.1

# Aurora conda env (torch built for ROCm + aurora deps). The Setonix base env is
# broken, so activate via PATH export, NOT `conda activate`.
CONDA_ENV="${CONDA_ENV:-/scratch/pawsey0115/hwang4/miniconda3/envs/aurora-gpu}"
export PATH="$CONDA_ENV/bin:$PATH"

# MI250X is gfx90a — a first-class ROCm target, so no HSA_OVERRIDE_GFX_VERSION
# is needed (that override is for consumer RDNA / MI300 gfx942 cards).

# AURORA_INPUT_FILENAME is inherited from the sbatch --export (set by
# realtime.py) so this run points at the exact .nc just downloaded/processed.

# Warm the ROCm context so torch's HIP init doesn't race a cold device.
echo "[warmup] rocminfo"
rocminfo >/dev/null 2>&1 || true
echo "[warmup] done"

echo "[preflight] torch"
python -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"

echo "[run] inference"
python -u scripts/run_inference.py
