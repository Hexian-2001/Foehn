#!/bin/bash
# =============================================================================
# GraphCast operational (0.25 deg) inference on Pawsey Setonix (AMD MI250X).
#
# This script exists to defeat the THREE hang sites that stop the GPU run on a
# cold / shared Setonix node:
#   1. `import jax` deadlock on a cold GPU  -> `rocminfo` warm-up (below).
#   2. TISR solar forcing jitted onto a contended GPU -> fixed in inference.py
#      (pinned to CPU) and this script now has a preflight to prove it.
#   3. jit-compile of the big GNN looks like a hang -> preflight + flushed
#      step markers tell "slow compile" from "true deadlock".
#
# Adjust the ---- SETONIX ---- block to your project/queue; everything else is
# portable.
# =============================================================================
#SBATCH --job-name=graphcast_oper
#SBATCH --account=pawsey0115-gpu
#SBATCH --partition=gpu                      # <-- set this (Setonix: `gpu` / `gpu-high-mem`)
#SBATCH --nodes=1
#SBATCH --exclusive                          # whole node (8x MI250X): reliable 24/7 runs, no
#                                            # GPU contention that can stall the run. This single
#                                            # forecast still uses ONE GCD. Cheaper alternative:
#                                            #   drop --exclusive, use --gpus-per-node=1
# NOTE: Setonix GPU allocations REJECT `--cpus-per-task`/`--mem` (each GCD
# auto-allocates 8 cores; `--exclusive` gives the whole node anyway).
#SBATCH --time=02:00:00
#SBATCH --output=logs/graphcast_%j.out
#SBATCH --error=logs/graphcast_%j.err

set -euo pipefail

# ---- SETONIX: environment ------------------------------------------------
module load rocm/6.4.1                       # per the infer-gpu env that was tested
# conda's base env on Setonix is broken (frozendict ImportError), so
# `conda activate` fails. Point PATH at the env directly instead — this is the
# verified-working activation for infer-gpu.
export PATH=/scratch/pawsey0115/hwang4/miniconda3/envs/infer-gpu/bin:$PATH
# --------------------------------------------------------------------------

# ---- 1. Warm the ROCm/HSA stack BEFORE anything imports jax --------------
# jax 0.4.35's ROCm backend deadlocks on the first GPU touch after the node
# went idle. `rocminfo` runs hsa_init + loads GPU firmware; after it,
# `import jax` returns instantly. This line is what makes a cold node work.
echo "[warmup] rocminfo (hsa_init + firmware load) ..."
rocminfo > /dev/null 2>&1 || true
echo "[warmup] done"

# ---- 2. Single-GCD visibility (avoid multi-device ambiguity) -------------
# Prefer letting Slurm set visibility via --gpus-per-node. Only uncomment the
# line below if you must force it manually; a wrong index here can target a
# GCD you don't own on a shared node.
# export ROCR_VISIBLE_DEVICES=0

# ---- 3. Tame MI250X memory behaviour -------------------------------------
# Preallocation on ROCm can make a large model look like it OOM-hung. Allow
# growth and cap the fraction so the OS + runtime stay responsive.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# Persist the XLA compile cache so the 24/7 reruns skip recompilation.
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax_compilation_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# ---- 4. Line-buffered output: the log must show WHERE it hangs -----------
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."                      # repo root: weathernext_forecast/
mkdir -p logs

# ---- 5. Preflight: isolate import/GPU-init from the model run ------------
echo "[preflight] import jax + device topology ..."
python -u - <<'PY' 2>&1
import jax
print("backend:", jax.default_backend())
print("devices:", jax.devices())
PY
echo "[preflight] import OK"

# ---- 6. The real run -----------------------------------------------------
# Step markers [0/6]..[6/6] are flushed to the log. Interpretation:
#   stop after [0/6]        -> import/GPU init deadlock (warm-up missing/failed)
#   stop after [3/6]        -> TISR forcing / data stage
#   stop after [5/6]        -> jit-compile (5-20 min is normal; beyond that + idle
#                             GPU in `rocm-smi` = a real 0.4.35 compile hang)
#   no [6/6] after rollout  -> rollout/execution hang
timeout 90m python -u scripts/run_inference.py 2>&1
echo "[done] inference finished"
