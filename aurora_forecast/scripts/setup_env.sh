#!/usr/bin/env bash
# =============================================================================
# Create the Aurora inference conda env on Setonix (run ONCE, on the login node).
#
# Aurora is PyTorch (NOT jax), so it needs its own env (`aurora-gpu`) with a
# ROCm-built torch. The login-node stages (download/adapt/visualize) keep using
# the existing `infer-gpu` env; only the GPU inference sbatch job uses this env.
#
# Hardware: MI250X = gfx90a, a first-class ROCm target (no HSA override needed).
#
# Usage:
#     bash scripts/setup_env.sh
# =============================================================================

set -euo pipefail

ENV_NAME="${1:-aurora-gpu}"
ENV_ROOT="${CONDA_ENVS_ROOT:-/scratch/pawsey0115/hwang4/miniconda3/envs}"
ENV_DIR="$ENV_ROOT/$ENV_NAME"
PY_VER="3.11"

module load rocm/6.4.1

# Pin to conda-forge only (`--override-channels`). The default anaconda.com
# channels now require accepting Terms of Service under conda 26.x, which aborts
# non-interactive `conda create`/`install` with CondaToSNonInteractiveError.
# conda-forge has no ToS gate, so this makes the env buildable unattended.
echo "[env] creating $ENV_DIR (python $PY_VER)"
if [ -d "$ENV_DIR" ]; then
    echo "      already exists; installing/upgrading deps in place"
else
    conda create -y -p "$ENV_DIR" -c conda-forge --override-channels python="$PY_VER"
fi
CONDA_PY="$ENV_DIR/bin/python"

echo "[deps] conda-forge (binary stack: numpy/scipy/xarray/netcdf4/pillow)"
# pillow is a transitive dep of torchvision; pull it from conda-forge up-front so
# the ROCm-index-only pip install below never has to look for it off-index.
conda install -y -p "$ENV_DIR" -c conda-forge --override-channels \
    numpy scipy xarray netcdf4 h5py pillow 2>&1 | tail -n 3

# torch AND torchvision MUST both come from the ROCm index. `timm` (installed
# next, from PyPI) depends on `torchvision`; if torchvision is missing here, pip
# resolves that dep to the latest PyPI wheel — the CUDA build — which then pins
# torch==2.13.0 and silently replaces the ROCm torch with the CUDA one. Installing
# the ROCm torchvision first satisfies timm's requirement in place.
echo "[deps] torch + torchvision (ROCm 6.4, gfx90a) via the official PyTorch index"
"$CONDA_PY" -m pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4

echo "[deps] aurora pure-python deps (from microsoft/aurora pyproject.toml)"
"$CONDA_PY" -m pip install --no-cache-dir \
    einops timm huggingface-hub pydantic azure-storage-blob

echo "[verify]"
"$CONDA_PY" -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"

echo "[done] Aurora env ready at $ENV_DIR"
echo "       Real-time inference: ./scripts/realtime.sh --conda-env $ENV_DIR"
