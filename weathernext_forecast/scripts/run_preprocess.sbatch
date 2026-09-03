#!/bin/bash
#SBATCH --job-name=preproc
#SBATCH --account=pawsey0115-gpu
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/preprocess_%j.log
#SBATCH --chdir=/scratch/pawsey0115/hwang4/projects/Foehn/weathernext_forecast

# Download (idempotent) + process GRIB -> GraphCast input .nc on a dedicated
# compute node, isolated from login-node contention. Does NOT run inference.
module load rocm/6.4.1 2>/dev/null || true

CONDA_ENV="${CONDA_ENV:-/scratch/pawsey0115/hwang4/miniconda3/envs/infer-gpu}"
export PATH="$CONDA_ENV/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u scripts/realtime.py --no-submit
