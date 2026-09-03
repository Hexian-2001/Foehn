#!/bin/bash
# =============================================================================
# Render the prediction figure set (GIF + static maps + time series) on a
# dedicated compute node. Reads one predictions .nc (~13 GB) and writes
# <project>/visualizations/<stem>/ — pure CPU + disk I/O, no GPU use.
# =============================================================================
#SBATCH --job-name=viz
#SBATCH --account=pawsey0115-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz_%j.log
#SBATCH --error=logs/viz_%j.err
#SBATCH --chdir=/scratch/pawsey0115/hwang4/projects/Foehn/weathernext_forecast
#SBATCH --export=ALL

set -euo pipefail

module load rocm/6.4.1 2>/dev/null || true

export PATH=/scratch/pawsey0115/hwang4/miniconda3/envs/infer-gpu/bin:$PATH
export PYTHONUNBUFFERED=1

# Predictions land in the external results tree (../results/<model>/<variant>/<init>Z/predictions/).
PRED="${PRED:-$(find ../results -path '*/predictions/*.nc' -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)}"
python -u scripts/visualize.py --predictions "$PRED"
