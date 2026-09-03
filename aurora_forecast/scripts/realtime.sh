#!/usr/bin/env bash
# =============================================================================
# End-to-end realtime Aurora forecast — entry point (run on the Setonix LOGIN node).
#
# Sets up the Aurora conda env (torch + aurora deps; the Setonix base env is
# broken, so activate via PATH export), then delegates to scripts/realtime.py:
#     download -> adapt -> Slurm inference -> visualize
#
# Usage:
#     ./scripts/realtime.sh                  # latest cycle, full pipeline
#     ./scripts/realtime.sh --no-submit      # download + adapt only
#     ./scripts/realtime.sh --date 2026-08-31 --time 00
# =============================================================================

set -euo pipefail

# Login-node stages (download -> adapt -> visualize) run in the existing infer-gpu
# env, which already carries cfgrib/eccodes/netcdf4/matplotlib/cartopy. Only the
# GPU inference job (sbatch) switches to the torch-based `aurora-gpu` env, whose
# path is passed to Slurm via realtime.py's --conda-env.
CONDA_ENV="${CONDA_ENV:-/scratch/pawsey0115/hwang4/miniconda3/envs/infer-gpu}"
export PATH="$CONDA_ENV/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$HERE/realtime.py" "$@"
