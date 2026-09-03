#!/usr/bin/env bash
# =============================================================================
# End-to-end realtime forecast — entry point (run on the Setonix LOGIN node).
#
# Sets up the inference conda env (the Setonix base env is broken, so activate
# via PATH export), then delegates to scripts/realtime.py for:
#     download -> process -> Slurm inference -> visualize
#
# Usage:
#     ./scripts/realtime.sh                  # latest cycle, full pipeline
#     ./scripts/realtime.sh --no-submit      # download + process only
#     ./scripts/realtime.sh --date 2026-08-31 --time 00
# =============================================================================

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-/scratch/pawsey0115/hwang4/miniconda3/envs/infer-gpu}"
export PATH="$CONDA_ENV/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$HERE/realtime.py" "$@"
