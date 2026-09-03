#!/usr/bin/env bash
# =============================================================================
# Unified end-to-end realtime forecast: GraphCast + Aurora, one command.
#
# Drives BOTH per-model pipelines for the SAME analysis cycle, in sequence:
#
#     download (shared open-data GRIB, idempotent)
#       -> GraphCast: process -> Slurm inference -> visualize
#       -> Aurora:   adapt    -> Slurm inference -> visualize
#
# Each model owns its own env / sbatch / visualizer (see each project's
# scripts/realtime.sh), so this runner stays a thin orchestrator and the models
# stay decoupled. The shared download is run twice but is a fast no-op the second
# time (existing files are skipped).
#
# Usage (run on the Setonix LOGIN node):
#     ./realtime_all.sh                     # latest cycle, both models
#     ./realtime_all.sh --date 2026-08-31 --time 00
#     ./realtime_all.sh --latest --source google
#
# NOTE: do NOT pass --conda-env here — each model uses its own default env.
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default to the newest available cycle when no cycle args are given.
if [ "$#" -eq 0 ]; then
    set -- --latest
fi

echo "########################################################################"
echo "# [1/2] GraphCast (WeatherNext 1 Graph, operational 0.25-deg)           #"
echo "########################################################################"
bash "$HERE/weathernext_forecast/scripts/realtime.sh" "$@"

echo
echo "########################################################################"
echo "# [2/2] Aurora 0.25 finetuned (IFS HRES T0)                             #"
echo "########################################################################"
bash "$HERE/aurora_forecast/scripts/realtime.sh" "$@"

echo
echo "=== realtime_all complete: GraphCast + Aurora predictions saved & visualized ==="
