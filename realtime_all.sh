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
CONTROL_ENV="${FOEHN_CONTROL_ENV:-/scratch/pawsey0115/hwang4/miniconda3/envs/infer-gpu}"
CONTROL_PYTHON="${FOEHN_CONTROL_PYTHON:-$CONTROL_ENV/bin/python}"
LOG_DIR="${FOEHN_LOG_DIR:-$HERE/logs}"
mkdir -p "$LOG_DIR"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="$LOG_DIR/realtime_all_${RUN_STAMP}.log"
exec > >(tee -a "$RUN_LOG") 2>&1
trap 'rc=$?; echo "[$(date -u +%FT%TZ)] FAILED line=$LINENO exit=$rc"; exit "$rc"' ERR

echo "[$(date -u +%FT%TZ)] Foehn dual-model run started"
echo "log: $RUN_LOG"

# Default to the newest available cycle when no cycle args are given.
if [ "$#" -eq 0 ]; then
    set -- --latest
fi

# Resolve --latest exactly once. Without this normalization, GraphCast and
# Aurora query the catalogue independently and can select different cycles if a
# new analysis appears between their sequential runs.
ARGS=("$@")
SOURCE="google"
HAS_LATEST=0
HAS_DATE=0
HAS_TIME=0
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    case "${ARGS[$i]}" in
        --latest) HAS_LATEST=1 ;;
        --date) HAS_DATE=1; ((i += 1)) ;;
        --date=*) HAS_DATE=1 ;;
        --time) HAS_TIME=1; ((i += 1)) ;;
        --time=*) HAS_TIME=1 ;;
        --source)
            ((i += 1))
            SOURCE="${ARGS[$i]}"
            ;;
        --source=*) SOURCE="${ARGS[$i]#*=}" ;;
    esac
done

if [ "$HAS_LATEST" -eq 1 ] || { [ "$HAS_DATE" -eq 0 ] && [ "$HAS_TIME" -eq 0 ]; }; then
    if [ "$HAS_DATE" -eq 1 ] || [ "$HAS_TIME" -eq 1 ]; then
        echo "error: --latest cannot be combined with --date/--time" >&2
        exit 2
    fi
    read -r CYCLE_DATE CYCLE_HOUR < <("$CONTROL_PYTHON" "$HERE/scripts/resolve_latest.py" --source "$SOURCE")
    NORMALIZED=()
    for arg in "${ARGS[@]}"; do
        [ "$arg" = "--latest" ] || NORMALIZED+=("$arg")
    done
    NORMALIZED+=(--date "$CYCLE_DATE" --time "$CYCLE_HOUR")
    set -- "${NORMALIZED[@]}"
    echo "resolved shared cycle: ${CYCLE_DATE}T${CYCLE_HOUR}Z (source=$SOURCE)"
elif [ "$HAS_DATE" -ne 1 ] || [ "$HAS_TIME" -ne 1 ]; then
    echo "error: provide both --date and --time, or --latest" >&2
    exit 2
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
echo "[$(date -u +%FT%TZ)] SUCCESS"
