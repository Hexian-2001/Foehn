#!/usr/bin/env bash
# =============================================================================
# Daily end-to-end forecast: inference (GraphCast + Aurora) then push.
#
# Runs on a Pawsey compute node as a self-resubmitting Slurm job (no cron on
# Setonix). One invocation:
#   1. resubmits itself for the next 02:00 AWST *first* (so the schedule survives
#      any failure below),
#   2. runs realtime_all.sh --latest (download + process + GPU inference),
#   3. pushes both models to the benchmark platform.
#
# Timing rationale (cluster TZ = AWST = UTC+8):
#   * 02:00 AWST = 18:00 UTC — just after the 12Z cycle's ~6 h availability delay,
#     so --latest resolves 12Z and the push uses the freshest initial field.
#   * push deadline = 08:00 AWST (= 00:00 UTC) and the issued forecast starts at
#     16:00 UTC (next-day Beijing 00:00); default --start-date is auto-computed
#     by push_to_platform.
# =============================================================================

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
SCHEDULE_HOUR="${DAILY_SCHEDULE_HOUR:-02}"   # AWST hour to resubmit for
PARTITION="${DAILY_PARTITION:-gpu-dev}"      # gpu partition is occupied; use gpu-dev

# ── 1. Resubmit-first: queue tomorrow's run before doing any work. ──
# Using --begin with an explicit timestamp keeps the schedule self-correcting
# even if today's job starts late (it always targets the next 02:00 AWST).
NEXT_BEGIN="$(date -d "tomorrow ${SCHEDULE_HOUR}:00" +%Y-%m-%dT%H:%M:%S)"
if sbatch --begin="$NEXT_BEGIN" "$HERE/daily_run.sbatch"; then
    echo "[daily] resubmitted next run for ${NEXT_BEGIN} AWST"
else
    echo "[daily] WARNING: failed to resubmit next run" >&2
fi

# ── 2. Inference. Tolerate a single-model failure so the push still runs. ──
echo "[daily] === inference (realtime_all.sh --latest --partition ${PARTITION}) ==="
if bash "$REPO_ROOT/realtime_all.sh" --latest --partition "$PARTITION"; then
    echo "[daily] inference OK"
else
    echo "[daily] inference FAILED — continuing to push whatever predictions exist" >&2
fi

# ── 3. Push. Always attempted; uses the newest prediction <= target start. ──
echo "[daily] === push (push_to_platform/realtime_push.sh) ==="
if bash "$REPO_ROOT/push_to_platform/realtime_push.sh"; then
    echo "[daily] push OK"
else
    echo "[daily] push FAILED" >&2
fi

echo "[daily] done"
