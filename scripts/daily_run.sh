#!/usr/bin/env bash
# =============================================================================
# Daily end-to-end forecast: inference (GraphCast + Aurora) then push.
#
# Runs on a Pawsey compute node as a self-resubmitting Slurm job (no cron on
# Setonix). One invocation:
#   1. resubmits itself for the next scheduled AWST time *first* (so the schedule
#      survives any failure below),
#   2. waits for the freshest feasible initial field (12Z), then runs
#      realtime_all.sh (download + process + GPU inference),
#   3. pushes both models to the benchmark platform.
#
# Timing rationale (cluster TZ = AWST = UTC+8 = Beijing):
#   * push deadline = 08:00 AWST (= 00:00 UTC). The forecast starts at 16:00 UTC
#     (= next-day Beijing 00:00), auto-computed by push_to_platform.
#   * The freshest ECMWF cycle we can actually use is the PREVIOUS-DAY 12Z:
#       - 12Z becomes available ~02:30 AWST (18:30 UTC),
#       - 18Z becomes available ~08:00 AWST (00:00 UTC) = the deadline itself,
#         so there is never time to download + infer with 18Z.
#   * We schedule at 03:00 AWST and poll until 12Z appears (capped), so an ECMWF
#     delay shifts the run later instead of silently falling back to 06Z.
# =============================================================================

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
SCHEDULE_HOUR="${DAILY_SCHEDULE_HOUR:-03}"        # AWST hour to (re)submit for
PARTITION="${DAILY_PARTITION:-gpu-dev}"           # gpu partition is occupied; use gpu-dev
CONTROL_PYTHON="${FOEHN_CONTROL_PYTHON:-/scratch/pawsey0115/hwang4/miniconda3/envs/infer-gpu/bin/python}"
TARGET_HOUR="${DAILY_TARGET_HOUR:-12}"            # freshest feasible cycle (12Z)
WAIT_MAX_MIN="${DAILY_WAIT_MAX_MIN:-120}"         # cap on waiting for 12Z (2 h)

# ── 1. Resubmit-first: queue tomorrow's run before doing any work. ──
NEXT_BEGIN="$(date -d "tomorrow ${SCHEDULE_HOUR}:00" +%Y-%m-%dT%H:%M:%S)"
if sbatch --begin="$NEXT_BEGIN" "$HERE/daily_run.sbatch"; then
    echo "[daily] resubmitted next run for ${NEXT_BEGIN} AWST"
else
    echo "[daily] WARNING: failed to resubmit next run" >&2
fi

# ── 2. Resolve the freshest feasible initial field (12Z), waiting if needed. ──
CYCLE_DATE=""; CYCLE_HOUR=""
deadline=$(( $(date +%s) + WAIT_MAX_MIN * 60 ))
while :; do
    if read -r CYCLE_DATE CYCLE_HOUR < <("$CONTROL_PYTHON" "$REPO_ROOT/scripts/resolve_latest.py" --source google 2>/dev/null) \
       && [ -n "$CYCLE_DATE" ] && [ -n "$CYCLE_HOUR" ]; then
        if [ "$CYCLE_HOUR" = "$TARGET_HOUR" ]; then
            echo "[daily] using freshest feasible cycle ${CYCLE_DATE}T${CYCLE_HOUR}Z"
            break
        fi
        echo "[daily] latest is ${CYCLE_DATE}T${CYCLE_HOUR}Z; waiting for ${TARGET_HOUR}Z ..."
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "[daily] WARNING: ${TARGET_HOUR}Z not available within ${WAIT_MAX_MIN} min; proceeding with latest (${CYCLE_DATE:-?}T${CYCLE_HOUR:-?}Z)" >&2
        break
    fi
    sleep 300
done

# ── 3. Inference. Tolerate a single-model failure so the push still runs. ──
RT_ARGS=(--partition "$PARTITION")
if [ -n "$CYCLE_DATE" ] && [ -n "$CYCLE_HOUR" ]; then
    RT_ARGS+=(--date "$CYCLE_DATE" --time "$CYCLE_HOUR")
else
    RT_ARGS+=(--latest)   # last resort: cycle resolution itself failed
fi
echo "[daily] === inference (realtime_all.sh ${RT_ARGS[*]}) ==="
if bash "$REPO_ROOT/realtime_all.sh" "${RT_ARGS[@]}"; then
    echo "[daily] inference OK"
else
    echo "[daily] inference FAILED — continuing to push whatever predictions exist" >&2
fi

# ── 4. Push. Always attempted; uses the newest prediction <= target start. ──
echo "[daily] === push (push_to_platform/realtime_push.sh) ==="
if bash "$REPO_ROOT/push_to_platform/realtime_push.sh"; then
    echo "[daily] push OK"
else
    echo "[daily] push FAILED" >&2
fi

echo "[daily] done"
