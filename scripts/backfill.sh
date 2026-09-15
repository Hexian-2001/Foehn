#!/usr/bin/env bash
# =============================================================================
# Backfill missed/late forecast days (inference + push) for GraphCast + Aurora.
#
# Runs on a Pawsey CPU node (see backfill.sbatch). For each missed day it:
#   1. runs inference for the freshest init cycle that was available at the
#      original push deadline  = (D-1)T12:00Z  (lead 28 h to start_date D T16:00Z),
#      via realtime_all.sh --date <D-1> --time 12 --partition gpu-dev;
#   2. pushes BOTH models for start_date D T16:00:00Z with --submission-type
#      backfill, immediately after that cycle (before the next cycle exists), so
#      find_prediction_nc() resolves to exactly this cycle (lead 28 h, valid).
#
# gpu-dev is used because the gpu partition is occupied by the user's training.
# =============================================================================

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
PARTITION="${BACKFILL_PARTITION:-gpu-dev}"
PY="${PYTHON:-/scratch/pawsey0115/hwang4/miniconda3/envs/infer-gpu/bin/python}"

# Credentials for the push stage (BENCHMARK_PUSH_URL / BENCHMARK_PUSH_KEYS).
if [ -f "$REPO_ROOT/push_to_platform/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/push_to_platform/.env"
  set +a
fi

# init_date|start_date  (init 12Z(D-1) -> start 16Z(D)).
DAYS=(
  "2026-09-10|2026-09-11T16:00:00Z"
  "2026-09-11|2026-09-12T16:00:00Z"
  "2026-09-12|2026-09-13T16:00:00Z"
  "2026-09-13|2026-09-14T16:00:00Z"
)

rc_all=0
for entry in "${DAYS[@]}"; do
  init_date="${entry%%|*}"
  start_date="${entry##*|}"
  echo ""
  echo "======================================================================"
  echo "[backfill] init=${init_date}T12Z -> start=${start_date}  (partition=${PARTITION})"
  echo "======================================================================"

  # 1. Inference: download + process + GPU inference, both models, --wait.
  if bash "$REPO_ROOT/realtime_all.sh" --date "$init_date" --time 12 --partition "$PARTITION"; then
    echo "[backfill] inference OK for ${init_date}T12Z"
  else
    echo "[backfill] inference FAILED for ${init_date}T12Z — skipping its push" >&2
    rc_all=1
    continue
  fi

  # 2. Push both models for this start_date, backfill type.
  cd "$REPO_ROOT" || exit 1
  for MODEL in graphcast aurora; do
    if "$PY" -m push_to_platform.cli \
         --model "$MODEL" --start-date "$start_date" --submission-type backfill; then
      echo "[backfill] $MODEL push OK for $start_date"
    else
      echo "[backfill] $MODEL push FAILED for $start_date" >&2
      rc_all=1
    fi
  done
done

echo ""
echo "[backfill] complete (rc=$rc_all)"
exit "$rc_all"
