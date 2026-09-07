#!/usr/bin/env bash
# Daily push stage: after GraphCast/Aurora inference finishes for today's 00Z
# cycle, extract the wind series and push them to the benchmark platform.
#
# Runs on Pawsey. Assumes the forecast_models tree layout (results/, and the
# stage-2 IFS input .nc used as the lead-0 anchor). The platform URL and the
# provider API keys come from the environment:
#   BENCHMARK_PUSH_URL   e.g. https://benchmark.mingyangai.com.cn
#   BENCHMARK_PUSH_KEYS  "graphcast=sk-...,aurora=sk-..."
#
# Usage:  bash push_to_platform/realtime_push.sh [YYYY-MM-DD]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# Python with xarray + numpy (the aurora or mymet conda env).
PY="${PYTHON:-python}"

DATE="${1:-$(date -u +%Y-%m-%d)}"
echo "[push] date=${DATE} (00Z)"

for MODEL in graphcast aurora; do
  echo "[push] ${MODEL} ..."
  if "$PY" -m push_to_platform.cli \
       --model "$MODEL" --date "$DATE" \
       --submission-type realtime; then
    echo "[push] ${MODEL} OK"
  else
    echo "[push] ${MODEL} FAILED (see log)" >&2
    # Continue with the next model; do not abort the whole stage.
  fi
done

echo "[push] done"
