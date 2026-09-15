#!/usr/bin/env bash
# Daily push stage: after GraphCast/Aurora inference finishes, extract the wind
# series and push them to the benchmark platform.
#
# Runs on Pawsey. Assumes the forecast_models tree layout (results/). The
# platform URL and provider API keys come from push_to_platform/.env (or the
# environment), so credentials never live in source:
#   BENCHMARK_PUSH_URL   e.g. https://benchmark.mingyangai.com.cn
#   BENCHMARK_PUSH_KEYS  "graphcast=sk-...,aurora=sk-..."
#
# The forecast is issued at a fixed 16:00 UTC instant (= next-day Beijing 00:00).
# When run during UTC day D, the default --start-date is (D+1)T16:00:00Z, which
# the platform expects (deadline (D+1)T00:00Z). An explicit start instant may be
# given for backfill/replay.
#
# Usage:  bash push_to_platform/realtime_push.sh [YYYY-MM-DDTHH:MM:SSZ]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# Credentials: prefer push_to_platform/.env, fall back to the ambient environment.
if [ -f "$HERE/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$HERE/.env"
  set +a
fi

# Python with xarray + numpy (the infer-gpu conda env on Pawsey). Note the
# repo lives at <home>/projects/Foehn, so the env is TWO levels up from ROOT
# (one level up is <home>/projects, not <home>).
PY="${PYTHON:-$ROOT/../../miniconda3/envs/infer-gpu/bin/python}"
# Fall back to whatever `python` resolves to if the canonical path is absent.
if ! command -v "$PY" >/dev/null 2>&1; then
  PY="$(command -v python || true)"
  [ -n "$PY" ] || { echo "[push] FATAL: no usable python found" >&2; exit 2; }
fi

START_DATE="${1:-}"
START_ARGS=()
if [ -n "$START_DATE" ]; then
  START_ARGS=(--start-date "$START_DATE")
fi

echo "[push] start_date=${START_DATE:-auto (tomorrow 16:00Z)}"

rc=0
for MODEL in graphcast aurora; do
  echo "[push] ${MODEL} ..."
  if "$PY" -m push_to_platform.cli \
       --model "$MODEL" "${START_ARGS[@]}" \
       --submission-type realtime; then
    echo "[push] ${MODEL} OK"
  else
    echo "[push] ${MODEL} FAILED (see log)" >&2
    rc=1
    # Continue with the next model; do not abort the whole stage.
  fi
done

echo "[push] done"
exit "$rc"
