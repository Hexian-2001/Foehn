#!/usr/bin/env bash
# =============================================================================
# Sync between local machine and Pawsey Setonix.
#
#     ./sync.sh          # pull the results tree down (default)
#     ./sync.sh pull     # same
#     ./sync.sh push     # push source code up (scripts + src of all packages)
#
# Rules (industrial-grade decoupling):
#   * `pull` transfers the EXTERNAL results tree (<repo>/results) — predictions
#     + visualizations organized by model — and NEVER any file > 1 GB (e.g. a
#     full-global prediction). The durable China-region prediction (~0.4 GB) and
#     the small visualization images move down; the full-global source does not.
#   * `push` transfers only source code (scripts + src), never data / models /
#     predictions.
# =============================================================================

set -euo pipefail

HOST="${PAWSEY_HOST:-hwang4@setonix.pawsey.org.au}"
REMOTE_ROOT="/scratch/pawsey0115/hwang4/projects/Foehn"
LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"   # forecast_models/

MODE="${1:-pull}"
MAX_SIZE="1G"   # never transfer anything at or above this (predictions are ~13 GB full-global)

if command -v rsync >/dev/null 2>&1; then
    RSYNC=1
else
    RSYNC=0
    echo "note: rsync not found; using scp (re-copies everything)" >&2
fi

case "$MODE" in
  pull)
    REMOTE="$REMOTE_ROOT/results"
    LOCAL="$LOCAL_ROOT/results"
    echo "pull: $HOST:$REMOTE/  ->  $LOCAL/   (excluding files >= $MAX_SIZE)"
    mkdir -p "$LOCAL"
    if [ "$RSYNC" = 1 ]; then
      rsync -av --delete --max-size="$MAX_SIZE" "$HOST:$REMOTE/" "$LOCAL/"
    else
      # scp cannot filter by size, so stream a tar of only the sub-1GB files.
      ssh "$HOST" "cd '$REMOTE_ROOT' && find results -type f -size -${MAX_SIZE} -print0 | tar --null -T - -cf -" \
        | tar -xf - -C "$LOCAL_ROOT"
    fi
    ;;
  push)
    # Only source, never data / models / predictions.
    # `aurora_forecast/upstream/aurora/aurora` is the importable `aurora` package
    # (pure Python, read-only dependency) — pushed as code so `import aurora`
    # resolves on Pawsey without a separate git clone. Model weights (>=1 GB) are
    # NOT pushed; fetch them on Pawsey with aurora_forecast/scripts/download_weights.sh.
    for pkg in \
        weathernext_forecast/scripts \
        weathernext_forecast/src \
        opendata_download/src \
        data_processing/src \
        aurora_forecast/scripts \
        aurora_forecast/src \
        aurora_forecast/upstream/aurora/aurora; do
      echo "push: $LOCAL_ROOT/$pkg/  ->  $HOST:$REMOTE_ROOT/$pkg/"
      if [ "$RSYNC" = 1 ]; then
        rsync -av "$LOCAL_ROOT/$pkg/" "$HOST:$REMOTE_ROOT/$pkg/"
      else
        scp -r "$LOCAL_ROOT/$pkg/." "$HOST:$REMOTE_ROOT/$pkg/"
      fi
    done
    ;;
  *)
    echo "usage: $0 [pull|push]" >&2
    exit 2
    ;;
esac

echo "Done."
