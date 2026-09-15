#!/usr/bin/env bash
# =============================================================================
# Sync between local machine and Pawsey Setonix.
#
#     ./sync.sh          # pull the results tree down (default)
#     ./sync.sh pull     # same
#     ./sync.sh push     # push source code/docs to Pawsey
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

# Pawsey explicitly recommends the data-mover nodes for transfers involving
# /scratch. Override with PAWSEY_TRANSFER_HOST if a site-specific route is
# required; PAWSEY_HOST is retained as a backwards-compatible fallback.
HOST="${PAWSEY_TRANSFER_HOST:-${PAWSEY_HOST:-hwang4@data-mover.pawsey.org.au}}"
REMOTE_ROOT="/scratch/pawsey0115/hwang4/projects/Foehn"
LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"   # forecast_models/

MODE="${1:-pull}"
MAX_SIZE="1G"   # never transfer anything at or above this (predictions are ~13 GB full-global)
MAX_BYTES=1073741824

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
      # No --delete: preserve local-only runs while merging server results.
      rsync -av --max-size="$MAX_SIZE" "$HOST:$REMOTE/" "$LOCAL/"
    else
      # scp cannot filter by size, so stream a tar of only the sub-1GB files.
      # Use bytes rather than `-size -1G`: GNU find rounds size units up, so
      # `-1G` accidentally excludes every non-empty file smaller than 1 GiB.
      ssh "$HOST" "cd '$REMOTE_ROOT' && find results -type f -size -${MAX_BYTES}c -print0 | tar --null -T - -cf -" \
        | tar -xf - -C "$LOCAL_ROOT"
    fi
    ;;
  push)
    # Only source, never data / models / predictions.
    # `aurora_forecast/upstream/aurora/aurora` is the importable `aurora` package
    # (pure Python, read-only dependency) — pushed as code so `import aurora`
    # resolves on Pawsey without a separate git clone. Model weights (>=1 GB) are
    # NOT pushed; fetch them on Pawsey with aurora_forecast/scripts/download_weights.sh.
    ssh "$HOST" "mkdir -p \
      '$REMOTE_ROOT/scripts' \
      '$REMOTE_ROOT/foehn_core/src' \
      '$REMOTE_ROOT/foehn_core/tests'"
    for pkg in \
        scripts \
        foehn_core/src \
        foehn_core/tests \
        weathernext_forecast/scripts \
        weathernext_forecast/src \
        opendata_download/scripts \
        opendata_download/src \
        data_processing/scripts \
        data_processing/src \
        aurora_forecast/scripts \
        aurora_forecast/src \
        aurora_forecast/upstream/aurora/aurora \
        push_to_platform; do
      echo "push: $LOCAL_ROOT/$pkg/  ->  $HOST:$REMOTE_ROOT/$pkg/"
      if [ "$RSYNC" = 1 ]; then
        # .env holds live API keys (never synced); __pycache__ is transient.
        rsync -av --exclude='.env' --exclude='__pycache__' \
          "$LOCAL_ROOT/$pkg/" "$HOST:$REMOTE_ROOT/$pkg/"
      else
        scp -r "$LOCAL_ROOT/$pkg/." "$HOST:$REMOTE_ROOT/$pkg/"
      fi
    done
    # Project entry points, packaging metadata and operational documentation.
    # These are intentionally explicit so weights/data/checkpoints can never be
    # swept into an upload by a broad repository-level copy.
    for file in \
        realtime_all.sh \
        sync.sh \
        README.md \
        OPERATION.md \
        .gitignore \
        foehn_core/pyproject.toml \
        weathernext_forecast/pyproject.toml \
        weathernext_forecast/README.md \
        weathernext_forecast/OPERATION.md \
        opendata_download/pyproject.toml \
        opendata_download/README.md \
        data_processing/pyproject.toml \
        data_processing/README.md \
        aurora_forecast/pyproject.toml \
        aurora_forecast/OPERATION.md \
        aurora_forecast/download_aurora_weights.py \
        aurora_forecast/probe_grib.py \
        aurora_forecast/test_adapter.py; do
      echo "push: $LOCAL_ROOT/$file  ->  $HOST:$REMOTE_ROOT/$file"
      if [ "$RSYNC" = 1 ]; then
        rsync -av "$LOCAL_ROOT/$file" "$HOST:$REMOTE_ROOT/$file"
      else
        scp "$LOCAL_ROOT/$file" "$HOST:$REMOTE_ROOT/$file"
      fi
    done
    ;;
  *)
    echo "usage: $0 [pull|push]" >&2
    exit 2
    ;;
esac

echo "Done."
