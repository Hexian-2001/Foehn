#!/usr/bin/env bash
# =============================================================================
# Download the Aurora weights actually needed for classic 0.25 finetuned
# inference — NOT the full ~20 GB repo. Run ONCE on Setonix (login node) in an
# env that has huggingface_hub (the aurora-gpu env does).
#
#   bash scripts/download_weights.sh          # -> <repo>/model_weights/model
#
# Files (both public, MIT, microsoft/aurora):
#   aurora-0.25-finetuned.ckpt   ~4.7 GB   IFS HRES T0, 0.25 deg (the model)
#   aurora-0.25-static.pickle     ~12 MB   {z, slt, lsm} static fields
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$(cd "$HERE/.." && pwd)/model_weights/model}"
mkdir -p "$TARGET"

echo "[weights] microsoft/aurora -> $TARGET"

python - "$TARGET" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

target = Path(sys.argv[1])
for name in ("aurora-0.25-finetuned.ckpt", "aurora-0.25-static.pickle"):
    dest = target / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip {name} (present)")
        continue
    print(f"  fetch {name} ...")
    hf_hub_download(repo_id="microsoft/aurora", filename=name, local_dir=target)
    print(f"  ok   {name}")

print("[weights] done")
PY
