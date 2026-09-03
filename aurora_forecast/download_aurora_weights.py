#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download ALL Microsoft Aurora weights (~41.3 GB total) with fast transfer + resume.

Pulls the complete file set of `microsoft/aurora` (public, MIT) into a flat local
directory, ready for `model.load_checkpoint_local(<dir>/<file>.ckpt)`.

Speed-ups
  - Xet high-performance transfer (built into recent huggingface_hub) is enabled on the
    direct route. Newer versions no longer use `hf_transfer`.
  - Downloads are parallel across files (`--threads`).
  - Interrupted downloads resume automatically (no restart from zero).
  - Files already present at the target (nonzero size) are skipped.

One-time setup:
    pip install -U huggingface_hub

Usage:
    python download_aurora_weights.py                       # -> ./model_weights/model
    python download_aurora_weights.py --target D:/x/y       # custom flat dir
    python download_aurora_weights.py --threads 8           # more parallel files
    python download_aurora_weights.py --mirror              # route via hf-mirror.com (CN)
    python download_aurora_weights.py --force               # re-download existing files

Network note: if huggingface.co is unreachable from your network (typical in mainland
China), run with `--mirror` to route through hf-mirror.com. The 10 files that fail with
"cannot find the requested files ... check your connection" will succeed via the mirror.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Fast-transfer is decided in main(): Xet on the direct route, off when using the mirror.
REPO_ID = "microsoft/aurora"

# Complete artifact list of microsoft/aurora (verified against the HF API, 2026-09-01).
# `.gitattributes` and `README.md` are omitted (not model artifacts).
EXPECTED_FILES = [
    "aurora-0.1-finetuned.ckpt",
    "aurora-0.1-static.nc",
    "aurora-0.1-static.pickle",
    "aurora-0.25-12h-pretrained.ckpt",
    "aurora-0.25-finetuned.ckpt",
    "aurora-0.25-pretrained.ckpt",
    "aurora-0.25-small-pretrained-test-input.pickle",
    "aurora-0.25-small-pretrained-test-output.pickle",
    "aurora-0.25-small-pretrained.ckpt",
    "aurora-0.25-static.pickle",
    "aurora-0.25-v1.5-ensemble.ckpt",
    "aurora-0.25-v1.5-static.pickle",
    "aurora-0.25-v1.5.ckpt",
    "aurora-0.25-wave-static.nc",
    "aurora-0.25-wave-static.pickle",
    "aurora-0.25-wave.ckpt",
    "aurora-0.4-air-pollution-static.nc",
    "aurora-0.4-air-pollution-static.pickle",
    "aurora-0.4-air-pollution.ckpt",
]

# Non-weather branches (wave / air-pollution / ensemble) — excluded by `--only-weather`.
NON_WEATHER_PREFIXES = (
    "aurora-0.25-wave",
    "aurora-0.4-air-pollution",
    "aurora-0.25-v1.5-ensemble",
)


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download_one(name: str, target: Path, force: bool) -> tuple[str, str, str | None]:
    from huggingface_hub import hf_hub_download

    dest = target / name
    if not force and dest.exists() and dest.stat().st_size > 0:
        return name, "skip", None
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=name,
            local_dir=target,
        )
        return name, "ok", None
    except Exception as e:  # noqa: BLE001 - isolate per-file errors, keep going
        return name, "fail", str(e)


def main() -> int:
    default_target = Path(__file__).resolve().parent / "model_weights" / "model"
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--target", default=str(default_target),
                    help="flat download dir (default: <script dir>/model_weights/model)")
    ap.add_argument("--threads", type=int, default=4, help="parallel files (default 4)")
    ap.add_argument("--mirror", action="store_true", help="route via hf-mirror.com (China)")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--only-weather", action="store_true",
                    help="download only the weather branch (skip wave / air-pollution / ensemble)")
    args = ap.parse_args()

    files = EXPECTED_FILES
    if args.only_weather:
        files = [f for f in EXPECTED_FILES if not f.startswith(NON_WEATHER_PREFIXES)]

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        # Xet talks to HF's own storage endpoints, which the mirror does not proxy.
        # Use the standard HTTP path through the mirror instead.
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
        print("[mirror] HF_ENDPOINT=https://hf-mirror.com (Xet off)")
    else:
        # Fast transfer on the direct route (new huggingface_hub uses Xet, not hf_transfer).
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("       Run: pip install -U huggingface_hub")
        return 1

    fast = os.environ.get("HF_XET_HIGH_PERFORMANCE") == "1"
    print(f"[fast]   Xet high-performance transfer: {'ON' if fast else 'off (mirror)'}")

    print(f"[repo]   {REPO_ID}")
    print(f"[target] {target}")
    print(f"[files]  {len(files)} files, {args.threads} parallel")
    print("-" * 62)

    results: dict[str, list[str]] = {"ok": [], "skip": [], "fail": []}
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(download_one, n, target, args.force): n for n in files}
        for fut in as_completed(futs):
            name, status, err = fut.result()
            results[status].append(name)
            mark = {"ok": "done ", "skip": "skip ", "fail": "FAIL "}[status]
            suffix = f"  ({err})" if err else ""
            print(f"[{len(results['ok']) + len(results['skip']) + len(results['fail'])}/"
                  f"{len(files)}] {mark}{name}{suffix}", flush=True)

    print("-" * 62)
    print(f"downloaded {len(results['ok'])}, skipped {len(results['skip'])}, "
          f"failed {len(results['fail'])}")

    # Verify every expected file landed with nonzero size.
    missing, total = [], 0
    for name in files:
        p = target / name
        if p.exists() and p.stat().st_size > 0:
            total += p.stat().st_size
        else:
            missing.append(name)

    print(f"[verify] {human(total)} on disk across present files")
    if results["fail"]:
        print("Failed (re-run to retry with resume):")
        for n in results["fail"]:
            print(f"  - {n}")
        return 2
    if missing:
        print(f"[verify] MISSING {len(missing)}: {missing}")
        return 2
    print("[verify] all expected files present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
