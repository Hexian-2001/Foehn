"""CLI: extract a model prediction into platform arrays and push (or dry-run).

Examples:
    python -m push_to_platform.cli --model graphcast --date 2026-08-27 --dry-run
    python -m push_to_platform.cli --model aurora   --date 2026-09-03
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import MODELS
from .assets_map import load_catalog
from .extract import find_prediction_nc, find_analysis_nc, build_wind_sites
from .payload import build_payload
from .push import push_payload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("push_to_platform.cli")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Push open-source model forecast to benchmark platform")
    ap.add_argument("--model", required=True, choices=sorted(MODELS), help="model key")
    ap.add_argument("--date", required=True, help="init date YYYY-MM-DD (00Z)")
    ap.add_argument("--submission-type", default="realtime", choices=["realtime", "backfill"])
    ap.add_argument("--request-id-suffix", default=None, help="override request_id")
    ap.add_argument("--out", default=None, help="write payload JSON here instead of pushing")
    ap.add_argument("--dry-run", action="store_true", help="build payload but do not push")
    args = ap.parse_args(argv)

    cfg = MODELS[args.model]
    provider = cfg["provider"]
    variant = cfg["variant"]

    pred_path = find_prediction_nc(args.model, variant, args.date)
    analysis_path = find_analysis_nc(args.date)
    logger.info("prediction: %s", pred_path)
    logger.info("analysis:   %s", analysis_path or "(none — zero-order hold for leads 1..5)")

    catalog = load_catalog()
    wind_sites = build_wind_sites(pred_path, analysis_path, catalog)

    n_turbines = sum(len(s["turbines"]) for s in wind_sites.values())
    logger.info("extracted %d wind sites, %d turbines", len(wind_sites), n_turbines)

    payload = build_payload(
        provider=provider,
        model_version=cfg["model_version"],
        start_date=args.date,
        wind_sites=wind_sites,
        catalog=catalog,
        submission_type=args.submission_type,
        request_id_suffix=args.request_id_suffix,
    )

    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("payload written to %s", args.out)
        return 0

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    result = push_payload(payload, provider)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
