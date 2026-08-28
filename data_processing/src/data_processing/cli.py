"""CLI for converting open-data GRIB -> GraphCast input .nc."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

from data_processing import config, ingest


def _setup_logging(level: int) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Convert ECMWF open-data fc0 GRIB into a GraphCast input .nc."
    )
    p.add_argument("--date", help="init date, YYYY-MM-DD (UTC)")
    p.add_argument("--time", help="init hour, one of 00/06/12/18 (UTC)")
    p.add_argument(
        "--raw-root",
        type=Path,
        default=config.RAW_IFS_ROOT,
        help="root of raw/ifs (default: env OPENDATA_DATA_ROOT or shared store)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=config.PROCESSED_ROOT,
        help="output dir for the processed .nc",
    )
    p.add_argument("--force", action="store_true", help="re-build existing file")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = p.parse_args(argv)

    _setup_logging(getattr(logging, args.log_level))

    if not (args.date and args.time):
        p.error("provide --date and --time")
    date = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
    hour = int(args.time)
    if hour not in (0, 6, 12, 18):
        p.error("--time must be one of 00/06/12/18")

    path = ingest.build_input(
        date, hour, raw_root=args.raw_root, out_dir=args.out_dir, force=args.force
    )
    logging.getLogger(__name__).info("output: %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
