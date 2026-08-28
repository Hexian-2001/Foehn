"""Command-line interface for the ECMWF open-data downloader."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

from opendata_download import config, downloader
from opendata_download.client import OpenDataClient


def _setup_logging(level: int) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_cycle(date_str: str, time_str: str) -> tuple[dt.date, int]:
    date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    hour = int(time_str)
    if hour not in config.CYCLE_HOURS:
        raise ValueError(f"hour must be one of {config.CYCLE_HOURS}, got {hour}")
    return date, hour
    

def _resolve_latest(client: OpenDataClient) -> tuple[dt.date, int]:
    """Return (date, hour) of the newest fully-available analysis cycle."""
    latest = client.latest(
        {
            "type": "fc",
            "step": config.ANALYSIS_STEP,
            "levtype": "sfc",
            "param": config.SURFACE_PARAMS[0],
        }
    )
    if isinstance(latest, dt.datetime):
        return latest.date(), latest.hour
    raise RuntimeError(f"unexpected latest() result: {latest!r}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Download ECMWF open-data IFS analysis fields (fc0)."
    )
    p.add_argument("--date", help="init date, YYYY-MM-DD (UTC)")
    p.add_argument("--time", help="init hour, one of 00/06/12/18 (UTC)")
    p.add_argument(
        "--latest",
        action="store_true",
        help="download the most recent available analysis cycle",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=config.DATA_ROOT,
        help="root dir for downloaded data (default: env OPENDATA_DATA_ROOT or <package>/data)",
    )
    p.add_argument("--force", action="store_true", help="re-download existing files")
    p.add_argument("--no-static", action="store_true", help="skip static fields")
    p.add_argument(
        "--source",
        default=config.SOURCE,
        help="data source: ecmwf (default) or a cloud mirror (aws/azure/google)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = p.parse_args(argv)
        
    _setup_logging(getattr(logging, args.log_level))
    client = OpenDataClient(source=args.source)

    if args.latest and (args.date or args.time):
        p.error("--latest cannot be combined with --date/--time")
    if args.latest:
        date, hour = _resolve_latest(client)
        logging.getLogger(__name__).info("latest available cycle: %s %02dZ", date, hour)
    else:
        if not (args.date and args.time):
            p.error("provide --date and --time, or use --latest")
        date, hour = _parse_cycle(args.date, args.time)

    paths = downloader.download_init(
        client,
        date=date,
        hour=hour,
        data_root=args.data_root,
        force=args.force,
        include_static=not args.no_static,
    )

    for label, path in paths.items():
        logging.getLogger(__name__).info("%-10s -> %s", label, path)
    return 0
