"""Download orchestration: which cycles/fields to fetch and where to store them.

Model-agnostic — downloads ECMWF GRIB fields by short name; no renaming, no
regridding, no model-specific logic.

Raw layout (the contract the downstream "processing" stage consumes):

    <data_root>/raw/ifs/<YYYY-MM-DD>/<HH>/sfc_fc0.grib2   # surface fields
    <data_root>/raw/ifs/<YYYY-MM-DD>/<HH>/pl_fc0.grib2    # pressure-level fields
    <data_root>/raw/ifs/static/static.grib2               # static fields (once)

Each cycle dir also gets a ``manifest.json`` recording exactly what was fetched,
for auditability in the unattended 24/7 pipeline.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

from opendata_download import config
from opendata_download.client import OpenDataClient

logger = logging.getLogger(__name__)


def cycle_dir(data_root: Path, date: dt.date, hour: int) -> Path:
    """Directory holding one cycle's raw files."""
    return data_root / "raw" / "ifs" / date.strftime("%Y-%m-%d") / f"{hour:02d}"


def _request(levtype: str, params, levelist, date: dt.date, hour: int) -> dict:
    req: dict = {
        "date": date.strftime("%Y%m%d"),
        "time": hour,
        "type": "fc",
        "step": config.ANALYSIS_STEP,
        "levtype": levtype,
        "param": list(params),
    }
    if levelist is not None:
        req["levelist"] = list(levelist)
    return req


def download_cycle(
    client: OpenDataClient,
    date: dt.date,
    hour: int,
    data_root: Path,
    force: bool = False,
) -> Path:
    """Download one cycle's fc0 (surface + pressure fields), idempotently."""
    out = cycle_dir(data_root, date, hour)
    out.mkdir(parents=True, exist_ok=True)

    files = (
        ("sfc_fc0.grib2", "sfc", config.SURFACE_PARAMS, None),
        ("pl_fc0.grib2", "pl", config.PRESSURE_PARAMS, config.PRESSURE_LEVELS),
    )
    for name, levtype, params, levelist in files:
        path = out / name
        if force or not path.exists() or path.stat().st_size == 0:
            client.retrieve(
                _request(levtype, params, levelist, date, hour), target=str(path)
            )
        else:
            logger.info("already present, skipping %s", path)

    _write_manifest(out, date, hour)
    return out


def download_static(
    client: OpenDataClient,
    data_root: Path,
    date: dt.date,
    hour: int,
    force: bool = False,
) -> Path:
    """Download the time-invariant static fields (once)."""
    out = data_root / "raw" / "ifs" / "static"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "static.grib2"
    if force or not path.exists() or path.stat().st_size == 0:
        client.retrieve(
            _request("sfc", config.STATIC_PARAMS, None, date, hour),
            target=str(path),
        )
    else:
        logger.info("already present, skipping %s", path)
    return path


def download_init(
    client: OpenDataClient,
    date: dt.date,
    hour: int,
    data_root: Path,
    force: bool = False,
    include_static: bool = True,
) -> dict[str, Path]:
    """Download the two input cycles (T-6h and T) plus static fields.

    One-step autoregressive models consume a 12h input window = two consecutive
    analyses, so a forecast initialized at ``T`` needs the analyses at ``T-6h``
    and ``T``.
    """
    t = dt.datetime.combine(date, dt.time(hour))
    prev = t - dt.timedelta(hours=6)

    paths = {
        "t_minus_6h": download_cycle(client, prev.date(), prev.hour, data_root, force),
        "t": download_cycle(client, date, hour, data_root, force),
    }
    if include_static:
        paths["static"] = download_static(client, data_root, date, hour, force)
    return paths


def _write_manifest(out: Path, date: dt.date, hour: int) -> None:
    manifest = {
        "cycle": f"{date.strftime('%Y-%m-%d')}T{hour:02d}Z",
        "surface_params": list(config.SURFACE_PARAMS),
        "pressure_params": list(config.PRESSURE_PARAMS),
        "pressure_levels": list(config.PRESSURE_LEVELS),
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
