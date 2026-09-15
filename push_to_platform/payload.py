"""Assemble the platform ``ForecastPushRequest`` JSON payload.

The platform requires ALL 10 sites (8 wind + 2 solar). Wind sites carry a
farm-level series plus one series per turbine; solar sites only need ``ghi_wm2``
and may be all-null (this integration does not produce solar, per project
decision). Wind turbine keys must exactly match the manifest.
"""

from __future__ import annotations

import datetime as dt
import uuid

from .config import (
    SCHEMA_VERSION, CATALOG_VERSION, LEAD_TIME_COUNT,
    HORIZON_HOURS, INTERVAL_MINUTES, FORECAST_ISSUANCE_UTC,
)
from .assets_map import Catalog


def _normalize_start(start_date: str) -> str:
    """Accept either ``YYYY-MM-DD`` or a full ISO instant; always emit 16:00Z.

    The platform requires ``start_date`` to end in ``T16:00:00Z``, so a bare date
    is expanded to that fixed issuance time. A full ISO value is passed through
    unchanged (callers already use ``default_start_datetime()``).
    """
    if "T" not in start_date:
        return f"{start_date}T{FORECAST_ISSUANCE_UTC:02d}:00:00Z"
    return start_date


def build_payload(
    provider: str,
    model_version: str,
    start_date: str,          # "YYYY-MM-DDT16:00:00Z" (or bare "YYYY-MM-DD")
    wind_sites: dict[str, dict],
    catalog: Catalog,
    submission_type: str = "realtime",
    request_id_suffix: str | None = None,
) -> dict:
    """Return the full JSON body for ``POST /api/v1/forecast/push``."""
    start_iso = _normalize_start(start_date)
    date_part = start_iso[:10]
    request_id = request_id_suffix or f"{provider}_{date_part.replace('-', '')}_16z"

    sites: dict = {}
    for site_id in sorted(wind_sites):
        sites[site_id] = wind_sites[site_id]
    # Solar sites: ghi all-null (not produced); still required by the schema.
    for solar in catalog.solar_sites:
        sites[solar["site_id"]] = {"ghi_wm2": [None] * LEAD_TIME_COUNT}

    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_version": CATALOG_VERSION,
        "request_id": request_id,
        "provider": provider,
        "model_version": model_version,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "start_date": start_iso,
        "interval_minutes": INTERVAL_MINUTES,
        "horizon_hours": HORIZON_HOURS,
        "submission_type": submission_type,
        "sites": sites,
    }
