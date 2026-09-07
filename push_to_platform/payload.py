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
    HORIZON_HOURS, INTERVAL_MINUTES,
)
from .assets_map import Catalog


def build_payload(
    provider: str,
    model_version: str,
    start_date: str,          # "YYYY-MM-DD"
    wind_sites: dict[str, dict],
    catalog: Catalog,
    submission_type: str = "realtime",
    request_id_suffix: str | None = None,
) -> dict:
    """Return the full JSON body for ``POST /api/v1/forecast/push``."""
    start_iso = f"{start_date}T00:00:00Z"
    request_id = request_id_suffix or f"{provider}_{start_date.replace('-', '')}_00z"

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
