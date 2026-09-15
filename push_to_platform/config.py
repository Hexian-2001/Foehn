"""Configuration for pushing open-source model forecasts to the benchmark platform.

Paths point at the sibling forecast_models tree (results/, processed input .nc).
Platform connection + provider API keys come from environment variables so the
same code runs locally (dry-run) and on Pawsey without edits.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

# ── Paths (relative to this package) ──
PACKAGE_DIR = Path(__file__).resolve().parent
FORECAST_MODELS_ROOT = PACKAGE_DIR.parent

# Model predictions: results/<model>/<variant>/<init>T00Z/predictions/*.nc
RESULTS_ROOT = FORECAST_MODELS_ROOT / "results"

# Analysis-field (lead-0) input .nc produced by data_processing stage 2. It is
# the IFS fc0 the model actually consumed, stored at 0.25 deg global grid with
# lead 0 at time=0 — used as the interpolation anchor for leads 1..5.
PROCESSED_IFS_ROOT = FORECAST_MODELS_ROOT / "weathernext_forecast" / "data" / "processed"

# Raw IFS GRIB (fallback / cross-check; requires cfgrib, only present on Pawsey).
RAW_IFS_ROOT = FORECAST_MODELS_ROOT / "data" / "raw" / "ifs"

# Platform asset manifest snapshot (must match the platform's catalog_version).
ASSET_CATALOG_PATH = PACKAGE_DIR / "asset_catalog.json"

# ── Platform endpoint ──
PLATFORM_URL = os.getenv("BENCHMARK_PUSH_URL", "http://localhost:8000").rstrip("/")
PUSH_ENDPOINT = "/api/v1/forecast/push"

# ── Protocol constants (single source of truth; must match the platform) ──
SCHEMA_VERSION = "1.0.0"
CATALOG_VERSION = "1.0.0"
LEAD_TIME_COUNT = 72
HORIZON_HOURS = 72
INTERVAL_MINUTES = 60

# Fixed forecast issuance instant: 16:00 UTC (= next-day Beijing 00:00 = the
# platform's "London 16:00"). Every realtime submission has ``start_date`` set to
# this time; the push deadline is 16 h earlier (00:00 UTC, = Beijing 08:00).
FORECAST_ISSUANCE_UTC = 16


def default_start_datetime(now_utc: dt.datetime | None = None) -> str:
    """Return the start_date for a submission made *now*.

    The forecast is issued at ``FORECAST_ISSUANCE_UTC`` (16:00 UTC = the next
    Beijing midnight). The correct target is the *next* such instant strictly
    after ``now``. ``now + 1 day`` was only right when ``now`` fell in the
    16:00–24:00 UTC window (the scheduled pre-deadline push); any run earlier in
    the day (e.g. a same-day backfill/replay after 00:00 UTC) was off by one day.
    """
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    start = now.replace(
        hour=FORECAST_ISSUANCE_UTC, minute=0, second=0, microsecond=0
    )
    if start <= now:
        start += dt.timedelta(days=1)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ")

# ── Model registry ──
# key  -> dict(provider, variant, model_version). `provider` must equal the
# platform-side API-key owner; `variant` selects the results subdirectory.
MODELS: dict[str, dict[str, str]] = {
    "graphcast": {
        "provider": "graphcast",
        "variant": "operational",
        "model_version": "graphcast-operational-0.25",
    },
    "aurora": {
        "provider": "aurora",
        "variant": "0.25-finetuned",
        "model_version": "aurora-0.25-finetuned",
    },
}


def _parse_keys(raw: str | None) -> dict[str, str]:
    """Parse ``provider=key,provider=key`` into ``{provider: key}``."""
    out: dict[str, str] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, _sep, key = part.partition("=")
        name, key = name.strip(), key.strip()
        if name and key:
            out[name] = key
    return out


# Provider API keys, e.g. BENCHMARK_PUSH_KEYS="graphcast=sk-...,aurora=sk-...".
PROVIDER_KEYS: dict[str, str] = _parse_keys(os.getenv("BENCHMARK_PUSH_KEYS"))
