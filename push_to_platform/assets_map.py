"""Load the platform asset manifest and expose site/turbine coordinates.

The manifest is a snapshot copy of the benchmark platform's
``assets/asset_catalog.json`` (catalog_version 1.0.0). It maps every wind
turbine (and wind-site centroid) to its lat/lon so the extractor can pull the
nearest 0.25-degree grid point from a model prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import ASSET_CATALOG_PATH


class Catalog:
    """In-memory view of the platform asset manifest."""

    def __init__(self, path: Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.wind_sites: list[dict] = data["wind_sites"]          # 8 sites
        self.solar_sites: list[dict] = data["solar_sites"]        # 2 sites
        assets: list[dict] = data["assets"]

        self.turbines: list[dict] = [
            a for a in assets if a["asset_type"] == "wind_turbine"
        ]
        self.inverters: list[dict] = [
            a for a in assets if a["asset_type"] == "solar_inverter"
        ]

        # site_id -> ordered list of turbine asset_ids (matches platform template).
        self.turbines_by_site: dict[str, list[dict]] = {}
        for t in self.turbines:
            self.turbines_by_site.setdefault(t["site_id"], []).append(t)

    def site_centroid(self, site_id: str) -> tuple[float, float]:
        """(lat, lon) of a wind site's centroid."""
        for s in self.wind_sites:
            if s["site_id"] == site_id:
                return s["lat"], s["lon"]
        raise KeyError(f"no wind site {site_id}")


def load_catalog() -> Catalog:
    return Catalog(ASSET_CATALOG_PATH)
