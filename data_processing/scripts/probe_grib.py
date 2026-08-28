"""Probe raw ECMWF open-data GRIB files.

Dumps per-message metadata (shortName, units, level, grid extent) so we can
confirm — *before* writing the ingest step — that:

  1. ``z``/``w``/``q`` carry the expected SI units (m2/s2, Pa/s, kg/kg),
     not gpm or anything else.
  2. the decoded lat/lon grid orientation matches ERA5 (lat 90 -> -90,
     lon 0 -> 360).
  3. all 13 pressure levels are present for the 6 pressure variables.

Usage:
    python scripts/probe_grib.py <path-to-grib> [more paths...]
"""

from __future__ import annotations

import sys

import eccodes
    

def _get(gid, key: str) -> str:
    try:
        return str(eccodes.codes_get(gid, key))
    except Exception:  # noqa: BLE001 - key may be absent for this levtype
        return "-"


def probe(path: str) -> None:
    print(f"\n=== {path} ===")
    keys = (
        "shortName", "paramId", "typeOfLevel", "level", "units",
        "gridType", "Ni", "Nj",
        "latitudeOfFirstGridPointInDegrees", "latitudeOfLastGridPointInDegrees",
        "longitudeOfFirstGridPointInDegrees", "longitudeOfLastGridPointInDegrees",
        "iDirectionIncrementInDegrees", "jDirectionIncrementInDegrees",
        "dataDate", "dataTime",
    )
    i = 0
    with open(path, "rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            i += 1
            vals = "  ".join(f"{k}={_get(gid, k)}" for k in keys)
            print(f"  [{i:02d}] {vals}")
            eccodes.codes_release(gid)
    print(f"  total messages: {i}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python probe_grib.py <grib path> [...]")
    for p in sys.argv[1:]:
        probe(p)
