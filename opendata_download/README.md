# opendata_download

Download ECMWF **open-data** IFS analysis fields (fc0), reusable across forecast
models. This is **stage 1 (数据下载)** of a decoupled pipeline:

```
opendata_download (this)  →  data_processing (stage 2, later)  →  weathernext_forecast (stage 3, inference)
```

It is deliberately **model-agnostic**: it fetches raw ECMWF GRIB fields by short
name (`2t`, `z`, `q`, …) and does *no* renaming / regridding / unit conversion.
That mapping belongs to the processing stage, so GraphCast, FourCastNet,
Pangu-Weather, … can all consume the same downloader unchanged.

## Install (once)

```bash
pip install -e .
```

Requires `ecmwf-opendata` (installed automatically). No ECMWF account is needed
— open data is served anonymously under CC-BY-4.0.

## Usage

```bash
# A forecast initialized at 2026-08-27 00 UTC — downloads the two input cycles
# (T-6h = 2026-08-26 18Z and T = 2026-08-27 00Z) plus the static fields:
python scripts/download_fc0.py --date 2026-08-27 --time 00

# Latest available analysis cycle (for the unattended 24/7 auto-run):
python scripts/download_fc0.py --latest
```

Or, if installed, `download-fc0 --date 2026-08-27 --time 00`.

`--time` must be one of `00/06/12/18` (the four IFS analysis cycles per day).

## What it downloads

Per init time `T`, it fetches **step 0 (fc0 = the analysis)** — the initial
condition, *not* a future forecast — for two cycles:

| file | levtype | params | purpose |
|---|---|---|---|
| `sfc_fc0.grib2` | surface | `2t, msl, 10u, 10v` | dynamic surface fields |
| `pl_fc0.grib2` | pressure | `z, t, u, v, w, q` @ 13 levels | dynamic pressure-level fields |
| `static.grib2` | surface | `z, lsm` | geopotential_at_surface + land-sea mask (once) |

The 13 pressure levels are the "WeatherBench 13" set (50 … 1000 hPa), which
matches `GraphCast_operational`'s levels exactly.

## Output layout (the contract for stage 2)

```
<data_root>/
└── raw/ifs/
    ├── 2026-08-26/18/{sfc_fc0.grib2, pl_fc0.grib2, manifest.json}
    ├── 2026-08-27/00/{sfc_fc0.grib2, pl_fc0.grib2, manifest.json}
    └── static/static.grib2
```

`manifest.json` records exactly what was fetched and when, for auditability.
Downloads are **atomic** (written to `.part`, renamed on success), so a file at
its final path is always complete — this makes re-running idempotent (existing
files are skipped unless `--force`).

`<data_root>` defaults to a **shared, external** store — `<forecast_models>/data`
(a sibling of the code folders, so raw fields are not tied to any single model
project). Model-specific `processed` files stay inside each model's own
`data/processed/`. Override via `OPENDATA_DATA_ROOT` or `--data-root` (e.g.
point it at a bigger/faster disk on the server).

## Programmatic API (for the scheduler)

```python
from opendata_download.client import OpenDataClient
from opendata_download import downloader

client = OpenDataClient()
paths = downloader.download_init(client, date, hour, data_root)  # idempotent
```

`client.latest(request)` returns the newest fully-available cycle (a
`datetime`), or raises `ValueError` while nothing is ready — poll that to detect
"new data published".

## Notes / verify on first run

- Open data is a **fixed 13-level set**; on first download, confirm all six
  pressure params (`z,t,u,v,w,q`) actually arrive at all 13 levels (a historical
  issue served only 9 levels for `u,v,r,t`).
- The static params (`z` at surface, `lsm`) are fetched at `type="fc", step=0`;
  if the bucket serves them differently, adjust `config.STATIC_PARAMS`/the
  request type accordingly.
- Analysis availability lags the nominal cycle time by a few hours (full
  dissemination ~7–9h); the scheduler must account for that delay.
- Transient connection breaks (`ChunkedEncodingError` / timeouts) are retried
  automatically with exponential backoff (6 attempts, starting at 15s). If the
  link to `data.ecmwf.int` is still unstable, switch to a cloud mirror with
  `--source aws` (also `azure` / `google`).
