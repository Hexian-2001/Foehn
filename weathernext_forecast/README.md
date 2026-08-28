# weathernext_forecast

GraphCast (WeatherNext 1 Graph) inference workspace, built on top of the
[`google-deepmind/weathernext`](https://github.com/google-deepmind/weathernext)
repository. This is one project under `D:\mingyang_tech_work\forecast_models\`;
sibling model projects (FourCastNet, Pangu-Weather, …) follow the same layout.

## Three-way separation

| layer | location | managed by |
|---|---|---|
| **upstream source** (the fork) | `upstream/weathernext/` | its own git (origin=your fork, upstream=deepmind) — read-only dependency |
| **your code** (the product) | `src/`, `scripts/`, `experiments/` | this repo (git) |
| **artifacts** (weights/data/output) | `models/`, `data/`, `predictions/` | nothing — git-ignored |

## Directory layout

```
weathernext_forecast/
├── README.md                  # this file
├── .gitignore                 # ignores upstream/ data/ models/ predictions/
├── requirements.txt           # pip install -r requirements.txt
├── src/weathernext_forecast/  # your code package (version-controlled)
│   ├── __init__.py
│   ├── config.py              # all paths + model/data selection (edit here)
│   └── inference.py           # the GraphCast inference pipeline
├── scripts/
│   └── run_inference.py       # thin CLI entry point
├── experiments/               # exploratory notebooks / one-off scripts
│   └── graphcast_demo.ipynb   # the demo notebook (with your PRESSURE_LEVELS fix)
├── upstream/weathernext/      # the fork (git-ignored, separate repo)
├── data/
│   ├── raw/                   # files exactly as downloaded (GRIB / original .nc)
│   └── processed/             # standard-format .nc that the script reads
├── models/
│   ├── weights/               # checkpoints (.npz)
│   └── stats/                 # normalization statistics (3 .nc files)
└── predictions/               # model output, one .nc per run
```

All paths in `config.py` are derived from that file's location, so the whole
directory is **portable** — copy it anywhere and it still runs.

## What is already here

| file | size | source (GCS `dm_graphcast`) |
|---|---|---|
| `models/weights/GraphCast_small.npz` | 144 MB | `params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz` |
| `models/stats/{mean,stddev,diffs_stddev}_by_level.nc` | ~6 KB each | `stats/*.nc` |
| `data/processed/source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc` | 131 MB | `dataset/…` (era5, 1.0°, 13 levels, 6 timesteps) |

The checkpoints in the bucket have **long descriptive names**; they were
renamed to clean local names on download. `GraphCast_small` (1.0°, 13 levels)
is the only one that runs on CPU; the 0.25° models need a TPU/GPU.

## Setup (one time)

```bash
# from this directory — editable-installs the fork and all its deps
pip install -r requirements.txt
```

On Windows, JAX is CPU-only. `GraphCast_small` runs on CPU (slow); the
0.25° models need a TPU/GPU.

## Run

```bash
python scripts/run_inference.py
```

Predictions land in `predictions/predictions_<ref-time>_<model>.nc` as an
xarray Dataset in real physical units.

## Where the data comes from

### 1. Demo / sample data (bundled in the GCS bucket)

The `dm_graphcast` GCS bucket ships everything needed to *try* the model:

| path in bucket | what it is |
|---|---|
| `graphcast/params/*.npz` | pretrained weights (3 models, long filenames) |
| `graphcast/stats/*.nc` | the 3 normalization statistics files |
| `graphcast/dataset/*.nc` | sample input datasets (fake/era5/hres, various resolutions/levels/steps) |

These are **static demo files, not a live feed** — good for validating setup,
but they do not give you *today's* weather. Download with `curl` (anonymous):

```bash
curl -o models/stats/mean_by_level.nc \
  https://storage.googleapis.com/dm_graphcast/graphcast/stats/mean_by_level.nc
# ... likewise for stddev_by_level.nc, diffs_stddev_by_level.nc, and the
# checkpoint under params/ (rename to a clean local name).
```

The smallest dataset files that still match `GraphCast_small` (1.0°, 13 levels)
are the `era5` 1.0° files (`steps-04` = 131 MB; it holds 6 timesteps = 2 input
+ 4 target = a 24h forecast). The even smaller `fake` 6.0° files are synthetic
and do not match any pretrained model's resolution.

### 2. Real-time operational data (for actual forecasts)

Operational providers deliver **GRIB/GRIB2**, which is **not** the format
GraphCast reads — a conversion step is required.

- **ECMWF open data** — real-time IFS/HRES and AIFS, 0.25°, GRIB2, CC-BY-4.0.
  → <https://data.ecmwf.int/forecasts/>
- **NOAA GFS** — real-time, 0.25°, GRIB2, free on AWS open data.
  → `s3://noaa-gfs-bdp-pds/gfs.YYYYMMDD/CC/atmos/...`
- **ERA5 reanalysis** — matches the `GraphCast` model, but historical with a
  ~5-day delay (Copernicus CDS). → <https://cds.climate.copernicus.eu/>

> **Gap:** this repo does not ship a "live GRIB → ready input" pipeline.
> `data_utils.extract_inputs_targets_forcings` only does the final slicing +
> derived-variable computation. GRIB → standard `.nc` needs a separate ingestion
> step (read GRIB with `cfgrib`/`xarray`, rename to ECMWF naming, regrid/select,
> convert units, accumulate precip to 6h, add a `batch` dim).

## The standard input format (`data/processed/*.nc`)

```python
# dims:
#   (batch, time, lat, lon)          -> surface & static variables
#   (batch, time, lat, lon, level)   -> pressure-level variables
# coords: time (timedelta, 6h), datetime (absolute), lat, lon, level (hPa)

# required variables (exact names):
#   surface : 2m_temperature, mean_sea_level_pressure,
#             10m_u_component_of_wind, 10m_v_component_of_wind,
#             total_precipitation_6hr
#   pressure: temperature, geopotential, u_component_of_wind,
#             v_component_of_wind, vertical_velocity, specific_humidity
#   static  : geopotential_at_surface, land_sea_mask
# (TISR + year/day progress are auto-computed if missing)
```

## Packing it up (e.g. when leaving)

The code is version-controlled and the artifacts are git-ignored, so to take
everything with you:

```bash
# 1. bundle the versioned code (src/scripts/experiments/docs + history)
git bundle create weathernext_forecast.gitbundle --all

# 2. copy the whole directory (includes the fork + artifacts + bundle)
#    — copy D:\mingyang_tech_work\forecast_models\weathernext_forecast\
```

The `.gitbundle` alone reconstructs the code + history anywhere; the directory
copy also carries the fork and the large artifacts.
