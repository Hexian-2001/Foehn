# Foehn

Foehn is a dual-model operational weather-inference workspace for Pawsey
Setonix. It runs GraphCast (WeatherNext 1 Graph) and Microsoft Aurora from one
ECMWF IFS analysis cycle and writes a common, China-region NetCDF contract plus
publication-ready visualizations.

Start with [OPERATION.md](OPERATION.md) for submission, monitoring, output and
synchronization procedures. Model-specific implementation notes live in
`weathernext_forecast/OPERATION.md` and `aurora_forecast/OPERATION.md`.

Runtime data, checkpoints, logs and generated results are intentionally outside
version control.
