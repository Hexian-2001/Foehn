# Download ECMWF open-data IFS analysis fields (fc0).
#
# Usage:
#   .\run_download.ps1                                       # latest available cycle
#   .\run_download.ps1 -Date 2026-08-27 -Time 00             # a specific init time
#   .\run_download.ps1 -Source aws                           # use the AWS mirror
#
# One-time setup (run once first, in your conda env):
#   pip install -e D:\mingyang_tech_work\forecast_models\opendata_download

param(
    [string]$Date = "",
    [string]$Time = "",
    [string]$Source = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$extra = @()
if ($Source) { $extra += "--source"; $extra += $Source }

if ($Date -and $Time) {
    python scripts/download_fc0.py --date $Date --time $Time @extra
} else {
    python scripts/download_fc0.py --latest @extra
}
