"""ECMWF open-data downloader.

Model-agnostic: fetches raw IFS analysis fields (fc0) by ECMWF GRIB short name.
The mapping to a specific model's variables belongs to a separate "processing"
stage, so this package can be reused by any forecast-model project.
"""

from opendata_download import cli, client, config, downloader

__all__ = ["cli", "client", "config", "downloader"]
