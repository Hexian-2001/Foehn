"""Convert open-data fc0 GRIB -> GraphCast input .nc (stage 2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_processing.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
