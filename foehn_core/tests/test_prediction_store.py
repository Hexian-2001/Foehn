from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foehn_core import prediction_store


class PredictionStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = xr.Dataset(
            {"2m_temperature": (("batch", "time", "lat", "lon"), np.ones((1, 2, 3, 3)))},
            coords={
                "batch": [0],
                "time": np.array([6, 12], dtype="timedelta64[h]"),
                "lat": [55.0, 35.0, 15.0],
                "lon": [70.0, 105.0, 140.0],
            },
        )

    def test_unified_contract_and_atomic_write(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = prediction_store.save_unified(
                self.ds,
                model="test-model",
                variant="test-variant",
                init=np.datetime64("2026-09-04T00"),
                out_root=root,
            )
            self.assertTrue(path.exists())
            self.assertEqual(list(path.parent.glob("*.tmp.nc")), [])
            with xr.open_dataset(path) as saved:
                self.assertNotIn("batch", saved.dims)
                self.assertEqual(saved.attrs["convention"], "unified-forecast-1")
                self.assertEqual(saved.attrs["step_h"], 6)
                self.assertEqual(saved.attrs["horizon_h"], 12)
                self.assertEqual(saved.time.values[0], np.datetime64("2026-09-04T06"))


if __name__ == "__main__":
    unittest.main()
