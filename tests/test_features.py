from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p1_qc.config import FeatureConfig
from p1_qc.features import build_features


def series_frame(values: list[float], *, gap_at: int | None = None) -> pd.DataFrame:
    timestamps = []
    current = pd.Timestamp("2025-05-01T00:00:00+09:00")
    for index in range(len(values)):
        if index:
            current += pd.Timedelta(minutes=20 if gap_at == index else 10)
        timestamps.append(current.isoformat())
    return pd.DataFrame(
        {
            "station": ["S-ORS"] * len(values),
            "year": [2025] * len(values),
            "layer": [1] * len(values),
            "time": timestamps,
            "temp": values,
            "psal": [33.0] * len(values),
            "depth": [4.1] * len(values),
        }
    )


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = FeatureConfig(
            rolling_hours=(1,),
            long_windows_days=(1,),
            min_period_fraction=0.05,
        )

    def test_offline_spike_and_full_plateau_features(self) -> None:
        frame = series_frame([10, 10, 10, 20, 10, 11, 11, 11, 11])
        bundle = build_features(frame, config=self.config, mode="offline")
        self.assertNotIn("label", bundle.feature_columns)
        self.assertIn("station", bundle.categorical_columns)
        self.assertEqual(float(bundle.frame.loc[3, "spike_min_abs_diff"]), 10.0)
        self.assertEqual(float(bundle.frame.loc[6, "plateau_full_length"]), 4.0)
        self.assertEqual(bundle.frame.index.tolist(), frame.index.tolist())

    def test_gap_resets_differences_and_plateau(self) -> None:
        frame = series_frame([7, 7, 7, 7], gap_at=2)
        features = build_features(frame, config=self.config, mode="causal").frame
        self.assertTrue(np.isnan(features.loc[2, "temp_diff_1"]))
        self.assertEqual(float(features.loc[1, "plateau_elapsed"]), 2.0)
        self.assertEqual(float(features.loc[2, "plateau_elapsed"]), 1.0)

    def test_causal_prefix_is_invariant_to_future_values(self) -> None:
        prefix = series_frame([10 + index * 0.1 for index in range(15)])
        full = series_frame([10 + index * 0.1 for index in range(20)])
        full.loc[15:, "temp"] = [50, -20, 80, -40, 100]
        prefix_features = build_features(prefix, config=self.config, mode="causal").frame
        full_features = build_features(full, config=self.config, mode="causal").frame.iloc[:15]
        pd.testing.assert_frame_equal(
            prefix_features,
            full_features,
            check_exact=False,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_peer_reference_and_single_layer_fallback(self) -> None:
        base = series_frame([10 + index * 0.1 for index in range(15)])
        single = build_features(base, config=self.config, mode="offline").frame
        self.assertTrue((single["peer_available"] == 0).all())
        self.assertTrue(single["peer_temp_mean"].isna().all())
        self.assertTrue(single["reference_resid_1d"].notna().any())

        peer = base.copy()
        peer["layer"] = 2
        peer["temp"] += 2.0
        combined = pd.concat([base, peer], ignore_index=True)
        features = build_features(combined, config=self.config, mode="offline").frame
        self.assertTrue((features["peer_available"] == 1).all())
        self.assertAlmostEqual(float(features.loc[0, "temp_peer_residual"]), -2.0)


if __name__ == "__main__":
    unittest.main()
