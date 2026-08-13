from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from p1_qc.augment import ANOMALY_TYPES, AugmentConfig, augment_training_fold


def make_frame(rows: int = 5000) -> pd.DataFrame:
    time = pd.date_range("2025-01-01", periods=rows, freq="10min", tz="Asia/Seoul")
    phase = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "station": ["S-ORS"] * rows,
            "layer": [2] * rows,
            "time": time,
            "temp": 15 + np.sin(phase / 25) + 0.02 * np.sin(phase / 3),
            "psal": 33 + 0.1 * np.cos(phase / 30),
            "depth": np.full(rows, 10.0),
            "label": np.zeros(rows, dtype=np.int8),
            "anomaly_type": [""] * rows,
        }
    )


class AugmentationTests(unittest.TestCase):
    def test_is_deterministic_and_does_not_mutate_fold(self) -> None:
        frame = make_frame()
        original = frame.copy(deep=True)
        config = AugmentConfig(target_fraction=0.20, overlap_fraction=0.5, seed=17)
        first = augment_training_fold(frame, config)
        second = augment_training_fold(frame, config)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(first.frame, second.frame)
        pd.testing.assert_frame_equal(first.events, second.events)
        self.assertGreater(int(first.injected_mask.sum()), 0)

    def test_five_types_and_overlap_are_injected_with_official_durations(self) -> None:
        result = augment_training_fold(
            make_frame(),
            AugmentConfig(target_fraction=0.30, overlap_fraction=1.0, seed=9),
        )
        primary = result.events[~result.events["is_overlap"]]
        self.assertTrue(set(ANOMALY_TYPES).issubset(set(primary["anomaly_type"])))
        for _, event in result.events.iterrows():
            minimum = {"spike": 1, "noise": 18, "flatline": 12, "offset": 48, "drift": 54}[
                event["anomaly_type"]
            ]
            maximum = {"spike": 1, "noise": 353, "flatline": 283, "offset": 519, "drift": 519}[
                event["anomaly_type"]
            ]
            self.assertLessEqual(minimum, int(event["rows"]))
            self.assertLessEqual(int(event["rows"]), maximum)
        self.assertTrue(result.injected_types.str.contains(r"\+").any())
        self.assertTrue((result.frame.loc[result.injected_mask, "label"] == 1).all())

    def test_existing_positive_is_never_modified_or_relabelled(self) -> None:
        frame = make_frame()
        frame.loc[200:260, "label"] = 1
        frame.loc[200:260, "anomaly_type"] = "offset"
        original_values = frame.loc[200:260, "temp"].copy()
        result = augment_training_fold(
            frame,
            AugmentConfig(target_fraction=0.15, overlap_fraction=0.2, seed=21),
        )
        pd.testing.assert_series_equal(result.frame.loc[200:260, "temp"], original_values)
        self.assertTrue((result.frame.loc[200:260, "anomaly_type"] == "offset").all())
        self.assertFalse(result.injected_mask.iloc[200:261].any())


if __name__ == "__main__":
    unittest.main()
