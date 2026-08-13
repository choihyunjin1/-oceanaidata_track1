from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p1_qc.config import FoldWindowConfig
from p1_qc.splits import outer_folds


class SplitTests(unittest.TestCase):
    def test_explicit_purge_and_positional_indices(self) -> None:
        times = pd.date_range("2025-03-01", "2025-04-03", freq="1D", tz="Asia/Seoul")
        frame = pd.DataFrame(
            {
                "station": ["S-ORS"] * len(times),
                "layer": [1] * len(times),
                "time": times.astype(str),
                "label": [0] * len(times),
            },
            index=np.arange(100, 100 + len(times)),
        )
        spec = FoldWindowConfig(
            "q2_test",
            "2025-03-24T00:00:00+09:00",
            "2025-04-01T00:00:00+09:00",
            "2025-04-04T00:00:00+09:00",
        )
        fold = outer_folds(
            frame,
            specs=[spec],
            cadence_minutes=1440,
            purge_days=7,
        )[0]
        self.assertTrue(np.issubdtype(fold.train_idx.dtype, np.integer))
        self.assertEqual(
            frame.iloc[fold.train_idx]["time"].max(),
            str(times[times <= pd.Timestamp("2025-03-24", tz="Asia/Seoul")].max()),
        )
        self.assertTrue(
            pd.to_datetime(frame.iloc[fold.val_idx]["time"], utc=True)
            .ge(pd.Timestamp("2025-04-01", tz="Asia/Seoul").tz_convert("UTC"))
            .all()
        )

    def test_positive_event_touching_boundary_is_included_whole(self) -> None:
        times = pd.date_range("2025-03-20", "2025-04-03", freq="1D", tz="Asia/Seoul")
        labels = [
            int(
                pd.Timestamp("2025-04-01", tz="Asia/Seoul")
                <= time
                <= pd.Timestamp("2025-04-03", tz="Asia/Seoul")
            )
            for time in times
        ]
        frame = pd.DataFrame(
            {
                "station": ["S-ORS"] * len(times),
                "layer": [1] * len(times),
                "time": times.astype(str),
                "label": labels,
            }
        )
        spec = FoldWindowConfig(
            "boundary",
            "2025-03-23T00:00:00+09:00",
            "2025-04-01T00:00:00+09:00",
            "2025-04-02T00:00:00+09:00",
        )
        protected = outer_folds(
            frame,
            specs=[spec],
            cadence_minutes=1440,
            purge_days=7,
            protect_positive_runs=True,
        )[0]
        validation_times = set(frame.iloc[protected.val_idx]["time"])
        self.assertIn(str(pd.Timestamp("2025-04-01", tz="Asia/Seoul")), validation_times)
        self.assertIn(str(pd.Timestamp("2025-04-03", tz="Asia/Seoul")), validation_times)
        self.assertFalse(set(protected.train_idx).intersection(protected.val_idx))

    def test_short_embargo_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "station": ["S-ORS", "S-ORS"],
                "layer": [1, 1],
                "time": ["2025-03-30T00:00:00+09:00", "2025-04-01T00:00:00+09:00"],
                "label": [0, 0],
            }
        )
        spec = FoldWindowConfig(
            "bad",
            "2025-03-30T00:00:00+09:00",
            "2025-04-01T00:00:00+09:00",
            "2025-04-02T00:00:00+09:00",
        )
        with self.assertRaises(ValueError):
            outer_folds(frame, specs=[spec], purge_days=7)


if __name__ == "__main__":
    unittest.main()
