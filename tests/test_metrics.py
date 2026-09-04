from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p1_qc.metrics import (
    anomaly_type_recall,
    binary_counts,
    event_report,
    group_report,
    micro_f1,
    weighted_group_counts,
)


class MetricTests(unittest.TestCase):
    def test_official_binary_f1(self) -> None:
        counts = binary_counts([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((counts.tp, counts.fp, counts.fn, counts.tn), (1, 1, 1, 1))
        self.assertEqual(counts.precision, 0.5)
        self.assertEqual(counts.recall, 0.5)
        self.assertEqual(micro_f1([1, 1, 0, 0], [1, 0, 1, 0]), 0.5)

    def test_group_and_target_share_weighting(self) -> None:
        metadata = pd.DataFrame({"station": ["A", "A", "B", "B"], "layer": [1, 1, 1, 1]})
        truth = [1, 0, 1, 0]
        prediction = [1, 0, 0, 0]
        groups = group_report(truth, prediction, metadata)
        self.assertEqual(len(groups), 2)
        self.assertEqual(float(groups.loc[groups.station.eq("A"), "f1"].iloc[0]), 1.0)
        weighted = weighted_group_counts(
            truth,
            prediction,
            metadata,
            {("A", 1): 0.25, ("B", 1): 0.75},
        )
        # A contributes weighted TP=.125; B contributes weighted FN=.375.
        self.assertAlmostEqual(weighted.tp, 0.125)
        self.assertAlmostEqual(weighted.fn, 0.375)
        self.assertAlmostEqual(weighted.f1, 0.4)

    def test_event_metrics_respect_false_gaps(self) -> None:
        metadata = pd.DataFrame(
            {
                "station": ["S"] * 8,
                "layer": [1] * 8,
                "time": pd.date_range(
                    "2025-01-01", periods=8, freq="10min", tz="Asia/Seoul"
                ).astype(str),
            }
        )
        truth = [0, 1, 1, 0, 0, 1, 0, 0]
        prediction = [0, 0, 1, 0, 0, 1, 0, 1]
        report = event_report(truth, prediction, metadata)
        self.assertEqual(report.true_events, 2)
        self.assertEqual(report.detected_true_events, 2)
        self.assertEqual(report.predicted_events, 3)
        self.assertEqual(report.matched_predicted_events, 2)
        self.assertAlmostEqual(report.precision, 2 / 3)
        self.assertEqual(report.recall, 1.0)
        self.assertAlmostEqual(report.f1, 0.8)

    def test_multilabel_type_recall(self) -> None:
        prediction = [1, 0, 1, 0]
        anomaly = pd.Series(["noise+drift", "drift", "flatline+flatline", None])
        recall = anomaly_type_recall(prediction, anomaly)
        self.assertEqual(recall["noise"], 1.0)
        self.assertEqual(recall["drift"], 0.5)
        self.assertEqual(recall["flatline"], 1.0)
        self.assertTrue(np.isnan(recall["spike"]))

    def test_invalid_labels_fail(self) -> None:
        with self.assertRaises(ValueError):
            binary_counts([0, 2], [0, 1])


if __name__ == "__main__":
    unittest.main()
