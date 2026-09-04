from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from p1_qc.rules import (
    PlateauRuleConfig,
    SpikeRuleConfig,
    apply_hard_rules,
    detect_plateaus,
    evaluate_binary_rule,
    plateau_runs,
)


class PlateauRuleTests(unittest.TestCase):
    def test_plateau_requires_six_contiguous_rows(self) -> None:
        values = np.array([1.0] * 5 + [2.0] + [3.0] * 6)
        expected = np.array([False] * 6 + [True] * 6)
        np.testing.assert_array_equal(plateau_runs(values, min_run=6), expected)

    def test_plateau_does_not_cross_observation_gap(self) -> None:
        frame = pd.DataFrame(
            {
                "station": ["S-ORS"] * 6,
                "layer": [1] * 6,
                "time": [
                    "2025-01-01T00:00:00+09:00",
                    "2025-01-01T00:10:00+09:00",
                    "2025-01-01T00:20:00+09:00",
                    "2025-01-01T01:00:00+09:00",
                    "2025-01-01T01:10:00+09:00",
                    "2025-01-01T01:20:00+09:00",
                ],
                "temp": [10.0] * 6,
            }
        )
        mask = detect_plateaus(frame, PlateauRuleConfig(min_run=6))
        self.assertFalse(mask.any())

    def test_component_masks_and_metrics_are_exposed(self) -> None:
        frame = pd.DataFrame(
            {
                "station": ["I-ORS"] * 10,
                "layer": [2] * 10,
                "time": pd.date_range("2025-01-01", periods=10, freq="10min", tz="Asia/Seoul"),
                "temp": [10.0, 10.1, 10.2, 20.0, 10.2, 8.0, 8.0, 8.0, 8.0, 8.0],
            }
        )
        result = apply_hard_rules(
            frame,
            plateau=PlateauRuleConfig(min_run=5),
            spike=SpikeRuleConfig(z_threshold=5, min_abs_jump=1.0),
        )
        self.assertEqual(
            result.singleton_spike.tolist(),
            [False, False, False, True, False, False, False, False, False, False],
        )
        self.assertEqual(int(result.plateau.sum()), 5)
        metric = evaluate_binary_rule(result.label, result.label)
        self.assertEqual(metric.f1, 1.0)
        self.assertEqual(metric.true_positive, 6)


if __name__ == "__main__":
    unittest.main()
