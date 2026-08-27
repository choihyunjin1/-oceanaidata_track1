from __future__ import annotations

import unittest

import numpy as np

from p1_qc.depth_shift_stress import (
    apply_depth_missing_counterfactual,
    apply_unknown_depth_regime_counterfactual,
    comparison_summary,
    depth_fallback_codes,
    weighted_group_counterfactual_summary,
)
from p1_qc.pipeline import TabularEncoder


class DepthShiftStressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.columns = (
            "temp_raw",
            "depth_raw",
            "nominal_depth_m",
            "depth_diff_1",
            "depth_abs_diff_1",
            "depth_missing",
            "depth_regime",
        )
        self.matrix = np.arange(21, dtype=np.float32).reshape(3, 7)

    def test_depth_missing_counterfactual_changes_only_affected_depth_inputs(self) -> None:
        result = apply_depth_missing_counterfactual(
            self.matrix,
            self.columns,
            [False, True, False],
            [-1.0, -1.0, -1.0],
        )
        np.testing.assert_array_equal(result[[0, 2]], self.matrix[[0, 2]])
        self.assertEqual(float(result[1, 0]), float(self.matrix[1, 0]))
        self.assertTrue(np.isnan(result[1, 1:5]).all())
        self.assertEqual(float(result[1, 5]), 1.0)
        self.assertEqual(float(result[1, 6]), -1.0)
        np.testing.assert_array_equal(self.matrix, np.arange(21, dtype=np.float32).reshape(3, 7))

    def test_unknown_regime_preserves_numeric_depth_inputs(self) -> None:
        result = apply_unknown_depth_regime_counterfactual(
            self.matrix, self.columns, [True, False, True]
        )
        np.testing.assert_array_equal(result[:, :6], self.matrix[:, :6])
        np.testing.assert_array_equal(result[:, 6], [-1.0, self.matrix[1, 6], -1.0])

    def test_fallback_codes_use_fitted_map_without_extending_it(self) -> None:
        encoder = TabularEncoder(category_maps={"depth_regime": {"G-ORS|unknown|l1": 7}})
        result = depth_fallback_codes(encoder, ["G-ORS", "S-ORS"], [1, 5])
        np.testing.assert_array_equal(result, [7.0, -1.0])
        self.assertEqual(encoder.category_maps["depth_regime"], {"G-ORS|unknown|l1": 7})

    def test_comparison_summary_reports_exact_f1_fpr_and_flip_deltas(self) -> None:
        result = comparison_summary(
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        )
        self.assertAlmostEqual(result["original"]["f1"], 0.5)
        self.assertAlmostEqual(result["original"]["fpr"], 0.5)
        self.assertAlmostEqual(result["counterfactual"]["f1"], 1.0)
        self.assertAlmostEqual(result["counterfactual"]["fpr"], 0.0)
        self.assertAlmostEqual(result["delta_f1"], 0.5)
        self.assertAlmostEqual(result["delta_fpr"], -0.5)
        self.assertEqual(result["flip_rows"], 2)
        self.assertEqual(result["zero_to_one_rows"], 1)
        self.assertEqual(result["one_to_zero_rows"], 1)

    def test_weighted_group_counterfactual_uses_target_row_shares(self) -> None:
        groups = [
            {"station": "G-ORS", "layer": 1, "rows": 4, "tp": 1, "fp": 1, "fn": 1, "tn": 1},
            {"station": "S-ORS", "layer": 5, "rows": 4, "tp": 2, "fp": 0, "fn": 0, "tn": 2},
        ]
        affected = comparison_summary(
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        )
        result = weighted_group_counterfactual_summary(
            groups,
            {("G-ORS", 1): 0.75, ("S-ORS", 5): 0.25},
            ("G-ORS", 1),
            affected,
        )
        self.assertAlmostEqual(result["original"]["f1"], 0.625)
        self.assertAlmostEqual(result["counterfactual"]["f1"], 1.0)
        self.assertAlmostEqual(result["delta_f1"], 0.375)


if __name__ == "__main__":
    unittest.main()
