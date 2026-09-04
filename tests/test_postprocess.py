from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from p1_qc.postprocess import (
    PostprocessConfig,
    close_short_gaps,
    hysteresis_threshold,
    postprocess_probabilities,
    remove_short_runs,
)


class PostprocessTests(unittest.TestCase):
    def test_hysteresis_grows_only_from_high_seed(self) -> None:
        probability = np.array([0.1, 0.4, 0.7, 0.45, 0.1, 0.4, 0.4])
        result = hysteresis_threshold(
            probability,
            high_threshold=0.65,
            low_threshold=0.35,
        )
        np.testing.assert_array_equal(
            result,
            [False, True, True, True, False, False, False],
        )

    def test_gap_closing_respects_physical_break(self) -> None:
        mask = np.array([True, False, True, True, False, True])
        breaks = np.array([True, False, False, True, True, False])
        result = close_short_gaps(mask, max_gap_rows=1, breaks=breaks)
        np.testing.assert_array_equal(result, [True, True, True, True, False, True])

    def test_singleton_spike_survives_minimum_run_filter(self) -> None:
        mask = np.array([False, True, False, False, True, False])
        preserved = np.array([False, True, False, False, False, False])
        result = remove_short_runs(mask, minimum_run=2, preserve=preserved)
        np.testing.assert_array_equal(result, [False, True, False, False, False, False])

    def test_full_pipeline_returns_original_row_order(self) -> None:
        frame = pd.DataFrame(
            {
                "station": ["S-ORS"] * 4,
                "layer": [1] * 4,
                "time": pd.to_datetime(
                    ["2026-01-01 00:20", "2026-01-01 00:00", "2026-01-01 00:30", "2026-01-01 00:10"]
                ).tz_localize("Asia/Seoul"),
            },
            index=[8, 2, 9, 4],
        )
        probability = np.array([0.4, 0.1, 0.1, 0.8])
        spike = np.array([False, True, False, False])
        result = postprocess_probabilities(
            frame,
            probability,
            config=PostprocessConfig(
                high_threshold=0.7,
                low_threshold=0.3,
                close_gap_rows=0,
                minimum_positive_run=2,
            ),
            singleton_spike_mask=spike,
        )
        self.assertEqual(result.label.index.tolist(), [8, 2, 9, 4])
        self.assertEqual(result.label.tolist(), [1, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
