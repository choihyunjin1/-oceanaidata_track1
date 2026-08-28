from __future__ import annotations

import numpy as np
import pandas as pd

from p3_wave.chronos2_transfer import (
    LEAD_INDICES,
    point_predictions,
    prediction_frame,
    prepare_context_inputs,
    score_frame,
)


def test_prepare_context_inputs_uses_20_minute_grid_and_finite_covariates() -> None:
    raw = np.ones((1, 289, 10), dtype=np.float32)
    raw[0, ::2, 3] = 90.0
    raw[0, ::2, 6] = 180.0
    raw[0, 2, 1] = np.nan
    item = prepare_context_inputs(raw, [0])[0]
    assert np.asarray(item["target"]).shape == (145,)
    assert set(item["past_covariates"]) == {
        "tp", "hmax", "wvdir_sin", "wvdir_cos", "wspd", "gust",
        "wdir_sin", "wdir_cos", "airt", "relh", "caph",
    }
    assert all(np.isfinite(values).all() for values in item["past_covariates"].values())


class _FakePipeline:
    quantiles = [0.1, 0.5, 0.9]

    def predict(self, inputs, **kwargs):
        import torch

        output = []
        for _ in inputs:
            values = torch.zeros((1, 3, 72), dtype=torch.float32)
            values[0, 1] = torch.arange(72)
            output.append(values)
        return output


def test_point_predictions_extract_official_leads() -> None:
    predictions = point_predictions(_FakePipeline(), [{"target": np.ones(145)}], batch_size=1)
    np.testing.assert_array_equal(predictions[0], np.clip(LEAD_INDICES, 0, 30))


def test_prediction_frame_and_score_are_case_lead_aligned() -> None:
    anchors = pd.DataFrame(
        {
            "anchor_id": [0],
            "station": ["G-ORS"],
            "current_hs": [1.5],
            **{f"target_{lead}": [2.0] for lead in (3, 6, 9, 12, 18, 24)},
        }
    )
    frame = prediction_frame(anchors, [0], np.full((1, 6), 2.0), fold="f")
    result = score_frame(frame)
    assert result["rows"] == 6
    assert result["cases"] == 1
    assert result["rmse_m"] == 0.0
