from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from p2_restore.deep_data import PanelNormalizer, build_panel, make_chunk_bounds, time_block_mask
from p2_restore.deep_models import MODEL_SPECS, ConditionalDiffusion, build_model, count_parameters
from p2_restore.deep_training import blend_grid


def _observations(rows: int = 36) -> pd.DataFrame:
    times = pd.date_range("2024-08-31 22:00", periods=rows, freq="10min", tz="Asia/Seoul")
    nominal = {1: 4.19, 2: 7.04, 3: 9.44, 4: 14.74, 5: 19.59, 6: 30.68, 7: 39.45, 8: 49.35}
    records = []
    for number, time in enumerate(times):
        for layer, depth in nominal.items():
            temp = 24.0 - depth * 0.08 + np.sin(number / 8) * 0.2
            psal = 31.0 + depth * 0.01
            records.append(
                {
                    "station": "S-ORS",
                    "year": time.year,
                    "layer": layer,
                    "time": time.isoformat(),
                    "temp": temp,
                    "psal": psal,
                    "depth": depth,
                    "nominal_depth": depth,
                }
            )
    return pd.DataFrame(records)


def test_panel_excludes_target_observations_and_normalizes_fold_locally() -> None:
    frame = _observations()
    panel = build_panel(frame)
    assert panel.inputs.shape[0] == 36
    assert not (
        {"temp_2", "temp_3", "temp_4", "psal_2", "psal_3", "psal_4"} & set(panel.input_names)
    )
    validation = time_block_mask(panel, "2024-09-01", "2024-09-02")
    normalizer = PanelNormalizer.fit(panel, ~validation)
    inputs = normalizer.transform_inputs(panel.inputs)
    target, mask = normalizer.transform_targets(panel)
    assert np.isfinite(inputs).all()
    assert np.isfinite(target).all()
    assert mask.all()


def test_gap_aware_chunks_do_not_cross_segments() -> None:
    segment = np.array([0] * 8 + [1] * 15)
    bounds = make_chunk_bounds(segment, length=10, stride=6)
    assert bounds
    assert all(np.unique(segment[start:stop]).size == 1 for start, stop in bounds)


@pytest.mark.parametrize("name", [spec.name for spec in MODEL_SPECS])
def test_every_tournament_model_has_aligned_output_and_backward(name: str) -> None:
    torch.manual_seed(7)
    model = build_model(name, 12)
    inputs = torch.randn(2, 48, 12)
    target = torch.randn(2, 48, 3)
    mask = torch.ones_like(target)
    if isinstance(model, ConditionalDiffusion):
        loss = model.training_loss(inputs, target, mask)
        prediction = model.predict(inputs, samples=1)
    else:
        prediction = model(inputs)
        loss = (prediction - target).square().mean()
    loss.backward()
    assert prediction.shape == (2, 48, 3)
    assert torch.isfinite(prediction).all()
    assert count_parameters(model) > 10_000


def test_blend_grid_normalizes_string_and_timezone_datetime_keys() -> None:
    deep = pd.DataFrame(
        {
            "time": ["2024-09-01 00:00:00+09:00", "2024-09-01 00:10:00+09:00"],
            "layer": [2, 2],
            "block": ["a", "a"],
            "truth": [20.0, 21.0],
            "prediction": [20.1, 21.1],
        }
    )
    incumbent = pd.DataFrame(
        {
            "time": pd.to_datetime(deep["time"]),
            "layer": [2, 2],
            "block": ["a", "a"],
            "truth": [20.0, 21.0],
            "router_400": [19.5, 20.5],
        }
    )
    result = blend_grid(deep, incumbent, weights=(0.0, 0.5, 1.0))
    assert result["rows"] == 2
    assert result["selected"]["deep_weight"] == 1.0
