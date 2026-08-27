from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from p2_restore.deep_data import build_panel
from p2_restore.structured_imputer import (
    StructuredMaskBiTCN,
    StructuredMaskConfig,
    build_hourly_panel,
    inference_window,
    interpolate_hourly_prediction,
    materialize_window,
    structured_mask_candidates,
)


def _observations(hours: int = 2400) -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=hours * 6, freq="10min", tz="Asia/Seoul")
    depths = {1: 4.19, 2: 7.04, 3: 9.44, 4: 14.74, 5: 19.59, 6: 30.68, 7: 39.45, 8: 49.35}
    rows = []
    for number, time in enumerate(times):
        for layer, depth in depths.items():
            rows.append(
                {
                    "station": "S-ORS",
                    "year": time.year,
                    "layer": layer,
                    "time": time.isoformat(),
                    "temp": 24 - 0.08 * depth + np.sin(number / 100),
                    "psal": 31 + 0.01 * depth,
                    "depth": depth,
                    "nominal_depth": depth,
                }
            )
    return pd.DataFrame(rows)


def test_hourly_panel_preserves_targets_separately_from_public_inputs() -> None:
    hourly = build_hourly_panel(build_panel(_observations(48)))
    assert len(hourly.times) == 48
    assert hourly.target_mask.all()
    assert not any(name.startswith("target_") for name in hourly.public_names)


def test_structured_blackout_removes_all_three_target_inputs_and_only_scores_mask() -> None:
    hourly = build_hourly_panel(build_panel(_observations()))
    public = np.zeros_like(hourly.public_inputs, dtype=np.float32)
    residual = np.ones_like(hourly.target, dtype=np.float32)
    observed = hourly.target_mask.copy()
    window = (0, 2160, 400, 1120)
    x, _, mask = materialize_window(public, residual, observed, hourly.target_mask, window, 2160)
    public_width = public.shape[1]
    assert np.all(x[400:1120, public_width : public_width + 3] == 0)
    assert np.all(x[400:1120, public_width + 3 :] == 0)
    assert mask[:400].sum() == 0
    assert mask[400:1120].all()


def test_candidates_and_inference_never_use_held_target_rows() -> None:
    hourly = build_hourly_panel(build_panel(_observations()))
    train = np.ones(len(hourly.times), dtype=bool)
    train[500:800] = False
    starts = structured_mask_candidates(hourly.target_mask, train, 168, min_coverage=1.0)
    assert not any(start < 800 and start + 168 > 500 for start in starts)
    config = StructuredMaskConfig(window_hours=2160, context_hours=336)
    public = np.zeros_like(hourly.public_inputs, dtype=np.float32)
    residual = np.ones_like(hourly.target, dtype=np.float32)
    block = np.zeros(len(hourly.times), dtype=bool)
    block[500:800] = True
    x, local, _ = inference_window(hourly, public, residual, hourly.target_mask, block, config)
    public_width = public.shape[1]
    assert np.all(x[local, public_width : public_width + 6] == 0)


def test_model_backward_and_hourly_interpolation_are_finite() -> None:
    model = StructuredMaskBiTCN(20, hidden=24, blocks=3)
    inputs = torch.randn(2, 96, 20)
    output = model(inputs)
    output.square().mean().backward()
    assert output.shape == (2, 96, 3)
    times = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    values = np.arange(9, dtype=float).reshape(3, 3)
    target = pd.date_range("2024-01-01", periods=5, freq="30min", tz="UTC")
    interpolated = interpolate_hourly_prediction(times, values, target)
    assert interpolated.shape == (5, 3)
    assert np.isfinite(interpolated).all()
