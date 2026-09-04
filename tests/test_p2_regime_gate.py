from __future__ import annotations

import numpy as np
import pandas as pd

from p2_restore.regime_gate import (
    STATE_FEATURES,
    RobustStateTransform,
    build_public_state_features,
    nested_lobo_soft_gate,
    predict_soft_gate,
    soft_gate_weights,
)


def _synthetic_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(42)
    for block_number, block in enumerate(("a", "b", "c")):
        for layer in (2, 3, 4):
            for number in range(36):
                state = -2.0 + 4.0 * number / 35 + 0.1 * block_number
                model_a = state + 0.25
                model_b = state - 0.25
                truth = model_a if state < 0 else model_b
                row: dict[str, float | int | str] = {
                    "block": block,
                    "layer": layer,
                    "truth": truth + rng.normal(0, 0.01),
                    "model_a": model_a,
                    "model_b": model_b,
                }
                for index, name in enumerate(STATE_FEATURES):
                    row[name] = state if index == 0 else 0.1 * index
                rows.append(row)
    return pd.DataFrame(rows)


def test_robust_transform_is_missing_aware_and_finite() -> None:
    frame = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [np.nan, np.nan, np.nan]})
    transform = RobustStateTransform.fit(frame, ("x", "y"))
    values = transform.transform(frame)
    assert values.shape == (3, 5)
    assert np.isfinite(values).all()
    assert np.array_equal(values[:, -2], [0.0, 1.0, 0.0])
    assert np.array_equal(values[:, -1], [1.0, 1.0, 1.0])


def test_public_features_read_only_public_layers_and_reset_at_gap() -> None:
    times = list(pd.date_range("2025-01-01", periods=150, freq="10min", tz="Asia/Seoul"))
    times.extend(pd.date_range("2025-01-03", periods=150, freq="10min", tz="Asia/Seoul"))
    rows = []
    for number, time in enumerate(times):
        for layer in (1, 2, 3, 4, 5, 6, 7, 8):
            rows.append(
                {
                    "time": time.isoformat(),
                    "layer": layer,
                    "temp": 10 + number * 0.01 + layer,
                    "psal": 30 + layer * 0.01,
                }
            )
    observations = pd.DataFrame(rows)
    keys = observations.loc[observations["layer"].eq(2), ["time", "layer"]].reset_index(drop=True)
    original = observations.copy(deep=True)
    result = build_public_state_features(observations, keys)
    assert observations.equals(original)
    assert tuple(result.columns[2:]) == STATE_FEATURES
    assert pd.isna(result.loc[150, "contrast_delta_past_24h"])


def test_nested_soft_gate_is_convex_and_outer_prediction_is_label_blind() -> None:
    frame = _synthetic_frame()
    first = nested_lobo_soft_gate(
        frame,
        regularization_grid=(0.0, 0.01, 1.0),
        prediction_columns=("model_a", "model_b"),
    )
    baseline_rmse = np.mean((first.baseline_prediction - frame["truth"].to_numpy()) ** 2) ** 0.5
    gate_rmse = np.mean((first.prediction - frame["truth"].to_numpy()) ** 2) ** 0.5
    assert gate_rmse < baseline_rmse
    weights = soft_gate_weights(first.final_gate, frame)
    assert (weights >= 0).all()
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.isfinite(predict_soft_gate(first.final_gate, frame)).all()

    changed = frame.copy()
    held = changed["block"].eq("c")
    changed.loc[held, "truth"] += 100.0
    second = nested_lobo_soft_gate(
        changed,
        regularization_grid=(0.0, 0.01, 1.0),
        prediction_columns=("model_a", "model_b"),
    )
    assert first.selected_regularization["c"] == second.selected_regularization["c"]
    assert np.allclose(first.prediction[held], second.prediction[held])
