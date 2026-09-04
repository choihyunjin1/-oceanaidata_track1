from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p2_restore.corrected_repeated_forward import (
    build_joint_masked_population,
    forward_training_mask,
    joint_mask_target_context,
    metric_report,
    nominal_target_rows,
)


def _observations(periods: int = 24) -> pd.DataFrame:
    times = pd.date_range("2024-08-01", periods=periods, freq="10min", tz="Asia/Seoul")
    depths = {1: 4.19, 2: 7.04, 3: 9.44, 4: 14.74, 5: 19.59, 6: 30.68, 7: 39.45, 8: 49.35}
    rows: list[dict[str, object]] = []
    for step, timestamp in enumerate(times):
        for layer, depth in depths.items():
            rows.append(
                {
                    "station": "S-ORS",
                    "year": 2024,
                    "layer": layer,
                    "time": timestamp.isoformat(),
                    "temp": 25.0 - 0.08 * depth + 0.01 * step,
                    "psal": 31.0 + 0.01 * depth,
                    "depth": depth,
                    "nominal_depth": depth,
                }
            )
    return pd.DataFrame(rows)


def test_joint_mask_features_are_invariant_to_target_temp_and_psal() -> None:
    source = _observations()
    masked, audit = joint_mask_target_context(source)
    original = build_joint_masked_population(source, masked)

    perturbed = source.copy()
    target = perturbed["layer"].isin([2, 3, 4])
    perturbed.loc[target, "temp"] += 50.0
    perturbed.loc[target, "psal"] -= 20.0
    perturbed_masked, _ = joint_mask_target_context(perturbed)
    changed = build_joint_masked_population(perturbed, perturbed_masked)

    assert audit.target_temp_non_null_after_mask == 0
    assert audit.target_psal_non_null_after_mask == 0
    assert original.feature_columns == changed.feature_columns
    assert original.frame[["station", "layer", "time"]].equals(
        changed.frame[["station", "layer", "time"]]
    )
    np.testing.assert_allclose(
        original.frame.loc[:, original.feature_columns],
        changed.frame.loc[:, changed.feature_columns],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        changed.frame["target"].to_numpy(float) - original.frame["target"].to_numpy(float),
        50.0,
    )


def test_hidden_target_population_fails_closed() -> None:
    source = _observations(periods=2)
    source["time"] = source["time"].str.replace("2024-08-01", "2025-09-01")
    with pytest.raises(ValueError, match="hidden target-layer"):
        joint_mask_target_context(source)


def test_forward_training_mask_obeys_seven_day_embargo() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.date_range(
                "2024-08-20", periods=20, freq="D", tz="Asia/Seoul"
            ).astype(str),
            "residual": np.ones(20),
        }
    )
    selected, cutoff = forward_training_mask(
        frame, "2024-09-10T00:00:00+09:00", embargo_days=7
    )
    selected_time = pd.to_datetime(frame.loc[selected, "time"], utc=True)
    assert selected_time.max() < cutoff
    assert not selected[pd.to_datetime(frame["time"], utc=True).ge(cutoff).to_numpy()].any()


def test_metric_report_uses_fold_equal_official_layer_weighted_mse() -> None:
    frame = pd.DataFrame(
        {
            "fold": ["a"] * 3 + ["b"] * 3,
            "layer": [2, 3, 4, 2, 3, 4],
            "truth": np.zeros(6),
            "prediction": [1.0, 2.0, 3.0, 2.0, 2.0, 2.0],
        }
    )
    report = metric_report(
        frame,
        prediction_column="prediction",
        official_layer_counts={"2": 1, "3": 1, "4": 1},
    )
    expected_a_mse = (1.0 + 4.0 + 9.0) / 3.0
    expected_b_mse = 4.0
    expected = np.sqrt((expected_a_mse + expected_b_mse) / 2.0)
    assert report["fold_equal_official_layer_weighted_rmse_c"] == pytest.approx(expected)


def test_nominal_rows_respect_10_minute_three_layer_grain() -> None:
    assert nominal_target_rows(
        "2024-09-01T00:00:00+09:00", "2024-11-01T00:00:00+09:00"
    ) == 26_352
