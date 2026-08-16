from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p2_restore.features import FeatureTable
from p2_restore.state_conditional import compute_state_partition, state_weights


def _table(contrast: np.ndarray) -> FeatureTable:
    size = len(contrast)
    frame = pd.DataFrame(
        {
            "station": ["S-ORS"] * size,
            "layer": np.resize([2, 3, 4], size),
            "time": pd.date_range("2024-01-01", periods=size, freq="10min", tz="Asia/Seoul").astype(
                str
            ),
            "baseline": np.full(size, 20.0),
            "target": np.full(size, 20.1),
            "residual": np.full(size, 0.1),
            "temp_1_minus_5": contrast,
        }
    )
    return FeatureTable(frame, ("baseline", "temp_1_minus_5"))


def test_partition_thresholds_use_training_rows_only() -> None:
    contrast = np.linspace(0.0, 10.0, 240)
    train = np.zeros(240, dtype=bool)
    train[:180] = True
    first = compute_state_partition(_table(contrast), train)
    changed = contrast.copy()
    changed[~train] = 10_000.0
    second = compute_state_partition(_table(changed), train)
    assert first.q40 == second.q40
    assert first.q60 == second.q60
    assert np.array_equal(first.mixed_rows, second.mixed_rows)
    assert np.array_equal(first.stratified_rows, second.stratified_rows)


def test_partition_has_fixed_overlap_and_missing_in_both_experts() -> None:
    contrast = np.linspace(0.0, 10.0, 300)
    contrast[10] = np.nan
    partition = compute_state_partition(_table(contrast))
    assert partition.mixed_rows[10]
    assert partition.stratified_rows[10]
    overlap = partition.mixed_rows & partition.stratified_rows
    assert 0.18 <= overlap.mean() <= 0.22


def test_state_weights_are_monotone_bounded_and_missing_half() -> None:
    table = _table(np.array([0.0, 1.0, 2.0, 3.0, np.nan] * 30))
    weights = state_weights(table, 1.0, 2.0)
    assert np.array_equal(weights[:4], np.array([0.0, 0.0, 1.0, 1.0]))
    assert weights[4] == 0.5
    assert np.all((weights >= 0.0) & (weights <= 1.0))


def test_state_threshold_contract_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="frozen"):
        compute_state_partition(_table(np.linspace(0.0, 1.0, 200)), quantile_low=0.3)
