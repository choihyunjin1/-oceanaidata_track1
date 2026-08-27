from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p1_qc.causal_raw_features_v4r2 import (
    CAUSAL_FEATURE_COLUMNS,
    assert_future_value_invariance,
    build_causal_raw_features,
    build_exact_prefix_causal_matrix,
)


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for station in ("G-ORS", "I-ORS"):
        for index, time in enumerate(pd.date_range("2025-01-01", periods=12, freq="10min")):
            rows.append(
                {
                    "station": station,
                    "layer": 1,
                    "time": time,
                    "temp": float(index),
                    "psal": np.nan if index == 4 else 30.0 + index,
                    "depth": 100.0 + index,
                    "label": 999,
                    "anomaly_type": "must_not_be_read",
                }
            )
    return pd.DataFrame(rows)


def test_builder_is_exact_backward_allowlist_and_never_emits_targets() -> None:
    features = build_causal_raw_features(_frame())
    assert tuple(features.columns) == CAUSAL_FEATURE_COLUMNS
    assert "label" not in features and "anomaly_type" not in features
    assert features.shape == (24, 9)
    assert features.loc[1, "temp_diff_1"] == 1.0
    assert features.loc[2, "temp_backward_acceleration"] == 0.0


def test_future_raw_value_perturbation_cannot_change_prefix_features() -> None:
    frame = _frame()
    prefix_ids = np.r_[np.arange(0, 8), np.arange(12, 20)]
    original = build_causal_raw_features(frame)
    changed = frame.copy()
    future_ids = np.setdiff1d(np.arange(len(frame)), prefix_ids)
    changed.loc[future_ids, ["temp", "psal", "depth"]] = 1_000_000.0
    perturbed = build_causal_raw_features(changed)
    np.testing.assert_array_equal(
        original.iloc[prefix_ids].to_numpy(),
        perturbed.iloc[prefix_ids].to_numpy(),
        strict=True,
    )
    assert len(assert_future_value_invariance(frame, prefix_ids)) == 64


def test_prefix_matrix_is_rebuilt_from_prefix_rows_and_rejects_unsafe_reference() -> None:
    frame = _frame()
    prefix_ids = np.r_[np.arange(0, 8), np.arange(12, 20)]
    full = build_causal_raw_features(frame).to_numpy(np.float32)
    matrix, digest = build_exact_prefix_causal_matrix(
        frame, prefix_ids, full_reference=full
    )
    assert len(digest) == 64
    np.testing.assert_array_equal(matrix[prefix_ids], full[prefix_ids], strict=True)
    assert np.count_nonzero(matrix[np.setdiff1d(np.arange(len(frame)), prefix_ids)]) == 0
    unsafe = full.copy()
    unsafe[prefix_ids[0], 0] += 1.0
    with pytest.raises(PermissionError, match="differs"):
        build_exact_prefix_causal_matrix(frame, prefix_ids, full_reference=unsafe)


def test_prefix_rebuild_is_invariant_to_every_future_row_field() -> None:
    frame = _frame()
    prefix_ids = np.r_[np.arange(0, 8), np.arange(12, 20)]
    full = build_causal_raw_features(frame).to_numpy(np.float32)
    future_ids = np.setdiff1d(np.arange(len(frame)), prefix_ids)
    changed = frame.copy()
    changed.loc[future_ids, "station"] = "FUTURE-MOVED"
    changed.loc[future_ids, "layer"] = 999
    changed.loc[future_ids, "time"] = pd.Timestamp("1900-01-01")
    changed.loc[future_ids, ["temp", "psal", "depth"]] = -1_000_000.0
    rebuilt, _ = build_exact_prefix_causal_matrix(
        changed, prefix_ids, full_reference=full
    )
    np.testing.assert_array_equal(rebuilt[prefix_ids], full[prefix_ids], strict=True)
