from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p1_qc.features import FeatureBundle, build_features
from p1_qc.ring_residual import (
    RING_RESIDUAL_FEATURES,
    RingResidualConfig,
    append_ring_residual_features,
    build_ring_residual_features,
    summarize_ring_residual_coverage,
)


def _frame(values: np.ndarray, *, gap_at: int | None = None) -> pd.DataFrame:
    start = pd.Timestamp("2025-01-01T00:00:00+09:00")
    timestamps: list[pd.Timestamp] = []
    current = start
    for position in range(len(values)):
        if position:
            current += pd.Timedelta(minutes=20 if position == gap_at else 10)
        timestamps.append(current)
    return pd.DataFrame(
        {
            "station": "SYNTH",
            "layer": 1,
            "time": [timestamp.isoformat() for timestamp in timestamps],
            "temp": values,
        },
        index=pd.Index(np.arange(len(values)) * 3 + 7, name="source_row"),
    )


def test_config_and_contract_are_not_tunable() -> None:
    config = RingResidualConfig()
    assert config.flank_rows == 432
    assert config.min_flank_observations == 216
    with pytest.raises(TypeError):
        RingResidualConfig(cadence_minutes=20)  # type: ignore[call-arg]
    with pytest.raises(FrozenInstanceError):
        config.exclusion_hours = 48  # type: ignore[misc]


def test_exact_ring_endpoints_and_label_blindness() -> None:
    target = 1_100
    values = np.zeros(2_300, dtype=np.float64)
    # Excluded endpoints must not affect either flank.  The included ring
    # endpoints receive constant values so their median is unambiguous.
    values[target - 576 : target + 577] = 999.0
    values[target - 1_008 : target - 576] = 2.0
    values[target + 577 : target + 1_009] = 4.0
    values[target] = 10.0
    frame = _frame(values)
    poisoned = frame.assign(label=np.arange(len(frame)) % 2, anomaly_type="offset")

    clean_features = build_ring_residual_features(frame)
    poisoned_features = build_ring_residual_features(poisoned)
    pd.testing.assert_frame_equal(clean_features, poisoned_features)
    row = clean_features.loc[frame.index[target]]
    assert row[RING_RESIDUAL_FEATURES[0]] == pytest.approx(8.0)
    assert row[RING_RESIDUAL_FEATURES[1]] == pytest.approx(6.0)
    assert row[RING_RESIDUAL_FEATURES[2]] == pytest.approx(7.0)
    assert row[RING_RESIDUAL_FEATURES[3]] == pytest.approx(2.0)
    assert clean_features.index.equals(frame.index)
    assert all(dtype == np.dtype("float32") for dtype in clean_features.dtypes)


def test_fixed_50_percent_flank_coverage_fails_closed() -> None:
    target = 1_100
    values = np.zeros(2_300, dtype=np.float64)
    past_slice = np.arange(target - 1_008, target - 576)
    values[past_slice[216:]] = np.nan
    covered = build_ring_residual_features(_frame(values))
    assert pd.notna(covered.iloc[target][RING_RESIDUAL_FEATURES[0]])

    values[past_slice[215]] = np.nan
    deficient = build_ring_residual_features(_frame(values))
    assert pd.isna(deficient.iloc[target][RING_RESIDUAL_FEATURES[0]])
    assert pd.isna(deficient.iloc[target][RING_RESIDUAL_FEATURES[2]])
    assert pd.isna(deficient.iloc[target][RING_RESIDUAL_FEATURES[3]])


def test_gap_is_never_bridged_and_input_order_is_restored() -> None:
    values = np.linspace(0.0, 1.0, 2_300)
    continuous = build_ring_residual_features(_frame(values))
    gapped = build_ring_residual_features(_frame(values, gap_at=1_100))
    assert pd.notna(continuous.iloc[1_100][RING_RESIDUAL_FEATURES[2]])
    assert pd.isna(gapped.iloc[1_100][RING_RESIDUAL_FEATURES[2]])

    shuffled = _frame(values).sample(frac=1.0, random_state=23)
    restored = build_ring_residual_features(shuffled)
    assert restored.index.equals(shuffled.index)
    sorted_features = build_ring_residual_features(shuffled.sort_values("time"))
    pd.testing.assert_frame_equal(restored.sort_index(), sorted_features.sort_index())


@pytest.mark.parametrize("kind", ["offset", "drift"])
def test_ring_reduces_long_event_absorption_vs_centered_7d(kind: str) -> None:
    values = np.zeros(2_500, dtype=np.float64)
    event_start = 992
    event_length = 517  # 86h10m, still below the published 86.5h maximum.
    event_stop = event_start + event_length
    if kind == "offset":
        values[event_start:event_stop] = 6.0
    else:
        values[event_start:event_stop] = np.linspace(0.0, 6.0, event_length)
    target = event_start + event_length // 2
    frame = _frame(values).assign(year=2025, psal=33.0, depth=10.0)

    base = build_features(frame, mode="offline")
    ring = build_ring_residual_features(frame)
    current_abs = abs(float(base.frame.iloc[target]["temp_long_resid_7d"]))
    ring_abs = abs(float(ring.iloc[target][RING_RESIDUAL_FEATURES[2]]))
    minimum_absorption_reduction = 2.0 if kind == "offset" else 0.1
    assert ring_abs > current_abs + minimum_absorption_reduction


def test_append_and_coverage_summary_are_fail_closed() -> None:
    frame = _frame(np.zeros(2_300, dtype=np.float64))
    base_frame = pd.DataFrame({"base": np.ones(len(frame), dtype=np.float32)}, index=frame.index)
    bundle = FeatureBundle(base_frame, ("base",), ())
    augmented = append_ring_residual_features(bundle, frame)
    assert augmented.feature_columns == ("base", *RING_RESIDUAL_FEATURES)
    assert augmented.categorical_columns == ()
    assert len(augmented.feature_columns) == len(bundle.feature_columns) + 4

    ring = build_ring_residual_features(frame)
    summary = summarize_ring_residual_coverage(ring)
    assert summary["row_count"] == len(frame)
    assert 0.0 < summary["both_flanks_covered_fraction"] < 1.0

    ring.attrs["min_flank_observations"] = 215
    with pytest.raises(ValueError, match="contract mismatch"):
        summarize_ring_residual_coverage(ring)
