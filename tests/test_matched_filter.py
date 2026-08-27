from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p1_qc.features import FeatureBundle
from p1_qc.matched_filter import (
    MATCHED_FILTER_FEATURES,
    MatchedFilterConfig,
    append_matched_filter_features,
    build_matched_filter_features,
)


def _frame(values: np.ndarray, *, gap_at: int | None = None) -> pd.DataFrame:
    count = len(values)
    time = pd.date_range("2024-01-01", periods=count, freq="10min", tz="Asia/Seoul")
    if gap_at is not None:
        time = time.to_series(index=np.arange(count))
        time.loc[gap_at:] += pd.Timedelta(minutes=10)
        time = pd.DatetimeIndex(time)
    return pd.DataFrame(
        {
            "station": "S-ORS",
            "layer": 1,
            "time": time.astype(str),
            "temp": values,
        }
    )


def test_config_is_immutable_and_problem_derived() -> None:
    config = MatchedFilterConfig()
    assert (config.offset_min_rows, config.drift_min_rows, config.maximum_rows) == (48, 54, 519)
    with pytest.raises(TypeError):
        MatchedFilterConfig(maximum_rows=100)  # type: ignore[call-arg]


def test_exact_offset_pair_is_projected_only_inside_interval() -> None:
    rng = np.random.default_rng(7)
    values = 15 + rng.normal(0, 0.01, 900).cumsum()
    start, stop = 200, 400
    values[start:stop] += 2.0
    features = build_matched_filter_features(_frame(values))
    score = features[MATCHED_FILTER_FEATURES[0]].to_numpy()
    assert np.median(score[start:stop]) > 20
    assert score[start - 1] < np.median(score[start:stop])
    assert features.loc[start : stop - 1, MATCHED_FILTER_FEATURES[1]].median() == stop - start


def test_linear_drift_and_reset_produce_high_drift_score() -> None:
    rng = np.random.default_rng(11)
    values = 12 + rng.normal(0, 0.008, 1_000).cumsum()
    start, stop = 250, 550
    values[start:stop] += np.linspace(0, 3.0, stop - start)
    features = build_matched_filter_features(_frame(values))
    score = features[MATCHED_FILTER_FEATURES[2]].to_numpy()
    assert np.median(score[start:stop]) > 20
    recovered = features.loc[start : stop - 1, MATCHED_FILTER_FEATURES[3]].median()
    assert abs(recovered - (stop - start)) <= 12


def test_gap_prevents_boundary_pairing() -> None:
    values = np.linspace(10, 11, 800)
    values[200:500] += 4
    no_gap = build_matched_filter_features(_frame(values))
    with_gap = build_matched_filter_features(_frame(values, gap_at=350))
    assert no_gap[MATCHED_FILTER_FEATURES[0]].max() > 0
    assert with_gap.loc[200:499, MATCHED_FILTER_FEATURES[0]].max() < (
        no_gap[MATCHED_FILTER_FEATURES[0]].max() * 0.01
    )


def test_labels_and_types_cannot_change_features() -> None:
    values = 10 + np.sin(np.arange(800) / 30)
    source = _frame(values)
    poisoned = source.assign(label=np.arange(len(source)) % 2, anomaly_type="drift")
    left = build_matched_filter_features(source)
    right = build_matched_filter_features(poisoned)
    pd.testing.assert_frame_equal(left, right)


def test_row_order_and_index_are_restored() -> None:
    values = 10 + np.sin(np.arange(700) / 20)
    source = _frame(values).iloc[::-1].copy()
    source.index = np.arange(10_000, 10_000 + len(source))
    result = build_matched_filter_features(source)
    assert result.index.equals(source.index)
    assert tuple(result.columns) == MATCHED_FILTER_FEATURES
    assert all(result[column].dtype == np.float32 for column in result)


def test_append_contract_and_duplicate_guard() -> None:
    source = _frame(10 + np.sin(np.arange(700) / 20))
    base_frame = pd.DataFrame({"base": np.arange(len(source), dtype=np.float32)})
    bundle = FeatureBundle(base_frame, ("base",), ())
    appended = append_matched_filter_features(bundle, source)
    assert appended.feature_columns == ("base", *MATCHED_FILTER_FEATURES)
    with pytest.raises(ValueError, match="already present"):
        append_matched_filter_features(appended, source)
