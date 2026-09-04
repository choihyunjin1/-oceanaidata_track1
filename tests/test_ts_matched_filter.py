from __future__ import annotations

import numpy as np
import pandas as pd

from p1_qc.ts_matched_filter import TS_MATCHED_FILTER_FEATURES, build_ts_matched_filter_features


def _frame(temp: np.ndarray, psal: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": "S-ORS",
            "layer": 1,
            "time": pd.date_range(
                "2024-01-01", periods=len(temp), freq="10min", tz="Asia/Seoul"
            ).astype(str),
            "temp": temp,
            "psal": psal,
        }
    )


def test_temperature_only_offset_is_retained_but_ts_step_is_softly_reduced() -> None:
    rng = np.random.default_rng(17)
    temp = 15 + rng.normal(0, 0.01, 900).cumsum()
    psal = 32 + rng.normal(0, 0.001, 900).cumsum()
    start, stop = 250, 450
    temp[start:stop] += 2
    temperature_only = build_ts_matched_filter_features(_frame(temp, psal))
    psal[start:stop] += 0.5
    natural_like = build_ts_matched_filter_features(_frame(temp, psal))
    column = TS_MATCHED_FILTER_FEATURES[0]
    assert temperature_only.loc[start : stop - 1, column].median() > 20
    assert natural_like.loc[start : stop - 1, column].median() < (
        temperature_only.loc[start : stop - 1, column].median() * 0.25
    )


def test_missing_psal_retains_finite_temperature_scores() -> None:
    rng = np.random.default_rng(19)
    temp = 10 + rng.normal(0, 0.01, 800).cumsum()
    temp[200:500] += 3
    result = build_ts_matched_filter_features(_frame(temp, np.full(len(temp), np.nan)))
    assert np.isfinite(result.to_numpy()).all()
    assert result[TS_MATCHED_FILTER_FEATURES[0]].max() > 0


def test_target_columns_are_ignored() -> None:
    x = np.arange(800)
    source = _frame(10 + np.sin(x / 30), 32 + np.cos(x / 50) * 0.01)
    poisoned = source.assign(label=x % 2, anomaly_type="offset")
    pd.testing.assert_frame_equal(
        build_ts_matched_filter_features(source),
        build_ts_matched_filter_features(poisoned),
    )
