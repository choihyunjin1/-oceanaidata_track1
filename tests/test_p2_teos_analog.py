from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from p2_teos_analog import (
    AnalogConfig,
    AnalogResidualModel,
    PublicState,
    blend_with_frozen,
    build_public_state,
    catalog_split,
    gap_aware_change,
    linear_density_anomaly,
    mask_target_interval,
)


def _observations(periods: int = 20) -> pd.DataFrame:
    times = pd.date_range("2024-08-31 15:00", periods=periods, freq="10min", tz="UTC")
    rows: list[dict[str, object]] = []
    for number, time in enumerate(times):
        for layer in range(1, 9):
            rows.append(
                {
                    "station": "S-ORS",
                    "year": 2024,
                    "layer": layer,
                    "time": time.isoformat(),
                    "temp": 25.0 - 0.2 * layer + 0.01 * number,
                    "psal": 31.0 + 0.05 * layer - 0.002 * number,
                    "depth": float(layer * 5),
                    "nominal_depth": float(layer * 5),
                }
            )
    return pd.DataFrame(rows)


def test_linear_density_proxy_has_expected_monotonicity() -> None:
    reference = linear_density_anomaly(np.array([15.0]), np.array([35.0]))[0]
    warmer = linear_density_anomaly(np.array([16.0]), np.array([35.0]))[0]
    saltier = linear_density_anomaly(np.array([15.0]), np.array([36.0]))[0]
    assert warmer < reference < saltier
    assert np.isnan(linear_density_anomaly(np.array([np.nan]), np.array([35.0]))[0])


def test_gap_aware_change_never_crosses_a_missing_cadence() -> None:
    times = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2024-01-01T00:00Z",
                "2024-01-01T00:10Z",
                "2024-01-01T00:20Z",
                "2024-01-01T00:40Z",
                "2024-01-01T00:50Z",
                "2024-01-01T01:00Z",
            ]
        )
    )
    result = gap_aware_change(np.arange(6.0), times, 20)
    assert result[2] == 2.0
    assert np.isnan(result[3])
    assert result[5] == 2.0


def test_target_temp_and_psal_are_masked_together_and_never_enter_state() -> None:
    observations = _observations()
    masked, diagnostics = mask_target_interval(observations, "2024-09-01", "2024-09-02")
    target = masked["layer"].isin((2, 3, 4))
    assert masked.loc[target, ["temp", "psal"]].isna().all().all()
    assert diagnostics["layers"] == [2, 3, 4]
    assert diagnostics["variables"] == ["temp", "psal"]

    original_state = build_public_state(observations)
    altered = observations.copy()
    altered.loc[altered["layer"].isin((2, 3, 4)), ["temp", "psal"]] = 9999.0
    altered_state = build_public_state(altered)
    assert not any("temp_2" in name or "psal_4" in name for name in original_state.feature_columns)
    assert_frame_equal(original_state.frame, altered_state.frame)


def test_catalog_split_excludes_full_validation_and_seven_day_purge() -> None:
    times = pd.date_range("2024-07-01", "2024-11-30", freq="1D", tz="UTC")
    target_mask = np.ones((len(times), 3), dtype=bool)
    split = catalog_split(
        times,
        target_mask,
        "2024-09-01",
        "2024-11-01",
        purge_days=7,
    )
    assert not np.any(split.training & split.validation)
    assert not split.training[(times >= "2024-08-25") & (times < "2024-11-08")].any()
    assert split.validation.sum() == 61


def _synthetic_state(rows: int, offset: float = 0.0) -> PublicState:
    x = np.linspace(-2.0, 2.0, rows) + offset
    frame = pd.DataFrame(
        {
            "x": x,
            "x2": x**2,
            "sin": np.sin(x),
            "cos": np.cos(x),
        },
        index=pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC"),
    )
    return PublicState(frame, tuple(frame.columns))


def test_local_linear_analog_is_deterministic_and_fallback_is_exact() -> None:
    training = _synthetic_state(200)
    x = training.frame["x"].to_numpy()
    residual = np.column_stack((0.2 + 0.4 * x, -0.1 + 0.2 * x, 0.3 - 0.1 * x))
    setting = AnalogConfig(
        neighbors=32,
        pca_components=3,
        ridge=0.01,
        blend=0.35,
        max_normalized_neighbor_distance=100.0,
        min_effective_neighbors=8.0,
        max_query_missing_fraction=1.0,
        minimum_feature_coverage=0.5,
        batch_size=16,
        n_jobs=1,
        seed=7,
    )
    model = AnalogResidualModel.fit(training, residual, config=setting)
    query = _synthetic_state(25, offset=0.05)
    first = model.predict(query)
    second = model.predict(query)
    assert np.array_equal(first.supported, second.supported)
    assert np.allclose(first.residual, second.residual, equal_nan=True)
    assert first.supported.all()

    base = np.linspace(10.0, 11.0, 25)
    unsupported = np.zeros(25, dtype=bool)
    combined = blend_with_frozen(base, base + 3.0, unsupported, blend=setting.blend)
    assert np.array_equal(combined, base)


def test_analog_support_rejects_far_state() -> None:
    training = _synthetic_state(100)
    residual = np.zeros((100, 3), dtype=float)
    setting = AnalogConfig(
        neighbors=16,
        pca_components=2,
        ridge=1.0,
        max_normalized_neighbor_distance=1e-8,
        min_effective_neighbors=4.0,
        max_query_missing_fraction=1.0,
        minimum_feature_coverage=0.5,
        batch_size=8,
        n_jobs=1,
        seed=11,
    )
    model = AnalogResidualModel.fit(training, residual, config=setting)
    prediction = model.predict(_synthetic_state(5, offset=100.0))
    assert not prediction.supported.any()
    assert np.isnan(prediction.residual).all()
