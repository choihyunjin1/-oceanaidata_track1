from __future__ import annotations

import numpy as np
import pandas as pd

from p2_tide_rts import (
    TARGET_LAYERS,
    PublicFactorEncoder,
    ResidualRegressor,
    TideRTSConfig,
    actual_depth_interpolation,
    build_tide_panel,
    cadence_segments,
    exact_fallback,
    kalman_rts_smoother,
    m2_relationship_diagnostics,
    observability_diagnostics,
    outer_split,
    residual_skill_r2,
)


def _observations(rows: int = 1_200) -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="Asia/Seoul")
    minute = np.arange(rows, dtype=float)
    m2 = np.sin(2 * np.pi * minute / (12.42 * 6))
    slow = np.sin(2 * np.pi * minute / (30 * 24 * 6))
    depth_by_layer = {1: 4.19, 2: 7.04, 3: 9.44, 4: 14.74, 5: 19.59, 6: 30.68, 7: 39.45, 8: 49.35}
    frames: list[pd.DataFrame] = []
    for layer in range(1, 9):
        depth = depth_by_layer[layer] + 0.02 * np.sin(2 * np.pi * minute / (24 * 6))
        frames.append(
            pd.DataFrame(
                {
                    "time": times.astype(str),
                    "layer": layer,
                    "temp": 25 - 0.08 * depth + 0.7 * slow + 0.12 * m2 * (1 + layer / 20),
                    "psal": 31 + 0.025 * depth - 0.2 * slow + 0.03 * m2,
                    "depth": depth,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_public_panel_is_invariant_to_hidden_target_values() -> None:
    observations = _observations(300)
    original = build_tide_panel(observations)
    changed = observations.copy()
    hidden = changed["layer"].isin(TARGET_LAYERS)
    changed.loc[hidden, "temp"] = -999.0
    changed.loc[hidden, "psal"] = 999.0
    rebuilt = build_tide_panel(changed)
    np.testing.assert_array_equal(original.public_values, rebuilt.public_values)
    assert original.public_feature_names == rebuilt.public_feature_names
    assert not np.array_equal(original.target_temp, rebuilt.target_temp)


def test_outer_split_purges_both_sides_and_gap_segments_are_separate() -> None:
    times = pd.date_range("2024-08-01", "2024-12-01", freq="10min", tz="UTC", inclusive="left")
    split = outer_split(times, "2024-09-01", "2024-11-01", purge_days=7)
    assert not np.any(split.training & split.validation)
    assert split.validation.sum() == 61 * 24 * 6
    assert split.purged.sum() == 75 * 24 * 6

    with_gap = times.delete(slice(100, 120))
    bounds = cadence_segments(with_gap)
    assert bounds == ((0, 100), (100, len(with_gap)))


def test_actual_depth_interpolation_uses_coordinates_not_layer_number() -> None:
    public_depth = np.array([[4.0, 20.0, 30.0, 40.0, 50.0]])
    public_value = 2.0 * public_depth + 1.0
    target_depth = np.array([[7.0, 10.0, 15.0]])
    restored = actual_depth_interpolation(public_value, public_depth, target_depth)
    np.testing.assert_allclose(restored, 2.0 * target_depth + 1.0)


def test_encoder_coverage_and_residual_transfer_have_positive_synthetic_skill() -> None:
    panel = build_tide_panel(_observations())
    training = np.ones(len(panel.times), dtype=bool)
    training[-200:] = False
    encoder = PublicFactorEncoder.fit(
        panel.public_values,
        panel.public_feature_names,
        panel.times,
        training,
        config=TideRTSConfig(factors=3),
    )
    state, observed_share = encoder.transform(panel.public_values, panel.times)
    labels = np.column_stack((0.3 * state[:, 0], -0.2 * state[:, 1], 0.1 * state[:, 2]))
    model = ResidualRegressor.fit(state, labels, training, alpha=1.0)
    prediction = model.predict(state)
    assert observed_share[-200:].min() == 1.0
    assert residual_skill_r2(labels[-200:], prediction[-200:]) > 0.95


def test_observability_reports_full_and_deficient_rank() -> None:
    transition = np.diag([0.9, 0.8])
    full = observability_diagnostics(
        transition,
        np.eye(2),
        np.ones(2),
        horizon_steps=20,
        rank_tolerance=1e-10,
    )
    deficient = observability_diagnostics(
        transition,
        np.array([[1.0, 0.0]]),
        np.ones(1),
        horizon_steps=20,
        rank_tolerance=1e-10,
    )
    assert full["rank"] == full["state_dimension"] == 2
    assert deficient["rank"] == 1


def test_m2_diagnostic_recovers_stable_phase_across_30_and_61_day_windows() -> None:
    times = pd.date_range("2024-01-01", periods=130 * 24 * 6, freq="10min", tz="UTC")
    seconds = times.as_unit("ns").asi8 / 1e9
    angle = 2 * np.pi * seconds / (12.42 * 3600)
    public = np.sin(angle)[:, None]
    target = np.sin(angle + 0.6)[:, None]
    diagnostics = m2_relationship_diagnostics(
        times,
        public,
        target,
        np.ones(len(times), dtype=bool),
        window_days=(30, 61),
    )
    assert diagnostics["30"]["targets"][0]["windows"] >= 4
    assert diagnostics["61"]["targets"][0]["windows"] >= 2
    assert diagnostics["30"]["median_best_coherence"] > 0.99
    assert diagnostics["61"]["median_phase_stability"] > 0.99


def test_rts_smoothing_and_exact_fallback() -> None:
    generator = np.random.default_rng(7)
    truth = np.zeros(240)
    for row in range(1, len(truth)):
        truth[row] = 0.97 * truth[row - 1] + generator.normal(0, 0.1)
    observations = (truth + generator.normal(0, 0.5, len(truth)))[:, None]
    result = kalman_rts_smoother(
        observations,
        np.ones_like(observations, dtype=bool),
        np.ones((1, 1)),
        np.array([0.25]),
        np.array([[0.97]]),
        np.array([[0.01]]),
    )
    assert np.mean(np.square(result.mean[:, 0] - truth)) < np.mean(
        np.square(observations[:, 0] - truth)
    )

    frozen = np.array([1.0, np.nan, 3.0])
    correction = np.array([0.5, 8.0, -0.5])
    supported = np.array([True, False, False])
    restored = exact_fallback(frozen, correction, supported)
    np.testing.assert_allclose(restored[[0, 2]], [1.5, 3.0])
    assert np.isnan(restored[1])
