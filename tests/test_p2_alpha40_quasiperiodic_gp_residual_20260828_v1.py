from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from p2_restore.p2_alpha40_quasiperiodic_gp_residual_20260828_v1 import (
    FittedResidualLayer,
    QuasiPeriodicFeatureMap,
    bounded_profile_correction,
    evaluate_gate,
)


def synthetic_features(rows: int = 800) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC")
    seconds = index.asi8 / 1e9
    phase = 2.0 * np.pi * seconds / (12.42 * 3600.0)
    annual = 2.0 * np.pi * index.dayofyear.to_numpy() / 365.2425
    return pd.DataFrame(
        {
            "temp_1": 10.0 + np.sin(phase),
            "psal_1": 31.0 + 0.1 * np.cos(phase),
            "temp_missing_1": 0.0,
            "psal_missing_1": 0.0,
            "public_temp_count": 5.0,
            "public_psal_count": 5.0,
            "public_depth_span": 45.0,
            "annual_sin": np.sin(annual),
            "annual_cos": np.cos(annual),
            "m2_sin": np.sin(phase),
            "m2_cos": np.cos(phase),
        },
        index=index,
    )


def test_quasiperiodic_feature_map_is_deterministic() -> None:
    features = synthetic_features()
    first = QuasiPeriodicFeatureMap.fit(features, gamma=0.02, components=16, seed=20260828)
    second = QuasiPeriodicFeatureMap.fit(features, gamma=0.02, components=16, seed=20260828)
    first_design, first_distance = first.transform(features)
    second_design, second_distance = second.transform(features)
    assert np.array_equal(first_design, second_design)
    assert np.array_equal(first_distance, second_distance)
    assert first_design.shape[1] > len(first.columns)


def test_bayesian_residual_prediction_is_deterministic() -> None:
    features = synthetic_features()
    target = 0.03 * features["m2_sin"].to_numpy() + 0.01 * features["m2_cos"].to_numpy()
    first = FittedResidualLayer.fit(
        features,
        target,
        gamma=0.02,
        components=16,
        seed=20260830,
        uncertainty_quantile=0.8,
        max_iter=100,
        tolerance=1e-5,
    )
    second = FittedResidualLayer.fit(
        features,
        target,
        gamma=0.02,
        components=16,
        seed=20260830,
        uncertainty_quantile=0.8,
        max_iter=100,
        tolerance=1e-5,
    )
    assert all(np.array_equal(left, right) for left, right in zip(first.predict(features), second.predict(features), strict=True))


def test_correction_caps_and_exact_noop_fallback() -> None:
    raw = np.linspace(-1.0, 1.0, 1000)
    enabled = np.arange(1000) % 3 != 0
    correction, diagnostics = bounded_profile_correction(raw, enabled, rms_cap=0.05, p99_cap=0.20)
    assert np.array_equal(correction[~enabled], np.zeros((~enabled).sum()))
    assert diagnostics["rms_c"] <= 0.05 + 1e-12
    assert diagnostics["p99_absolute_c"] <= 0.20 + 1e-12


def test_gate_requires_every_preregistered_check() -> None:
    thresholds = {
        "aggregate_delta_rmse_max_c": -0.003,
        "paired_kst_day_bootstrap_ci90_upper_max_c": 0.0,
        "minimum_improved_folds": 2,
        "maximum_worst_fold_regression_c": 0.010,
        "maximum_layer_regression_c": 0.005,
        "maximum_correction_rms_c": 0.05,
        "maximum_correction_p99_c": 0.20,
    }
    passed = evaluate_gate(
        aggregate_delta=-0.004,
        ci90_high=-0.001,
        fold_deltas={"a": -0.01, "b": -0.002, "c": 0.005},
        layer_deltas={"2": -0.003, "3": 0.001, "4": -0.002},
        correction_rms=0.04,
        correction_p99=0.15,
        thresholds=thresholds,
    )
    assert passed["passed"]
    failed = evaluate_gate(
        aggregate_delta=-0.002,
        ci90_high=-0.001,
        fold_deltas={"a": -0.01, "b": -0.002, "c": 0.005},
        layer_deltas={"2": -0.003, "3": 0.001, "4": -0.002},
        correction_rms=0.04,
        correction_p99=0.15,
        thresholds=thresholds,
    )
    assert not failed["passed"]


def test_runner_has_no_official_or_submission_file_contract() -> None:
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "scripts" / "run_p2_alpha40_quasiperiodic_gp_residual_20260828_v1.py").read_text(encoding="utf-8")
    for forbidden in ("test_index.csv", "sample_submission.csv", "P2_submission.csv"):
        assert forbidden not in source
    assert source.index("sha256(prediction_path)") < source.index("truth = block_anchor")
