from __future__ import annotations

import numpy as np
import pandas as pd

from p2_restore.p2_oas40_target_weighted_nonlinear_thermocline_residual_20260828_v1 import (
    bounded_fixed_correction,
    evaluate_gate,
    fit_covariate_shift_weights,
    vector_cosine,
    weighted_quantile,
)


def test_shift_weights_are_clipped_and_report_effective_sample_size() -> None:
    rows = 400
    source = pd.DataFrame(
        {"x": np.linspace(-1.0, 1.0, rows), "z": np.sin(np.linspace(0.0, 4.0, rows))}
    )
    query = pd.DataFrame(
        {"x": np.linspace(-0.8, 1.2, rows), "z": np.sin(np.linspace(0.2, 4.2, rows))}
    )
    source_time = pd.date_range("2024-01-01", periods=rows, freq="6h", tz="UTC")
    query_time = pd.date_range("2024-07-01", periods=rows, freq="6h", tz="UTC")
    fitted = fit_covariate_shift_weights(
        source,
        query,
        source_time=source_time,
        query_time=query_time,
        columns=("x", "z"),
        logistic_c=1.0,
        max_iter=300,
        seed=7,
        weight_clip=(0.25, 4.0),
        minimum_effective_sample_fraction=0.30,
        maximum_auc=0.90,
    )
    assert fitted.source_weights.shape == (rows,)
    assert np.min(fitted.source_weights) >= 0.25
    assert np.max(fitted.source_weights) <= 4.0
    assert 0.0 < fitted.effective_sample_fraction <= 1.0
    assert 0.5 <= fitted.cross_day_auc <= 1.0


def test_bounded_correction_preserves_exact_no_op() -> None:
    raw = np.array([0.4, -0.3, 0.1, -0.2])
    enabled = np.array([True, False, True, False])
    correction, receipt = bounded_fixed_correction(raw, enabled, rms_cap=0.05, p99_cap=0.20)
    assert np.array_equal(correction[~enabled], np.zeros(2))
    assert receipt["fallback_maximum_absolute_c"] == 0.0
    assert receipt["rms_c"] <= 0.05 + 1e-12
    assert receipt["p99_absolute_c"] <= 0.20 + 1e-12


def test_weighted_quantile_and_cosine() -> None:
    assert weighted_quantile(np.array([1.0, 2.0, 3.0]), np.ones(3), 0.5) == 2.0
    assert np.isclose(vector_cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 0.0)


def test_gate_requires_every_preregistered_dimension() -> None:
    thresholds = {
        "adapted_pooled_delta_rmse_max_c": -0.002,
        "paired_kst_day_bootstrap_ci90_upper_max_c": 0.001,
        "minimum_improved_folds": 2,
        "maximum_unweighted_pooled_regression_c": 0.005,
        "maximum_worst_layer_regression_c": 0.010,
        "minimum_active_share": 0.10,
        "maximum_active_share": 0.70,
        "minimum_correction_rms_c": 0.01,
        "maximum_correction_rms_c": 0.05,
        "maximum_correction_p99_c": 0.20,
        "maximum_absolute_cosine_with_alpha20_to_alpha40": 0.85,
    }
    passed = evaluate_gate(
        aggregate_delta=-0.003,
        ci90_high=-0.0001,
        fold_deltas={"a": -0.003, "b": -0.002, "c": 0.001},
        layer_deltas={"2": -0.002, "3": -0.001, "4": 0.001},
        active_share=0.4,
        correction_rms=0.02,
        correction_p99=0.10,
        cosine=0.2,
        all_shift_folds_passed=True,
        thresholds=thresholds,
    )
    assert passed["passed"] is True
    failed = evaluate_gate(
        aggregate_delta=-0.003,
        ci90_high=-0.0001,
        fold_deltas={"a": -0.003, "b": -0.002, "c": 0.001},
        layer_deltas={"2": -0.002, "3": -0.001, "4": 0.001},
        active_share=0.9,
        correction_rms=0.02,
        correction_p99=0.10,
        cosine=0.2,
        all_shift_folds_passed=True,
        thresholds=thresholds,
    )
    assert failed["passed"] is False
    assert failed["checks"]["active_share_maximum"] is False
