from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from p2_restore.p2_public_heave_tangent_incumbent_20260828_v1 import (
    BackgroundProfile,
    apply_heave_to_incumbent,
    estimate_public_heave,
    evaluate_gate,
    mask_validation_targets,
)


def curved_background() -> BackgroundProfile:
    return BackgroundProfile(
        season_bin=0,
        depth=np.asarray([4.0, 20.0, 31.0, 40.0, 50.0]),
        temperature=np.asarray([20.0, 13.0, 10.0, 8.5, 7.8]),
        source_layers=(1, 5, 6, 7, 8),
        source_rows={1: 100, 5: 100, 6: 100, 7: 100, 8: 100},
    )


def test_public_heave_recovers_intercept_and_displacement() -> None:
    background = curved_background()
    depth = background.depth.copy()
    interpolator = background.interpolator()
    mode = -interpolator.derivative()(depth)
    temperature = interpolator(depth) + 0.3 + 1.75 * mode
    estimate = estimate_public_heave(
        public_temperature=temperature,
        public_depth=depth,
        background=background,
        target_depth=np.asarray([7.04, 9.44, 14.74]),
        minimum_public_layers=4,
        minimum_public_span_m=30.0,
        minimum_gradient_rms_c_per_m=3e-5,
        maximum_design_condition_number=50.0,
    )
    assert estimate.supported
    assert np.isclose(estimate.intercept_c, 0.3)
    assert np.isclose(estimate.eta_m, 1.75)
    assert np.isfinite(estimate.target_mode).all()


def test_target_temp_and_psal_are_masked_together() -> None:
    time = pd.date_range("2024-09-01", periods=6, freq="10min", tz="UTC")
    frame = pd.DataFrame(
        {
            "time": np.repeat(time, 3),
            "layer": np.tile([2, 3, 4], len(time)),
            "temp": 1.0,
            "psal": 2.0,
        }
    )
    folds = {
        "fold": {
            "start": "2024-09-01T09:00:00+09:00",
            "stop": "2024-09-01T10:00:00+09:00",
        }
    }
    masked, rows = mask_validation_targets(frame, folds)
    assert rows == len(frame)
    assert masked[["temp", "psal"]].isna().all().all()


def test_missing_endpoint_is_bit_exact_incumbent_noop() -> None:
    time = pd.Timestamp("2024-01-08T00:00:00Z")
    incumbent = pd.DataFrame(
        {
            "time": [time] * 3,
            "layer": [2, 3, 4],
            "block": ["fold"] * 3,
            "reference": [18.0, 17.0, 15.0],
        }
    )
    panel = pd.DataFrame(
        {
            "temp_1": [np.nan],
            "temp_5": [13.0],
            "temp_6": [10.0],
            "temp_7": [8.5],
            "temp_8": [7.8],
            "depth_1": [4.0],
            "depth_5": [20.0],
            "depth_6": [31.0],
            "depth_7": [40.0],
            "depth_8": [50.0],
        },
        index=pd.DatetimeIndex([time]),
    )
    candidate, diagnostics = apply_heave_to_incumbent(
        incumbent,
        panel,
        {0: curved_background()},
        season_bin_days=14,
        target_depth_by_layer={2: 7.04, 3: 9.44, 4: 14.74},
        eta_cap_m=10.0,
        maximum_correction_c=0.2,
        support={
            "minimum_public_layers": 4,
            "minimum_public_span_m": 30.0,
            "minimum_gradient_rms_c_per_m": 3e-5,
            "maximum_design_condition_number": 50.0,
        },
    )
    assert np.array_equal(candidate["candidate"].to_numpy(), incumbent["reference"].to_numpy())
    assert diagnostics["enabled_rows"] == 0
    assert diagnostics["reason_counts_by_profile"]["missing_l1_or_l5_endpoint"] == 1


def test_gate_requires_active_share_and_all_safety_checks() -> None:
    thresholds = {
        "aggregate_delta_rmse_max_c": -0.003,
        "paired_kst_day_bootstrap_ci90_upper_max_c": 0.0,
        "minimum_improved_folds": 2,
        "maximum_worst_fold_regression_c": 0.005,
        "maximum_layer_regression_c": 0.005,
        "minimum_active_fraction": 0.05,
        "maximum_correction_rms_c": 0.05,
        "maximum_correction_p99_c": 0.20,
        "maximum_correction_absolute_c": 0.20,
    }
    passed = evaluate_gate(
        aggregate_delta=-0.004,
        ci90_high=-0.001,
        fold_deltas={"a": -0.01, "b": -0.002, "c": 0.003},
        layer_deltas={"2": -0.004, "3": 0.001, "4": -0.002},
        active_fraction=0.10,
        correction_rms=0.04,
        correction_p99=0.15,
        correction_maximum=0.19,
        thresholds=thresholds,
    )
    assert passed["passed"]
    failed = evaluate_gate(
        aggregate_delta=-0.004,
        ci90_high=-0.001,
        fold_deltas={"a": -0.01, "b": -0.002, "c": 0.003},
        layer_deltas={"2": -0.004, "3": 0.001, "4": -0.002},
        active_fraction=0.01,
        correction_rms=0.04,
        correction_p99=0.15,
        correction_maximum=0.19,
        thresholds=thresholds,
    )
    assert not failed["passed"]


def test_runner_has_no_deployment_file_contract() -> None:
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "scripts" / "run_p2_public_heave_tangent_incumbent_20260828_v1.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("test_index.csv", "sample_submission.csv", "P2_submission.csv"):
        assert forbidden not in source
    assert source.index("prediction_commitment.json") < source.index("load_truth_after_seal")
