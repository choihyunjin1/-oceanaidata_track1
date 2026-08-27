from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p2_restore.normalized_curvature_residual import (
    PUBLIC_LAYERS,
    NormalizedCurvatureDesign,
    align_exact_incumbent,
    assert_safe_input_path,
    build_normalized_curvature_design,
    compute_profile_scale,
    decode_normalized_curvature,
    encode_normalized_curvature,
    evaluate_stage1_gate,
    make_stage1_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts/run_p2_normalized_curvature_residual_stage1.py"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("ncr_stage1_runner_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _feature_frame() -> pd.DataFrame:
    rows = 4
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-08-01", periods=rows, freq="h", tz="UTC"),
            "layer": [2, 3, 4, 2],
            "target": [9.6, 9.1, 8.2, 9.4],
            "baseline": [9.5, 9.0, 8.0, 9.2],
            "target_depth": [5.0, 10.0, 15.0, 5.0],
            "public_temp_range": [2.0, 0.2, 3.0, 1.5],
            "public_temp_count": [5, 4, 5, 5],
            "doy_sin": [0.1, 0.2, 0.3, 0.4],
            "doy_cos": [0.9, 0.8, 0.7, 0.6],
            "hour_sin": [0.0, 0.2, 0.4, 0.6],
            "hour_cos": [1.0, 0.8, 0.6, 0.4],
            "m2_sin": [0.2, 0.3, 0.4, 0.5],
            "m2_cos": [0.8, 0.7, 0.6, 0.5],
            "year": [2024] * rows,
            "elapsed_days": [0.0, 0.1, 0.2, 0.3],
        }
    )
    nominal = {1: 0.0, 5: 20.0, 6: 30.0, 7: 40.0, 8: 50.0}
    temperatures = {
        1: [10.5, np.nan, 10.0, 10.0],
        5: [8.5, 8.8, 7.0, 8.5],
        6: [8.0, 8.9, 6.9, 8.4],
        7: [7.8, 8.7, 6.8, 8.3],
        8: [7.5, 8.6, 6.7, 8.2],
    }
    for public_layer in PUBLIC_LAYERS:
        frame[f"temp_{public_layer}"] = temperatures[public_layer]
        frame[f"psal_{public_layer}"] = np.asarray(
            [34.0, 34.1, 34.2, 34.3], dtype=float
        ) + public_layer * 0.001
        frame[f"nominal_{public_layer}"] = nominal[public_layer]
        frame[f"depth_{public_layer}"] = nominal[public_layer] + np.asarray(
            [0.0, 0.1, -0.1, 0.2]
        )
    return frame


def test_profile_scale_endpoint_then_public_range_fallback_and_floor() -> None:
    frame = _feature_frame()
    scale = compute_profile_scale(frame, floor_c=0.5)
    assert scale[0] == pytest.approx(2.0)
    assert scale[1] == pytest.approx(0.5)
    assert scale[2] == pytest.approx(3.0)


def test_curvature_encode_decode_round_trip() -> None:
    truth = np.asarray([8.0, 9.0, 10.0])
    baseline = np.asarray([7.5, 9.2, 9.0])
    scale = np.asarray([0.5, 2.0, 4.0])
    encoded = encode_normalized_curvature(truth, baseline, scale)
    decoded = decode_normalized_curvature(encoded, baseline, scale)
    np.testing.assert_allclose(decoded, truth, rtol=0, atol=1e-12)


def test_design_is_temperature_shift_invariant_and_has_no_forbidden_features() -> None:
    original = _feature_frame()
    shifted = original.copy()
    shifted[["target", "baseline", *(f"temp_{layer}" for layer in PUBLIC_LAYERS)]] += 7.25

    design = build_normalized_curvature_design(original)
    shifted_design = build_normalized_curvature_design(shifted)

    assert tuple(design.features.columns) == tuple(shifted_design.features.columns)
    np.testing.assert_allclose(
        design.features.to_numpy(),
        shifted_design.features.to_numpy(),
        rtol=0,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        design.normalized_target, shifted_design.normalized_target, rtol=0, atol=1e-12
    )
    forbidden = {"year", "elapsed_days", "baseline", "target", "residual"}
    forbidden.update(f"temp_{layer}" for layer in (2, 3, 4))
    forbidden.update(f"psal_{layer}" for layer in (2, 3, 4))
    assert forbidden.isdisjoint(design.features.columns)


def test_stage1_split_is_strict_past_with_seven_day_embargo() -> None:
    time = pd.to_datetime(
        [
            "2024-08-24T23:59:59+09:00",
            "2024-08-25T00:00:00+09:00",
            "2024-08-31T23:00:00+09:00",
            "2024-09-01T00:00:00+09:00",
            "2024-10-31T23:59:59+09:00",
            "2024-11-01T00:00:00+09:00",
        ],
        utc=True,
    )
    split = make_stage1_split(
        time,
        validation_start="2024-09-01T00:00:00+09:00",
        validation_end="2024-11-01T00:00:00+09:00",
        embargo_days=7,
    )
    assert split.train_mask.tolist() == [True, False, False, False, False, False]
    assert split.validation_mask.tolist() == [False, False, False, True, True, False]


def test_exact_alignment_is_keyed_and_reorders_incumbent() -> None:
    keys = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-09-01T00:00:00Z", "2024-09-02T00:00:00Z"]),
            "layer": [2, 3],
        }
    )
    design = NormalizedCurvatureDesign(
        keys=keys,
        features=pd.DataFrame({"x": [1.0, 2.0]}),
        normalized_target=np.asarray([0.0, 0.0]),
        truth=np.asarray([10.0, 20.0]),
        baseline=np.asarray([10.0, 20.0]),
        profile_scale=np.asarray([1.0, 1.0]),
    )
    oof = pd.DataFrame(
        {
            "time": ["2024-09-02T00:00:00Z", "2024-09-01T00:00:00Z"],
            "layer": [3, 2],
            "block": ["2024_sep_oct", "2024_sep_oct"],
            "truth": [20.0, 10.0],
            "prediction": [19.0, 9.0],
        }
    )
    aligned = align_exact_incumbent(
        design, oof, block="2024_sep_oct", expected_rows=2
    )
    assert aligned.candidate_positions.tolist() == [0, 1]
    assert aligned.incumbent_prediction.tolist() == [9.0, 19.0]


def test_stage1_gate_requires_primary_ci_and_every_layer() -> None:
    incumbent = {
        "row_pooled_rmse_c": 0.5,
        "by_layer_rmse_c": {"2": 0.2, "3": 0.4, "4": 0.7},
    }
    candidate = {
        "row_pooled_rmse_c": 0.48,
        "by_layer_rmse_c": {"2": 0.19, "3": 0.39, "4": 0.702},
    }
    gate = {
        "candidate_minus_incumbent_row_pooled_rmse_c_max": -0.015,
        "paired_day_bootstrap_ci90_upper_c_max": -0.005,
        "candidate_minus_incumbent_each_layer_rmse_c_max": 0.003,
        "on_pass": "PASS",
        "on_fail": "FAIL",
    }
    passed = evaluate_stage1_gate(
        incumbent, candidate, {"ci90_c": [-0.03, -0.006]}, gate
    )
    assert passed["passed"] is True
    failed = evaluate_stage1_gate(
        incumbent, candidate, {"ci90_c": [-0.03, -0.004]}, gate
    )
    assert failed["passed"] is False
    assert failed["checks"]["bootstrap_ci_upper"] is False


@pytest.mark.parametrize(
    "path",
    [
        Path("test_index.csv"),
        Path("sample_submission.csv"),
        Path("artifacts/submissions/rows.parquet"),
        Path("artifacts/model_candidate_v2.parquet"),
    ],
)
def test_input_firewall_rejects_official_or_submission_like_paths(path: Path) -> None:
    with pytest.raises(RuntimeError):
        assert_safe_input_path(path)


def test_input_firewall_allows_registered_historical_oof() -> None:
    assert_safe_input_path(Path("artifacts/p2_extrapolated_soft_gate_v2/oof.parquet"))


def test_strict_prereg_seal_pins_runner_and_transitive_implementation() -> None:
    runner = _load_runner_module()
    config, verified = runner._verify_static_bundle(runner.DEFAULT_CONFIG)
    assert runner._sha256(runner.DEFAULT_CONFIG) == runner.EXPECTED_CONFIG_SHA256
    assert set(verified["implementation_pins"]) == {
        "runner",
        "normalized_curvature_module",
        "training_feature_builder",
        "feature_builder_data_dependency",
        "package_initializer",
    }
    assert verified["runtime"] == {
        "python": "3.12.10",
        "numpy": "2.3.5",
        "pandas": "3.0.1",
        "lightgbm": "4.7.0",
        "pyarrow": "25.0.1",
        "scikit-learn": "1.9.0",
    }
    assert config["seal_protocol"]["pre_numerical_import_verification"] is True


def test_runner_imports_no_numerical_or_p2_module_before_preflight() -> None:
    code = f"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('ncr_runner_isolation', {str(RUNNER_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
blocked = sorted(name for name in sys.modules if name in {{'numpy', 'pandas', 'lightgbm'}} or name.startswith('p2_restore'))
print(json.dumps(blocked))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(completed.stdout) == []
