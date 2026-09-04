from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_p2_public_sensor_influence_shrink_20260901_v10.py"
CONFIG_PATH = ROOT / "configs" / "experiments" / "p2_public_sensor_influence_shrink_20260901_v10.json"
SPEC = importlib.util.spec_from_file_location("p2_v10_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_contract_is_single_candidate_zero_fit_and_no_deletion() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["candidate"]["candidate_count"] == 1
    assert config["candidate"]["model_fit_count"] == 0
    assert config["operation_limits"]["maximum_fit_count"] == 0
    assert config["training_only_influence"]["row_deletion"] is False
    assert config["result_adaptive_tuning"] is False
    assert config["automatic_retry_count"] == 0


def test_preflight_is_byte_identical_and_zero_operation() -> None:
    first = runner.preflight()
    second = runner.preflight()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["status"] == "ZERO_OPERATION_PREFLIGHT_PASS"
    assert first["data_rows_read"] == 0
    assert first["model_fits"] == 0
    assert first["official_rows_read"] == 0
    assert first["submission_csv_created"] == 0
    assert first["uploads"] == 0


def test_training_window_mask_is_explicit_and_nonoverlapping() -> None:
    times = pd.Series(pd.to_datetime(["2024-05-01T00:00:00+09:00", "2024-09-01T00:00:00+09:00"], utc=True))
    windows = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["training_only_influence"]["registered_windows_kst"]
    assert runner._training_window_mask(times, windows).tolist() == [True, False]


def test_metric_slice_identity_is_exact_zero_delta() -> None:
    truth = np.asarray([0.0, 2.0])
    reference = np.asarray([1.0, 1.0])
    value = runner._metric_slice(truth, reference, reference.copy(), np.asarray([True, True]))
    assert value["delta_rmse"] == 0.0


def test_huber_formula_has_exact_noop_and_floor() -> None:
    endpoint = np.asarray([1.0, 1.0, 1.0])
    reference = np.asarray([2.0, 2.0, 2.0])
    score = np.asarray([5.0, 12.0, 120.0])
    weight = np.ones(3)
    active = score > 6.0
    weight[active] = np.maximum(0.5, 6.0 / score[active])
    candidate = endpoint + weight * (reference - endpoint)
    assert candidate[0] == reference[0]
    assert candidate[1] == 1.5
    assert candidate[2] == 1.5
    assert weight.min() == 0.5


def test_semantic_audit_excludes_named_closed_families() -> None:
    audit = runner.semantic_audit(runner.load_config())
    closed = set(audit["closed_or_excluded"])
    assert "learned_soft_gate_or_benefit_gate" in closed
    assert "raw_Celsius_L1_L2_residual" in closed
    assert "normalized_vertical_curvature_Ridge" in closed
    assert "row_deletion_or_truth_selected_outlier_filter" in closed


def test_runner_has_no_official_materializer_tokens() -> None:
    text = RUNNER_PATH.read_text(encoding="utf-8").lower()
    assert "test_index.csv" not in text
    assert "sample_submission" not in text
    assert "to_csv(" not in text
    assert "requests.post" not in text
