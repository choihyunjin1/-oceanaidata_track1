from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT / "scripts" / "run_p2_domain_invariant_vertical_curvature_20260901_v9.py"
)
CONFIG_PATH = (
    ROOT / "configs" / "experiments" / "p2_domain_invariant_vertical_curvature_20260901_v9.json"
)
SPEC = importlib.util.spec_from_file_location("p2_v9_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_config_seals_two_orthogonal_candidates_and_no_external_action() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["experiment_id"] == runner.EXPERIMENT_ID
    assert len(config["candidates"]) == 2
    assert config["operation_limits"]["maximum_fit_count"] == 6
    assert config["training"]["row_deletion"] is False
    assert config["training"]["target"].startswith("dimensionless_")
    assert config["postprocess"] == "NONE"
    assert config["result_adaptive_tuning"] is False
    assert config["source_contract"]["official_inputs_allowed"] is False
    assert config["source_contract"]["hidden_truth_allowed"] is False
    assert config["source_contract"]["submission_csv_allowed"] is False
    assert config["source_contract"]["upload_allowed"] is False


def test_robust_winsorization_never_deletes_rows() -> None:
    values = np.asarray([0.0, 0.1, -0.1, 100.0])
    clipped, receipt = runner.robust_winsorize(values, 4.0)
    assert clipped.shape == values.shape
    assert receipt["rows"] == len(values)
    assert receipt["rows_deleted"] == 0
    assert receipt["rows_clipped"] == 1
    assert np.isfinite(clipped).all()


def test_month_center_uses_global_fallback_for_unseen_month() -> None:
    train = pd.DataFrame({"a": [1.0, 3.0, 10.0, 14.0], "b": [5.0, 7.0, 9.0, 11.0]})
    query = pd.DataFrame({"a": [2.0, 20.0], "b": [6.0, 13.0]})
    train_month = np.asarray([5, 5, 6, 6])
    query_month = np.asarray([5, 9])
    centered_train, centered_query, receipt = runner.month_center_features(
        train, query, train_month, query_month
    )
    assert centered_train.shape == train.shape
    assert centered_query.shape == query.shape
    assert receipt["query_rows_using_layer_global_fallback"] == 1
    assert centered_query.loc[0, "a"] == 0.0
    assert centered_query.loc[1, "a"] == 13.5


def test_climatology_target_fallback_is_training_only() -> None:
    target = np.asarray([1.0, 3.0, 10.0, 14.0])
    centered, query_center, receipt = runner.climatology_center_target(
        target, np.asarray([5, 5, 6, 6]), np.asarray([5, 9])
    )
    assert centered.tolist() == [-1.0, 1.0, -2.0, 2.0]
    assert query_center.tolist() == [2.0, 6.5]
    assert receipt["query_rows_using_layer_global_fallback"] == 1


def test_preflight_is_zero_operation_and_deterministic() -> None:
    first = runner.preflight()
    second = runner.preflight()
    assert first == second
    assert first["status"] == "ZERO_OPERATION_PREFLIGHT_PASS"
    assert first["data_rows_read"] == 0
    assert first["model_fits"] == 0
    assert first["official_rows_read"] == 0
    assert first["submission_csv_created"] == 0
    assert first["uploads"] == 0


def test_runner_has_no_official_materialization_api() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "to_csv(" not in source
    assert "requests." not in source
    assert "selenium" not in source.lower()
    assert "--upload" not in source
