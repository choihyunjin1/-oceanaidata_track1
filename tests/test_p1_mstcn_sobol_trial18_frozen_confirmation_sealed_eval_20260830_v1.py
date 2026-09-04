from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_mstcn_sobol_trial18_frozen_confirmation_sealed_eval_20260830_v1"
RUNNER_PATH = ROOT / "scripts" / f"run_{EXPERIMENT_ID}.py"


def _load_runner():
    name = f"test_{EXPERIMENT_ID}"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_synthetic_holdout_surface_key_regression() -> None:
    runner = _load_runner()
    result = runner.run_smoke()
    assert result["decision"] == "PASS"
    assert result["holdout_surface_key_regression"] is True
    assert result["additional_model_fits_authorized"] == 0
    assert result["historical_truth_metric_evaluations_authorized"] == 1


def test_config_freezes_zero_fit_one_evaluation_contract() -> None:
    runner = _load_runner()
    config = runner._config()
    assert config["frozen_recipe_identity"] == {
        "trial_id": "trial_18",
        "threshold": 0.8,
        "epoch": 150,
        "seeds": [20260827, 20260839, 20260863],
        "fit_count_already_completed": 6,
        "recipe_sha256": "0206dbed2d509ecb930593d4621eb53ccce2e528adbbdec09696911bffbb2d6e",
    }
    assert all(config["prohibitions"].values())


def test_preflight_verifies_failure_and_both_blind_seals_without_truth() -> None:
    runner = _load_runner()
    before = runner._verify_pins(runner._config())
    result = runner.check_only()
    after = runner._verify_pins(runner._config())
    assert result["decision"] == "PASS"
    assert all(result["checks"].values())
    assert before == after
    assert result["additional_model_fits_authorized"] == 0
    assert result["official_test_sample_submission_hidden_rows_read"] == 0


def test_exclusive_json_never_overwrites(tmp_path: Path) -> None:
    runner = _load_runner()
    path = tmp_path / "receipt.json"
    runner._exclusive_json(path, {"value": 1})
    with pytest.raises(FileExistsError):
        runner._exclusive_json(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}


def test_evaluator_has_no_fit_predict_csv_or_upload_call() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    prohibited_calls = (
        "._fit_one(",
        "._train_epoch(",
        ".predict_encoded(",
        ".to_csv(",
        ".upload(",
    )
    assert not any(token in source for token in prohibited_calls)


def test_original_failure_terminal_is_performance_blind() -> None:
    runner = _load_runner()
    config = runner._config()
    terminal = runner._load_json(
        ROOT / config["pinned_inputs"]["original_failure_terminal"]["path"]
    )
    assert runner._original_failure_matches(config, terminal)
    assert terminal["claim_scope"] == "NO_PERFORMANCE_CLAIM"
    assert terminal["completed_fit_count"] == 6
    assert terminal["official_test_sample_submission_hidden_rows_read"] == 0
