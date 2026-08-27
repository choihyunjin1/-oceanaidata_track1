from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p1_binary_event_tcn_dense_natural_v3.py"
SPEC = importlib.util.spec_from_file_location("p1_binary_event_v3_runner_test", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _deep_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _data_dir() -> Path:
    return ROOT / "데이터셋 원본/데이터셋_P1/P1_qc_anomaly"


def test_config_bytes_deep_structure_and_fixed_training_contract() -> None:
    path = ROOT / runner.CANONICAL_CONFIG
    config = json.loads(path.read_text(encoding="utf-8"))
    assert _sha(path) == runner.EXPECTED_CONFIG_SHA256
    assert _deep_sha(config) == runner.EXPECTED_CONFIG_DEEP_SHA256
    assert config["comparison_mode"] == "EXACT_OFFICIAL_PREFIX_REFIT"
    assert config["training"]["main_event_loss"].startswith("unweighted_BCEWithLogits")
    assert config["training"]["phase_balanced_resampling"] is False
    assert config["training"]["natural_prior_probability_correction"] is False
    assert config["training"]["expected_curve_fit_cells"] == 45
    assert config["training"]["expected_curve_optimizer_steps"] == 5400
    assert config["model"]["probability_rule"] == "sigmoid(binary_event_logit)_only"
    assert config["model"]["auxiliary_probability_union_forbidden"] is True
    assert config["on_pass"]["test_value_reads"] == 0
    assert config["on_pass"]["candidate_creation"] is False
    assert config["on_pass"]["upload"] is False


def test_authorization_pins_v3_and_all_implementations() -> None:
    paths = runner._paths(ROOT)
    config, observed_paths, pins = runner.authorize_entry(
        root=ROOT,
        data_dir=_data_dir(),
        requested_config=paths["config"],
        requested_artifact=paths["artifact"],
    )
    assert observed_paths == paths
    assert len(pins) == len(config["immutable_inputs"]) == 16
    assert config["implementation_sha256"]["goal_contract"] == _sha(paths["goal"])
    assert config["implementation_sha256"]["goal_evaluator"] == _sha(
        ROOT / "src/ocean_goal/meaningful_score_v3.py"
    )


def test_authorization_rejects_config_copy_and_arbitrary_artifact(tmp_path: Path) -> None:
    paths = runner._paths(ROOT)
    copied = tmp_path / "config.json"
    copied.write_bytes(paths["config"].read_bytes())
    with pytest.raises(PermissionError, match="non-canonical config"):
        runner.authorize_entry(
            root=ROOT,
            data_dir=_data_dir(),
            requested_config=copied,
            requested_artifact=paths["artifact"],
        )
    with pytest.raises(PermissionError, match="non-canonical artifact"):
        runner.authorize_entry(
            root=ROOT,
            data_dir=_data_dir(),
            requested_config=paths["config"],
            requested_artifact=tmp_path / "artifact",
        )


def test_shared_engine_is_bound_to_binary_only_v3_surface() -> None:
    assert runner.shared.fit_fixed_step_temporal_event_model is runner.fit_fixed_step_binary_event_model
    assert runner.shared.predict_temporal_event_probability is runner.predict_binary_event_probability
    assert runner.shared.evaluate_learning_curve is runner.evaluate_learning_curve
    assert runner.shared.load_contract is runner.load_contract
    assert runner.shared.CANONICAL_CONFIG == runner.CANONICAL_CONFIG
    assert runner.shared.HYPOTHESIS == runner.HYPOTHESIS
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "test_value_reads" not in source or "test_value_reads" in (
        ROOT / runner.CANONICAL_CONFIG
    ).read_text(encoding="utf-8")


def test_canonical_check_has_zero_test_candidate_upload_surface() -> None:
    result = runner.check_only(root=ROOT, data_dir=_data_dir())
    assert result["status"] == "CANONICAL_CHECK_ONLY_PASS"
    assert result["curve_fit_cells"] == 45
    assert result["curve_optimizer_steps"] == 5400
    assert result["binary_event_probability_only"] is True
    assert result["auxiliary_probability_union"] is False
    assert result["test_value_reads"] == 0
    assert result["candidate_files"] == 0
    assert result["uploads"] == 0
