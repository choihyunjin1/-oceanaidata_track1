from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p1_masked_pretrain_binary_event_v4r2.py"
SPEC = importlib.util.spec_from_file_location("p1_masked_pretrain_v4r2_runner_test", RUNNER_PATH)
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


def test_config_bytes_deep_and_two_stage_contract() -> None:
    path = ROOT / runner.CANONICAL_CONFIG
    config = json.loads(path.read_text(encoding="utf-8"))
    assert _sha(path) == runner.EXPECTED_CONFIG_SHA256
    assert _deep_sha(config) == runner.EXPECTED_CONFIG_DEEP_SHA256
    training = config["training"]
    assert training["optimizer_steps_per_cell"] == 120
    assert training["pretrain_steps_per_cell"] == 30
    assert training["finetune_steps_per_cell"] == 90
    assert training["pretrain_steps_per_cell"] + training["finetune_steps_per_cell"] == 120
    assert training["expected_curve_pretrain_steps"] == 1350
    assert training["expected_curve_finetune_steps"] == 4050
    assert training["pretraining_label_reads"] == 0
    assert config["model"]["probability_rule"] == "sigmoid(binary_event_logit)_only"
    assert config["model"]["reconstruction_or_auxiliary_probability_use_forbidden"] is True
    assert config["features"]["out_of_prefix_context_zero_masked"] is True
    assert config["features"]["raw_prefix_rebuilt_before_every_fit_cell"] is True
    assert config["features"]["forward_centered_or_terminal_run_features"] == []
    assert tuple(config["features"]["selected_numeric_columns"]) == (
        runner.CAUSAL_FEATURE_COLUMNS
    )
    assert config["on_pass"]["test_value_reads"] == 0
    assert config["on_pass"]["candidate_creation"] is False
    assert config["on_pass"]["upload"] is False


def test_authorization_pins_all_inputs_v3_and_exact_v5_head() -> None:
    paths = runner._paths(ROOT)
    config, observed_paths, pins = runner.authorize_entry(
        root=ROOT,
        data_dir=_data_dir(),
        requested_config=paths["config"],
        requested_artifact=paths["artifact"],
    )
    assert observed_paths == paths
    assert len(pins) == len(config["immutable_inputs"]) + 1
    for name, expected in config["implementation_sha256"].items():
        assert len(expected) == 64, name
    assert config["implementation_sha256"]["goal_contract"] == _sha(paths["goal"])
    assert config["implementation_sha256"]["goal_evaluator"] == _sha(
        ROOT / "src/ocean_goal/meaningful_score_v3.py"
    )
    ledger = pins["canonical_v5_ledger_semantic_binding"]
    assert ledger["event_count"] == 7
    assert ledger["head_event_sha256"] == (
        "ef6689eb9ea5e4b25c0bf3ed85bfa75411634eb6482354fa8c6cb9b71da4df3a"
    )


def test_authorization_rejects_noncanonical_paths(tmp_path: Path) -> None:
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


def test_authorization_rejects_clone_wrong_data_and_hardlink(tmp_path: Path) -> None:
    paths = runner._paths(ROOT)
    with pytest.raises(PermissionError, match="workspace clone"):
        runner.authorize_entry(
            root=tmp_path,
            data_dir=_data_dir(),
            requested_config=paths["config"],
            requested_artifact=paths["artifact"],
        )
    with pytest.raises(PermissionError, match="data directory"):
        runner.authorize_entry(
            root=ROOT,
            data_dir=tmp_path,
            requested_config=paths["config"],
            requested_artifact=paths["artifact"],
        )
    if os.name == "nt":
        original = tmp_path / "anchor.txt"
        alias = tmp_path / "anchor_alias.txt"
        original.write_text("identity", encoding="utf-8")
        os.link(original, alias)
        with pytest.raises(PermissionError, match="hardlinked"):
            runner._assert_single_link(original, role="synthetic")
    with pytest.raises(PermissionError, match="non-canonical artifact"):
        runner.authorize_entry(
            root=ROOT,
            data_dir=_data_dir(),
            requested_config=paths["config"],
            requested_artifact=tmp_path / "artifact",
        )


def test_shared_engine_is_bound_to_gen4_and_check_is_zero_side_effect() -> None:
    assert runner.shared.fit_fixed_step_temporal_event_model is (
        runner.fit_masked_pretrain_binary_event_model
    )
    assert runner.shared.predict_temporal_event_probability is (
        runner.predict_masked_pretrain_binary_probability
    )
    assert runner.shared.evaluate_learning_curve is runner.evaluate_learning_curve
    assert runner.shared._run_curve is runner._run_curve
    assert runner.shared._full_fit_models is runner._full_fit_models
    result = runner.check_only(root=ROOT, data_dir=_data_dir())
    assert result["status"] == "CANONICAL_CHECK_ONLY_PASS"
    assert result["curve_fit_cells"] == 45
    assert result["curve_optimizer_steps"] == 5400
    assert result["pretrain_steps"] == 30
    assert result["finetune_steps"] == 90
    assert result["pretraining_label_reads"] == 0
    assert result["event_probability_only"] is True
    assert result["reconstruction_or_auxiliary_probability_use"] is False
    assert result["test_value_reads"] == 0
    assert result["candidate_files"] == 0
    assert result["uploads"] == 0
    assert result["v5_ledger_event_count"] == 7
    assert result["v5_ledger_semantic_upload_count"] == 0
    assert result["causal_feature_audit"]["cache_exact_to_raw_rebuild"] is True
    assert result["causal_feature_audit"]["future_value_perturbation_invariant"] is True
    assert result["artifact_absent"] is False
    assert result["attempt_lock_absent"] is True
    artifact = ROOT / runner.CANONICAL_ARTIFACT
    assert {path.name for path in artifact.iterdir()} == {
        "preregistration.json",
        "preseal_static_qa.json",
    }
    prereg = json.loads((artifact / "preregistration.json").read_text(encoding="utf-8"))
    assert prereg["created_before_first_fit"] is True
    assert prereg["operation_counters_at_seal"]["curve_model_fits"] == 0
