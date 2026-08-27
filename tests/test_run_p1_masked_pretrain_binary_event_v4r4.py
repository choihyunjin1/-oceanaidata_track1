from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p1_masked_pretrain_binary_event_v4r4.py"
SPEC = importlib.util.spec_from_file_location("p1_masked_pretrain_v4r4_runner_test", RUNNER_PATH)
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
    assert training["runtime_prefix_selector_target_reads"] == 0
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
    protocol = config["prefix_protocol"]
    assert protocol["current_run_selector_target_reads"] == 0
    assert protocol["current_run_fold_selector_target_reads"] == 0
    assert (
        protocol["current_run_pretraining_target_tensor_materializations_before_pretraining"]
        == 0
    )
    assert protocol["exact_incumbent_prefix_row_count_and_sha_required"] is True
    assert protocol["historical_event_safe_cutoff_lineage_disclosed"] is True
    assert protocol["nominal_event_boundary_split_risk_documented"] is True
    assert protocol["expected_historical_boundary_retreat_cells"] == 4


def test_authorization_pins_all_inputs_v3_and_exact_v5_head() -> None:
    paths = runner._paths(ROOT)
    config, observed_paths, pins = runner.authorize_entry(
        root=ROOT,
        data_dir=_data_dir(),
        requested_config=paths["config"],
        requested_artifact=paths["artifact"],
    )
    assert observed_paths == paths
    assert len(pins) == len(config["immutable_inputs"])
    assert set(pins) == set(config["immutable_inputs"])
    assert pins == runner.shared._verify_input_pins(ROOT, _data_dir(), config)
    for name, expected in config["implementation_sha256"].items():
        assert len(expected) == 64, name
    assert config["implementation_sha256"]["goal_contract"] == _sha(paths["goal"])
    assert config["implementation_sha256"]["goal_evaluator"] == _sha(
        ROOT / "src/ocean_goal/meaningful_score_v3.py"
    )
    ledger = runner._verify_v5_ledger_binding(ROOT, config, paths["ledger"])
    assert ledger["event_count"] == 7
    assert ledger["head_event_sha256"] == (
        "ef6689eb9ea5e4b25c0bf3ed85bfa75411634eb6482354fa8c6cb9b71da4df3a"
    )
    guards = runner._verify_superseded_execution_guards(ROOT, config, paths)
    assert set(guards) == {"v4", "v4r2", "v4r3"}
    assert all(row["authorization_guard_is_first_statement"] for row in guards.values())
    assert all(row["attempt_lock_absent"] for row in guards.values())


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


def test_prefix_selector_has_no_target_access_and_reproduces_all_exact_cells() -> None:
    source = textwrap.dedent(inspect.getsource(runner._pinned_label_free_prefixes))
    tree = ast.parse(source)
    target_subscripts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value in {"label", "anomaly_type"}
    ]
    assert target_subscripts == []
    assert "_safe_prefix" not in source
    config = json.loads((ROOT / runner.CANONICAL_CONFIG).read_text(encoding="utf-8"))
    audit = runner._prefix_selector_audit(
        data_dir=_data_dir(), paths=runner._paths(ROOT), config=config
    )
    assert audit["target_columns_read"] == 0
    assert audit["fold_selector_target_reads"] == 0
    assert audit["cell_count"] == 15
    assert audit["exact_incumbent_prefix_row_count_and_sha_cells"] == 15
    assert audit["historical_event_boundary_retreat_cells"] == 4
    assert len(audit["cell_id_hashes"]) == 15


def test_predecessor_runners_are_statically_tombstoned_before_authorization() -> None:
    paths = runner._paths(ROOT)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    guards = runner._verify_superseded_execution_guards(ROOT, config, paths)
    assert set(guards) == {"v4", "v4r2", "v4r3"}
    assert not paths["v4_lock"].exists()
    assert not paths["v4r2_lock"].exists()
    assert not paths["v4r3_lock"].exists()


def test_shared_engine_is_bound_to_gen4r4_and_check_is_zero_side_effect() -> None:
    assert runner.shared.fit_fixed_step_temporal_event_model is (
        runner.fit_masked_pretrain_binary_event_model
    )
    assert runner.shared.predict_temporal_event_probability is (
        runner.predict_masked_pretrain_binary_probability
    )
    assert runner.shared.evaluate_learning_curve is runner.evaluate_learning_curve
    assert runner.shared._run_curve is runner._run_curve
    assert runner.shared._full_fit_models is runner._full_fit_models
    artifact = ROOT / runner.CANONICAL_ARTIFACT
    before = (
        {path.relative_to(artifact).as_posix(): _sha(path) for path in artifact.rglob("*") if path.is_file()}
        if artifact.exists()
        else None
    )
    result = runner.check_only(root=ROOT, data_dir=_data_dir())
    assert result["status"] == "CANONICAL_CHECK_ONLY_PASS"
    assert result["curve_fit_cells"] == 45
    assert result["curve_optimizer_steps"] == 5400
    assert result["pretrain_steps"] == 30
    assert result["finetune_steps"] == 90
    assert result["pretraining_label_reads"] == 0
    assert result["runtime_prefix_selector_target_reads"] == 0
    assert result["pretraining_target_tensor_materializations_before_pretraining"] == 0
    assert result["event_probability_only"] is True
    assert result["reconstruction_or_auxiliary_probability_use"] is False
    assert result["test_value_reads"] == 0
    assert result["candidate_files"] == 0
    assert result["uploads"] == 0
    assert result["v5_ledger_event_count"] == 7
    assert result["v5_ledger_semantic_upload_count"] == 0
    assert result["end_of_run_pin_surface_exact_to_authorized_start"] is True
    assert result["causal_feature_audit"]["cache_exact_to_raw_rebuild"] is True
    assert result["causal_feature_audit"]["future_value_perturbation_invariant"] is True
    assert result["prefix_selector_audit"]["target_columns_read"] == 0
    assert result["prefix_selector_audit"]["cell_count"] == 15
    assert result["attempt_lock_absent"] is True
    after = (
        {path.relative_to(artifact).as_posix(): _sha(path) for path in artifact.rglob("*") if path.is_file()}
        if artifact.exists()
        else None
    )
    assert after == before


def test_sealed_preregistration_and_preseal_receipt_if_present() -> None:
    artifact = ROOT / runner.CANONICAL_ARTIFACT
    if not artifact.exists():
        pytest.skip("Gen4r3 preregistration has not been append-only sealed yet")
    assert {path.name for path in artifact.iterdir()} == {
        "preregistration.json",
        "preseal_static_qa.json",
    }
    prereg = json.loads((artifact / "preregistration.json").read_text(encoding="utf-8"))
    preseal = json.loads((artifact / "preseal_static_qa.json").read_text(encoding="utf-8"))
    assert prereg["created_before_first_fit"] is True
    assert prereg["operation_counters_at_seal"]["curve_model_fits"] == 0
    assert prereg["current_run_prefix_selector_contract"]["target_reads"] == 0
    assert preseal["prefix_selector_audit"]["target_columns_read"] == 0
    assert preseal["prefix_selector_audit"]["cell_count"] == 15
    assert preseal["operation_counters"]["curve_model_fits"] == 0
