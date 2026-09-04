from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from p3_wave.meaningful_learning_curve import PREFIX_FRACTIONS

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p3_station_stable_energy_state_space_v1.py"
SPEC = importlib.util.spec_from_file_location("p3_stable_energy_ssm_runner_test", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _deep_sha(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_config_byte_deep_and_structural_contract_are_compiled() -> None:
    path = ROOT / runner.CANONICAL_CONFIG_RELATIVE
    config = json.loads(path.read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == runner.EXPECTED_CONFIG_SHA256
    assert _deep_sha(config) == runner.EXPECTED_CONFIG_DEEP_SHA256
    assert [item["id"] for item in config["hypotheses"]] == [runner.HYPOTHESIS]
    assert config["validation"]["training_prefix_fractions"] == list(PREFIX_FRACTIONS)
    assert config["model"]["deterministic"] is True
    assert config["model"]["seed_metric_replica_ids"] == [20260816, 20260817, 20260818]
    assert config["model"]["expected_actual_fit_cells"] == 15
    assert config["model"]["expected_protocol_seed_metric_cells"] == 45
    assert config["model"]["expected_closed_form_solves"] == 60
    assert config["model"]["global_ridge"] == 1e-3
    assert config["model"]["station_residual_ridge"] == 1e-2
    assert config["model"]["maximum_spectral_radius"] == 0.995
    assert config["postprocess"]["fixed_persistence_weight"] == 0.2


def test_canonical_authorization_rejects_config_copy_output_and_root(
    tmp_path: Path,
) -> None:
    paths = runner._canonical_paths(ROOT)
    config, authorized = runner.authorize_entry(
        root=ROOT,
        data_dir=runner.CANONICAL_DATA_DIR,
        requested_config=paths["config"],
        requested_output=paths["output"],
    )
    assert config["experiment_id"] == "p3_station_stable_energy_state_space_v1"
    assert authorized == paths
    copied = tmp_path / "copied.json"
    copied.write_bytes(paths["config"].read_bytes())
    with pytest.raises(PermissionError, match="non-canonical config"):
        runner.authorize_entry(
            root=ROOT,
            data_dir=runner.CANONICAL_DATA_DIR,
            requested_config=copied,
            requested_output=paths["output"],
        )
    with pytest.raises(PermissionError, match="non-canonical output"):
        runner.authorize_entry(
            root=ROOT,
            data_dir=runner.CANONICAL_DATA_DIR,
            requested_config=paths["config"],
            requested_output=tmp_path / "other-output",
        )
    with pytest.raises(PermissionError, match="workspace root"):
        runner.authorize_entry(
            root=tmp_path,
            data_dir=runner.CANONICAL_DATA_DIR,
            requested_config=paths["config"],
            requested_output=paths["output"],
        )


def test_cli_direct_call_blind_order_and_fail_close_are_static() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'add_argument("--config"' not in source
    assert 'add_argument("--output"' not in source
    assert "acquire_persistent_attempt_lock" in source
    assert "safe_new_stage_path" in source
    assert "known false exact-reference check must fail closed" in source
    assert '"test_sequence_cache_value_reads": 0' in source
    assert '"upload_attempts": 0' in source
    assert source.index(
        "blind_prediction_sealed_before_validation_truth_attachment"
    ) < source.index("_load_comparator_truth_after_blind")
    assert "prefix_scored_after_all_15_blind_predictions_sealed" in source


def test_private_run_after_lock_rejects_fake_receipt_before_preflight() -> None:
    paths = runner._canonical_paths(ROOT)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    fake = {
        "created_at": "2026-08-23T00:00:00+09:00",
        "status": "ATTEMPT_CONSUMED_ONE_SHOT",
        "experiment_id": config["experiment_id"],
        "canonical_config_sha256": runner.EXPECTED_CONFIG_SHA256,
        "o_excl": True,
        "rerun_forbidden": True,
        "sha256": "0" * 64,
    }
    with pytest.raises(FileNotFoundError, match="attempt lock is absent"):
        runner._run_after_lock(
            root=ROOT,
            data_dir=runner.CANONICAL_DATA_DIR,
            config=config,
            paths=paths,
            attempt=fake,
        )


def test_persisted_receipt_replay_rejected_before_second_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = json.loads((ROOT / runner.CANONICAL_CONFIG_RELATIVE).read_text(encoding="utf-8"))
    paths = runner._canonical_paths(ROOT)
    paths["lock"] = tmp_path / "attempt.json"
    paths["claim"] = tmp_path / "claim.json"
    paths["output"] = tmp_path / "output"
    attempt = runner.acquire_persistent_attempt_lock(
        paths["lock"],
        experiment_id=config["experiment_id"],
        config_sha256=runner.EXPECTED_CONFIG_SHA256,
        created_at="2026-08-23T00:00:00+09:00",
    )
    monkeypatch.setattr(runner, "authorize_entry", lambda **_: (config, paths))
    calls = 0

    def stop_after_claim(**_: object) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("STOP_AFTER_CLAIM")

    monkeypatch.setattr(runner, "_preflight", stop_after_claim)
    with pytest.raises(RuntimeError, match="STOP_AFTER_CLAIM"):
        runner._run_after_lock(
            root=ROOT,
            data_dir=runner.CANONICAL_DATA_DIR,
            config=config,
            paths=paths,
            attempt=attempt,
        )
    assert calls == 1
    with pytest.raises(FileExistsError):
        runner._run_after_lock(
            root=ROOT,
            data_dir=runner.CANONICAL_DATA_DIR,
            config=config,
            paths=paths,
            attempt=attempt,
        )
    assert calls == 1


def test_implementation_reference_and_v5_anchor_pins_match() -> None:
    paths = runner._canonical_paths(ROOT)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    for name, path in runner._implementation_paths(ROOT, paths).items():
        assert runner.gen3.gen2.gen1.base.sha256_file(path) == config["implementation_sha256"][name]
    for name, path in runner._reference_paths(ROOT, paths).items():
        assert (
            runner.gen3.gen2.gen1.base.sha256_file(path)
            == config["reference_evidence_sha256"][name]
        )
    runner._verify_v5_anchor(ROOT, paths, config)


def test_fixed_postprocess_only_changes_registered_long_leads() -> None:
    current = np.asarray([2.0, 4.0])
    absolute = np.repeat(current[:, None] + 1.0, 6, axis=1)
    result = runner._postprocess(absolute, current)
    np.testing.assert_array_equal(result[:, :3], absolute[:, :3])
    np.testing.assert_allclose(
        result[:, 3:], np.repeat(current[:, None] + 0.8, 3, axis=1), atol=1e-12
    )


def test_append_only_targets_absent_before_one_shot() -> None:
    paths = runner._canonical_paths(ROOT)
    assert not paths["output"].exists()
    assert not paths["lock"].exists()
    assert not paths["claim"].exists()


def test_inherited_reference_false_deterministic_semantics_and_no_placeholder() -> None:
    config_text = (ROOT / runner.CANONICAL_CONFIG_RELATIVE).read_text(encoding="utf-8")
    config = json.loads(config_text)
    assert (
        config["comparator"]["reference_seed_full_prediction_exact_to_historical_frozen_oof"]
        is False
    )
    assert config["comparator"]["fail_closed_surrogate_only_if_false"] is True
    assert config["model"]["deterministic"] is True
    assert (
        "15" in config["model"]["seed_replica_rule"]
        or "deterministic" in config["model"]["seed_replica_rule"]
    )
    assert "PENDING_" not in config_text + RUNNER_PATH.read_text(encoding="utf-8")


def test_model_path_contains_no_target_or_test_api() -> None:
    helper = (ROOT / "src/p3_wave/stable_energy_state_space.py").read_text(encoding="utf-8")
    fit_body = helper[helper.index("def fit_stable_energy_state_space") :]
    fit_body = fit_body[: fit_body.index("def predict_stable_energy_state_space")]
    assert "target_hs" not in fit_body
    assert "target_delta" not in fit_body
    assert "test_" not in fit_body
    assert "forbidden_ids: np.ndarray" in fit_body
