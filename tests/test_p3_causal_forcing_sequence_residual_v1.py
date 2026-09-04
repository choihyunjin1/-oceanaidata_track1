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
RUNNER_PATH = ROOT / "scripts/run_p3_causal_forcing_sequence_residual_v1.py"
SPEC = importlib.util.spec_from_file_location("p3_causal_sequence_runner_test", RUNNER_PATH)
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
    assert config["model"]["seed_replicates"] == [20260816, 20260817, 20260818]
    assert config["training"]["expected_fit_cells"] == 45
    assert config["training"]["expected_optimizer_steps"] == 6840
    assert config["training"]["hyperparameter_search"] is False
    assert config["postprocess"]["fixed_persistence_weight"] == 0.2


def test_canonical_authorization_rejects_config_copy_and_arbitrary_output(
    tmp_path: Path,
) -> None:
    paths = runner._canonical_paths(ROOT)
    config, authorized = runner.authorize_entry(
        root=ROOT,
        data_dir=runner.CANONICAL_DATA_DIR,
        requested_config=paths["config"],
        requested_compact_cache=paths["compact_cache"],
        requested_sequence_cache=paths["sequence_cache"],
        requested_gen1=paths["gen1"],
        requested_output=paths["output"],
    )
    assert config["experiment_id"] == "p3_causal_forcing_sequence_residual_v1"
    assert authorized == paths

    copied = tmp_path / "copied-config.json"
    copied.write_bytes(paths["config"].read_bytes())
    with pytest.raises(PermissionError, match="non-canonical config"):
        runner.authorize_entry(
            root=ROOT,
            data_dir=runner.CANONICAL_DATA_DIR,
            requested_config=copied,
            requested_compact_cache=paths["compact_cache"],
            requested_sequence_cache=paths["sequence_cache"],
            requested_gen1=paths["gen1"],
            requested_output=paths["output"],
        )
    with pytest.raises(PermissionError, match="non-canonical output"):
        runner.authorize_entry(
            root=ROOT,
            data_dir=runner.CANONICAL_DATA_DIR,
            requested_config=paths["config"],
            requested_compact_cache=paths["compact_cache"],
            requested_sequence_cache=paths["sequence_cache"],
            requested_gen1=paths["gen1"],
            requested_output=tmp_path / "other-output",
        )


def test_direct_call_and_cli_are_fail_closed() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'add_argument("--config"' not in source
    assert 'add_argument("--output"' not in source
    assert 'add_argument("--compact-cache"' not in source
    assert 'add_argument("--sequence-cache"' not in source
    assert "acquire_persistent_attempt_lock" in source
    assert "safe_new_stage_path" in source
    assert "known false exact-reference check must fail closed" in source
    assert '"test_sequence_cache_value_reads": 0' in source
    assert '"upload_attempts": 0' in source


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


def test_valid_persisted_receipt_replay_is_rejected_before_second_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = json.loads((ROOT / runner.CANONICAL_CONFIG_RELATIVE).read_text(encoding="utf-8"))
    paths = runner._canonical_paths(ROOT)
    paths["lock"] = tmp_path / "attempt.json"
    paths["claim"] = tmp_path / "execution-claim.json"
    paths["output"] = tmp_path / "output"
    attempt = runner.acquire_persistent_attempt_lock(
        paths["lock"],
        experiment_id=config["experiment_id"],
        config_sha256=runner.EXPECTED_CONFIG_SHA256,
        created_at="2026-08-23T00:00:00+09:00",
    )
    monkeypatch.setattr(runner, "authorize_entry", lambda **_: (config, paths))
    preflight_calls = 0

    def stop_after_claim(**_: object) -> None:
        nonlocal preflight_calls
        preflight_calls += 1
        raise RuntimeError("STOP_AFTER_EXECUTION_CLAIM")

    monkeypatch.setattr(runner, "_preflight", stop_after_claim)
    with pytest.raises(RuntimeError, match="STOP_AFTER_EXECUTION_CLAIM"):
        runner._run_after_lock(
            root=ROOT,
            data_dir=runner.CANONICAL_DATA_DIR,
            config=config,
            paths=paths,
            attempt=attempt,
        )
    assert paths["claim"].is_file()
    assert preflight_calls == 1
    with pytest.raises(FileExistsError):
        runner._run_after_lock(
            root=ROOT,
            data_dir=runner.CANONICAL_DATA_DIR,
            config=config,
            paths=paths,
            attempt=attempt,
        )
    assert preflight_calls == 1


def test_implementation_and_reference_pins_match_current_bytes() -> None:
    config = json.loads((ROOT / runner.CANONICAL_CONFIG_RELATIVE).read_text(encoding="utf-8"))
    implementation = {
        "gen1_runner": ROOT / "scripts/run_p3_meaningful_learning_curve_v1.py",
        "corrected_split_module": ROOT / "src/p3_wave/corrected_repeated_forward.py",
        "learning_curve_module": ROOT / "src/p3_wave/meaningful_learning_curve.py",
        "sequence_module": ROOT / "src/p3_wave/sequences.py",
        "revin_preparation_module": ROOT / "src/p3_wave/revin_patch.py",
        "causal_forcing_analog_module": ROOT / "src/p3_wave/causal_forcing_analog.py",
        "causal_forcing_sequence_module": ROOT / "src/p3_wave/causal_forcing_sequence.py",
        "models_module": ROOT / "src/p3_wave/models.py",
        "persistence_shrink_module": ROOT / "src/p3_wave/persistence_shrink.py",
        "one_shot_guard_module": ROOT / "src/p3_wave/one_shot_guard.py",
        "goal_contract": ROOT / runner.CANONICAL_GOAL_RELATIVE,
        "goal_evaluator": ROOT / "src/ocean_goal/meaningful_score.py",
    }
    for name, path in implementation.items():
        assert runner.gen1.base.sha256_file(path) == config["implementation_sha256"][name]
    for name, relative in runner.REFERENCE_PATHS.items():
        assert (
            runner.gen1.base.sha256_file(ROOT / relative)
            == config["reference_evidence_sha256"][name]
        )


def test_fixed_postprocess_only_changes_registered_long_leads() -> None:
    delta = np.full((2, 6), 1.0, dtype=np.float32)
    current = np.asarray([2.0, 4.0])
    result = runner._postprocess(delta, current)
    np.testing.assert_array_equal(result[:, :3], np.repeat(current[:, None] + 1.0, 3, axis=1))
    np.testing.assert_allclose(
        result[:, 3:], np.repeat(current[:, None] + 0.8, 3, axis=1), atol=1e-12
    )


def test_append_only_targets_absent_before_one_shot() -> None:
    paths = runner._canonical_paths(ROOT)
    assert not paths["output"].exists()
    assert not paths["lock"].exists()
    assert not paths["claim"].exists()


def test_inherited_exact_reference_fact_is_explicitly_false() -> None:
    config = json.loads((ROOT / runner.CANONICAL_CONFIG_RELATIVE).read_text(encoding="utf-8"))
    assert (
        config["comparator"]["reference_seed_full_prediction_exact_to_historical_frozen_oof"]
        is False
    )
    assert config["comparator"]["fail_closed_surrogate_only_if_false"] is True
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"incumbent_reference_seed_full_prediction_exact_to_frozen_oof": False' in source


def test_no_placeholder_survives_static_seal() -> None:
    content = (ROOT / runner.CANONICAL_CONFIG_RELATIVE).read_text(encoding="utf-8")
    content += RUNNER_PATH.read_text(encoding="utf-8")
    assert "PENDING_" not in content
