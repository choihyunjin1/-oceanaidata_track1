from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_p1_mstcn_checkpoint_diagnostic_20260827_v2.py"
CONFIG_PATH = ROOT / "configs" / "experiments" / "p1_mstcn_checkpoint_diagnostic_20260827_v2.json"


def _load_runner():
    name = "p1_mstcn_checkpoint_diagnostic_v2_tested"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_v2_config_is_hash_pinned_and_scientifically_identical() -> None:
    runner = _load_runner()
    assert runner._sha256(CONFIG_PATH) == runner.EXPECTED_CONFIG_SHA256
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["experiment_id"] == runner.EXPERIMENT_ID
    assert config["fixed_recipe"] == {
        "width": 512,
        "epoch": 150,
        "threshold": 0.8,
        "representation": "raw_three_seed_ensemble_mean",
        "seeds": [20260827, 20260839, 20260863],
        "blind_prediction_epochs": [120, 125, 130, 145, 150],
        "saved_state_epochs": [145, 150],
    }
    assert config["training_contract"]["source_schedule_horizon_epochs"] == 300
    assert config["training_contract"]["stop_epoch"] == 150
    assert config["training_contract"]["schedule_change_from_source"] is False
    assert config["evaluation_contract"]["truth_scored_epochs"] == [150]
    assert config["evaluation_contract"]["same_truth_oracle_diagnostic_epochs"] == [
        120,
        125,
        130,
        145,
    ]
    assert config["evaluation_contract"]["same_truth_oracle_promotion_evidence"] is False
    assert config["execution_recovery"]["scientific_change_from_v1"] is False
    assert config["execution_recovery"]["reuse_v1_model_or_optimizer_state"] is False
    assert all(config["prohibitions"].values())


def test_v2_loads_exact_v1_science_into_isolated_namespace() -> None:
    runner = _load_runner()
    implementation = runner._load_implementation()
    assert runner._sha256(runner.BASE_RUNNER_PATH) == runner.EXPECTED_BASE_RUNNER_SHA256
    assert implementation.EXPERIMENT_ID == runner.EXPERIMENT_ID
    assert implementation.CONFIG_PATH == runner.CONFIG_PATH
    assert implementation.ARTIFACT_DIR == runner.ARTIFACT_DIR
    assert implementation.ATTEMPT_LOCK == runner.ATTEMPT_LOCK
    assert implementation._atomic_torch_save is runner._safe_atomic_torch_save
    assert implementation._fit_seed_checkpoint_curve.__code__.co_filename.endswith(
        "run_p1_mstcn_checkpoint_diagnostic_20260827_v1.py"
    )
    assert runner.ARTIFACT_DIR.name.endswith("_v2")
    assert runner.BASE_EXPERIMENT_ID.endswith("_v1")


def test_windows_safe_torch_save_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    runner = _load_runner()
    target = tmp_path / "state.pt"
    payload = {
        "schema_version": "smoke.v1",
        "epoch": 145,
        "tensor": torch.arange(12, dtype=torch.float32).reshape(3, 4),
    }
    digest = runner._safe_atomic_torch_save(target, payload, torch)
    assert target.is_file()
    assert digest == runner._sha256(target)
    restored = torch.load(target, map_location="cpu", weights_only=True)
    assert restored["schema_version"] == payload["schema_version"]
    assert restored["epoch"] == payload["epoch"]
    assert torch.equal(restored["tensor"], payload["tensor"])
    assert list(tmp_path.glob(".state.pt.*.tmp")) == []
    with pytest.raises(FileExistsError):
        runner._safe_atomic_torch_save(target, payload, torch)


def test_safe_save_fsyncs_the_writable_handle() -> None:
    runner = _load_runner()
    source = inspect.getsource(runner._safe_atomic_torch_save)
    assert 'mode="w+b"' in source
    assert "torch.save(value, handle)" in source
    assert "handle.flush()" in source
    assert "os.fsync(handle.fileno())" in source
    assert '.open("rb")' not in source


def test_cli_requires_explicit_mode_and_reviewed_hash() -> None:
    runner = _load_runner()
    assert runner._parse_args(["--check-only"]).check_only
    args = runner._parse_args(["--execute", "--expected-runner-sha256", "0" * 64])
    assert args.execute
    with pytest.raises(SystemExit):
        runner._parse_args([])
    with pytest.raises(SystemExit):
        runner._parse_args(["--check-only", "--execute"])
    with pytest.raises(runner.ContractError, match="reviewed v2 launcher bytes"):
        runner.execute(expected_runner_sha256="0" * 64)


def test_check_only_is_read_only_for_fresh_or_consumed_v2_namespace() -> None:
    runner = _load_runner()
    artifact_existed = runner.ARTIFACT_DIR.exists()
    lock_existed = runner.ATTEMPT_LOCK.exists()
    result = runner.check_only()
    assert result["result"] == "PASS"
    assert result["experiment_id"] == runner.EXPERIMENT_ID
    assert result["config_sha256"] == runner.EXPECTED_CONFIG_SHA256
    assert result["runner_sha256"] == runner._sha256(RUNNER_PATH)
    assert result["q3_q4_truth_columns_read"] == 0
    assert result["official_interface_reads"] == 0
    assert result["failed_v1_state_reused"] is False
    assert result["scientific_contract_changed"] is False
    assert result["artifact_namespace_available"] is (not artifact_existed and not lock_existed)
    assert runner.ARTIFACT_DIR.exists() is artifact_existed
    assert runner.ATTEMPT_LOCK.exists() is lock_existed


def test_launcher_contains_no_protected_interface_filename_literals() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8").casefold()
    protected_fragments = ["sample_" + "submission", "test." + "csv", "submission." + "csv"]
    assert not any(fragment in source for fragment in protected_fragments)
    assert "requests." not in source
    assert "selenium" not in source
