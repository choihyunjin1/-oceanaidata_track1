from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/migrate_p3_era5_incomplete_manifest_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("p3_manifest_recovery", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_surface(tmp_path: Path):
    module = _module()
    canonical = tmp_path / (
        "external_data/quarantine/era5_p3_context_pretrain_v1/manifests/manifest.json"
    )
    module._write_fixture_incomplete(canonical)
    config_path, config = module._fixture_config(tmp_path, canonical)
    return module, canonical, config_path, config


def test_default_dry_run_is_eligible_only_after_process_exit(tmp_path: Path) -> None:
    module, _, config_path, _ = _fixture_surface(tmp_path)
    eligible = module.evaluate_migration(
        tmp_path,
        config_path,
        active_process_override=[],
        verify_upstream=False,
    )
    assert eligible["verdict"] == "DRY_RUN_ELIGIBLE_AFTER_PROCESS_EXIT"
    assert eligible["writes"] == 0
    active = module.evaluate_migration(
        tmp_path,
        config_path,
        active_process_override=[{"pid": 123, "matched_script": "prepare"}],
        verify_upstream=False,
    )
    assert active["verdict"] == "BLOCKED_ACTIVE_PREPARE_PROCESS"


def test_unknown_and_completed_manifest_hard_fail(tmp_path: Path) -> None:
    module, canonical, config_path, _ = _fixture_surface(tmp_path)
    payload = json.loads(canonical.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    canonical.write_text(json.dumps(payload), encoding="utf-8")
    unknown = module.evaluate_migration(
        tmp_path,
        config_path,
        active_process_override=[],
        verify_upstream=False,
    )
    assert unknown["verdict"] == "BLOCKED_UNKNOWN_OR_CHANGED_MANIFEST"

    payload["stage"] = "combine"
    payload["row_count"] = 262917
    payload["local_file"] = "external_data/quarantine/final.parquet"
    payload["file_sha256"] = "a" * 64
    canonical.write_text(json.dumps(payload), encoding="utf-8")
    completed = module.evaluate_migration(
        tmp_path,
        config_path,
        active_process_override=[],
        verify_upstream=False,
    )
    assert completed["verdict"] == "BLOCKED_COMPLETED_MANIFEST"


def test_apply_requires_acknowledgement_and_remaining_budget(tmp_path: Path) -> None:
    module, _, config_path, _ = _fixture_surface(tmp_path)
    eligible = module.evaluate_migration(
        tmp_path,
        config_path,
        active_process_override=[],
        verify_upstream=False,
    )
    with pytest.raises(module.RecoveryError, match="abnormal-termination"):
        module._apply_from_eligible(
            tmp_path,
            config_path,
            eligible,
            acknowledge_abnormal_termination=False,
            recovery_budget="remaining-two",
        )
    with pytest.raises(module.RecoveryError, match="two remaining"):
        module._apply_from_eligible(
            tmp_path,
            config_path,
            eligible,
            acknowledge_abnormal_termination=True,
            recovery_budget="remaining-one",
        )


def test_rollback_requires_explicit_abandon_of_last_automatic_resume(tmp_path: Path) -> None:
    module, canonical, config_path, config = _fixture_surface(tmp_path)
    eligible = module.evaluate_migration(
        tmp_path,
        config_path,
        active_process_override=[],
        verify_upstream=False,
    )
    module._apply_from_eligible(
        tmp_path,
        config_path,
        eligible,
        acknowledge_abnormal_termination=True,
        recovery_budget="remaining-two",
    )
    with pytest.raises(module.RecoveryError, match="manual recovery only"):
        module.rollback_migration(
            tmp_path,
            config_path,
            acknowledge_rollback=True,
            recovery_budget="remaining-one",
            active_process_override=[],
        )
    assert not canonical.exists()
    assert module.resolve_inside(tmp_path, config["recoverable_backup"]).is_file()
    restored = module.rollback_migration(
        tmp_path,
        config_path,
        acknowledge_rollback=True,
        recovery_budget="abandoned",
        active_process_override=[],
    )
    assert restored["status"] == "ROLLBACK_COMPLETE_MANUAL_ABANDON"
    assert canonical.is_file()


def test_temp_fixture_proves_raw_reuse_final_write_and_rollback() -> None:
    result = _module().run_temp_fixture_qa()
    assert result["passed"] is True
    assert result["actual_era5_path_accessed"] is False
    assert result["temporary_fixture_deleted"] is True
    assert result["checks"] == {
        "migration_fixture_eligible": True,
        "atomic_backup_preserved_known_hash": True,
        "existing_raw_reused_without_client": True,
        "same_prepare_stage_years_completed": True,
        "same_prepare_resume_download_count_zero": True,
        "same_prepare_wrote_final_manifest": True,
        "rollback_restored_exact_known_manifest": True,
        "first_resume_is_attempt_2_of_3": True,
        "one_automatic_resume_remains_after_attempt_2": True,
    }


def test_apply_surface_never_invokes_prepare_or_scientific_runner() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    apply_body = source.split("def _apply_from_eligible", 1)[1].split(
        "def apply_migration", 1
    )[0]
    assert "subprocess" not in apply_body
    assert "prepare.run" not in apply_body
    assert "--execute" not in apply_body
    assert "os.replace(canonical, backup)" in apply_body


def test_real_contract_pins_exact_known_hash_and_dry_run_default() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/experiments/p3_era5_manifest_collision_recovery_20260825_v2.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        config["known_incomplete_manifest"]["sha256"]
        == "55e5545a3cc7df6df288eb04f2f437f2a157b3968f9ea10a3c7f669223d94525"
    )
    assert config["known_incomplete_manifest"]["row_count"] == 0
    assert config["known_incomplete_manifest"]["local_file"] is None
    assert config["known_incomplete_manifest"]["file_sha256"] is None
    assert config["exact_prepare_resume_argv"][0] == ".venv-era5/Scripts/python.exe"
    assert (
        config["pinned_upstream"]["exact_prepare_python"]["sha256"]
        == "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14"
    )
    assert config["recovery_budget"]["current_attempt"] == 1
    assert config["recovery_budget"]["maximum_attempts"] == 3
    assert config["recovery_budget"]["remaining_automatic_resumes_before_migration"] == 2
    assert config["recovery_budget"]["migration_apply_consumes_automatic_resume"] == 0
    assert config["recovery_budget"]["first_post_migration_resume_attempt"] == 2
    assert (
        config["recovery_budget"]["remaining_automatic_resumes_after_first_post_migration_resume"]
        == 1
    )
    assert config["rollback_policy"]["automatic_rollback_after_first_migrated_resume_failure"] is False
    source = SCRIPT.read_text(encoding="utf-8")
    main_body = source.split("def main", 1)[1]
    assert "else:\n        result = evaluate_migration(root, config_path)" in main_body
