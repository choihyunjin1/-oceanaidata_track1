from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/guard_p3_era5_context_launch_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("p3_era5_guard", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_temp_fixture_proves_manifest_collision_and_crash_lock_semantics() -> None:
    result = _module().run_temp_fixture_qa()
    assert result["passed"] is True
    assert result["actual_era5_path_accessed"] is False
    assert result["temporary_fixture_deleted"] is True
    assert result["checks"] == {
        "incomplete_existing_manifest_is_collision": True,
        "low_level_atomic_writer_can_replace_incomplete_manifest": True,
        "completed_same_receipt_reuse_is_valid": True,
        "tampered_completed_surface_is_collision": True,
        "attempt_initially_available": True,
        "lock_without_result_is_crash_lock": True,
        "lock_with_result_is_consumed": True,
        "output_without_lock_is_inconsistent": True,
    }


def test_attempt_state_is_fail_closed(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "output"
    lock = tmp_path / "output.attempt.lock"
    assert module.classify_attempt_state(output, lock)["state"] == "ATTEMPT_AVAILABLE"
    lock.write_text("lock", encoding="utf-8")
    assert module.classify_attempt_state(output, lock)["state"] == "BLOCKED_CRASH_LOCK"
    output.mkdir()
    (output / "result.json").write_text("{}", encoding="utf-8")
    assert module.classify_attempt_state(output, lock)["state"] == "BLOCKED_ATTEMPT_CONSUMED"
    lock.unlink()
    assert (
        module.classify_attempt_state(output, lock)["state"]
        == "BLOCKED_INCONSISTENT_ATTEMPT_STATE"
    )


def test_final_manifest_requires_exact_registered_surface(tmp_path: Path) -> None:
    module = _module()
    quarantine = tmp_path / "external_data/quarantine/era5_p3_context_pretrain_v1"
    manifest_path = quarantine / "manifests/manifest.json"
    candidate = quarantine / "derived/era5_p3_context_pretrain_2014_2023.parquet"
    candidate.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    candidate.write_bytes(b"fixture")
    digest = module.sha256_file(candidate)
    expected = {
        "schema_version": "1.0",
        "source_id": "era5_pre2024",
        "stage_any_of": ["years", "combine"],
        "local_file": "external_data/quarantine/era5_p3_context_pretrain_v1/derived/era5_p3_context_pretrain_2014_2023.parquet",
        "final_receipt_relative_path": "derived/era5_p3_context_pretrain_2014_2023.parquet",
        "row_count": 262917,
        "observed_start": "2014-01-01T00:00:00+00:00",
        "observed_end": "2023-12-31T14:00:00+00:00",
        "selected_cells": 3,
        "selected_single_cell_year_requests": 363,
        "official_test_or_submission_accessed": False,
        "partial_file_count": 0,
    }
    payload = {
        "schema_version": "1.0",
        "source_id": "era5_pre2024",
        "stage": "combine",
        "local_file": expected["local_file"],
        "file_sha256": digest,
        "row_count": 262917,
        "observed_start": expected["observed_start"],
        "observed_end": expected["observed_end"],
        "selected_cells": [{}, {}, {}],
        "requests": {"selected_single_cell_years": [{}] * 363},
        "official_test_or_submission_accessed": False,
        "files": [
            {
                "role": "final_combined_selected_cell_hourly_parquet",
                "relative_path": expected["final_receipt_relative_path"],
                "sha256": digest,
                "row_count": 262917,
                "time_start_utc": expected["observed_start"],
                "time_end_utc": expected["observed_end"],
            }
        ],
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    observed = module.inspect_final_manifest(tmp_path, manifest_path, quarantine, expected)
    assert observed["ready"] is True
    payload["row_count"] = 0
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    observed = module.inspect_final_manifest(tmp_path, manifest_path, quarantine, expected)
    assert observed["ready"] is False
    assert "row_count" in observed["failed_checks"]


def test_check_only_receipt_requires_ready_non_null_matching_preflight() -> None:
    module = _module()
    required = {
        "mode": "check-only",
        "passed": True,
        "writes": 0,
        "model_fits": 0,
        "outcome_values_read": 0,
        "common_feature_count": 286,
        "source_quarantine_ready": True,
        "source_preflight_non_null": True,
        "source_preflight_accepted": True,
        "source_preflight_problem": "P3",
        "source_preflight_source_id": "era5_pre2024",
        "source_preflight_purpose": "pretraining",
    }
    manifest = {"sha256": "a" * 64, "file_sha256": "b" * 64, "row_count": 262917}
    observed = {
        "mode": "check-only",
        "passed": True,
        "writes": 0,
        "model_fits": 0,
        "outcome_values_read": 0,
        "common_feature_count": 286,
        "source_quarantine_ready": True,
        "source_preflight": {
            "accepted": True,
            "problem": "P3",
            "source_id": "era5_pre2024",
            "purpose": "pretraining",
            "manifest_sha256": "a" * 64,
            "candidate_sha256": "b" * 64,
            "row_count": 262917,
        },
    }
    assert module.validate_check_only_receipt(observed, required, manifest)["passed"] is True
    observed["source_preflight"] = None
    failed = module.validate_check_only_receipt(observed, required, manifest)
    assert failed["passed"] is False
    assert "source_preflight_non_null" in failed["failed_checks"]


def test_guard_subprocess_surface_is_check_only_and_never_execute() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    function = source.split("def run_fixed_check_only", 1)[1].split(
        "def validate_check_only_receipt", 1
    )[0]
    assert '"--check-only"' in function
    assert '"--execute"' not in function
    assert function.count("subprocess.run(") == 1


def test_guard_config_is_fail_closed() -> None:
    config = json.loads(
        (ROOT / "configs/experiments/p3_era5_guarded_launch_20260825_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["required_check_only_receipt"]["source_quarantine_ready"] is True
    assert config["required_check_only_receipt"]["source_preflight_non_null"] is True
    assert config["expected_final_manifest"]["row_count"] == 262917
    assert config["expected_final_manifest"]["selected_single_cell_year_requests"] == 363
    assert config["expected_final_manifest"]["partial_file_count"] == 0
