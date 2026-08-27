"""Guarded, recoverable migration for one exact stale P3 ERA5 manifest.

Dry-run is the default.  Apply is fail-closed and only relocates the exact known
incomplete manifest after the prepare process has exited abnormally, the
context-transfer attempt remains unused, and the one allowed prepare-resume
budget token is explicitly acknowledged.  The current contract records attempt
1/3 with two automatic resumes remaining.  It never reads or modifies ERA5 value
files and never starts either the prepare or scientific runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class RecoveryError(RuntimeError):
    """A recovery precondition or append-only provenance check failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryError(f"JSON root must be an object: {path}")
    return value


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def resolve_inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not _inside(path, root):
        raise RecoveryError(f"configured recovery path escapes repository: {value}")
    return path


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def exclusive_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def verify_pinned_upstream(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for label, receipt in config["pinned_upstream"].items():
        path = resolve_inside(root, str(receipt["path"]))
        if not path.is_file():
            raise RecoveryError(f"pinned upstream file is absent: {label}")
        actual = sha256_file(path)
        if actual != str(receipt["sha256"]).casefold():
            raise RecoveryError(f"pinned upstream hash drift: {label}")
        observed[str(label)] = actual
    return observed


def find_active_prepare_processes(script_basename: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    scan_errors: list[int] = []
    current_pid = os.getpid()
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            info = process.info
            pid = int(info["pid"])
            if pid == current_pid:
                continue
            command = [str(value) for value in (info.get("cmdline") or [])]
            if any(Path(value).name.casefold() == script_basename.casefold() for value in command):
                matches.append(
                    {
                        "pid": pid,
                        "process_name": str(info.get("name") or ""),
                        "matched_script": script_basename,
                    }
                )
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            # An inaccessible non-matching system process is not evidence of the
            # named Python prepare runner. Record it without mutating anything.
            try:
                scan_errors.append(int(process.pid))
            except (AttributeError, TypeError, ValueError):
                pass
    return {
        "active_matches": sorted(matches, key=lambda item: item["pid"]),
        "active_match_count": len(matches),
        "read_only_process_scan": True,
        "uninspectable_process_count": len(scan_errors),
    }


def classify_attempt_state(output: Path, lock: Path) -> dict[str, Any]:
    output_exists = output.exists()
    lock_exists = lock.is_file()
    result_exists = (output / "result.json").is_file()
    state = (
        "ATTEMPT_UNUSED"
        if not output_exists and not lock_exists
        else "ATTEMPT_NOT_UNUSED"
    )
    return {
        "state": state,
        "output_exists": output_exists,
        "attempt_lock_exists": lock_exists,
        "result_exists": result_exists,
    }


def inspect_known_incomplete_manifest(
    manifest_path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if not manifest_path.is_file():
        return {
            "exists": False,
            "exact_known_incomplete": False,
            "completed_or_claims_values": False,
            "failed_checks": ["manifest_exists"],
        }
    payload = read_json(manifest_path)
    requests = payload.get("requests")
    year_requests = (
        requests.get("selected_single_cell_years")
        if isinstance(requests, Mapping)
        else None
    )
    selected = payload.get("selected_cells")
    files = payload.get("files")
    checksums = payload.get("checksums_sha256")
    actual_hash = sha256_file(manifest_path)
    checks = {
        "exact_sha256": actual_hash == expected["sha256"],
        "schema_version": payload.get("schema_version") == expected["schema_version"],
        "source_id": payload.get("source_id") == expected["source_id"],
        "stage": payload.get("stage") is expected["stage"],
        "row_count_zero": payload.get("row_count") == expected["row_count"],
        "local_file_null": payload.get("local_file") is expected["local_file"],
        "file_sha256_null": payload.get("file_sha256") is expected["file_sha256"],
        "observed_start_null": payload.get("observed_start") is expected["observed_start"],
        "observed_end_null": payload.get("observed_end") is expected["observed_end"],
        "selected_cells_empty": isinstance(selected, list)
        and len(selected) == expected["selected_cells"],
        "year_requests_empty": isinstance(year_requests, list)
        and len(year_requests) == expected["selected_single_cell_year_requests"],
        "files_empty": isinstance(files, list) and len(files) == expected["files"],
        "checksums_empty": isinstance(checksums, Mapping)
        and len(checksums) == expected["checksums"],
        "official_access_false": payload.get("official_test_or_submission_accessed")
        is expected["official_test_or_submission_accessed"],
    }
    completed_or_claims_values = bool(
        payload.get("stage") in {"years", "combine"}
        or (isinstance(payload.get("row_count"), int) and payload["row_count"] > 0)
        or payload.get("local_file")
        or payload.get("file_sha256")
    )
    return {
        "exists": True,
        "sha256": actual_hash,
        "exact_known_incomplete": all(checks.values()),
        "completed_or_claims_values": completed_or_claims_values,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "stage": payload.get("stage"),
        "row_count": payload.get("row_count"),
        "local_file": payload.get("local_file"),
        "file_sha256": payload.get("file_sha256"),
    }


def exact_prepare_resume_argv(root: Path, config: Mapping[str, Any]) -> list[str]:
    values = list(config["exact_prepare_resume_argv"])
    return [
        str(resolve_inside(root, values[0])),
        str(resolve_inside(root, values[1])),
        *[str(value) for value in values[2:]],
    ]


def rollback_argv(root: Path) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        str((root / "scripts/migrate_p3_era5_incomplete_manifest_v1.py").resolve()),
        "--root",
        str(root.resolve()),
        "--rollback",
        "--acknowledge-abandon-remaining-automatic-resume",
        "--recovery-budget",
        "abandoned",
    ]


def evaluate_migration(
    root: Path,
    config_path: Path,
    *,
    active_process_override: Sequence[Mapping[str, Any]] | None = None,
    verify_upstream: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path)
    base: dict[str, Any] = {
        "schema_version": "p3.era5_manifest_collision_recovery.dry_run.v2",
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "mode": "dry-run",
        "writes": 0,
        "era5_value_file_reads": 0,
        "process_mutations": 0,
        "prepare_runner_invocations": 0,
        "scientific_runner_invocations": 0,
        "recovery_budget": config["recovery_budget"],
    }
    try:
        upstream = verify_pinned_upstream(root, config) if verify_upstream else {}
    except (RecoveryError, OSError) as exc:
        return {**base, "verdict": "BLOCKED_UNKNOWN_OR_CHANGED_MANIFEST", "reason": str(exc)}
    canonical = resolve_inside(root, str(config["canonical_manifest"]))
    backup = resolve_inside(root, str(config["recoverable_backup"]))
    apply_receipt = resolve_inside(root, str(config["apply_receipt_dir"]))
    rollback_receipt = resolve_inside(root, str(config["rollback_receipt_dir"]))
    output = resolve_inside(root, str(config["attempt_output"]))
    attempt_lock = resolve_inside(root, str(config["attempt_lock"]))
    manifest = inspect_known_incomplete_manifest(canonical, config["known_incomplete_manifest"])
    attempt = classify_attempt_state(output, attempt_lock)
    if active_process_override is None:
        process = find_active_prepare_processes(
            str(config["process_guard"]["matched_script_basename"])
        )
    else:
        matches = [dict(item) for item in active_process_override]
        process = {
            "active_matches": matches,
            "active_match_count": len(matches),
            "read_only_process_scan": False,
            "fixture_override": True,
        }
    plan = {
        "canonical_manifest": str(canonical),
        "recoverable_backup": str(backup),
        "apply_receipt_dir": str(apply_receipt),
        "rollback_receipt_dir": str(rollback_receipt),
        "exact_prepare_resume_argv": exact_prepare_resume_argv(root, config),
        "rollback_argv": rollback_argv(root),
        "migration_consumes_prepare_resume_budget": False,
        "first_post_migration_resume_is_attempt": config["recovery_budget"][
            "first_post_migration_resume_attempt"
        ],
        "remaining_automatic_resumes_after_first_post_migration_resume": config[
            "recovery_budget"
        ]["remaining_automatic_resumes_after_first_post_migration_resume"],
        "automatic_rollback_after_first_migrated_resume_failure": False,
    }
    if manifest["completed_or_claims_values"]:
        verdict = "BLOCKED_COMPLETED_MANIFEST"
        reason = "completed or value-bearing manifest must never be migrated"
    elif not manifest["exact_known_incomplete"]:
        verdict = "BLOCKED_UNKNOWN_OR_CHANGED_MANIFEST"
        reason = "manifest is not the exact known incomplete collision surface"
    elif attempt["state"] != "ATTEMPT_UNUSED":
        verdict = "BLOCKED_ATTEMPT_STATE"
        reason = "context-transfer attempt/output is not unused"
    elif backup.exists() or apply_receipt.exists() or rollback_receipt.exists():
        verdict = "BLOCKED_BACKUP_OR_RECEIPT_COLLISION"
        reason = "recovery backup or append-only receipt path already exists"
    elif process["active_match_count"] != config["process_guard"][
        "apply_requires_active_matches"
    ]:
        verdict = "BLOCKED_ACTIVE_PREPARE_PROCESS"
        reason = "prepare process is still active; abnormal termination is not established"
    else:
        verdict = "DRY_RUN_ELIGIBLE_AFTER_PROCESS_EXIT"
        reason = "all exact-manifest, unused-attempt, process-exit, and collision checks passed"
    return {
        **base,
        "verdict": verdict,
        "reason": reason,
        "pinned_upstream_sha256": upstream,
        "manifest_state": manifest,
        "attempt_state": attempt,
        "prepare_process_state": process,
        "plan": plan,
    }


def _apply_from_eligible(
    root: Path,
    config_path: Path,
    evaluation: Mapping[str, Any],
    *,
    acknowledge_abnormal_termination: bool,
    recovery_budget: str | None,
) -> dict[str, Any]:
    config = read_json(config_path)
    if evaluation.get("verdict") != "DRY_RUN_ELIGIBLE_AFTER_PROCESS_EXIT":
        raise RecoveryError(f"apply refused from verdict: {evaluation.get('verdict')}")
    if not acknowledge_abnormal_termination:
        raise RecoveryError("apply requires explicit abnormal-termination acknowledgement")
    if recovery_budget != config["recovery_budget"]["apply_required_budget_token"]:
        raise RecoveryError("apply requires exactly two remaining automatic prepare resumes")
    canonical = resolve_inside(root, str(config["canonical_manifest"]))
    backup = resolve_inside(root, str(config["recoverable_backup"]))
    receipt_dir = resolve_inside(root, str(config["apply_receipt_dir"]))
    if receipt_dir.exists() or backup.exists():
        raise FileExistsError("append-only apply receipt or recovery backup already exists")
    receipt_dir.mkdir(parents=True, exist_ok=False)
    backup.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat()
    intent = {
        "schema_version": "p3.era5_manifest_collision_recovery.apply_intent.v1",
        "created_at_kst": timestamp,
        "status": "APPLY_INTENT_SEALED_BEFORE_MOVE",
        "before_sha256": config["known_incomplete_manifest"]["sha256"],
        "canonical_manifest": str(canonical),
        "recoverable_backup": str(backup),
        "automation_attempt_before_migration": config["recovery_budget"]["current_attempt"],
        "automatic_resumes_before_migration": config["recovery_budget"][
            "remaining_automatic_resumes_before_migration"
        ],
        "migration_consumes_automatic_resume": 0,
        "exact_prepare_resume_argv": exact_prepare_resume_argv(root, config),
        "rollback_argv": rollback_argv(root),
    }
    exclusive_json(receipt_dir / "intent.json", intent)
    os.replace(canonical, backup)
    if canonical.exists() or sha256_file(backup) != config["known_incomplete_manifest"]["sha256"]:
        raise RecoveryError("atomic manifest relocation did not preserve the exact known bytes")
    complete = {
        **intent,
        "status": "APPLY_COMPLETE_PREPARE_RESUME_AUTHORIZED",
        "completed_at_kst": datetime.now().astimezone().isoformat(),
        "canonical_absent": True,
        "backup_sha256": sha256_file(backup),
        "automatic_resumes_after_migration": config["recovery_budget"][
            "remaining_automatic_resumes_before_migration"
        ],
        "first_post_migration_resume_attempt": config["recovery_budget"][
            "first_post_migration_resume_attempt"
        ],
        "remaining_automatic_resumes_after_first_post_migration_resume": config[
            "recovery_budget"
        ]["remaining_automatic_resumes_after_first_post_migration_resume"],
    }
    exclusive_json(receipt_dir / "complete.json", complete)
    return complete


def apply_migration(
    root: Path,
    config_path: Path,
    *,
    acknowledge_abnormal_termination: bool,
    recovery_budget: str | None,
) -> dict[str, Any]:
    evaluation = evaluate_migration(root, config_path)
    return _apply_from_eligible(
        root.resolve(),
        config_path,
        evaluation,
        acknowledge_abnormal_termination=acknowledge_abnormal_termination,
        recovery_budget=recovery_budget,
    )


def rollback_migration(
    root: Path,
    config_path: Path,
    *,
    acknowledge_rollback: bool,
    recovery_budget: str | None,
    active_process_override: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not acknowledge_rollback:
        raise RecoveryError(
            "rollback requires explicit abandonment of the remaining automatic resume"
        )
    root = root.resolve()
    config = read_json(config_path)
    if recovery_budget != config["recovery_budget"]["rollback_required_budget_token"]:
        raise RecoveryError("rollback is manual recovery only after budget abandonment")
    if not config["rollback_policy"][
        "allowed_only_after_explicit_abandon_of_remaining_automatic_resume"
    ]:
        raise RecoveryError("rollback policy does not authorize manual abandonment")
    verify_pinned_upstream(root, config)
    process = (
        find_active_prepare_processes(str(config["process_guard"]["matched_script_basename"]))
        if active_process_override is None
        else {"active_match_count": len(active_process_override)}
    )
    if process["active_match_count"] != 0:
        raise RecoveryError("rollback refused while prepare process is active")
    canonical = resolve_inside(root, str(config["canonical_manifest"]))
    backup = resolve_inside(root, str(config["recoverable_backup"]))
    output = resolve_inside(root, str(config["attempt_output"]))
    lock = resolve_inside(root, str(config["attempt_lock"]))
    apply_receipt = resolve_inside(root, str(config["apply_receipt_dir"]))
    rollback_receipt = resolve_inside(root, str(config["rollback_receipt_dir"]))
    if canonical.exists():
        payload = read_json(canonical)
        if payload.get("stage") in {"years", "combine"} or payload.get("row_count", 0) > 0:
            raise RecoveryError("rollback must never overwrite a completed canonical manifest")
        raise RecoveryError("rollback requires canonical manifest to be absent")
    if classify_attempt_state(output, lock)["state"] != "ATTEMPT_UNUSED":
        raise RecoveryError("rollback refused because context-transfer attempt is not unused")
    if not (apply_receipt / "intent.json").is_file() or not (
        apply_receipt / "complete.json"
    ).is_file():
        raise RecoveryError("verified apply receipts are required for rollback")
    if not backup.is_file() or sha256_file(backup) != config["known_incomplete_manifest"]["sha256"]:
        raise RecoveryError("rollback backup is absent or changed")
    if rollback_receipt.exists():
        raise FileExistsError("append-only rollback receipt already exists")
    rollback_receipt.mkdir(parents=True, exist_ok=False)
    intent = {
        "schema_version": "p3.era5_manifest_collision_recovery.rollback_intent.v1",
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "status": "ROLLBACK_INTENT_SEALED_BEFORE_MOVE",
        "backup_sha256": sha256_file(backup),
        "canonical_manifest": str(canonical),
        "recoverable_backup": str(backup),
    }
    exclusive_json(rollback_receipt / "intent.json", intent)
    os.replace(backup, canonical)
    if backup.exists() or sha256_file(canonical) != config["known_incomplete_manifest"]["sha256"]:
        raise RecoveryError("rollback did not restore exact known manifest bytes")
    complete = {
        **intent,
        "status": "ROLLBACK_COMPLETE_MANUAL_ABANDON",
        "completed_at_kst": datetime.now().astimezone().isoformat(),
        "canonical_sha256": sha256_file(canonical),
        "automatic_resume_budget_state": "abandoned",
    }
    exclusive_json(rollback_receipt / "complete.json", complete)
    return complete


@contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _prepare_module():
    path = ROOT / "scripts/prepare_p3_era5_pretrain.py"
    spec = importlib.util.spec_from_file_location("prepare_p3_era5_pretrain_fixture", path)
    if spec is None or spec.loader is None:
        raise RecoveryError("fixed prepare runner cannot be imported for fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_config(temp_root: Path, incomplete: Path) -> tuple[Path, dict[str, Any]]:
    config = {
        "schema_version": "fixture.v2",
        "canonical_manifest": "external_data/quarantine/era5_p3_context_pretrain_v1/manifests/manifest.json",
        "recoverable_backup": "external_data/quarantine/era5_p3_context_pretrain_v1/manifests/recovery/manifest.incomplete.fixture.json",
        "apply_receipt_dir": "artifacts/recovery/apply_receipt",
        "rollback_receipt_dir": "artifacts/recovery/rollback_receipt",
        "attempt_output": "artifacts/context_output",
        "attempt_lock": "artifacts/context_output.attempt.lock",
        "known_incomplete_manifest": {
            "sha256": sha256_file(incomplete),
            "schema_version": "1.0",
            "source_id": "era5_pre2024",
            "stage": None,
            "row_count": 0,
            "local_file": None,
            "file_sha256": None,
            "observed_start": None,
            "observed_end": None,
            "selected_cells": 0,
            "selected_single_cell_year_requests": 0,
            "files": 0,
            "checksums": 0,
            "official_test_or_submission_accessed": False,
        },
        "pinned_upstream": {},
        "exact_prepare_resume_argv": [
            ".venv-era5/Scripts/python.exe",
            "scripts/prepare_p3_era5_pretrain.py",
            "--stage",
            "years",
            "--execute-download",
        ],
        "process_guard": {
            "matched_script_basename": "prepare_p3_era5_pretrain.py",
            "apply_requires_active_matches": 0,
        },
        "recovery_budget": {
            "current_attempt": 1,
            "maximum_attempts": 3,
            "remaining_automatic_resumes_before_migration": 2,
            "migration_apply_consumes_automatic_resume": 0,
            "first_post_migration_resume_attempt": 2,
            "remaining_automatic_resumes_after_first_post_migration_resume": 1,
            "apply_required_budget_token": "remaining-two",
            "rollback_required_budget_token": "abandoned",
        },
        "rollback_policy": {
            "automatic_rollback_after_first_migrated_resume_failure": False,
            "allowed_only_after_explicit_abandon_of_remaining_automatic_resume": True,
            "manual_recovery_only": True,
        },
    }
    # Fixture argv paths must exist without copying or reading actual ERA5 data.
    (temp_root / ".venv-era5/Scripts").mkdir(parents=True, exist_ok=True)
    (temp_root / ".venv-era5/Scripts/python.exe").write_bytes(b"fixture-python")
    (temp_root / "scripts").mkdir(parents=True, exist_ok=True)
    (temp_root / "scripts/prepare_p3_era5_pretrain.py").write_text(
        "# fixture path only\n", encoding="utf-8"
    )
    config_path = temp_root / "fixture_recovery.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path, config


def _write_fixture_incomplete(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "source_id": "era5_pre2024",
        "stage": None,
        "row_count": 0,
        "local_file": None,
        "file_sha256": None,
        "observed_start": None,
        "observed_end": None,
        "selected_cells": [],
        "requests": {"selected_single_cell_years": []},
        "files": [],
        "checksums_sha256": {},
        "official_test_or_submission_accessed": False,
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def run_temp_fixture_qa() -> dict[str, Any]:
    from p3_wave.era5_pretrain_data import (
        COMBINED_FILE_NAME,
        DYNAMIC_VARIABLES,
        STATIONS,
        FileReceipt,
        QuarantineLayout,
        SelectedCell,
        build_year_plan,
        sha256_file as source_sha256,
    )

    prepare = _prepare_module()
    with tempfile.TemporaryDirectory(prefix="p3-era5-manifest-recovery-fixture-") as temporary:
        fixture_root = Path(temporary).resolve()
        success_root = fixture_root / "success"
        canonical = success_root / (
            "external_data/quarantine/era5_p3_context_pretrain_v1/manifests/manifest.json"
        )
        _write_fixture_incomplete(canonical)
        config_path, config = _fixture_config(success_root, canonical)
        eligible = evaluate_migration(
            success_root,
            config_path,
            active_process_override=[],
            verify_upstream=False,
        )
        applied = _apply_from_eligible(
            success_root,
            config_path,
            eligible,
            acknowledge_abnormal_termination=True,
            recovery_budget="remaining-two",
        )
        layout = QuarantineLayout.from_repo_root(success_root)
        layout.ensure()
        combined = layout.derived / COMBINED_FILE_NAME
        combined.write_bytes(b"synthetic-combined-fixture")
        combined_receipt = FileReceipt(
            request_id="combined_era5_p3_2014_2023",
            role="final_combined_selected_cell_hourly_parquet",
            relative_path=f"derived/{COMBINED_FILE_NAME}",
            bytes=combined.stat().st_size,
            sha256=source_sha256(combined),
            time_start_utc="2014-01-01T00:00:00+00:00",
            time_end_utc="2023-12-31T14:00:00+00:00",
            row_count=262_917,
        )
        selections = {
            station: SelectedCell(
                station=station,
                station_latitude=point.latitude,
                station_longitude=point.longitude,
                latitude=round(point.latitude * 4) / 4,
                longitude=round(point.longitude * 4) / 4,
                distance_km=0.0,
                mean_land_sea_mask=0.0,
                finite_fraction={name: 1.0 for name in DYNAMIC_VARIABLES},
            )
            for station, point in STATIONS.items()
        }
        first_request = build_year_plan(selections)[0]
        first_raw = layout.raw_path(first_request)
        first_raw.parent.mkdir(parents=True, exist_ok=True)
        first_raw.write_bytes(b"already-complete-raw-fixture")

        def forbidden_client():
            raise AssertionError("raw reuse fixture must not construct a CDS client")

        _, downloaded = prepare._existing_or_download(
            first_request,
            layout=layout,
            execute_download=True,
            client_factory=forbidden_client,
        )
        original_run_years = prepare._run_years

        def reused_years(**_: Any):
            return selections, [combined_receipt], 0

        prepare._run_years = reused_years
        try:
            with _temporary_cwd(success_root):
                resumed = prepare.run(
                    stage="years",
                    execute_download=True,
                    repo_root=success_root,
                    client_factory=forbidden_client,
                )
        finally:
            prepare._run_years = original_run_years
        final_payload = read_json(canonical)

        rollback_root = fixture_root / "rollback"
        rollback_canonical = rollback_root / (
            "external_data/quarantine/era5_p3_context_pretrain_v1/manifests/manifest.json"
        )
        _write_fixture_incomplete(rollback_canonical)
        rollback_config_path, rollback_config = _fixture_config(
            rollback_root, rollback_canonical
        )
        rollback_eligible = evaluate_migration(
            rollback_root,
            rollback_config_path,
            active_process_override=[],
            verify_upstream=False,
        )
        _apply_from_eligible(
            rollback_root,
            rollback_config_path,
            rollback_eligible,
            acknowledge_abnormal_termination=True,
            recovery_budget="remaining-two",
        )
        rollback = rollback_migration(
            rollback_root,
            rollback_config_path,
            acknowledge_rollback=True,
            recovery_budget="abandoned",
            active_process_override=[],
        )
        restored = inspect_known_incomplete_manifest(
            rollback_canonical,
            rollback_config["known_incomplete_manifest"],
        )

    checks = {
        "migration_fixture_eligible": eligible["verdict"]
        == "DRY_RUN_ELIGIBLE_AFTER_PROCESS_EXIT",
        "atomic_backup_preserved_known_hash": applied["backup_sha256"]
        == config["known_incomplete_manifest"]["sha256"],
        "existing_raw_reused_without_client": downloaded is False,
        "same_prepare_stage_years_completed": resumed["status"] == "complete"
        and resumed["stage"] == "years",
        "same_prepare_resume_download_count_zero": resumed["download_count"] == 0
        and resumed["network_action_taken"] is False,
        "same_prepare_wrote_final_manifest": final_payload.get("stage") == "years"
        and final_payload.get("row_count") == 262_917
        and final_payload.get("local_file") is not None
        and final_payload.get("file_sha256") == combined_receipt.sha256,
        "rollback_restored_exact_known_manifest": rollback["status"]
        == "ROLLBACK_COMPLETE_MANUAL_ABANDON"
        and restored["exact_known_incomplete"] is True,
        "first_resume_is_attempt_2_of_3": applied["first_post_migration_resume_attempt"] == 2,
        "one_automatic_resume_remains_after_attempt_2": applied[
            "remaining_automatic_resumes_after_first_post_migration_resume"
        ]
        == 1,
    }
    return {
        "schema_version": "p3.era5_manifest_collision_recovery.fixture_qa.v2",
        "passed": all(checks.values()),
        "checks": checks,
        "same_prepare_semantics": {
            "launcher": ".venv-era5/Scripts/python.exe",
            "stage": "years",
            "execute_download": True,
            "completed_raw_reuse_constructed_client": False,
            "simulated_remaining_download_count": 0,
            "year_request_count": resumed["year_request_count"],
            "final_row_count": final_payload.get("row_count"),
            "first_resume_attempt": 2,
            "remaining_automatic_resumes_after_first_resume": 1,
        },
        "actual_era5_path_accessed": False,
        "temporary_fixture_deleted": True,
    }


def render_report(dry_run: Mapping[str, Any], fixture: Mapping[str, Any]) -> str:
    process = dry_run["prepare_process_state"]
    plan = dry_run["plan"]
    return f"""# P3 ERA5 incomplete manifest guarded recovery

## 결론

현재 dry-run 판정은 **`{dry_run['verdict']}`**다. active prepare match는 `{process['active_match_count']}`개다. 이번 감사에서는 `--apply`를 호출하지 않았고 canonical manifest, ERA5 value files, prepare process를 수정하지 않았다.

도구는 manifest SHA256가 정확히 `55e5545a3cc7df6df288eb04f2f437f2a157b3968f9ea10a3c7f669223d94525`이며 schema/source가 일치하고 stage·local_file·file_sha256·coverage가 null, row_count=0, selected/year/files/checksums가 모두 빈 상태인지 확인한다. context attempt/output도 완전히 미소비여야 한다. completed 또는 hash/schema가 다른 manifest는 hard fail이다.

## 비정상 종료 후 단일 apply 명령

```powershell
& .\\.venv-p1\\Scripts\\python.exe scripts\\migrate_p3_era5_incomplete_manifest_v1.py --root . --apply --acknowledge-abnormal-termination --recovery-budget remaining-two
```

이 명령은 active prepare process가 0개일 때만 known incomplete manifest를 다음 recoverable backup으로 atomic relocation하고 append-only intent/complete receipts를 남긴다.

- Backup: `{plan['recoverable_backup']}`
- Apply receipt: `{plan['apply_receipt_dir']}`
- 이후 동일 prepare command: `{plan['exact_prepare_resume_argv']}`

Migration 자체는 automation의 automatic prepare-resume budget을 소비하지 않는다. 현재는 attempt 1/3이고 automatic resume 2회가 남아 있다. Migration 후 첫 prepare 재개는 attempt 2/3으로 1회를 소비하며, 실패하더라도 attempt 3/3에 쓸 automatic resume 1회가 남는다. Exact prepare launcher는 `.venv-era5\\Scripts\\python.exe`다.

## Temp fixture QA

Fixture QA는 `{fixture['passed']}`다. 실제 ERA5 경로가 아닌 OS temp root에서 exact prepare `stage=years, execute_download=True` 호출면을 사용했다. 기존 raw가 있으면 CDS client를 만들지 않고 재사용했으며, migration 후 final manifest 262,917행을 쓸 수 있음을 확인했다.

## Rollback

Prepare 재개가 final canonical manifest를 쓰기 전에 실패하더라도 남은 attempt 3/3을 우선 보존한다. 이후 그 remaining automatic resume를 사용하지 않기로 명시적으로 포기하고 manual recovery로 전환했으며, active process가 0개이고 canonical이 여전히 absent일 때만 다음 rollback을 쓴다.

```powershell
& .\\.venv-p1\\Scripts\\python.exe scripts\\migrate_p3_era5_incomplete_manifest_v1.py --root . --rollback --acknowledge-abandon-remaining-automatic-resume --recovery-budget abandoned
```

첫 migrated resume가 final manifest 전에 실패해도 canonical은 absent이므로 attempt 3/3이 raw를 재사용할 수 있다. 따라서 rollback은 자동화 절차가 아니며, 남은 automatic resume를 명시적으로 포기한 뒤의 manual recovery로만 허용된다. Rollback은 apply receipts와 backup hash를 검증해 known incomplete manifest를 atomic restore하고 append-only rollback receipts를 남긴다. final 또는 unknown canonical이 존재하면 절대 덮어쓰지 않는다.
"""


def write_audit_artifacts(root: Path, config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    output = resolve_inside(root, str(config["artifact_dir"]))
    protocol = output / "migration_protocol.json"
    test_receipt = output / "test_receipt.json"
    if not protocol.is_file() or not test_receipt.is_file():
        raise RecoveryError("sealed migration protocol and test receipt are required")
    protected = [
        output / "dry_run.json",
        output / "fixture_qa.json",
        output / "report_ko.md",
        output / "manifest.json",
        output / "manifest.sha256",
    ]
    if any(path.exists() for path in protected):
        raise FileExistsError("recovery audit artifacts are append-only")
    fixture = run_temp_fixture_qa()
    if not fixture["passed"]:
        raise RecoveryError("temporary recovery fixture QA failed")
    dry_run = evaluate_migration(root, config_path)
    exclusive_json(output / "dry_run.json", dry_run)
    exclusive_json(output / "fixture_qa.json", fixture)
    exclusive_text(output / "report_ko.md", render_report(dry_run, fixture))
    outputs = {
        path.name: sha256_file(path)
        for path in [
            protocol,
            test_receipt,
            output / "dry_run.json",
            output / "fixture_qa.json",
            output / "report_ko.md",
        ]
    }
    manifest = {
        "schema_version": "p3.era5_manifest_collision_recovery.manifest.v2",
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE_DRY_RUN_RECOVERY_AUDIT",
        "dry_run_verdict": dry_run["verdict"],
        "apply_call_count": 0,
        "rollback_call_count": 0,
        "canonical_manifest_mutations": 0,
        "era5_value_file_reads_or_mutations": 0,
        "prepare_process_mutations": 0,
        "scientific_runner_invocations": 0,
        "config_sha256": sha256_file(config_path),
        "migration_tool_sha256": sha256_file(Path(__file__).resolve()),
        "outputs_sha256": outputs,
    }
    exclusive_json(output / "manifest.json", manifest)
    manifest_hash = sha256_file(output / "manifest.json")
    exclusive_text(output / "manifest.sha256", f"{manifest_hash}  manifest.json\n")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/experiments/p3_era5_manifest_collision_recovery_20260825_v2.json",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--rollback", action="store_true")
    mode.add_argument("--write-audit-artifacts", action="store_true")
    parser.add_argument("--acknowledge-abnormal-termination", action="store_true")
    parser.add_argument(
        "--acknowledge-abandon-remaining-automatic-resume", action="store_true"
    )
    parser.add_argument(
        "--recovery-budget",
        choices=("remaining-two", "remaining-one", "already-used", "abandoned"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    config_path = args.config.resolve()
    if args.apply:
        result = apply_migration(
            root,
            config_path,
            acknowledge_abnormal_termination=args.acknowledge_abnormal_termination,
            recovery_budget=args.recovery_budget,
        )
    elif args.rollback:
        result = rollback_migration(
            root,
            config_path,
            acknowledge_rollback=args.acknowledge_abandon_remaining_automatic_resume,
            recovery_budget=args.recovery_budget,
        )
    elif args.write_audit_artifacts:
        result = write_audit_artifacts(root, config_path)
    else:
        result = evaluate_migration(root, config_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
