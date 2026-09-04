"""Read-only external guard for the frozen P3 ERA5 one-shot runner.

The guard never calls the scientific ``--execute`` mode.  It verifies an
append-only bridge seal, the final canonical manifest, the unconsumed attempt
state, and the frozen runner's zero-write ``--check-only`` receipt.  A guarded
READY verdict contains the exact argv that may subsequently be invoked once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class GuardError(RuntimeError):
    """The external launch contract could not be established."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GuardError(f"JSON root must be an object: {path}")
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
        raise GuardError(f"configured path escapes canonical repository root: {value}")
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


def verify_bridge(root: Path, seal_path: Path) -> dict[str, Any]:
    if not seal_path.is_file():
        raise GuardError("immutable bridge seal is absent")
    seal = read_json(seal_path)
    if seal.get("status") != "IMMUTABLE_BRIDGE_SEALED_BEFORE_GUARD_RESULT":
        raise GuardError("immutable bridge seal status changed")
    pinned = seal.get("pinned_files")
    if not isinstance(pinned, Mapping) or not pinned:
        raise GuardError("immutable bridge seal has no pinned files")
    observed: dict[str, str] = {}
    for label, receipt in pinned.items():
        if not isinstance(receipt, Mapping):
            raise GuardError(f"bridge receipt is malformed: {label}")
        path = resolve_inside(root, str(receipt.get("path", "")))
        if not path.is_file():
            raise GuardError(f"bridge-pinned file is absent: {label}")
        actual = sha256_file(path)
        expected = str(receipt.get("sha256", "")).casefold()
        if actual != expected:
            raise GuardError(f"bridge-pinned hash drift: {label}")
        observed[str(label)] = actual
    return {
        "seal_sha256": sha256_file(seal_path),
        "pinned_hashes": observed,
        "all_pinned_hashes_match": True,
    }


def classify_attempt_state(output: Path, attempt_lock: Path) -> dict[str, Any]:
    output_exists = output.exists()
    lock_exists = attempt_lock.is_file()
    result_exists = (output / "result.json").is_file()
    if lock_exists and result_exists:
        state = "BLOCKED_ATTEMPT_CONSUMED"
    elif lock_exists:
        state = "BLOCKED_CRASH_LOCK"
    elif output_exists:
        state = "BLOCKED_INCONSISTENT_ATTEMPT_STATE"
    else:
        state = "ATTEMPT_AVAILABLE"
    return {
        "state": state,
        "output_exists": output_exists,
        "attempt_lock_exists": lock_exists,
        "result_exists": result_exists,
    }


def inspect_final_manifest(
    root: Path,
    manifest_path: Path,
    quarantine: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if not manifest_path.is_file():
        return {
            "exists": False,
            "ready": False,
            "incomplete_collision": False,
            "checks": {},
            "failed_checks": ["manifest_exists"],
            "partial_file_count": sum(1 for _ in quarantine.rglob("*.partial"))
            if quarantine.is_dir()
            else 0,
        }
    payload = read_json(manifest_path)
    local_file = payload.get("local_file")
    row_count = payload.get("row_count")
    stage = payload.get("stage")
    incomplete_collision = bool(
        stage not in set(expected["stage_any_of"])
        and (not isinstance(row_count, int) or row_count <= 0)
        and not local_file
    )
    candidate = (
        resolve_inside(root, str(local_file))
        if isinstance(local_file, str) and local_file
        else None
    )
    selected = payload.get("selected_cells")
    requests = payload.get("requests")
    year_requests = (
        requests.get("selected_single_cell_years")
        if isinstance(requests, Mapping)
        else None
    )
    files = payload.get("files")
    final_entries = [
        entry
        for entry in files
        if isinstance(entry, Mapping)
        and entry.get("role") == "final_combined_selected_cell_hourly_parquet"
    ] if isinstance(files, list) else []
    final_entry = final_entries[0] if len(final_entries) == 1 else {}
    file_sha = payload.get("file_sha256")
    checks = {
        "schema_version": payload.get("schema_version") == expected["schema_version"],
        "source_id": payload.get("source_id") == expected["source_id"],
        "completed_stage": stage in set(expected["stage_any_of"]),
        "local_file": local_file == expected["local_file"],
        "candidate_inside_quarantine": bool(
            candidate is not None and _inside(candidate, quarantine)
        ),
        "candidate_exists": bool(candidate is not None and candidate.is_file()),
        "file_sha256_shape": isinstance(file_sha, str)
        and len(file_sha) == 64
        and all(character in "0123456789abcdef" for character in file_sha.casefold()),
        "row_count": row_count == expected["row_count"],
        "observed_start": payload.get("observed_start") == expected["observed_start"],
        "observed_end": payload.get("observed_end") == expected["observed_end"],
        "selected_cells": isinstance(selected, list)
        and len(selected) == expected["selected_cells"],
        "year_requests": isinstance(year_requests, list)
        and len(year_requests) == expected["selected_single_cell_year_requests"],
        "official_access_false": payload.get("official_test_or_submission_accessed")
        is expected["official_test_or_submission_accessed"],
        "one_final_receipt": len(final_entries) == 1,
        "final_receipt_relative_path": final_entry.get("relative_path")
        == expected["final_receipt_relative_path"],
        "final_receipt_sha256": final_entry.get("sha256") == file_sha,
        "final_receipt_row_count": final_entry.get("row_count") == row_count,
        "final_receipt_start": final_entry.get("time_start_utc")
        == payload.get("observed_start"),
        "final_receipt_end": final_entry.get("time_end_utc")
        == payload.get("observed_end"),
    }
    partial_count = (
        sum(1 for _ in quarantine.rglob("*.partial")) if quarantine.is_dir() else 0
    )
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "exists": True,
        "ready": not failed and partial_count == expected["partial_file_count"],
        "incomplete_collision": incomplete_collision,
        "sha256": sha256_file(manifest_path),
        "stage": stage,
        "row_count": row_count,
        "local_file": local_file,
        "file_sha256": file_sha,
        "checks": checks,
        "failed_checks": failed,
        "partial_file_count": partial_count,
    }


def run_fixed_check_only(root: Path, runner: Path) -> dict[str, Any]:
    argv = [
        str(Path(sys.executable).resolve()),
        str(runner.resolve()),
        "--root",
        str(root.resolve()),
        "--check-only",
    ]
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    completed = subprocess.run(
        argv,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout) if completed.returncode == 0 else None
    except json.JSONDecodeError:
        payload = None
    return {
        "argv": argv,
        "returncode": completed.returncode,
        "stderr": completed.stderr[-4000:],
        "payload": payload,
    }


def validate_check_only_receipt(
    observed: Mapping[str, Any],
    required: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    preflight = observed.get("source_preflight")
    checks = {
        "mode": observed.get("mode") == required["mode"],
        "passed": observed.get("passed") is required["passed"],
        "writes_zero": observed.get("writes") == required["writes"],
        "model_fits_zero": observed.get("model_fits") == required["model_fits"],
        "outcome_values_read_zero": observed.get("outcome_values_read")
        == required["outcome_values_read"],
        "feature_count": observed.get("common_feature_count")
        == required["common_feature_count"],
        "source_quarantine_ready": observed.get("source_quarantine_ready")
        is required["source_quarantine_ready"],
        "source_preflight_non_null": isinstance(preflight, Mapping) and bool(preflight),
        "preflight_accepted": isinstance(preflight, Mapping)
        and preflight.get("accepted") is required["source_preflight_accepted"],
        "preflight_problem": isinstance(preflight, Mapping)
        and preflight.get("problem") == required["source_preflight_problem"],
        "preflight_source_id": isinstance(preflight, Mapping)
        and preflight.get("source_id") == required["source_preflight_source_id"],
        "preflight_purpose": isinstance(preflight, Mapping)
        and preflight.get("purpose") == required["source_preflight_purpose"],
        "preflight_manifest_sha256": isinstance(preflight, Mapping)
        and preflight.get("manifest_sha256") == manifest.get("sha256"),
        "preflight_candidate_sha256": isinstance(preflight, Mapping)
        and preflight.get("candidate_sha256") == manifest.get("file_sha256"),
        "preflight_row_count": isinstance(preflight, Mapping)
        and preflight.get("row_count") == manifest.get("row_count"),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def evaluate_guard(
    root: Path,
    config_path: Path,
    *,
    check_only_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    canonical_root = ROOT.resolve()
    root = root.resolve()
    created = datetime.now().astimezone().isoformat()
    base: dict[str, Any] = {
        "schema_version": "p3.era5_guarded_launch.evaluation.v1",
        "created_at_kst": created,
        "guard_auto_executed_scientific_runner": False,
        "scientific_runner_execute_call_count": 0,
    }
    if root != canonical_root:
        return {
            **base,
            "verdict": "BLOCKED_BRIDGE_SEAL",
            "reason": "guard root is not the canonical repository containing this script",
        }
    config = read_json(config_path)
    if config.get("guard_id") != "p3_era5_guarded_launch_20260825_v1":
        return {**base, "verdict": "BLOCKED_BRIDGE_SEAL", "reason": "guard id changed"}
    seal_path = resolve_inside(root, str(config["bridge_seal"]))
    try:
        bridge = verify_bridge(root, seal_path)
    except (GuardError, OSError, json.JSONDecodeError) as exc:
        status = "BLOCKED_HASH_DRIFT" if "hash drift" in str(exc) else "BLOCKED_BRIDGE_SEAL"
        return {**base, "verdict": status, "reason": str(exc)}

    output = resolve_inside(root, str(config["context_output"]))
    attempt_lock = resolve_inside(root, str(config["attempt_lock"]))
    attempt = classify_attempt_state(output, attempt_lock)
    if attempt["state"] != "ATTEMPT_AVAILABLE":
        return {
            **base,
            "verdict": attempt["state"],
            "reason": "the canonical one-shot attempt surface is not unused",
            "bridge": bridge,
            "attempt_state": attempt,
        }

    quarantine = resolve_inside(root, str(config["canonical_quarantine"]))
    manifest_path = resolve_inside(root, str(config["canonical_manifest"]))
    manifest = inspect_final_manifest(
        root,
        manifest_path,
        quarantine,
        config["expected_final_manifest"],
    )
    if not manifest["exists"]:
        verdict = "BLOCKED_DOWNLOAD_INCOMPLETE"
    elif manifest["incomplete_collision"]:
        verdict = "BLOCKED_INCOMPLETE_CANONICAL_COLLISION"
    elif manifest["partial_file_count"] != config["expected_final_manifest"][
        "partial_file_count"
    ]:
        verdict = "BLOCKED_DOWNLOAD_PARTIALS"
    elif not manifest["ready"]:
        verdict = "BLOCKED_FINAL_MANIFEST_CONTRACT"
    else:
        verdict = None
    if verdict is not None:
        return {
            **base,
            "verdict": verdict,
            "reason": "canonical source has not reached the guarded final-manifest state",
            "bridge": bridge,
            "attempt_state": attempt,
            "manifest_state": manifest,
            "check_only_invoked": False,
        }

    runner = resolve_inside(root, str(config["fixed_runner"]))
    check_call = (
        {
            "argv": ["fixture_override"],
            "returncode": 0,
            "stderr": "",
            "payload": dict(check_only_override),
        }
        if check_only_override is not None
        else run_fixed_check_only(root, runner)
    )
    if check_call["returncode"] != 0 or not isinstance(check_call["payload"], Mapping):
        return {
            **base,
            "verdict": "BLOCKED_CHECK_ONLY",
            "reason": "fixed runner check-only did not return a valid zero-write receipt",
            "bridge": bridge,
            "attempt_state": attempt,
            "manifest_state": manifest,
            "check_only": check_call,
        }
    receipt = validate_check_only_receipt(
        check_call["payload"],
        config["required_check_only_receipt"],
        manifest,
    )
    if not receipt["passed"]:
        source_ready = bool(check_call["payload"].get("source_quarantine_ready"))
        source_preflight = check_call["payload"].get("source_preflight")
        verdict = (
            "BLOCKED_SOURCE_NOT_READY"
            if not source_ready or source_preflight is None
            else "BLOCKED_PREFLIGHT_RECEIPT"
        )
        return {
            **base,
            "verdict": verdict,
            "reason": "source-ready and non-null accepted preflight are both mandatory",
            "bridge": bridge,
            "attempt_state": attempt,
            "manifest_state": manifest,
            "check_only": check_call,
            "check_only_validation": receipt,
        }

    execute_argv = [
        str(Path(sys.executable).resolve()),
        str(runner.resolve()),
        "--root",
        str(root),
        "--execute",
    ]
    return {
        **base,
        "verdict": "READY_GUARDED",
        "reason": "all immutable, final-manifest, attempt, source-ready, and preflight checks passed",
        "bridge": bridge,
        "attempt_state": attempt,
        "manifest_state": manifest,
        "check_only": check_call,
        "check_only_validation": receipt,
        "authorized_execute_argv_exactly_once": execute_argv,
        "authorization_scope": "one invocation only; this guard does not invoke it",
    }


@contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_temp_fixture_qa() -> dict[str, Any]:
    from p3_wave.era5_pretrain_data import (
        COMBINED_FILE_NAME,
        DYNAMIC_VARIABLES,
        STATIONS,
        FileReceipt,
        QuarantineLayout,
        SelectedCell,
        build_manifest,
        build_smoke_plan,
        build_year_plan,
        sha256_file as source_sha256,
        validate_existing_canonical_manifest,
        write_manifest,
    )

    with tempfile.TemporaryDirectory(prefix="p3-era5-guard-fixture-") as temporary:
        temp_root = Path(temporary).resolve()
        layout = QuarantineLayout.from_repo_root(temp_root)
        layout.ensure()
        combined = layout.derived / COMBINED_FILE_NAME
        combined.write_bytes(b"synthetic-combined-fixture")
        receipt = FileReceipt(
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
        with _temporary_cwd(temp_root):
            final_manifest = build_manifest(
                stage="combine",
                smoke_requests=build_smoke_plan(),
                year_requests=build_year_plan(selections),
                selections=selections,
                files=[receipt],
                network_action_taken=False,
            )
        canonical = layout.manifests / "manifest.json"
        canonical.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "source_id": "era5_pre2024",
                    "stage": None,
                    "row_count": 0,
                    "local_file": None,
                    "file_sha256": None,
                }
            ),
            encoding="utf-8",
        )
        incomplete_refused = False
        incomplete_error = ""
        try:
            validate_existing_canonical_manifest(layout, combined_receipt=receipt)
        except FileExistsError as exc:
            incomplete_refused = True
            incomplete_error = str(exc)

        written = write_manifest(layout, final_manifest, stage="combine")
        direct_writer_replaced = written == canonical and read_json(canonical).get("row_count") == 262_917
        completed_reuse_valid = validate_existing_canonical_manifest(
            layout,
            combined_receipt=receipt,
        )
        combined.write_bytes(b"tampered-synthetic-combined-fixture")
        tamper_refused = False
        tamper_error = ""
        try:
            validate_existing_canonical_manifest(layout, combined_receipt=receipt)
        except FileExistsError as exc:
            tamper_refused = True
            tamper_error = str(exc)

        attempt_root = temp_root / "attempt-fixture"
        output = attempt_root / "output"
        lock = attempt_root / "output.attempt.lock"
        initial = classify_attempt_state(output, lock)["state"]
        attempt_root.mkdir(parents=True)
        lock.write_text("fixture lock", encoding="utf-8")
        crash = classify_attempt_state(output, lock)["state"]
        output.mkdir()
        (output / "result.json").write_text("{}", encoding="utf-8")
        consumed = classify_attempt_state(output, lock)["state"]
        lock.unlink()
        inconsistent = classify_attempt_state(output, lock)["state"]

    checks = {
        "incomplete_existing_manifest_is_collision": incomplete_refused,
        "low_level_atomic_writer_can_replace_incomplete_manifest": direct_writer_replaced,
        "completed_same_receipt_reuse_is_valid": bool(completed_reuse_valid),
        "tampered_completed_surface_is_collision": tamper_refused,
        "attempt_initially_available": initial == "ATTEMPT_AVAILABLE",
        "lock_without_result_is_crash_lock": crash == "BLOCKED_CRASH_LOCK",
        "lock_with_result_is_consumed": consumed == "BLOCKED_ATTEMPT_CONSUMED",
        "output_without_lock_is_inconsistent": inconsistent
        == "BLOCKED_INCONSISTENT_ATTEMPT_STATE",
    }
    return {
        "schema_version": "p3.era5_guarded_launch.temp_fixture_qa.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "production_semantics": {
            "prepare_sequence_with_existing_incomplete_manifest": "refuses before write_manifest",
            "write_manifest_low_level_semantics": "atomic replacement is possible if called directly",
            "completed_manifest_or_combined_file_mismatch": "collision refusal",
            "crash_after_attempt_lock_before_result": "BLOCKED_CRASH_LOCK; no retry authorization",
        },
        "observed_errors": {
            "incomplete": incomplete_error,
            "tampered": tamper_error,
        },
        "actual_era5_path_accessed": False,
        "temporary_fixture_deleted": True,
    }


def render_report(evaluation: Mapping[str, Any], fixture: Mapping[str, Any]) -> str:
    manifest = evaluation.get("manifest_state", {})
    failed = manifest.get("failed_checks", []) if isinstance(manifest, Mapping) else []
    return f"""# P3 ERA5 guarded-launch 후속 감사

## 결론

현재 판정은 **`{evaluation['verdict']}`**다. Guard는 과학 runner의 `--execute`를 호출하지 않았으며, 모델 fit·prediction도 수행하지 않았다.

현재 canonical manifest stage는 `{manifest.get('stage')}`이고 row_count는 `{manifest.get('row_count')}`다. 실패한 final-manifest checks는 `{failed}`다. `READY_GUARDED`는 bridge의 모든 hash, final manifest, partial 0개, 미소비 attempt, fixed runner의 `source_quarantine_ready=true`, non-null·accepted preflight가 모두 확인될 때만 반환된다.

## Temp fixture로 확인한 실제 production semantics

- fixture QA: `{fixture['passed']}`
- 기존 incomplete canonical manifest는 `validate_existing_canonical_manifest`에서 collision으로 거부되어 현재 prepare 호출 순서에서는 final write에 도달하지 못한다.
- 저수준 `write_manifest` 자체는 direct call 시 atomic replacement가 가능하지만, fixed prepare는 그 전에 validation을 호출한다.
- 동일한 completed manifest/combined receipt 재사용은 허용되고, combined file 변조·불일치는 collision으로 거부된다.
- attempt lock만 있고 result가 없으면 `BLOCKED_CRASH_LOCK`; lock과 result가 있으면 `BLOCKED_ATTEMPT_CONSUMED`; output만 있으면 inconsistent state다.

모든 fixture 파일은 OS 임시 디렉터리에만 생성되었고 삭제됐다. 실제 ERA5 quarantine이나 실행 프로세스는 수정·검사하지 않았다.

## 단일 read-only guard 명령

```powershell
& .\\.venv-p1\\Scripts\\python.exe scripts\\guard_p3_era5_context_launch_v1.py --root . --evaluate
```

이 명령은 scientific execute를 자동 호출하지 않는다. `READY_GUARDED`일 때 출력되는 `authorized_execute_argv_exactly_once`만 이후 별도 승인된 1회 호출에 사용할 수 있다. 그 외 모든 `BLOCKED_*` 상태에서는 호출하면 안 된다.
"""


def write_audit_artifacts(root: Path, config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    output = resolve_inside(root, str(config["output_dir"]))
    required_existing = [output / "immutable_bridge_seal.json", output / "test_receipt.json"]
    if not all(path.is_file() for path in required_existing):
        raise GuardError("bridge seal and test receipt must exist before artifact write")
    protected = [
        output / "guard_evaluation.json",
        output / "fixture_qa.json",
        output / "report_ko.md",
        output / "manifest.json",
        output / "manifest.sha256",
    ]
    if any(path.exists() for path in protected):
        raise FileExistsError("guard audit outputs are append-only")
    fixture = run_temp_fixture_qa()
    if not fixture["passed"]:
        raise GuardError("temporary production-semantics fixture QA failed")
    evaluation = evaluate_guard(root, config_path)
    exclusive_json(output / "guard_evaluation.json", evaluation)
    exclusive_json(output / "fixture_qa.json", fixture)
    exclusive_text(output / "report_ko.md", render_report(evaluation, fixture))
    outputs = {
        path.name: sha256_file(path)
        for path in [
            output / "immutable_bridge_seal.json",
            output / "test_receipt.json",
            output / "guard_evaluation.json",
            output / "fixture_qa.json",
            output / "report_ko.md",
        ]
    }
    manifest = {
        "schema_version": "p3.era5_guarded_launch.manifest.v1",
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE_EXTERNAL_GUARD_AUDIT",
        "guard_verdict": evaluation["verdict"],
        "guard_auto_executed_scientific_runner": False,
        "scientific_runner_execute_call_count": 0,
        "temporary_fixture_actual_era5_path_accessed": False,
        "config_sha256": sha256_file(config_path),
        "guard_sha256": sha256_file(Path(__file__).resolve()),
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
        default=ROOT / "configs/experiments/p3_era5_guarded_launch_20260825_v1.json",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--write-audit-artifacts", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    config_path = args.config.resolve()
    if args.write_audit_artifacts:
        result = write_audit_artifacts(root, config_path)
    else:
        result = evaluate_guard(root, config_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
