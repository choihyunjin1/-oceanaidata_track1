#!/usr/bin/env python
"""Independent aggregate-only QA for the sealed P2 45-cell readiness artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/p2_authoritative_nested_surrogate_execution_20260825_v2.json"
)
DEFAULT_ARTIFACT = (
    PROJECT_ROOT / "artifacts/p2_authoritative_nested_surrogate_execution_ready_20260825_v2"
)
EXPECTED_CONFIG_SHA256 = "b05fe56730ef8116b0aa6b914823dedfbb878595190d5d7a9366987eb07685b4"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    overlay = _read(path)
    base_pin = dict(overlay["base_config"])
    base_path = (PROJECT_ROOT / base_pin["path"]).resolve(strict=True)
    _require(_sha256(base_path) == base_pin["sha256"], "base config hash changed")
    return _deep_merge(_read(base_path), overlay), base_pin


def _write_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run(config_path: Path, artifact_dir: Path) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    artifact_dir = artifact_dir.resolve(strict=True)
    _require(_sha256(config_path) == EXPECTED_CONFIG_SHA256, "config hash changed")
    config, base_pin = _load_config(config_path)
    manifest_path = artifact_dir / "manifest.json"
    manifest = _read(manifest_path)
    _require(manifest["status"] == "EXECUTION_READY_NOT_AUTHORIZED", "status changed")
    _require(
        manifest["config"]["sha256"] == EXPECTED_CONFIG_SHA256,
        "manifest config pin changed",
    )
    for name, pin in manifest["outputs"].items():
        path = (artifact_dir / name).resolve(strict=True)
        _require(path.parent == artifact_dir, f"manifest output escaped: {name}")
        _require(_sha256(path) == pin["sha256"], f"output hash changed: {name}")
        _require(path.stat().st_size == int(pin["bytes"]), f"output size changed: {name}")
    module_path = (PROJECT_ROOT / manifest["module"]["path"]).resolve(strict=True)
    runner_path = (PROJECT_ROOT / manifest["runner"]["path"]).resolve(strict=True)
    _require(_sha256(module_path) == manifest["module"]["sha256"], "module hash changed")
    _require(_sha256(runner_path) == manifest["runner"]["sha256"], "runner hash changed")

    static = _read(artifact_dir / "static_verification.json")
    metadata = _read(artifact_dir / "train_metadata_receipt.json")
    plan = _read(artifact_dir / "execution_plan.json")
    tiny = _read(artifact_dir / "tiny_fixture_receipt.json")
    shapes = _read(artifact_dir / "deep_model_shape_receipt.json")
    atomic_publish = _read(artifact_dir / "atomic_publish_receipt.json")
    command_namespace = _read(artifact_dir / "exact_command_namespace_receipt.json")
    superseded_v1 = _read(artifact_dir / "superseded_v1_fail_closed_receipt.json")
    resource = _read(artifact_dir / "resource_estimate.json")
    seal = _read(artifact_dir / "preexecution_seal.json")
    qa = _read(artifact_dir / "qa.json")

    _require(static["status"] == "PASS_STATIC_SOURCE_CONFIG_AND_PARENT_PINS", "pins failed")
    _require(metadata["columns_read"] == ["station", "layer", "time"], "metadata scope changed")
    _require(metadata["value_columns_read"] == [], "dry-run read value columns")
    _require(
        (plan["outer_prefix_cells"], plan["seeded_cells"], plan["top_level_component_jobs"])
        == (15, 45, 900),
        "15/45/900 graph changed",
    )
    _require(plan["underlying_base_estimator_fits"] == 1440, "base fit count changed")
    _require(plan["meta_optimizations"] == 405, "meta optimizer count changed")
    _require(tiny["synthetic_component_callbacks"] == 20, "tiny first pass changed")
    _require(tiny["second_pass_reused_jobs"] == 20, "tiny resume changed")
    _require(tiny["second_pass_callbacks"] == 0, "tiny resume reran callback")
    _require(tiny["actual_model_fits"] == 0, "tiny reports a fit")
    _require(len(shapes["components"]) == 4, "deep shape component count changed")
    _require(shapes["actual_model_fits"] == 0, "shape receipt reports a fit")
    _require(
        atomic_publish["status"]
        == "PASS_STALE_PARTIAL_PRESERVED_ATOMIC_COMMIT_AND_VERIFIED_RESUME",
        "atomic evaluated OOF fixture failed",
    )
    _require(
        atomic_publish["first_publication"]["status"]
        == "COMMITTED_BY_FSYNC_AND_ATOMIC_RENAME",
        "atomic evaluated OOF commit was not exercised",
    )
    _require(
        atomic_publish["second_publication"]["status"] == "REUSED_VERIFIED_FINAL",
        "atomic evaluated OOF resume was not hash verified",
    )
    _require(atomic_publish["stale_failed_partial_preserved"] is True, "partial policy changed")
    _require(
        command_namespace["status"]
        == "PASS_EXACT_COMMAND_PINS_V2_CONFIG_SEAL_AND_AUTHORIZATION",
        "exact command namespace receipt failed",
    )
    _require(
        superseded_v1["status"] == "PASS_V1_SEAL_FAILS_CLOSED_ON_CURRENT_SOURCE_HASH",
        "superseded v1 seal did not fail closed",
    )
    _require(superseded_v1["authorization_usable"] is False, "v1 seal became usable")
    _require(
        base_pin["sha256"] == superseded_v1["base_config_sha256"],
        "base config receipt mismatch",
    )
    _require(
        (resource["single_rtx5090_wall_hours_low"], resource["single_rtx5090_wall_hours_high"])
        == (4.0, 8.0),
        "resource planning range changed",
    )
    _require(resource["actual_45_cell_benchmark_performed"] is False, "estimate became benchmark")

    command_sha = hashlib.sha256(str(config["exact_command"]).encode()).hexdigest()
    command = str(config["exact_command"])
    for fragment in (
        'p2_authoritative_nested_surrogate_execution_20260825_v2.json"',
        'p2_authoritative_nested_surrogate_execution_ready_20260825_v2\\preexecution_seal.json"',
        'p2_authoritative_nested_surrogate_execution_ready_20260825_v2\\EXECUTION_AUTHORIZATION.json"',
    ):
        _require(fragment in command, f"exact command lacks v2 namespace: {fragment}")
    _require(
        "p2_authoritative_nested_surrogate_execution_ready_20260825_v1\\" not in command,
        "exact command references superseded v1 readiness",
    )
    _require(seal["status"] == "EXECUTION_READY_NOT_AUTHORIZED", "seal status changed")
    _require(seal["config_sha256"] == EXPECTED_CONFIG_SHA256, "seal config pin changed")
    _require(seal["module_sha256"] == _sha256(module_path), "seal module pin changed")
    _require(seal["runner_sha256"] == _sha256(runner_path), "seal runner pin changed")
    _require(seal["exact_command_sha256"] == command_sha, "seal command pin changed")
    _require(seal["authorization_receipt_created"] is False, "seal claims authorization")
    _require(
        seal["evaluated_oof_publication"]
        == "UNIQUE_PARTIAL_FSYNC_ATOMIC_RENAME_HASH_VERIFIED_RESUME",
        "evaluated OOF publish contract changed",
    )
    _require(
        seal["failed_partial_policy"] == "PRESERVE_FOR_AUDIT_NEVER_TREAT_AS_FINAL",
        "failed partial policy changed",
    )
    _require(
        not (artifact_dir / "EXECUTION_AUTHORIZATION.json").exists(), "authorization file exists"
    )

    order = list(qa["receipt_timestamp_order"])
    timestamps = [datetime.fromisoformat(qa["receipt_timestamps"][name]) for name in order]
    _require(
        all(left < right for left, right in zip(timestamps, timestamps[1:], strict=False)),
        "receipt timestamps are not strictly monotonic",
    )
    _require(
        datetime.fromisoformat(config["created_at_kst"]) < timestamps[0],
        "config timestamp is not earlier than receipts",
    )
    _require(qa["receipt_timestamps_monotonic_strict"] is True, "timestamp QA absent")
    _require(qa["status"] == "PASS_EXECUTION_READY_NOT_AUTHORIZED", "QA status changed")
    _require(
        qa["exact_command_namespace"]
        == "PASS_EXACT_COMMAND_PINS_V2_CONFIG_SEAL_AND_AUTHORIZATION",
        "QA lacks v2 command namespace proof",
    )
    _require(
        qa["superseded_v1_fail_closed"]
        == "PASS_V1_SEAL_FAILS_CLOSED_ON_CURRENT_SOURCE_HASH",
        "QA lacks v1 fail-closed proof",
    )
    for name in ("actual_model_fits", "actual_scores", "actual_predictions"):
        _require(qa[name] == 0 and seal[name] == 0, f"nonzero readiness counter: {name}")
    for name in (
        "official_test_reads",
        "sample_submission_reads",
        "submission_candidate_reads",
        "submission_files_generated",
        "uploads",
        "p3_process_mutations",
    ):
        _require(qa[name] == 0, f"forbidden action count changed: {name}")

    forbidden_suffixes = {".csv", ".parquet", ".pt", ".joblib"}
    unexpected = [
        path.name
        for path in artifact_dir.iterdir()
        if path.is_dir() or path.suffix.lower() in forbidden_suffixes
    ]
    _require(not unexpected, f"readiness artifact contains prediction/model payload: {unexpected}")
    actual_dir = (PROJECT_ROOT / config["output"]["actual_directory"]).resolve()
    _require(not actual_dir.exists(), "actual execution directory already exists")

    return {
        "schema_version": "p2_authoritative_nested_surrogate_execution_ready_independent_qa.v2",
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "status": "PASS_EXECUTION_READY_NOT_AUTHORIZED",
        "main_manifest_sha256": _sha256(manifest_path),
        "preexecution_seal_sha256": _sha256(artifact_dir / "preexecution_seal.json"),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "module_sha256": _sha256(module_path),
        "runner_sha256": _sha256(runner_path),
        "outer_prefix_cells": 15,
        "seeded_cells": 45,
        "top_level_component_jobs": 900,
        "underlying_base_estimator_fits_if_authorized": 1440,
        "meta_optimizations_if_authorized": 405,
        "actual_model_fits": 0,
        "actual_scores": 0,
        "actual_predictions": 0,
        "authorization_receipt_present": False,
        "evaluated_oof_atomic_publish": atomic_publish["status"],
        "exact_command_v2_namespace_pinned": True,
        "superseded_v1_seal_fails_closed": True,
        "official_test_sample_submission_reads": 0,
        "public_score_selection_or_tuning": False,
        "submission_files_generated": 0,
        "uploads": 0,
        "p3_process_mutations": 0,
        "receipt_timestamps_strictly_monotonic": True,
        "resource_range_is_planning_only_not_benchmark": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    result = run(args.config, args.artifact_dir)
    output = args.artifact_dir.resolve() / "independent_qa.json"
    _write_exclusive(output, result)
    qa_manifest = {
        "schema_version": "p2_authoritative_nested_surrogate_execution_ready_qa_manifest.v2",
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "status": result["status"],
        "files": {
            "manifest.json": _sha256(args.artifact_dir.resolve() / "manifest.json"),
            "qa.json": _sha256(args.artifact_dir.resolve() / "qa.json"),
            "preexecution_seal.json": _sha256(
                args.artifact_dir.resolve() / "preexecution_seal.json"
            ),
            "independent_qa.json": _sha256(output),
        },
        "independent_qa_runner": {
            "path": "scripts/qa_p2_authoritative_nested_surrogate_execution_ready_v1.py",
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    qa_manifest_path = args.artifact_dir.resolve() / "independent_qa_manifest.json"
    _write_exclusive(qa_manifest_path, qa_manifest)
    print(
        json.dumps(
            {
                **result,
                "independent_qa_sha256": _sha256(output),
                "independent_qa_manifest_sha256": _sha256(qa_manifest_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
