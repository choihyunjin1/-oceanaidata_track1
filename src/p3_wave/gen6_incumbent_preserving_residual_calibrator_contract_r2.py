"""Security and append-only contract for P3 Gen6r2.

This module owns static authorization, the one-shot attempt lock, live phase
capabilities, exclusive output writes, and independent post-publish verification.
It never fits a model or decodes a target scalar during static preflight.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import secrets
import stat
import sys
import threading
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np
import pyarrow.parquet as pq

SCHEMA_VERSION: Final = "p3_gen6_incumbent_preserving_residual_calibrator.v1r2"
EXPERIMENT_ID: Final = "p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2"
CONFIG_RELATIVE: Final = (
    "configs/experiments/p3_gen6_incumbent_preserving_residual_calibrator_v1r2.json"
)
CONTRACT_RELATIVE: Final = (
    "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_contract_r2.py"
)
ENGINE_RELATIVE: Final = (
    "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_execution_r2.py"
)
RUNNER_RELATIVE: Final = (
    "scripts/run_p3_gen6_incumbent_preserving_residual_calibrator_v1r2.py"
)
TESTS_RELATIVE: Final = (
    "tests/test_p3_gen6_incumbent_preserving_residual_calibrator_v1r2.py"
)
EXPECTED_CONFIG_BYTES: Final = 16968
EXPECTED_CONFIG_SHA256: Final = (
    "77bcd142862795c1d5e8dbcc900c4bc728ce867ca018afc44ef6ff3038202f1f"
)
EXPECTED_CONFIG_DEEP_SHA256: Final = (
    "4d3f6ec4e8e4cff4dd04a427d8159b4667d14f8c4312226437f1bad479e23355"
)
EXPECTED_SCIENCE_DEEP_SHA256: Final = (
    "b2431a5f7a1e4746648b0be2139533273f40c31a61898dff4a680648c7bf27e0"
)

FOLD_ORDER: Final = ("2024_h2_storm", "winter_transition", "2025_h1")
PREFIX_FRACTIONS: Final = (0.2, 0.4, 0.6, 0.8, 1.0)
PHASE_ORDER: Final = (
    "ENGINE_LOAD_INPUTS",
    "FOLD_0_PREDICT_COMMIT",
    "FOLD_0_RELEASE_PRIOR_TRUTH",
    "FOLD_1_PREDICT_COMMIT",
    "FOLD_1_RELEASE_PRIOR_TRUTH",
    "FOLD_2_PREDICT_COMMIT",
    "PREDICTIONS_COMPLETE_COMMIT",
    "FOLD_2_RELEASE_SCORING_TRUTH",
    "SCORE_AND_WRITE_CORE",
    "PUBLISH_MANIFEST_SIDECAR_SEAL",
    "PUBLISH_IN_PROGRESS",
    "REVOKED",
)
ALLOWED_DIRECTORIES: Final = (".", "blind", "commitments")
ALLOWED_FILES: Final = (
    "blind/fold_00_2024_h2_storm.npy",
    "blind/fold_01_winter_transition.npy",
    "blind/fold_02_2025_h1.npy",
    "commitments/fold_00_2024_h2_storm.json",
    "commitments/fold_01_winter_transition.json",
    "commitments/fold_02_2025_h1.json",
    "commitments/predictions_complete.json",
    "calibrator_models.json",
    "learning_curve_evidence.json",
    "metrics.json",
    "oof.parquet",
    "manifest.json",
    "manifest.sha256",
    "seal.json",
)
CORE_FILES: Final = ALLOWED_FILES[:11]
WRITE_PHASES: Final = {
    "blind/fold_00_2024_h2_storm.npy": "FOLD_0_RELEASE_PRIOR_TRUTH",
    "commitments/fold_00_2024_h2_storm.json": "FOLD_0_RELEASE_PRIOR_TRUTH",
    "blind/fold_01_winter_transition.npy": "FOLD_1_RELEASE_PRIOR_TRUTH",
    "commitments/fold_01_winter_transition.json": "FOLD_1_RELEASE_PRIOR_TRUTH",
    "blind/fold_02_2025_h1.npy": "PREDICTIONS_COMPLETE_COMMIT",
    "commitments/fold_02_2025_h1.json": "PREDICTIONS_COMPLETE_COMMIT",
    "commitments/predictions_complete.json": "FOLD_2_RELEASE_SCORING_TRUTH",
    "calibrator_models.json": "PUBLISH_MANIFEST_SIDECAR_SEAL",
    "learning_curve_evidence.json": "PUBLISH_MANIFEST_SIDECAR_SEAL",
    "metrics.json": "PUBLISH_MANIFEST_SIDECAR_SEAL",
    "oof.parquet": "PUBLISH_MANIFEST_SIDECAR_SEAL",
    "manifest.json": "PUBLISH_IN_PROGRESS",
    "manifest.sha256": "PUBLISH_IN_PROGRESS",
    "seal.json": "PUBLISH_IN_PROGRESS",
}
THREAD_ENVIRONMENT: Final = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}
LOWER_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class R2ContractError(RuntimeError):
    """Fail-closed Gen6r2 contract violation."""


class R2LedgerChanged(R2ContractError):
    """The exact preregistered v9 anchor moved."""


class R2CapabilityError(PermissionError):
    """A direct, forged, stale, replayed, or revoked capability call."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def deep_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _has_reparse(path: Path) -> bool:
    info = path.lstat()
    attributes = int(getattr(info, "st_file_attributes", 0))
    return path.is_symlink() or bool(attributes & REPARSE_ATTRIBUTE)


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def assert_plain_existing(path: Path, *, kind: str | None = None) -> Path:
    candidate = path.absolute()
    if _lexists(candidate) and _has_reparse(candidate):
        raise R2ContractError(f"symlink/reparse path is forbidden: {candidate.name}")
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    if kind == "file" and not candidate.is_file():
        raise R2ContractError(f"regular file required: {candidate.name}")
    if kind == "directory" and not candidate.is_dir():
        raise R2ContractError(f"directory required: {candidate.name}")
    return candidate.resolve(strict=True)


def assert_contained(root: Path, path: Path, *, must_exist: bool) -> Path:
    workspace = assert_plain_existing(root, kind="directory")
    candidate = path.absolute()
    probe = candidate if _lexists(candidate) else candidate.parent
    while probe != workspace and workspace in probe.absolute().parents:
        if _lexists(probe) and _has_reparse(probe):
            raise R2ContractError("contained path has a symlink/reparse ancestor")
        probe = probe.parent
    resolved = candidate.resolve(strict=must_exist)
    if resolved != workspace and workspace not in resolved.parents:
        raise R2ContractError("path escapes the canonical workspace")
    return resolved


def file_pin(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = assert_plain_existing(path, kind="file")
    display = resolved.as_posix() if root is None else resolved.relative_to(root).as_posix()
    return {
        "path": display,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _assert_pin(path: Path, pin: Mapping[str, Any], *, label: str) -> None:
    resolved = assert_plain_existing(path, kind="file")
    if resolved.stat().st_size != int(pin["bytes"]):
        raise R2ContractError(f"{label} byte size changed")
    if sha256_file(resolved) != str(pin["sha256"]):
        raise R2ContractError(f"{label} SHA-256 changed")


def _canonical_paths_projection() -> dict[str, str]:
    return {
        "config": CONFIG_RELATIVE,
        "contract": CONTRACT_RELATIVE,
        "engine": ENGINE_RELATIVE,
        "runner": RUNNER_RELATIVE,
        "tests": TESTS_RELATIVE,
        "output": "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2",
        "control": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2_control"
        ),
        "pre_execution_qa": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2_control/"
            "pre_execution_qa.json"
        ),
        "authorization": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2_control/"
            "authorization.json"
        ),
        "attempt_lock": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2_control/"
            "attempt.lock"
        ),
        "run_failure_receipt": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2_control/"
            "run_failure_receipt.json"
        ),
        "central_v9_ledger": "artifacts/meaningful_score_goal_v9/registry.jsonl",
        "sealed_gen1_oof": (
            "artifacts/p3_meaningful_learning_curve_20260823_v1/oof/"
            "learning_curve_oof.parquet"
        ),
        "sealed_validation_keys": (
            "artifacts/p3_meaningful_learning_curve_20260823_v1/validation_keys.parquet"
        ),
        "train_anchor_metadata": "artifacts/p3/features_all20_v1/train_anchors.parquet",
    }


def load_canonical_config(
    root: Path, requested_config: Path | None = None
) -> tuple[dict[str, Any], bytes]:
    workspace = assert_plain_existing(root, kind="directory")
    canonical_lexical = workspace / CONFIG_RELATIVE
    requested_lexical = requested_config or canonical_lexical
    if requested_lexical.absolute() != canonical_lexical.absolute():
        raise R2ContractError("alternate Gen6r2 config path is forbidden")
    canonical = assert_contained(workspace, canonical_lexical, must_exist=True)
    assert_plain_existing(canonical, kind="file")
    raw = canonical.read_bytes()
    if len(raw) != EXPECTED_CONFIG_BYTES or sha256_bytes(raw) != EXPECTED_CONFIG_SHA256:
        raise R2ContractError("Gen6r2 config byte identity changed")
    config = json.loads(raw)
    if deep_sha256(config) != EXPECTED_CONFIG_DEEP_SHA256:
        raise R2ContractError("Gen6r2 config deep-JSON identity changed")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise R2ContractError("Gen6r2 schema changed")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise R2ContractError("Gen6r2 experiment identity changed")
    if config.get("canonical_paths") != _canonical_paths_projection():
        raise R2ContractError("Gen6r2 canonical paths changed")
    if tuple(config.get("capability_phase_order", ())) != PHASE_ORDER:
        raise R2ContractError("Gen6r2 phase order changed")
    output = config.get("output_contract", {})
    if tuple(output.get("allowed_directories", ())) != ALLOWED_DIRECTORIES:
        raise R2ContractError("Gen6r2 output directory allowlist changed")
    if tuple(output.get("allowed_files", ())) != ALLOWED_FILES:
        raise R2ContractError("Gen6r2 output file allowlist changed")
    if output.get("write_phase_by_file") != WRITE_PHASES:
        raise R2ContractError("Gen6r2 output write-phase map changed")
    if output.get("file_count") != len(ALLOWED_FILES):
        raise R2ContractError("Gen6r2 output file count changed")
    if any(int(value) != 0 for value in config["static_counters"].values()):
        raise R2ContractError("Gen6r2 static counters are nonzero")
    if config["execution_policy"].get("run_now_authorized") is not False:
        raise R2ContractError("Gen6r2 static config improperly authorizes a run")
    return config, raw


def _verify_science(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["science_source"]
    path = assert_contained(root, root / source["path"], must_exist=True)
    _assert_pin(path, source, label="v1 science config")
    v1 = json.loads(path.read_bytes())
    projection = {key: v1[key] for key in source["deep_equal_keys"]}
    observed = deep_sha256(projection)
    if observed != source["deep_sha256"] or observed != EXPECTED_SCIENCE_DEEP_SHA256:
        raise R2ContractError("Gen6r2 science differs from tombstoned v1")
    if source.get("hypothesis_or_science_changed") is not False:
        raise R2ContractError("Gen6r2 science-change flag differs")
    return {
        "source": file_pin(path, root=root),
        "deep_equal_keys": list(source["deep_equal_keys"]),
        "deep_sha256": observed,
        "hypothesis_or_science_changed": False,
    }


def _verify_workspace(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    workspace = assert_plain_existing(root, kind="directory")
    expected = config["canonical_workspace_identity"]
    info = workspace.stat()
    if info.st_dev != int(expected["root_st_dev"]) or info.st_ino != int(
        expected["root_st_ino"]
    ):
        raise R2ContractError("canonical root filesystem identity changed")
    git_config = workspace / ".git/config"
    assert_contained(workspace, git_config, must_exist=True)
    assert_plain_existing(git_config, kind="file")
    if sha256_file(git_config) != expected["git_config_sha256"]:
        raise R2ContractError("canonical Git config changed")
    if f"url = {expected['origin_url']}" not in git_config.read_text(encoding="utf-8"):
        raise R2ContractError("canonical Git origin changed")
    return {
        "root_st_dev": info.st_dev,
        "root_st_ino": info.st_ino,
        "git_config_sha256": expected["git_config_sha256"],
        "origin_url": expected["origin_url"],
    }


def _verify_environment(root: Path, data_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    expected = config["canonical_environment"]
    workspace_value = os.environ.get(expected["workspace_variable"])
    data_value = os.environ.get(expected["data_variable"])
    if not workspace_value or not data_value:
        raise R2ContractError("P3_WORKSPACE_ROOT and P3_DATA_DIR are both required")
    workspace_env = Path(workspace_value)
    data_env = Path(data_value)
    assert_plain_existing(workspace_env, kind="directory")
    assert_plain_existing(data_env, kind="directory")
    if (
        workspace_env.absolute() != root.absolute()
        or workspace_env.resolve(strict=True) != root.resolve(strict=True)
    ):
        raise R2ContractError("P3_WORKSPACE_ROOT differs from the canonical root")
    if (
        data_env.absolute() != data_dir.absolute()
        or data_env.resolve(strict=True) != data_dir.resolve(strict=True)
    ):
        raise R2ContractError("P3_DATA_DIR differs from the requested canonical data root")
    for key, value in THREAD_ENVIRONMENT.items():
        if os.environ.get(key) != value:
            raise R2ContractError(f"canonical process environment differs: {key}")
    source = assert_plain_existing(data_dir, kind="directory")
    return {
        "workspace_variable_present_and_exact": True,
        "data_variable_present_and_exact": True,
        "thread_environment": dict(THREAD_ENVIRONMENT),
        "data_root_plain_directory": source.is_dir(),
        "personal_absolute_path_persisted": False,
    }


def _normalized_blas_identity() -> dict[str, str]:
    details = np.show_config(mode="dicts")
    blas = details["Build Dependencies"]["blas"]
    return {
        "name": str(blas["name"]),
        "version": str(blas["version"]),
        "openblas_configuration": str(blas["openblas configuration"]),
    }


def _verify_runtime(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    expected = config["runtime_identity"]
    executable = assert_contained(
        root,
        root / expected["python_executable"]["path"],
        must_exist=True,
    )
    _assert_pin(executable, expected["python_executable"], label="Python executable")
    if Path(sys.executable).resolve(strict=True) != executable.resolve(strict=True):
        raise R2ContractError("Gen6r2 is running under a non-canonical Python executable")
    observed_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if observed_version != expected["python_version"]:
        raise R2ContractError("Python version changed")
    if platform.python_implementation() != expected["implementation"]:
        raise R2ContractError("Python implementation changed")
    if platform.system() != expected["platform_system"] or platform.machine() != expected[
        "machine"
    ]:
        raise R2ContractError("runtime platform identity changed")
    packages = {
        name: importlib.metadata.version(name) for name in expected["packages"]
    }
    if packages != expected["packages"]:
        raise R2ContractError("runtime package identity changed")
    blas = _normalized_blas_identity()
    if blas != expected["numpy_blas"]:
        raise R2ContractError("NumPy BLAS identity changed")
    for name in ("requirements_lock", "pyproject"):
        pinned = assert_contained(
            root, root / expected[name]["path"], must_exist=True
        )
        _assert_pin(pinned, expected[name], label=name)
    return {
        "python_executable": file_pin(executable, root=root),
        "python_version": observed_version,
        "implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
        "numpy_blas": blas,
        "requirements_lock": dict(expected["requirements_lock"]),
        "pyproject": dict(expected["pyproject"]),
        "thread_environment": dict(THREAD_ENVIRONMENT),
    }


def verify_central_ledger(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    anchor = config["central_ledger_anchor"]
    path = assert_contained(root, root / anchor["path"], must_exist=True)
    assert_plain_existing(path, kind="file")
    raw = path.read_bytes()
    if len(raw) != int(anchor["bytes"]) or sha256_bytes(raw) != anchor["sha256"]:
        raise R2LedgerChanged("central v9 moved; append-only rebase is required")
    if b"\r" in raw or not raw.endswith(b"\n"):
        raise R2LedgerChanged("central v9 is not LF-only JSONL")
    lines = raw.splitlines()
    if len(lines) != int(anchor["physical_event_lines"]):
        raise R2LedgerChanged("central v9 physical event count changed")
    events = [json.loads(line) for line in lines]
    previous: str | None = None
    for event in events:
        event_sha = str(event["event_sha256"])
        unsigned = {key: value for key, value in event.items() if key != "event_sha256"}
        if deep_sha256(unsigned) != event_sha:
            raise R2LedgerChanged("central v9 event hash is invalid")
        if previous is not None and event["previous_event_sha256"] != previous:
            raise R2LedgerChanged("central v9 event chain is invalid")
        previous = event_sha
    head = events[-1]
    if head["seq"] != anchor["global_head_seq"] or head["event_sha256"] != anchor[
        "head_event_sha256"
    ]:
        raise R2LedgerChanged("central v9 head changed")
    if '"upload_performed":true' in raw.decode("utf-8"):
        raise R2LedgerChanged("central v9 records an unexpected upload")
    return {
        "path": anchor["path"],
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "physical_event_lines": len(lines),
        "global_head_seq": int(head["seq"]),
        "head_event_sha256": str(head["event_sha256"]),
        "official_uploads": 0,
    }


def _verify_superseded_v1(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = config["superseded_v1"]
    for label, pin in predecessor["files"].items():
        path = assert_contained(root, root / pin["path"], must_exist=True)
        _assert_pin(path, pin, label=f"superseded v1 {label}")
    no_go = json.loads(
        (root / predecessor["files"]["OWNER_NO_GO"]["path"]).read_bytes()
    )
    tombstone = json.loads(
        (root / predecessor["files"]["EXECUTION_TOMBSTONE"]["path"]).read_bytes()
    )
    if no_go["review"]["verdict"] != "NO-GO" or no_go["review"]["p1_count"] != 3:
        raise R2ContractError("v1 owner NO-GO semantics changed")
    if tombstone["status"] != "PERMANENTLY_TOMBSTONED_NEVER_EXECUTE":
        raise R2ContractError("v1 execution tombstone semantics changed")
    counters = (
        "fit_count",
        "prediction_count",
        "score_count",
        "test_value_read_count",
        "candidate_count",
        "registry_append_count",
        "upload_count",
    )
    if any(predecessor[key] != 0 for key in counters):
        raise R2ContractError("v1 predecessor counters changed")
    return {label: dict(pin) for label, pin in predecessor["files"].items()}


def _verify_inputs(root: Path, data_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for label, pin in config["immutable_inputs"].items():
        base = data_dir if label.startswith("source/") else root
        path = assert_contained(base, base / pin["path"], must_exist=True)
        _assert_pin(path, pin, label=label)
        verified[label] = dict(pin)
    for relative, expected in config["transitive_science_pins"].items():
        path = assert_contained(root, root / relative, must_exist=True)
        if sha256_file(path) != expected:
            raise R2ContractError(f"transitive science implementation changed: {relative}")
    return verified


def implementation_pins(root: Path) -> dict[str, dict[str, Any]]:
    return {
        label: file_pin(
            assert_contained(root, root / relative, must_exist=True), root=root
        )
        for label, relative in {
            "CONFIG": CONFIG_RELATIVE,
            "CONTRACT": CONTRACT_RELATIVE,
            "ENGINE": ENGINE_RELATIVE,
            "RUNNER": RUNNER_RELATIVE,
            "TESTS": TESTS_RELATIVE,
        }.items()
    }


def _verify_parquet_metadata(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = config["canonical_paths"]
    oof_path = assert_contained(
        root, root / paths["sealed_gen1_oof"], must_exist=True
    )
    oof = pq.ParquetFile(oof_path)
    expected_oof = (
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "target_hs",
        "current_hs",
        "persistence",
        "incumbent_prediction",
        "single_horizon_residual_head",
        "multi_trajectory_residual_head",
        "fixed_horizon_splice",
        "prefix_fraction",
    )
    if oof.metadata.num_rows != 5430 or tuple(oof.schema_arrow.names) != expected_oof:
        raise R2ContractError("sealed Gen1 OOF metadata changed")
    keys_path = assert_contained(
        root, root / paths["sealed_validation_keys"], must_exist=True
    )
    keys = pq.ParquetFile(keys_path)
    if keys.metadata.num_rows != 181 or tuple(keys.schema_arrow.names) != (
        "fold",
        "anchor_id",
        "station",
        "episode_id",
    ):
        raise R2ContractError("sealed validation-key metadata changed")
    anchors_path = assert_contained(
        root, root / paths["train_anchor_metadata"], must_exist=True
    )
    anchors = pq.ParquetFile(anchors_path)
    if anchors.metadata.num_rows != 24360:
        raise R2ContractError("train-anchor metadata row count changed")
    if not {"anchor_id", "station", "anchor_time"}.issubset(anchors.schema_arrow.names):
        raise R2ContractError("train-anchor key/input schema changed")
    return {
        "oof_rows": 5430,
        "validation_cases": 181,
        "anchor_rows": 24360,
        "target_scalar_decodes": 0,
        "anonymous_test_value_reads": 0,
    }


def _assert_output_subset(root: Path, config: Mapping[str, Any], *, final: bool) -> dict[str, Any]:
    output = root / config["canonical_paths"]["output"]
    assert_contained(root, output, must_exist=_lexists(output))
    if not output.exists():
        if final:
            raise R2ContractError("sealed output is missing")
        return {"exists": False, "directories": [], "files": []}
    assert_plain_existing(output, kind="directory")
    directories: set[str] = {"."}
    files: set[str] = set()
    for path in output.rglob("*"):
        if _has_reparse(path):
            raise R2ContractError("output contains a symlink/reparse object")
        relative = path.relative_to(output).as_posix()
        if path.is_dir():
            directories.add(relative)
        elif path.is_file():
            files.add(relative)
        else:
            raise R2ContractError("output contains a non-file non-directory object")
    if not directories.issubset(ALLOWED_DIRECTORIES):
        raise R2ContractError("output contains an extra directory")
    if not files.issubset(ALLOWED_FILES):
        raise R2ContractError("output contains an extra file")
    if final and (directories != set(ALLOWED_DIRECTORIES) or files != set(ALLOWED_FILES)):
        raise R2ContractError("sealed output has missing allowlisted paths")
    return {
        "exists": True,
        "directories": sorted(directories),
        "files": sorted(files),
    }


def _control_inventory(root: Path, config: Mapping[str, Any]) -> list[str]:
    control = assert_contained(
        root,
        root / config["canonical_paths"]["control"],
        must_exist=_lexists(root / config["canonical_paths"]["control"]),
    )
    if not control.exists():
        return []
    assert_plain_existing(control, kind="directory")
    names: list[str] = []
    for path in control.iterdir():
        if _has_reparse(path) or not path.is_file():
            raise R2ContractError("control contains a non-plain file")
        names.append(path.name)
    allowed = {
        "pre_execution_qa.json",
        "authorization.json",
        "attempt.lock",
        "run_failure_receipt.json",
    }
    if not set(names).issubset(allowed):
        raise R2ContractError("control contains an unexpected file")
    return sorted(names)


def _static_lineage(
    *,
    root: Path,
    data_dir: Path,
    config: Mapping[str, Any],
    config_raw: bytes,
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "config": {
            "path": CONFIG_RELATIVE,
            "bytes": len(config_raw),
            "sha256": sha256_bytes(config_raw),
            "deep_sha256": deep_sha256(config),
        },
        "science": _verify_science(root, config),
        "superseded_v1": _verify_superseded_v1(root, config),
        "implementation_pins": implementation_pins(root),
        "immutable_inputs": _verify_inputs(root, data_dir, config),
        "runtime": _verify_runtime(root, config),
        "workspace": _verify_workspace(root, config),
        "environment": _verify_environment(root, data_dir, config),
        "central_v9_anchor": verify_central_ledger(root, config),
        "parquet_metadata": _verify_parquet_metadata(root, config),
        "output_allowlist": {
            "directories": list(ALLOWED_DIRECTORIES),
            "files": list(ALLOWED_FILES),
            "file_count": len(ALLOWED_FILES),
        },
        "fit_call_authorized_upper_bound": 20,
        "static_observed_fit_calls": 0,
    }


def static_preflight(
    root: Path,
    data_dir: Path,
    *,
    requested_config: Path | None = None,
    execution_documents_allowed: bool = False,
) -> dict[str, Any]:
    """Read-only preflight. It cannot create QA, auth, lock, output, or a result."""

    workspace = assert_plain_existing(root, kind="directory")
    source = assert_plain_existing(data_dir, kind="directory")
    config, raw = load_canonical_config(workspace, requested_config)
    output = _assert_output_subset(workspace, config, final=False)
    if output["exists"]:
        raise R2ContractError("canonical Gen6r2 output already exists")
    inventory = _control_inventory(workspace, config)
    if execution_documents_allowed:
        if not set(inventory).issubset({"pre_execution_qa.json", "authorization.json"}):
            raise R2ContractError("pre-lock control inventory changed")
    elif inventory or (workspace / config["canonical_paths"]["control"]).exists():
        raise R2ContractError("Gen6r2 control must be absent during static-only QA")
    lineage = _static_lineage(
        root=workspace,
        data_dir=source,
        config=config,
        config_raw=raw,
    )
    return {
        "schema_version": (
            "p3_gen6_incumbent_preserving_residual_calibrator.static_preflight.r2.v1"
        ),
        "status": "STATIC_PREFLIGHT_PASS_NO_WRITES",
        "required_qa_static_lineage": lineage,
        "control_inventory": inventory,
        "output_exists": False,
        "target_scalar_decodes": 0,
        "independent_qa_receipts_created": 0,
        "authorizations_created": 0,
        "attempt_locks_created": 0,
        "fit_calls": 0,
        "prediction_cells": 0,
        "score_calls": 0,
        "test_value_reads": 0,
        "candidate_files": 0,
        "registry_appends": 0,
        "uploads": 0,
    }


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise R2ContractError(f"{label} must be a JSON object")
    return value, raw


def _load_canonical_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    value, raw = _load_json(path, label=label)
    if raw != canonical_json_bytes(value) + b"\n":
        raise R2ContractError(f"{label} is not canonical LF-terminated JSON")
    return value, raw


def _require_aware_timestamp(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise R2ContractError(f"{label} timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exception:
        raise R2ContractError(f"{label} timestamp is invalid") from exception
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise R2ContractError(f"{label} timestamp lacks a timezone")
    return value


def verify_execution_documents(
    root: Path,
    data_dir: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any], str, dict[str, Any]]:
    """Deep-verify the canonical independent QA and authorization before locking."""

    config_current, raw = load_canonical_config(root, root / CONFIG_RELATIVE)
    if config_current != config:
        raise R2ContractError("in-memory config differs from canonical deep JSON")
    lineage = _static_lineage(
        root=root,
        data_dir=data_dir,
        config=config_current,
        config_raw=raw,
    )
    qa_path = assert_contained(
        root,
        root / config["canonical_paths"]["pre_execution_qa"],
        must_exist=True,
    )
    auth_path = assert_contained(
        root,
        root / config["canonical_paths"]["authorization"],
        must_exist=True,
    )
    assert_plain_existing(qa_path, kind="file")
    assert_plain_existing(auth_path, kind="file")
    qa, qa_raw = _load_json(qa_path, label="independent QA")
    expected_qa_keys = {
        "schema_version",
        "experiment_id",
        "recorded_at_kst",
        "reviewer",
        "reviewer_independent_of_implementation_owner",
        "verdict",
        "p0_count",
        "p1_count",
        "findings",
        "static_lineage",
        "output_control_lock_absence_before_receipt",
    }
    if set(qa) != expected_qa_keys:
        raise R2ContractError("independent QA field set differs")
    expected_qa = config["qa_receipt_contract"]
    if qa["schema_version"] != expected_qa["schema_version"]:
        raise R2ContractError("independent QA schema differs")
    if qa["experiment_id"] != EXPERIMENT_ID:
        raise R2ContractError("independent QA experiment differs")
    _require_aware_timestamp(qa["recorded_at_kst"], label="independent QA")
    if not isinstance(qa["reviewer"], str) or not qa["reviewer"]:
        raise R2ContractError("independent QA reviewer identity is missing")
    if qa["reviewer_independent_of_implementation_owner"] is not True:
        raise R2ContractError("independent QA reviewer independence is false")
    if qa["verdict"] != "GO" or qa["p0_count"] != 0 or qa["p1_count"] != 0:
        raise R2ContractError("independent QA is not P0=0/P1=0 GO")
    if qa["findings"] != [] or qa["static_lineage"] != lineage:
        raise R2ContractError("independent QA deep static lineage differs")
    absence = qa["output_control_lock_absence_before_receipt"]
    if absence != {
        "output": True,
        "control": True,
        "authorization": True,
        "attempt_lock": True,
        "run_failure_receipt": True,
    }:
        raise R2ContractError("independent QA absence assertions differ")
    qa_sha = sha256_bytes(qa_raw)

    auth, auth_raw = _load_json(auth_path, label="execution authorization")
    expected_auth_keys = {
        "schema_version",
        "experiment_id",
        "recorded_at_kst",
        "authorizer",
        "qa_sha256",
        "static_lineage",
        "execute_once",
        "candidate_or_test_prediction_allowed",
        "registry_append_allowed",
        "upload_allowed",
    }
    if set(auth) != expected_auth_keys:
        raise R2ContractError("execution authorization field set differs")
    expected_auth = config["authorization_contract"]
    if auth["schema_version"] != expected_auth["schema_version"]:
        raise R2ContractError("execution authorization schema differs")
    _require_aware_timestamp(auth["recorded_at_kst"], label="execution authorization")
    if not isinstance(auth["authorizer"], str) or not auth["authorizer"]:
        raise R2ContractError("execution authorizer identity is missing")
    if auth["experiment_id"] != EXPERIMENT_ID or auth["qa_sha256"] != qa_sha:
        raise R2ContractError("execution authorization lineage differs")
    if auth["static_lineage"] != lineage:
        raise R2ContractError("execution authorization static lineage differs")
    permissions = {
        "execute_once": True,
        "candidate_or_test_prediction_allowed": False,
        "registry_append_allowed": False,
        "upload_allowed": False,
    }
    if any(auth[key] is not value for key, value in permissions.items()):
        raise R2ContractError("execution authorization permissions differ")
    return qa, qa_sha, auth, sha256_bytes(auth_raw), lineage


def robust_write_exclusive(path: Path, payload: bytes) -> None:
    """Write one regular file with O_EXCL/O_BINARY and a complete write loop."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("exclusive writer made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    assert_plain_existing(path, kind="file")


def _expected_lock_payload(
    *,
    root: Path,
    config: Mapping[str, Any],
    qa_sha: str,
    auth_sha: str,
    lineage: Mapping[str, Any],
    created_at_kst: str,
    execution_nonce: str,
    process_id: int,
) -> dict[str, Any]:
    return {
        "schema_version": config["attempt_lock_contract"]["schema_version"],
        "experiment_id": EXPERIMENT_ID,
        "created_at_kst": created_at_kst,
        "execution_nonce": execution_nonce,
        "process_id": process_id,
        "root_identity": lineage["workspace"],
        "config": lineage["config"],
        "science": lineage["science"],
        "qa_sha256": qa_sha,
        "authorization_sha256": auth_sha,
        "implementation_pins": lineage["implementation_pins"],
        "immutable_inputs": lineage["immutable_inputs"],
        "runtime": lineage["runtime"],
        "central_v9_anchor": lineage["central_v9_anchor"],
        "output_relative_path": config["canonical_paths"]["output"],
        "capability_phase_order": list(PHASE_ORDER),
        "permissions": {
            "execute_once": True,
            "candidate_or_test_prediction_allowed": False,
            "registry_append_allowed": False,
            "upload_allowed": False,
        },
    }


def create_attempt_lock(
    root: Path,
    data_dir: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str, str, dict[str, Any]]:
    """Verify QA/auth first, then create and full-deep-verify the one-shot lock."""

    _qa, qa_sha, _auth, auth_sha, lineage = verify_execution_documents(
        root, data_dir, config
    )
    if _control_inventory(root, config) != [
        "authorization.json",
        "pre_execution_qa.json",
    ]:
        raise R2ContractError("pre-lock control inventory is not exactly QA plus authorization")
    output = root / config["canonical_paths"]["output"]
    if _lexists(output):
        raise FileExistsError("canonical Gen6r2 output already exists")
    lock = assert_contained(
        root,
        root / config["canonical_paths"]["attempt_lock"],
        must_exist=False,
    )
    if _lexists(lock):
        raise FileExistsError("canonical Gen6r2 attempt lock already exists")
    lock.parent.mkdir(parents=False, exist_ok=True)
    assert_plain_existing(lock.parent, kind="directory")
    created = datetime.now().astimezone().isoformat()
    nonce = secrets.token_hex(32)
    expected = _expected_lock_payload(
        root=root,
        config=config,
        qa_sha=qa_sha,
        auth_sha=auth_sha,
        lineage=lineage,
        created_at_kst=created,
        execution_nonce=nonce,
        process_id=os.getpid(),
    )
    raw = canonical_json_bytes(expected) + b"\n"
    robust_write_exclusive(lock, raw)
    reread = lock.read_bytes()
    parsed = json.loads(reread)
    if reread != raw or parsed != expected or deep_sha256(parsed) != deep_sha256(expected):
        raise R2ContractError("full expected attempt-lock payload differs after reread")
    return expected, sha256_bytes(raw), qa_sha, auth_sha, lineage


class _LiveCapability:
    __slots__ = ("__identity",)

    def __init__(self, identity: object, sentinel: object) -> None:
        if sentinel is not _CAPABILITY_SENTINEL:
            raise R2CapabilityError("capability construction is private")
        self.__identity = identity


@dataclass
class _LiveRecord:
    capability: _LiveCapability
    identity: object
    root: Path
    data_dir: Path
    config: dict[str, Any]
    phase: str
    lock_payload: dict[str, Any]
    lock_sha256: str
    qa_sha256: str
    authorization_sha256: str
    static_lineage: dict[str, Any]
    process_id: int
    revoked: bool = False


_CAPABILITY_SENTINEL = object()
_REGISTRY_LOCK = threading.RLock()
_LIVE_REGISTRY: dict[int, _LiveRecord] = {}


def _verify_existing_lock(
    *,
    root: Path,
    data_dir: Path,
    config: Mapping[str, Any],
    require_output_absent: bool,
    require_current_process: bool,
) -> tuple[dict[str, Any], bytes, str, str, dict[str, Any]]:
    """Reconstruct and deep-verify every field of the canonical attempt lock."""

    current, _raw = load_canonical_config(root, root / CONFIG_RELATIVE)
    if current != config:
        raise R2CapabilityError("lock config differs from canonical deep JSON")
    _qa, qa_sha, _auth, auth_sha, lineage = verify_execution_documents(
        root, data_dir, current
    )
    lock_path = assert_contained(
        root,
        root / current["canonical_paths"]["attempt_lock"],
        must_exist=True,
    )
    assert_plain_existing(lock_path, kind="file")
    lock, lock_raw = _load_json(lock_path, label="attempt lock")
    if lock_raw != canonical_json_bytes(lock) + b"\n":
        raise R2CapabilityError("attempt lock is not canonical LF-terminated JSON")
    nonce = lock.get("execution_nonce")
    created = lock.get("created_at_kst")
    process_id = lock.get("process_id")
    if not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{64}", nonce):
        raise R2CapabilityError("attempt-lock execution nonce differs")
    try:
        _require_aware_timestamp(created, label="attempt lock")
    except R2ContractError as exception:
        raise R2CapabilityError("attempt-lock creation timestamp differs") from exception
    if (
        not isinstance(process_id, int)
        or isinstance(process_id, bool)
        or process_id <= 0
        or (require_current_process and process_id != os.getpid())
    ):
        raise R2CapabilityError("attempt-lock process identity differs")
    expected = _expected_lock_payload(
        root=root,
        config=current,
        qa_sha=qa_sha,
        auth_sha=auth_sha,
        lineage=lineage,
        created_at_kst=created,
        execution_nonce=nonce,
        process_id=process_id,
    )
    if lock != expected or deep_sha256(lock) != deep_sha256(expected):
        raise R2CapabilityError("full expected attempt-lock payload differs")
    inventory = _control_inventory(root, current)
    if inventory != [
        "attempt.lock",
        "authorization.json",
        "pre_execution_qa.json",
    ]:
        raise R2CapabilityError("live control inventory differs")
    output = root / current["canonical_paths"]["output"]
    if require_output_absent and _lexists(output):
        raise R2CapabilityError("output exists before capability issuance")
    _assert_output_subset(root, current, final=False)
    return lock, lock_raw, qa_sha, auth_sha, lineage


def issue_execution_capability(
    *,
    root: Path,
    data_dir: Path,
    config: Mapping[str, Any],
    lock_payload: Mapping[str, Any],
    lock_sha256: str,
    qa_sha256: str,
    authorization_sha256: str,
    static_lineage: Mapping[str, Any],
) -> object:
    if not LOWER_SHA_RE.fullmatch(lock_sha256):
        raise R2CapabilityError("attempt-lock digest is invalid")
    canonical_config, _raw = load_canonical_config(root, root / CONFIG_RELATIVE)
    if canonical_config != config:
        raise R2CapabilityError("capability config differs from canonical deep JSON")
    observed_lock, lock_raw, observed_qa, observed_auth, observed_lineage = (
        _verify_existing_lock(
            root=root,
            data_dir=data_dir,
            config=canonical_config,
            require_output_absent=True,
            require_current_process=True,
        )
    )
    if (
        sha256_bytes(lock_raw) != lock_sha256
        or observed_lock != lock_payload
        or observed_qa != qa_sha256
        or observed_auth != authorization_sha256
        or observed_lineage != static_lineage
    ):
        raise R2CapabilityError("attempt lock differs before capability issuance")
    identity = object()
    capability = _LiveCapability(identity, _CAPABILITY_SENTINEL)
    record = _LiveRecord(
        capability=capability,
        identity=identity,
        root=root.resolve(strict=True),
        data_dir=data_dir.resolve(strict=True),
        config=deepcopy(canonical_config),
        phase=PHASE_ORDER[0],
        lock_payload=deepcopy(observed_lock),
        lock_sha256=lock_sha256,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
        static_lineage=deepcopy(observed_lineage),
        process_id=os.getpid(),
    )
    with _REGISTRY_LOCK:
        if id(capability) in _LIVE_REGISTRY:
            raise R2CapabilityError("capability object identity collision")
        _LIVE_REGISTRY[id(capability)] = record
    return capability


def _record_for(capability: object) -> _LiveRecord:
    with _REGISTRY_LOCK:
        record = _LIVE_REGISTRY.get(id(capability))
        if record is None or record.capability is not capability:
            raise R2CapabilityError("direct or forged capability call rejected")
        identity = object.__getattribute__(capability, "_LiveCapability__identity")
        if identity is not record.identity:
            raise R2CapabilityError("capability opaque identity changed")
        if record.revoked or record.phase == "REVOKED":
            raise R2CapabilityError("revoked capability call rejected")
        if record.process_id != os.getpid():
            raise R2CapabilityError("capability process identity changed")
        return record


def _revalidate_live(record: _LiveRecord) -> None:
    config, raw = load_canonical_config(record.root, record.root / CONFIG_RELATIVE)
    if config != record.config:
        raise R2CapabilityError("live canonical config deep equality failed")
    _verify_workspace(record.root, config)
    _verify_environment(record.root, record.data_dir, config)
    runtime = _verify_runtime(record.root, config)
    ledger = verify_central_ledger(record.root, config)
    science = _verify_science(record.root, config)
    predecessor = _verify_superseded_v1(record.root, config)
    inputs = _verify_inputs(record.root, record.data_dir, config)
    implementation = implementation_pins(record.root)
    lock, lock_raw, qa_sha, auth_sha, lineage = _verify_existing_lock(
        root=record.root,
        data_dir=record.data_dir,
        config=config,
        require_output_absent=False,
        require_current_process=True,
    )
    if qa_sha != record.qa_sha256 or auth_sha != record.authorization_sha256:
        raise R2CapabilityError("live QA or authorization changed")
    if lineage != record.static_lineage:
        raise R2CapabilityError("live static lineage changed")
    if (
        runtime != record.static_lineage["runtime"]
        or ledger != record.static_lineage["central_v9_anchor"]
        or science != record.static_lineage["science"]
        or predecessor != record.static_lineage["superseded_v1"]
        or inputs != record.static_lineage["immutable_inputs"]
        or implementation != record.static_lineage["implementation_pins"]
        or deep_sha256(config) != deep_sha256(record.config)
        or sha256_bytes(raw) != EXPECTED_CONFIG_SHA256
    ):
        raise R2CapabilityError("live lineage revalidation failed")
    if (
        sha256_bytes(lock_raw) != record.lock_sha256
        or lock != record.lock_payload
        or deep_sha256(lock) != deep_sha256(record.lock_payload)
    ):
        raise R2CapabilityError("live full attempt-lock payload changed")
    _assert_output_subset(record.root, config, final=False)


def enter_engine_phase(
    capability: object,
    *,
    expected_phase: str,
    next_phase: str,
    entry_name: str,
) -> dict[str, Any]:
    """Consume one phase exactly once after full live-lineage revalidation."""

    if expected_phase not in PHASE_ORDER or next_phase not in PHASE_ORDER:
        raise R2CapabilityError("unknown capability phase")
    if PHASE_ORDER.index(next_phase) != PHASE_ORDER.index(expected_phase) + 1:
        raise R2CapabilityError("non-adjacent capability phase transition")
    with _REGISTRY_LOCK:
        record = _record_for(capability)
        if record.phase != expected_phase:
            raise R2CapabilityError(
                f"phase replay/order rejection at {entry_name}: {record.phase}"
            )
        _revalidate_live(record)
        record.phase = next_phase
        return {
            "entry": entry_name,
            "consumed_phase": expected_phase,
            "live_phase": next_phase,
            "lock_sha256": record.lock_sha256,
            "qa_sha256": record.qa_sha256,
            "authorization_sha256": record.authorization_sha256,
            "implementation_pins": record.static_lineage["implementation_pins"],
            "central_v9_anchor": record.static_lineage["central_v9_anchor"],
        }


def verify_live_phase(capability: object, *, phase: str, entry_name: str) -> dict[str, Any]:
    with _REGISTRY_LOCK:
        record = _record_for(capability)
        if record.phase != phase:
            raise R2CapabilityError(f"wrong live phase at {entry_name}: {record.phase}")
        _revalidate_live(record)
        return {
            "entry": entry_name,
            "live_phase": phase,
            "lock_sha256": record.lock_sha256,
        }


def capability_context(capability: object) -> dict[str, Any]:
    record = _record_for(capability)
    return {
        "root": record.root,
        "data_dir": record.data_dir,
        "config": deepcopy(record.config),
        "phase": record.phase,
        "lock_payload": deepcopy(record.lock_payload),
        "lock_sha256": record.lock_sha256,
        "qa_sha256": record.qa_sha256,
        "authorization_sha256": record.authorization_sha256,
        "static_lineage": deepcopy(record.static_lineage),
    }


def revoke_capability(capability: object, *, expected_phase: str) -> None:
    with _REGISTRY_LOCK:
        record = _record_for(capability)
        if record.phase != expected_phase:
            raise R2CapabilityError("capability revoke phase differs")
        _revalidate_live(record)
        record.phase = "REVOKED"
        record.revoked = True


def revoke_capability_after_failure(capability: object) -> None:
    with _REGISTRY_LOCK:
        record = _LIVE_REGISTRY.get(id(capability))
        if record is not None and record.capability is capability:
            record.phase = "REVOKED"
            record.revoked = True


def capability_registry_snapshot(capability: object) -> dict[str, Any]:
    with _REGISTRY_LOCK:
        record = _LIVE_REGISTRY.get(id(capability))
        if record is None or record.capability is not capability:
            return {"registered": False}
        return {
            "registered": True,
            "phase": record.phase,
            "revoked": record.revoked,
            "object_identity_exact": record.capability is capability,
            "process_id": record.process_id,
        }


def create_output_directories(capability: object, *, phase: str) -> Path:
    verify_live_phase(capability, phase=phase, entry_name="create_output_directories")
    context = capability_context(capability)
    root: Path = context["root"]
    config = context["config"]
    output = assert_contained(root, root / config["canonical_paths"]["output"], must_exist=False)
    output.mkdir(parents=False, exist_ok=False)
    assert_plain_existing(output, kind="directory")
    for relative in ALLOWED_DIRECTORIES[1:]:
        child = output / relative
        child.mkdir(parents=False, exist_ok=False)
        assert_plain_existing(child, kind="directory")
    _assert_output_subset(root, config, final=False)
    return output


def write_output_exclusive(
    capability: object,
    *,
    phase: str,
    relative_path: str,
    payload: bytes,
) -> dict[str, Any]:
    verify_live_phase(capability, phase=phase, entry_name=f"write:{relative_path}")
    if relative_path not in ALLOWED_FILES:
        raise R2ContractError("output write path is not allowlisted")
    if WRITE_PHASES[relative_path] != phase:
        raise R2CapabilityError("output write is not authorized in this phase")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise R2ContractError("output write path is not canonical relative POSIX")
    context = capability_context(capability)
    root: Path = context["root"]
    output = root / context["config"]["canonical_paths"]["output"]
    target = assert_contained(root, output / Path(*pure.parts), must_exist=False)
    if target.parent.relative_to(output).as_posix() not in ALLOWED_DIRECTORIES:
        raise R2ContractError("output parent directory is not allowlisted")
    robust_write_exclusive(target, payload)
    reread = target.read_bytes()
    if reread != payload:
        raise R2ContractError("exclusive output reread differs")
    _assert_output_subset(root, context["config"], final=False)
    return {
        "path": relative_path,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def write_failure_receipt(
    *, root: Path, config: Mapping[str, Any], exception: BaseException
) -> dict[str, Any]:
    path = assert_contained(
        root,
        root / config["canonical_paths"]["run_failure_receipt"],
        must_exist=False,
    )
    assert_plain_existing(path.parent, kind="directory")
    payload = {
        "schema_version": (
            "p3_gen6_incumbent_preserving_residual_calibrator.run_failure_receipt.r2.v1"
        ),
        "experiment_id": EXPERIMENT_ID,
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "exception_type": type(exception).__name__,
        "message_sha256": sha256_bytes(str(exception).encode("utf-8")),
        "capability_revoked": True,
        "candidate_created": False,
        "test_prediction_created": False,
        "registry_appended": False,
        "uploads": 0,
    }
    raw = canonical_json_bytes(payload) + b"\n"
    robust_write_exclusive(path, raw)
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_bytes(raw)}


def _verify_fold_commitments(output: Path) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    all_receipts: list[dict[str, Any]] = []
    total_cases = 0
    total_fits = 0
    prior_target_decodes = 0
    for index, fold in enumerate(FOLD_ORDER):
        stem = f"fold_{index:02d}_{fold}"
        prediction = output / f"blind/{stem}.npy"
        commitment_path = output / f"commitments/{stem}.json"
        commitment, raw = _load_json(commitment_path, label=f"fold {index} commitment")
        if raw != canonical_json_bytes(commitment) + b"\n":
            raise R2ContractError("fold commitment is not canonical LF-terminated JSON")
        if set(commitment) != {
            "schema_version",
            "fold_index",
            "fold",
            "prefix_fractions",
            "case_count",
            "row_count",
            "key_sha256",
            "validation_ids_sha256",
            "blind_prediction",
            "cell_receipts",
            "fit_count",
            "target_scalar_decodes_before_fold_commitment",
            "active_fold_target_scalar_decodes_before_commitment",
            "truth_attached",
            "candidate_or_test_prediction",
        }:
            raise R2ContractError("fold commitment field set differs")
        if (
            commitment["schema_version"]
            != "p3_gen6_incumbent_preserving_residual_calibrator.blind_fold_commitment.r2.v1"
            or commitment["fold_index"] != index
            or commitment["fold"] != fold
            or tuple(commitment["prefix_fractions"]) != PREFIX_FRACTIONS
        ):
            raise R2ContractError("sealed fold commitment identity differs")
        if commitment["blind_prediction"] != file_pin(prediction, root=output):
            raise R2ContractError("sealed blind prediction pin differs")
        cases = int(commitment["case_count"])
        rows = int(commitment["row_count"])
        receipts = commitment["cell_receipts"]
        if cases <= 0 or rows != cases * len(PREFIX_FRACTIONS) * 6:
            raise R2ContractError("sealed fold commitment row count differs")
        if not isinstance(receipts, list) or len(receipts) != len(PREFIX_FRACTIONS):
            raise R2ContractError("sealed fold cell-receipt count differs")
        if [float(item["prefix_fraction"]) for item in receipts] != list(
            PREFIX_FRACTIONS
        ) or any(item["outer_fold"] != fold for item in receipts):
            raise R2ContractError("sealed fold cell-receipt identity differs")
        observed_fits = sum(int(item["fit_count"]) for item in receipts)
        if (
            any(int(item["fit_count"]) not in {0, 1, 2} for item in receipts)
            or int(commitment["fit_count"]) != observed_fits
            or observed_fits > 10
        ):
            raise R2ContractError("sealed fold fit-count semantics differ")
        if index == 0 and observed_fits != 0:
            raise R2ContractError("first fold unexpectedly fit a calibrator")
        if (
            int(commitment["target_scalar_decodes_before_fold_commitment"])
            != prior_target_decodes
            or int(commitment["active_fold_target_scalar_decodes_before_commitment"])
            != 0
            or commitment["truth_attached"] is not False
            or commitment["candidate_or_test_prediction"] is not False
        ):
            raise R2ContractError("fold commitment blind-target ordering differs")
        if not LOWER_SHA_RE.fullmatch(str(commitment["key_sha256"])) or not LOWER_SHA_RE.fullmatch(
            str(commitment["validation_ids_sha256"])
        ):
            raise R2ContractError("fold commitment key digest differs")
        saved = np.load(prediction, allow_pickle=False)
        if saved.dtype != np.dtype("float64") or saved.shape != (rows,):
            raise R2ContractError("sealed blind prediction dtype/shape differs")
        if not np.isfinite(saved).all():
            raise R2ContractError("sealed blind prediction is non-finite")
        folds.append(
            {
                "path": commitment_path.relative_to(output).as_posix(),
                "sha256": sha256_file(commitment_path),
                "blind_prediction": commitment["blind_prediction"],
            }
        )
        total_cases += cases
        total_fits += observed_fits
        prior_target_decodes += cases * 6
        all_receipts.extend(receipts)
    complete_path = output / "commitments/predictions_complete.json"
    complete, complete_raw = _load_json(
        complete_path, label="predictions-complete commitment"
    )
    if complete_raw != canonical_json_bytes(complete) + b"\n":
        raise R2ContractError("predictions-complete is not canonical LF-terminated JSON")
    if set(complete) != {
        "schema_version",
        "fold_order",
        "fold_commitments",
        "fold_0_and_1_truth_released_only_after_own_commitment",
        "fold_2_truth_released",
        "fit_count_authorized_upper_bound",
        "fit_count_observed_so_far",
        "scoring_truth_attached",
        "anonymous_test_value_reads",
        "candidate_or_test_prediction",
    }:
        raise R2ContractError("predictions-complete field set differs")
    if (
        complete["schema_version"]
        != "p3_gen6_incumbent_preserving_residual_calibrator.predictions_complete.r2.v1"
        or complete["fold_order"] != list(FOLD_ORDER)
        or complete["fold_commitments"] != folds
        or complete["fold_0_and_1_truth_released_only_after_own_commitment"] is not True
        or complete["fold_2_truth_released"] is not False
        or complete["fit_count_authorized_upper_bound"] != 20
        or complete["fit_count_observed_so_far"] != total_fits
        or complete["scoring_truth_attached"] is not False
        or complete["anonymous_test_value_reads"] != 0
        or complete["candidate_or_test_prediction"] is not False
    ):
        raise R2ContractError("predictions-complete aggregate differs")
    if total_cases != 181 or total_fits > 20 or prior_target_decodes != 1086:
        raise R2ContractError("predictions-complete aggregate counts differ")
    return {
        "fold_commitments": folds,
        "predictions_complete": file_pin(complete_path, root=output),
        "validation_cases": total_cases,
        "prediction_rows": total_cases * len(PREFIX_FRACTIONS) * 6,
        "fit_count_observed_exact": total_fits,
        "source_target_scalar_decodes_after_all_release": prior_target_decodes,
        "_cell_receipts": all_receipts,
    }


def verify_published_output(
    root: Path,
    data_dir: Path,
    *,
    requested_config: Path | None = None,
) -> dict[str, Any]:
    """Independent, registry-free, read-only verifier for a sealed final output."""

    workspace = assert_plain_existing(root, kind="directory")
    source = assert_plain_existing(data_dir, kind="directory")
    config, raw = load_canonical_config(workspace, requested_config)
    _verify_environment(workspace, source, config)
    _verify_workspace(workspace, config)
    lock, lock_raw, qa_sha, auth_sha, lineage = _verify_existing_lock(
        root=workspace,
        data_dir=source,
        config=config,
        require_output_absent=False,
        require_current_process=False,
    )
    runtime = lineage["runtime"]
    ledger = lineage["central_v9_anchor"]
    science = lineage["science"]
    predecessor = lineage["superseded_v1"]
    inputs = lineage["immutable_inputs"]
    implementation = lineage["implementation_pins"]
    output = workspace / config["canonical_paths"]["output"]
    tree = _assert_output_subset(workspace, config, final=True)
    commitment_report = _verify_fold_commitments(output)
    cell_receipts = commitment_report.pop("_cell_receipts")

    manifest_path = output / "manifest.json"
    models, _models_raw = _load_canonical_json(
        output / "calibrator_models.json", label="calibrator models"
    )
    evidence, _evidence_raw = _load_canonical_json(
        output / "learning_curve_evidence.json", label="learning-curve evidence"
    )
    metrics, _metrics_raw = _load_canonical_json(
        output / "metrics.json", label="metrics"
    )
    manifest, manifest_raw = _load_canonical_json(manifest_path, label="manifest")
    core = {
        relative: file_pin(output / relative, root=output) for relative in CORE_FILES
    }
    expected_manifest_lineage = {
        "config": {
            "path": CONFIG_RELATIVE,
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "deep_sha256": deep_sha256(config),
        },
        "science": science,
        "superseded_v1": predecessor,
        "qa_sha256": qa_sha,
        "authorization_sha256": auth_sha,
        "attempt_lock": {
            "path": config["canonical_paths"]["attempt_lock"],
            "bytes": len(lock_raw),
            "sha256": sha256_bytes(lock_raw),
            "deep_sha256": deep_sha256(lock),
        },
        "implementation_pins": implementation,
        "immutable_inputs": inputs,
        "runtime": runtime,
        "central_v9_anchor": ledger,
    }
    if set(manifest) != {
        "schema_version",
        "experiment_id",
        "created_at_kst",
        "status",
        "lineage",
        "core_files",
        "fit_count_authorized_upper_bound",
        "fit_count_observed_exact",
        "candidate_created",
        "test_prediction_created",
        "registry_appended",
        "official_upload_count",
    }:
        raise R2ContractError("manifest field set differs")
    _require_aware_timestamp(manifest["created_at_kst"], label="manifest")
    if (
        manifest["schema_version"]
        != "p3_gen6_incumbent_preserving_residual_calibrator.manifest.r2.v1"
        or manifest["experiment_id"] != EXPERIMENT_ID
        or manifest["status"] != metrics.get("status")
        or manifest["fit_count_authorized_upper_bound"] != 20
        or manifest["fit_count_observed_exact"]
        != commitment_report["fit_count_observed_exact"]
        or manifest["candidate_created"] is not False
        or manifest["test_prediction_created"] is not False
        or manifest["registry_appended"] is not False
        or manifest["official_upload_count"] != 0
    ):
        raise R2ContractError("manifest fixed semantics differ")
    if manifest["lineage"] != expected_manifest_lineage:
        raise R2ContractError("manifest full lineage differs")
    if manifest["core_files"] != core:
        raise R2ContractError("manifest core-file map differs")
    sidecar = output / "manifest.sha256"
    expected_sidecar = f"{sha256_bytes(manifest_raw)}  manifest.json\n".encode("ascii")
    if sidecar.read_bytes() != expected_sidecar:
        raise R2ContractError("manifest sidecar differs")
    seal, seal_raw = _load_canonical_json(output / "seal.json", label="seal")
    expected_seal_lineage = {
        "qa_sha256": qa_sha,
        "authorization_sha256": auth_sha,
        "attempt_lock_sha256": sha256_bytes(lock_raw),
        "implementation_pins": implementation,
        "immutable_inputs": inputs,
        "runtime": runtime,
        "central_v9_anchor": ledger,
    }
    if set(seal) != {
        "schema_version",
        "experiment_id",
        "created_at_kst",
        "status",
        "lineage",
        "manifest",
        "manifest_sidecar",
        "allowed_directories",
        "allowed_files",
        "candidate_created",
        "test_prediction_created",
        "registry_appended",
        "uploads",
    }:
        raise R2ContractError("seal field set differs")
    _require_aware_timestamp(seal["created_at_kst"], label="seal")
    if (
        seal["schema_version"]
        != "p3_gen6_incumbent_preserving_residual_calibrator.output_seal.r2.v1"
        or seal["experiment_id"] != EXPERIMENT_ID
        or seal["status"] != "SEALED_RESEARCH_ONLY"
    ):
        raise R2ContractError("seal fixed semantics differ")
    if seal["lineage"] != expected_seal_lineage:
        raise R2ContractError("seal full lineage differs")
    if seal["manifest"] != file_pin(manifest_path, root=output):
        raise R2ContractError("seal manifest pin differs")
    if seal["manifest_sidecar"] != file_pin(sidecar, root=output):
        raise R2ContractError("seal sidecar pin differs")
    if seal["allowed_directories"] != list(ALLOWED_DIRECTORIES) or seal[
        "allowed_files"
    ] != list(ALLOWED_FILES):
        raise R2ContractError("seal exact allowlist differs")
    if seal["candidate_created"] is not False or seal["test_prediction_created"] is not False:
        raise R2ContractError("seal candidate/test prohibition differs")
    if seal["registry_appended"] is not False or seal["uploads"] != 0:
        raise R2ContractError("seal registry/upload prohibition differs")
    expected_oof_columns = (
        "prefix_fraction",
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "current_hs",
        "persistence",
        "incumbent_prediction",
        "target_hs",
        "gen6_prediction",
    )
    oof_path = output / "oof.parquet"
    oof = pq.ParquetFile(oof_path)
    if oof.metadata.num_rows != 5430 or tuple(oof.schema_arrow.names) != expected_oof_columns:
        raise R2ContractError("sealed Gen6r2 OOF metadata differs")
    table = pq.read_table(
        oof_path,
        columns=[
            "prefix_fraction",
            "fold",
            "incumbent_prediction",
            "target_hs",
            "gen6_prediction",
        ],
    )
    prefix_values = table["prefix_fraction"].to_numpy(zero_copy_only=False)
    fold_values = np.asarray(table["fold"].to_pylist(), dtype=object)
    incumbent_values = table["incumbent_prediction"].to_numpy(zero_copy_only=False)
    target_values = table["target_hs"].to_numpy(zero_copy_only=False)
    candidate_values = table["gen6_prediction"].to_numpy(zero_copy_only=False)
    if not np.isfinite(target_values).all() or not np.isfinite(candidate_values).all():
        raise R2ContractError("sealed Gen6r2 OOF contains non-finite values")
    for index, fold in enumerate(FOLD_ORDER):
        saved = np.load(output / f"blind/fold_{index:02d}_{fold}.npy", allow_pickle=False)
        if not np.array_equal(candidate_values[fold_values == fold], saved):
            raise R2ContractError("sealed OOF differs from durable blind predictions")
    for receipt in cell_receipts:
        mask = (fold_values == receipt["outer_fold"]) & (
            prefix_values == float(receipt["prefix_fraction"])
        )
        incumbent_cell = np.ascontiguousarray(incumbent_values[mask], dtype=np.float64)
        candidate_cell = np.ascontiguousarray(candidate_values[mask], dtype=np.float64)
        if len(candidate_cell) == 0:
            raise R2ContractError("sealed cell receipt has no OOF rows")
        if "IDENTITY" in str(receipt["decision"]) and (
            candidate_cell.tobytes() != incumbent_cell.tobytes()
            or receipt.get("identity_bytes_equal") is not True
        ):
            raise R2ContractError("sealed identity fallback is not byte-exact")
        if "APPLY_BOUNDED_CORRECTION" in str(receipt["decision"]) and float(
            np.max(np.abs(candidate_cell - incumbent_cell), initial=0.0)
        ) > 0.120000000001:
            raise R2ContractError("sealed correction exceeds the fixed bound")

    observed_fits = commitment_report["fit_count_observed_exact"]
    record_identities = [
        (
            float(record.get("prefix_fraction", -1.0)),
            int(record.get("fold_index", -1)),
            str(record.get("outer_fold", "")),
        )
        for record in models.get("records", [])
    ]
    expected_record_identities = [
        (prefix, index, fold)
        for index, fold in enumerate(FOLD_ORDER)
        for prefix in PREFIX_FRACTIONS
    ]
    if (
        models.get("schema_version")
        != "p3_gen6_incumbent_preserving_residual_calibrator.models.r2.v1"
        or models.get("experiment_id") != EXPERIMENT_ID
        or models.get("science_deep_sha256") != EXPECTED_SCIENCE_DEEP_SHA256
        or models.get("fit_count_observed_exact") != observed_fits
        or models.get("fit_count_authorized_upper_bound") != 20
        or not isinstance(models.get("records"), list)
        or len(models["records"]) != 15
        or record_identities != expected_record_identities
        or sum(int(record["fit_count"]) for record in models["records"])
        != observed_fits
    ):
        raise R2ContractError("sealed calibrator-model receipt semantics differ")
    fit_contract = metrics.get("fit_count_contract", {})
    target_audit = metrics.get("target_access_audit", {})
    _require_aware_timestamp(metrics.get("created_at_kst"), label="metrics")
    if (
        metrics.get("schema_version")
        != "p3_gen6_incumbent_preserving_residual_calibrator.metrics.r2.v1"
        or metrics.get("experiment_id") != EXPERIMENT_ID
        or fit_contract.get("authorized_upper_bound") != 20
        or fit_contract.get("observed_exact") != observed_fits
        or fit_contract.get("observed_equals_cell_receipt_sum") is not True
        or target_audit.get("released_folds") != list(FOLD_ORDER)
        or target_audit.get("unique_source_target_scalar_decodes") != 1086
        or target_audit.get("expected_total_source_target_scalar_decodes_after_all_release")
        != 1086
        or target_audit.get("float_target_decodes_during_identity_index") != 0
        or target_audit.get("forbidden_release_attempts") != 0
        or target_audit.get("anonymous_test_value_reads") != 0
        or metrics.get("inner_gate_receipts") != cell_receipts
        or metrics.get("predictions_complete")
        != commitment_report["predictions_complete"]
        or metrics.get("full_fit_performed") is not False
        or metrics.get("candidate_created") is not False
        or metrics.get("test_prediction_created") is not False
        or metrics.get("official_promotion_allowed") is not False
        or metrics.get("registry_append_allowed") is not False
        or metrics.get("official_upload_count") != 0
        or metrics.get("status")
        not in {
            "LOCAL_CURVE_QUALIFIED_RESEARCH_ONLY_STOPPED_BEFORE_TEST",
            "NO_LOCAL_CURVE_QUALIFICATION_RESEARCH_ONLY_STOPPED_BEFORE_TEST",
        }
        or not all(metrics.get("leakage_checks", {}).values())
        or not all(metrics.get("reproducibility_checks", {}).values())
    ):
        raise R2ContractError("sealed metrics count or prohibition semantics differ")
    if (
        evidence.get("comparison_mode")
        != "SEALED_GEN1_OOF_INCUMBENT_PRESERVING_RESEARCH_ONLY"
        or evidence.get("official_promotion", {}).get("allowed") is not False
        or evidence.get("preregistration", {}).get("hypothesis_count") != 1
        or evidence.get("preregistration", {}).get(
            "alpha_threshold_seed_or_weight_search_count"
        )
        != 0
    ):
        raise R2ContractError("sealed evidence research-only semantics differ")
    return {
        "schema_version": (
            "p3_gen6_incumbent_preserving_residual_calibrator.post_publish_verifier.r2.v1"
        ),
        "status": "POST_PUBLISH_VERIFIED_EXACT_ALLOWLIST_AND_LINEAGE",
        "tree": tree,
        "commitments": commitment_report,
        "manifest": file_pin(manifest_path, root=output),
        "manifest_sidecar": file_pin(sidecar, root=output),
        "seal": {
            **file_pin(output / "seal.json", root=output),
            "deep_sha256": deep_sha256(seal),
            "bytes_rechecked": len(seal_raw),
        },
        "qa_sha256": qa_sha,
        "authorization_sha256": auth_sha,
        "attempt_lock_sha256": sha256_bytes(lock_raw),
        "central_v9_anchor": ledger,
        "candidate_created": False,
        "test_prediction_created": False,
        "registry_appended": False,
        "uploads": 0,
    }


__all__ = [
    "ALLOWED_DIRECTORIES",
    "ALLOWED_FILES",
    "CONFIG_RELATIVE",
    "CORE_FILES",
    "EXPECTED_CONFIG_BYTES",
    "EXPECTED_CONFIG_DEEP_SHA256",
    "EXPECTED_CONFIG_SHA256",
    "EXPERIMENT_ID",
    "FOLD_ORDER",
    "PHASE_ORDER",
    "R2CapabilityError",
    "R2ContractError",
    "R2LedgerChanged",
    "THREAD_ENVIRONMENT",
    "WRITE_PHASES",
    "capability_context",
    "capability_registry_snapshot",
    "canonical_json_bytes",
    "create_attempt_lock",
    "create_output_directories",
    "deep_sha256",
    "enter_engine_phase",
    "file_pin",
    "implementation_pins",
    "issue_execution_capability",
    "load_canonical_config",
    "revoke_capability",
    "revoke_capability_after_failure",
    "robust_write_exclusive",
    "sha256_file",
    "static_preflight",
    "verify_central_ledger",
    "verify_execution_documents",
    "verify_live_phase",
    "verify_published_output",
    "write_failure_receipt",
    "write_output_exclusive",
]
