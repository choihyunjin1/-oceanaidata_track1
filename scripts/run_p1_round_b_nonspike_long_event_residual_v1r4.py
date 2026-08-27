"""Externally authorized, parent-supervised P1 Round-B residual screen.

The file imports only the Python standard library before external authorization
and seal verification.  Both the parent and hidden worker execute this exact
sealed file.  All numerical code and data are loaded from a private verified
snapshot; model, feature, decoder, scoring, and gate definitions remain the
immutable v1 implementation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_round_b_nonspike_long_event_residual_v1r4.json"
)
AUTHORIZATION_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_round_b_nonspike_long_event_residual_v1r4_execution_authorization.json"
)
AUTHORIZATION_ENV_VAR = "P1_V1R4_EXECUTION_AUTHORIZATION_SHA256"
AUTHORIZATION_SHA256 = "a0d737c97221a400052621065eec5fc5e4d86d23b8205690cf9cdc6a6a5b54e0"
FOLD_ORDER = ("2025_q2", "2025_q3", "2025_q4")
KEY_COLUMNS = ("station", "year", "layer", "time")
BASE_PREFIX = "event_day_balanced_binary_lgbm"
MATCHED_PREFIX = "event_day_balanced_lightgbm__default"
NUMERICAL_SECTIONS = (
    "base_config",
    "immutable_inputs",
    "surface",
    "residual_target",
    "residual_model",
    "rescue_decoder",
    "outer_protocol",
    "fail_fast_gates",
    "resource_budget",
    "interpretation",
    "prohibitions",
)
REQUIRED_DEPENDENCY_PATHS = (
    "src/p1_qc/__init__.py",
    "src/p1_qc/audit.py",
    "src/p1_qc/augment.py",
    "src/p1_qc/config.py",
    "src/p1_qc/data.py",
    "src/p1_qc/experiment.py",
    "src/p1_qc/features.py",
    "src/p1_qc/metrics.py",
    "src/p1_qc/models_tabular.py",
    "src/p1_qc/nonspike_long_event_residual.py",
    "src/p1_qc/pipeline.py",
    "src/p1_qc/postprocess.py",
    "src/p1_qc/rules.py",
    "src/p1_qc/splits.py",
    "src/p1_qc/submission.py",
    "src/p1_qc/validation.py",
    "scripts/run_p1_round_b_nonspike_long_event_residual_v1.py",
)
REQUIRED_VERIFICATION_PATHS = (
    "tests/test_run_p1_round_b_nonspike_long_event_residual_v1r4.py",
)
REQUIRED_PROVENANCE_PATHS = (
    "scripts/build_p1_v1r3_feature_key_binding.py",
)
EXPECTED_RUNTIME_VERSIONS = {
    "python": "3.12.10",
    "joblib": "1.5.3",
    "lightgbm": "4.7.0",
    "narwhals": "2.24.0",
    "numpy": "2.3.5",
    "p1-qc": "0.1.0",
    "pandas": "3.0.1",
    "psutil": "7.2.2",
    "pyarrow": "25.0.1",
    "python-dateutil": "2.9.0.post0",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.0",
    "six": "1.17.0",
    "threadpoolctl": "3.6.0",
}
PACKAGE_IMPORT_ROOTS = {
    "joblib": "joblib",
    "lightgbm": "lightgbm",
    "narwhals": "narwhals",
    "numpy": "numpy",
    "p1-qc": "p1_qc",
    "pandas": "pandas",
    "psutil": "psutil",
    "pyarrow": "pyarrow",
    "python-dateutil": "dateutil",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "six": "six",
    "threadpoolctl": "threadpoolctl",
}
FORBIDDEN_PREAUTH_ROOTS = set(PACKAGE_IMPORT_ROOTS.values())
_AUTH_LINE_PATTERN = re.compile(
    rb'(?m)^AUTHORIZATION_SHA256 = "[^"]+"$',
)


def _now_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_bound_bytes(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Read one held handle once and bind the returned bytes to its digest."""
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        value = handle.read()
        after = os.fstat(handle.fileno())
    before_signature = (before.st_size, before.st_mtime_ns, before.st_ino)
    after_signature = (after.st_size, after.st_mtime_ns, after.st_ino)
    if before_signature != after_signature or len(value) != before.st_size:
        raise RuntimeError(f"held file changed while being read: {path}")
    observed = _sha256_bytes(value)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(f"held file digest mismatch: {path}")
    return value, {
        "bytes": len(value),
        "actual_read_sha256": observed,
        "sha256": observed,
    }


def _json_load_bound(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, receipt = _read_bound_bytes(path, expected_sha256=expected_sha256)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value, receipt


def _json_load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return f"{text}\n".encode()


def _normalised_runner_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    matches = list(_AUTH_LINE_PATTERN.finditer(raw))
    if len(matches) != 1:
        raise RuntimeError("runner authorization anchor must occur exactly once")
    replacement = b'AUTHORIZATION_SHA256 = "<EXTERNAL_AUTHORIZATION_SHA256>"'
    return raw[: matches[0].start()] + replacement + raw[matches[0].end() :]


def _normalised_runner_sha256(path: Path | None = None) -> str:
    return _sha256_bytes(_normalised_runner_bytes(path or Path(__file__).resolve()))


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        flush = kernel32.FlushFileBuffers
        flush.argtypes = [wintypes.HANDLE]
        flush.restype = wintypes.BOOL
        close = kernel32.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close.restype = wintypes.BOOL
        handle = create_file(
            str(path.resolve(strict=True)),
            0x40000000,  # GENERIC_WRITE is required for FlushFileBuffers.
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,  # OPEN_EXISTING
            0x02000000,  # FILE_FLAG_BACKUP_SEMANTICS opens a directory.
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in (None, invalid):
            error = ctypes.get_last_error()
            raise OSError(error, f"CreateFileW directory flush failed: {path}")
        try:
            if not flush(handle):
                error = ctypes.get_last_error()
                raise OSError(error, f"FlushFileBuffers directory flush failed: {path}")
        finally:
            close(handle)
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_temp_create_only(temporary: Path, target: Path) -> None:
    """Atomically create target without any overwrite fallback."""
    try:
        os.link(temporary, target)
    except FileExistsError:
        raise
    except OSError as error:
        raise RuntimeError(
            "atomic hardlink create-only publication is unavailable; fail closed"
        ) from error
    _fsync_directory(target.parent)
    temporary.unlink()


def _atomic_bytes_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temp_create_only(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_new(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes_new(path, _json_bytes(value))


def _atomic_parquet_new(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp.parquet"
    try:
        with temporary.open("xb") as handle:
            frame.to_parquet(handle, index=False, compression="zstd")
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temp_create_only(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_repo_path(value: str, *, must_exist: bool = True) -> Path:
    path = (PROJECT_ROOT / value).resolve(strict=must_exist)
    if not path.is_relative_to(PROJECT_ROOT):
        raise RuntimeError(f"path escapes repository: {value}")
    return path


def _artifact_dir(config: Mapping[str, Any]) -> Path:
    path = _resolve_repo_path(str(config["artifact_dir"]), must_exist=False)
    root = (PROJECT_ROOT / "artifacts").resolve()
    if not path.is_relative_to(root):
        raise RuntimeError("artifact directory must stay below project artifacts")
    return path


def _assert_preauth_clean() -> None:
    loaded = {
        name.split(".", maxsplit=1)[0]
        for name in sys.modules
        if name.split(".", maxsplit=1)[0] in FORBIDDEN_PREAUTH_ROOTS
    }
    if loaded:
        raise RuntimeError(
            "third-party/project numerical modules loaded before authorization: "
            + ", ".join(sorted(loaded))
        )


def _project_module_path(module_name: str) -> Path | None:
    if module_name != "p1_qc" and not module_name.startswith("p1_qc."):
        return None
    candidate = PROJECT_ROOT / "src" / Path(*module_name.split("."))
    module_file = candidate.with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_file = candidate / "__init__.py"
    if package_file.is_file():
        return package_file
    raise RuntimeError(f"unresolved project import: {module_name}")


def _discover_project_dependency_closure(entrypoint: Path) -> set[str]:
    pending = [entrypoint, PROJECT_ROOT / "src" / "p1_qc" / "__init__.py"]
    visited: set[Path] = set()
    source_root = (PROJECT_ROOT / "src").resolve()
    while pending:
        path = pending.pop().resolve(strict=True)
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(source_root) if path.is_relative_to(source_root) else None
        package_parts = [] if relative is None else list(relative.parent.parts)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level and package_parts:
                    retain = len(package_parts) - (node.level - 1)
                    if retain < 0:
                        raise RuntimeError(f"invalid relative import in {path}")
                    parts = package_parts[:retain]
                    if node.module:
                        parts.extend(node.module.split("."))
                    names.append(".".join(parts))
                elif node.module:
                    names.append(node.module)
            for name in names:
                dependency = _project_module_path(name)
                if dependency is not None and dependency not in visited:
                    pending.append(dependency)
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in visited
    }


def _runtime_versions() -> dict[str, str]:
    observed = {"python": platform.python_version()}
    for distribution in EXPECTED_RUNTIME_VERSIONS:
        if distribution == "python":
            continue
        observed[distribution] = importlib.metadata.version(distribution)
    return observed


def _native_lightgbm_record(config: Mapping[str, Any]) -> dict[str, Any]:
    specification = config["trust_contract"]["lightgbm_native"]
    distribution = importlib.metadata.distribution("lightgbm")
    relative = Path(str(specification["distribution_relative_path"]))
    path = Path(distribution.locate_file(relative)).resolve(strict=True)
    record = {
        "distribution_relative_path": str(relative).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if record != specification:
        raise RuntimeError("LightGBM native binary fingerprint mismatch")
    return record


def _validate_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != (
        "p1_round_b_nonspike_long_event_residual.preregistration.v1r4"
    ):
        raise RuntimeError("unexpected preregistration schema")
    if config.get("experiment_id") != "p1_round_b_nonspike_long_event_residual_v1r4":
        raise RuntimeError("unexpected experiment id")
    if config.get("artifact_dir") != (
        "artifacts/p1_round_b_nonspike_long_event_residual_v1r4"
    ):
        raise RuntimeError("unexpected artifact namespace")
    trust = config["trust_contract"]
    if trust.get("stdlib_only_before_external_authorization") is not True:
        raise RuntimeError("stdlib-only authorization stage disabled")
    if trust.get("execution_authorization_env_var") != AUTHORIZATION_ENV_VAR:
        raise RuntimeError("authorization environment binding changed")
    if trust.get("execution_authorization_path") != str(
        AUTHORIZATION_PATH.relative_to(PROJECT_ROOT)
    ).replace("\\", "/"):
        raise RuntimeError("authorization path binding changed")
    if trust.get("runner_path") != str(Path(__file__).resolve().relative_to(PROJECT_ROOT)).replace(
        "\\", "/"
    ):
        raise RuntimeError("runner path binding changed")
    if trust.get("runner_normalized_sha256") != _normalised_runner_sha256():
        raise RuntimeError("normalized runner hash differs from config")
    if tuple(trust["project_dependency_paths"]) != REQUIRED_DEPENDENCY_PATHS:
        raise RuntimeError("project dependency list changed")
    if tuple(trust["verification_paths"]) != REQUIRED_VERIFICATION_PATHS:
        raise RuntimeError("verification-file list changed")
    if tuple(trust["provenance_paths"]) != REQUIRED_PROVENANCE_PATHS:
        raise RuntimeError("provenance-file list changed")
    if trust.get("provenance_sha256") != _sha256(
        _resolve_repo_path(REQUIRED_PROVENANCE_PATHS[0])
    ):
        raise RuntimeError("feature-key provenance builder hash changed")
    discovered = _discover_project_dependency_closure(
        _resolve_repo_path(str(trust["numerical_entrypoint_path"]))
    )
    if discovered != set(REQUIRED_DEPENDENCY_PATHS):
        raise RuntimeError("actual project import closure differs from preregistration")
    if trust["runtime_versions"] != EXPECTED_RUNTIME_VERSIONS:
        raise RuntimeError("runtime version contract changed")

    safety = config["execution_safety"]
    if safety["parent_worker_same_runner"] is not True:
        raise RuntimeError("same-runner parent/worker contract changed")
    if int(safety["parent_absolute_timeout_seconds"]) != 1800:
        raise RuntimeError("parent timeout changed")
    if int(safety["maximum_lifetime_physical_model_fits"]) != 9:
        raise RuntimeError("physical fit ceiling changed")
    if safety["claim_after_complete_readiness_only"] is not True:
        raise RuntimeError("preclaim readiness contract changed")
    if safety["feature_row_key_policy"] != (
        "REGENERATE_ALL_80_MODEL_INPUT_FEATURES_FROM_VERIFIED_SOURCE_"
        "AND_EXACT_ORDERED_COMPARE_TO_CACHE_BEFORE_CLAIM"
    ):
        raise RuntimeError("full feature binding contract changed")
    ambiguity = safety["ambiguous_raw_swap_probe"]
    if ambiguity != {
        "expected_ambiguous_groups": 299,
        "expected_ambiguous_rows": 603,
        "expected_groups_with_multiple_derived_variants": 299,
        "synthetic_swap_must_fail": True,
    }:
        raise RuntimeError("ambiguous same-raw swap probe contract changed")
    if safety["scoring_truth_policy"] != (
        "PASS_HELD_VERIFIED_IN_MEMORY_TRUTH_DIRECTLY_TO_SEALED_SCORE_"
        "LOGIC_WITH_ZERO_PATH_REOPEN"
    ):
        raise RuntimeError("held scoring truth contract changed")
    if safety["claimed_failure_policy"] != (
        "EXACTLY_ONE_CREATE_ONLY_HASH_CHAINED_FAILED_TERMINAL_WITH_"
        "PHASE_ERROR_FIT_STATE_AND_PREFIX_THEN_KEEP_LOCK"
    ):
        raise RuntimeError("claimed failure terminal contract changed")
    if safety["timeout_termination_policy"] != (
        "TASKKILL_RC_ZERO_AND_PID_PLUS_DESCENDANTS_CONFIRMED_GONE_"
        "OR_FAIL_CLOSED_WITH_TERMINAL_PROVENANCE"
    ):
        raise RuntimeError("timeout termination contract changed")
    if safety["atomic_publish_policy"] != (
        "SAME_DIRECTORY_TEMP_FSYNC_THEN_ATOMIC_HARDLINK_CREATE_ONLY"
    ):
        raise RuntimeError("create-only publish contract changed")
    if safety["directory_flush_policy"] != (
        "WINDOWS_CREATEFILE_GENERIC_WRITE_FLUSHFILEBUFFERS_OR_"
        "POSIX_DIRECTORY_FSYNC_FAIL_CLOSED"
    ):
        raise RuntimeError("directory durability contract changed")
    if int(safety["pre_fit_left_censor_gate"]["maximum_count_per_fold"]) != 0:
        raise RuntimeError("left-censor gate changed")
    if safety["terminal_order"] != [
        "result.json",
        "manifest.json",
        "998_worker_terminal.json",
        "parent independent verification",
        "999_completed.json",
        "release execution.lock",
    ]:
        raise RuntimeError("terminal ordering changed")

    if tuple(config["surface"]["fold_order"]) != FOLD_ORDER:
        raise RuntimeError("fold order changed")
    if list(config["surface"]["seeds"]) != [20260813, 20260829, 20260847]:
        raise RuntimeError("seed list changed")
    if float(config["rescue_decoder"]["probability_threshold"]) != 0.8:
        raise RuntimeError("cutoff changed")
    if int(config["rescue_decoder"]["maximum_anchor_distance_rows"]) != 18:
        raise RuntimeError("anchor distance changed")
    if int(config["resource_budget"]["residual_model_fits"]) != 9:
        raise RuntimeError("fit budget changed")
    if not all(int(value) == 0 for value in config["prohibitions"].values()):
        raise RuntimeError("a prohibition counter is nonzero")

    prior = config["supersedes"]
    for label, path_key, hash_key in (
        ("v1r3 config", "config_path", "config_sha256"),
        ("v1r3 runner", "runner_path", "runner_sha256"),
        ("v1r3 test", "test_path", "test_sha256"),
        ("v1r3 seal", "seal_path", "seal_sha256"),
    ):
        if _sha256(_resolve_repo_path(str(prior[path_key]))) != prior[hash_key]:
            raise RuntimeError(f"{label} hash mismatch")
    old, _old_read = _json_load_bound(
        _resolve_repo_path(str(prior["config_path"])),
        expected_sha256=str(prior["config_sha256"]),
    )
    changed = [name for name in NUMERICAL_SECTIONS if config[name] != old[name]]
    if changed:
        raise RuntimeError("numerical sections changed: " + ", ".join(changed))


def _external_authorization_value() -> str:
    value = os.environ.get(AUTHORIZATION_ENV_VAR)
    if not value:
        raise RuntimeError(f"{AUTHORIZATION_ENV_VAR} is required before any config read")
    if value != AUTHORIZATION_SHA256:
        raise RuntimeError("external authorization hash differs from sealed runner anchor")
    return value


def _load_authorized(
    config_path: Path,
    *,
    require_seal: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    _assert_preauth_clean()
    external = _external_authorization_value()
    if config_path.resolve(strict=True) != DEFAULT_CONFIG.resolve(strict=True):
        raise RuntimeError("only the fixed v1r4 config is authorized")
    authorization, authorization_read = _json_load_bound(
        AUTHORIZATION_PATH,
        expected_sha256=external,
    )
    expected_config_sha256 = str(authorization.get("config_sha256", ""))
    config, config_read = _json_load_bound(
        config_path,
        expected_sha256=expected_config_sha256,
    )
    _validate_contract(config)
    checks = {
        "status": "AUTHORIZED_FOR_ONE_LOCAL_OUTER_SCREEN",
        "experiment_id": config["experiment_id"],
        "config_sha256": config_read["actual_read_sha256"],
        "runner_normalized_sha256": _normalised_runner_sha256(),
    }
    for key, expected in checks.items():
        if authorization.get(key) != expected:
            raise RuntimeError(f"authorization binding mismatch: {key}")
    if authorization.get("runtime_versions") != EXPECTED_RUNTIME_VERSIONS:
        raise RuntimeError("authorization runtime closure changed")
    if authorization.get("lightgbm_native") != config["trust_contract"][
        "lightgbm_native"
    ]:
        raise RuntimeError("authorization native fingerprint changed")
    if authorization.get("supersedes") != config["supersedes"]:
        raise RuntimeError("authorization supersession evidence changed")
    expected_dependencies = {
        name: _sha256(_resolve_repo_path(name)) for name in REQUIRED_DEPENDENCY_PATHS
    }
    if authorization.get("dependency_sha256") != expected_dependencies:
        raise RuntimeError("authorization project dependency closure changed")
    expected_verification = {
        name: _sha256(_resolve_repo_path(name)) for name in REQUIRED_VERIFICATION_PATHS
    }
    if authorization.get("verification_sha256") != expected_verification:
        raise RuntimeError("authorization verification-file closure changed")
    expected_provenance = {
        name: _sha256(_resolve_repo_path(name)) for name in REQUIRED_PROVENANCE_PATHS
    }
    if authorization.get("provenance_sha256") != expected_provenance:
        raise RuntimeError("authorization provenance-file closure changed")
    expected_evidence = {
        name: str(specification["sha256"])
        for name, specification in config["preexecution_evidence"].items()
    }
    if authorization.get("preexecution_evidence_sha256") != expected_evidence:
        raise RuntimeError("authorization preexecution evidence closure changed")

    seal: dict[str, Any] | None = None
    if require_seal:
        seal_path = _artifact_dir(config) / "preexecution_seal.json"
        seal, seal_read = _json_load_bound(seal_path)
        if seal.get("status") != "SEALED_EXTERNAL_AUTH_PREIMPORT_V1R4":
            raise RuntimeError("invalid v1r4 seal status")
        seal_checks = {
            "config_sha256": config_read["actual_read_sha256"],
            "runner_normalized_sha256": _normalised_runner_sha256(),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "authorization_sha256": authorization_read["actual_read_sha256"],
        }
        for key, expected in seal_checks.items():
            if seal.get(key) != expected:
                raise RuntimeError(f"seal binding mismatch: {key}")
        for name, expected in seal["dependency_sha256"].items():
            if _sha256(_resolve_repo_path(name)) != expected:
                raise RuntimeError(f"dependency changed after seal: {name}")
        for name, expected in seal["verification_sha256"].items():
            if _sha256(_resolve_repo_path(name)) != expected:
                raise RuntimeError(f"verification file changed after seal: {name}")
        for name, expected in seal["provenance_sha256"].items():
            if _sha256(_resolve_repo_path(name)) != expected:
                raise RuntimeError(f"provenance file changed after seal: {name}")
        if seal_read["actual_read_sha256"] != _sha256_bytes(_json_bytes(seal)):
            raise RuntimeError("seal bytes are not canonical JSON")
    return config, authorization, seal


def _input_specs(config: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    immutable = config["immutable_inputs"]
    evidence = config["preexecution_evidence"]
    specs: list[tuple[str, Mapping[str, Any]]] = [
        ("base_config", config["base_config"]),
        ("feature_cache", immutable["feature_cache"]),
        ("feature_metadata", immutable["feature_metadata"]),
        ("feature_cache_key_binding", evidence["feature_cache_key_binding"]),
        (
            "feature_cache_key_binding_receipt",
            evidence["feature_cache_key_binding_receipt"],
        ),
        ("frozen_truth_oof", immutable["frozen_truth_oof"]),
        ("matched_budget_predictions", immutable["matched_budget_predictions"]),
    ]
    specs.extend(
        (f"round_b_full_prefix:{part['fold']}", part)
        for part in immutable["round_b_full_prefix_parts"]
    )
    return specs


def _training_source_path(config: Mapping[str, Any]) -> Path:
    raw = os.environ.get("P1_DATA_DIR")
    if not raw:
        raise RuntimeError("P1_DATA_DIR is missing; no execution claim was created")
    directory = Path(raw).expanduser().resolve(strict=True)
    path = (directory / "train.csv").resolve(strict=True)
    if path.parent != directory or path.name != "train.csv" or not path.is_file():
        raise RuntimeError("only P1 train.csv is an authorized external source")
    expected = str(config["immutable_inputs"]["feature_cache"]["source_sha256"])
    if _sha256(path) != expected:
        raise RuntimeError("P1 training source hash mismatch before claim")
    return path


def _stdlib_readiness(config: Mapping[str, Any]) -> dict[str, Any]:
    versions = _runtime_versions()
    if versions != config["trust_contract"]["runtime_versions"]:
        raise RuntimeError(f"runtime version mismatch: {versions}")
    native = _native_lightgbm_record(config)
    source = _training_source_path(config)
    files: dict[str, dict[str, Any]] = {
        "training_source": {
            "filename": "train.csv",
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
        }
    }
    for name, specification in _input_specs(config):
        path = _resolve_repo_path(str(specification["path"]))
        digest = _sha256(path)
        if digest != specification["sha256"]:
            raise RuntimeError(f"immutable input mismatch before claim: {name}")
        files[name] = {
            "project_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    return {
        "status": "PASS_STDLIB_READINESS_NO_CLAIM",
        "runtime_versions": versions,
        "lightgbm_native": native,
        "files": files,
        "source_path": source,
        "official_test_reads": 0,
        "sample_format_reads": 0,
        "submission_candidate_reads": 0,
    }


def seal(config_path: Path) -> Path:
    config, authorization, _ = _load_authorized(config_path, require_seal=False)
    readiness = _stdlib_readiness(config)
    dependencies = {
        name: _sha256(_resolve_repo_path(name)) for name in REQUIRED_DEPENDENCY_PATHS
    }
    verification = {
        name: _sha256(_resolve_repo_path(name)) for name in REQUIRED_VERIFICATION_PATHS
    }
    provenance = {
        name: _sha256(_resolve_repo_path(name)) for name in REQUIRED_PROVENANCE_PATHS
    }
    receipt = {
        "schema_version": "p1_round_b_nonspike_long_event_residual.seal.v1r4",
        "experiment_id": config["experiment_id"],
        "status": "SEALED_EXTERNAL_AUTH_PREIMPORT_V1R4",
        "sealed_at_kst": _now_kst(),
        "config_path": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "config_sha256": _sha256(config_path),
        "runner_path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        ),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "runner_normalized_sha256": _normalised_runner_sha256(),
        "authorization_path": str(AUTHORIZATION_PATH.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        ),
        "authorization_sha256": _sha256(AUTHORIZATION_PATH),
        "authorization_status": authorization["status"],
        "dependency_sha256": dependencies,
        "verification_sha256": verification,
        "provenance_sha256": provenance,
        "runtime_versions": readiness["runtime_versions"],
        "lightgbm_native": readiness["lightgbm_native"],
        "immutable_inputs": readiness["files"],
        "supersedes": config["supersedes"],
        "numerical_sections_equal_v1r3": list(NUMERICAL_SECTIONS),
        "registered_model_fits": 9,
        "operation_counters_at_seal": {
            "residual_model_fits": 0,
            "round_b_base_model_fits": 0,
            "outer_scores": 0,
            "candidate_files": 0,
            "uploads": 0,
            "official_test_reads": 0,
            "sample_format_reads": 0,
            "submission_candidate_reads": 0,
        },
    }
    path = _artifact_dir(config) / "preexecution_seal.json"
    _atomic_json_new(path, receipt)
    return path


def _copy_verified_file(source: Path, destination: Path, expected_sha256: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.snapshot"
    digest = hashlib.sha256()
    try:
        with source.open("rb") as source_handle:
            before = os.fstat(source_handle.fileno())
            with temporary.open("xb") as destination_handle:
                for block in iter(lambda: source_handle.read(1024 * 1024), b""):
                    digest.update(block)
                    destination_handle.write(block)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
            after = os.fstat(source_handle.fileno())
        signature_before = (before.st_size, before.st_mtime_ns, before.st_ino)
        signature_after = (after.st_size, after.st_mtime_ns, after.st_ino)
        if signature_before != signature_after:
            raise RuntimeError(f"held source changed during snapshot: {source}")
        observed = digest.hexdigest()
        if observed != expected_sha256:
            raise RuntimeError(f"held source digest mismatch: {source}")
        _publish_temp_create_only(temporary, destination)
        snapshot_digest = _sha256(destination)
        if snapshot_digest != expected_sha256:
            raise RuntimeError(f"snapshot digest mismatch after publication: {destination}")
        return {
            "bytes": destination.stat().st_size,
            "sha256": snapshot_digest,
            "actual_read_sha256": observed,
        }
    finally:
        if temporary.exists():
            temporary.unlink()


def _prepare_snapshot(
    config: Mapping[str, Any],
    seal_receipt: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> tuple[Path, dict[str, dict[str, Any]]]:
    root = Path(tempfile.mkdtemp(prefix="p1_v1r4_verified_snapshot_"))
    records: dict[str, dict[str, Any]] = {}
    try:
        source = Path(readiness["source_path"])
        destination = root / "inputs" / "train.csv"
        records["inputs/train.csv"] = _copy_verified_file(
            source,
            destination,
            str(config["immutable_inputs"]["feature_cache"]["source_sha256"]),
        )
        for _name, specification in _input_specs(config):
            relative = str(specification["path"])
            if relative in records:
                continue
            source_path = _resolve_repo_path(relative)
            records[relative] = _copy_verified_file(
                source_path,
                root / relative,
                str(specification["sha256"]),
            )
        for relative, expected in seal_receipt["dependency_sha256"].items():
            if relative in records:
                continue
            records[relative] = _copy_verified_file(
                _resolve_repo_path(relative),
                root / relative,
                str(expected),
            )
        return root, records
    except BaseException:
        _cleanup_snapshot(root)
        raise


def _cleanup_snapshot(root: Path) -> None:
    resolved = root.resolve()
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if resolved.parent != temporary_root or not resolved.name.startswith(
        "p1_v1r4_verified_snapshot_"
    ):
        raise RuntimeError(f"refusing unsafe snapshot cleanup: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _verify_snapshot_manifest(
    manifest_path: Path,
    expected_sha256: str,
    config: Mapping[str, Any],
    authorization: Mapping[str, Any],
    seal_receipt: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    manifest, _manifest_read = _json_load_bound(
        manifest_path,
        expected_sha256=expected_sha256,
    )
    snapshot_root = manifest_path.parent.resolve(strict=True)
    checks = {
        "config_sha256": _sha256(DEFAULT_CONFIG),
        "authorization_sha256": _sha256(AUTHORIZATION_PATH),
        "seal_sha256": _sha256(_artifact_dir(config) / "preexecution_seal.json"),
        "runner_normalized_sha256": _normalised_runner_sha256(),
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"snapshot manifest trust mismatch: {key}")
    if manifest.get("authorization_status") != authorization["status"]:
        raise RuntimeError("snapshot authorization status mismatch")
    if manifest.get("seal_status") != seal_receipt["status"]:
        raise RuntimeError("snapshot seal status mismatch")
    entries = manifest.get("files")
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError("snapshot manifest has no file inventory")
    for relative, record in entries.items():
        path = (snapshot_root / relative).resolve(strict=True)
        if not path.is_relative_to(snapshot_root):
            raise RuntimeError("snapshot entry escapes private root")
        if path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"snapshot size mismatch: {relative}")
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"snapshot digest mismatch: {relative}")
    return snapshot_root, manifest


def _loaded_runtime_file_receipt(
    added_modules: set[str],
    snapshot_root: Path,
) -> tuple[str, dict[str, dict[str, Any]], set[str]]:
    packages = importlib.metadata.packages_distributions()
    records: dict[str, dict[str, Any]] = {}
    distributions: set[str] = set()
    for module_name in sorted(added_modules):
        module = sys.modules.get(module_name)
        path_value = getattr(module, "__file__", None)
        if not path_value:
            continue
        path = Path(path_value).resolve()
        if path.is_relative_to(snapshot_root):
            continue
        root_name = module_name.split(".", maxsplit=1)[0]
        candidates = packages.get(root_name, [])
        candidates = [name for name in candidates if name in EXPECTED_RUNTIME_VERSIONS]
        if not candidates:
            continue
        digest = _sha256(path)
        records[module_name] = {
            "distributions": sorted(candidates),
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        distributions.update(candidates)
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(payload), records, distributions


def _load_snapshot_numerical(
    snapshot_root: Path,
    config: Mapping[str, Any],
) -> tuple[ModuleType, dict[str, Any]]:
    before = set(sys.modules)
    sys.dont_write_bytecode = True
    source_root = snapshot_root / "src"
    sys.path.insert(0, str(source_root))
    entrypoint = snapshot_root / str(
        config["trust_contract"]["numerical_entrypoint_path"]
    )
    specification = importlib.util.spec_from_file_location(
        f"_p1_v1r4_snapshot_{uuid.uuid4().hex}", entrypoint
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("unable to load verified numerical snapshot")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    added = set(sys.modules) - before
    aggregate, files, distributions = _loaded_runtime_file_receipt(
        added, snapshot_root
    )
    expected_aggregate = config["trust_contract"][
        "loaded_runtime_file_aggregate_sha256"
    ]
    if aggregate != expected_aggregate:
        raise RuntimeError(
            f"loaded runtime file aggregate mismatch: {aggregate} != {expected_aggregate}"
        )
    required_distributions = set(EXPECTED_RUNTIME_VERSIONS) - {"python", "p1-qc"}
    if distributions != required_distributions:
        raise RuntimeError(
            "actual imported distribution closure changed: "
            f"observed={sorted(distributions)}, expected={sorted(required_distributions)}"
        )
    native_path = Path(str(module.lgb.basic._LIB._name)).resolve(strict=True)
    native = {
        "distribution_relative_path": config["trust_contract"]["lightgbm_native"][
            "distribution_relative_path"
        ],
        "bytes": native_path.stat().st_size,
        "sha256": _sha256(native_path),
    }
    if native != config["trust_contract"]["lightgbm_native"]:
        raise RuntimeError("actually loaded LightGBM native binary differs from pin")
    return module, {
        "runtime_versions": _runtime_versions(),
        "loaded_distribution_names": sorted(distributions),
        "loaded_runtime_file_count": len(files),
        "loaded_runtime_file_aggregate_sha256": aggregate,
        "loaded_runtime_files": files,
        "lightgbm_native": native,
    }


class HeldSnapshotInputs:
    """Parse each private snapshot input from one held, pre/post-hashed handle."""

    def __init__(self, snapshot_root: Path) -> None:
        self.snapshot_root = snapshot_root.resolve(strict=True)
        self.receipts: dict[str, dict[str, Any]] = {}

    def _path(self, relative: str) -> Path:
        path = (self.snapshot_root / relative).resolve(strict=True)
        if not path.is_relative_to(self.snapshot_root) or not path.is_file():
            raise RuntimeError(f"snapshot input escapes private root: {relative}")
        return path

    @staticmethod
    def _hash_handle(handle: Any) -> str:
        handle.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        handle.seek(0)
        return digest.hexdigest()

    def _parse(
        self,
        relative: str,
        expected_sha256: str,
        parser: Any,
    ) -> Any:
        if relative in self.receipts:
            raise RuntimeError(f"snapshot input path would be reopened: {relative}")
        path = self._path(relative)
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            before_digest = self._hash_handle(handle)
            if before_digest != expected_sha256:
                raise RuntimeError(f"snapshot parse input digest mismatch: {relative}")
            value = parser(handle)
            if handle.closed:
                raise RuntimeError(f"parser unexpectedly closed held input: {relative}")
            after_digest = self._hash_handle(handle)
            after = os.fstat(handle.fileno())
        signature_before = (before.st_size, before.st_mtime_ns, before.st_ino)
        signature_after = (after.st_size, after.st_mtime_ns, after.st_ino)
        if signature_before != signature_after or after_digest != before_digest:
            raise RuntimeError(f"snapshot input changed during parse: {relative}")
        self.receipts[relative] = {
            "bytes": before.st_size,
            "expected_sha256": expected_sha256,
            "actual_read_sha256_before_parse": before_digest,
            "actual_read_sha256_after_parse": after_digest,
            "parsed_from_same_held_handle": True,
            "path_reopens": 0,
        }
        return value

    def read_csv(
        self,
        relative: str,
        expected_sha256: str,
        numerical: ModuleType,
    ) -> Any:
        return self._parse(
            relative,
            expected_sha256,
            lambda handle: numerical.pd.read_csv(handle, low_memory=False),
        )

    def read_parquet(
        self,
        relative: str,
        expected_sha256: str,
        numerical: ModuleType,
        *,
        columns: Sequence[str] | None = None,
    ) -> Any:
        return self._parse(
            relative,
            expected_sha256,
            lambda handle: numerical.pd.read_parquet(handle, columns=columns),
        )

    def read_json(self, relative: str, expected_sha256: str) -> dict[str, Any]:
        value = self._parse(
            relative,
            expected_sha256,
            lambda handle: json.loads(handle.read().decode("utf-8")),
        )
        if not isinstance(value, dict):
            raise TypeError(f"expected snapshot JSON object: {relative}")
        return value

    def read_bytes(self, relative: str, expected_sha256: str) -> bytes:
        return self._parse(relative, expected_sha256, lambda handle: handle.read())


def _load_base_config_from_held(
    snapshot_root: Path,
    config: Mapping[str, Any],
    numerical: ModuleType,
    reader: HeldSnapshotInputs,
) -> Any:
    relative = str(config["base_config"]["path"])
    raw = reader.read_bytes(relative, str(config["base_config"]["sha256"]))
    parsed = tomllib.loads(raw.decode("utf-8"))
    module = sys.modules.get(str(numerical.load_config.__module__))
    if module is None or not str(module.__file__).startswith(str(snapshot_root)):
        raise RuntimeError("snapshot config module provenance changed")
    effective = module._defaults_mapping()
    module._deep_merge(effective, module._normalise_legacy_sections(parsed))
    data_section = effective.get("data", {})
    paths_section = effective.setdefault("paths", {})
    config_path = snapshot_root / relative
    config_base = (
        config_path.parent.parent
        if config_path.parent.name == "configs"
        else config_path.parent
    )
    if isinstance(data_section, Mapping) and isinstance(paths_section, dict):
        if paths_section.get("data_dir") in {None, ""} and data_section.get(
            "relative_dir"
        ):
            paths_section["data_dir"] = str(
                (config_base / str(data_section["relative_dir"])).resolve()
            )
    return module._build_config(effective)


def _load_base_surface_from_held(
    config: Mapping[str, Any],
    numerical: ModuleType,
    reader: HeldSnapshotInputs,
) -> Any:
    """Exact v1 Round-B surface logic with held-handle input parsing only."""
    seeds = [int(value) for value in config["surface"]["seeds"]]
    key_columns = [str(value) for value in config["surface"]["key_columns"]]
    columns = [
        *key_columns,
        "row_position",
        "fold",
        "fraction",
        f"{BASE_PREFIX}__probability",
        f"{BASE_PREFIX}__prediction",
        "spike_candidate",
    ]
    for seed in seeds:
        columns.extend(
            [
                f"{BASE_PREFIX}__seed_{seed}__probability",
                f"{BASE_PREFIX}__seed_{seed}__prediction",
            ]
        )
    parts: list[Any] = []
    immutable = config["immutable_inputs"]
    for specification in immutable["round_b_full_prefix_parts"]:
        relative = str(specification["path"])
        part = reader.read_parquet(
            relative,
            str(specification["sha256"]),
            numerical,
            columns=columns,
        )
        if len(part) == 0 or not part["fold"].eq(str(specification["fold"])).all():
            raise RuntimeError(f"invalid Round-B part: {specification['fold']}")
        if not numerical.np.isclose(
            part["fraction"].to_numpy(dtype=float), 1.0
        ).all():
            raise RuntimeError("Round-B part is not full-prefix p100")
        parts.append(part)
    surface = numerical.pd.concat(parts, ignore_index=True)
    if len(surface) != int(config["surface"]["expected_rows"]):
        raise RuntimeError("Round-B surface row count changed")
    if surface.duplicated(key_columns).any() or not surface["row_position"].is_unique:
        raise RuntimeError("Round-B surface keys or row positions are duplicated")
    if tuple(surface["fold"].drop_duplicates().tolist()) != FOLD_ORDER:
        raise RuntimeError("Round-B surface fold order changed")

    matched_columns = [*key_columns, MATCHED_PREFIX]
    matched_columns.extend(f"{MATCHED_PREFIX}__seed_{seed}" for seed in seeds)
    matched_spec = immutable["matched_budget_predictions"]
    matched = reader.read_parquet(
        str(matched_spec["path"]),
        str(matched_spec["sha256"]),
        numerical,
        columns=matched_columns,
    )
    aligned = surface.loc[:, key_columns].merge(
        matched,
        on=key_columns,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(aligned) != len(surface) or aligned[MATCHED_PREFIX].isna().any():
        raise RuntimeError("matched-budget Round-B alignment failed")
    comparisons = [
        (
            surface[f"{BASE_PREFIX}__prediction"].to_numpy(dtype=numerical.np.int8),
            aligned[MATCHED_PREFIX].to_numpy(dtype=numerical.np.int8),
        )
    ]
    for seed in seeds:
        comparisons.append(
            (
                surface[f"{BASE_PREFIX}__seed_{seed}__prediction"].to_numpy(
                    dtype=numerical.np.int8
                ),
                aligned[f"{MATCHED_PREFIX}__seed_{seed}"].to_numpy(
                    dtype=numerical.np.int8
                ),
            )
        )
    if any(not numerical.np.array_equal(left, right) for left, right in comparisons):
        raise RuntimeError("p100 Round-B predictions differ from matched-budget default")
    return surface


def _canonical_key_digest(frame: Any) -> str:
    digest = hashlib.sha256()
    for ordinal, row in enumerate(frame.loc[:, list(KEY_COLUMNS)].itertuples(index=False, name=None)):
        values = (
            str(ordinal),
            str(row[0]),
            str(int(row[1])),
            str(int(row[2])),
            str(row[3]),
        )
        for value in values:
            encoded = value.encode()
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
    return digest.hexdigest()


def _validate_feature_row_binding(
    train: Any,
    features: Any,
    numerical: ModuleType,
) -> dict[str, Any]:
    expected_index = numerical.pd.RangeIndex(start=0, stop=len(train), step=1)
    if not train.index.equals(expected_index):
        raise RuntimeError("training frame is not exact positional RangeIndex")
    if not features.index.equals(expected_index):
        raise RuntimeError("feature cache is not exact positional RangeIndex")
    if len(features) != len(train):
        raise RuntimeError("feature cache row count differs from source")
    station_equal = numerical.np.array_equal(
        features["station"].astype(str).to_numpy(),
        train["station"].astype(str).to_numpy(),
    )
    layer_equal = numerical.np.array_equal(
        features["layer_category"].astype(str).to_numpy(),
        train["layer"].astype(str).to_numpy(),
    )
    if not station_equal or not layer_equal:
        raise RuntimeError("feature cache station/layer row binding mismatch")
    raw_columns = {
        "temp_raw": "temp",
        "psal_raw": "psal",
        "depth_raw": "depth",
    }
    raw_checks: dict[str, bool] = {}
    for feature_column, source_column in raw_columns.items():
        left = features[feature_column].to_numpy(dtype=numerical.np.float32)
        right = numerical.pd.to_numeric(
            train[source_column], errors="coerce"
        ).to_numpy(dtype=numerical.np.float32)
        equal = bool(numerical.np.array_equal(left, right, equal_nan=True))
        raw_checks[feature_column] = equal
        if not equal:
            raise RuntimeError(f"feature cache raw row binding mismatch: {feature_column}")
    if train.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError("training source contains duplicate exact row keys")
    return {
        "rows": len(train),
        "range_index_exact": True,
        "station_exact": station_equal,
        "layer_exact": layer_equal,
        "raw_float32_nan_aware_exact": raw_checks,
        "source_train_key_digest": _canonical_key_digest(train),
    }


def _left_censored_positive_event_counts(
    train: Any,
    folds: Sequence[Mapping[str, Any]],
    numerical: ModuleType,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fold in folds:
        name = str(fold["name"])
        indices = numerical.np.asarray(fold["train_idx"], dtype=numerical.np.int64)
        frame = train.iloc[indices][["station", "layer", "time", "label"]].copy()
        frame["_source_row"] = indices
        frame["_parsed"] = numerical.pd.to_datetime(
            frame["time"], errors="raise", utc=True, format="mixed"
        )
        frame.sort_values(
            ["station", "layer", "_parsed", "_source_row"],
            kind="stable",
            inplace=True,
        )
        first = frame.groupby(["station", "layer"], sort=False, observed=True).head(1)
        labels = numerical.pd.to_numeric(first["label"], errors="raise").to_numpy(
            dtype=numerical.np.int8
        )
        if not numerical.np.isin(labels, [0, 1]).all():
            raise RuntimeError(f"non-binary left-censor audit label: {name}")
        counts[name] = int((labels == 1).sum())
    if tuple(counts) != FOLD_ORDER or any(counts.values()):
        raise RuntimeError(f"NO_GO_LEFT_CENSORED_POSITIVE_EVENT_BEFORE_FIT: {counts}")
    return counts


def _baseline_truth_readiness(
    config: Mapping[str, Any],
    surface: Any,
    truth: Any,
    numerical: ModuleType,
) -> dict[str, Any]:
    if len(truth) != int(config["immutable_inputs"]["frozen_truth_oof"]["rows"]):
        raise RuntimeError("truth OOF row count changed")
    if truth.duplicated([*KEY_COLUMNS, "fold"]).any():
        raise RuntimeError("truth OOF keys are duplicated")
    baseline_column = f"{BASE_PREFIX}__prediction"
    aligned = surface.loc[:, [*KEY_COLUMNS, "fold", baseline_column]].merge(
        truth.loc[:, [*KEY_COLUMNS, "fold", "label"]],
        on=[*KEY_COLUMNS, "fold"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(aligned) != len(surface) or aligned["label"].isna().any():
        raise RuntimeError("truth OOF and exact Round-B surface do not align")
    values = numerical.binary_metrics(
        aligned["label"].to_numpy(dtype=numerical.np.int8),
        aligned[baseline_column].to_numpy(dtype=numerical.np.int8),
    )
    expected = config["surface"]["expected_base_metrics"]
    for name in ("tp", "fp", "fn"):
        if int(values[name]) != int(expected[name]):
            raise RuntimeError(f"exact Round-B baseline count changed: {name}")
    for name in ("f1", "precision", "recall"):
        if not numerical.np.isclose(
            float(values[name]), float(expected[name]), rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"exact Round-B baseline metric changed: {name}")
    return {
        "rows": len(aligned),
        "truth_key_alignment": "one_to_one",
        "expected_base_metrics_exact": True,
        "baseline_metrics": values,
    }


def _ordered_feature_frame_digest(frame: Any, numerical: ModuleType) -> str:
    schema = json.dumps(
        {
            "columns": [str(value) for value in frame.columns],
            "dtypes": [str(value) for value in frame.dtypes],
            "rows": len(frame),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    row_hashes = numerical.pd.util.hash_pandas_object(
        frame,
        index=True,
        categorize=False,
    ).to_numpy(dtype=numerical.np.uint64)
    return _sha256_bytes(schema + row_hashes.tobytes(order="C"))


def _feature_frames_exact(
    expected: Any,
    observed: Any,
    numerical: ModuleType,
) -> tuple[bool, list[str]]:
    if not expected.index.equals(observed.index):
        return False, ["<index>"]
    if list(expected.columns) != list(observed.columns):
        return False, ["<columns>"]
    if [str(value) for value in expected.dtypes] != [
        str(value) for value in observed.dtypes
    ]:
        return False, ["<dtypes>"]
    unequal: list[str] = []
    for column in expected.columns:
        left = expected[column]
        right = observed[column]
        if numerical.pd.api.types.is_numeric_dtype(left.dtype):
            left_values = left.to_numpy()
            right_values = right.to_numpy()
            if numerical.pd.api.types.is_float_dtype(left.dtype):
                equal = numerical.np.array_equal(
                    left_values,
                    right_values,
                    equal_nan=True,
                )
            else:
                equal = numerical.np.array_equal(left_values, right_values)
        else:
            equal = left.equals(right)
        if not bool(equal):
            unequal.append(str(column))
    return not unequal, unequal


def _full_feature_regeneration_readiness(
    train: Any,
    cached: Any,
    p1_config: Any,
    numerical: ModuleType,
    config: Mapping[str, Any],
    snapshot_root: Path,
) -> dict[str, Any]:
    feature_module = sys.modules.get("p1_qc.features")
    if feature_module is None:
        raise RuntimeError("feature regeneration module is not loaded")
    module_path = Path(str(getattr(feature_module, "__file__", ""))).resolve()
    if not module_path.is_relative_to(snapshot_root):
        raise RuntimeError("feature regeneration module is not from verified snapshot")
    if str(p1_config.features.mode) != "offline":
        raise RuntimeError("registered feature regeneration mode changed")
    rebuilt_bundle = feature_module.build_features(
        train,
        config=p1_config,
        mode="offline",
        cadence_minutes=int(p1_config.data.cadence_minutes),
    )
    rebuilt = rebuilt_bundle.frame
    actual_equal, unequal_columns = _feature_frames_exact(
        rebuilt,
        cached,
        numerical,
    )
    if not actual_equal:
        raise RuntimeError(
            "NO_GO_REGENERATED_FEATURE_CACHE_ORDER_OR_VALUE_MISMATCH: "
            + ", ".join(unequal_columns[:10])
        )
    rebuilt_digest = _ordered_feature_frame_digest(rebuilt, numerical)
    cached_digest = _ordered_feature_frame_digest(cached, numerical)
    if rebuilt_digest != cached_digest:
        raise RuntimeError("regenerated/cache ordered full-row digest mismatch")

    raw = numerical.pd.DataFrame(
        {
            "station": train["station"].astype(str),
            "layer": train["layer"].astype(str),
            "temp": numerical.pd.to_numeric(train["temp"], errors="coerce").astype(
                numerical.np.float32
            ),
            "psal": numerical.pd.to_numeric(train["psal"], errors="coerce").astype(
                numerical.np.float32
            ),
            "depth": numerical.pd.to_numeric(train["depth"], errors="coerce").astype(
                numerical.np.float32
            ),
            "feature_hash": numerical.pd.util.hash_pandas_object(
                cached,
                index=False,
                categorize=False,
            ).to_numpy(dtype=numerical.np.uint64),
        }
    )
    raw_keys = ["station", "layer", "temp", "psal", "depth"]
    raw["group_code"] = raw.groupby(
        raw_keys,
        dropna=False,
        sort=False,
    ).ngroup()
    group_stats = raw.groupby("group_code", sort=False)["feature_hash"].agg(
        ["size", "nunique"]
    )
    ambiguous = group_stats.loc[group_stats["size"].gt(1)]
    variants = ambiguous.loc[ambiguous["nunique"].gt(1)]
    expected_probe = config["execution_safety"]["ambiguous_raw_swap_probe"]
    observed_counts = {
        "ambiguous_groups": len(ambiguous),
        "ambiguous_rows": int(ambiguous["size"].sum()),
        "groups_with_multiple_derived_variants": len(variants),
    }
    registered_counts = {
        "ambiguous_groups": int(expected_probe["expected_ambiguous_groups"]),
        "ambiguous_rows": int(expected_probe["expected_ambiguous_rows"]),
        "groups_with_multiple_derived_variants": int(
            expected_probe["expected_groups_with_multiple_derived_variants"]
        ),
    }
    if observed_counts != registered_counts or not len(variants):
        raise RuntimeError(
            f"ambiguous raw-fingerprint audit changed: {observed_counts}"
        )
    group_code = int(variants.index[0])
    group_rows = raw.loc[raw["group_code"].eq(group_code)]
    first_index = int(group_rows.index[0])
    different = group_rows.loc[
        group_rows["feature_hash"].ne(group_rows.iloc[0]["feature_hash"])
    ]
    if different.empty:
        raise RuntimeError("unable to construct registered ambiguous swap probe")
    second_index = int(different.index[0])
    expected_pair = rebuilt.iloc[[first_index, second_index]].reset_index(drop=True)
    swapped_pair = cached.iloc[[second_index, first_index]].reset_index(drop=True)
    raw_columns = ["station", "layer_category", "temp_raw", "psal_raw", "depth_raw"]
    raw_equal, _raw_unequal = _feature_frames_exact(
        expected_pair.loc[:, raw_columns],
        swapped_pair.loc[:, raw_columns],
        numerical,
    )
    swap_equal, swap_unequal = _feature_frames_exact(
        expected_pair,
        swapped_pair,
        numerical,
    )
    if not raw_equal or swap_equal or not swap_unequal:
        raise RuntimeError("synthetic ambiguous same-raw swap was not detected")
    return {
        "status": "PASS_REGENERATED_ALL_MODEL_INPUT_FEATURES_EXACT_ORDERED",
        "rows": len(cached),
        "columns": len(cached.columns),
        "actual_cache_order_and_values_exact": True,
        "dtypes_exact": True,
        "regenerated_ordered_full_row_digest": rebuilt_digest,
        "cached_ordered_full_row_digest": cached_digest,
        "ambiguous_raw_fingerprint": observed_counts,
        "synthetic_swap_probe": {
            "raw_fingerprint_unchanged": True,
            "full_feature_mismatch_detected": True,
            "unequal_derived_feature_count": len(swap_unequal),
        },
        "source_regenerations": 1,
        "model_fits": 0,
        "outer_scores": 0,
    }


def _validate_pinned_feature_key_binding(
    train: Any,
    sidecar: Any,
    feature_binding: dict[str, Any],
    cache_spec: Mapping[str, Any],
    sidecar_spec: Mapping[str, Any],
    binding_receipt_spec: Mapping[str, Any],
    binding_receipt: Mapping[str, Any],
    numerical: ModuleType,
) -> dict[str, Any]:
    expected_sidecar_columns = ["ordinal", *KEY_COLUMNS]
    if list(sidecar.columns) != expected_sidecar_columns:
        raise RuntimeError("feature key sidecar schema changed")
    if len(sidecar) != int(sidecar_spec["rows"]) or len(sidecar) != len(train):
        raise RuntimeError("feature key sidecar row count changed")
    ordinal = sidecar["ordinal"].to_numpy(dtype=numerical.np.int64)
    if not numerical.np.array_equal(
        ordinal, numerical.np.arange(len(train), dtype=numerical.np.int64)
    ):
        raise RuntimeError("feature key sidecar ordinal order changed")
    sidecar_digest = _canonical_key_digest(sidecar)
    expected_key_digest = str(sidecar_spec["source_train_key_digest"])
    if (
        sidecar_digest != expected_key_digest
        or feature_binding["source_train_key_digest"] != expected_key_digest
    ):
        raise RuntimeError("feature cache row-key digest differs from source train")
    exact_keys = sidecar.loc[:, list(KEY_COLUMNS)].equals(
        train.loc[:, list(KEY_COLUMNS)]
    )
    if not exact_keys:
        raise RuntimeError("feature key sidecar values differ from source train")
    binding_checks = {
        "schema_version": "p1_round_b_residual.feature_cache_key_binding.v1r3",
        "status": "SEALED_ZERO_FIT_SOURCE_CACHE_POSITIONAL_BINDING",
        "rows": len(train),
        "source_sha256": cache_spec["source_sha256"],
        "feature_cache_sha256": cache_spec["sha256"],
        "source_train_key_digest": expected_key_digest,
        "feature_cache_key_digest": expected_key_digest,
        "key_digests_equal": True,
        "sidecar_path": sidecar_spec["path"],
        "sidecar_bytes": int(sidecar_spec["bytes"]),
        "sidecar_sha256": sidecar_spec["sha256"],
    }
    for name, expected in binding_checks.items():
        if binding_receipt.get(name) != expected:
            raise RuntimeError(f"feature-cache key binding receipt changed: {name}")
    if binding_receipt.get("operation_counters") != {
        "candidate_files": 0,
        "model_fits": 0,
        "official_test_reads": 0,
        "predictions": 0,
        "sample_format_reads": 0,
        "scores": 0,
        "submission_candidate_reads": 0,
        "uploads": 0,
    }:
        raise RuntimeError("feature-cache key binding operation counters changed")
    feature_binding["feature_cache_key_sidecar_digest"] = sidecar_digest
    feature_binding["feature_cache_key_digest_equals_source"] = True
    feature_binding["feature_cache_exact_key_values_equal_source"] = exact_keys
    feature_binding["key_binding_receipt_sha256"] = binding_receipt_spec["sha256"]
    return feature_binding


def _strict_snapshot_readiness(
    snapshot_root: Path,
    config: Mapping[str, Any],
    numerical: ModuleType,
    runtime_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reader = HeldSnapshotInputs(snapshot_root)
    immutable = config["immutable_inputs"]
    cache_spec = immutable["feature_cache"]
    train = reader.read_csv(
        "inputs/train.csv",
        str(cache_spec["source_sha256"]),
        numerical,
    )
    train.attrs.update(
        {
            "source_path": "PRIVATE_VERIFIED_SNAPSHOT/inputs/train.csv",
            "source_size": reader.receipts["inputs/train.csv"]["bytes"],
            "source_sha256": str(cache_spec["source_sha256"]),
            "dataset_kind": "train",
        }
    )
    feature_relative = str(cache_spec["path"])
    features = reader.read_parquet(
        feature_relative,
        str(cache_spec["sha256"]),
        numerical,
    )
    metadata_spec = immutable["feature_metadata"]
    metadata = reader.read_json(
        str(metadata_spec["path"]),
        str(metadata_spec["sha256"]),
    )
    if int(metadata["rows"]) != len(train) or len(features) != int(cache_spec["rows"]):
        raise RuntimeError("feature cache row contract changed")
    if metadata.get("source_sha256") != cache_spec["source_sha256"]:
        raise RuntimeError("feature metadata source binding changed")
    if metadata.get("parquet_sha256") != cache_spec["sha256"]:
        raise RuntimeError("feature metadata cache binding changed")
    if tuple(features.columns) != tuple(metadata["feature_columns"]):
        raise RuntimeError("feature cache schema differs from metadata")
    if len(features.columns) != int(cache_spec["columns"]):
        raise RuntimeError("feature cache column count changed")
    if {"label", "anomaly_type"}.intersection(features.columns):
        raise RuntimeError("feature cache contains protected target columns")
    feature_binding = _validate_feature_row_binding(train, features, numerical)

    evidence = config["preexecution_evidence"]
    sidecar_spec = evidence["feature_cache_key_binding"]
    sidecar_relative = str(sidecar_spec["path"])
    sidecar = reader.read_parquet(
        sidecar_relative,
        str(sidecar_spec["sha256"]),
        numerical,
    )
    binding_receipt_spec = evidence["feature_cache_key_binding_receipt"]
    binding_receipt = reader.read_json(
        str(binding_receipt_spec["path"]),
        str(binding_receipt_spec["sha256"]),
    )
    feature_binding = _validate_pinned_feature_key_binding(
        train,
        sidecar,
        feature_binding,
        cache_spec,
        sidecar_spec,
        binding_receipt_spec,
        binding_receipt,
        numerical,
    )
    p1_config = _load_base_config_from_held(snapshot_root, config, numerical, reader)
    full_feature_binding = _full_feature_regeneration_readiness(
        train,
        features,
        p1_config,
        numerical,
        config,
        snapshot_root,
    )

    # No positional index overwrite: the full 80-column source regeneration,
    # ordered value comparison, and synthetic same-raw swap probe were proven.
    bundle = numerical.FeatureBundle(
        features,
        tuple(str(value) for value in metadata["feature_columns"]),
        tuple(str(value) for value in metadata["categorical_columns"]),
    )
    surface = _load_base_surface_from_held(config, numerical, reader)
    truth_spec = immutable["frozen_truth_oof"]
    truth = reader.read_parquet(
        str(truth_spec["path"]),
        str(truth_spec["sha256"]),
        numerical,
        columns=[*KEY_COLUMNS, "label", "anomaly_type", "fold"],
    )
    folds = numerical._fold_runtime(train, p1_config, surface)
    left_counts = _left_censored_positive_event_counts(train, folds, numerical)
    baseline = _baseline_truth_readiness(config, surface, truth, numerical)
    receipt = {
        "status": "PASS_COMPLETE_READINESS_BEFORE_CLAIM",
        "rows": len(train),
        "oof_rows": len(surface),
        "fold_rows": {fold: int(surface["fold"].eq(fold).sum()) for fold in FOLD_ORDER},
        "feature_row_binding": feature_binding,
        "full_feature_cache_binding": full_feature_binding,
        "left_censored_positive_connected_event_count_by_fold": left_counts,
        "exact_round_b_equivalence": baseline,
        "held_snapshot_input_reads": reader.receipts,
        "runtime": runtime_receipt,
        "residual_model_fits": 0,
        "outer_scores": 0,
        "official_test_reads": 0,
        "sample_format_reads": 0,
        "submission_candidate_reads": 0,
    }
    state = {
        "train": train,
        "bundle": bundle,
        "feature_metadata": metadata,
        "surface": surface,
        "truth": truth,
        "truth_read_receipt": dict(reader.receipts[str(truth_spec["path"])]),
        "folds": folds,
        "left_counts": left_counts,
    }
    return receipt, state


def _finalize_snapshot_manifest(
    snapshot_root: Path,
    records: dict[str, dict[str, Any]],
    readiness_receipt: Mapping[str, Any],
    config: Mapping[str, Any],
    authorization: Mapping[str, Any],
    seal_receipt: Mapping[str, Any],
) -> Path:
    manifest = {
        "schema_version": "p1_round_b_residual.verified_snapshot.v1r4",
        "experiment_id": config["experiment_id"],
        "status": "COMPLETE_PRIVATE_SNAPSHOT_READY_FOR_WORKER_REVERIFY",
        "created_at_kst": _now_kst(),
        "config_sha256": _sha256(DEFAULT_CONFIG),
        "authorization_sha256": _sha256(AUTHORIZATION_PATH),
        "authorization_status": authorization["status"],
        "seal_sha256": _sha256(_artifact_dir(config) / "preexecution_seal.json"),
        "seal_status": seal_receipt["status"],
        "runner_normalized_sha256": _normalised_runner_sha256(),
        "immutability_policy": (
            "UNIQUE_PARENT_OWNED_TEMP_ROOT_COMPLETE_DIGEST_INVENTORY_"
            "NO_MUTATION_AFTER_MANIFEST_WORKER_FULL_REHASH"
        ),
        "files": records,
        "strict_readiness": readiness_receipt,
    }
    path = snapshot_root / "snapshot_manifest.json"
    _atomic_json_new(path, manifest)
    return path


class AttemptJournal:
    """Create-only lifetime fit ledger owned by the worker until parent finalization."""

    def __init__(
        self,
        artifact: Path,
        lock_path: Path,
        lock_descriptor: int,
        journal_dir: Path,
        attempt_id: str,
        deadline_epoch: float,
    ) -> None:
        self.artifact = artifact
        self.lock_path = lock_path
        self.lock_descriptor = lock_descriptor
        self.journal_dir = journal_dir
        self.attempt_id = attempt_id
        self.deadline_epoch = deadline_epoch
        self.reserved_fits = 0
        self.completed_fits = 0
        self._entry_hashes: dict[str, str] = {}
        self._last_sha256: str | None = None
        self._fold_reservations: dict[str, int] = {fold: 0 for fold in FOLD_ORDER}

    @classmethod
    def begin(
        cls,
        artifact: Path,
        *,
        deadline_epoch: float,
        snapshot_manifest_sha256: str,
    ) -> AttemptJournal:
        if time.time() >= deadline_epoch:
            raise TimeoutError("deadline expired before execution claim")
        artifact.mkdir(parents=True, exist_ok=True)
        lock_path = artifact / "execution.lock"
        journal_dir = artifact / "attempt_journal"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        attempt_id = uuid.uuid4().hex
        created_journal = False
        try:
            payload = _json_bytes(
                {
                    "attempt_id": attempt_id,
                    "pid": os.getpid(),
                    "deadline_epoch": deadline_epoch,
                    "created_at_kst": _now_kst(),
                }
            )
            os.write(descriptor, payload)
            os.fsync(descriptor)
            _fsync_directory(artifact)
            if journal_dir.exists():
                raise FileExistsError("lifetime attempt journal already exists")
            os.mkdir(journal_dir)
            created_journal = True
            _fsync_directory(artifact)
            journal = cls(
                artifact,
                lock_path,
                descriptor,
                journal_dir,
                attempt_id,
                deadline_epoch,
            )
            journal._entry(
                "000_started.json",
                {
                    "schema_version": "p1_round_b_residual.attempt.started.v1r4",
                    "attempt_id": attempt_id,
                    "deadline_epoch": deadline_epoch,
                    "snapshot_manifest_sha256": snapshot_manifest_sha256,
                    "maximum_lifetime_physical_model_fits": 9,
                    "created_at_kst": _now_kst(),
                },
            )
            return journal
        except BaseException:
            os.close(descriptor)
            if not created_journal and lock_path.exists():
                lock_path.unlink()
                _fsync_directory(artifact)
            raise

    def _verify_entries(self) -> None:
        names = {path.name for path in self.journal_dir.glob("*.json")}
        if names != set(self._entry_hashes):
            raise RuntimeError("attempt journal membership changed")
        for name, expected in self._entry_hashes.items():
            if _sha256(self.journal_dir / name) != expected:
                raise RuntimeError(f"attempt journal entry changed: {name}")

    def _entry(self, name: str, value: Mapping[str, Any]) -> Path:
        self._verify_entries()
        record = dict(value)
        record["previous_entry_sha256"] = self._last_sha256
        path = self.journal_dir / name
        _atomic_json_new(path, record)
        digest = _sha256(path)
        self._entry_hashes[name] = digest
        self._last_sha256 = digest
        return path

    def record_readiness(self, receipt: Mapping[str, Any]) -> None:
        if self.reserved_fits or self.completed_fits:
            raise RuntimeError("readiness must precede every fit reservation")
        self._entry(
            "005_preclaim_readiness_reconfirmed.json",
            {
                "schema_version": "p1_round_b_residual.readiness.v1r4",
                "attempt_id": self.attempt_id,
                "status": receipt["status"],
                "feature_row_binding": receipt["feature_row_binding"],
                "full_feature_cache_binding": receipt[
                    "full_feature_cache_binding"
                ],
                "left_censor_counts": receipt[
                    "left_censored_positive_connected_event_count_by_fold"
                ],
                "exact_round_b_equivalence": receipt["exact_round_b_equivalence"],
                "physical_model_fits_before_receipt": 0,
                "created_at_kst": _now_kst(),
            },
        )

    def begin_fold(self, ordinal: int, fold: str) -> None:
        self._entry(
            f"{10 + ordinal * 30:03d}_{fold}_intent.json",
            {
                "schema_version": "p1_round_b_residual.fold_intent.v1r4",
                "attempt_id": self.attempt_id,
                "fold": fold,
                "ordinal": ordinal,
                "registered_fit_count": 3,
                "cumulative_reserved_before_fold": self.reserved_fits,
                "created_at_kst": _now_kst(),
            },
        )

    def reserve_fit(self, fold: str, fold_ordinal: int, fit_seed: int) -> int:
        if time.time() >= self.deadline_epoch:
            raise TimeoutError("absolute deadline expired before physical fit reservation")
        if self.reserved_fits >= 9:
            raise RuntimeError("lifetime physical fit ceiling would be exceeded")
        within_fold = self._fold_reservations[fold]
        seeds = [20260813, 20260829, 20260847]
        if within_fold >= 3 or fit_seed != seeds[within_fold] + fold_ordinal:
            raise RuntimeError("physical fit seed/order differs from preregistration")
        reservation = self.reserved_fits + 1
        self._entry(
            f"{11 + fold_ordinal * 30 + within_fold * 2:03d}_{fold}_fit_{within_fold + 1}_reserved.json",
            {
                "schema_version": "p1_round_b_residual.fit_reserved.v1r4",
                "attempt_id": self.attempt_id,
                "reservation": reservation,
                "fold": fold,
                "fold_ordinal": fold_ordinal,
                "within_fold_ordinal": within_fold,
                "fit_seed": fit_seed,
                "deadline_epoch": self.deadline_epoch,
                "reserved_at_epoch": time.time(),
                "created_at_kst": _now_kst(),
            },
        )
        self.reserved_fits += 1
        self._fold_reservations[fold] += 1
        return reservation

    def complete_fit(self, reservation: int, fold: str, fit_seed: int) -> None:
        if reservation != self.completed_fits + 1:
            raise RuntimeError("physical fit completion order changed")
        self._entry(
            f"{12 + (reservation - 1) // 3 * 30 + ((reservation - 1) % 3) * 2:03d}_{fold}_fit_{(reservation - 1) % 3 + 1}_completed.json",
            {
                "schema_version": "p1_round_b_residual.fit_completed.v1r4",
                "attempt_id": self.attempt_id,
                "reservation": reservation,
                "fold": fold,
                "fit_seed": fit_seed,
                "completed_at_kst": _now_kst(),
            },
        )
        self.completed_fits += 1

    def complete_fold(
        self,
        ordinal: int,
        fold: str,
        part_path: Path,
        audit_path: Path,
    ) -> None:
        if self._fold_reservations[fold] != 3:
            raise RuntimeError("fold did not reserve exactly three physical fits")
        self._entry(
            f"{29 + ordinal * 30:03d}_{fold}_completed.json",
            {
                "schema_version": "p1_round_b_residual.fold_completed.v1r4",
                "attempt_id": self.attempt_id,
                "fold": fold,
                "cumulative_reserved_fits": self.reserved_fits,
                "cumulative_completed_fits": self.completed_fits,
                "part_sha256": _sha256(part_path),
                "audit_sha256": _sha256(audit_path),
                "completed_at_kst": _now_kst(),
            },
        )

    def manifest_records(self) -> dict[str, dict[str, Any]]:
        self._verify_entries()
        return {
            str((self.journal_dir / name).relative_to(PROJECT_ROOT)).replace("\\", "/"): {
                "bytes": (self.journal_dir / name).stat().st_size,
                "sha256": digest,
            }
            for name, digest in self._entry_hashes.items()
        }

    def worker_terminal(self, result_path: Path, manifest_path: Path) -> Path:
        if self.reserved_fits != 9 or self.completed_fits != 9:
            raise RuntimeError("worker terminal requires exactly nine completed fits")
        path = self._entry(
            "998_worker_terminal.json",
            {
                "schema_version": "p1_round_b_residual.worker_terminal.v1r4",
                "attempt_id": self.attempt_id,
                "physical_fit_reservations": self.reserved_fits,
                "physical_fits_completed": self.completed_fits,
                "result_sha256": _sha256(result_path),
                "manifest_sha256": _sha256(manifest_path),
                "status": "WORKER_SUCCESS_PARENT_QA_REQUIRED",
                "completed_at_kst": _now_kst(),
            },
        )
        self.close_handle_keep_lock()
        return path

    def fail_terminal(
        self,
        phase: str,
        error: BaseException,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> Path:
        """Publish exactly one durable FAILED terminal and retain the lock."""
        terminals = [
            path
            for path in self.journal_dir.glob("*.json")
            if path.name in {
                "997_failed.json",
                "998_worker_terminal.json",
                "999_completed.json",
                "999_failed.json",
                "999_postcompletion_failed.json",
            }
        ]
        failed = [
            path
            for path in terminals
            if path.name
            in {
                "997_failed.json",
                "999_failed.json",
                "999_postcompletion_failed.json",
            }
        ]
        if len(failed) == 1:
            self.close_handle_keep_lock()
            return failed[0]
        if len(failed) > 1 or any(path.name == "999_completed.json" for path in terminals):
            raise RuntimeError("attempt already has an incompatible terminal")
        terminal_name = (
            "999_failed.json"
            if any(path.name == "998_worker_terminal.json" for path in terminals)
            else "997_failed.json"
        )
        prefix_names = sorted(self._entry_hashes)
        path = self._entry(
            terminal_name,
            {
                "schema_version": "p1_round_b_residual.failed_terminal.v1r4",
                "attempt_id": self.attempt_id,
                "status": "FAILED_FAIL_CLOSED_LOCK_RETAINED",
                "failure_actor": "worker",
                "phase": str(phase),
                "error_type": type(error).__name__,
                "error_message": str(error)[-2000:],
                "physical_fit_reservations": self.reserved_fits,
                "physical_fits_completed": self.completed_fits,
                "fit_slot_state": {
                    "maximum_lifetime_physical_model_fits": 9,
                    "next_slot": self.reserved_fits + 1
                    if self.reserved_fits < 9
                    else None,
                    "fold_reservations": dict(self._fold_reservations),
                },
                "journal_prefix": {
                    "entry_count": len(prefix_names),
                    "entry_names": prefix_names,
                    "last_entry_sha256": self._last_sha256,
                },
                "failure_provenance": dict(provenance or {}),
                "failed_at_kst": _now_kst(),
            },
        )
        self.close_handle_keep_lock()
        return path

    def close_handle_keep_lock(self) -> None:
        if self.lock_descriptor >= 0:
            os.close(self.lock_descriptor)
            self.lock_descriptor = -1


def _install_fit_guard(
    numerical: ModuleType,
    attempt: AttemptJournal,
    fold_context: dict[str, Any],
) -> type:
    original = numerical.lgb.LGBMClassifier

    class DeadlineCheckedLGBMClassifier:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._fit_seed = int(kwargs["random_state"])
            self._delegate = original(*args, **kwargs)

        def fit(self, *args: Any, **kwargs: Any) -> DeadlineCheckedLGBMClassifier:
            fold = str(fold_context["name"])
            ordinal = int(fold_context["ordinal"])
            reservation = attempt.reserve_fit(fold, ordinal, self._fit_seed)
            self._delegate.fit(*args, **kwargs)
            attempt.complete_fit(reservation, fold, self._fit_seed)
            return self

        def predict_proba(self, *args: Any, **kwargs: Any) -> Any:
            return self._delegate.predict_proba(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._delegate, name)

    numerical.lgb.LGBMClassifier = DeadlineCheckedLGBMClassifier
    return original


def _file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _assert_output_namespace_ready(artifact: Path) -> None:
    existing = [
        artifact / "predictions.parquet",
        artifact / "predictions_complete.json",
        artifact / "metrics.json",
        artifact / "result.json",
        artifact / "manifest.json",
        artifact / "attempt_journal" / "998_worker_terminal.json",
        artifact / "attempt_journal" / "997_failed.json",
        artifact / "attempt_journal" / "999_failed.json",
        artifact / "attempt_journal" / "999_postcompletion_failed.json",
        artifact / "attempt_journal" / "999_completed.json",
    ]
    existing.extend(
        artifact / "prediction_parts" / f"{fold}.parquet" for fold in FOLD_ORDER
    )
    existing.extend(
        artifact / "prediction_parts" / f"{fold}.json" for fold in FOLD_ORDER
    )
    present = [path for path in existing if path.exists()]
    if present:
        names = ", ".join(path.name for path in present[:5])
        raise FileExistsError(f"one-shot output already exists before claim: {names}")


def _commit_worker_terminal_artifacts(
    artifact: Path,
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    attempt: AttemptJournal,
    *,
    crash_after: str | None = None,
) -> tuple[Path, Path]:
    """Durably publish result, manifest, then worker terminal in that order."""
    result_path = artifact / "result.json"
    manifest_path = artifact / "manifest.json"
    _atomic_json_new(result_path, result)
    if crash_after == "result":
        raise RuntimeError("injected crash after result")
    _atomic_json_new(manifest_path, manifest)
    if crash_after == "manifest":
        raise RuntimeError("injected crash after manifest")
    attempt.worker_terminal(result_path, manifest_path)
    if crash_after == "worker_terminal":
        raise RuntimeError("injected crash after worker terminal")
    return result_path, manifest_path


def _score_with_verified_truth(
    numerical: ModuleType,
    config: Mapping[str, Any],
    predictions: Any,
    truth: Any,
    truth_read_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool], dict[str, Any]]:
    """Run the sealed v1 score while serving truth from verified held memory.

    The sealed v1 scorer asks its module globals to resolve and read the frozen
    truth parquet.  This narrow adapter preserves every downstream scoring
    operation while replacing only that I/O boundary with the exact frame
    already parsed from verified snapshot bytes during preclaim readiness.
    """
    truth_spec = config["immutable_inputs"]["frozen_truth_oof"]
    expected_relative = str(truth_spec["path"])
    expected_columns = [*KEY_COLUMNS, "label", "anomaly_type", "fold"]
    if list(truth.columns) != expected_columns:
        raise RuntimeError("held scoring truth schema changed")
    if len(truth) != int(truth_spec["rows"]):
        raise RuntimeError("held scoring truth row count changed")
    if (
        truth_read_receipt.get("actual_read_sha256_before_parse")
        != truth_spec["sha256"]
        or truth_read_receipt.get("actual_read_sha256_after_parse")
        != truth_spec["sha256"]
    ):
        raise RuntimeError("held scoring truth receipt digest changed")
    if truth_read_receipt.get("path_reopens") != 0:
        raise RuntimeError("held scoring truth was reopened before scoring")

    sentinel = object()
    resolver_calls = 0
    in_memory_reads = 0
    original_resolver = numerical._resolve_repo_path
    original_read_parquet = numerical.pd.read_parquet

    def resolve_truth_only(value: str, *_args: Any, **_kwargs: Any) -> object:
        nonlocal resolver_calls
        if str(value) != expected_relative:
            raise RuntimeError("sealed scorer requested an unexpected path")
        resolver_calls += 1
        return sentinel

    def read_truth_only(
        source: object,
        *,
        columns: Sequence[str] | None = None,
        **_kwargs: Any,
    ) -> Any:
        nonlocal in_memory_reads
        if source is not sentinel:
            raise RuntimeError("sealed scorer attempted an unverified parquet read")
        requested = expected_columns if columns is None else list(columns)
        if requested != expected_columns:
            raise RuntimeError("sealed scorer requested an unexpected truth schema")
        in_memory_reads += 1
        return truth.loc[:, requested].copy(deep=True)

    numerical._resolve_repo_path = resolve_truth_only
    numerical.pd.read_parquet = read_truth_only
    try:
        metrics, checks = numerical._score(config, predictions)
    finally:
        numerical.pd.read_parquet = original_read_parquet
        numerical._resolve_repo_path = original_resolver
    if resolver_calls != 1 or in_memory_reads != 1:
        raise RuntimeError("sealed scorer truth injection count changed")
    receipt = {
        "status": "PASS_HELD_VERIFIED_TRUTH_IN_MEMORY",
        "truth_sha256": truth_spec["sha256"],
        "truth_rows": len(truth),
        "sealed_resolver_intercepts": resolver_calls,
        "held_in_memory_truth_injections": in_memory_reads,
        "truth_path_reopens": 0,
        "outer_scores": 1,
    }
    return metrics, checks, receipt


def _worker_execute(
    config_path: Path,
    snapshot_manifest_path: Path,
    snapshot_manifest_sha256: str,
    deadline_epoch: float,
) -> Path:
    config, authorization, seal_receipt = _load_authorized(
        config_path, require_seal=True
    )
    assert seal_receipt is not None
    snapshot_root, snapshot_manifest = _verify_snapshot_manifest(
        snapshot_manifest_path,
        snapshot_manifest_sha256,
        config,
        authorization,
        seal_receipt,
    )
    numerical, runtime_receipt = _load_snapshot_numerical(snapshot_root, config)
    readiness, state = _strict_snapshot_readiness(
        snapshot_root,
        config,
        numerical,
        runtime_receipt,
    )
    if readiness != snapshot_manifest["strict_readiness"]:
        raise RuntimeError("worker readiness differs from parent snapshot receipt")
    if time.time() >= deadline_epoch:
        raise TimeoutError("absolute deadline expired before claim")

    artifact = _artifact_dir(config)
    _assert_output_namespace_ready(artifact)
    attempt = AttemptJournal.begin(
        artifact,
        deadline_epoch=deadline_epoch,
        snapshot_manifest_sha256=snapshot_manifest_sha256,
    )
    phase = "CLAIM_CREATED"
    try:
        phase = "READINESS_JOURNALED"
        attempt.record_readiness(readiness)
        # Defense in depth after the exclusive claim. Proper concurrent workers
        # cannot publish without this lock; any preexisting output is fail-closed.
        _assert_output_namespace_ready(artifact)

        phase = "MODEL_FIT_SETUP"
        train = state["train"]
        bundle = state["bundle"]
        folds = state["folds"]
        fold_context: dict[str, Any] = {}
        original_classifier = _install_fit_guard(numerical, attempt, fold_context)
        part_frames: list[Any] = []
        fit_audits: dict[str, Any] = {}
        try:
            for ordinal, fold in enumerate(folds):
                name = str(fold["name"])
                phase = f"FOLD_{name}_PRE_RESERVATION"
                if time.time() >= deadline_epoch:
                    raise TimeoutError("absolute deadline expired before fold")
                attempt.begin_fold(ordinal, name)
                fold_context.clear()
                fold_context.update({"name": name, "ordinal": ordinal})
                output, audit = numerical._fit_fold(train, bundle, config, fold)
                phase = f"FOLD_{name}_OUTPUT_PUBLISH"
                if int(audit["model_fits"]) != 3:
                    raise RuntimeError("sealed fold did not report three fits")
                part_path = artifact / "prediction_parts" / f"{name}.parquet"
                audit_path = artifact / "prediction_parts" / f"{name}.json"
                _atomic_parquet_new(part_path, output)
                audit.update(
                    {
                        "parquet_path": str(part_path.relative_to(PROJECT_ROOT)).replace(
                            "\\", "/"
                        ),
                        "parquet_sha256": _sha256(part_path),
                        "completed_at_kst": _now_kst(),
                    }
                )
                _atomic_json_new(audit_path, audit)
                attempt.complete_fold(ordinal, name, part_path, audit_path)
                part_frames.append(output)
                fit_audits[name] = audit
        finally:
            numerical.lgb.LGBMClassifier = original_classifier

        if attempt.reserved_fits != 9 or attempt.completed_fits != 9:
            raise RuntimeError("physical fit ledger differs from exactly nine")
        phase = "PREDICTION_ASSEMBLY_AND_PUBLISH"
        predictions = numerical.pd.concat(part_frames, ignore_index=True)
        if len(predictions) != int(config["surface"]["expected_rows"]):
            raise RuntimeError("prediction surface row count changed")
        predictions_path = artifact / "predictions.parquet"
        _atomic_parquet_new(predictions_path, predictions)
        complete_path = artifact / "predictions_complete.json"
        _atomic_json_new(
            complete_path,
            {
                "schema_version": "p1_round_b_residual.predictions_complete.v1r4",
                "experiment_id": config["experiment_id"],
                "status": "ALL_OUTER_PREDICTIONS_FROZEN_BEFORE_SCORING",
                "rows": len(predictions),
                "physical_fit_reservations": attempt.reserved_fits,
                "physical_fits_completed": attempt.completed_fits,
                "prediction_sha256": _sha256(predictions_path),
                "fold_audits": fit_audits,
                "left_censor_counts": state["left_counts"],
                "target_fold_scores_computed_before_receipt": 0,
                "completed_at_kst": _now_kst(),
            },
        )
        if time.time() >= deadline_epoch:
            raise TimeoutError("absolute deadline expired before outer scoring")

        phase = "HELD_TRUTH_OUTER_SCORING"
        metrics, checks, scoring_truth_receipt = _score_with_verified_truth(
            numerical,
            config,
            predictions,
            state["truth"],
            state["truth_read_receipt"],
        )
        metrics_path = artifact / "metrics.json"
        _atomic_json_new(metrics_path, metrics)
        passed = all(checks.values())
        result = {
            "schema_version": "p1_round_b_nonspike_long_event_residual.result.v1r4",
            "experiment_id": config["experiment_id"],
            "status": "COMPLETE_LOCAL_SCREEN_ONLY_PARENT_QA_PENDING",
            "decision": config["interpretation"]["pass_label"]
            if passed
            else config["interpretation"]["fail_label"],
            "passed_all_gates": passed,
            "attempt_id": attempt.attempt_id,
            "completed_at_kst": _now_kst(),
            "config_sha256": _sha256(config_path),
            "authorization_sha256": _sha256(AUTHORIZATION_PATH),
            "preexecution_seal_sha256": _sha256(
                artifact / "preexecution_seal.json"
            ),
            "snapshot_manifest_sha256": snapshot_manifest_sha256,
            "prediction_sha256": _sha256(predictions_path),
            "metrics_sha256": _sha256(metrics_path),
            "feature_cache_sha256": state["feature_metadata"]["parquet_sha256"],
            "feature_row_binding": readiness["feature_row_binding"],
            "full_feature_cache_binding": readiness["full_feature_cache_binding"],
            "scoring_truth_receipt": scoring_truth_receipt,
            "operation_counters": {
                "round_b_base_model_fits": 0,
                "residual_model_fit_reservations": attempt.reserved_fits,
                "residual_model_fits_completed": attempt.completed_fits,
                "outer_scores": 1,
                "full_fits": 0,
                "candidate_files": 0,
                "uploads": 0,
                "official_test_reads": 0,
                "sample_format_reads": 0,
                "submission_candidate_reads": 0,
            },
            "runtime": runtime_receipt,
            "independent_confirmation": config["interpretation"][
                "independent_confirmation"
            ],
        }
        result_payload = _json_bytes(result)
        result_path = artifact / "result.json"
        files: list[Path] = [
            config_path,
            AUTHORIZATION_PATH,
            artifact / "preexecution_seal.json",
            predictions_path,
            complete_path,
            metrics_path,
        ]
        files.extend(
            artifact / "prediction_parts" / f"{fold}.parquet" for fold in FOLD_ORDER
        )
        files.extend(
            artifact / "prediction_parts" / f"{fold}.json" for fold in FOLD_ORDER
        )
        artifacts = {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _file_record(path)
            for path in files
        }
        artifacts.update(attempt.manifest_records())
        artifacts[str(result_path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = {
            "bytes": len(result_payload),
            "sha256": _sha256_bytes(result_payload),
        }
        manifest = {
            "schema_version": "p1_round_b_nonspike_long_event_residual.manifest.v1r4",
            "experiment_id": config["experiment_id"],
            "attempt_id": attempt.attempt_id,
            "created_at_kst": _now_kst(),
            "authorization_sha256": _sha256(AUTHORIZATION_PATH),
            "snapshot_manifest_sha256": snapshot_manifest_sha256,
            "artifacts": artifacts,
            "terminal_order": [
                "result.json",
                "manifest.json",
                "attempt_journal/998_worker_terminal.json",
                "parent verification",
                "attempt_journal/999_completed.json",
                "release execution.lock",
            ],
        }
        phase = "RESULT_MANIFEST_WORKER_TERMINAL_PUBLISH"
        result_path, _manifest_path = _commit_worker_terminal_artifacts(
            artifact,
            result,
            manifest,
            attempt,
        )
        return result_path
    except BaseException as error:
        try:
            attempt.fail_terminal(phase, error)
        except BaseException as terminal_error:
            attempt.close_handle_keep_lock()
            raise RuntimeError(
                "claimed worker failure terminal could not be published"
            ) from terminal_error
        raise


def _worker_command(
    config_path: Path,
    snapshot_manifest_path: Path,
    snapshot_manifest_sha256: str,
    deadline_epoch: float,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--config",
        str(config_path),
        "--snapshot-manifest",
        str(snapshot_manifest_path),
        "--snapshot-manifest-sha256",
        snapshot_manifest_sha256,
        "--deadline-epoch",
        repr(deadline_epoch),
    ]


class ProcessTreeTerminationError(RuntimeError):
    def __init__(self, message: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.termination_receipt = dict(receipt)


class WorkerTimeoutError(TimeoutError):
    def __init__(self, message: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.termination_receipt = dict(receipt)


def _windows_process_table() -> dict[int, int]:
    command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        (
            "$ErrorActionPreference='Stop';"
            "@(Get-CimInstance Win32_Process | "
            "Select-Object ProcessId,ParentProcessId) | ConvertTo-Json -Compress"
        ),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Windows process-table query failed with exit code "
            f"{completed.returncode}: {completed.stderr[-1000:]}"
        )
    raw = completed.stdout.strip()
    values = [] if not raw else json.loads(raw)
    if isinstance(values, dict):
        values = [values]
    if not isinstance(values, list):
        raise RuntimeError("Windows process-table query returned an invalid payload")
    return {
        int(value["ProcessId"]): int(value["ParentProcessId"])
        for value in values
    }


def _descendant_pids(table: Mapping[int, int], root_pid: int) -> set[int]:
    descendants: set[int] = set()
    frontier = {int(root_pid)}
    while frontier:
        children = {
            int(pid)
            for pid, parent in table.items()
            if int(parent) in frontier and int(pid) not in descendants
        }
        descendants.update(children)
        frontier = children
    descendants.discard(int(root_pid))
    return descendants


def _terminate_windows_process_tree(
    process: subprocess.Popen[str],
    *,
    table_loader: Any | None = None,
    taskkill_runner: Any | None = None,
    confirmation_timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    loader = _windows_process_table if table_loader is None else table_loader
    killer = subprocess.run if taskkill_runner is None else taskkill_runner
    target_pid = int(process.pid)
    if process.poll() is not None:
        return {
            "platform": "windows",
            "target_pid": target_pid,
            "known_descendant_pids": [],
            "taskkill_returncode": None,
            "target_and_descendants_confirmed_gone": True,
            "already_exited": True,
        }

    before_error: str | None = None
    try:
        before = loader()
        descendants = _descendant_pids(before, target_pid)
    except BaseException as error:
        before_error = f"{type(error).__name__}: {str(error)[-1000:]}"
        descendants = set()
    known = {target_pid, *descendants}
    completed = killer(
        ["taskkill", "/PID", str(target_pid), "/T", "/F"],
        check=False,
        capture_output=True,
        text=True,
    )
    receipt: dict[str, Any] = {
        "platform": "windows",
        "target_pid": target_pid,
        "known_descendant_pids": sorted(descendants),
        "descendants_enumerated_before_taskkill": before_error is None,
        "pretermination_query_error": before_error,
        "taskkill_returncode": int(completed.returncode),
        "taskkill_stdout_tail": str(completed.stdout)[-1000:],
        "taskkill_stderr_tail": str(completed.stderr)[-1000:],
        "target_and_descendants_confirmed_gone": False,
        "already_exited": False,
    }
    try:
        process.wait(timeout=10)
        receipt["worker_wait_returncode"] = process.poll()
    except subprocess.TimeoutExpired:
        receipt["worker_wait_returncode"] = None

    alive = set(known)
    query_error: str | None = None
    confirmation_deadline = time.time() + confirmation_timeout_seconds
    while time.time() < confirmation_deadline:
        try:
            after = loader()
            alive = known.intersection(after)
            query_error = None
        except BaseException as error:
            query_error = f"{type(error).__name__}: {str(error)[-1000:]}"
            break
        if not alive:
            break
        time.sleep(0.05)
    receipt["posttermination_query_error"] = query_error
    receipt["pids_still_present"] = sorted(alive)
    confirmed = (
        int(completed.returncode) == 0
        and before_error is None
        and query_error is None
        and not alive
        and process.poll() is not None
    )
    receipt["target_and_descendants_confirmed_gone"] = confirmed
    if not confirmed:
        raise ProcessTreeTerminationError(
            "Windows taskkill rc/PID-descendant termination was not fully confirmed",
            receipt,
        )
    return receipt


def _terminate_process_tree(process: subprocess.Popen[str]) -> dict[str, Any]:
    if os.name == "nt":
        return _terminate_windows_process_tree(process)
    target_pid = int(process.pid)
    if process.poll() is None:
        try:
            os.killpg(target_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired as error:
        receipt = {
            "platform": os.name,
            "target_pid": target_pid,
            "target_process_group_confirmed_gone": False,
        }
        raise ProcessTreeTerminationError(
            "worker process group termination was not confirmed", receipt
        ) from error
    return {
        "platform": os.name,
        "target_pid": target_pid,
        "target_process_group_confirmed_gone": process.poll() is not None,
    }


def _run_supervised(command: Sequence[str], deadline_epoch: float) -> tuple[str, str]:
    remaining = deadline_epoch - time.time()
    if remaining <= 0:
        raise TimeoutError("absolute deadline expired before worker spawn")
    kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(list(command), **kwargs)
    try:
        stdout, stderr = process.communicate(timeout=max(0.0, deadline_epoch - time.time()))
    except subprocess.TimeoutExpired as error:
        try:
            termination = _terminate_process_tree(process)
        except ProcessTreeTerminationError as termination_error:
            raise WorkerTimeoutError(
                "absolute 1800-second timeout; worker tree termination failed closed",
                termination_error.termination_receipt,
            ) from termination_error
        raise WorkerTimeoutError(
            "absolute 1800-second parent timeout killed worker tree",
            termination,
        ) from error
    if process.returncode != 0:
        raise RuntimeError(
            f"worker failed with exit code {process.returncode}: {stderr[-4000:]}"
        )
    return stdout, stderr


def _verify_journal_chain(
    journal_dir: Path,
    *,
    required_last_name: str | None = "998_worker_terminal.json",
) -> tuple[str, list[dict[str, Any]]]:
    paths = sorted(journal_dir.glob("*.json"))
    if not paths:
        raise RuntimeError("attempt journal is empty")
    if required_last_name is not None and paths[-1].name != required_last_name:
        raise RuntimeError(f"required journal terminal is missing: {required_last_name}")
    previous: str | None = None
    entries: list[dict[str, Any]] = []
    for path in paths:
        value, read_receipt = _json_load_bound(path)
        if value.get("previous_entry_sha256") != previous:
            raise RuntimeError(f"journal hash chain mismatch: {path.name}")
        previous = str(read_receipt["actual_read_sha256"])
        entries.append(value)
    assert previous is not None
    return previous, entries


def _record_parent_failure_if_claimed(
    artifact: Path,
    phase: str,
    error: BaseException,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one create-only FAILED record only when a worker claim exists."""
    lock_path = artifact / "execution.lock"
    if not lock_path.is_file():
        return {"claim_present": False, "failed_terminal_created": False}
    journal_dir = artifact / "attempt_journal"
    if not journal_dir.exists():
        os.mkdir(journal_dir)
        _fsync_directory(artifact)

    paths = sorted(journal_dir.glob("*.json"))
    if paths:
        previous, entries = _verify_journal_chain(
            journal_dir,
            required_last_name=None,
        )
    else:
        previous = None
        entries = []
    failed_paths = [
        path
        for path in paths
        if path.name
        in {
            "997_failed.json",
            "999_failed.json",
            "999_postcompletion_failed.json",
        }
    ]
    if len(failed_paths) > 1:
        raise RuntimeError("multiple FAILED terminals already exist")
    if len(failed_paths) == 1:
        return {
            "claim_present": True,
            "failed_terminal_created": False,
            "failed_terminal_path": str(failed_paths[0]),
            "failed_terminal_sha256": _sha256(failed_paths[0]),
            "status": "FAILED_TERMINAL_ALREADY_PRESENT_EXACT_ONCE",
        }
    parent_completion_present = any(
        path.name == "999_completed.json" for path in paths
    )

    lock_value: dict[str, Any] = {}
    try:
        lock_value, _lock_read = _json_load_bound(lock_path)
    except BaseException:
        lock_value = {}
    attempt_id = str(
        entries[0].get("attempt_id")
        if entries
        else lock_value.get("attempt_id", "UNKNOWN_LOCKED_ATTEMPT")
    )
    reservations = [
        entry
        for entry in entries
        if entry.get("schema_version") == "p1_round_b_residual.fit_reserved.v1r4"
    ]
    completions = [
        entry
        for entry in entries
        if entry.get("schema_version") == "p1_round_b_residual.fit_completed.v1r4"
    ]
    worker_terminal_present = any(
        path.name == "998_worker_terminal.json" for path in paths
    )
    if parent_completion_present:
        terminal_name = "999_postcompletion_failed.json"
    elif worker_terminal_present:
        terminal_name = "999_failed.json"
    else:
        terminal_name = "997_failed.json"
    target = journal_dir / terminal_name
    payload = {
        "schema_version": "p1_round_b_residual.failed_terminal.v1r4",
        "attempt_id": attempt_id,
        "status": "FAILED_FAIL_CLOSED_LOCK_RETAINED",
        "failure_actor": "parent",
        "phase": str(phase),
        "error_type": type(error).__name__,
        "error_message": str(error)[-2000:],
        "physical_fit_reservations": len(reservations),
        "physical_fits_completed": len(completions),
        "fit_slot_state": {
            "maximum_lifetime_physical_model_fits": 9,
            "next_slot": len(reservations) + 1 if len(reservations) < 9 else None,
            "worker_success_terminal_present": worker_terminal_present,
            "parent_completion_record_present": parent_completion_present,
        },
        "journal_prefix": {
            "entry_count": len(paths),
            "entry_names": [path.name for path in paths],
            "last_entry_sha256": previous,
        },
        "failure_provenance": dict(provenance or {}),
        "previous_entry_sha256": previous,
        "failed_at_kst": _now_kst(),
    }
    _atomic_json_new(target, payload)
    final_sha, final_entries = _verify_journal_chain(
        journal_dir,
        required_last_name=target.name,
    )
    if final_entries[-1].get("status") != "FAILED_FAIL_CLOSED_LOCK_RETAINED":
        raise RuntimeError("parent FAILED terminal verification changed")
    return {
        "claim_present": True,
        "failed_terminal_created": True,
        "failed_terminal_path": str(target),
        "failed_terminal_sha256": final_sha,
        "status": "FAILED_TERMINAL_CREATED_EXACT_ONCE",
    }


def _publish_parent_completion(
    artifact: Path,
    payload: Mapping[str, Any],
    *,
    crash_after_completion: bool = False,
) -> Path:
    """Create 999 durably before releasing the exclusive namespace lock."""
    completion_path = artifact / "attempt_journal" / "999_completed.json"
    _atomic_json_new(completion_path, payload)
    if crash_after_completion:
        raise RuntimeError("injected crash after parent completion")
    lock_path = artifact / "execution.lock"
    lock_path.unlink()
    _fsync_directory(artifact)
    return completion_path


def _parent_finalize(config: Mapping[str, Any], expected_result_path: Path) -> Path:
    artifact = _artifact_dir(config)
    result_path = artifact / "result.json"
    manifest_path = artifact / "manifest.json"
    if expected_result_path.resolve(strict=True) != result_path.resolve(strict=True):
        raise RuntimeError("worker returned an unexpected result path")
    previous, entries = _verify_journal_chain(artifact / "attempt_journal")
    terminal = entries[-1]
    result, _result_read = _json_load_bound(
        result_path,
        expected_sha256=str(terminal["result_sha256"]),
    )
    manifest, _manifest_read = _json_load_bound(
        manifest_path,
        expected_sha256=str(terminal["manifest_sha256"]),
    )
    if result.get("attempt_id") != manifest.get("attempt_id"):
        raise RuntimeError("result/manifest attempt mismatch")
    if result.get("authorization_sha256") != AUTHORIZATION_SHA256:
        raise RuntimeError("result authorization mismatch")
    for relative, record in manifest["artifacts"].items():
        path = _resolve_repo_path(relative)
        if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"parent manifest verification failed: {relative}")
    if terminal.get("manifest_sha256") != _manifest_read["actual_read_sha256"]:
        raise RuntimeError("worker terminal manifest hash mismatch")
    reservations = [entry for entry in entries if entry.get("schema_version") == (
        "p1_round_b_residual.fit_reserved.v1r4"
    )]
    completions = [entry for entry in entries if entry.get("schema_version") == (
        "p1_round_b_residual.fit_completed.v1r4"
    )]
    if len(reservations) != 9 or len(completions) != 9:
        raise RuntimeError("parent physical-fit journal count differs from nine")
    _publish_parent_completion(
        artifact,
        {
            "schema_version": "p1_round_b_residual.parent_completed.v1r4",
            "attempt_id": result["attempt_id"],
            "status": "PARENT_INDEPENDENT_MANIFEST_QA_PASS",
            "physical_fit_reservations": 9,
            "physical_fits_completed": 9,
            "result_sha256": _sha256(result_path),
            "manifest_sha256": _sha256(manifest_path),
            "previous_entry_sha256": previous,
            "completed_at_kst": _now_kst(),
        },
    )
    return result_path


def _prepare_complete_snapshot(
    config: Mapping[str, Any],
    authorization: Mapping[str, Any],
    seal_receipt: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    stdlib_readiness = _stdlib_readiness(config)
    snapshot_root, records = _prepare_snapshot(config, seal_receipt, stdlib_readiness)
    try:
        numerical, runtime = _load_snapshot_numerical(snapshot_root, config)
        readiness, _state = _strict_snapshot_readiness(
            snapshot_root,
            config,
            numerical,
            runtime,
        )
        manifest_path = _finalize_snapshot_manifest(
            snapshot_root,
            records,
            readiness,
            config,
            authorization,
            seal_receipt,
        )
        return snapshot_root, manifest_path, readiness
    except BaseException:
        _cleanup_snapshot(snapshot_root)
        raise


def preflight(config_path: Path) -> dict[str, Any]:
    config, authorization, seal_receipt = _load_authorized(
        config_path, require_seal=True
    )
    assert seal_receipt is not None
    snapshot_root, manifest_path, readiness = _prepare_complete_snapshot(
        config, authorization, seal_receipt
    )
    try:
        manifest_sha = _sha256(manifest_path)
        _verify_snapshot_manifest(
            manifest_path,
            manifest_sha,
            config,
            authorization,
            seal_receipt,
        )
        return {
            "schema_version": "p1_round_b_nonspike_long_event_residual.preflight.v1r4",
            "experiment_id": config["experiment_id"],
            "status": "PASS_STRICT_COMPLETE_READINESS_NO_CLAIM_NO_FIT",
            "strict_readiness": readiness,
            "snapshot_manifest_sha256": manifest_sha,
            "artifact_claim_created": False,
            "residual_model_fits": 0,
            "outer_scores": 0,
            "official_test_reads": 0,
            "sample_format_reads": 0,
            "submission_candidate_reads": 0,
        }
    finally:
        _cleanup_snapshot(snapshot_root)


def execute_parent(config_path: Path) -> Path:
    started_epoch = time.time()
    config, authorization, seal_receipt = _load_authorized(
        config_path, require_seal=True
    )
    assert seal_receipt is not None
    artifact = _artifact_dir(config)
    deadline_epoch = started_epoch + float(
        config["execution_safety"]["parent_absolute_timeout_seconds"]
    )
    snapshot_root: Path | None = None
    phase = "PARENT_PRECLAIM_COMPLETE_READINESS"
    try:
        snapshot_root, manifest_path, _readiness = _prepare_complete_snapshot(
            config, authorization, seal_receipt
        )
        manifest_sha = _sha256(manifest_path)
        command = _worker_command(
            config_path,
            manifest_path,
            manifest_sha,
            deadline_epoch,
        )
        phase = "PARENT_WORKER_SUPERVISION"
        stdout, _stderr = _run_supervised(command, deadline_epoch)
        phase = "PARENT_WORKER_RECEIPT_VALIDATION"
        lines = [line for line in stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("worker returned no receipt")
        worker_receipt = json.loads(lines[-1])
        if worker_receipt.get("status") != "worker_ok":
            raise RuntimeError("worker terminal receipt is invalid")
        phase = "PARENT_INDEPENDENT_RESULT_MANIFEST_QA"
        return _parent_finalize(config, Path(str(worker_receipt["result_path"])))
    except BaseException as error:
        termination = getattr(error, "termination_receipt", None)
        _record_parent_failure_if_claimed(
            artifact,
            phase,
            error,
            provenance=termination if isinstance(termination, Mapping) else None,
        )
        raise
    finally:
        if snapshot_root is not None:
            _cleanup_snapshot(snapshot_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--seal", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-manifest-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--deadline-epoch", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    if args.seal:
        output: Any = str(seal(config_path))
        status = "ok"
    elif args.preflight:
        output = preflight(config_path)
        status = "ok"
    elif args.execute:
        output = str(execute_parent(config_path))
        status = "ok"
    else:
        if (
            args.snapshot_manifest is None
            or args.snapshot_manifest_sha256 is None
            or args.deadline_epoch is None
        ):
            raise RuntimeError("hidden worker requires a complete parent capability")
        result = _worker_execute(
            config_path,
            args.snapshot_manifest.resolve(strict=True),
            str(args.snapshot_manifest_sha256),
            float(args.deadline_epoch),
        )
        print(
            json.dumps(
                {"status": "worker_ok", "result_path": str(result)},
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return
    print(json.dumps({"status": status, "output": output}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
