"""Final zero-fit r8 trust successor for the frozen P1 Cycle-1 screen.

The parent builds a sealed private Python home and runtime.  The hidden worker
must authenticate its true OS parent through three inherited handles and a
32-byte pipe-only capability, then independently hold every canonical/live and
private byte before the unchanged r6 numerical graph can be reached.
"""

from __future__ import annotations

import argparse
import ast
import csv
import ctypes
import functools
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType
from typing import Any

sys.dont_write_bytecode = True

CANONICAL_PROJECT_ROOT = Path("C:/Users/cedis/PycharmProjects/PythonProject")
CANONICAL_PROJECT_ROOT_TEXT = "C:/Users/cedis/PycharmProjects/PythonProject"
CANONICAL_ARTIFACT_RELATIVE = (
    "artifacts/p1_long_event_segment_proposal_rescore_20260826_v2_infrastructure_v8"
)
CANONICAL_ARTIFACT_DIR = CANONICAL_PROJECT_ROOT / CANONICAL_ARTIFACT_RELATIVE
SEAL_PATH = CANONICAL_ARTIFACT_DIR / "preexecution_seal.json"
LAUNCH_CLAIM_PATH = CANONICAL_ARTIFACT_DIR / "canonical_launch.claim"
WORKER_LEASE_PATH = CANONICAL_ARTIFACT_DIR / "worker_start.lease"

EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_20260826_v2_infrastructure_v8"
SCIENTIFIC_EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_20260826_v1"
SINGLE_ATTEMPT_NAMESPACE = "P1_CYCLE1_SEGMENT_RESCORE_CANONICAL_V8_ONE_SHOT"
AUTHORIZATION_ENV_VAR = "P1_LONG_EVENT_SEGMENT_V8_AUTHORIZATION_SHA256"
AUTHORIZATION_SCHEMA_VERSION = (
    "p1_long_event_segment_proposal_rescore.execution_authorization.v8"
)
SEAL_SCHEMA_VERSION = "p1_long_event_segment_proposal_rescore.preexecution_seal.v8"
MANIFEST_SCHEMA_VERSION = "p1_segment_rescore.private_transport_manifest.v8"
CLAIM_SCHEMA_VERSION = "p1_segment_rescore.canonical_launch_claim.v8"
PYTHON_HOME_SCHEMA_VERSION = "p1_segment_rescore.isolated_python_home_inventory.v8"
SNAPSHOT_SCHEMA_VERSION = "p1_segment_rescore.full_snapshot_inventory.v8"

MAXIMUM_LIFETIME_PHYSICAL_FITS = 72
MAXIMUM_SCIENTIFIC_MATERIALIZATIONS = 21
MAXIMUM_OUTER_SCORES = 1
HARD_WALL_SECONDS = 21600
MAXIMUM_LAUNCH_AGE_SECONDS = 300
CAPABILITY_BYTES = 32

R8_AMENDMENT_PATH = CANONICAL_PROJECT_ROOT / (
    "configs/experiments/"
    "p1_long_event_segment_proposal_rescore_v8_isolated_python_parent_capability_amendment.json"
)
R8_EXECUTION_MODULE_PATH = CANONICAL_PROJECT_ROOT / (
    "src/p1_qc/long_event_segment_proposal_rescore_execution_v8.py"
)
R8_TEST_PATH = CANONICAL_PROJECT_ROOT / (
    "tests/test_run_p1_long_event_segment_proposal_rescore_v8.py"
)
R8_AUTHORIZATION_TEMPLATE_PATH = CANONICAL_PROJECT_ROOT / (
    "configs/experiments/"
    "p1_long_event_segment_proposal_rescore_v8_execution_authorization_template.json"
)
R8_AUTHORIZATION_PATH = CANONICAL_PROJECT_ROOT / (
    "configs/experiments/"
    "p1_long_event_segment_proposal_rescore_v8_execution_authorization.json"
)
R8_QA_PATH = CANONICAL_PROJECT_ROOT / (
    "reports/p1_long_event_segment_proposal_rescore_v8_independent_preexecution_qa_20260826.json"
)
R7_QA_PATH = CANONICAL_PROJECT_ROOT / (
    "reports/p1_long_event_segment_proposal_rescore_v7_independent_preexecution_qa_20260826.json"
)
R7_RUNNER_PATH = CANONICAL_PROJECT_ROOT / (
    "scripts/run_p1_long_event_segment_proposal_rescore_v7.py"
)
R7_SEAL_PATH = CANONICAL_PROJECT_ROOT / (
    "artifacts/p1_long_event_segment_proposal_rescore_20260826_v2_infrastructure_v7/"
    "preexecution_seal.json"
)

R8_AMENDMENT_SHA256 = (
    "003672cac416b5a691357ef108c0401b5bfb9661d4daa3adbe3c6efce68f4b61"
)
R7_QA_SHA256 = "066d51a32a16175a1e0fb5829ff44d6699ea398901d38a1fbbd92e51a20fe6ff"
R7_RUNNER_SHA256 = "0f481298beabe9f1a16fdf47320ba107159db6fd91aa40d7a0f18007a5b23b6a"
R7_SEAL_SHA256 = "4763719879ecd677a5be8193050efe1334def50dd44c289d0adb80e22e263776"
R6_EXECUTION_MODULE_SHA256 = (
    "8b91db9234bde301730728dcffa4fd2c014b099d0c518571ddaa733608b81636"
)
REQUIRED_VC_RUNTIME = {
    "vcruntime140.dll": {
        "bytes": 120400,
        "sha256": "052ad6a20d375957e82aa6a3c441ea548d89be0981516ca7eb306e063d5027f4",
    },
    "vcruntime140_1.dll": {
        "bytes": 49776,
        "sha256": "6a99bc0128e0c7d6cbbf615fcc26909565e17d4ca3451b97f8987f9c6acbc6c8",
    },
}
NONCE_RE = re.compile(r"[0-9a-f]{64}\Z")
_SENSITIVE_NAMES = {"test.csv", "sample_submission.csv", "submission.csv"}


class AuthorizationError(RuntimeError):
    """Fail before a scientific/canonical namespace mutation."""


class WireFormatError(ValueError):
    """Exact security JSON wire contract failed."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_record(value: Any) -> dict[str, Any]:
    raw = _canonical_bytes(value)
    return {"bytes": len(raw), "sha256": _sha256_bytes(raw)}


def _reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WireFormatError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise WireFormatError(f"non-finite JSON constant: {value}")


def _reject_nonfinite(value: Any, label: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise WireFormatError(f"non-finite JSON number at {label}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{label}[{index}]")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, f"{label}.{key}")


def _json_from_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WireFormatError(f"invalid JSON bytes for {label}") from error
    _reject_nonfinite(value)
    if type(value) is not dict:
        raise WireFormatError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise AuthorizationError(f"{label} exact key set changed")
    return value


def _exact_digest(value: Any, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise AuthorizationError(f"{label} must be lowercase SHA-256")
    return value


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise AuthorizationError(f"{label} must be an integer")
    return value


def _reject_sensitive_path(path: Path) -> None:
    lowered = path.name.lower()
    if lowered in _SENSITIVE_NAMES or "submission" in lowered or "candidate" in lowered:
        raise RuntimeError("forbidden official/submission/candidate path")


def _assert_canonical_root() -> Path:
    actual = CANONICAL_PROJECT_ROOT.resolve(strict=True)
    if os.path.normcase(str(actual)) != os.path.normcase(CANONICAL_PROJECT_ROOT_TEXT):
        raise RuntimeError("canonical project root identity changed")
    return actual


def _project_relative(path: Path) -> str:
    root = _assert_canonical_root()
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise RuntimeError("path escapes canonical repository")
    return resolved.relative_to(root).as_posix()


def _atomic_create_bytes(path: Path, raw: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_BINARY, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    _fsync_directory(path.parent)
    return path


def _atomic_create_json(path: Path, value: Any) -> Path:
    return _atomic_create_bytes(path, _json_bytes(value))


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_module(path: Path, name: str, expected_sha256: str) -> ModuleType:
    if _sha256(path) != expected_sha256:
        raise RuntimeError(f"sealed dependency changed: {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sealed dependency: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_r7(path: Path = R7_RUNNER_PATH) -> ModuleType:
    return _load_module(path, "_p1_segment_rescore_r7_for_v8", R7_RUNNER_SHA256)


def _verify_top_level_stdlib_only(path: Path | None = None) -> list[str]:
    target = path or Path(__file__)
    tree = ast.parse(target.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module.split(".", 1)[0])
    forbidden = {"numpy", "pandas", "lightgbm", "sklearn", "scipy", "pyarrow", "p1_qc"}
    if forbidden.intersection(names):
        raise RuntimeError("numerical/project import exists before trust activation")
    return names


def _source_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    attributes = int(getattr(stat, "st_file_attributes", 0))
    if resolved.is_symlink() or attributes & 0x00000400:
        raise RuntimeError("source inventory contains a link/reparse point")
    return {
        "resolved_path": str(resolved),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _file_inventory_record(path: Path) -> tuple[str, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    return str(resolved), {
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
        "source_identity": _source_identity(resolved),
    }


def _base_python_root() -> Path:
    executable = Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
    root = executable.parent
    if executable.name.lower() != "python.exe" or not (root / "Lib/os.py").is_file():
        raise RuntimeError("base CPython installation cannot be sealed")
    return root


def _python_home_source_paths(root: Path | None = None) -> list[Path]:
    base = (root or _base_python_root()).resolve(strict=True)
    selected: set[Path] = set()
    root_names = {
        "python.exe",
        "python3.dll",
        f"python{sys.version_info.major}{sys.version_info.minor}.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
    }
    for name in root_names:
        path = base / name
        if path.exists():
            selected.add(path.resolve(strict=True))
    for directory in (base / "DLLs", base / "Lib", base / "tcl"):
        if not directory.is_dir():
            raise RuntimeError(f"isolated Python source directory missing: {directory.name}")
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            resolved = path.resolve(strict=True)
            if directory.name == "Lib" and resolved.is_relative_to(
                (base / "Lib/site-packages").resolve()
            ):
                continue
            selected.add(resolved)
    paths = sorted(selected, key=lambda value: str(value).casefold())
    if len(paths) != 4044:
        raise RuntimeError(f"isolated Python inventory count changed: {len(paths)} != 4044")
    return paths


@functools.lru_cache(maxsize=2)
def _python_home_inventory(root: Path | None = None) -> dict[str, Any]:
    base = (root or _base_python_root()).resolve(strict=True)
    paths = _python_home_source_paths(base)
    workers = min(16, max(4, (os.cpu_count() or 4)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        absolute_records = dict(pool.map(_file_inventory_record, paths))
    files: dict[str, dict[str, Any]] = {}
    seen_casefold: set[str] = set()
    for path in paths:
        relative = path.relative_to(base).as_posix()
        folded = relative.casefold()
        if folded in seen_casefold:
            raise RuntimeError("isolated Python inventory has a case collision")
        seen_casefold.add(folded)
        files[relative] = absolute_records[str(path)]
    for name, expected in REQUIRED_VC_RUNTIME.items():
        observed = files.get(name)
        if observed is None or {
            "bytes": observed["bytes"],
            "sha256": observed["sha256"],
        } != expected:
            raise RuntimeError(f"required VC runtime pin changed: {name}")
    executable = files.get("python.exe")
    if executable is None:
        raise RuntimeError("isolated Python executable is absent")
    inventory = {
        "schema_version": PYTHON_HOME_SCHEMA_VERSION,
        "source_root": str(base),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "executable_relative_path": "python.exe",
        "complete_stdlib": True,
        "live_site_packages_included": False,
        "file_count": len(files),
        "total_bytes": sum(int(value["bytes"]) for value in files.values()),
        "files": files,
        "required_vc_runtime": REQUIRED_VC_RUNTIME,
    }
    return inventory


@functools.lru_cache(maxsize=1)
def _snapshot_static_inventory() -> tuple[dict[str, Any], dict[str, Any]]:
    r7 = _load_r7()
    inherited, inherited_bindings = r7._snapshot_static_inventory()
    files = dict(inherited)
    bindings = dict(inherited_bindings)
    additions = (
        R7_QA_PATH,
        R8_AMENDMENT_PATH,
        Path(__file__).resolve(strict=True),
        R8_EXECUTION_MODULE_PATH,
        R8_TEST_PATH,
    )
    for path in additions:
        relative = _project_relative(path)
        record = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        existing = files.get(relative)
        if existing is not None and {
            "bytes": existing["bytes"],
            "sha256": existing["sha256"],
        } != {"bytes": record["bytes"], "sha256": record["sha256"]}:
            raise RuntimeError(f"r8 snapshot collides with inherited bytes: {relative}")
        files[relative] = record
        bindings[relative] = {
            "source_path": relative,
            "source_identity": _source_identity(path),
        }
    return dict(sorted(files.items())), dict(sorted(bindings.items()))


def _fixed_pins() -> dict[str, dict[str, Any]]:
    paths = (
        R8_AMENDMENT_PATH,
        R7_QA_PATH,
        R7_RUNNER_PATH,
        R7_SEAL_PATH,
        Path(__file__).resolve(strict=True),
        R8_EXECUTION_MODULE_PATH,
        R8_TEST_PATH,
    )
    return {
        _project_relative(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in paths
    }


def _verify_fixed_anchors() -> None:
    expected = {
        _project_relative(R8_AMENDMENT_PATH): R8_AMENDMENT_SHA256,
        _project_relative(R7_QA_PATH): R7_QA_SHA256,
        _project_relative(R7_RUNNER_PATH): R7_RUNNER_SHA256,
        _project_relative(R7_SEAL_PATH): R7_SEAL_SHA256,
    }
    for relative, digest in expected.items():
        if _sha256(CANONICAL_PROJECT_ROOT / relative) != digest:
            raise RuntimeError(f"immutable predecessor anchor changed: {relative}")


@functools.lru_cache(maxsize=1)
def _science_preflight() -> dict[str, Any]:
    r7 = _load_r7()
    receipt = r7.read_only_preflight()
    if receipt["operation_counters"] != {
        "claims": 0,
        "physical_fits": 0,
        "scientific_materializations": 0,
        "outer_scores": 0,
        "candidate_files": 0,
        "official_reads": 0,
        "uploads": 0,
    }:
        raise RuntimeError("r7 inherited readiness is not zero-operation")
    r7_seal = _json_from_bytes(
        R7_SEAL_PATH.read_bytes(), label="immutable r7 preexecution seal"
    )
    science = r7_seal["science_preflight"]
    return {
        "r7_preflight_verification_sha256": receipt["verification_sha256"],
        "selected_scientific_readiness": science["selected_scientific_readiness"],
        "selected_scientific_readiness_sha256": science[
            "selected_scientific_readiness_sha256"
        ],
        "status": science["selected_scientific_readiness"]["status"],
    }


@functools.lru_cache(maxsize=1)
def _full_runtime_inventory() -> dict[str, Any]:
    return _load_r7()._full_runtime_inventory()


def _preflight_core() -> dict[str, Any]:
    _verify_top_level_stdlib_only()
    _verify_fixed_anchors()
    runtime = _full_runtime_inventory()
    snapshot, bindings = _snapshot_static_inventory()
    python_home = _python_home_inventory()
    security_providers = _security_provider_inventory()
    science = _science_preflight()
    return {
        "experiment_id": EXPERIMENT_ID,
        "scientific_experiment_id": SCIENTIFIC_EXPERIMENT_ID,
        "canonical_project_root": CANONICAL_PROJECT_ROOT_TEXT,
        "canonical_artifact_relative_path": CANONICAL_ARTIFACT_RELATIVE,
        "single_attempt_namespace": SINGLE_ATTEMPT_NAMESPACE,
        "fixed_files": _fixed_pins(),
        "python_home_inventory_binding": _canonical_record(python_home),
        "runtime_inventory_binding": _canonical_record(runtime),
        "snapshot_inventory_binding": _canonical_record(snapshot),
        "security_provider_inventory_binding": _canonical_record(security_providers),
        "snapshot_source_binding_sha256": _sha256_bytes(_canonical_bytes(bindings)),
        "python_home_file_count": len(python_home["files"]),
        "runtime_file_count": len(runtime["files"]),
        "snapshot_file_count": len(snapshot),
        "security_provider_file_count": len(security_providers),
        "science_preflight": science,
        "operation_counters": {
            "actual_authorizations": 0,
            "claims": 0,
            "physical_fits": 0,
            "scientific_materializations": 0,
            "outer_scores": 0,
            "candidate_files": 0,
            "uploads": 0,
        },
    }


def _now_kst_after_tzdata_activation() -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def _read_exact_json(path: Path, expected_sha256: str, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if _sha256_bytes(raw) != expected_sha256:
        raise RuntimeError(f"{label} digest changed")
    return _json_from_bytes(raw, label=label), raw


def _seal_value() -> dict[str, Any]:
    core = _preflight_core()
    runtime = _full_runtime_inventory()
    snapshot, bindings = _snapshot_static_inventory()
    python_home = _python_home_inventory()
    security_providers = _security_provider_inventory()
    return {
        "schema_version": SEAL_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "scientific_experiment_id": SCIENTIFIC_EXPERIMENT_ID,
        "status": "SEALED_ZERO_OPERATION_PENDING_INDEPENDENT_QA",
        "sealed_at_kst": _now_kst_after_tzdata_activation(),
        "canonical_project_root": CANONICAL_PROJECT_ROOT_TEXT,
        "canonical_artifact_relative_path": CANONICAL_ARTIFACT_RELATIVE,
        "single_attempt_namespace": SINGLE_ATTEMPT_NAMESPACE,
        "hard_wall_seconds": HARD_WALL_SECONDS,
        "maximum_launch_age_seconds": MAXIMUM_LAUNCH_AGE_SECONDS,
        "maximum_lifetime_physical_fits": MAXIMUM_LIFETIME_PHYSICAL_FITS,
        "maximum_scientific_materializations": MAXIMUM_SCIENTIFIC_MATERIALIZATIONS,
        "maximum_outer_scores": MAXIMUM_OUTER_SCORES,
        "r8_amendment": {
            "path": _project_relative(R8_AMENDMENT_PATH),
            "bytes": R8_AMENDMENT_PATH.stat().st_size,
            "sha256": R8_AMENDMENT_SHA256,
        },
        "r7_independent_qa": {
            "path": _project_relative(R7_QA_PATH),
            "bytes": R7_QA_PATH.stat().st_size,
            "sha256": R7_QA_SHA256,
            "verdict": "NO_GO",
        },
        "fixed_files": core["fixed_files"],
        "python_home_inventory_binding": core["python_home_inventory_binding"],
        "runtime_inventory_binding": core["runtime_inventory_binding"],
        "snapshot_inventory_binding": core["snapshot_inventory_binding"],
        "security_provider_inventory_binding": core[
            "security_provider_inventory_binding"
        ],
        "full_python_home_inventory": python_home,
        "full_runtime_inventory": runtime,
        "snapshot_static_inventory": snapshot,
        "snapshot_source_bindings": bindings,
        "host_security_provider_inventory": security_providers,
        "science_preflight": core["science_preflight"],
        "authorization_contract": {
            "schema_version": AUTHORIZATION_SCHEMA_VERSION,
            "canonical_live_authority_required": True,
            "manifest_role": "TRANSPORT_RECEIPT_NEVER_AUTHORITY",
            "pipe_only_parent_capability_bytes": CAPABILITY_BYTES,
            "inherited_handles": ["anonymous_pipe", "true_parent_process", "canonical_claim"],
            "claim_created_before_worker": True,
            "existing_or_partial_claim_permanently_blocks_replay": True,
            "actual_authorization_absent_at_seal": not R8_AUTHORIZATION_PATH.exists(),
            "fresh_independent_qa_required": True,
        },
        "bootstrap_contract": {
            "copied_interpreter_flags": ["-I", "-S", "-B"],
            "complete_python_home": True,
            "live_python_home_forbidden": True,
            "live_site_packages_forbidden": True,
            "KST_before_verified_tzdata": False,
            "destination_directory_acl_freeze_required": True,
            "worker_lifetime_holds_required": True,
        },
        "operation_counters": core["operation_counters"],
        "forbidden_actions": {
            "official_test_reads": 0,
            "sample_submission_reads": 0,
            "submission_candidate_reads": 0,
            "candidate_files": 0,
            "uploads": 0,
            "P3_actions": 0,
        },
    }


def seal() -> Path:
    if R8_AUTHORIZATION_PATH.exists() or LAUNCH_CLAIM_PATH.exists():
        raise RuntimeError("r8 cannot seal after authorization or claim")
    if CANONICAL_ARTIFACT_DIR.exists():
        existing = list(CANONICAL_ARTIFACT_DIR.iterdir())
        if existing:
            raise FileExistsError("r8 seal namespace already contains bytes")
    else:
        CANONICAL_ARTIFACT_DIR.mkdir(parents=True)
    value = _seal_value()
    _atomic_create_json(SEAL_PATH, value)
    return SEAL_PATH


def _read_seal() -> tuple[dict[str, Any], bytes, str]:
    raw = SEAL_PATH.read_bytes()
    value = _json_from_bytes(raw, label="r8 preexecution seal")
    if value.get("schema_version") != SEAL_SCHEMA_VERSION:
        raise RuntimeError("r8 seal schema changed")
    return value, raw, _sha256_bytes(raw)


def _verify_seal_against_live(seal_value: Mapping[str, Any]) -> dict[str, Any]:
    core = _preflight_core()
    runtime = _full_runtime_inventory()
    snapshot, bindings = _snapshot_static_inventory()
    python_home = _python_home_inventory()
    security_providers = _security_provider_inventory()
    comparisons = {
        "fixed_files": core["fixed_files"],
        "python_home_inventory_binding": _canonical_record(python_home),
        "runtime_inventory_binding": _canonical_record(runtime),
        "snapshot_inventory_binding": _canonical_record(snapshot),
        "security_provider_inventory_binding": _canonical_record(security_providers),
        "full_python_home_inventory": python_home,
        "full_runtime_inventory": runtime,
        "snapshot_static_inventory": snapshot,
        "snapshot_source_bindings": bindings,
        "host_security_provider_inventory": security_providers,
        "science_preflight": core["science_preflight"],
        "operation_counters": core["operation_counters"],
    }
    for key, expected in comparisons.items():
        if seal_value.get(key) != expected:
            raise RuntimeError(f"live state differs from r8 seal: {key}")
    if seal_value.get("r8_amendment", {}).get("sha256") != R8_AMENDMENT_SHA256:
        raise RuntimeError("r8 amendment seal pin changed")
    if seal_value.get("r7_independent_qa", {}).get("sha256") != R7_QA_SHA256:
        raise RuntimeError("r7 QA seal pin changed")
    return core


def read_only_preflight() -> dict[str, Any]:
    if R8_AUTHORIZATION_PATH.exists():
        raise RuntimeError("actual r8 authorization unexpectedly exists")
    if LAUNCH_CLAIM_PATH.exists():
        raise RuntimeError("canonical r8 claim unexpectedly exists")
    seal_value, _raw, seal_sha = _read_seal()
    core = _verify_seal_against_live(seal_value)
    allowed = {SEAL_PATH.name}
    actual = {path.name for path in CANONICAL_ARTIFACT_DIR.iterdir()}
    if actual != allowed:
        raise RuntimeError("r8 zero-operation namespace is not seal-only")
    stable = {
        "status": "PASS_ZERO_OPERATION_NOT_AUTHORIZED",
        "experiment_id": EXPERIMENT_ID,
        "preexecution_seal_sha256": seal_sha,
        "r8_amendment_sha256": R8_AMENDMENT_SHA256,
        "r7_qa_sha256": R7_QA_SHA256,
        "python_home_inventory_binding": core["python_home_inventory_binding"],
        "runtime_inventory_binding": core["runtime_inventory_binding"],
        "snapshot_inventory_binding": core["snapshot_inventory_binding"],
        "security_provider_inventory_binding": core[
            "security_provider_inventory_binding"
        ],
        "python_home_file_count": core["python_home_file_count"],
        "runtime_file_count": core["runtime_file_count"],
        "snapshot_file_count": core["snapshot_file_count"],
        "security_provider_file_count": core["security_provider_file_count"],
        "operation_counters": core["operation_counters"],
    }
    stable["verification_sha256"] = _sha256_bytes(_canonical_bytes(stable))
    return stable


def _require_authorization_schema(value: Any) -> dict[str, Any]:
    auth = _exact_keys(
        value,
        {
            "schema_version",
            "status",
            "authorized",
            "experiment_id",
            "scientific_experiment_id",
            "canonical_project_root",
            "canonical_artifact_relative_path",
            "single_attempt_namespace",
            "r8_amendment",
            "runner",
            "execution_module",
            "preexecution_seal",
            "r7_independent_qa",
            "r8_independent_qa",
            "python_home_inventory_binding",
            "runtime_inventory_binding",
            "snapshot_inventory_binding",
            "security_provider_inventory_binding",
            "operation_authorization",
            "zero_prior_state",
        },
        "r8 authorization",
    )
    if auth["schema_version"] != AUTHORIZATION_SCHEMA_VERSION:
        raise AuthorizationError("r8 authorization schema changed")
    for name in (
        "r8_amendment",
        "runner",
        "execution_module",
        "preexecution_seal",
        "r7_independent_qa",
        "r8_independent_qa",
        "python_home_inventory_binding",
        "runtime_inventory_binding",
        "snapshot_inventory_binding",
        "security_provider_inventory_binding",
    ):
        item = _exact_keys(auth[name], {"path", "bytes", "sha256"}, f"authorization.{name}")
        if type(item["path"]) is not str or type(item["bytes"]) is not int:
            raise AuthorizationError(f"authorization.{name} primitive type changed")
        _exact_digest(item["sha256"], f"authorization.{name}.sha256")
    operations = _exact_keys(
        auth["operation_authorization"],
        {
            "single_attempt",
            "maximum_lifetime_physical_fits",
            "maximum_scientific_materializations",
            "outer_scores",
            "candidate_files",
            "uploads",
        },
        "operation_authorization",
    )
    zero = _exact_keys(
        auth["zero_prior_state"],
        {
            "actual_authorizations",
            "claims",
            "physical_fits",
            "scientific_materializations",
            "outer_scores",
            "candidate_files",
            "uploads",
        },
        "zero_prior_state",
    )
    if any(type(item) is not int or item != 0 for item in zero.values()):
        raise AuthorizationError("r8 zero-prior state changed")
    expected_operations = {
        "single_attempt": True,
        "maximum_lifetime_physical_fits": 72,
        "maximum_scientific_materializations": 21,
        "outer_scores": 1,
        "candidate_files": 0,
        "uploads": 0,
    }
    if operations != expected_operations:
        raise AuthorizationError("r8 operation authorization changed")
    return auth


def _validate_live_authorization(
    authorization: Mapping[str, Any],
    seal_value: Mapping[str, Any],
    seal_sha256: str,
    qa_value: Mapping[str, Any],
    qa_sha256: str,
) -> None:
    if authorization.get("authorized") is not True:
        raise AuthorizationError("r8 actual authorization is false")
    if authorization.get("status") != "AUTHORIZED_AFTER_INDEPENDENT_QA_PASS":
        raise AuthorizationError("r8 authorization status is not final PASS")
    exact_literals = {
        "experiment_id": EXPERIMENT_ID,
        "scientific_experiment_id": SCIENTIFIC_EXPERIMENT_ID,
        "canonical_project_root": CANONICAL_PROJECT_ROOT_TEXT,
        "canonical_artifact_relative_path": CANONICAL_ARTIFACT_RELATIVE,
        "single_attempt_namespace": SINGLE_ATTEMPT_NAMESPACE,
    }
    for key, expected in exact_literals.items():
        if authorization.get(key) != expected:
            raise AuthorizationError(f"authorization canonical literal changed: {key}")
    expected_pins = {
        "r8_amendment": {
            "path": _project_relative(R8_AMENDMENT_PATH),
            "bytes": R8_AMENDMENT_PATH.stat().st_size,
            "sha256": R8_AMENDMENT_SHA256,
        },
        "runner": {
            "path": _project_relative(Path(__file__).resolve(strict=True)),
            "bytes": Path(__file__).stat().st_size,
            "sha256": _sha256(Path(__file__)),
        },
        "execution_module": {
            "path": _project_relative(R8_EXECUTION_MODULE_PATH),
            "bytes": R8_EXECUTION_MODULE_PATH.stat().st_size,
            "sha256": _sha256(R8_EXECUTION_MODULE_PATH),
        },
        "preexecution_seal": {
            "path": _project_relative(SEAL_PATH),
            "bytes": SEAL_PATH.stat().st_size,
            "sha256": seal_sha256,
        },
        "r7_independent_qa": {
            "path": _project_relative(R7_QA_PATH),
            "bytes": R7_QA_PATH.stat().st_size,
            "sha256": R7_QA_SHA256,
        },
        "r8_independent_qa": {
            "path": _project_relative(R8_QA_PATH),
            "bytes": R8_QA_PATH.stat().st_size,
            "sha256": qa_sha256,
        },
    }
    for key, expected in expected_pins.items():
        if authorization.get(key) != expected:
            raise AuthorizationError(f"authorization file pin changed: {key}")
    for key in (
        "python_home_inventory_binding",
        "runtime_inventory_binding",
        "snapshot_inventory_binding",
        "security_provider_inventory_binding",
    ):
        expected = dict(seal_value[key], path=f"seal:{key}")
        if authorization.get(key) != expected:
            raise AuthorizationError(f"authorization inventory pin changed: {key}")
    if qa_value.get("verdict") != "PASS" or qa_value.get("authorization_recommended") is not True:
        raise AuthorizationError("independent r8 QA did not authorize execution")
    if qa_value.get("experiment_id") != EXPERIMENT_ID:
        raise AuthorizationError("independent r8 QA experiment identity changed")
    if qa_value.get("preexecution_seal_sha256") != seal_sha256:
        raise AuthorizationError("independent r8 QA seal pin changed")


def _validate_nonce(value: Any, seen: set[str] | None = None) -> str:
    if type(value) is not str or NONCE_RE.fullmatch(value) is None:
        raise AuthorizationError("launch nonce must encode exactly 32 random bytes")
    if len(bytes.fromhex(value)) != 32:
        raise AuthorizationError("launch nonce decoded length changed")
    if seen is not None:
        if value in seen:
            raise AuthorizationError("launch nonce was reused")
        seen.add(value)
    return value


def _validate_clock_fields(
    *,
    created_epoch_ns: Any,
    deadline_epoch_ns: Any,
    created_monotonic_ns: Any,
    deadline_monotonic_ns: Any,
    now_epoch_ns: int | None = None,
    now_monotonic_ns: int | None = None,
) -> None:
    created = _exact_int(created_epoch_ns, "created_epoch_ns")
    deadline = _exact_int(deadline_epoch_ns, "deadline_epoch_ns")
    created_mono = _exact_int(created_monotonic_ns, "created_monotonic_ns")
    deadline_mono = _exact_int(deadline_monotonic_ns, "deadline_monotonic_ns")
    span = HARD_WALL_SECONDS * 1_000_000_000
    if deadline - created != span or deadline_mono - created_mono != span:
        raise AuthorizationError("r8 deadline span must be exactly 21600 seconds")
    now = time.time_ns() if now_epoch_ns is None else now_epoch_ns
    now_mono = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
    if not (created <= now < deadline):
        raise AuthorizationError("r8 wall-clock freshness failed")
    if now - created > MAXIMUM_LAUNCH_AGE_SECONDS * 1_000_000_000:
        raise AuthorizationError("r8 worker launch age exceeded 300 seconds")
    if not (created_mono <= now_mono < deadline_mono):
        raise AuthorizationError("r8 monotonic freshness failed")


def _process_identity(pid: int) -> dict[str, Any]:
    if os.name != "nt":
        return {
            "pid": pid,
            "creation_filetime": 0,
            "image_path": str(Path(sys.executable).resolve(strict=True)),
            "image_sha256": _sha256(Path(sys.executable)),
        }
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x1000 | 0x00100000, False, pid)
    if not handle:
        raise AuthorizationError("cannot open true parent process")
    try:
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise OSError(ctypes.get_last_error(), "GetProcessTimes failed")
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            raise OSError(ctypes.get_last_error(), "QueryFullProcessImageNameW failed")
        image = Path(buffer.value).resolve(strict=True)
        return {
            "pid": int(pid),
            "creation_filetime": (int(creation.dwHighDateTime) << 32)
            | int(creation.dwLowDateTime),
            "image_path": str(image),
            "image_sha256": _sha256(image),
        }
    finally:
        kernel32.CloseHandle(handle)


def _actual_os_parent_pid() -> int:
    if os.name != "nt":
        return os.getppid()
    from ctypes import wintypes

    class PROCESS_BASIC_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("Reserved1", wintypes.LPVOID),
            ("PebBaseAddress", wintypes.LPVOID),
            ("Reserved2", wintypes.LPVOID * 2),
            ("UniqueProcessId", ctypes.c_size_t),
            ("InheritedFromUniqueProcessId", ctypes.c_size_t),
        ]

    ntdll = ctypes.WinDLL("ntdll")
    info = PROCESS_BASIC_INFORMATION()
    returned = wintypes.ULONG()
    status = ntdll.NtQueryInformationProcess(
        ctypes.windll.kernel32.GetCurrentProcess(),
        0,
        ctypes.byref(info),
        ctypes.sizeof(info),
        ctypes.byref(returned),
    )
    if status != 0:
        raise AuthorizationError("cannot query actual OS parent")
    return int(info.InheritedFromUniqueProcessId)


def _process_identity_from_handle(handle: int) -> dict[str, Any]:
    if os.name != "nt":
        raise AuthorizationError("inherited process-handle proof requires Windows")
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetProcessId.argtypes = [wintypes.HANDLE]
    kernel32.GetProcessId.restype = wintypes.DWORD
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    pid = int(kernel32.GetProcessId(wintypes.HANDLE(handle)))
    if pid <= 0:
        raise AuthorizationError("inherited parent process handle is invalid")
    if kernel32.WaitForSingleObject(wintypes.HANDLE(handle), 0) != 0x00000102:
        raise AuthorizationError("inherited parent process is not live")
    creation = wintypes.FILETIME()
    exit_time = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    if not kernel32.GetProcessTimes(
        wintypes.HANDLE(handle),
        ctypes.byref(creation),
        ctypes.byref(exit_time),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        raise AuthorizationError("cannot verify inherited parent creation time")
    size = wintypes.DWORD(32768)
    buffer = ctypes.create_unicode_buffer(size.value)
    if not kernel32.QueryFullProcessImageNameW(
        wintypes.HANDLE(handle), 0, buffer, ctypes.byref(size)
    ):
        raise AuthorizationError("cannot verify inherited parent image")
    image = Path(buffer.value).resolve(strict=True)
    return {
        "pid": pid,
        "creation_filetime": (int(creation.dwHighDateTime) << 32)
        | int(creation.dwLowDateTime),
        "image_path": str(image),
        "image_sha256": _sha256(image),
    }


def _handle_file_identity(handle: int) -> dict[str, Any]:
    if os.name != "nt":
        raise AuthorizationError("inherited claim-handle proof requires Windows")
    from ctypes import wintypes

    class FILE_ID_128(ctypes.Structure):
        _fields_ = [("Identifier", ctypes.c_ubyte * 16)]

    class FILE_ID_INFO(ctypes.Structure):
        _fields_ = [("VolumeSerialNumber", ctypes.c_ulonglong), ("FileId", FILE_ID_128)]

    class FILE_STANDARD_INFO(ctypes.Structure):
        _fields_ = [
            ("AllocationSize", ctypes.c_longlong),
            ("EndOfFile", ctypes.c_longlong),
            ("NumberOfLinks", wintypes.DWORD),
            ("DeletePending", wintypes.BOOLEAN),
            ("Directory", wintypes.BOOLEAN),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    info = FILE_ID_INFO()
    standard = FILE_STANDARD_INFO()
    if not kernel32.GetFileInformationByHandleEx(
        wintypes.HANDLE(handle), 18, ctypes.byref(info), ctypes.sizeof(info)
    ):
        raise AuthorizationError("cannot verify inherited claim FileId")
    if not kernel32.GetFileInformationByHandleEx(
        wintypes.HANDLE(handle), 1, ctypes.byref(standard), ctypes.sizeof(standard)
    ):
        raise AuthorizationError("cannot verify inherited claim size")
    needed = kernel32.GetFinalPathNameByHandleW(wintypes.HANDLE(handle), None, 0, 0)
    if needed <= 0:
        raise AuthorizationError("cannot size inherited claim final path")
    buffer = ctypes.create_unicode_buffer(needed + 1)
    if not kernel32.GetFinalPathNameByHandleW(
        wintypes.HANDLE(handle), buffer, len(buffer), 0
    ):
        raise AuthorizationError("cannot read inherited claim final path")
    return {
        "file_id": f"{int(info.VolumeSerialNumber)}:{bytes(info.FileId.Identifier).hex()}",
        "bytes": int(standard.EndOfFile),
        "number_of_links": int(standard.NumberOfLinks),
        "delete_pending": bool(standard.DeletePending),
        "directory": bool(standard.Directory),
        "final_path": buffer.value.removeprefix("\\\\?\\"),
    }


def _read_pipe_capability(handle: int) -> tuple[bytes, dict[str, Any]]:
    if os.name != "nt":
        raise AuthorizationError("pipe-only capability requires Windows")
    import msvcrt

    try:
        descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
    except OSError as error:
        raise AuthorizationError("anonymous capability-pipe handle is invalid") from error
    with os.fdopen(descriptor, "rb", closefd=True) as reader:
        raw = reader.read()
    if len(raw) < CAPABILITY_BYTES + 4:
        raise AuthorizationError("parent capability frame is truncated")
    secret = raw[:CAPABILITY_BYTES]
    payload_length = struct.unpack(">I", raw[CAPABILITY_BYTES : CAPABILITY_BYTES + 4])[0]
    payload_raw = raw[CAPABILITY_BYTES + 4 :]
    if len(payload_raw) != payload_length:
        raise AuthorizationError("parent capability frame length changed")
    payload = _json_from_bytes(payload_raw, label="parent pipe capability")
    expected = {
        "schema_version",
        "authorization_sha256",
        "seal_sha256",
        "qa_sha256",
        "claim_sha256",
        "manifest_sha256",
        "launch_nonce",
        "parent_identity",
    }
    _exact_keys(payload, expected, "parent pipe capability")
    for name in (
        "authorization_sha256",
        "seal_sha256",
        "qa_sha256",
        "claim_sha256",
        "manifest_sha256",
    ):
        _exact_digest(payload[name], f"parent capability.{name}")
    _validate_nonce(payload["launch_nonce"])
    return secret, payload


def _capability_frame(secret: bytes, payload: Mapping[str, Any]) -> bytes:
    if type(secret) is not bytes or len(secret) != CAPABILITY_BYTES:
        raise AuthorizationError("parent capability must be exactly 32 bytes")
    raw = _canonical_bytes(payload)
    return secret + struct.pack(">I", len(raw)) + raw


def _manifest_schema(value: Any) -> dict[str, Any]:
    manifest = _exact_keys(
        value,
        {
            "schema_version",
            "manifest_role",
            "experiment_id",
            "canonical_project_root",
            "canonical_artifact_relative_path",
            "single_attempt_namespace",
            "snapshot_root",
            "runtime_root",
            "python_home_root",
            "snapshot_inventory_binding",
            "runtime_inventory_binding",
            "python_home_inventory_binding",
            "authorization_sha256",
            "seal_sha256",
            "qa_sha256",
            "launch_nonce",
            "parent_identity",
            "created_epoch_ns",
            "deadline_epoch_ns",
            "created_monotonic_ns",
            "deadline_monotonic_ns",
        },
        "transport manifest",
    )
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise AuthorizationError("transport manifest schema changed")
    if manifest["manifest_role"] != "TRANSPORT_RECEIPT_NEVER_AUTHORITY":
        raise AuthorizationError("transport manifest attempted authority")
    return manifest


def _claim_schema(value: Any) -> dict[str, Any]:
    claim = _exact_keys(
        value,
        {
            "schema_version",
            "experiment_id",
            "canonical_project_root",
            "canonical_artifact_relative_path",
            "single_attempt_namespace",
            "launch_nonce",
            "capability_commitment_sha256",
            "parent_identity",
            "authorization_sha256",
            "seal_sha256",
            "qa_sha256",
            "manifest_sha256",
            "python_home_inventory_binding",
            "runtime_inventory_binding",
            "snapshot_inventory_binding",
            "created_epoch_ns",
            "deadline_epoch_ns",
            "created_monotonic_ns",
            "deadline_monotonic_ns",
            "replay_rule",
        },
        "canonical launch claim",
    )
    if claim["schema_version"] != CLAIM_SCHEMA_VERSION:
        raise AuthorizationError("canonical claim schema changed")
    if claim["replay_rule"] != "ANY_EXISTING_OR_PARTIAL_CLAIM_PERMANENTLY_BLOCKS_REPLAY":
        raise AuthorizationError("canonical claim replay rule changed")
    _validate_nonce(claim["launch_nonce"])
    _exact_digest(claim["capability_commitment_sha256"], "capability commitment")
    _validate_clock_fields(
        created_epoch_ns=claim["created_epoch_ns"],
        deadline_epoch_ns=claim["deadline_epoch_ns"],
        created_monotonic_ns=claim["created_monotonic_ns"],
        deadline_monotonic_ns=claim["deadline_monotonic_ns"],
    )
    return claim


def _current_user_sid() -> str:
    if os.name != "nt":
        raise RuntimeError("directory ACL freeze requires Windows")
    system32 = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32"
    result = subprocess.run(
        [str(system32 / "whoami.exe"), "/user", "/fo", "csv", "/nh"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) < 2 or not rows[0][1].startswith("S-1-"):
        raise RuntimeError("cannot resolve current Windows SID")
    return rows[0][1]


class DirectoryAclFreeze:
    """Deny additions/deletes below private load roots for worker lifetime."""

    def __init__(self, roots: Sequence[Path]) -> None:
        self.roots = tuple(path.resolve(strict=True) for path in roots)
        self.sid = ""
        self._frozen: list[Path] = []

    def __enter__(self) -> DirectoryAclFreeze:
        if os.name != "nt":
            return self
        system32 = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32"
        icacls = str(system32 / "icacls.exe")
        self.sid = _current_user_sid()
        try:
            for root in self.roots:
                result = subprocess.run(
                    [
                        icacls,
                        str(root),
                        "/deny",
                        f"*{self.sid}:(OI)(CI)(WD,AD,DC)",
                        "/T",
                        "/C",
                        "/Q",
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
                if result.returncode != 0:
                    raise RuntimeError(f"directory ACL freeze failed: {root.name}")
                self._frozen.append(root)
                self._assert_insertion_denied(root)
            return self
        except BaseException:
            self.close(suppress=True)
            raise

    @staticmethod
    def _assert_insertion_denied(root: Path) -> None:
        probe = root / "__p1_r8_forbidden_insertion_probe__.dll"
        try:
            descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_BINARY, 0o600)
        except PermissionError:
            return
        else:
            os.close(descriptor)
            try:
                probe.unlink()
            except OSError:
                pass
            raise RuntimeError("directory ACL did not deny native-search insertion")

    def close(self, *, suppress: bool = False) -> None:
        errors: list[BaseException] = []
        if os.name == "nt" and self.sid:
            system32 = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32"
            icacls = str(system32 / "icacls.exe")
            while self._frozen:
                root = self._frozen.pop()
                try:
                    result = subprocess.run(
                        [icacls, str(root), "/remove:d", f"*{self.sid}", "/T", "/C", "/Q"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                    )
                    if result.returncode != 0:
                        raise RuntimeError(f"directory ACL restore failed: {root.name}")
                except BaseException as error:
                    errors.append(error)
        if errors and not suppress:
            raise RuntimeError("one or more private directory ACLs failed to restore") from errors[0]

    def __exit__(self, exc_type: Any, exc: Any, traceback_value: Any) -> None:
        del exc_type, exc, traceback_value
        self.close()


class HeldDirectoryHandles:
    """Retain exact directory identities; ACL freeze supplies insertion denial."""

    def __init__(self, roots: Sequence[Path]) -> None:
        paths: set[Path] = set()
        for root in roots:
            resolved_root = root.resolve(strict=True)
            paths.add(resolved_root)
            paths.update(
                value.resolve(strict=True)
                for value in resolved_root.rglob("*")
                if value.is_dir()
            )
        self.paths = tuple(sorted(paths, key=lambda value: str(value).casefold()))
        self._handles: list[int] = []
        self.records: dict[str, str] = {}

    def __enter__(self) -> HeldDirectoryHandles:
        if os.name != "nt":
            return self
        from ctypes import wintypes

        class FILE_ID_128(ctypes.Structure):
            _fields_ = [("Identifier", ctypes.c_ubyte * 16)]

        class FILE_ID_INFO(ctypes.Structure):
            _fields_ = [
                ("VolumeSerialNumber", ctypes.c_ulonglong),
                ("FileId", FILE_ID_128),
            ]

        class FILE_ATTRIBUTE_TAG_INFO(ctypes.Structure):
            _fields_ = [
                ("FileAttributes", wintypes.DWORD),
                ("ReparseTag", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel32.CreateFileW
        create.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create.restype = wintypes.HANDLE
        invalid = wintypes.HANDLE(-1).value
        try:
            for path in self.paths:
                handle = int(
                    create(
                        str(path),
                        0x0001 | 0x0080 | 0x00100000,
                        0x00000001,
                        None,
                        3,
                        0x02000000 | 0x00200000,
                        None,
                    )
                )
                if handle in {0, invalid}:
                    raise OSError(ctypes.get_last_error(), "cannot hold private directory")
                info = FILE_ID_INFO()
                attributes = FILE_ATTRIBUTE_TAG_INFO()
                if not kernel32.GetFileInformationByHandleEx(
                    wintypes.HANDLE(handle), 18, ctypes.byref(info), ctypes.sizeof(info)
                ):
                    kernel32.CloseHandle(handle)
                    raise OSError(ctypes.get_last_error(), "cannot read directory FileId")
                if not kernel32.GetFileInformationByHandleEx(
                    wintypes.HANDLE(handle),
                    9,
                    ctypes.byref(attributes),
                    ctypes.sizeof(attributes),
                ):
                    kernel32.CloseHandle(handle)
                    raise OSError(ctypes.get_last_error(), "cannot read directory attributes")
                if int(attributes.FileAttributes) & 0x00000400:
                    kernel32.CloseHandle(handle)
                    raise RuntimeError("private directory is a reparse point")
                file_id = (
                    f"{int(info.VolumeSerialNumber)}:"
                    f"{bytes(info.FileId.Identifier).hex()}"
                )
                if file_id in self.records.values():
                    kernel32.CloseHandle(handle)
                    raise RuntimeError("private directory FileId alias detected")
                self._handles.append(handle)
                self.records[str(path)] = file_id
            return self
        except BaseException:
            self.close(suppress=True)
            raise

    def close(self, *, suppress: bool = False) -> None:
        errors: list[BaseException] = []
        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            while self._handles:
                handle = self._handles.pop()
                if not kernel32.CloseHandle(handle):
                    errors.append(OSError(ctypes.get_last_error(), "CloseHandle failed"))
        if errors and not suppress:
            raise RuntimeError("private directory handle release failed") from errors[0]

    def __exit__(self, exc_type: Any, exc: Any, traceback_value: Any) -> None:
        del exc_type, exc, traceback_value
        self.close()


def _copy_from_held(guard: Any, source: Path, target: Path, expected: Mapping[str, Any]) -> None:
    raw = guard.read_bytes(source)
    if len(raw) != int(expected["bytes"]) or _sha256_bytes(raw) != expected["sha256"]:
        raise RuntimeError("held source differs before private copy")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_BINARY, 0o500)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_private_bundle(
    source_guard: Any,
    seal_value: Mapping[str, Any],
    bundle_root: Path,
) -> dict[str, Any]:
    python_root = bundle_root / "python_home"
    runtime_root = bundle_root / "isolated_runtime"
    snapshot_root = bundle_root / "project_snapshot"
    python_root.mkdir(parents=True)
    runtime_root.mkdir()
    snapshot_root.mkdir()
    python_inventory = seal_value["full_python_home_inventory"]
    for relative, record in python_inventory["files"].items():
        source = Path(record["source_identity"]["resolved_path"])
        _copy_from_held(source_guard, source, python_root / relative, record)
    runtime = seal_value["full_runtime_inventory"]
    for relative, record in runtime["files"].items():
        source = Path(record["source_identity"]["resolved_path"])
        _copy_from_held(source_guard, source, runtime_root / relative, record)
    snapshot = seal_value["snapshot_static_inventory"]
    snapshot_bindings = seal_value["snapshot_source_bindings"]
    for relative, record in snapshot.items():
        binding = snapshot_bindings[relative]
        source = Path(binding["source_identity"]["resolved_path"])
        _copy_from_held(source_guard, source, snapshot_root / relative, record)
    r7 = _load_r7()
    r7._verify_exact_tree(python_root, python_inventory["files"], label="isolated Python home")
    r7._verify_exact_tree(runtime_root, runtime["files"], label="isolated third-party runtime")
    r7._verify_exact_tree(snapshot_root, snapshot, label="private project snapshot")
    return {
        "python_home_root": python_root,
        "runtime_root": runtime_root,
        "snapshot_root": snapshot_root,
    }


def _mapped_image_paths(pid: int | None = None) -> list[Path]:
    if os.name != "nt":
        return [Path(sys.executable).resolve(strict=True)]
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModulesEx.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.DWORD,
    ]
    psapi.EnumProcessModulesEx.restype = wintypes.BOOL
    psapi.GetModuleFileNameExW.argtypes = [
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    psapi.GetModuleFileNameExW.restype = wintypes.DWORD
    target_pid = os.getpid() if pid is None else pid
    process = kernel32.OpenProcess(0x0400 | 0x0010, False, target_pid)
    if not process:
        raise RuntimeError("cannot open process for module-map validation")
    try:
        capacity = 4096
        modules = (wintypes.HMODULE * capacity)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModulesEx(
            process,
            modules,
            ctypes.sizeof(modules),
            ctypes.byref(needed),
            0x03,
        ):
            raise OSError(ctypes.get_last_error(), "EnumProcessModulesEx failed")
        count = min(capacity, int(needed.value // ctypes.sizeof(wintypes.HMODULE)))
        paths: set[Path] = set()
        for module in modules[:count]:
            buffer = ctypes.create_unicode_buffer(32768)
            length = psapi.GetModuleFileNameExW(process, module, buffer, len(buffer))
            if length:
                try:
                    paths.add(Path(buffer.value).resolve(strict=True))
                except OSError:
                    pass
        return sorted(paths, key=lambda value: str(value).casefold())
    finally:
        kernel32.CloseHandle(process)


@functools.lru_cache(maxsize=1)
def _security_provider_inventory() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in _mapped_image_paths():
        if path.name.casefold() != "mpoav.dll":
            continue
        records[str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "source_identity": _source_identity(path),
            "classification": "OS_SECURITY_PROVIDER_EXACT_LIVE_HELD",
        }
    return dict(sorted(records.items(), key=lambda item: item[0].casefold()))


def _validate_initial_native_map(
    python_root: Path,
    python_inventory: Mapping[str, Any],
    security_providers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    system_root = Path(os.environ.get("SystemRoot", "C:/Windows")).resolve(strict=True)
    system32 = (system_root / "System32").resolve(strict=True)
    allowed = {
        str((python_root / relative).resolve(strict=True)).casefold(): record
        for relative, record in python_inventory["files"].items()
        if Path(relative).suffix.lower() in {".exe", ".dll", ".pyd"}
    }
    providers = {
        str(Path(path).resolve(strict=True)).casefold(): record
        for path, record in (security_providers or {}).items()
    }
    observed: dict[str, Any] = {}
    for path in _mapped_image_paths():
        suffix = path.suffix.lower()
        if suffix not in {".exe", ".dll", ".pyd"}:
            continue
        if path.is_relative_to(system32) or path.is_relative_to(system_root / "WinSxS"):
            observed[str(path)] = {"class": "TRUSTED_WINDOWS_COMPONENT"}
            continue
        record = allowed.get(str(path).casefold())
        if record is None:
            provider = providers.get(str(path).casefold())
            if provider is not None:
                if (
                    path.stat().st_size != int(provider["bytes"])
                    or _sha256(path) != provider["sha256"]
                ):
                    raise AuthorizationError("host security-provider digest changed")
                observed[str(path)] = {
                    "class": "OS_SECURITY_PROVIDER_EXACT_LIVE",
                    "bytes": provider["bytes"],
                    "sha256": provider["sha256"],
                }
                continue
            raise AuthorizationError(f"unsealed initial native image: {path.name}")
        if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise AuthorizationError(f"initial native image digest changed: {path.name}")
        observed[str(path)] = {"class": "ISOLATED_PYTHON_HOME", "sha256": record["sha256"]}
    vc = [path for path in observed if Path(path).name.casefold() == "vcruntime140.dll"]
    if len(vc) != 1:
        raise AuthorizationError("exact VCRUNTIME140.dll is not loaded from isolated home")
    return observed


def _runtime_for_private_python(
    runtime: Mapping[str, Any], python_root: Path, security_providers: Mapping[str, Any]
) -> dict[str, Any]:
    private = dict(runtime)
    host: dict[str, dict[str, Any]] = {}
    for name, original in runtime["host_files"].items():
        if name == "python_executable":
            continue
        if not name.startswith("python_home/"):
            continue
        relative = name.removeprefix("python_home/")
        record = dict(original)
        record["resolved_source_path"] = str((python_root / relative).resolve(strict=True))
        record.pop("source_identity", None)
        host[name] = record
    python_inventory = _python_home_inventory_from_sealed_tree(
        python_root, runtime=None, include_source_identity=False
    )
    for relative in (
        "python.exe",
        "python3.dll",
        f"python{sys.version_info.major}{sys.version_info.minor}.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
    ):
        record = python_inventory["files"].get(relative)
        if record is not None:
            host[f"isolated_python_home/{relative}"] = {
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "native_payload": True,
                "resolved_source_path": str((python_root / relative).resolve(strict=True)),
            }
    for path_text, record in security_providers.items():
        host[f"security_provider/{Path(path_text).name}"] = {
            "bytes": record["bytes"],
            "sha256": record["sha256"],
            "native_payload": True,
            "resolved_source_path": path_text,
        }
    private["host_files"] = host
    return private


def _python_home_inventory_from_sealed_tree(
    root: Path,
    runtime: Mapping[str, Any] | None,
    *,
    include_source_identity: bool,
) -> dict[str, Any]:
    del runtime
    base = root.resolve(strict=True)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(
        (value for value in base.rglob("*") if value.is_file()),
        key=lambda value: str(value).casefold(),
    ):
        relative = path.relative_to(base).as_posix()
        record: dict[str, Any] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if include_source_identity:
            record["source_identity"] = _source_identity(path)
        files[relative] = record
    return {
        "schema_version": PYTHON_HOME_SCHEMA_VERSION,
        "source_root": str(base),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "executable_relative_path": "python.exe",
        "complete_stdlib": True,
        "live_site_packages_included": False,
        "file_count": len(files),
        "total_bytes": sum(int(value["bytes"]) for value in files.values()),
        "files": files,
        "required_vc_runtime": REQUIRED_VC_RUNTIME,
    }


def _assert_worker_bootstrap_flags(python_root: Path) -> None:
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise AuthorizationError("worker must start with exact -I -S -B isolation")
    executable = Path(sys.executable).resolve(strict=True)
    expected = (python_root / "python.exe").resolve(strict=True)
    if os.path.normcase(str(executable)) != os.path.normcase(str(expected)):
        raise AuthorizationError("worker is not running the sealed copied interpreter")
    base_prefix = Path(sys.base_prefix).resolve(strict=True)
    if os.path.normcase(str(base_prefix)) != os.path.normcase(str(python_root.resolve())):
        raise AuthorizationError("worker Python prefix escaped the copied home")
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        if os.environ.get(key):
            raise AuthorizationError(f"forbidden worker environment variable: {key}")


def _worker_self_authorize(args: argparse.Namespace) -> dict[str, Any]:
    if os.name != "nt":
        raise AuthorizationError("r8 authorized worker is Windows-only")
    if min(args.capability_pipe_handle, args.parent_process_handle, args.claim_handle) <= 0:
        raise AuthorizationError("hidden worker inherited-handle set is incomplete")
    secret, payload = _read_pipe_capability(args.capability_pipe_handle)
    parent_from_handle = _process_identity_from_handle(args.parent_process_handle)
    actual_parent_pid = _actual_os_parent_pid()
    if parent_from_handle["pid"] != actual_parent_pid:
        raise AuthorizationError("inherited process handle is not the true OS parent")
    if payload["parent_identity"] != parent_from_handle:
        raise AuthorizationError("pipe capability parent identity changed")

    manifest_path = Path(args.transport_manifest).resolve(strict=True)
    manifest_raw = manifest_path.read_bytes()
    if _sha256_bytes(manifest_raw) != payload["manifest_sha256"]:
        raise AuthorizationError("transport manifest differs from parent pipe")
    manifest = _manifest_schema(
        _json_from_bytes(manifest_raw, label="private transport manifest")
    )
    roots = {
        "python": Path(manifest["python_home_root"]).resolve(strict=True),
        "runtime": Path(manifest["runtime_root"]).resolve(strict=True),
        "snapshot": Path(manifest["snapshot_root"]).resolve(strict=True),
    }
    bundle_parent = roots["python"].parent
    if any(path.parent != bundle_parent for path in roots.values()):
        raise AuthorizationError("transport roots do not share one private bundle")
    _assert_worker_bootstrap_flags(roots["python"])
    bootstrap = Path(__file__).resolve(strict=True)
    expected_bootstrap = (
        roots["snapshot"] / "scripts/run_p1_long_event_segment_proposal_rescore_v8.py"
    ).resolve(strict=True)
    if os.path.normcase(str(bootstrap)) != os.path.normcase(str(expected_bootstrap)):
        raise AuthorizationError("copied/direct worker bootstrap path changed")

    r7_path = roots["snapshot"] / _project_relative(R7_RUNNER_PATH)
    r7 = _load_r7(r7_path)
    stack = ExitStack()
    try:
        authority_guard = stack.enter_context(
            r7.HeldReadDenyMutation(
                [R8_AUTHORIZATION_PATH, R8_QA_PATH, SEAL_PATH, LAUNCH_CLAIM_PATH]
            )
        )
        auth_raw = authority_guard.read_bytes(R8_AUTHORIZATION_PATH)
        qa_raw = authority_guard.read_bytes(R8_QA_PATH)
        seal_raw = authority_guard.read_bytes(SEAL_PATH)
        claim_raw = authority_guard.read_bytes(LAUNCH_CLAIM_PATH)
        observed_digests = {
            "authorization_sha256": _sha256_bytes(auth_raw),
            "qa_sha256": _sha256_bytes(qa_raw),
            "seal_sha256": _sha256_bytes(seal_raw),
            "claim_sha256": _sha256_bytes(claim_raw),
        }
        for key, digest in observed_digests.items():
            if payload[key] != digest:
                raise AuthorizationError(f"canonical live {key} differs from parent pipe")
        canonical_claim_record = authority_guard.records[
            str(LAUNCH_CLAIM_PATH.resolve(strict=True))
        ]
        inherited_claim_record = _handle_file_identity(args.claim_handle)
        if (
            inherited_claim_record["file_id"] != canonical_claim_record["file_id"]
            or os.path.normcase(inherited_claim_record["final_path"])
            != os.path.normcase(str(LAUNCH_CLAIM_PATH.resolve(strict=True)))
            or inherited_claim_record["bytes"] != len(claim_raw)
        ):
            raise AuthorizationError("inherited claim handle is not canonical live claim")

        seal_value = _json_from_bytes(seal_raw, label="canonical live r8 seal")
        if seal_value.get("schema_version") != SEAL_SCHEMA_VERSION:
            raise AuthorizationError("canonical live seal schema changed")
        authorization = _require_authorization_schema(
            _json_from_bytes(auth_raw, label="canonical live r8 authorization")
        )
        qa_value = _json_from_bytes(qa_raw, label="canonical live r8 QA")
        claim = _claim_schema(
            _json_from_bytes(claim_raw, label="canonical live r8 claim")
        )
        _validate_live_authorization(
            authorization,
            seal_value,
            observed_digests["seal_sha256"],
            qa_value,
            observed_digests["qa_sha256"],
        )
        if _sha256_bytes(secret) != claim["capability_commitment_sha256"]:
            raise AuthorizationError("pipe-only parent capability commitment changed")
        canonical_literals = {
            "experiment_id": EXPERIMENT_ID,
            "canonical_project_root": CANONICAL_PROJECT_ROOT_TEXT,
            "canonical_artifact_relative_path": CANONICAL_ARTIFACT_RELATIVE,
            "single_attempt_namespace": SINGLE_ATTEMPT_NAMESPACE,
        }
        for key, expected in canonical_literals.items():
            if claim.get(key) != expected or manifest.get(key) != expected:
                raise AuthorizationError(f"canonical worker literal changed: {key}")
        for key in (
            "launch_nonce",
            "parent_identity",
            "created_epoch_ns",
            "deadline_epoch_ns",
            "created_monotonic_ns",
            "deadline_monotonic_ns",
            "authorization_sha256",
            "seal_sha256",
            "qa_sha256",
            "python_home_inventory_binding",
            "runtime_inventory_binding",
            "snapshot_inventory_binding",
        ):
            if claim.get(key) != manifest.get(key):
                raise AuthorizationError(f"claim/manifest binding changed: {key}")
        if claim["parent_identity"] != parent_from_handle:
            raise AuthorizationError("canonical claim is not bound to true parent")
        if claim["manifest_sha256"] != payload["manifest_sha256"]:
            raise AuthorizationError("canonical claim manifest binding changed")
        if claim["launch_nonce"] != payload["launch_nonce"]:
            raise AuthorizationError("canonical claim nonce differs from pipe")
        for key in (
            "python_home_inventory_binding",
            "runtime_inventory_binding",
            "snapshot_inventory_binding",
        ):
            if claim[key] != seal_value[key]:
                raise AuthorizationError(f"canonical claim inventory changed: {key}")

        python_inventory = seal_value["full_python_home_inventory"]
        runtime = seal_value["full_runtime_inventory"]
        snapshot = seal_value["snapshot_static_inventory"]
        snapshot_bindings = seal_value["snapshot_source_bindings"]
        security_providers = seal_value["host_security_provider_inventory"]
        if manifest["python_home_inventory_binding"] != _canonical_record(python_inventory):
            raise AuthorizationError("transport Python-home binding changed")
        if manifest["runtime_inventory_binding"] != _canonical_record(runtime):
            raise AuthorizationError("transport runtime binding changed")
        if manifest["snapshot_inventory_binding"] != _canonical_record(snapshot):
            raise AuthorizationError("transport snapshot binding changed")

        source_paths = {
            Path(record["source_identity"]["resolved_path"])
            for record in python_inventory["files"].values()
        }
        source_paths.update(
            Path(record["source_identity"]["resolved_path"])
            for record in runtime["files"].values()
        )
        source_paths.update(
            Path(binding["source_identity"]["resolved_path"])
            for binding in snapshot_bindings.values()
        )
        source_paths.update(Path(path) for path in security_providers)
        source_guard = stack.enter_context(r7.HeldReadDenyMutation(source_paths))
        destination_paths = {
            *(roots["python"] / relative for relative in python_inventory["files"]),
            *(roots["runtime"] / relative for relative in runtime["files"]),
            *(roots["snapshot"] / relative for relative in snapshot),
            manifest_path,
        }
        destination_guard = stack.enter_context(r7.HeldReadDenyMutation(destination_paths))
        directory_guard = stack.enter_context(HeldDirectoryHandles(list(roots.values())))
        if not directory_guard.records:
            raise AuthorizationError("worker private directory lifetime holds are absent")
        for relative, record in python_inventory["files"].items():
            source_guard.assert_record(
                Path(record["source_identity"]["resolved_path"]),
                record,
                require_source_identity=True,
            )
            destination_guard.assert_record(roots["python"] / relative, record)
        for relative, record in runtime["files"].items():
            source_guard.assert_record(
                Path(record["source_identity"]["resolved_path"]),
                record,
                require_source_identity=True,
            )
            destination_guard.assert_record(roots["runtime"] / relative, record)
        for relative, record in snapshot.items():
            binding = snapshot_bindings[relative]
            expected_source = dict(record, source_identity=binding["source_identity"])
            source_guard.assert_record(
                Path(binding["source_identity"]["resolved_path"]),
                expected_source,
                require_source_identity=True,
            )
            destination_guard.assert_record(roots["snapshot"] / relative, record)
        r7._verify_exact_tree(
            roots["python"], python_inventory["files"], label="worker isolated Python home"
        )
        r7._verify_exact_tree(
            roots["runtime"], runtime["files"], label="worker isolated runtime"
        )
        r7._verify_exact_tree(
            roots["snapshot"], snapshot, label="worker private project snapshot"
        )
        for root in roots.values():
            DirectoryAclFreeze._assert_insertion_denied(root)
        initial_map = _validate_initial_native_map(
            roots["python"], python_inventory, security_providers
        )
        _validate_clock_fields(
            created_epoch_ns=claim["created_epoch_ns"],
            deadline_epoch_ns=claim["deadline_epoch_ns"],
            created_monotonic_ns=claim["created_monotonic_ns"],
            deadline_monotonic_ns=claim["deadline_monotonic_ns"],
        )
        return {
            "stack": stack.pop_all(),
            "r7": r7,
            "authorization": authorization,
            "authorization_sha256": observed_digests["authorization_sha256"],
            "qa": qa_value,
            "qa_sha256": observed_digests["qa_sha256"],
            "seal": seal_value,
            "seal_sha256": observed_digests["seal_sha256"],
            "claim": claim,
            "claim_sha256": observed_digests["claim_sha256"],
            "manifest": manifest,
            "manifest_sha256": payload["manifest_sha256"],
            "python_root": roots["python"],
            "runtime_root": roots["runtime"],
            "snapshot_root": roots["snapshot"],
            "python_inventory": python_inventory,
            "runtime": runtime,
            "snapshot_inventory": snapshot,
            "security_providers": security_providers,
            "initial_native_map": initial_map,
        }
    except BaseException:
        stack.close()
        raise


def _configure_r7_for_r8(r7: ModuleType) -> None:
    r7.EXPERIMENT_ID = EXPERIMENT_ID
    r7.SCIENTIFIC_EXPERIMENT_ID = SCIENTIFIC_EXPERIMENT_ID
    r7.CANONICAL_PROJECT_ROOT = CANONICAL_PROJECT_ROOT
    r7.CANONICAL_PROJECT_ROOT_TEXT = CANONICAL_PROJECT_ROOT_TEXT
    r7.CANONICAL_ARTIFACT_RELATIVE = CANONICAL_ARTIFACT_RELATIVE
    r7.CANONICAL_ARTIFACT_DIR = CANONICAL_ARTIFACT_DIR
    r7.SEAL_PATH = SEAL_PATH
    r7.LAUNCH_CLAIM_PATH = LAUNCH_CLAIM_PATH
    r7.SINGLE_ATTEMPT_NAMESPACE = SINGLE_ATTEMPT_NAMESPACE


def _load_science_r8(
    authority: Mapping[str, Any],
) -> tuple[ModuleType, ModuleType, ModuleType, dict[str, Any], dict[str, Any]]:
    r7 = authority["r7"]
    snapshot_root = authority["snapshot_root"]
    r6_path = snapshot_root / r7._project_relative_literal(r7.R6_RUNNER_PATH)
    r6 = r7._load_r6_runner(r6_path)
    r6.EXPERIMENT_ID = EXPERIMENT_ID
    r6.OUTPUT_PROJECT_ROOT = CANONICAL_PROJECT_ROOT
    r6.ARTIFACT_DIR = CANONICAL_ARTIFACT_DIR
    r6.SEAL_PATH = SEAL_PATH
    legacy_raw, _receipt = r7._read_bound_bytes(
        r6.LEGACY_CONFIG_PATH,
        expected_sha256=r6.LEGACY_CONFIG_SHA256,
    )
    legacy_config = r7._json_from_bytes(legacy_raw, label=r6.LEGACY_CONFIG_PATH.name)
    legacy = r6._load_snapshot_legacy_runner(snapshot_root, "_p1_v8_legacy")
    r6_execution_raw, _receipt = r7._read_bound_bytes(
        snapshot_root / r7._project_relative_literal(r7.R6_EXECUTION_MODULE_PATH),
        expected_sha256=R6_EXECUTION_MODULE_SHA256,
    )
    numerical, _frozen_execution, runtime_receipt = r6._load_snapshot_numerical(
        snapshot_root,
        legacy_config,
        legacy,
        r6_execution_raw,
    )
    readiness, state = r6._strict_target_free_snapshot_readiness(
        snapshot_root,
        legacy_config,
        numerical,
        runtime_receipt,
        legacy,
    )
    selected = r6._selected_readiness(readiness)
    selected_science = r7._selected_science_readiness(selected)
    expected = authority["seal"]["science_preflight"][
        "selected_scientific_readiness_sha256"
    ]
    if _sha256_bytes(_canonical_bytes(selected_science)) != expected:
        raise RuntimeError("worker scientific readiness differs from r8 seal")
    wrapper = __import__(
        "p1_qc.long_event_segment_proposal_rescore_execution_v8",
        fromlist=["run_authorized_screen"],
    )
    if wrapper.R8_AMENDMENT_SHA256 != R8_AMENDMENT_SHA256:
        raise RuntimeError("r8 execution wrapper amendment anchor changed")
    return r6, wrapper, numerical, selected, state


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _render_report(result: Mapping[str, Any]) -> bytes:
    metrics = result["metrics"]
    pooled = metrics["pooled"]
    bootstrap = metrics["paired_bootstrap"]
    selected = result["selected_inner_cell"]
    lines = [
        "# P1 장기 이벤트 구간 제안·재채점 고정 실험 (r8 infrastructure)",
        "",
        f"결론: **{result['decision']}**",
        "",
        "공식 평가·제출이 아닌 사전등록 historical screen 집계다.",
        "행별 예측·제출 후보·업로드는 생성하지 않았다.",
        "",
        f"- 후보 F1: {pooled['candidate']['f1']:.9f}",
        f"- Round-B anchor F1: {pooled['anchor']['f1']:.9f}",
        f"- 후보−anchor F1 Δ: {pooled['f1_delta']:+.9f}",
        f"- paired bootstrap 90% CI: {bootstrap['difference_ci90']}",
        f"- 선택 구조: {selected['cell_id']}, threshold={selected['threshold']}",
        f"- RESEARCH_GO: {result['RESEARCH_GO']}",
        f"- SUBMISSION_GO_RESEARCH_ONLY: {result['SUBMISSION_GO_RESEARCH_ONLY']}",
        "",
        "고정 계수: 9 anchor + 54 inner segment + 9 outer segment = 72 fits; "
        "21 scientific materializations; 1 outer score.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _publish_success_r8(
    screen: Mapping[str, Any], journal: Any, authority: Mapping[str, Any], r6: ModuleType
) -> Path:
    r7 = authority["r7"]
    metrics_path = CANONICAL_ARTIFACT_DIR / "metrics.json"
    report_path = CANONICAL_ARTIFACT_DIR / "report_ko.md"
    result_path = CANONICAL_ARTIFACT_DIR / "result.json"
    manifest_path = CANONICAL_ARTIFACT_DIR / "manifest.json"
    result = dict(screen)
    result.update(
        {
            "attempt_id": journal.attempt_id,
            "authorization_sha256": authority["authorization_sha256"],
            "preexecution_seal_sha256": authority["seal_sha256"],
            "independent_qa_sha256": authority["qa_sha256"],
            "transport_manifest_sha256": authority["manifest_sha256"],
            "canonical_launch_claim_sha256": authority["claim_sha256"],
            "python_home_inventory_binding": authority["seal"][
                "python_home_inventory_binding"
            ],
            "runtime_inventory_binding": authority["seal"]["runtime_inventory_binding"],
            "snapshot_inventory_binding": authority["seal"]["snapshot_inventory_binding"],
            "completed_at_kst": _now_kst_after_tzdata_activation(),
        }
    )
    journal.record_aggregate(result)
    r7._atomic_create_json(metrics_path, result["metrics"])
    r7._atomic_create_bytes(report_path, _render_report(result))
    r7._atomic_create_json(result_path, result)
    inventory = {
        _project_relative(path): _artifact_record(path)
        for path in (metrics_path, report_path, result_path, SEAL_PATH, LAUNCH_CLAIM_PATH)
    }
    inventory.update(journal.manifest_records())
    manifest = {
        "schema_version": "p1_long_event_segment_proposal_rescore.manifest.v8",
        "experiment_id": EXPERIMENT_ID,
        "scientific_experiment_id": SCIENTIFIC_EXPERIMENT_ID,
        "attempt_id": journal.attempt_id,
        "status": "WORKER_OUTPUTS_COMPLETE_BEFORE_FINAL_COMMIT",
        "authorization_sha256": authority["authorization_sha256"],
        "preexecution_seal_sha256": authority["seal_sha256"],
        "independent_qa_sha256": authority["qa_sha256"],
        "transport_manifest_sha256": authority["manifest_sha256"],
        "canonical_launch_claim_sha256": authority["claim_sha256"],
        "python_home_inventory_binding": authority["seal"][
            "python_home_inventory_binding"
        ],
        "runtime_inventory_binding": authority["seal"]["runtime_inventory_binding"],
        "snapshot_inventory_binding": authority["seal"]["snapshot_inventory_binding"],
        "artifacts": inventory,
        "operation_counters": result["operation_counters"],
        "forbidden_outputs": {
            "row_level_prediction_files": 0,
            "submission_candidate_files": 0,
            "official_test_reads": 0,
            "sample_format_reads": 0,
            "submission_candidate_reads": 0,
            "uploads": 0,
        },
        "created_at_kst": _now_kst_after_tzdata_activation(),
    }
    r7._atomic_create_json(manifest_path, manifest)
    for relative, record in inventory.items():
        if _artifact_record(CANONICAL_PROJECT_ROOT / relative) != record:
            raise RuntimeError(f"r8 worker manifest verification failed: {relative}")
    journal.terminal_entry(
        "0998_worker_terminal.json",
        {
            "schema_version": "p1_segment_rescore.worker_terminal.v8",
            "status": "WORKER_SUCCESS_COMMIT_PREPARED",
            "result_sha256": _sha256(result_path),
            "manifest_sha256": _sha256(manifest_path),
            "canonical_launch_claim_sha256": authority["claim_sha256"],
            "physical_fit_reservations": journal.fit_reservations,
            "physical_fits_completed": journal.fits_completed,
            "scientific_materializations": journal.materializations,
            "outer_scores": 1,
            "created_at_kst": _now_kst_after_tzdata_activation(),
        },
    )
    previous, _entries = r6._verify_journal_chain(
        journal.journal_dir, required_last="0998_worker_terminal.json"
    )
    journal.terminal_entry(
        "0999_completed.json",
        {
            "schema_version": "p1_segment_rescore.completed.v8",
            "status": "SUCCESS_ALL_OUTPUTS_VERIFIED_READY_FOR_LOCK_RELEASE",
            "result_sha256": _sha256(result_path),
            "manifest_sha256": _sha256(manifest_path),
            "worker_terminal_sha256": previous,
            "physical_fit_reservations": 72,
            "physical_fits_completed": 72,
            "scientific_materializations": 21,
            "outer_scores": 1,
            "created_at_kst": _now_kst_after_tzdata_activation(),
        },
    )
    r6._final_success_commit(journal)
    return result_path


def _worker_execute(args: argparse.Namespace) -> Path:
    authority = _worker_self_authorize(args)
    r7 = authority["r7"]
    _configure_r7_for_r8(r7)
    runtime_private = _runtime_for_private_python(
        authority["runtime"], authority["python_root"], authority["security_providers"]
    )
    authority["runtime"] = runtime_private
    guard, dll_handles = r7._activate_isolated_runtime(
        authority["snapshot_root"],
        authority["snapshot_inventory"],
        authority["runtime_root"],
        runtime_private,
    )
    r7._install_runtime_audit_hook(authority["runtime_root"], runtime_private)
    journal: Any | None = None
    phase = "R8_RUNTIME_ACTIVATION"
    try:
        r7._eager_runtime_imports(runtime_private, authority["runtime_root"])
        phase = "R8_POSTIMPORT_TRUST"
        r7._trust_checkpoint(authority, guard, "POSTIMPORT")
        _now_kst_after_tzdata_activation()
        allowed_before_lease = {SEAL_PATH.name, LAUNCH_CLAIM_PATH.name}
        if {path.name for path in CANONICAL_ARTIFACT_DIR.iterdir()} != allowed_before_lease:
            raise FileExistsError("r8 namespace changed before worker lease")
        _atomic_create_json(
            WORKER_LEASE_PATH,
            {
                "schema_version": "p1_segment_rescore.worker_start_lease.v8",
                "status": "TRUE_PARENT_CAPABILITY_AND_ALL_LIFETIME_HOLDS_VERIFIED",
                "worker_pid": os.getpid(),
                "true_parent_identity": authority["claim"]["parent_identity"],
                "launch_nonce": authority["claim"]["launch_nonce"],
                "claim_sha256": authority["claim_sha256"],
                "initial_native_map": authority["initial_native_map"],
                "created_at_kst": _now_kst_after_tzdata_activation(),
            },
        )
        phase = "R8_LOAD_FROZEN_SCIENCE"
        r6, wrapper, numerical, readiness, state = _load_science_r8(authority)
        phase = "R8_PRECLAIM_TRUST"
        r7._trust_checkpoint(authority, guard, "PRECLAIM")
        expected = {SEAL_PATH.name, LAUNCH_CLAIM_PATH.name, WORKER_LEASE_PATH.name}
        if {path.name for path in CANONICAL_ARTIFACT_DIR.iterdir()} != expected:
            raise FileExistsError("r8 canonical namespace changed before scientific claim")
        deadline_epoch = authority["claim"]["deadline_epoch_ns"] / 1_000_000_000
        journal = r6.AttemptJournal.begin(
            CANONICAL_ARTIFACT_DIR,
            deadline_epoch,
            snapshot_manifest_sha256=authority["manifest_sha256"],
        )
        journal.record_readiness(readiness)

        def checkpoint(name: str) -> None:
            _validate_clock_fields(
                created_epoch_ns=authority["claim"]["created_epoch_ns"],
                deadline_epoch_ns=authority["claim"]["deadline_epoch_ns"],
                created_monotonic_ns=authority["claim"]["created_monotonic_ns"],
                deadline_monotonic_ns=authority["claim"]["deadline_monotonic_ns"],
            )
            r7._trust_checkpoint(authority, guard, name)

        guarded_journal = r7.FirstFitTrustJournal(journal, checkpoint)
        closure_raw, _receipt = r7._read_bound_bytes(
            r6.CLOSURE_V3_PATH, expected_sha256=r6.CLOSURE_V3_SHA256
        )
        closure = r7._json_from_bytes(closure_raw, label=r6.CLOSURE_V3_PATH.name)
        phase = "R8_FIXED_72_FIT_21_MATERIALIZATION_SCREEN"
        screen = wrapper.run_authorized_screen(
            state, numerical, closure, guarded_journal, deadline_epoch
        )
        phase = "R8_SUCCESS_PUBLICATION"
        return _publish_success_r8(screen, journal, authority, r6)
    except BaseException as error:
        if journal is not None and journal.lock_path.exists():
            try:
                journal.fail_terminal(
                    phase,
                    error,
                    provenance={
                        "snapshot": "WORKER_HELD_PRIVATE_TREE",
                        "runtime": "WORKER_HELD_ISOLATED_TREE",
                        "python_home": "WORKER_HELD_ISOLATED_PYTHON_HOME",
                        "claim_sha256": authority["claim_sha256"],
                    },
                )
            except BaseException as terminal_error:
                journal.close_handle_keep_lock()
                raise RuntimeError(
                    "r8 claimed failure terminal publication failed; lock retained"
                ) from terminal_error
        raise
    finally:
        for handle in reversed(dll_handles):
            try:
                handle.close()
            except BaseException:
                pass
        authority["stack"].close()


def _set_handle_inheritable(handle: int, enabled: bool) -> None:
    if os.name != "nt":
        raise RuntimeError("r8 handle inheritance requires Windows")
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    flags = 0x00000001 if enabled else 0
    if not kernel32.SetHandleInformation(
        wintypes.HANDLE(handle), 0x00000001, flags
    ):
        raise OSError(ctypes.get_last_error(), "SetHandleInformation failed")


def _open_self_process_handle() -> int:
    if os.name != "nt":
        raise RuntimeError("r8 parent process handle requires Windows")
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = int(kernel32.OpenProcess(0x1000 | 0x00100000, False, os.getpid()))
    if not handle:
        raise OSError(ctypes.get_last_error(), "OpenProcess(self) failed")
    return handle


def _native_handle_from_guard(guard: Any, path: Path) -> int:
    import msvcrt

    reader = guard._readers[str(path.resolve(strict=True))]
    return int(msvcrt.get_osfhandle(reader.fileno()))


def _manifest_value(
    *,
    roots: Mapping[str, Path],
    seal_value: Mapping[str, Any],
    authorization_sha256: str,
    seal_sha256: str,
    qa_sha256: str,
    parent_identity: Mapping[str, Any],
    launch_nonce: str,
    created_epoch_ns: int,
    deadline_epoch_ns: int,
    created_monotonic_ns: int,
    deadline_monotonic_ns: int,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_role": "TRANSPORT_RECEIPT_NEVER_AUTHORITY",
        "experiment_id": EXPERIMENT_ID,
        "canonical_project_root": CANONICAL_PROJECT_ROOT_TEXT,
        "canonical_artifact_relative_path": CANONICAL_ARTIFACT_RELATIVE,
        "single_attempt_namespace": SINGLE_ATTEMPT_NAMESPACE,
        "snapshot_root": str(roots["snapshot_root"].resolve(strict=True)),
        "runtime_root": str(roots["runtime_root"].resolve(strict=True)),
        "python_home_root": str(roots["python_home_root"].resolve(strict=True)),
        "snapshot_inventory_binding": seal_value["snapshot_inventory_binding"],
        "runtime_inventory_binding": seal_value["runtime_inventory_binding"],
        "python_home_inventory_binding": seal_value["python_home_inventory_binding"],
        "authorization_sha256": authorization_sha256,
        "seal_sha256": seal_sha256,
        "qa_sha256": qa_sha256,
        "launch_nonce": launch_nonce,
        "parent_identity": dict(parent_identity),
        "created_epoch_ns": created_epoch_ns,
        "deadline_epoch_ns": deadline_epoch_ns,
        "created_monotonic_ns": created_monotonic_ns,
        "deadline_monotonic_ns": deadline_monotonic_ns,
    }


def _claim_value(
    manifest: Mapping[str, Any], manifest_sha256: str, secret: bytes
) -> dict[str, Any]:
    return {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "canonical_project_root": CANONICAL_PROJECT_ROOT_TEXT,
        "canonical_artifact_relative_path": CANONICAL_ARTIFACT_RELATIVE,
        "single_attempt_namespace": SINGLE_ATTEMPT_NAMESPACE,
        "launch_nonce": manifest["launch_nonce"],
        "capability_commitment_sha256": _sha256_bytes(secret),
        "parent_identity": manifest["parent_identity"],
        "authorization_sha256": manifest["authorization_sha256"],
        "seal_sha256": manifest["seal_sha256"],
        "qa_sha256": manifest["qa_sha256"],
        "manifest_sha256": manifest_sha256,
        "python_home_inventory_binding": manifest["python_home_inventory_binding"],
        "runtime_inventory_binding": manifest["runtime_inventory_binding"],
        "snapshot_inventory_binding": manifest["snapshot_inventory_binding"],
        "created_epoch_ns": manifest["created_epoch_ns"],
        "deadline_epoch_ns": manifest["deadline_epoch_ns"],
        "created_monotonic_ns": manifest["created_monotonic_ns"],
        "deadline_monotonic_ns": manifest["deadline_monotonic_ns"],
        "replay_rule": "ANY_EXISTING_OR_PARTIAL_CLAIM_PERMANENTLY_BLOCKS_REPLAY",
    }


def _worker_command(
    roots: Mapping[str, Path],
    manifest_path: Path,
    capability_pipe_handle: int,
    parent_process_handle: int,
    claim_handle: int,
) -> list[str]:
    return [
        str((roots["python_home_root"] / "python.exe").resolve(strict=True)),
        "-I",
        "-S",
        "-B",
        str(
            (
                roots["snapshot_root"]
                / "scripts/run_p1_long_event_segment_proposal_rescore_v8.py"
            ).resolve(strict=True)
        ),
        "--worker",
        "--transport-manifest",
        str(manifest_path.resolve(strict=True)),
        "--capability-pipe-handle",
        str(capability_pipe_handle),
        "--parent-process-handle",
        str(parent_process_handle),
        "--claim-handle",
        str(claim_handle),
    ]


def _clean_worker_environment() -> dict[str, str]:
    forbidden_prefixes = ("PYTHON",)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(forbidden_prefixes)
    }
    data_dir = os.environ.get("P1_DATA_DIR")
    if not data_dir:
        raise AuthorizationError("P1_DATA_DIR must be set before canonical claim")
    expected_data = (
        CANONICAL_PROJECT_ROOT
        / "데이터셋 원본/데이터셋_P1/P1_qc_anomaly"
    ).resolve(strict=True)
    actual_data = Path(data_dir).resolve(strict=True)
    if os.path.normcase(str(actual_data)) != os.path.normcase(str(expected_data)):
        raise AuthorizationError("P1_DATA_DIR canonical path changed")
    environment["P1_DATA_DIR"] = str(actual_data)
    return environment


def _terminate_tree_and_verify(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    system32 = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32"
    result = subprocess.run(
        [str(system32 / "taskkill.exe"), "/PID", str(process.pid), "/T", "/F"],
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 128}:
        raise RuntimeError(f"taskkill failed with rc={result.returncode}")
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("worker tree termination could not be proved") from error
    if process.poll() is None:
        raise RuntimeError("worker remained live after taskkill")


def _supervise_worker(
    process: subprocess.Popen[str], deadline_monotonic_ns: int
) -> tuple[str, str]:
    remaining = (deadline_monotonic_ns - time.monotonic_ns()) / 1_000_000_000
    if remaining <= 0:
        _terminate_tree_and_verify(process)
        raise TimeoutError("r8 monotonic deadline expired before supervision")
    try:
        stdout, stderr = process.communicate(timeout=remaining)
    except subprocess.TimeoutExpired as error:
        _terminate_tree_and_verify(process)
        raise TimeoutError("r8 fixed 21600-second worker timeout") from error
    if process.returncode != 0:
        raise RuntimeError(f"r8 isolated worker exited {process.returncode}: {stderr[-4000:]}")
    return stdout, stderr


def _execute_parent_started(
    execute_start_epoch_ns: int, execute_start_monotonic_ns: int
) -> tuple[Path, dict[str, Any]]:
    if os.name != "nt":
        raise AuthorizationError("r8 execution is Windows-only")
    expected_auth_sha = os.environ.get(AUTHORIZATION_ENV_VAR, "")
    _exact_digest(expected_auth_sha, AUTHORIZATION_ENV_VAR)
    if LAUNCH_CLAIM_PATH.exists():
        raise FileExistsError("existing or partial r8 claim permanently blocks replay")
    seal_value, seal_raw, seal_sha = _read_seal()
    _verify_seal_against_live(seal_value)
    r7 = _load_r7()
    with r7.HeldReadDenyMutation([R8_AUTHORIZATION_PATH, R8_QA_PATH, SEAL_PATH]) as live_guard:
        auth_raw = live_guard.read_bytes(R8_AUTHORIZATION_PATH)
        qa_raw = live_guard.read_bytes(R8_QA_PATH)
        held_seal_raw = live_guard.read_bytes(SEAL_PATH)
        if held_seal_raw != seal_raw:
            raise AuthorizationError("held canonical seal differs from verified seal")
        authorization_sha = _sha256_bytes(auth_raw)
        qa_sha = _sha256_bytes(qa_raw)
        if authorization_sha != expected_auth_sha:
            raise AuthorizationError("external authorization digest changed")
        authorization = _require_authorization_schema(
            _json_from_bytes(auth_raw, label="canonical r8 authorization")
        )
        qa_value = _json_from_bytes(qa_raw, label="canonical r8 QA")
        _validate_live_authorization(
            authorization, seal_value, seal_sha, qa_value, qa_sha
        )
        allowed = {SEAL_PATH.name}
        if {path.name for path in CANONICAL_ARTIFACT_DIR.iterdir()} != allowed:
            raise FileExistsError("r8 namespace is not seal-only before execution")

        python_inventory = seal_value["full_python_home_inventory"]
        runtime = seal_value["full_runtime_inventory"]
        snapshot = seal_value["snapshot_static_inventory"]
        bindings = seal_value["snapshot_source_bindings"]
        security_providers = seal_value["host_security_provider_inventory"]
        source_paths = {
            Path(record["source_identity"]["resolved_path"])
            for record in python_inventory["files"].values()
        }
        source_paths.update(
            Path(record["source_identity"]["resolved_path"])
            for record in runtime["files"].values()
        )
        source_paths.update(
            Path(binding["source_identity"]["resolved_path"])
            for binding in bindings.values()
        )
        source_paths.update(Path(path) for path in security_providers)
        with r7.HeldReadDenyMutation(source_paths) as source_guard:
            with tempfile.TemporaryDirectory(prefix="p1-r8-private-") as temporary:
                bundle = Path(temporary).resolve(strict=True)
                roots = _copy_private_bundle(source_guard, seal_value, bundle)
                parent_identity = _process_identity(os.getpid())
                secret = secrets.token_bytes(CAPABILITY_BYTES)
                launch_nonce = secrets.token_hex(32)
                _validate_nonce(launch_nonce)
                deadline_epoch_ns = execute_start_epoch_ns + HARD_WALL_SECONDS * 1_000_000_000
                deadline_monotonic_ns = (
                    execute_start_monotonic_ns + HARD_WALL_SECONDS * 1_000_000_000
                )
                _validate_clock_fields(
                    created_epoch_ns=execute_start_epoch_ns,
                    deadline_epoch_ns=deadline_epoch_ns,
                    created_monotonic_ns=execute_start_monotonic_ns,
                    deadline_monotonic_ns=deadline_monotonic_ns,
                )
                manifest = _manifest_value(
                    roots=roots,
                    seal_value=seal_value,
                    authorization_sha256=authorization_sha,
                    seal_sha256=seal_sha,
                    qa_sha256=qa_sha,
                    parent_identity=parent_identity,
                    launch_nonce=launch_nonce,
                    created_epoch_ns=execute_start_epoch_ns,
                    deadline_epoch_ns=deadline_epoch_ns,
                    created_monotonic_ns=execute_start_monotonic_ns,
                    deadline_monotonic_ns=deadline_monotonic_ns,
                )
                manifest_path = bundle / "transport_manifest.json"
                _atomic_create_json(manifest_path, manifest)
                manifest_sha = _sha256(manifest_path)
                destination_paths = {
                    *(roots["python_home_root"] / relative for relative in python_inventory["files"]),
                    *(roots["runtime_root"] / relative for relative in runtime["files"]),
                    *(roots["snapshot_root"] / relative for relative in snapshot),
                    manifest_path,
                }
                with r7.HeldReadDenyMutation(destination_paths) as destination_guard:
                    if source_guard.file_ids().intersection(destination_guard.file_ids()):
                        raise RuntimeError("private bundle contains a source hardlink")
                    with HeldDirectoryHandles(list(roots.values())) as directory_guard:
                        if not directory_guard.records:
                            raise RuntimeError("parent private directory holds are absent")
                        with DirectoryAclFreeze(
                            [
                                roots["python_home_root"],
                                roots["runtime_root"],
                                roots["snapshot_root"],
                            ]
                        ):
                            claim = _claim_value(manifest, manifest_sha, secret)
                            _atomic_create_json(LAUNCH_CLAIM_PATH, claim)
                            claim_sha = _sha256(LAUNCH_CLAIM_PATH)
                            with r7.HeldReadDenyMutation(
                                [LAUNCH_CLAIM_PATH]
                            ) as claim_guard:
                                claim_handle = _native_handle_from_guard(
                                    claim_guard, LAUNCH_CLAIM_PATH
                                )
                                parent_handle = _open_self_process_handle()
                                read_fd, write_fd = os.pipe()
                                import msvcrt

                                pipe_handle = int(msvcrt.get_osfhandle(read_fd))
                                inherited = [pipe_handle, parent_handle, claim_handle]
                                for handle in inherited:
                                    _set_handle_inheritable(handle, True)
                                startup = subprocess.STARTUPINFO()
                                startup.lpAttributeList = {"handle_list": inherited}
                                command = _worker_command(
                                    roots,
                                    manifest_path,
                                    pipe_handle,
                                    parent_handle,
                                    claim_handle,
                                )
                                environment = _clean_worker_environment()
                                process: subprocess.Popen[str] | None = None
                                try:
                                    process = subprocess.Popen(
                                        command,
                                        cwd=roots["python_home_root"],
                                        env=environment,
                                        stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        encoding="utf-8",
                                        close_fds=True,
                                        startupinfo=startup,
                                        creationflags=subprocess.CREATE_NO_WINDOW,
                                    )
                                finally:
                                    for handle in inherited:
                                        _set_handle_inheritable(handle, False)
                                    os.close(read_fd)
                                if process is None:
                                    os.close(write_fd)
                                    ctypes.windll.kernel32.CloseHandle(parent_handle)
                                    raise RuntimeError("r8 worker process was not created")
                                payload = {
                                    "schema_version": (
                                        "p1_segment_rescore.parent_pipe_capability.v8"
                                    ),
                                    "authorization_sha256": authorization_sha,
                                    "seal_sha256": seal_sha,
                                    "qa_sha256": qa_sha,
                                    "claim_sha256": claim_sha,
                                    "manifest_sha256": manifest_sha,
                                    "launch_nonce": launch_nonce,
                                    "parent_identity": parent_identity,
                                }
                                frame = _capability_frame(secret, payload)
                                secret = b"\x00" * CAPABILITY_BYTES
                                try:
                                    os.write(write_fd, frame)
                                finally:
                                    os.close(write_fd)
                                    ctypes.windll.kernel32.CloseHandle(parent_handle)
                                stdout, _stderr = _supervise_worker(
                                    process, deadline_monotonic_ns
                                )
                                lines = [
                                    line for line in stdout.splitlines() if line.strip()
                                ]
                                if not lines:
                                    raise RuntimeError(
                                        "r8 worker returned no success receipt"
                                    )
                                receipt = _json_from_bytes(
                                    lines[-1].encode("utf-8"),
                                    label="worker success receipt",
                                )
                                if receipt.get("status") != "worker_ok":
                                    raise RuntimeError(
                                        "r8 worker success receipt changed"
                                    )
                                result_path = Path(receipt["result_path"]).resolve(
                                    strict=True
                                )
                                expected_result = (
                                    CANONICAL_ARTIFACT_DIR / "result.json"
                                ).resolve(strict=True)
                                if result_path != expected_result:
                                    raise RuntimeError(
                                        "r8 worker result path is noncanonical"
                                    )
                                result = _json_from_bytes(
                                    result_path.read_bytes(), label="r8 result"
                                )
                                return result_path, result


def execute_parent() -> tuple[Path, dict[str, Any]]:
    execute_start_epoch_ns = time.time_ns()
    execute_start_monotonic_ns = time.monotonic_ns()
    return _execute_parent_started(execute_start_epoch_ns, execute_start_monotonic_ns)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--seal", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--transport-manifest", help=argparse.SUPPRESS)
    parser.add_argument(
        "--capability-pipe-handle", type=int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--parent-process-handle", type=int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument("--claim-handle", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.seal:
        path = seal()
        print(_json_bytes({"status": "sealed", "path": str(path)}).decode(), end="")
        return
    if args.preflight:
        print(_json_bytes(read_only_preflight()).decode(), end="")
        return
    if args.execute:
        path, result = execute_parent()
        print(
            _json_bytes(
                {
                    "status": "complete",
                    "result_path": str(path),
                    "decision": result["decision"],
                }
            ).decode(),
            end="",
        )
        return
    if not args.transport_manifest:
        raise AuthorizationError("hidden worker requires a transport manifest locator")
    result = _worker_execute(args)
    print(
        _canonical_bytes({"status": "worker_ok", "result_path": str(result)}).decode(
            "utf-8"
        )
    )


if __name__ == "__main__":
    main()
