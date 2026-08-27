"""Full-runtime, canonical one-shot runner for the frozen P1 Cycle-1 screen.

Only stdlib modules are imported before an isolated worker authenticates the
private project snapshot and clean runtime.  Scientific execution remains
disabled until a separate, digest-pinned independent-QA PASS authorization is
created.  This file never accepts official test/sample/submission paths.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True

CANONICAL_PROJECT_ROOT = Path("C:/Users/cedis/PycharmProjects/PythonProject")
CANONICAL_PROJECT_ROOT_TEXT = "C:/Users/cedis/PycharmProjects/PythonProject"
BOOTSTRAP_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ARTIFACT_RELATIVE = (
    "artifacts/p1_long_event_segment_proposal_rescore_20260826_v2_infrastructure_v7"
)
CANONICAL_ARTIFACT_DIR = CANONICAL_PROJECT_ROOT / CANONICAL_ARTIFACT_RELATIVE
SEAL_PATH = CANONICAL_ARTIFACT_DIR / "preexecution_seal.json"
LAUNCH_CLAIM_PATH = CANONICAL_ARTIFACT_DIR / "canonical_launch.claim"

EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_20260826_v2_infrastructure_v7"
SCIENTIFIC_EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_20260826_v1"
SINGLE_ATTEMPT_NAMESPACE = "P1_CYCLE1_SEGMENT_RESCORE_CANONICAL_V7_ONE_SHOT"
AUTHORIZATION_ENV_VAR = "P1_LONG_EVENT_SEGMENT_V7_AUTHORIZATION_SHA256"
AUTHORIZATION_SCHEMA_VERSION = (
    "p1_long_event_segment_proposal_rescore.execution_authorization.v7"
)
MANIFEST_SCHEMA_VERSION = "p1_segment_rescore.private_capability_manifest.v7"
SEAL_SCHEMA_VERSION = "p1_long_event_segment_proposal_rescore.preexecution_seal.v7"
RUNTIME_SCHEMA_VERSION = "p1_segment_rescore.full_runtime_inventory.v7"
SNAPSHOT_SCHEMA_VERSION = "p1_segment_rescore.full_snapshot_inventory.v7"

MAXIMUM_LIFETIME_PHYSICAL_FITS = 72
MAXIMUM_SCIENTIFIC_MATERIALIZATIONS = 21
HARD_WALL_SECONDS = 21600
EXPECTED_RUNTIME_FILE_COUNT = 10745
EXPECTED_RUNTIME_EXCLUSION_COUNT = 2

R7_AMENDMENT_PATH = (
    CANONICAL_PROJECT_ROOT
    / "configs/experiments/p1_long_event_segment_proposal_rescore_v7_full_runtime_replay_firewall_amendment.json"
)
R6_QA_PATH = (
    CANONICAL_PROJECT_ROOT
    / "reports/p1_long_event_segment_proposal_rescore_v6_independent_preexecution_qa_20260826.json"
)
R6_RUNNER_PATH = (
    CANONICAL_PROJECT_ROOT / "scripts/run_p1_long_event_segment_proposal_rescore_v6.py"
)
R6_EXECUTION_MODULE_PATH = (
    CANONICAL_PROJECT_ROOT
    / "src/p1_qc/long_event_segment_proposal_rescore_execution_v6.py"
)
R6_TEST_PATH = (
    CANONICAL_PROJECT_ROOT / "tests/test_run_p1_long_event_segment_proposal_rescore_v6.py"
)
R6_SEAL_PATH = (
    CANONICAL_PROJECT_ROOT
    / "artifacts/p1_long_event_segment_proposal_rescore_20260826_v2_infrastructure_v6/preexecution_seal.json"
)
R7_EXECUTION_MODULE_PATH = (
    CANONICAL_PROJECT_ROOT
    / "src/p1_qc/long_event_segment_proposal_rescore_execution_v7.py"
)
R7_TEST_PATH = (
    CANONICAL_PROJECT_ROOT / "tests/test_run_p1_long_event_segment_proposal_rescore_v7.py"
)
AUTHORIZATION_TEMPLATE_PATH = (
    CANONICAL_PROJECT_ROOT
    / "configs/experiments/p1_long_event_segment_proposal_rescore_v7_execution_authorization_template.json"
)
AUTHORIZATION_PATH = (
    CANONICAL_PROJECT_ROOT
    / "configs/experiments/p1_long_event_segment_proposal_rescore_v7_execution_authorization.json"
)
R7_QA_PATH = (
    CANONICAL_PROJECT_ROOT
    / "reports/p1_long_event_segment_proposal_rescore_v7_independent_preexecution_qa_20260826.json"
)

R7_AMENDMENT_SHA256 = (
    "71563c954a5c529044d82c63af0e44ddf313dcc55b787c784afd153fc14434ff"
)
R6_QA_SHA256 = "932662de49fd1bdfcd7a546bdeb2a88732dd36e9db54e849354fbdd5c34e2c08"
R6_RUNNER_SHA256 = "001109137fa4daa5408977d14473d95639ba51b8c70301e802626c60ea59509e"
R6_EXECUTION_MODULE_SHA256 = (
    "8b91db9234bde301730728dcffa4fd2c014b099d0c518571ddaa733608b81636"
)
R6_TEST_SHA256 = "49b99eca65f3676414b459314e5ec7366852ad25cdff9367fe0f61c0fb71cabd"
R6_SEAL_SHA256 = "0eb3667ed032b931c84b0948f1bcb6e0e9762eac378b4abcc99f784d60511bbf"
R7_EXECUTION_MODULE_SHA256 = (
    "9f532581bfda500f4bdf1f923ad602c16965ccb39a4ab6157c3f88c3af88c061"
)

EXPECTED_DISTRIBUTIONS = {
    "joblib": "1.5.3",
    "lightgbm": "4.7.0",
    "narwhals": "2.24.0",
    "numpy": "2.3.5",
    "pandas": "3.0.1",
    "psutil": "7.2.2",
    "pyarrow": "25.0.1",
    "python-dateutil": "2.9.0.post0",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.0",
    "six": "1.17.0",
    "threadpoolctl": "3.6.0",
}
RESOURCE_DISTRIBUTIONS = {"tzdata": "2026.3"}
EXPECTED_NUMPY_CLI_NAMES = {"f2py.exe", "numpy-config.exe"}
EXPECTED_NUMPY_CLI = {
    "f2py.exe": {
        "bytes": 108372,
        "sha256": "8c2d76346997109b5195961b18f29901ae96e4bc0891a020e92243fc2c26f9ad",
    },
    "numpy-config.exe": {
        "bytes": 108372,
        "sha256": "23961034f9f7ecfcd8667d674e0620f98b1cef6ffb22aa175bcaebab6fc13736",
    },
}

_SENSITIVE_NAMES = {"test.csv", "sample_submission.csv", "submission.csv"}


class AuthorizationError(RuntimeError):
    """Authorization failed before any scientific operation."""


class WireFormatError(ValueError):
    """JSON bytes violate the exact wire contract."""


def _now_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


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
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_record(value: Any) -> dict[str, Any]:
    raw = _canonical_bytes(value)
    return {"bytes": len(raw), "sha256": _sha256_bytes(raw)}


def _reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise WireFormatError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> Any:
    raise WireFormatError(f"non-finite JSON number: {value}")


def _reject_nonfinite(value: Any, label: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise WireFormatError(f"non-finite JSON value at {label}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for ordinal, item in enumerate(value):
            _reject_nonfinite(item, f"{label}[{ordinal}]")


def _json_from_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WireFormatError(f"invalid JSON bytes: {label}") from error
    _reject_nonfinite(value)
    if not isinstance(value, dict):
        raise WireFormatError(f"expected JSON object: {label}")
    return value


def _read_bound_bytes(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        raw = handle.read()
        after = os.fstat(handle.fileno())
    identity = (before.st_size, before.st_mtime_ns, before.st_ino)
    if identity != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError(f"held read identity changed: {path.name}")
    if len(raw) != before.st_size:
        raise RuntimeError(f"held read byte count changed: {path.name}")
    observed = _sha256_bytes(raw)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(f"held read digest mismatch: {path.name}")
    return raw, {
        "bytes": len(raw),
        "sha256": observed,
        "identity": {
            "device": int(before.st_dev),
            "inode": int(before.st_ino),
            "mtime_ns": int(before.st_mtime_ns),
            "size": int(before.st_size),
        },
        "single_held_byte_snapshot": True,
    }


def _read_bound_json(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    raw, receipt = _read_bound_bytes(path, expected_sha256=expected_sha256)
    return _json_from_bytes(raw, label=path.name), raw, receipt


def _reject_sensitive_path(path: Path) -> None:
    name = path.name.lower()
    if name in _SENSITIVE_NAMES or (
        "submission" in name and path.suffix.lower() in {".csv", ".parquet"}
    ):
        raise RuntimeError("official/sample/submission path is prohibited")


def _canonical_relative(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(CANONICAL_PROJECT_ROOT):
        raise RuntimeError("path escapes canonical repository")
    _reject_sensitive_path(resolved)
    return resolved.relative_to(CANONICAL_PROJECT_ROOT).as_posix()


def _project_relative_literal(path: Path) -> str:
    """Map a compile-time canonical path without touching the live filesystem."""

    try:
        relative = path.relative_to(CANONICAL_PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise RuntimeError("literal path escapes canonical repository") from error
    _validate_tree_relative(relative)
    if Path(relative).name.lower() in _SENSITIVE_NAMES:
        raise RuntimeError("sensitive literal path is prohibited")
    return relative


def _assert_no_reparse_components(path: Path) -> None:
    """Reject symlink/junction/reparse components without following them as trust."""

    candidate = Path(os.path.abspath(os.fspath(path)))
    chain = [candidate, *candidate.parents]
    for component in chain:
        if not component.exists():
            continue
        stat = component.lstat()
        attributes = int(getattr(stat, "st_file_attributes", 0))
        if (
            component.is_symlink()
            or bool(getattr(component, "is_junction", lambda: False)())
            or attributes & 0x00000400
        ):
            raise RuntimeError(f"path contains a reparse component: {component.name}")


def _source_identity(path: Path) -> dict[str, Any]:
    _assert_no_reparse_components(path)
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or bool(getattr(resolved, "is_junction", lambda: False)()):
        raise RuntimeError(f"source is a link/reparse point: {resolved.name}")
    if not resolved.is_file():
        raise RuntimeError(f"source is not a regular file: {resolved.name}")
    stat = resolved.stat()
    return {
        "resolved_path": str(resolved),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def _atomic_create_bytes(path: Path, raw: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        view = memoryview(raw)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short create-only write")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
        temporary.unlink()
        if path.stat().st_nlink != 1:
            raise RuntimeError("create-only publication retained multiple hard links")
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
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


def _verify_top_level_stdlib_only(path: Path | None = None) -> list[str]:
    source = (path or Path(__file__).resolve()).read_text(encoding="utf-8")
    tree = ast.parse(source)
    allowed = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        for name in names:
            if name and name not in allowed:
                raise RuntimeError(f"non-stdlib top-level import: {name}")
            imports.append(name)
    return sorted(set(imports))


def _load_module_from_path(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"unable to load sealed module: {path.name}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_r6_runner(path: Path = R6_RUNNER_PATH) -> ModuleType:
    _read_bound_bytes(path, expected_sha256=R6_RUNNER_SHA256)
    module = _load_module_from_path(path, f"_p1_segment_r6_{uuid.uuid4().hex}")
    _read_bound_bytes(path, expected_sha256=R6_RUNNER_SHA256)
    return module


def _normalise_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _runtime_site_root(distribution: importlib.metadata.Distribution) -> Path:
    located = Path(distribution.locate_file(""))
    _assert_no_reparse_components(located)
    root = located.resolve(strict=True)
    if root.name.lower() != "site-packages":
        raise RuntimeError("runtime distribution root is not site-packages")
    return root


def _full_runtime_inventory() -> dict[str, Any]:
    """Inventory complete RECORD payloads without importing their packages."""

    owners: dict[str, set[str]] = {}
    records: dict[str, dict[str, Any]] = {}
    exclusions: list[dict[str, Any]] = []
    site_roots: set[Path] = set()
    versions: dict[str, str] = {}
    declared = {**EXPECTED_DISTRIBUTIONS, **RESOURCE_DISTRIBUTIONS}
    initially_resolved = {
        name: importlib.metadata.distribution(name) for name in declared
    }
    initial_roots = {_runtime_site_root(value) for value in initially_resolved.values()}
    if len(initial_roots) != 1:
        raise RuntimeError("runtime distributions do not share one initial site root")
    sealed_site_root = next(iter(initial_roots))
    for distribution_name, expected_version in declared.items():
        matches = list(
            importlib.metadata.Distribution.discover(
                name=distribution_name,
                path=[str(sealed_site_root)],
            )
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"runtime distribution multiplicity changed: {distribution_name}"
            )
        distribution = matches[0]
        if distribution.version != expected_version:
            raise RuntimeError(f"runtime version changed: {distribution_name}")
        versions[distribution_name] = distribution.version
        root = _runtime_site_root(distribution)
        site_roots.add(root)
        files = distribution.files
        if files is None:
            raise RuntimeError(f"distribution RECORD is unavailable: {distribution_name}")
        for package_path in files:
            located = Path(distribution.locate_file(package_path))
            _assert_no_reparse_components(located)
            source = located.resolve(strict=True)
            if not source.is_file():
                raise RuntimeError(
                    f"distribution RECORD entry is not a regular file: {distribution_name}"
                )
            try:
                relative = source.relative_to(root).as_posix()
            except ValueError:
                if (
                    distribution_name == "numpy"
                    and source.parent.name.lower() == "scripts"
                    and source.name.lower() in EXPECTED_NUMPY_CLI_NAMES
                ):
                    exclusions.append(
                        {
                            "distribution": distribution_name,
                            "record_path": str(package_path).replace("\\", "/"),
                            "resolved_source_path": str(source),
                            "name": source.name,
                            "classification": (
                                "NON_IMPORTABLE_CONSOLE_ENTRY_OUTSIDE_SITE_PACKAGES"
                            ),
                            "bytes": source.stat().st_size,
                            "sha256": _sha256(source),
                            "source_identity": _source_identity(source),
                        }
                    )
                    continue
                raise RuntimeError(
                    f"runtime RECORD entry escapes site-packages: {distribution_name}"
                ) from None
            if source.suffix.lower() == ".pth" or source.name.lower() in {
                "sitecustomize.py",
                "usercustomize.py",
            }:
                raise RuntimeError("runtime inventory contains a startup hook")
            identity = _source_identity(source)
            observed = {
                "relative_path": relative,
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
                "source_identity": identity,
                "native_payload": source.suffix.lower() in {".pyd", ".dll"},
                "resource_only": distribution_name in RESOURCE_DISTRIBUTIONS,
            }
            if relative in records:
                previous = records[relative]
                comparable = {key: previous[key] for key in observed}
                if comparable != observed:
                    raise RuntimeError(f"runtime distribution collision differs: {relative}")
            else:
                records[relative] = observed
            owners.setdefault(relative, set()).add(distribution_name)

    if len(site_roots) != 1:
        raise RuntimeError("runtime distributions do not share one sealed site root")
    for relative, record in records.items():
        record["owners"] = sorted(owners[relative])
    exclusions.sort(key=lambda value: (value["distribution"], value["record_path"]))
    if {value["name"].lower() for value in exclusions} != EXPECTED_NUMPY_CLI_NAMES:
        raise RuntimeError("sealed numpy CLI exclusion membership changed")
    if len(exclusions) != EXPECTED_RUNTIME_EXCLUSION_COUNT:
        raise RuntimeError("sealed runtime exclusion count changed")
    for value in exclusions:
        expected = EXPECTED_NUMPY_CLI[value["name"].lower()]
        if value["bytes"] != expected["bytes"] or value["sha256"] != expected["sha256"]:
            raise RuntimeError("sealed numpy CLI exclusion bytes changed")
    if len(records) != EXPECTED_RUNTIME_FILE_COUNT:
        raise RuntimeError(
            f"clean runtime file count changed: {len(records)} != {EXPECTED_RUNTIME_FILE_COUNT}"
        )

    python_home = Path(sys.base_prefix).resolve(strict=True)
    host_candidates = {
        Path(sys.executable).resolve(strict=True),
        Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True),
    }
    host_candidates.update(python_home.glob("python*.dll"))
    dll_root = python_home / "DLLs"
    if dll_root.is_dir():
        host_candidates.update(dll_root.glob("*.dll"))
        host_candidates.update(dll_root.glob("*.pyd"))
    host_files: dict[str, Any] = {}
    for path in sorted(host_candidates, key=lambda item: str(item).lower()):
        resolved = path.resolve(strict=True)
        relative = (
            "python_executable"
            if resolved == Path(sys.executable).resolve(strict=True)
            else f"python_home/{resolved.relative_to(python_home).as_posix()}"
        )
        host_files[relative] = {
            "resolved_source_path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": _sha256(resolved),
            "source_identity": _source_identity(resolved),
            "native_payload": resolved.suffix.lower() in {".pyd", ".dll", ".exe"},
        }
    if "python_executable" not in host_files or not any(
        key.lower().endswith("python312.dll") for key in host_files
    ):
        raise RuntimeError("Python executable/native host closure is incomplete")

    value = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "distribution_versions": versions,
        "import_distributions": sorted(EXPECTED_DISTRIBUTIONS),
        "resource_distributions": sorted(RESOURCE_DISTRIBUTIONS),
        "site_source_root": str(next(iter(site_roots))),
        "files": dict(sorted(records.items())),
        "excluded_nonimport_cli": exclusions,
        "host_files": host_files,
        "trusted_os_native_roots": [
            str((Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32").resolve())
        ],
        "policy": {
            "copy_mode": "VERIFIED_BYTE_COPY_NO_HARDLINK",
            "live_site_packages_on_worker_sys_path": False,
            "pth_files_processed": 0,
            "site_module_processed": False,
            "source_and_destination_handles_held_through_worker_termination": True,
            "os_native_root_is_platform_trust_anchor": True,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "architecture": platform.machine(),
        },
    }
    if value["python"]["version"] != "3.12.10":
        raise RuntimeError("Python runtime version changed")
    return value


def _record_matches_path(record: Mapping[str, Any], path: Path) -> None:
    resolved = path.resolve(strict=True)
    identity = _source_identity(resolved)
    if (
        resolved.stat().st_size != int(record["bytes"])
        or _sha256(resolved) != record["sha256"]
        or identity != record["source_identity"]
    ):
        raise RuntimeError(f"sealed source identity/digest changed: {resolved.name}")


class HeldReadDenyMutation:
    """Hold FILE_SHARE_READ-only handles and preserve exact file identities."""

    def __init__(self, paths: Sequence[Path]) -> None:
        resolved = [path.resolve(strict=True) for path in paths]
        if len(resolved) != len(set(resolved)):
            raise RuntimeError("held path inventory contains duplicates")
        self.paths = tuple(sorted(resolved, key=lambda value: str(value).lower()))
        self._handles: list[Any] = []
        self._readers: dict[str, Any] = {}
        self.records: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> HeldReadDenyMutation:
        try:
            if os.name == "nt":
                self._open_windows()
            else:
                for path in self.paths:
                    handle = path.open("rb")
                    self._handles.append(handle)
                    self._readers[str(path)] = handle
                    stat = os.fstat(handle.fileno())
                    raw = handle.read()
                    handle.seek(0)
                    self.records[str(path)] = {
                        "bytes": len(raw),
                        "sha256": _sha256_bytes(raw),
                        "file_id": f"{stat.st_dev}:{stat.st_ino}",
                        "final_path": str(path),
                    }
            return self
        except BaseException:
            self.close(suppress=True)
            raise

    def _open_windows(self) -> None:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        class FILE_ID_128(ctypes.Structure):
            _fields_ = [("Identifier", ctypes.c_ubyte * 16)]

        class FILE_ID_INFO(ctypes.Structure):
            _fields_ = [
                ("VolumeSerialNumber", ctypes.c_ulonglong),
                ("FileId", FILE_ID_128),
            ]

        class FILE_STANDARD_INFO(ctypes.Structure):
            _fields_ = [
                ("AllocationSize", ctypes.c_longlong),
                ("EndOfFile", ctypes.c_longlong),
                ("NumberOfLinks", wintypes.DWORD),
                ("DeletePending", wintypes.BOOLEAN),
                ("Directory", wintypes.BOOLEAN),
            ]

        class FILE_ATTRIBUTE_TAG_INFO(ctypes.Structure):
            _fields_ = [
                ("FileAttributes", wintypes.DWORD),
                ("ReparseTag", wintypes.DWORD),
            ]

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
        invalid = wintypes.HANDLE(-1).value
        get_info = kernel32.GetFileInformationByHandleEx
        get_info.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        get_final = kernel32.GetFinalPathNameByHandleW
        get_final.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
        get_final.restype = wintypes.DWORD
        for path in self.paths:
            handle = create_file(
                str(path),
                0x80000000 | 0x00000080,
                0x00000001,
                None,
                3,
                0x00200000 | 0x08000000,
                None,
            )
            if handle in {None, invalid}:
                raise OSError(ctypes.get_last_error(), f"cannot hold file: {path.name}")
            info = FILE_ID_INFO()
            if not get_info(handle, 18, ctypes.byref(info), ctypes.sizeof(info)):
                kernel32.CloseHandle(handle)
                raise OSError(ctypes.get_last_error(), "cannot read held FileIdInfo")
            standard = FILE_STANDARD_INFO()
            if not get_info(handle, 1, ctypes.byref(standard), ctypes.sizeof(standard)):
                kernel32.CloseHandle(handle)
                raise OSError(ctypes.get_last_error(), "cannot read held FileStandardInfo")
            attributes = FILE_ATTRIBUTE_TAG_INFO()
            if not get_info(handle, 9, ctypes.byref(attributes), ctypes.sizeof(attributes)):
                kernel32.CloseHandle(handle)
                raise OSError(ctypes.get_last_error(), "cannot read held FileAttributeTagInfo")
            if (
                bool(standard.Directory)
                or bool(standard.DeletePending)
                or int(standard.NumberOfLinks) < 1
                or int(attributes.FileAttributes) & 0x00000400
            ):
                kernel32.CloseHandle(handle)
                raise RuntimeError("held entry is directory/deleting/reparse")
            needed = get_final(handle, None, 0, 0)
            if not needed:
                kernel32.CloseHandle(handle)
                raise OSError(ctypes.get_last_error(), "cannot size held final path")
            buffer = ctypes.create_unicode_buffer(needed + 1)
            if not get_final(handle, buffer, len(buffer), 0):
                kernel32.CloseHandle(handle)
                raise OSError(ctypes.get_last_error(), "cannot read held final path")
            file_id = (
                f"{int(info.VolumeSerialNumber)}:"
                f"{bytes(info.FileId.Identifier).hex()}"
            )
            try:
                descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
            except BaseException:
                kernel32.CloseHandle(handle)
                raise
            reader = os.fdopen(descriptor, "rb", closefd=True)
            try:
                raw = reader.read()
                reader.seek(0)
                locked_stat = path.stat()
                final_path = buffer.value.removeprefix("\\\\?\\")
                if os.path.normcase(final_path) != os.path.normcase(str(path)):
                    raise RuntimeError(
                        "held final path differs from requested source identity"
                    )
            except BaseException:
                reader.close()
                raise
            self._handles.append(reader)
            self._readers[str(path)] = reader
            self.records[str(path)] = {
                "bytes": len(raw),
                "sha256": _sha256_bytes(raw),
                "file_id": file_id,
                "final_path": final_path,
                "number_of_links": int(standard.NumberOfLinks),
                "device": int(locked_stat.st_dev),
                "inode": int(locked_stat.st_ino),
                "mtime_ns": int(locked_stat.st_mtime_ns),
                "read_from_held_handle": True,
            }

    def assert_record(
        self,
        path: Path,
        expected: Mapping[str, Any],
        *,
        require_source_identity: bool = False,
    ) -> None:
        key = str(path.resolve(strict=True))
        record = self.records.get(key)
        if record is None:
            raise RuntimeError("required lifetime hold is absent")
        reader = self._readers[key]
        reader.seek(0)
        raw = reader.read()
        reader.seek(0)
        if (
            len(raw) != int(expected["bytes"])
            or _sha256_bytes(raw) != expected["sha256"]
            or int(record["bytes"]) != int(expected["bytes"])
            or record["sha256"] != expected["sha256"]
        ):
            raise RuntimeError(f"held file differs from sealed inventory: {path.name}")
        identity = expected.get("source_identity")
        if require_source_identity:
            if not isinstance(identity, Mapping):
                raise RuntimeError("sealed source identity is unavailable")
            if (
                os.path.normcase(str(identity["resolved_path"]))
                != os.path.normcase(str(record["final_path"]))
                or int(identity["device"]) != int(record.get("device", identity["device"]))
                or int(identity["inode"]) != int(record.get("inode", identity["inode"]))
                or int(identity["mtime_ns"])
                != int(record.get("mtime_ns", identity["mtime_ns"]))
            ):
                raise RuntimeError("held source identity differs from sealed inventory")

    def read_bytes(self, path: Path) -> bytes:
        key = str(path.resolve(strict=True))
        reader = self._readers.get(key)
        if reader is None:
            raise RuntimeError("requested bytes are not lifetime-held")
        reader.seek(0)
        raw = reader.read()
        reader.seek(0)
        record = self.records[key]
        if len(raw) != int(record["bytes"]) or _sha256_bytes(raw) != record["sha256"]:
            raise RuntimeError("held bytes changed during lifetime read")
        return raw

    def file_ids(self) -> set[str]:
        return {str(record["file_id"]) for record in self.records.values()}

    def close(self, *, suppress: bool = False) -> None:
        errors: list[BaseException] = []
        while self._handles:
            value = self._handles.pop()
            try:
                if isinstance(value, tuple):
                    kernel32, handle = value
                    if not kernel32.CloseHandle(handle):
                        raise OSError("unable to close lifetime trust handle")
                else:
                    value.close()
            except BaseException as error:
                errors.append(error)
        if errors and not suppress:
            raise RuntimeError("one or more lifetime trust handles failed to close") from errors[0]
        self._readers.clear()

    def __exit__(self, exc_type: Any, exc: Any, traceback_value: Any) -> None:
        del exc_type, exc, traceback_value
        self.close()


def _training_source_path() -> Path:
    raw = os.environ.get("P1_DATA_DIR", "")
    if not raw:
        raise RuntimeError("P1_DATA_DIR is required for the sealed historical train source")
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = CANONICAL_PROJECT_ROOT / root
    root = root.resolve(strict=True)
    if not root.is_relative_to(CANONICAL_PROJECT_ROOT):
        raise RuntimeError("P1_DATA_DIR must resolve inside the canonical repository")
    source = (root / "train.csv").resolve(strict=True)
    if source.parent != root or not source.is_file():
        raise RuntimeError("P1_DATA_DIR may expose only its direct historical train.csv")
    return source


def _fixed_file_pins() -> dict[Path, str]:
    return {
        R7_AMENDMENT_PATH: R7_AMENDMENT_SHA256,
        R6_QA_PATH: R6_QA_SHA256,
        R6_RUNNER_PATH: R6_RUNNER_SHA256,
        R6_EXECUTION_MODULE_PATH: R6_EXECUTION_MODULE_SHA256,
        R6_TEST_PATH: R6_TEST_SHA256,
        R6_SEAL_PATH: R6_SEAL_SHA256,
        R7_EXECUTION_MODULE_PATH: R7_EXECUTION_MODULE_SHA256,
    }


def _verify_fixed_files(*, require_successor_files: bool) -> dict[str, dict[str, Any]]:
    pins = dict(_fixed_file_pins())
    if require_successor_files:
        pins.update(
            {
                Path(__file__).resolve(): _sha256(Path(__file__).resolve()),
                R7_TEST_PATH: _sha256(R7_TEST_PATH),
                AUTHORIZATION_TEMPLATE_PATH: _sha256(AUTHORIZATION_TEMPLATE_PATH),
            }
        )
    records: dict[str, dict[str, Any]] = {}
    for path, expected in pins.items():
        raw, receipt = _read_bound_bytes(path, expected_sha256=expected)
        records[_canonical_relative(path)] = {
            "bytes": len(raw),
            "sha256": receipt["sha256"],
        }
    if _verify_top_level_stdlib_only() != sorted(_verify_top_level_stdlib_only()):
        raise RuntimeError("stdlib import audit is nondeterministic")
    amendment, _raw, _receipt = _read_bound_json(
        R7_AMENDMENT_PATH,
        expected_sha256=R7_AMENDMENT_SHA256,
    )
    if (
        amendment.get("status")
        != "PROSPECTIVE_INFRASTRUCTURE_FIX_ONLY_NOT_IMPLEMENTED_NOT_AUTHORIZED"
        or amendment.get("implementation_and_QA_contract", {}).get(
            "scientific_execution_authorized"
        )
        is not False
    ):
        raise RuntimeError("r7 amendment authorization state changed")
    qa, _raw, _receipt = _read_bound_json(R6_QA_PATH, expected_sha256=R6_QA_SHA256)
    if qa.get("verdict") != "NO_GO" or qa.get("severity_counts") != {
        "P0": 2,
        "P1": 1,
        "P2": 1,
    }:
        raise RuntimeError("r6 QA NO_GO evidence changed")
    return records


def _snapshot_source_path(relative: str) -> Path:
    if relative == "inputs/train.csv":
        return _training_source_path()
    path = (CANONICAL_PROJECT_ROOT / relative).resolve(strict=True)
    if not path.is_relative_to(CANONICAL_PROJECT_ROOT):
        raise RuntimeError("snapshot source escapes canonical repository")
    _reject_sensitive_path(path)
    return path


def _snapshot_static_inventory() -> tuple[dict[str, Any], dict[str, Any]]:
    r6_seal, _raw, _receipt = _read_bound_json(
        R6_SEAL_PATH,
        expected_sha256=R6_SEAL_SHA256,
    )
    inherited = r6_seal.get("snapshot_static_inventory")
    if not isinstance(inherited, dict) or not inherited:
        raise RuntimeError("r6 snapshot inventory is unavailable")
    inventory: dict[str, Any] = {
        str(relative): {
            "bytes": int(record["bytes"]),
            "sha256": str(record["sha256"]),
        }
        for relative, record in inherited.items()
    }
    additions = {
        _canonical_relative(R7_AMENDMENT_PATH): R7_AMENDMENT_SHA256,
        _canonical_relative(R6_QA_PATH): R6_QA_SHA256,
        _canonical_relative(R6_SEAL_PATH): R6_SEAL_SHA256,
        _canonical_relative(Path(__file__).resolve()): _sha256(Path(__file__).resolve()),
        _canonical_relative(R7_EXECUTION_MODULE_PATH): R7_EXECUTION_MODULE_SHA256,
        _canonical_relative(R7_TEST_PATH): _sha256(R7_TEST_PATH),
        _canonical_relative(AUTHORIZATION_TEMPLATE_PATH): _sha256(
            AUTHORIZATION_TEMPLATE_PATH
        ),
    }
    for relative, expected in additions.items():
        source = _snapshot_source_path(relative)
        record = {"bytes": source.stat().st_size, "sha256": _sha256(source)}
        if record["sha256"] != expected:
            raise RuntimeError(f"successor snapshot pin changed: {relative}")
        inventory[relative] = record
    bindings: dict[str, Any] = {}
    for relative, record in sorted(inventory.items()):
        source = _snapshot_source_path(relative)
        if source.stat().st_size != int(record["bytes"]) or _sha256(source) != record[
            "sha256"
        ]:
            raise RuntimeError(f"snapshot source differs from sealed bytes: {relative}")
        bindings[relative] = {
            "source_path": (
                _canonical_relative(source)
                if source.is_relative_to(CANONICAL_PROJECT_ROOT)
                else str(source)
            ),
            "source_identity": _source_identity(source),
        }
    return dict(sorted(inventory.items())), bindings


def _selected_science_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: readiness[key]
        for key in (
            "status",
            "source",
            "feature_cache",
            "full_feature_cache_binding",
            "exact_round_b_equivalence",
            "left_censored_positive_connected_event_count_by_fold",
            "target_firewall",
            "closure",
        )
        if key in readiness
    }


def _science_preflight_receipt() -> dict[str, Any]:
    r6 = _load_r6_runner()
    receipt = r6.read_only_preflight()
    if receipt.get("verification_sha256") != (
        "15922024421af623f20cf5b3ba0015804d61158021f4b4e20b371d193788be45"
    ):
        raise RuntimeError("r6 sealed scientific readiness changed")
    readiness = receipt.get("readiness", {})
    selected = _selected_science_readiness(readiness)
    return {
        "r6_readonly_preflight_verification_sha256": receipt["verification_sha256"],
        "selected_scientific_readiness_sha256": _sha256_bytes(_canonical_bytes(selected)),
        "selected_scientific_readiness": selected,
    }


def _preflight_core() -> dict[str, Any]:
    _verify_fixed_files(require_successor_files=True)
    runtime = _full_runtime_inventory()
    snapshot, bindings = _snapshot_static_inventory()
    science = _science_preflight_receipt()
    return {
        "schema_version": "p1_segment_rescore.readonly_preflight.v7",
        "experiment_id": EXPERIMENT_ID,
        "scientific_experiment_id": SCIENTIFIC_EXPERIMENT_ID,
        "status": "PASS_ZERO_OPERATION_IMPLEMENTATION_READINESS",
        "r7_amendment_sha256": R7_AMENDMENT_SHA256,
        "r6_qa_no_go_sha256": R6_QA_SHA256,
        "canonical_project_root": CANONICAL_PROJECT_ROOT_TEXT,
        "canonical_artifact_relative_path": CANONICAL_ARTIFACT_RELATIVE,
        "single_attempt_namespace": SINGLE_ATTEMPT_NAMESPACE,
        "deadline_policy": {
            "derivation": "parent_execute_started_epoch_plus_hard_wall_seconds",
            "hard_wall_seconds": HARD_WALL_SECONDS,
            "caller_deadline_override_allowed": False,
        },
        "runtime_inventory_binding": _canonical_record(runtime),
        "snapshot_inventory_binding": _canonical_record(snapshot),
        "runtime_file_count": len(runtime["files"]),
        "runtime_exclusion_count": len(runtime["excluded_nonimport_cli"]),
        "snapshot_file_count": len(snapshot),
        "snapshot_source_binding_sha256": _sha256_bytes(_canonical_bytes(bindings)),
        "science": science,
        "fixed_operation_graph": {
            "inner_anchor_fits": 9,
            "inner_segment_fits": 54,
            "outer_segment_fits": 9,
            "maximum_lifetime_physical_fits": 72,
            "maximum_scientific_materializations": 21,
            "outer_scores": 1,
        },
        "operation_counters": {
            "claims": 0,
            "physical_fits": 0,
            "scientific_materializations": 0,
            "outer_scores": 0,
            "candidate_files": 0,
            "official_reads": 0,
            "uploads": 0,
        },
    }


def _verify_seal_value(seal: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    if seal.get("schema_version") != SEAL_SCHEMA_VERSION:
        raise RuntimeError("r7 seal schema changed")
    if seal.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("r7 seal experiment identity changed")
    if seal.get("status") != "SEALED_STRICT_ZERO_OPERATION_PENDING_INDEPENDENT_QA":
        raise RuntimeError("r7 seal status changed")
    checks = {
        "r7_amendment_sha256": R7_AMENDMENT_SHA256,
        "r6_independent_qa_no_go_sha256": R6_QA_SHA256,
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "execution_module_sha256": R7_EXECUTION_MODULE_SHA256,
        "canonical_project_root": CANONICAL_PROJECT_ROOT_TEXT,
        "canonical_artifact_relative_path": CANONICAL_ARTIFACT_RELATIVE,
        "single_attempt_namespace": SINGLE_ATTEMPT_NAMESPACE,
    }
    for key, expected in checks.items():
        if seal.get(key) != expected:
            raise RuntimeError(f"r7 seal binding changed: {key}")
    runtime = seal.get("full_runtime_inventory")
    snapshot = seal.get("snapshot_static_inventory")
    if not isinstance(runtime, dict) or not isinstance(snapshot, dict):
        raise RuntimeError("r7 seal inventories are unavailable")
    if seal.get("runtime_inventory_binding") != _canonical_record(runtime):
        raise RuntimeError("r7 runtime inventory binding changed")
    if seal.get("snapshot_inventory_binding") != _canonical_record(snapshot):
        raise RuntimeError("r7 snapshot inventory binding changed")
    if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("sha256", ""))):
        raise RuntimeError("r7 seal held receipt is malformed")


def _read_seal() -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    value, raw, receipt = _read_bound_json(SEAL_PATH)
    _verify_seal_value(value, receipt)
    return value, raw, receipt


def seal() -> Path:
    if Path(__file__).resolve().parents[1] != CANONICAL_PROJECT_ROOT.resolve(strict=True):
        raise RuntimeError("seal must run from the canonical repository runner")
    if SEAL_PATH.exists():
        raise FileExistsError("r7 preexecution seal already exists")
    CANONICAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if any(CANONICAL_ARTIFACT_DIR.iterdir()):
        raise FileExistsError("r7 artifact namespace is not empty before seal")
    core = _preflight_core()
    runtime = _full_runtime_inventory()
    snapshot, bindings = _snapshot_static_inventory()
    receipt = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "scientific_experiment_id": SCIENTIFIC_EXPERIMENT_ID,
        "status": "SEALED_STRICT_ZERO_OPERATION_PENDING_INDEPENDENT_QA",
        "sealed_at_kst": _now_kst(),
        "r7_amendment_sha256": R7_AMENDMENT_SHA256,
        "r6_independent_qa_no_go_sha256": R6_QA_SHA256,
        "r6_zero_fit_package": {
            "runner_sha256": R6_RUNNER_SHA256,
            "execution_module_sha256": R6_EXECUTION_MODULE_SHA256,
            "test_sha256": R6_TEST_SHA256,
            "seal_sha256": R6_SEAL_SHA256,
        },
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "runner_normalized_sha256": _sha256(Path(__file__).resolve()),
        "execution_module_sha256": R7_EXECUTION_MODULE_SHA256,
        "test_sha256": _sha256(R7_TEST_PATH),
        "authorization_template_sha256": _sha256(AUTHORIZATION_TEMPLATE_PATH),
        "canonical_project_root": CANONICAL_PROJECT_ROOT_TEXT,
        "canonical_artifact_relative_path": CANONICAL_ARTIFACT_RELATIVE,
        "single_attempt_namespace": SINGLE_ATTEMPT_NAMESPACE,
        "deadline_policy": core["deadline_policy"],
        "runtime_inventory_binding": _canonical_record(runtime),
        "snapshot_inventory_binding": _canonical_record(snapshot),
        "full_runtime_inventory": runtime,
        "snapshot_static_inventory": snapshot,
        "snapshot_source_bindings": bindings,
        "readonly_preflight_verification_sha256": _sha256_bytes(_canonical_bytes(core)),
        "science_preflight": core["science"],
        "fixed_operation_graph": core["fixed_operation_graph"],
        "operation_counters_at_seal": core["operation_counters"],
        "fresh_independent_qa_required": True,
        "actual_authorization_exists": False,
        "manifest_is_transport_only": True,
    }
    return _atomic_create_json(SEAL_PATH, receipt)


def read_only_preflight() -> dict[str, Any]:
    core = _preflight_core()
    verification = _sha256_bytes(_canonical_bytes(core))
    if SEAL_PATH.is_file():
        seal_value, _raw, seal_receipt = _read_seal()
        if verification != seal_value["readonly_preflight_verification_sha256"]:
            raise RuntimeError("r7 read-only preflight differs from seal")
        if _full_runtime_inventory() != seal_value["full_runtime_inventory"]:
            raise RuntimeError("live full runtime differs from seal")
        snapshot, bindings = _snapshot_static_inventory()
        if snapshot != seal_value["snapshot_static_inventory"]:
            raise RuntimeError("live full snapshot differs from seal")
        if bindings != seal_value["snapshot_source_bindings"]:
            raise RuntimeError("live snapshot source identity differs from seal")
        seal_sha256: str | None = seal_receipt["sha256"]
    else:
        seal_sha256 = None
    return {
        "schema_version": "p1_segment_rescore.preflight_receipt.v7",
        "status": "PASS_READ_ONLY_ZERO_OPERATION_PREFLIGHT",
        "verification_sha256": verification,
        "seal_sha256": seal_sha256,
        "runtime_inventory_binding": core["runtime_inventory_binding"],
        "snapshot_inventory_binding": core["snapshot_inventory_binding"],
        "runtime_file_count": core["runtime_file_count"],
        "runtime_exclusion_count": core["runtime_exclusion_count"],
        "snapshot_file_count": core["snapshot_file_count"],
        "canonical_project_root": core["canonical_project_root"],
        "canonical_artifact_relative_path": CANONICAL_ARTIFACT_RELATIVE,
        "single_attempt_namespace": SINGLE_ATTEMPT_NAMESPACE,
        "operation_counters": core["operation_counters"],
    }


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise AuthorizationError(f"exact object membership changed: {label}")
    return value


def _exact_digest(value: Any, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise AuthorizationError(f"digest format changed: {label}")
    return value


def _require_authorization_schema(value: Any) -> dict[str, Any]:
    authorization = _exact_keys(
        value,
        {
            "schema_version",
            "experiment_id",
            "status",
            "authorized",
            "r7_amendment_sha256",
            "r6_independent_qa_no_go_sha256",
            "runner_sha256",
            "runner_normalized_sha256",
            "execution_module_sha256",
            "canonical_project_root",
            "canonical_project_root_identity",
            "canonical_artifact_relative_path",
            "single_attempt_namespace",
            "deadline_policy",
            "preexecution_seal",
            "independent_qa",
            "readonly_preflight_verification_sha256",
            "runtime_inventory_binding",
            "snapshot_inventory_binding",
            "zero_prior_state",
            "operation_authorization",
        },
        "authorization",
    )
    scalar_types = {
        "schema_version": str,
        "experiment_id": str,
        "status": str,
        "authorized": bool,
        "r7_amendment_sha256": str,
        "r6_independent_qa_no_go_sha256": str,
        "runner_sha256": str,
        "runner_normalized_sha256": str,
        "execution_module_sha256": str,
        "canonical_project_root": str,
        "canonical_artifact_relative_path": str,
        "single_attempt_namespace": str,
        "readonly_preflight_verification_sha256": str,
    }
    for key, expected in scalar_types.items():
        if type(authorization[key]) is not expected:
            raise AuthorizationError(f"authorization primitive type changed: {key}")
    for key in (
        "r7_amendment_sha256",
        "r6_independent_qa_no_go_sha256",
        "runner_sha256",
        "runner_normalized_sha256",
        "execution_module_sha256",
        "readonly_preflight_verification_sha256",
    ):
        _exact_digest(authorization[key], key)
    identity = _exact_keys(
        authorization["canonical_project_root_identity"],
        {"device", "inode"},
        "canonical_project_root_identity",
    )
    if type(identity["device"]) is not int or type(identity["inode"]) is not int:
        raise AuthorizationError("canonical root identity primitive changed")
    deadline = _exact_keys(
        authorization["deadline_policy"],
        {"derivation", "hard_wall_seconds", "caller_deadline_override_allowed"},
        "deadline_policy",
    )
    if (
        type(deadline["derivation"]) is not str
        or type(deadline["hard_wall_seconds"]) is not int
        or type(deadline["caller_deadline_override_allowed"]) is not bool
    ):
        raise AuthorizationError("deadline policy primitive changed")
    for name in ("preexecution_seal", "runtime_inventory_binding", "snapshot_inventory_binding"):
        record = authorization[name]
        expected = {"bytes", "sha256"}
        if name == "preexecution_seal":
            expected.add("path")
        _exact_keys(record, expected, name)
        if type(record["bytes"]) is not int or record["bytes"] <= 0:
            raise AuthorizationError(f"authorization byte binding changed: {name}")
        _exact_digest(record["sha256"], f"{name}.sha256")
        if name == "preexecution_seal" and type(record["path"]) is not str:
            raise AuthorizationError("authorization seal path type changed")
    qa = _exact_keys(
        authorization["independent_qa"],
        {"path", "bytes", "sha256", "verdict"},
        "independent_qa",
    )
    if (
        type(qa["path"]) is not str
        or type(qa["bytes"]) is not int
        or qa["bytes"] <= 0
        or type(qa["verdict"]) is not str
    ):
        raise AuthorizationError("independent QA primitive changed")
    _exact_digest(qa["sha256"], "independent_qa.sha256")
    zero = _exact_keys(
        authorization["zero_prior_state"],
        {"claims", "physical_fits", "scientific_materializations", "outer_scores", "candidate_files"},
        "zero_prior_state",
    )
    if any(type(value) is not int for value in zero.values()):
        raise AuthorizationError("zero-state primitive changed")
    operations = _exact_keys(
        authorization["operation_authorization"],
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
    if type(operations["single_attempt"]) is not bool or any(
        type(operations[key]) is not int for key in set(operations) - {"single_attempt"}
    ):
        raise AuthorizationError("operation authorization primitive changed")
    return authorization


def _canonical_root_identity() -> dict[str, int]:
    _assert_no_reparse_components(CANONICAL_PROJECT_ROOT)
    stat = CANONICAL_PROJECT_ROOT.resolve(strict=True).stat()
    return {"device": int(stat.st_dev), "inode": int(stat.st_ino)}


def _validate_authorization_values(
    authorization: Mapping[str, Any],
    seal_value: Mapping[str, Any],
    seal_raw: bytes,
    qa_value: Mapping[str, Any],
    qa_raw: bytes,
) -> None:
    expected_scalars = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "AUTHORIZED_INDEPENDENT_QA_PASS_CANONICAL_ONE_SHOT",
        "authorized": True,
        "r7_amendment_sha256": R7_AMENDMENT_SHA256,
        "r6_independent_qa_no_go_sha256": R6_QA_SHA256,
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "runner_normalized_sha256": _sha256(Path(__file__).resolve()),
        "execution_module_sha256": R7_EXECUTION_MODULE_SHA256,
        "canonical_project_root": CANONICAL_PROJECT_ROOT_TEXT,
        "canonical_project_root_identity": _canonical_root_identity(),
        "canonical_artifact_relative_path": CANONICAL_ARTIFACT_RELATIVE,
        "single_attempt_namespace": SINGLE_ATTEMPT_NAMESPACE,
        "deadline_policy": {
            "derivation": "parent_execute_started_epoch_plus_hard_wall_seconds",
            "hard_wall_seconds": HARD_WALL_SECONDS,
            "caller_deadline_override_allowed": False,
        },
        "readonly_preflight_verification_sha256": seal_value[
            "readonly_preflight_verification_sha256"
        ],
        "runtime_inventory_binding": seal_value["runtime_inventory_binding"],
        "snapshot_inventory_binding": seal_value["snapshot_inventory_binding"],
        "zero_prior_state": {
            "claims": 0,
            "physical_fits": 0,
            "scientific_materializations": 0,
            "outer_scores": 0,
            "candidate_files": 0,
        },
        "operation_authorization": {
            "single_attempt": True,
            "maximum_lifetime_physical_fits": 72,
            "maximum_scientific_materializations": 21,
            "outer_scores": 1,
            "candidate_files": 0,
            "uploads": 0,
        },
    }
    for key, expected in expected_scalars.items():
        if authorization.get(key) != expected:
            raise AuthorizationError(f"authorization binding changed: {key}")
    seal_spec = authorization["preexecution_seal"]
    if seal_spec != {
        "path": CANONICAL_ARTIFACT_RELATIVE + "/preexecution_seal.json",
        "bytes": len(seal_raw),
        "sha256": _sha256_bytes(seal_raw),
    }:
        raise AuthorizationError("authorization seal binding changed")
    qa_spec = authorization["independent_qa"]
    if qa_spec != {
        "path": _canonical_relative(R7_QA_PATH),
        "bytes": len(qa_raw),
        "sha256": _sha256_bytes(qa_raw),
        "verdict": "PASS",
    }:
        raise AuthorizationError("authorization QA binding changed")
    if (
        qa_value.get("verdict") != "PASS"
        or qa_value.get("preexecution_seal_sha256") != _sha256_bytes(seal_raw)
        or qa_value.get("r7_amendment_sha256") != R7_AMENDMENT_SHA256
        or qa_value.get("runner_sha256") != _sha256(Path(__file__).resolve())
        or qa_value.get("execution_module_sha256") != R7_EXECUTION_MODULE_SHA256
    ):
        raise AuthorizationError("fresh independent QA lineage changed")


def _validate_tree_relative(relative: str) -> None:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeError(f"inventory path is not canonical relative: {relative}")
    for part in path.parts:
        if (
            part.endswith((" ", "."))
            or ":" in part
            or part.upper().split(".")[0]
            in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
        ):
            raise RuntimeError(f"inventory path has unsafe Windows spelling: {relative}")


def _verify_exact_tree(
    root: Path,
    expected: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
) -> None:
    _assert_no_reparse_components(root)
    root = root.resolve(strict=True)
    expected_casefold: dict[str, str] = {}
    for relative in expected:
        _validate_tree_relative(relative)
        folded = relative.casefold()
        if folded in expected_casefold and expected_casefold[folded] != relative:
            raise RuntimeError(f"{label} inventory has case-fold collision")
        expected_casefold[folded] = relative
    actual: dict[str, Path] = {}
    actual_casefold: dict[str, str] = {}
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink() or bool(getattr(candidate, "is_junction", lambda: False)()):
            raise RuntimeError(f"{label} tree contains a link/reparse point: {relative}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise RuntimeError(f"{label} tree contains a non-regular entry: {relative}")
        folded = relative.casefold()
        if folded in actual_casefold and actual_casefold[folded] != relative:
            raise RuntimeError(f"{label} actual tree has case-fold collision")
        actual_casefold[folded] = relative
        actual[relative] = candidate
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(f"{label} exact membership changed: missing={missing}, extra={extra}")
    for relative, record in expected.items():
        path = actual[relative]
        if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"{label} file digest changed: {relative}")


def _copy_held_file(
    source_guard: HeldReadDenyMutation,
    source: Path,
    destination: Path,
    expected: Mapping[str, Any],
) -> None:
    source_guard.assert_record(source, expected, require_source_identity=True)
    raw = source_guard.read_bytes(source)
    if len(raw) != int(expected["bytes"]) or _sha256_bytes(raw) != expected["sha256"]:
        raise RuntimeError("held source differs before verified byte copy")
    _atomic_create_bytes(destination, raw)
    if destination.stat().st_size != len(raw) or _sha256(destination) != expected["sha256"]:
        raise RuntimeError("verified byte-copy destination differs")


def _runtime_source_paths(runtime: Mapping[str, Any]) -> list[Path]:
    values = {
        Path(record["source_identity"]["resolved_path"]).resolve(strict=True)
        for record in runtime["files"].values()
    }
    values.update(
        Path(record["source_identity"]["resolved_path"]).resolve(strict=True)
        for record in runtime["excluded_nonimport_cli"]
    )
    values.update(
        Path(record["resolved_source_path"]).resolve(strict=True)
        for record in runtime["host_files"].values()
    )
    return sorted(values, key=lambda value: str(value).lower())


def _snapshot_source_paths(
    inventory: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> list[Path]:
    del inventory
    values = []
    for binding in bindings.values():
        raw = str(binding["source_path"])
        path = Path(raw)
        if not path.is_absolute():
            path = CANONICAL_PROJECT_ROOT / path
        values.append(path.resolve(strict=True))
    return sorted(set(values), key=lambda value: str(value).lower())


def _copy_snapshot_and_runtime(
    seal_value: Mapping[str, Any],
    authorization_raw: bytes,
    seal_raw: bytes,
    qa_raw: bytes,
    source_guard: HeldReadDenyMutation,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    bundle = Path(tempfile.mkdtemp(prefix="p1_segment_v7_private_bundle_"))
    snapshot_root = bundle / "project_snapshot"
    runtime_root = bundle / "isolated_runtime"
    snapshot_root.mkdir()
    runtime_root.mkdir()
    try:
        snapshot_inventory = dict(seal_value["snapshot_static_inventory"])
        bindings = seal_value["snapshot_source_bindings"]
        for relative, record in sorted(snapshot_inventory.items()):
            raw_source = str(bindings[relative]["source_path"])
            source = Path(raw_source)
            if not source.is_absolute():
                source = CANONICAL_PROJECT_ROOT / source
            source_record = dict(record)
            source_record["source_identity"] = bindings[relative]["source_identity"]
            _copy_held_file(
                source_guard,
                source.resolve(strict=True),
                snapshot_root / relative,
                source_record,
            )
        dynamic = {
            _project_relative_literal(AUTHORIZATION_PATH): {
                "bytes": len(authorization_raw),
                "sha256": _sha256_bytes(authorization_raw),
            },
            CANONICAL_ARTIFACT_RELATIVE + "/preexecution_seal.json": {
                "bytes": len(seal_raw),
                "sha256": _sha256_bytes(seal_raw),
            },
            _project_relative_literal(R7_QA_PATH): {
                "bytes": len(qa_raw),
                "sha256": _sha256_bytes(qa_raw),
            },
        }
        dynamic_raw = {
            _project_relative_literal(AUTHORIZATION_PATH): authorization_raw,
            CANONICAL_ARTIFACT_RELATIVE + "/preexecution_seal.json": seal_raw,
            _project_relative_literal(R7_QA_PATH): qa_raw,
        }
        for relative, record in dynamic.items():
            if relative in snapshot_inventory:
                raise RuntimeError("dynamic authority collides with static snapshot")
            _atomic_create_bytes(snapshot_root / relative, dynamic_raw[relative])
            snapshot_inventory[relative] = record

        runtime = seal_value["full_runtime_inventory"]
        for relative, record in sorted(runtime["files"].items()):
            source = Path(record["source_identity"]["resolved_path"])
            _copy_held_file(
                source_guard,
                source.resolve(strict=True),
                runtime_root / relative,
                record,
            )
        _verify_exact_tree(snapshot_root, snapshot_inventory, label="private snapshot")
        _verify_exact_tree(runtime_root, runtime["files"], label="isolated runtime")
        return bundle, snapshot_root, snapshot_inventory, {
            "runtime_root": runtime_root,
            "runtime_inventory": runtime,
        }
    except BaseException:
        shutil.rmtree(bundle, ignore_errors=True)
        raise


def _process_identity(pid: int) -> dict[str, Any]:
    if pid <= 0:
        raise RuntimeError("process identity PID is invalid")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        handle = open_process(0x1000, False, pid)
        if not handle:
            raise OSError(ctypes.get_last_error(), "cannot open parent process")
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
                raise OSError(ctypes.get_last_error(), "cannot read parent process times")
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                raise OSError(ctypes.get_last_error(), "cannot read parent image path")
            creation_filetime = (int(creation.dwHighDateTime) << 32) | int(
                creation.dwLowDateTime
            )
            image = Path(buffer.value).resolve(strict=True)
            return {
                "pid": pid,
                "creation_filetime": creation_filetime,
                "image_path": str(image),
                "image_sha256": _sha256(image),
            }
        finally:
            kernel32.CloseHandle(handle)
    image = Path(sys.executable).resolve(strict=True)
    return {
        "pid": pid,
        "creation_filetime": int(Path(f"/proc/{pid}").stat().st_ctime_ns),
        "image_path": str(image),
        "image_sha256": _sha256(image),
    }


def _manifest_value(
    *,
    snapshot_root: Path,
    runtime_root: Path,
    snapshot_inventory: Mapping[str, Any],
    runtime_inventory_binding: Mapping[str, Any],
    authorization_sha256: str,
    seal_sha256: str,
    qa_sha256: str,
    created_epoch_ns: int,
    deadline_epoch_ns: int,
    parent_identity: Mapping[str, Any],
    launch_nonce: str,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "PRIVATE_SNAPSHOT_RUNTIME_READY_TRANSPORT_ONLY",
        "created_epoch_ns": created_epoch_ns,
        "deadline_epoch_ns": deadline_epoch_ns,
        "hard_wall_seconds": HARD_WALL_SECONDS,
        "parent_identity": dict(parent_identity),
        "launch_nonce": launch_nonce,
        "canonical_project_root": CANONICAL_PROJECT_ROOT_TEXT,
        "canonical_project_root_identity": _canonical_root_identity(),
        "canonical_artifact_relative_path": CANONICAL_ARTIFACT_RELATIVE,
        "single_attempt_namespace": SINGLE_ATTEMPT_NAMESPACE,
        "authorization_sha256": authorization_sha256,
        "seal_sha256": seal_sha256,
        "qa_sha256": qa_sha256,
        "snapshot_inventory_binding": _canonical_record(snapshot_inventory),
        "runtime_inventory_binding": dict(runtime_inventory_binding),
        "snapshot_root": str(snapshot_root.resolve(strict=True)),
        "runtime_root": str(runtime_root.resolve(strict=True)),
        "manifest_role": "TRANSPORT_RECEIPT_NEVER_AUTHORITY",
    }


def _launch_claim_value(manifest: Mapping[str, Any], manifest_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "p1_segment_rescore.canonical_launch_claim.v7",
        "experiment_id": EXPERIMENT_ID,
        "status": "CANONICAL_ONE_SHOT_CLAIMED_BEFORE_WORKER",
        "single_attempt_namespace": SINGLE_ATTEMPT_NAMESPACE,
        "canonical_project_root": manifest["canonical_project_root"],
        "canonical_project_root_identity": manifest["canonical_project_root_identity"],
        "canonical_artifact_relative_path": CANONICAL_ARTIFACT_RELATIVE,
        "created_epoch_ns": manifest["created_epoch_ns"],
        "deadline_epoch_ns": manifest["deadline_epoch_ns"],
        "hard_wall_seconds": HARD_WALL_SECONDS,
        "parent_identity": manifest["parent_identity"],
        "launch_nonce": manifest["launch_nonce"],
        "authorization_sha256": manifest["authorization_sha256"],
        "seal_sha256": manifest["seal_sha256"],
        "qa_sha256": manifest["qa_sha256"],
        "snapshot_inventory_binding": manifest["snapshot_inventory_binding"],
        "runtime_inventory_binding": manifest["runtime_inventory_binding"],
        "transport_manifest_sha256": manifest_sha256,
        "replay_policy": "ANY_EXISTING_CLAIM_PERMANENTLY_BLOCKS_REPLAY",
    }


def _require_manifest_schema(value: Any) -> dict[str, Any]:
    manifest = _exact_keys(
        value,
        {
            "schema_version",
            "experiment_id",
            "status",
            "created_epoch_ns",
            "deadline_epoch_ns",
            "hard_wall_seconds",
            "parent_identity",
            "launch_nonce",
            "canonical_project_root",
            "canonical_project_root_identity",
            "canonical_artifact_relative_path",
            "single_attempt_namespace",
            "authorization_sha256",
            "seal_sha256",
            "qa_sha256",
            "snapshot_inventory_binding",
            "runtime_inventory_binding",
            "snapshot_root",
            "runtime_root",
            "manifest_role",
        },
        "transport_manifest",
    )
    for key in ("created_epoch_ns", "deadline_epoch_ns", "hard_wall_seconds"):
        if type(manifest[key]) is not int:
            raise AuthorizationError(f"manifest integer primitive changed: {key}")
    for key in (
        "schema_version",
        "experiment_id",
        "status",
        "launch_nonce",
        "canonical_project_root",
        "canonical_artifact_relative_path",
        "single_attempt_namespace",
        "snapshot_root",
        "runtime_root",
        "manifest_role",
    ):
        if type(manifest[key]) is not str:
            raise AuthorizationError(f"manifest string primitive changed: {key}")
    for key in ("authorization_sha256", "seal_sha256", "qa_sha256"):
        _exact_digest(manifest[key], f"manifest.{key}")
    _exact_keys(
        manifest["canonical_project_root_identity"],
        {"device", "inode"},
        "manifest.root_identity",
    )
    _exact_keys(
        manifest["parent_identity"],
        {"pid", "creation_filetime", "image_path", "image_sha256"},
        "manifest.parent_identity",
    )
    for key in ("snapshot_inventory_binding", "runtime_inventory_binding"):
        binding = _exact_keys(manifest[key], {"bytes", "sha256"}, f"manifest.{key}")
        if type(binding["bytes"]) is not int or binding["bytes"] <= 0:
            raise AuthorizationError("manifest inventory byte count changed")
        _exact_digest(binding["sha256"], f"manifest.{key}.sha256")
    return manifest


def _require_claim_schema(value: Any) -> dict[str, Any]:
    return _exact_keys(
        value,
        {
            "schema_version",
            "experiment_id",
            "status",
            "single_attempt_namespace",
            "canonical_project_root",
            "canonical_project_root_identity",
            "canonical_artifact_relative_path",
            "created_epoch_ns",
            "deadline_epoch_ns",
            "hard_wall_seconds",
            "parent_identity",
            "launch_nonce",
            "authorization_sha256",
            "seal_sha256",
            "qa_sha256",
            "snapshot_inventory_binding",
            "runtime_inventory_binding",
            "transport_manifest_sha256",
            "replay_policy",
        },
        "canonical_launch_claim",
    )


def _assert_isolated_bootstrap(expected_sha256: str) -> Path:
    if {
        "isolated": int(sys.flags.isolated),
        "no_site": int(sys.flags.no_site),
        "dont_write_bytecode": int(sys.flags.dont_write_bytecode),
    } != {"isolated": 1, "no_site": 1, "dont_write_bytecode": 1}:
        raise AuthorizationError("worker isolation flags changed")
    if any(name in sys.modules for name in ("site", "sitecustomize", "usercustomize")):
        raise AuthorizationError("startup hook module was imported")
    _exact_digest(expected_sha256, "bootstrap_sha256")
    runner = Path(__file__).resolve(strict=True)
    root = runner.parents[1]
    if root == CANONICAL_PROJECT_ROOT.resolve(strict=True):
        raise AuthorizationError("worker bootstrap executed from live repository")
    if runner != root / "scripts/run_p1_long_event_segment_proposal_rescore_v7.py":
        raise AuthorizationError("worker bootstrap path changed")
    if _sha256(runner) != expected_sha256:
        raise AuthorizationError("worker bootstrap bytes changed")
    return root


class SealedOriginFinder(importlib.abc.MetaPathFinder):
    """Resolve a spec first, then reject an undeclared origin before execution."""

    def __init__(
        self,
        delegates: Sequence[Any],
        snapshot_root: Path,
        snapshot_inventory: Mapping[str, Any],
        runtime_root: Path,
        runtime_inventory: Mapping[str, Any],
        host_files: Mapping[str, Any],
        stdlib_roots: Sequence[Path],
    ) -> None:
        self.delegates = tuple(delegates)
        self.snapshot_root = snapshot_root.resolve(strict=True)
        self.snapshot_inventory = snapshot_inventory
        self.runtime_root = runtime_root.resolve(strict=True)
        self.runtime_inventory = runtime_inventory
        self.host_by_path = {
            str(Path(record["resolved_source_path"]).resolve(strict=True)).casefold(): record
            for record in host_files.values()
        }
        self.stdlib_roots = tuple(path.resolve(strict=True) for path in stdlib_roots)

    def _validate_file(self, raw_path: str, *, cached: bool = False) -> None:
        path = Path(raw_path).resolve(strict=True)
        if path.is_relative_to(self.runtime_root):
            relative = path.relative_to(self.runtime_root).as_posix()
            record = self.runtime_inventory.get(relative)
            if record is None:
                raise ImportError(f"undeclared isolated runtime origin: {relative}")
            if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
                raise ImportError(f"isolated runtime origin digest changed: {relative}")
            return
        if path.is_relative_to(self.snapshot_root):
            relative = path.relative_to(self.snapshot_root).as_posix()
            record = self.snapshot_inventory.get(relative)
            if record is None:
                raise ImportError(f"undeclared private snapshot origin: {relative}")
            if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
                raise ImportError(f"private snapshot origin digest changed: {relative}")
            return
        if any(path.is_relative_to(root) for root in self.stdlib_roots):
            if path.suffix.lower() in {".pyd", ".dll"}:
                record = self.host_by_path.get(str(path).casefold())
                if record is None:
                    raise ImportError(f"unsealed Python native origin: {path.name}")
                if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record[
                    "sha256"
                ]:
                    raise ImportError(f"Python native origin digest changed: {path.name}")
            return
        kind = "cached" if cached else "origin"
        raise ImportError(f"module {kind} escapes sealed roots: {path.name}")

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname in sys.modules:
            return None
        for finder in self.delegates:
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            specification = find_spec(fullname, path, target)
            if specification is None:
                continue
            if specification.origin not in {None, "built-in", "frozen"}:
                self._validate_file(str(specification.origin))
            cached = getattr(specification, "cached", None)
            if cached and Path(str(cached)).exists():
                self._validate_file(str(cached), cached=True)
            locations = specification.submodule_search_locations
            if locations is not None:
                for raw in locations:
                    location = Path(str(raw)).resolve(strict=True)
                    if not (
                        location.is_relative_to(self.runtime_root)
                        or location.is_relative_to(self.snapshot_root)
                        or any(location.is_relative_to(root) for root in self.stdlib_roots)
                    ):
                        raise ImportError("namespace package location escapes sealed roots")
            return specification
        return None


def _stdlib_roots() -> list[Path]:
    import sysconfig

    roots: set[Path] = set()
    for key in ("stdlib", "platstdlib"):
        value = sysconfig.get_path(key)
        if value:
            path = Path(value).resolve(strict=True)
            if "site-packages" not in str(path).lower():
                roots.add(path)
    if not roots:
        raise RuntimeError("stdlib trust roots are unavailable")
    return sorted(roots, key=lambda value: str(value).lower())


def _activate_isolated_runtime(
    snapshot_root: Path,
    snapshot_inventory: Mapping[str, Any],
    runtime_root: Path,
    runtime: Mapping[str, Any],
) -> tuple[SealedOriginFinder, list[Any]]:
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        os.environ.pop(key, None)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    stdlib = _stdlib_roots()
    project_source = (snapshot_root / "src").resolve(strict=True)
    runtime_root = runtime_root.resolve(strict=True)
    sys.path[:] = [str(project_source), str(runtime_root), *(str(path) for path in stdlib)]
    sys.path_importer_cache.clear()
    importlib.invalidate_caches()
    if any("site-packages" in value.lower() for value in sys.path):
        raise RuntimeError("live site-packages remained on worker sys.path")

    dll_handles: list[Any] = []
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.SetDefaultDllDirectories(0x00001000 | 0x00000400):
            raise OSError(ctypes.get_last_error(), "cannot restrict default DLL directories")
        kernel32.SetDllDirectoryW("")
        dll_dirs = sorted(
            {
                (runtime_root / relative).parent.resolve(strict=True)
                for relative, record in runtime["files"].items()
                if record.get("native_payload") and relative.lower().endswith(".dll")
            },
            key=lambda value: str(value).lower(),
        )
        for directory in dll_dirs:
            dll_handles.append(os.add_dll_directory(str(directory)))

    delegates = tuple(sys.meta_path)
    guard = SealedOriginFinder(
        delegates,
        snapshot_root,
        snapshot_inventory,
        runtime_root,
        runtime["files"],
        runtime["host_files"],
        stdlib,
    )
    sys.meta_path.insert(0, guard)
    return guard, dll_handles


def _eager_runtime_imports(runtime: Mapping[str, Any], runtime_root: Path) -> None:
    expected_modules = (
        "joblib",
        "lightgbm",
        "narwhals",
        "numpy",
        "pandas",
        "psutil",
        "pyarrow",
        "dateutil",
        "sklearn",
        "scipy",
        "six",
        "threadpoolctl",
        "tzdata",
    )
    for name in expected_modules:
        importlib.import_module(name)
    for distribution_name, expected_version in {
        **EXPECTED_DISTRIBUTIONS,
        **RESOURCE_DISTRIBUTIONS,
    }.items():
        matches = list(
            importlib.metadata.Distribution.discover(
                name=distribution_name,
                path=[str(runtime_root)],
            )
        )
        if len(matches) != 1 or matches[0].version != expected_version:
            raise RuntimeError(f"isolated distribution closure changed: {distribution_name}")
    from zoneinfo import ZoneInfo as WorkerZoneInfo

    if WorkerZoneInfo("Asia/Seoul").key != "Asia/Seoul":
        raise RuntimeError("sealed tzdata resource is unavailable")


def _validate_loaded_module_origins(guard: SealedOriginFinder) -> None:
    for module in tuple(sys.modules.values()):
        raw = getattr(module, "__file__", None)
        if raw:
            guard._validate_file(str(raw))


def _validate_loaded_native_modules(runtime: Mapping[str, Any], runtime_root: Path) -> None:
    psutil = importlib.import_module("psutil")
    runtime_by_path = {
        str((runtime_root / relative).resolve(strict=True)).casefold(): record
        for relative, record in runtime["files"].items()
        if record.get("native_payload")
    }
    host_by_path = {
        str(Path(record["resolved_source_path"]).resolve(strict=True)).casefold(): record
        for record in runtime["host_files"].values()
    }
    os_roots = [Path(value).resolve(strict=True) for value in runtime["trusted_os_native_roots"]]
    for mapping in psutil.Process().memory_maps(grouped=False):
        raw = str(getattr(mapping, "path", ""))
        if not raw or raw.startswith("["):
            continue
        path = Path(raw)
        if path.suffix.lower() not in {".pyd", ".dll"} or not path.exists():
            continue
        resolved = path.resolve(strict=True)
        key = str(resolved).casefold()
        record = runtime_by_path.get(key) or host_by_path.get(key)
        if record is not None:
            if resolved.stat().st_size != int(record["bytes"]) or _sha256(resolved) != record[
                "sha256"
            ]:
                raise RuntimeError(f"loaded native digest changed: {resolved.name}")
            continue
        if any(resolved.is_relative_to(root) for root in os_roots):
            continue
        raise RuntimeError(f"loaded native module escapes sealed roots: {resolved}")


def _worker_authority_state(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_root = _assert_isolated_bootstrap(str(args.bootstrap_sha256))
    manifest_path = args.capability_manifest.resolve(strict=True)
    manifest_raw, manifest_receipt = _read_bound_bytes(
        manifest_path,
        expected_sha256=str(args.capability_manifest_sha256),
    )
    manifest = _require_manifest_schema(
        _json_from_bytes(manifest_raw, label=manifest_path.name)
    )
    runtime_root = Path(manifest["runtime_root"]).resolve(strict=True)
    if (
        Path(manifest["snapshot_root"]).resolve(strict=True) != snapshot_root
        or manifest_path.parent.resolve(strict=True) != snapshot_root.parent
        or runtime_root.parent != snapshot_root.parent
        or runtime_root.name != "isolated_runtime"
    ):
        raise AuthorizationError("private bundle transport paths changed")
    cli_checks = {
        "launch_nonce": args.launch_nonce,
        "deadline_epoch_ns": int(args.deadline_epoch_ns),
        "parent_pid": int(args.parent_pid),
        "authorization_sha256": args.authorization_sha256,
        "seal_sha256": args.seal_sha256,
        "qa_sha256": args.qa_sha256,
        "snapshot_inventory_sha256": args.snapshot_inventory_sha256,
        "runtime_inventory_sha256": args.runtime_inventory_sha256,
        "canonical_project_root": args.canonical_project_root,
        "canonical_artifact_relative_path": args.canonical_artifact_relative_path,
        "single_attempt_namespace": args.single_attempt_namespace,
    }
    expected_cli = {
        "launch_nonce": manifest["launch_nonce"],
        "deadline_epoch_ns": manifest["deadline_epoch_ns"],
        "parent_pid": manifest["parent_identity"]["pid"],
        "authorization_sha256": manifest["authorization_sha256"],
        "seal_sha256": manifest["seal_sha256"],
        "qa_sha256": manifest["qa_sha256"],
        "snapshot_inventory_sha256": manifest["snapshot_inventory_binding"]["sha256"],
        "runtime_inventory_sha256": manifest["runtime_inventory_binding"]["sha256"],
        "canonical_project_root": manifest["canonical_project_root"],
        "canonical_artifact_relative_path": manifest["canonical_artifact_relative_path"],
        "single_attempt_namespace": manifest["single_attempt_namespace"],
    }
    if cli_checks != expected_cli:
        raise AuthorizationError("hidden-worker CLI differs from transport manifest")
    if (
        manifest["schema_version"] != MANIFEST_SCHEMA_VERSION
        or manifest["experiment_id"] != EXPERIMENT_ID
        or manifest["status"] != "PRIVATE_SNAPSHOT_RUNTIME_READY_TRANSPORT_ONLY"
        or manifest["manifest_role"] != "TRANSPORT_RECEIPT_NEVER_AUTHORITY"
        or manifest["canonical_project_root"]
        != CANONICAL_PROJECT_ROOT_TEXT
        or manifest["canonical_project_root_identity"] != _canonical_root_identity()
        or manifest["canonical_artifact_relative_path"] != CANONICAL_ARTIFACT_RELATIVE
        or manifest["single_attempt_namespace"] != SINGLE_ATTEMPT_NAMESPACE
        or manifest["hard_wall_seconds"] != HARD_WALL_SECONDS
        or manifest["deadline_epoch_ns"] - manifest["created_epoch_ns"]
        != HARD_WALL_SECONDS * 1_000_000_000
        or time.time_ns() >= manifest["deadline_epoch_ns"]
    ):
        raise AuthorizationError("canonical manifest/deadline binding changed")
    if _process_identity(int(args.parent_pid)) != manifest["parent_identity"]:
        raise AuthorizationError("live parent process identity changed")

    auth_relative = _project_relative_literal(AUTHORIZATION_PATH)
    seal_relative = CANONICAL_ARTIFACT_RELATIVE + "/preexecution_seal.json"
    qa_relative = _project_relative_literal(R7_QA_PATH)
    auth_raw, auth_receipt = _read_bound_bytes(
        snapshot_root / auth_relative,
        expected_sha256=manifest["authorization_sha256"],
    )
    supplied = os.environ.get(AUTHORIZATION_ENV_VAR, "")
    if supplied != manifest["authorization_sha256"]:
        raise AuthorizationError("worker external authorization digest changed")
    authorization = _require_authorization_schema(
        _json_from_bytes(auth_raw, label=AUTHORIZATION_PATH.name)
    )
    seal_raw, seal_receipt = _read_bound_bytes(
        snapshot_root / seal_relative,
        expected_sha256=manifest["seal_sha256"],
    )
    seal_value = _json_from_bytes(seal_raw, label=SEAL_PATH.name)
    _verify_seal_value(seal_value, seal_receipt)
    qa_raw, qa_receipt = _read_bound_bytes(
        snapshot_root / qa_relative,
        expected_sha256=manifest["qa_sha256"],
    )
    qa_value = _json_from_bytes(qa_raw, label=R7_QA_PATH.name)
    _validate_authorization_values(
        authorization,
        seal_value,
        seal_raw,
        qa_value,
        qa_raw,
    )
    expected_snapshot = dict(seal_value["snapshot_static_inventory"])
    expected_snapshot.update(
        {
            auth_relative: {"bytes": len(auth_raw), "sha256": auth_receipt["sha256"]},
            seal_relative: {"bytes": len(seal_raw), "sha256": seal_receipt["sha256"]},
            qa_relative: {"bytes": len(qa_raw), "sha256": qa_receipt["sha256"]},
        }
    )
    if manifest["snapshot_inventory_binding"] != _canonical_record(expected_snapshot):
        raise AuthorizationError("transport snapshot binding differs from held authorities")
    runtime = seal_value["full_runtime_inventory"]
    if (
        seal_value["runtime_inventory_binding"] != _canonical_record(runtime)
        or authorization["runtime_inventory_binding"] != _canonical_record(runtime)
        or manifest["runtime_inventory_binding"] != _canonical_record(runtime)
    ):
        raise AuthorizationError("runtime authority binding changed")
    if authorization["snapshot_inventory_binding"] != seal_value[
        "snapshot_inventory_binding"
    ]:
        raise AuthorizationError("static snapshot authority binding changed")
    _verify_exact_tree(snapshot_root, expected_snapshot, label="private snapshot preimport")
    _verify_exact_tree(runtime_root, runtime["files"], label="isolated runtime preimport")

    claim_value, claim_raw, claim_receipt = _read_bound_json(
        LAUNCH_CLAIM_PATH,
        expected_sha256=str(args.launch_claim_sha256),
    )
    _require_claim_schema(claim_value)
    expected_claim = _launch_claim_value(manifest, manifest_receipt["sha256"])
    if claim_value != expected_claim or claim_receipt["sha256"] != args.launch_claim_sha256:
        raise AuthorizationError("canonical launch claim differs from parent capability")
    lease_path = CANONICAL_ARTIFACT_DIR / "worker_start.lease"
    _atomic_create_json(
        lease_path,
        {
            "schema_version": "p1_segment_rescore.worker_start_lease.v7",
            "experiment_id": EXPERIMENT_ID,
            "launch_nonce": manifest["launch_nonce"],
            "parent_identity": manifest["parent_identity"],
            "deadline_epoch_ns": manifest["deadline_epoch_ns"],
            "launch_claim_sha256": claim_receipt["sha256"],
            "worker_pid": os.getpid(),
            "created_at_kst": _now_kst(),
        },
    )
    return {
        "snapshot_root": snapshot_root,
        "runtime_root": runtime_root,
        "manifest": manifest,
        "manifest_sha256": manifest_receipt["sha256"],
        "authorization": authorization,
        "authorization_sha256": auth_receipt["sha256"],
        "seal": seal_value,
        "seal_sha256": seal_receipt["sha256"],
        "qa_sha256": qa_receipt["sha256"],
        "snapshot_inventory": expected_snapshot,
        "runtime": runtime,
        "worker_lease_path": lease_path,
    }


class FirstFitTrustJournal:
    def __init__(self, journal: Any, checkpoint: Any) -> None:
        self._journal = journal
        self._checkpoint = checkpoint
        self._first_fit_checked = False

    def reserve_fit(self, *args: Any, **kwargs: Any) -> int:
        if not self._first_fit_checked:
            self._checkpoint("PREFIT")
            self._first_fit_checked = True
        return self._journal.reserve_fit(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._journal, name)


def _load_worker_science(
    authority: Mapping[str, Any],
) -> tuple[ModuleType, ModuleType, ModuleType, dict[str, Any], dict[str, Any]]:
    snapshot_root = authority["snapshot_root"]
    r6_path = snapshot_root / _project_relative_literal(R6_RUNNER_PATH)
    r6 = _load_r6_runner(r6_path)
    r6.EXPERIMENT_ID = EXPERIMENT_ID
    r6.OUTPUT_PROJECT_ROOT = CANONICAL_PROJECT_ROOT
    r6.ARTIFACT_DIR = CANONICAL_ARTIFACT_DIR
    r6.SEAL_PATH = SEAL_PATH
    legacy_raw, _receipt = _read_bound_bytes(
        r6.LEGACY_CONFIG_PATH,
        expected_sha256=r6.LEGACY_CONFIG_SHA256,
    )
    legacy_config = _json_from_bytes(legacy_raw, label=r6.LEGACY_CONFIG_PATH.name)
    legacy = r6._load_snapshot_legacy_runner(snapshot_root, "_p1_v7_legacy")
    r6_execution_raw, _receipt = _read_bound_bytes(
        snapshot_root / _project_relative_literal(R6_EXECUTION_MODULE_PATH),
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
    selected_science = _selected_science_readiness(selected)
    sealed_science = authority["seal"]["science_preflight"]
    if _sha256_bytes(_canonical_bytes(selected_science)) != sealed_science[
        "selected_scientific_readiness_sha256"
    ]:
        raise RuntimeError("worker scientific readiness differs from sealed preflight")
    wrapper = importlib.import_module(
        "p1_qc.long_event_segment_proposal_rescore_execution_v7"
    )
    if wrapper.R7_AMENDMENT_SHA256 != R7_AMENDMENT_SHA256:
        raise RuntimeError("r7 execution wrapper amendment anchor changed")
    return r6, wrapper, numerical, selected, state


def _artifact_relative(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(CANONICAL_PROJECT_ROOT.resolve(strict=True)):
        raise RuntimeError("artifact path escapes canonical repository")
    return resolved.relative_to(CANONICAL_PROJECT_ROOT.resolve(strict=True)).as_posix()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}


def _trust_checkpoint(
    authority: Mapping[str, Any],
    guard: SealedOriginFinder,
    phase: str,
) -> None:
    if time.time_ns() >= int(authority["manifest"]["deadline_epoch_ns"]):
        raise TimeoutError(f"deadline expired before trust checkpoint {phase}")
    _verify_exact_tree(
        authority["snapshot_root"],
        authority["snapshot_inventory"],
        label=f"private snapshot {phase}",
    )
    _verify_exact_tree(
        authority["runtime_root"],
        authority["runtime"]["files"],
        label=f"isolated runtime {phase}",
    )
    _validate_loaded_module_origins(guard)
    _validate_loaded_native_modules(authority["runtime"], authority["runtime_root"])


def _install_runtime_audit_hook(
    runtime_root: Path,
    runtime: Mapping[str, Any],
) -> None:
    allowed_files = {
        str((runtime_root / relative).resolve(strict=True)).casefold()
        for relative in runtime["files"]
    }
    host_files = {
        str(Path(record["resolved_source_path"]).resolve(strict=True)).casefold()
        for record in runtime["host_files"].values()
    }
    os_roots = tuple(
        Path(value).resolve(strict=True) for value in runtime["trusted_os_native_roots"]
    )

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event not in {"ctypes.dlopen", "os.add_dll_directory"} or not args:
            return
        raw = args[0]
        if raw in {None, ""}:
            return
        path = Path(os.fspath(raw))
        if not path.is_absolute():
            raise RuntimeError(f"unqualified native load rejected at {event}")
        resolved = path.resolve(strict=True)
        key = str(resolved).casefold()
        if event == "os.add_dll_directory":
            if not resolved.is_relative_to(runtime_root.resolve(strict=True)):
                raise RuntimeError("DLL directory escapes isolated runtime")
            return
        if key in allowed_files or key in host_files:
            return
        if any(resolved.is_relative_to(root) for root in os_roots):
            return
        raise RuntimeError("native load escapes sealed runtime and platform roots")

    sys.addaudithook(audit)


def _render_report(result: Mapping[str, Any]) -> bytes:
    metrics = result["metrics"]
    pooled = metrics["pooled"]
    bootstrap = metrics["paired_bootstrap"]
    selected = result["selected_inner_cell"]
    lines = [
        "# P1 장기 이벤트 구간 제안·재채점 고정 실험 (r7 infrastructure)",
        "",
        f"결론: **{result['decision']}**",
        "",
        "공식 평가·제출이 아닌, 사전등록된 historical screen의 집계 결과다.",
        "행별 예측·제출 후보·업로드는 생성하지 않았다.",
        "",
        "## 핵심 수치",
        "",
        f"- 후보 F1: {pooled['candidate']['f1']:.9f}",
        f"- Round-B anchor F1: {pooled['anchor']['f1']:.9f}",
        f"- 후보−anchor F1 Δ: {pooled['f1_delta']:+.9f}",
        f"- paired bootstrap 90% CI: {bootstrap['difference_ci90']}",
        f"- 선택 구조: {selected['cell_id']}, threshold={selected['threshold']}",
        f"- RESEARCH_GO: {result['RESEARCH_GO']}",
        f"- SUBMISSION_GO_RESEARCH_ONLY: {result['SUBMISSION_GO_RESEARCH_ONLY']}",
        "",
        "## 고정 계수",
        "",
        "- inner Round-B anchors 9 + inner segment 54 + outer segment 9 = 72 fits",
        "- scientific materializations 21; outer score 1",
        "",
        "## 한계",
        "",
        "- 로컬 historical window와 공식 평가 분포의 차이는 남는다.",
        "- 결과 사용에는 별도 독립 사후 QA와 명시적 제출 승인이 필요하다.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _publish_success(
    screen: Mapping[str, Any],
    journal: Any,
    authority: Mapping[str, Any],
    r6: ModuleType,
) -> Path:
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
            "canonical_launch_claim_sha256": _sha256(LAUNCH_CLAIM_PATH),
            "runtime_inventory_binding": authority["seal"]["runtime_inventory_binding"],
            "snapshot_static_inventory_binding": authority["seal"][
                "snapshot_inventory_binding"
            ],
            "completed_at_kst": _now_kst(),
        }
    )
    journal.record_aggregate(result)
    _atomic_create_json(metrics_path, result["metrics"])
    _atomic_create_bytes(report_path, _render_report(result))
    _atomic_create_json(result_path, result)
    inventory = {
        _artifact_relative(path): _file_record(path)
        for path in (metrics_path, report_path, result_path, SEAL_PATH, LAUNCH_CLAIM_PATH)
    }
    inventory.update(journal.manifest_records())
    manifest = {
        "schema_version": "p1_long_event_segment_proposal_rescore.manifest.v7",
        "experiment_id": EXPERIMENT_ID,
        "scientific_experiment_id": SCIENTIFIC_EXPERIMENT_ID,
        "attempt_id": journal.attempt_id,
        "status": "WORKER_OUTPUTS_COMPLETE_BEFORE_FINAL_COMMIT",
        "authorization_sha256": authority["authorization_sha256"],
        "preexecution_seal_sha256": authority["seal_sha256"],
        "independent_qa_sha256": authority["qa_sha256"],
        "transport_manifest_sha256": authority["manifest_sha256"],
        "canonical_launch_claim_sha256": _sha256(LAUNCH_CLAIM_PATH),
        "runtime_inventory_binding": authority["seal"]["runtime_inventory_binding"],
        "snapshot_inventory_binding": _canonical_record(authority["snapshot_inventory"]),
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
        "created_at_kst": _now_kst(),
    }
    _atomic_create_json(manifest_path, manifest)
    for relative, record in manifest["artifacts"].items():
        path = CANONICAL_PROJECT_ROOT / relative
        if _file_record(path) != record:
            raise RuntimeError(f"worker manifest verification failed: {relative}")
    journal.terminal_entry(
        "0998_worker_terminal.json",
        {
            "schema_version": "p1_segment_rescore.worker_terminal.v7",
            "status": "WORKER_SUCCESS_COMMIT_PREPARED",
            "result_sha256": _sha256(result_path),
            "manifest_sha256": _sha256(manifest_path),
            "canonical_launch_claim_sha256": _sha256(LAUNCH_CLAIM_PATH),
            "physical_fit_reservations": journal.fit_reservations,
            "physical_fits_completed": journal.fits_completed,
            "scientific_materializations": journal.materializations,
            "outer_scores": 1,
            "created_at_kst": _now_kst(),
        },
    )
    previous, _entries = r6._verify_journal_chain(
        journal.journal_dir,
        required_last="0998_worker_terminal.json",
    )
    journal.terminal_entry(
        "0999_completed.json",
        {
            "schema_version": "p1_segment_rescore.completed.v7",
            "status": "SUCCESS_ALL_OUTPUTS_VERIFIED_READY_FOR_LOCK_RELEASE",
            "result_sha256": _sha256(result_path),
            "manifest_sha256": _sha256(manifest_path),
            "worker_terminal_sha256": previous,
            "physical_fit_reservations": 72,
            "physical_fits_completed": 72,
            "scientific_materializations": 21,
            "outer_scores": 1,
            "created_at_kst": _now_kst(),
        },
    )
    r6._final_success_commit(journal)
    return result_path


def _worker_execute(args: argparse.Namespace) -> Path:
    authority = _worker_authority_state(args)
    guard, dll_handles = _activate_isolated_runtime(
        authority["snapshot_root"],
        authority["snapshot_inventory"],
        authority["runtime_root"],
        authority["runtime"],
    )
    _install_runtime_audit_hook(authority["runtime_root"], authority["runtime"])
    journal: Any | None = None
    phase = "RUNTIME_ACTIVATION"
    try:
        _eager_runtime_imports(authority["runtime"], authority["runtime_root"])
        phase = "POSTIMPORT_RUNTIME_CHECKPOINT"
        _trust_checkpoint(authority, guard, "POSTIMPORT")
        phase = "LOAD_FROZEN_SCIENCE"
        r6, wrapper, numerical, readiness, state = _load_worker_science(authority)
        phase = "PRECLAIM_RUNTIME_CHECKPOINT"
        _trust_checkpoint(authority, guard, "PRECLAIM")
        allowed = {SEAL_PATH.name, LAUNCH_CLAIM_PATH.name, "worker_start.lease"}
        actual = {path.name for path in CANONICAL_ARTIFACT_DIR.iterdir()}
        if actual != allowed:
            raise FileExistsError("canonical one-shot namespace changed before claim")
        deadline_epoch = int(authority["manifest"]["deadline_epoch_ns"]) / 1_000_000_000
        journal = r6.AttemptJournal.begin(
            CANONICAL_ARTIFACT_DIR,
            deadline_epoch,
            snapshot_manifest_sha256=authority["manifest_sha256"],
        )
        phase = "READINESS_RECORDED"
        journal.record_readiness(readiness)

        def checkpoint(name: str) -> None:
            _trust_checkpoint(authority, guard, name)

        guarded_journal = FirstFitTrustJournal(journal, checkpoint)
        closure_raw, _receipt = _read_bound_bytes(
            r6.CLOSURE_V3_PATH,
            expected_sha256=r6.CLOSURE_V3_SHA256,
        )
        closure = _json_from_bytes(closure_raw, label=r6.CLOSURE_V3_PATH.name)
        phase = "FIXED_72_FIT_21_MATERIALIZATION_SCREEN"
        screen = wrapper.run_authorized_screen(
            state,
            numerical,
            closure,
            guarded_journal,
            deadline_epoch,
        )
        phase = "AGGREGATE_ONLY_SUCCESS_PUBLICATION"
        return _publish_success(screen, journal, authority, r6)
    except BaseException as error:
        if journal is not None and journal.lock_path.exists():
            try:
                journal.fail_terminal(
                    phase,
                    error,
                    provenance={
                        "snapshot": "PRIVATE_HELD_TREE",
                        "runtime": "ISOLATED_HELD_TREE",
                        "launch_claim_sha256": _sha256(LAUNCH_CLAIM_PATH),
                    },
                )
            except BaseException as terminal_error:
                journal.close_handle_keep_lock()
                raise RuntimeError(
                    "claimed failure terminal publication failed; lock retained"
                ) from terminal_error
        raise
    finally:
        for handle in reversed(dll_handles):
            try:
                handle.close()
            except BaseException:
                pass


def _worker_command(
    snapshot_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    claim_sha256: str,
    manifest: Mapping[str, Any],
) -> list[str]:
    runner = snapshot_root / "scripts/run_p1_long_event_segment_proposal_rescore_v7.py"
    return [
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(runner),
        "--worker",
        "--capability-manifest",
        str(manifest_path),
        "--capability-manifest-sha256",
        manifest_sha256,
        "--bootstrap-sha256",
        _sha256(runner),
        "--launch-claim-sha256",
        claim_sha256,
        "--launch-nonce",
        manifest["launch_nonce"],
        "--deadline-epoch-ns",
        str(manifest["deadline_epoch_ns"]),
        "--parent-pid",
        str(manifest["parent_identity"]["pid"]),
        "--authorization-sha256",
        manifest["authorization_sha256"],
        "--seal-sha256",
        manifest["seal_sha256"],
        "--qa-sha256",
        manifest["qa_sha256"],
        "--snapshot-inventory-sha256",
        manifest["snapshot_inventory_binding"]["sha256"],
        "--runtime-inventory-sha256",
        manifest["runtime_inventory_binding"]["sha256"],
        "--canonical-project-root",
        manifest["canonical_project_root"],
        "--canonical-artifact-relative-path",
        manifest["canonical_artifact_relative_path"],
        "--single-attempt-namespace",
        manifest["single_attempt_namespace"],
    ]


def _run_supervised(
    command: Sequence[str],
    deadline_epoch_ns: int,
    snapshot_root: Path,
) -> tuple[str, str]:
    r6 = _load_r6_runner(
        snapshot_root / "scripts/run_p1_long_event_segment_proposal_rescore_v6.py"
    )
    remaining = (deadline_epoch_ns - time.time_ns()) / 1_000_000_000
    if remaining <= 0 or remaining > HARD_WALL_SECONDS:
        raise TimeoutError("fixed parent deadline is unavailable before worker spawn")
    kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(list(command), **kwargs)
    try:
        stdout, stderr = process.communicate(timeout=remaining)
    except subprocess.TimeoutExpired as error:
        try:
            termination = r6._terminate_process_tree(process)
        except BaseException as termination_error:
            raise RuntimeError(
                "fixed 21600-second timeout could not prove worker-tree termination"
            ) from termination_error
        timeout_error = TimeoutError(
            "fixed 21600-second monotonic communicate timeout terminated worker tree"
        )
        timeout_error.termination_receipt = termination
        raise timeout_error from error
    if process.returncode != 0:
        raise RuntimeError(
            f"isolated worker exited {process.returncode}: {stderr[-4000:]}"
        )
    return stdout, stderr


def _record_parent_failure(phase: str, error: BaseException) -> None:
    lock = CANONICAL_ARTIFACT_DIR / "execution.lock"
    if lock.exists():
        r6 = _load_r6_runner()
        r6.EXPERIMENT_ID = EXPERIMENT_ID
        r6.ARTIFACT_DIR = CANONICAL_ARTIFACT_DIR
        r6.SEAL_PATH = SEAL_PATH
        r6.OUTPUT_PROJECT_ROOT = CANONICAL_PROJECT_ROOT
        r6._record_parent_failure_if_claimed(
            phase,
            error,
            provenance={"canonical_launch_claim_sha256": _sha256(LAUNCH_CLAIM_PATH)},
        )
        return
    terminal = CANONICAL_ARTIFACT_DIR / "parent_worker_failed_before_scientific_claim.json"
    message = str(error)
    for root, token in (
        (CANONICAL_PROJECT_ROOT, "<PROJECT_ROOT>"),
        (Path.home(), "<USER_HOME>"),
        (Path(tempfile.gettempdir()), "<TEMP_ROOT>"),
    ):
        message = message.replace(str(root), token)
    _atomic_create_json(
        terminal,
        {
            "schema_version": "p1_segment_rescore.parent_preclaim_failed.v7",
            "experiment_id": EXPERIMENT_ID,
            "status": "FAILED_AFTER_CANONICAL_LAUNCH_CLAIM_REPLAY_BLOCKED",
            "phase": phase,
            "error": {"type": type(error).__name__, "message": message[:500]},
            "canonical_launch_claim_sha256": _sha256(LAUNCH_CLAIM_PATH),
            "physical_fits": 0,
            "scientific_materializations": 0,
            "outer_scores": 0,
            "created_at_kst": _now_kst(),
        },
    )


def _parent_read_only_verify(result_path: Path) -> dict[str, Any]:
    if (CANONICAL_ARTIFACT_DIR / "execution.lock").exists():
        raise RuntimeError("success returned while execution lock remains")
    expected = CANONICAL_ARTIFACT_DIR / "result.json"
    if result_path.resolve(strict=True) != expected.resolve(strict=True):
        raise RuntimeError("worker returned a noncanonical result path")
    result, _raw, result_receipt = _read_bound_json(result_path)
    manifest, _raw, manifest_receipt = _read_bound_json(
        CANONICAL_ARTIFACT_DIR / "manifest.json"
    )
    r6 = _load_r6_runner()
    previous, entries = r6._verify_journal_chain(
        CANONICAL_ARTIFACT_DIR / "attempt_journal",
        required_last="0999_completed.json",
    )
    completed = entries[-1]
    if (
        completed.get("status")
        != "SUCCESS_ALL_OUTPUTS_VERIFIED_READY_FOR_LOCK_RELEASE"
        or completed.get("result_sha256") != result_receipt["sha256"]
        or completed.get("manifest_sha256") != manifest_receipt["sha256"]
    ):
        raise RuntimeError("completed terminal/output hashes changed")
    for relative, record in manifest["artifacts"].items():
        if _file_record(CANONICAL_PROJECT_ROOT / relative) != record:
            raise RuntimeError(f"parent artifact verification failed: {relative}")
    reservations = [
        item
        for item in entries
        if item.get("schema_version") == "p1_segment_rescore.fit_reserved.v2"
    ]
    completions = [
        item
        for item in entries
        if item.get("schema_version") == "p1_segment_rescore.fit_completed.v2"
    ]
    materializations = [
        item
        for item in entries
        if item.get("schema_version")
        == "p1_segment_rescore.materialization_reserved.v2"
    ]
    if (len(reservations), len(completions), len(materializations)) != (72, 72, 21):
        raise RuntimeError("journal operation accounting changed")
    if result["operation_counters"] != manifest["operation_counters"]:
        raise RuntimeError("result/manifest operation counters differ")
    if result["operation_counters"]["physical_fits"] != 72:
        raise RuntimeError("result does not contain exactly 72 physical fits")
    return {
        "status": "PASS_PARENT_READ_ONLY_DURABLE_SUCCESS_QA",
        "result_sha256": result_receipt["sha256"],
        "manifest_sha256": manifest_receipt["sha256"],
        "completed_terminal_sha256": previous,
        "physical_fit_reservations": 72,
        "physical_fits_completed": 72,
        "scientific_materializations": 21,
        "outer_scores": 1,
    }


def _recover_committed_success_after_supervision_error(
    error: BaseException,
) -> tuple[Path, dict[str, Any]] | None:
    if (CANONICAL_ARTIFACT_DIR / "execution.lock").exists():
        return None
    result = CANONICAL_ARTIFACT_DIR / "result.json"
    manifest = CANONICAL_ARTIFACT_DIR / "manifest.json"
    completed = CANONICAL_ARTIFACT_DIR / "attempt_journal/0999_completed.json"
    states = (result.is_file(), manifest.is_file(), completed.is_file())
    if not any(states):
        return None
    if not all(states):
        raise RuntimeError("lock-free namespace contains partial success evidence") from error
    verification = _parent_read_only_verify(result)
    verification = dict(verification)
    verification["status"] = "PASS_RECOVERED_DURABLE_SUCCESS_AFTER_STDOUT_LOSS"
    verification["recovery_writes"] = 0
    return result, verification


def _execute_parent_held(
    authorization: Mapping[str, Any],
    authorization_raw: bytes,
    authorization_receipt: Mapping[str, Any],
    seal_value: Mapping[str, Any],
    seal_raw: bytes,
    seal_receipt: Mapping[str, Any],
    qa_raw: bytes,
    qa_receipt: Mapping[str, Any],
    authority_source_file_ids: set[str],
) -> tuple[Path, dict[str, Any]]:
    runtime = seal_value["full_runtime_inventory"]
    snapshot, bindings = _snapshot_static_inventory()
    source_paths = set(_runtime_source_paths(runtime))
    source_paths.update(_snapshot_source_paths(snapshot, bindings))
    bundle: Path | None = None
    with HeldReadDenyMutation(tuple(source_paths)) as source_guard:
        for _relative, record in runtime["files"].items():
            source_guard.assert_record(
                Path(record["source_identity"]["resolved_path"]),
                record,
                require_source_identity=True,
            )
        for record in runtime["excluded_nonimport_cli"]:
            source_guard.assert_record(
                Path(record["source_identity"]["resolved_path"]),
                record,
                require_source_identity=True,
            )
        for record in runtime["host_files"].values():
            source_guard.assert_record(
                Path(record["resolved_source_path"]),
                record,
                require_source_identity=True,
            )
        for relative, record in snapshot.items():
            raw_source = bindings[relative]["source_path"]
            source = Path(raw_source)
            if not source.is_absolute():
                source = CANONICAL_PROJECT_ROOT / source
            source_record = dict(record)
            source_record["source_identity"] = bindings[relative]["source_identity"]
            source_guard.assert_record(
                source,
                source_record,
                require_source_identity=True,
            )
        bundle, snapshot_root, snapshot_inventory, runtime_state = (
            _copy_snapshot_and_runtime(
                seal_value,
                authorization_raw,
                seal_raw,
                qa_raw,
                source_guard,
            )
        )
        runtime_root = runtime_state["runtime_root"]
        created_ns = time.time_ns()
        deadline_ns = created_ns + HARD_WALL_SECONDS * 1_000_000_000
        parent_identity = _process_identity(os.getpid())
        launch_nonce = secrets.token_hex(32)
        manifest = _manifest_value(
            snapshot_root=snapshot_root,
            runtime_root=runtime_root,
            snapshot_inventory=snapshot_inventory,
            runtime_inventory_binding=seal_value["runtime_inventory_binding"],
            authorization_sha256=authorization_receipt["sha256"],
            seal_sha256=seal_receipt["sha256"],
            qa_sha256=qa_receipt["sha256"],
            created_epoch_ns=created_ns,
            deadline_epoch_ns=deadline_ns,
            parent_identity=parent_identity,
            launch_nonce=launch_nonce,
        )
        manifest_path = bundle / "capability_manifest.json"
        _atomic_create_json(manifest_path, manifest)
        manifest_sha = _sha256(manifest_path)
        destination_paths = [
            *(snapshot_root / relative for relative in snapshot_inventory),
            *(runtime_root / relative for relative in runtime["files"]),
            manifest_path,
        ]
        outcome: tuple[Path, dict[str, Any]] | None = None
        with HeldReadDenyMutation(destination_paths) as destination_guard:
            if len(destination_guard.file_ids()) != len(destination_paths):
                raise RuntimeError("private bundle contains a duplicate destination FileId")
            if destination_guard.file_ids() & (
                source_guard.file_ids() | authority_source_file_ids
            ):
                raise RuntimeError("private runtime/snapshot used a source hardlink")
            for relative, record in snapshot_inventory.items():
                destination_guard.assert_record(snapshot_root / relative, record)
            for relative, record in runtime["files"].items():
                destination_guard.assert_record(runtime_root / relative, record)
            destination_guard.assert_record(
                manifest_path,
                {"bytes": manifest_path.stat().st_size, "sha256": manifest_sha},
            )
            claim = _launch_claim_value(manifest, manifest_sha)
            _atomic_create_json(LAUNCH_CLAIM_PATH, claim)
            claim_sha = _sha256(LAUNCH_CLAIM_PATH)
            command = _worker_command(
                snapshot_root,
                manifest_path,
                manifest_sha,
                claim_sha,
                manifest,
            )
            try:
                stdout, _stderr = _run_supervised(command, deadline_ns, snapshot_root)
                lines = [line for line in stdout.splitlines() if line.strip()]
                if not lines:
                    raise RuntimeError("worker returned no receipt")
                receipt = _json_from_bytes(lines[-1].encode("utf-8"), label="worker stdout")
                if receipt.get("status") != "worker_ok":
                    raise RuntimeError("worker success receipt changed")
                result_path = Path(str(receipt["result_path"]))
                outcome = (result_path, _parent_read_only_verify(result_path))
            except BaseException as error:
                recovered = _recover_committed_success_after_supervision_error(error)
                if recovered is not None:
                    outcome = recovered
                else:
                    _record_parent_failure("PARENT_WORKER_SUPERVISION", error)
                    raise
        if outcome is None:
            raise AssertionError("worker supervision escaped without an outcome")
        shutil.rmtree(bundle)
        bundle = None
        return outcome
    raise AssertionError("parent held execution escaped without a result")


def _execute_parent_authenticated() -> tuple[Path, dict[str, Any]]:
    supplied = os.environ.get(AUTHORIZATION_ENV_VAR, "")
    _exact_digest(supplied, "external authorization environment")
    if BOOTSTRAP_PROJECT_ROOT != CANONICAL_PROJECT_ROOT.resolve(strict=True):
        raise AuthorizationError("parent execute requires canonical live runner")
    if AUTHORIZATION_PATH.exists() is False:
        raise AuthorizationError("actual authorization does not exist")
    if set(CANONICAL_ARTIFACT_DIR.iterdir()) != {SEAL_PATH}:
        raise FileExistsError("canonical one-shot namespace is not pristine")
    with HeldReadDenyMutation([AUTHORIZATION_PATH]) as auth_guard:
        authorization_raw = auth_guard.read_bytes(AUTHORIZATION_PATH)
        if _sha256_bytes(authorization_raw) != supplied:
            raise AuthorizationError("external digest differs from held authorization bytes")
        authorization = _require_authorization_schema(
            _json_from_bytes(authorization_raw, label=AUTHORIZATION_PATH.name)
        )
        authorization_receipt = {
            "bytes": len(authorization_raw),
            "sha256": supplied,
        }
        with HeldReadDenyMutation([SEAL_PATH, R7_QA_PATH]) as authority_guard:
            seal_raw = authority_guard.read_bytes(SEAL_PATH)
            qa_raw = authority_guard.read_bytes(R7_QA_PATH)
            seal_receipt = {"bytes": len(seal_raw), "sha256": _sha256_bytes(seal_raw)}
            qa_receipt = {"bytes": len(qa_raw), "sha256": _sha256_bytes(qa_raw)}
            seal_value = _json_from_bytes(seal_raw, label=SEAL_PATH.name)
            _verify_seal_value(seal_value, seal_receipt)
            qa_value = _json_from_bytes(qa_raw, label=R7_QA_PATH.name)
            _validate_authorization_values(
                authorization,
                seal_value,
                seal_raw,
                qa_value,
                qa_raw,
            )
            return _execute_parent_held(
                authorization,
                authorization_raw,
                authorization_receipt,
                seal_value,
                seal_raw,
                seal_receipt,
                qa_raw,
                qa_receipt,
                auth_guard.file_ids() | authority_guard.file_ids(),
            )


def execute_parent() -> tuple[Path, dict[str, Any]]:
    try:
        return _execute_parent_authenticated()
    except BaseException as error:
        recovered = _recover_committed_success_after_supervision_error(error)
        if recovered is not None:
            return recovered
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--seal", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--capability-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--capability-manifest-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--launch-claim-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--launch-nonce", help=argparse.SUPPRESS)
    parser.add_argument("--deadline-epoch-ns", help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", help=argparse.SUPPRESS)
    parser.add_argument("--authorization-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--seal-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--qa-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-inventory-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--runtime-inventory-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--canonical-project-root", help=argparse.SUPPRESS)
    parser.add_argument("--canonical-artifact-relative-path", help=argparse.SUPPRESS)
    parser.add_argument("--single-attempt-namespace", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.seal:
        output: Any = str(seal())
    elif args.preflight:
        output = read_only_preflight()
    elif args.execute:
        path, verification = execute_parent()
        output = {"result_path": str(path), "parent_verification": verification}
    else:
        required = (
            "capability_manifest",
            "capability_manifest_sha256",
            "bootstrap_sha256",
            "launch_claim_sha256",
            "launch_nonce",
            "deadline_epoch_ns",
            "parent_pid",
            "authorization_sha256",
            "seal_sha256",
            "qa_sha256",
            "snapshot_inventory_sha256",
            "runtime_inventory_sha256",
            "canonical_project_root",
            "canonical_artifact_relative_path",
            "single_attempt_namespace",
        )
        if any(getattr(args, key) is None for key in required):
            raise AuthorizationError("hidden worker requires complete exact capability")
        result = _worker_execute(args)
        print(
            json.dumps(
                {"status": "worker_ok", "result_path": str(result)},
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return
    print(json.dumps({"status": "ok", "output": output}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
