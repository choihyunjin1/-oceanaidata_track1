"""Noncyclic authenticated bootstrap for P1 multiscale Gen6r2.

The independent QA receipt is the trust root for this bootstrap.  This file
pins every subordinate owner role; no subordinate file pins this bootstrap.
"""

# ruff: noqa: E402 -- import order is the enforced pre-import trust boundary.

from __future__ import annotations

import sys

# Nothing backed by a filesystem path may execute before this boundary.  These
# three modules are frozen/builtin in the exact CPython runtime pinned below.
if not (
    sys.flags.isolated == 1
    and sys.flags.no_site == 1
    and sys.flags.dont_write_bytecode == 1
    and sys.flags.ignore_environment == 1
    and sys.flags.safe_path is True
):
    raise RuntimeError("canonical bootstrap requires python -I -S -B")
if len(sys.argv) != 2 or sys.argv[1] not in {"--check-only", "--execute"}:
    raise RuntimeError("canonical bootstrap accepts exactly one explicit mode")

_BOOT_MODE = "check-only" if sys.argv[1] == "--check-only" else "execute"
_EARLY_AUDIT_EVENTS: list[tuple[str, tuple[object, ...]]] = []
_EARLY_WRITES = 0
_NUMERICAL_IMPORTS_ENABLED = False
_THIRD_PARTY_ROOTS = {
    "dateutil",
    "joblib",
    "narwhals",
    "numpy",
    "pandas",
    "psutil",
    "pyarrow",
    "scipy",
    "six",
    "sklearn",
    "threadpoolctl",
    "tzdata",
}
_THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _earliest_audit_hook(event: str, arguments: tuple[object, ...]) -> None:
    """Install before every path-backed import and retain a fail-closed trace."""

    global _EARLY_WRITES
    if event == "import":
        name = arguments[0] if arguments else ""
        if (
            type(name) is str
            and name.split(".", 1)[0] in _THIRD_PARTY_ROOTS
            and not _NUMERICAL_IMPORTS_ENABLED
        ):
            raise PermissionError("third-party import preceded runtime authentication")
    if event == "open":
        mode = arguments[1] if len(arguments) > 1 else "r"
        flags = arguments[2] if len(arguments) > 2 else 0
        writable = (type(mode) is str and any(character in mode for character in "wax+")) or (
            type(flags) is int and bool(flags & (1 | 2 | 8 | 256 | 512))
        )
        if writable and _BOOT_MODE == "check-only":
            _EARLY_WRITES += 1
            raise PermissionError("check-only write forbidden from process start")
    elif event in {
        "os.remove",
        "os.rename",
        "os.replace",
        "os.rmdir",
        "os.mkdir",
        "os.symlink",
        "os.link",
        "subprocess.Popen",
        "socket.connect",
    }:
        if _BOOT_MODE == "check-only" or event in {"subprocess.Popen", "socket.connect"}:
            _EARLY_WRITES += 1
            raise PermissionError(f"forbidden bootstrap side effect: {event}")
    if len(_EARLY_AUDIT_EVENTS) < 4096 and event in {"import", "open"}:
        _EARLY_AUDIT_EVENTS.append((event, arguments[:3]))


sys.addaudithook(_earliest_audit_hook)

import _sha2
import os
import stat

if (
    getattr(os.__spec__, "origin", None) != "frozen"
    or getattr(stat.__spec__, "origin", None) != "frozen"
    or getattr(_sha2.__spec__, "origin", None) != "built-in"
):
    raise RuntimeError("earliest runtime modules are not frozen/builtin")

sys.dont_write_bytecode = True

_BASE_PREFIX = os.path.abspath(sys.base_prefix)
_EXPECTED_STARTUP_SYS_PATH = (
    f"{_BASE_PREFIX}\\python312.zip",
    f"{_BASE_PREFIX}\\DLLs",
    f"{_BASE_PREFIX}\\Lib",
    _BASE_PREFIX,
)
_EXPECTED_META_PATH = (
    ("_frozen_importlib", "BuiltinImporter"),
    ("_frozen_importlib", "FrozenImporter"),
    ("_frozen_importlib_external", "PathFinder"),
)


def _meta_path_identity(item: object) -> tuple[str, str]:
    target = item if isinstance(item, type) else type(item)
    return target.__module__, target.__qualname__


_EARLY_RUNTIME_PINS = {
    "python312.dll": (6945272, "9a0e3435aaa680d868150f87ab3e388ad2eebc22f87e036155c7b4eda8cd2120"),
    "python3.dll": (70376, "fb975a606e7fbf74f64260e3f60c3490b4f74a183c0926fd6ed1ac4c52ac7b1c"),
    "vcruntime140.dll": (
        120400,
        "052ad6a20d375957e82aa6a3c441ea548d89be0981516ca7eb306e063d5027f4",
    ),
}
_STDLIB_SNAPSHOTS = {
    "Lib": {
        "files": 2397,
        "bytes": 43971009,
        "sha256": "81c5fbf6d2e0d268b9f8b2193a778e22393d9b8d4cded3dc7743eee659a3e519",
    },
    "DLLs": {
        "files": 42,
        "bytes": 15330590,
        "sha256": "f552f359861f2ce695d3c8e28af427c02571455e2b85e73bddf716fdf19b3500",
    },
}
_RUNTIME_FILE_BYTES: dict[str, bytes] = {}
_RUNTIME_FILE_PINS: dict[str, tuple[int, str]] = {}
_HELD_NATIVE_DESCRIPTORS: list[int] = []
_HELD_NATIVE_IDENTITIES: dict[str, tuple[int, int, int]] = {}


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _early_plain_file(path: str) -> tuple[bytes, tuple[int, int, int, int, int]]:
    absolute = _norm(path)
    before = os.lstat(absolute)
    attributes = int(getattr(before, "st_file_attributes", 0))
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or bool(attributes & 0x400):
        raise RuntimeError("early runtime file identity differs")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        opened = os.fstat(descriptor)
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            blocks.append(block)
        closed_identity = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(absolute)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_mode)
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
    ) or (opened.st_dev, opened.st_ino, opened.st_size) != (
        closed_identity.st_dev,
        closed_identity.st_ino,
        closed_identity.st_size,
    ):
        raise RuntimeError("early runtime file changed during authenticated read")
    payload = b"".join(blocks)
    if len(payload) != before.st_size:
        raise RuntimeError("early runtime file length differs")
    return payload, identity


def _early_assert_pin(path: str, expected: tuple[int, str], *, retain_native: bool) -> bytes:
    payload, _identity = _early_plain_file(path)
    if len(payload) != expected[0] or _sha2.sha256(payload).hexdigest() != expected[1]:
        raise RuntimeError(f"early runtime pin differs: {path}")
    normalized = _norm(path)
    _RUNTIME_FILE_PINS[normalized] = expected
    if path.casefold().endswith(".py"):
        _RUNTIME_FILE_BYTES[normalized] = payload
    if retain_native:
        descriptor = os.open(normalized, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        _HELD_NATIVE_DESCRIPTORS.append(descriptor)
        opened = os.fstat(descriptor)
        _HELD_NATIVE_IDENTITIES[normalized] = (
            int(opened.st_dev),
            int(opened.st_ino),
            int(opened.st_size),
        )
    return payload


def _authenticate_stdlib_snapshot(root: str, expected: dict[str, object]) -> None:
    rows: list[str] = []
    total = 0
    files = 0
    for base, directories, names in os.walk(root):
        directories[:] = sorted(
            name
            for name in directories
            if name != "__pycache__"
            and not (
                root.casefold().endswith("\\lib")
                and os.path.relpath(os.path.join(base, name), root)
                .replace("\\", "/")
                .split("/", 1)[0]
                == "site-packages"
            )
        )
        for name in sorted(names):
            if name.casefold().endswith((".pyc", ".pyo")):
                continue
            path = os.path.join(base, name)
            payload, _identity = _early_plain_file(path)
            relative = os.path.relpath(path, root).replace("\\", "/")
            digest = _sha2.sha256(payload).hexdigest()
            rows.append(f"{relative}\0{len(payload)}\0{digest}\n")
            normalized = _norm(path)
            _RUNTIME_FILE_PINS[normalized] = (len(payload), digest)
            if name.casefold().endswith(".py"):
                _RUNTIME_FILE_BYTES[normalized] = payload
            if name.casefold().endswith((".pyd", ".dll")):
                descriptor = os.open(normalized, os.O_RDONLY | getattr(os, "O_BINARY", 0))
                _HELD_NATIVE_DESCRIPTORS.append(descriptor)
                opened = os.fstat(descriptor)
                _HELD_NATIVE_IDENTITIES[normalized] = (
                    int(opened.st_dev),
                    int(opened.st_ino),
                    int(opened.st_size),
                )
            total += len(payload)
            files += 1
    snapshot = _sha2.sha256("".join(sorted(rows)).encode("utf-8")).hexdigest()
    if (
        type(expected.get("files")) is not int
        or files != expected["files"]
        or type(expected.get("bytes")) is not int
        or total != expected["bytes"]
        or snapshot != expected.get("sha256")
    ):
        raise RuntimeError(f"stdlib snapshot differs: {root}")


_RAW_WORKSPACE = os.environ.get("P1_WORKSPACE_ROOT")
if not _RAW_WORKSPACE or not os.path.isabs(_RAW_WORKSPACE):
    raise RuntimeError("P1_WORKSPACE_ROOT must be present and absolute before runtime trust")
_EARLY_WORKSPACE = _norm(_RAW_WORKSPACE)
_EXPECTED_PYCACHE = _norm(os.path.join(_EARLY_WORKSPACE, "artifacts", "p1_v6r2_forbidden_pycache"))
if _norm(sys.pycache_prefix or "") != _EXPECTED_PYCACHE or os.path.lexists(_EXPECTED_PYCACHE):
    raise RuntimeError("canonical absent -X pycache_prefix boundary differs")
if _norm(sys.base_prefix) != _norm(_BASE_PREFIX) or tuple(map(_norm, sys.path)) != tuple(
    map(_norm, _EXPECTED_STARTUP_SYS_PATH)
):
    raise RuntimeError("canonical isolated startup path differs")
if tuple(_meta_path_identity(item) for item in sys.meta_path) != _EXPECTED_META_PATH:
    raise RuntimeError("canonical startup meta_path differs")
_EXPECTED_EXECUTABLE = _norm(os.path.join(_EARLY_WORKSPACE, ".venv-p1", "Scripts", "python.exe"))
_early_assert_pin(
    _EXPECTED_EXECUTABLE,
    (274424, "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14"),
    retain_native=True,
)
if _norm(sys.executable) != _EXPECTED_EXECUTABLE:
    raise RuntimeError("canonical venv interpreter path differs")
_early_assert_pin(
    os.path.join(_EARLY_WORKSPACE, ".venv-p1", "pyvenv.cfg"),
    (339, "d1fb970854073922d49959ae01539088550613e316cb67f9fac858f586361174"),
    retain_native=False,
)
for _name, _expected in _EARLY_RUNTIME_PINS.items():
    _early_assert_pin(os.path.join(_BASE_PREFIX, _name), _expected, retain_native=True)
for _name, _expected in _STDLIB_SNAPSHOTS.items():
    _authenticate_stdlib_snapshot(os.path.join(_BASE_PREFIX, _name), _expected)

# Only frozen import machinery is used to install loaders that serve verified
# Python source from the snapshot buffers and reject all bytecode caches.
import _frozen_importlib_external


class _AuthenticatedSourceLoader(_frozen_importlib_external.SourceFileLoader):
    def get_data(self, path: str) -> bytes:
        normalized = _norm(path)
        if normalized.casefold().endswith((".pyc", ".pyo")):
            raise OSError("bytecode loading is forbidden")
        try:
            return _RUNTIME_FILE_BYTES[normalized]
        except KeyError as exc:
            raise ImportError(f"source is outside authenticated runtime: {path}") from exc

    def set_data(self, path: str, data: bytes, *_args: object, **_kwargs: object) -> None:
        del path, data
        return None


class _AuthenticatedExtensionLoader(_frozen_importlib_external.ExtensionFileLoader):
    def create_module(self, spec: object) -> object:
        normalized = _norm(self.path)
        current = os.lstat(normalized)
        if (
            normalized not in _RUNTIME_FILE_PINS
            or normalized not in _HELD_NATIVE_IDENTITIES
            or _HELD_NATIVE_IDENTITIES[normalized]
            != (int(current.st_dev), int(current.st_ino), int(current.st_size))
        ):
            raise ImportError(f"native extension is outside authenticated runtime: {self.path}")
        return super().create_module(spec)

    def exec_module(self, module: object) -> None:
        normalized = _norm(self.path)
        current = os.lstat(normalized)
        if _HELD_NATIVE_IDENTITIES.get(normalized) != (
            int(current.st_dev),
            int(current.st_ino),
            int(current.st_size),
        ):
            raise ImportError(f"native extension identity changed: {self.path}")
        super().exec_module(module)
        after = os.lstat(normalized)
        if _HELD_NATIVE_IDENTITIES.get(normalized) != (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
        ):
            raise ImportError(f"native extension changed during load: {self.path}")


def _trusted_path_hook(path: str) -> object:
    normalized = _norm(path)
    trusted_roots = (
        _norm(os.path.join(_BASE_PREFIX, "Lib")),
        _norm(os.path.join(_BASE_PREFIX, "DLLs")),
        _norm(_BASE_PREFIX),
        _norm(os.path.join(_EARLY_WORKSPACE, ".venv-p1", "Lib", "site-packages")),
    )
    if not any(
        normalized == root or normalized.startswith(root + os.sep) for root in trusted_roots
    ):
        raise ImportError
    return _frozen_importlib_external.FileFinder(
        path,
        (_AuthenticatedSourceLoader, _frozen_importlib_external.SOURCE_SUFFIXES),
        (_AuthenticatedExtensionLoader, _frozen_importlib_external.EXTENSION_SUFFIXES),
    )


sys.path_hooks.insert(0, _trusted_path_hook)
for _entry in tuple(sys.path_importer_cache):
    _normalized = _norm(_entry)
    if any(
        _normalized == _norm(root) or _normalized.startswith(_norm(root) + os.sep)
        for root in (
            _BASE_PREFIX,
            os.path.join(_BASE_PREFIX, "Lib"),
            os.path.join(_BASE_PREFIX, "DLLs"),
        )
    ):
        sys.path_importer_cache.pop(_entry, None)

import argparse
import base64
import csv
import hashlib
import io
import json
import types
from pathlib import Path, PurePosixPath
from typing import Any

GENERATION = "p1_multiscale_cross_layer_offset_drift_unary_v6r2"
BOOTSTRAP_RELATIVE = "scripts/bootstrap_p1_multiscale_cross_layer_offset_drift_unary_v6r2.py"
QA_SCHEMA = "p1_multiscale_cross_layer_offset_drift_unary.v6r2.independent_qa.v1"
AUTH_SCHEMA = "p1_multiscale_cross_layer_offset_drift_unary.v6r2.execution_authorization.v1"
AUTHORIZATION_PHRASE = (
    "AUTHORIZE_P1_MULTISCALE_CROSS_LAYER_OFFSET_DRIFT_V6R2_ONE_SHOT_RESEARCH_CURVE_ONLY"
)
REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
O_BINARY = getattr(os, "O_BINARY", 0)
LOWER_SHA = set("0123456789abcdef")

# Fresh independent QA pins this bootstrap.  The bootstrap owns only this
# noncyclic subordinate pin map, which is filled after static verification.
PINNED_OWNER_ROLES: dict[str, dict[str, Any]] = {
    "CONFIG": {
        "path": "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r2.json",
        "bytes": 21292,
        "sha256": "5343b6d9a15ac7e0b2728b30f84db5974431b80070ea8519d51d9bfd8ad1dc12",
    },
    "SCIENCE_PROJECTION": {
        "path": "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r2_science_projection.json",
        "bytes": 9537,
        "sha256": "c8bc59c7dc78568fbf79a54a8dfdbfe799242d44cfa31d155b31000d4cafcaef",
    },
    "SCIENCE": {
        "path": "src/p1_qc/multiscale_cross_layer_offset_drift_v6r2.py",
        "bytes": 70846,
        "sha256": "4b5c74aeb54406416cda09576fac1d7e1569c6cbc46534ad92a9c2ea154a03af",
    },
    "CONTRACT": {
        "path": "src/p1_qc/multiscale_cross_layer_offset_drift_contract_v6r2.py",
        "bytes": 63371,
        "sha256": "b8867e61a54855d78bd3c85ed0bc88f2cce9d51d63be661b111c4136dd3d2bdb",
    },
    "ENGINE": {
        "path": "src/p1_qc/multiscale_cross_layer_offset_drift_execution_v6r2.py",
        "bytes": 63055,
        "sha256": "4d46d30cb1895a952e2925cbf6811609d776d133528dd43ff06f520a276fffc9",
    },
    "RUNNER": {
        "path": "scripts/run_p1_multiscale_cross_layer_offset_drift_unary_v6r2.py",
        "bytes": 2609,
        "sha256": "2d7e1314f4964230189f70505df026b135cb0f348e4dcaab4462b1e25c646817",
    },
    "TESTS": {
        "path": "tests/test_run_p1_multiscale_cross_layer_offset_drift_unary_v6r2.py",
        "bytes": 25232,
        "sha256": "7e1787677905cfe8cd2ec9710fa6a75f42f6a0497d2edc461f6ef813835822b6",
    },
}

PINNED_DISTRIBUTION_RECORDS: dict[str, dict[str, Any]] = {
    "joblib": {
        "path": ".venv-p1/Lib/site-packages/joblib-1.5.3.dist-info/RECORD",
        "bytes": 18933,
        "sha256": "efce4d32af613f56006d66133a1c038922e6f4b2206459686c77a52343e88b4e",
    },
    "narwhals": {
        "path": ".venv-p1/Lib/site-packages/narwhals-2.24.0.dist-info/RECORD",
        "bytes": 23282,
        "sha256": "8bfbfcd69a78d0de565d4d955d0587920363e30a03c95ff33a182a2ac0b6a088",
    },
    "numpy": {
        "path": ".venv-p1/Lib/site-packages/numpy-2.3.5.dist-info/RECORD",
        "bytes": 110571,
        "sha256": "28e9a4fcb2fa550a51a5c6f6639c2a5d3aed11407aded2cb8981747eb9640ca9",
    },
    "pandas": {
        "path": ".venv-p1/Lib/site-packages/pandas-3.0.1.dist-info/RECORD",
        "bytes": 249268,
        "sha256": "98e530a3a2f22b4865b342652b94846b0d75b4aa7590f7a53ba6aea322e0d0e2",
    },
    "psutil": {
        "path": ".venv-p1/Lib/site-packages/psutil-7.2.2.dist-info/RECORD",
        "bytes": 1875,
        "sha256": "55fd2f55e72c18fd0017a0a033af4661d0227e339c5d772a40a29375e6f740d7",
    },
    "pyarrow": {
        "path": ".venv-p1/Lib/site-packages/pyarrow-25.0.1.dist-info/RECORD",
        "bytes": 78570,
        "sha256": "1eddf4fb72b1b071868dc02d6fc8242125d98c6557ae6af8f783b1c84ef6a797",
    },
    "python-dateutil": {
        "path": ".venv-p1/Lib/site-packages/python_dateutil-2.9.0.post0.dist-info/RECORD",
        "bytes": 3125,
        "sha256": "0c26b4b1542dbd1ebd8d2babdd501aed583d6ada9595517f936f00fe4ff9d254",
    },
    "scikit-learn": {
        "path": ".venv-p1/Lib/site-packages/scikit_learn-1.9.0.dist-info/RECORD",
        "bytes": 151140,
        "sha256": "641dc828354fda88576c8c34bc13c7991bf66a13fa9be1a8496f0dd2a687c0f9",
    },
    "scipy": {
        "path": ".venv-p1/Lib/site-packages/scipy-1.18.0.dist-info/RECORD",
        "bytes": 211030,
        "sha256": "61107cec4e3c47006ac4d28012680573adeef02d32a6472084dd96092543f579",
    },
    "six": {
        "path": ".venv-p1/Lib/site-packages/six-1.17.0.dist-info/RECORD",
        "bytes": 561,
        "sha256": "d834e846ba51c0e7371968d0b5a0cdebdaa2f9ea2f0447a40b594fa96ca5d89f",
    },
    "threadpoolctl": {
        "path": ".venv-p1/Lib/site-packages/threadpoolctl-3.6.0.dist-info/RECORD",
        "bytes": 640,
        "sha256": "110920dc9e6b4942c485d23e80de9a826bbfd80acbc19e6c16324152905e766d",
    },
    "tzdata": {
        "path": ".venv-p1/Lib/site-packages/tzdata-2026.3.dist-info/RECORD",
        "bytes": 56831,
        "sha256": "4e88bcacd1b80a7aff99638129717f504c6225c887a28f4ba0e193783f03c30e",
    },
}


class BootstrapError(RuntimeError):
    """A pre-import authentication boundary failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def deep_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= LOWER_SHA


def _runtime_declaration() -> dict[str, Any]:
    return {
        "interpreter": {
            "path": ".venv-p1/Scripts/python.exe",
            "bytes": 274424,
            "sha256": "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14",
        },
        "pyvenv_config": {
            "path": ".venv-p1/pyvenv.cfg",
            "bytes": 339,
            "sha256": "d1fb970854073922d49959ae01539088550613e316cb67f9fac858f586361174",
        },
        "base_prefix_source": "observed_sys_base_prefix_never_personal_path_literal",
        "base_runtime_files": {
            name: {"bytes": pin[0], "sha256": pin[1]} for name, pin in _EARLY_RUNTIME_PINS.items()
        },
        "stdlib_snapshots": _STDLIB_SNAPSHOTS,
        "distribution_records": PINNED_DISTRIBUTION_RECORDS,
        "third_party_import_enabled_only_after_qa_authorization_and_record_verification": True,
        "numerical_thread_environment": dict(_THREAD_ENVIRONMENT),
        "bytecode_loader_present": False,
        "source_loader_uses_authenticated_buffers": True,
    }


def parse_json_text(text: str, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise BootstrapError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise BootstrapError(f"nonfinite JSON constant in {label}: {value}")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"invalid JSON in {label}") from exc


def _relative_parts(relative: str) -> tuple[str, ...]:
    if type(relative) is not str or not relative or "\\" in relative:
        raise BootstrapError("path must be canonical POSIX relative text")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise BootstrapError("path is not a contained relative path")
    return pure.parts


def _has_reparse(path: Path) -> bool:
    info = path.lstat()
    return path.is_symlink() or bool(
        int(getattr(info, "st_file_attributes", 0)) & REPARSE_ATTRIBUTE
    )


def _plain_chain(path: Path, *, require_target: bool) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    for entry in (*reversed(lexical.parents), lexical):
        if os.path.lexists(entry) and _has_reparse(entry):
            raise BootstrapError(f"link/reparse path forbidden: {entry}")
    if require_target and not os.path.lexists(lexical):
        raise FileNotFoundError(lexical)
    return lexical


def contained_path(root: Path, relative: str, *, must_exist: bool) -> Path:
    base = _plain_chain(root, require_target=True).resolve(strict=True)
    candidate = _plain_chain(base.joinpath(*_relative_parts(relative)), require_target=must_exist)
    resolved = candidate.resolve(strict=must_exist)
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise BootstrapError("path escaped its declared root") from exc
    return resolved


def _read_identity(path: Path) -> tuple[bytes, dict[str, Any]]:
    checked = _plain_chain(path, require_target=True).resolve(strict=True)
    before = checked.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise BootstrapError("authenticated source must be a single-link regular file")
    descriptor = os.open(checked, os.O_RDONLY | O_BINARY)
    try:
        fd_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        fd_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = checked.lstat()
    path_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    path_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    fd_identity_before = (fd_before.st_dev, fd_before.st_ino, fd_before.st_size)
    fd_identity_after = (fd_after.st_dev, fd_after.st_ino, fd_after.st_size)
    if (
        path_before != path_after
        or fd_identity_before != fd_identity_after
        or fd_identity_before != (before.st_dev, before.st_ino, before.st_size)
        or fd_identity_after != (after.st_dev, after.st_ino, after.st_size)
    ):
        raise BootstrapError("source identity changed across authenticated read")
    payload = b"".join(chunks)
    if len(payload) != after.st_size:
        raise BootstrapError("authenticated source read length differs")
    return payload, {
        "bytes": int(after.st_size),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "non_reparse": True,
    }


def _pin3(relative: str, observed: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": relative,
        "bytes": observed["bytes"],
        "sha256": observed["sha256"],
    }


def _authenticate_pin(root: Path, pin: dict[str, Any], label: str) -> tuple[bytes, dict[str, Any]]:
    if (
        type(pin) is not dict
        or set(pin) != {"path", "bytes", "sha256"}
        or type(pin["bytes"]) is not int
        or pin["bytes"] < 0
        or not _is_sha(pin["sha256"])
    ):
        raise BootstrapError(f"{label} expected pin schema differs")
    path = contained_path(root, pin["path"], must_exist=True)
    payload, observed = _read_identity(path)
    if observed["bytes"] != pin["bytes"] or observed["sha256"] != pin["sha256"]:
        raise BootstrapError(f"{label} byte pin differs")
    return payload, _pin3(pin["path"], observed)


def _distribution_record_static_audit(root: Path) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for distribution, expected in PINNED_DISTRIBUTION_RECORDS.items():
        _payload, pin = _authenticate_pin(root, expected, f"{distribution} RECORD")
        observed[distribution] = pin
    return observed


def _authenticate_numerical_runtime(root: Path) -> dict[str, Any]:
    """Authenticate complete wheel RECORD contents before enabling imports."""

    global _NUMERICAL_IMPORTS_ENABLED
    if _NUMERICAL_IMPORTS_ENABLED:
        raise BootstrapError("numerical runtime authentication is one-shot")
    if tuple(map(_norm, sys.path)) != tuple(map(_norm, _EXPECTED_STARTUP_SYS_PATH)):
        raise BootstrapError("startup sys.path changed before numerical authentication")
    site = contained_path(root, ".venv-p1/Lib/site-packages", must_exist=True)
    distribution_report: dict[str, Any] = {}
    verified_paths: set[str] = set()
    native_paths: set[str] = set()
    for distribution, expected in PINNED_DISTRIBUTION_RECORDS.items():
        record_raw, record_pin = _authenticate_pin(root, expected, f"{distribution} RECORD")
        try:
            rows = list(csv.reader(io.StringIO(record_raw.decode("utf-8", errors="strict"))))
        except (UnicodeError, csv.Error) as exc:
            raise BootstrapError(f"invalid wheel RECORD: {distribution}") from exc
        if not rows or any(len(row) != 3 for row in rows):
            raise BootstrapError(f"wheel RECORD row schema differs: {distribution}")
        files = 0
        payload_bytes = 0
        for relative, encoded_digest, encoded_size in rows:
            pure = PurePosixPath(relative)
            if pure.is_absolute() or any(part in {"", "."} for part in pure.parts):
                raise BootstrapError(f"wheel RECORD path differs: {distribution}")
            candidate = Path(os.path.abspath(os.fspath(site.joinpath(*pure.parts))))
            try:
                candidate.relative_to(site)
            except ValueError:
                # Console entry points outside site-packages are never importable in
                # the isolated execution and therefore remain outside this TCB.
                continue
            normalized = _norm(os.fspath(candidate))
            if normalized == _norm(os.fspath(Path(record_pin["path"]))):
                continue
            if not encoded_digest or not encoded_digest.startswith("sha256="):
                if candidate.name == "RECORD" or candidate.suffix.casefold() in {".pyc", ".pyo"}:
                    continue
                raise BootstrapError(f"unhashed importable wheel member: {relative}")
            if not encoded_size.isdecimal():
                raise BootstrapError(f"wheel RECORD size differs: {relative}")
            path = _plain_chain(candidate, require_target=True).resolve(strict=True)
            payload, identity = _read_identity(path)
            digest_text = encoded_digest.removeprefix("sha256=")
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
            )
            if len(payload) != int(encoded_size) or digest != digest_text:
                raise BootstrapError(f"wheel member pin differs: {relative}")
            if normalized in _RUNTIME_FILE_PINS and _RUNTIME_FILE_PINS[normalized] != (
                len(payload),
                identity["sha256"],
            ):
                raise BootstrapError(f"wheel member has conflicting pins: {relative}")
            _RUNTIME_FILE_PINS[normalized] = (len(payload), identity["sha256"])
            if candidate.suffix.casefold() == ".py":
                _RUNTIME_FILE_BYTES[normalized] = payload
            if candidate.suffix.casefold() in {".pyd", ".dll"} and normalized not in native_paths:
                descriptor = os.open(normalized, os.O_RDONLY | getattr(os, "O_BINARY", 0))
                opened = os.fstat(descriptor)
                current = os.lstat(normalized)
                if (opened.st_dev, opened.st_ino, opened.st_size) != (
                    current.st_dev,
                    current.st_ino,
                    current.st_size,
                ):
                    os.close(descriptor)
                    raise BootstrapError(f"native wheel member identity differs: {relative}")
                _HELD_NATIVE_DESCRIPTORS.append(descriptor)
                _HELD_NATIVE_IDENTITIES[normalized] = (
                    int(opened.st_dev),
                    int(opened.st_ino),
                    int(opened.st_size),
                )
                native_paths.add(normalized)
            verified_paths.add(normalized)
            payload_bytes += len(payload)
            files += 1
        distribution_report[distribution] = {
            "record": record_pin,
            "verified_importable_files": files,
            "verified_importable_bytes": payload_bytes,
        }
    if not verified_paths or not native_paths:
        raise BootstrapError("numerical runtime inventory is empty")
    for name, value in _THREAD_ENVIRONMENT.items():
        os.environ[name] = value
    sys.path.append(os.fspath(site))
    sys.path_importer_cache.pop(os.fspath(site), None)
    _NUMERICAL_IMPORTS_ENABLED = True
    return {
        "site_packages": ".venv-p1/Lib/site-packages",
        "distributions": distribution_report,
        "verified_unique_files": len(verified_paths),
        "held_native_files": len(native_paths),
        "bytecode_loading_forbidden": True,
        "thread_environment": dict(_THREAD_ENVIRONMENT),
    }


def _verify_loaded_numerical_origins() -> dict[str, Any]:
    if not _NUMERICAL_IMPORTS_ENABLED:
        raise BootstrapError("numerical runtime was not authenticated")
    site = _norm(os.path.join(_EARLY_WORKSPACE, ".venv-p1", "Lib", "site-packages"))
    expected_path = (*tuple(map(_norm, _EXPECTED_STARTUP_SYS_PATH)), site)
    if tuple(map(_norm, sys.path)) != expected_path:
        raise BootstrapError("runtime sys.path changed after numerical authentication")
    meta_path = tuple(_meta_path_identity(item) for item in sys.meta_path)
    if meta_path not in {
        _EXPECTED_META_PATH,
        (*_EXPECTED_META_PATH, ("six", "_SixMetaPathImporter")),
    }:
        raise BootstrapError("runtime meta_path changed after numerical authentication")
    if not sys.path_hooks or sys.path_hooks[0] is not _trusted_path_hook:
        raise BootstrapError("authenticated runtime path hook was displaced")
    observed: dict[str, str] = {}
    for name, module in tuple(sys.modules.items()):
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if origin in {None, "built-in", "frozen"}:
            continue
        normalized = _norm(os.fspath(origin))
        inside_site = normalized == site or normalized.startswith(site + os.sep)
        if inside_site and normalized not in _RUNTIME_FILE_PINS:
            raise BootstrapError(f"loaded third-party origin is unauthenticated: {name}")
        if name.split(".", 1)[0] in _THIRD_PARTY_ROOTS and not inside_site:
            raise BootstrapError(f"third-party module escaped canonical site root: {name}")
        if inside_site:
            observed[name] = normalized
    return {
        "loaded_modules": len(observed),
        "all_origins_record_authenticated": True,
    }


def _resolve_environment() -> tuple[Path, Path]:
    raw_workspace = os.environ.get("P1_WORKSPACE_ROOT")
    raw_data = os.environ.get("P1_DATA_DIR")
    if not raw_workspace or not raw_data:
        raise BootstrapError("P1_WORKSPACE_ROOT and P1_DATA_DIR are required")
    if not Path(raw_workspace).is_absolute() or not Path(raw_data).is_absolute():
        raise BootstrapError("workspace and data environment paths must be absolute")
    workspace = _plain_chain(Path(raw_workspace), require_target=True).resolve(strict=True)
    data = _plain_chain(Path(raw_data), require_target=True).resolve(strict=True)
    if not workspace.is_dir() or not data.is_dir():
        raise BootstrapError("workspace and data roots must be plain directories")
    if workspace.stat().st_dev != data.stat().st_dev:
        raise BootstrapError("workspace and data must remain on one authenticated device")
    return workspace, data


def _execution_closure(
    *, config: dict[str, Any], bootstrap_pin: dict[str, Any], owner_pins: dict[str, Any]
) -> dict[str, Any]:
    return {
        "generation": GENERATION,
        "bootstrap_observed_pin": bootstrap_pin,
        "owner_role_pins": owner_pins,
        "science_projection_sha256": config["science_projection"]["sha256"],
        "v6_disposition": config["append_only_lineage"],
        "source_pins": config["source_pins"],
        "runtime_trust_contract": config["runtime_trust_contract"],
        "incumbent_binding": config["incumbent_binding"],
        "inner_incumbent_binding": config["inner_incumbent_binding"],
        "split_and_dependency_contract": config["split_and_dependency_contract"],
        "selective_target_contract": config["selective_target_contract"],
        "commitment_contract": config["commitment_contract"],
        "metric_contract": config["metric_contract"],
        "resource_ceiling": config["resource_ceiling"],
        "output_contract": config["output_contract"],
        "v9_binding": config["v9_binding"],
    }


def _strict_dynamic_pin(root: Path, relative: str, label: str) -> tuple[bytes, dict[str, Any]]:
    path = contained_path(root, relative, must_exist=True)
    payload, observed = _read_identity(path)
    return payload, _pin3(relative, observed)


def _source_identity(
    data: Path,
    config: dict[str, Any],
    authenticated_data_bytes_for_pin: Any | None = None,
) -> dict[str, Any]:
    info = data.lstat()
    result: dict[str, Any] = {
        "directory": {
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
            "non_reparse": True,
        },
        "files": {},
    }
    for name, pin in config["source_pins"].items():
        path = contained_path(data, pin["path"], must_exist=True)
        if authenticated_data_bytes_for_pin is None:
            payload, observed = _read_identity(path)
        else:
            payload = authenticated_data_bytes_for_pin(pin, f"source {name}")
            info = path.lstat()
            observed = {
                "bytes": int(info.st_size),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "device": int(info.st_dev),
                "inode": int(info.st_ino),
                "nlink": int(info.st_nlink),
                "non_reparse": True,
            }
        if len(payload) != pin["bytes"] or observed["sha256"] != pin["sha256"]:
            raise BootstrapError(f"source {name} byte pin differs")
        result["files"][name] = {
            "path": pin["path"],
            "bytes": observed["bytes"],
            "device": observed["device"],
            "inode": observed["inode"],
            "nlink": 1,
            "non_reparse": True,
            "opened": True,
            "sha256": observed["sha256"],
        }
    return result


def _prevalidate_documents(
    *,
    workspace: Path,
    data: Path,
    config: dict[str, Any],
    bootstrap_pin: dict[str, Any],
    owner_pins: dict[str, Any],
    authenticated_data_bytes_for_pin: Any,
) -> dict[str, Any]:
    paths = config["canonical_paths"]
    qa_raw, qa_pin = _strict_dynamic_pin(workspace, paths["independent_qa"], "independent QA")
    auth_raw, auth_pin = _strict_dynamic_pin(
        workspace, paths["execution_authorization"], "execution authorization"
    )
    qa = parse_json_text(qa_raw.decode("utf-8", errors="strict"), "independent QA")
    auth = parse_json_text(auth_raw.decode("utf-8", errors="strict"), "execution authorization")
    closure_sha = deep_sha256(
        _execution_closure(config=config, bootstrap_pin=bootstrap_pin, owner_pins=owner_pins)
    )
    qa_fields = {
        "schema_version",
        "created_at_kst",
        "problem",
        "generation",
        "reviewer",
        "reviewer_independent",
        "verdict",
        "p0_count",
        "p1_count",
        "bootstrap",
        "owner_role_pins",
        "execution_closure_sha256",
        "v6_disposition_verified",
        "v9_binding",
        "source_identity",
        "incumbent_binding_verified",
        "inner_incumbent_binding_verified",
        "resource_ceiling_verified",
        "static_report_sha256",
        "actual_run_performed",
        "counters",
    }
    zero_counters = {
        "authorizations": 0,
        "attempt_locks": 0,
        "fits": 0,
        "predictions": 0,
        "target_decodes": 0,
        "scores": 0,
        "candidates": 0,
        "test_predictions": 0,
        "ledger_appends": 0,
        "uploads": 0,
    }
    if (
        type(qa) is not dict
        or set(qa) != qa_fields
        or qa["schema_version"] != QA_SCHEMA
        or qa["problem"] != "P1"
        or qa["generation"] != GENERATION
        or type(qa["reviewer"]) is not str
        or not qa["reviewer"]
        or qa["reviewer_independent"] is not True
        or qa["verdict"] != "GO"
        or type(qa["p0_count"]) is not int
        or qa["p0_count"] != 0
        or type(qa["p1_count"]) is not int
        or qa["p1_count"] != 0
        or qa["bootstrap"] != bootstrap_pin
        or qa["owner_role_pins"] != owner_pins
        or qa["execution_closure_sha256"] != closure_sha
        or qa["v6_disposition_verified"] is not True
        or qa["v9_binding"] != config["v9_binding"]
        or qa["source_identity"] != _source_identity(data, config, authenticated_data_bytes_for_pin)
        or qa["incumbent_binding_verified"] is not True
        or qa["inner_incumbent_binding_verified"] is not True
        or qa["resource_ceiling_verified"] is not True
        or not _is_sha(qa["static_report_sha256"])
        or qa["actual_run_performed"] is not False
        or type(qa["counters"]) is not dict
        or set(qa["counters"]) != set(zero_counters)
        or any(
            type(qa["counters"][name]) is not int or qa["counters"][name] != expected
            for name, expected in zero_counters.items()
        )
    ):
        raise BootstrapError("independent QA receipt semantics differ")
    auth_fields = {
        "schema_version",
        "created_at_kst",
        "problem",
        "generation",
        "authorization",
        "user_message_reference",
        "qa_receipt",
        "bootstrap",
        "owner_role_pins_sha256",
        "execution_closure_sha256",
        "v9_binding",
        "execution_authorized",
        "research_curve_only",
        "one_shot_no_resume",
        "test_prediction_allowed",
        "candidate_creation_allowed",
        "ledger_append_allowed",
        "upload_allowed",
    }
    if (
        type(auth) is not dict
        or set(auth) != auth_fields
        or auth["schema_version"] != AUTH_SCHEMA
        or auth["problem"] != "P1"
        or auth["generation"] != GENERATION
        or auth["authorization"] != AUTHORIZATION_PHRASE
        or type(auth["user_message_reference"]) is not str
        or not auth["user_message_reference"]
        or auth["qa_receipt"] != qa_pin
        or auth["bootstrap"] != bootstrap_pin
        or auth["owner_role_pins_sha256"] != deep_sha256(owner_pins)
        or auth["execution_closure_sha256"] != closure_sha
        or auth["v9_binding"] != config["v9_binding"]
        or auth["execution_authorized"] is not True
        or auth["research_curve_only"] is not True
        or auth["one_shot_no_resume"] is not True
        or auth["test_prediction_allowed"] is not False
        or auth["candidate_creation_allowed"] is not False
        or auth["ledger_append_allowed"] is not False
        or auth["upload_allowed"] is not False
    ):
        raise BootstrapError("execution authorization semantics differ")
    return {
        "qa_raw": qa_raw,
        "qa": qa,
        "qa_pin": qa_pin,
        "auth_raw": auth_raw,
        "auth": auth,
        "auth_pin": auth_pin,
    }


def _compile_module(
    *, name: str, path: str, payload: bytes, injected: dict[str, Any]
) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = path
    module.__package__ = ""
    module.__dict__.update(injected)
    sys.modules[name] = module
    try:
        code = compile(payload.decode("utf-8", errors="strict"), path, "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _install_check_only_write_firewall(context: dict[str, Any]) -> None:
    write_events = {
        "os.mkdir",
        "os.makedirs",
        "os.remove",
        "os.rmdir",
        "os.rename",
        "os.replace",
        "os.unlink",
        "shutil.copyfile",
        "shutil.copytree",
        "subprocess.Popen",
        "socket.connect",
    }

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event == "open":
            mode = args[1] if len(args) > 1 else "r"
            flags = args[2] if len(args) > 2 else 0
            writable_mode = type(mode) is str and any(character in mode for character in "wax+")
            writable_flags = type(flags) is int and bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
            if writable_mode or writable_flags:
                context["writes_observed"] += 1
                raise PermissionError("check-only filesystem write forbidden")
        elif event in write_events:
            context["writes_observed"] += 1
            raise PermissionError(f"check-only side effect forbidden: {event}")

    sys.addaudithook(audit)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    mode = "check-only" if args.check_only else "execute"
    workspace, data = _resolve_environment()
    canonical_bootstrap = contained_path(workspace, BOOTSTRAP_RELATIVE, must_exist=True)
    if Path(__file__).resolve(strict=True) != canonical_bootstrap:
        raise BootstrapError("bootstrap is not running from its canonical path")
    bootstrap_raw, bootstrap_observed = _read_identity(canonical_bootstrap)
    del bootstrap_raw
    bootstrap_pin = _pin3(BOOTSTRAP_RELATIVE, bootstrap_observed)
    role_buffers: dict[str, bytes] = {}
    owner_pins: dict[str, dict[str, Any]] = {}
    for role, expected in PINNED_OWNER_ROLES.items():
        payload, observed = _authenticate_pin(workspace, expected, role)
        role_buffers[role] = payload
        owner_pins[role] = observed
    config = parse_json_text(role_buffers["CONFIG"].decode("utf-8"), "CONFIG")
    projection = parse_json_text(
        role_buffers["SCIENCE_PROJECTION"].decode("utf-8"), "SCIENCE_PROJECTION"
    )
    if config.get("implementation_roles") != {
        role: PINNED_OWNER_ROLES[role]["path"] for role in PINNED_OWNER_ROLES
    } | {"BOOTSTRAP": BOOTSTRAP_RELATIVE}:
        raise BootstrapError("config implementation role paths differ")
    if config.get("science_projection") != owner_pins["SCIENCE_PROJECTION"]:
        raise BootstrapError("config science projection pin differs")
    if config.get("runtime_trust_contract") != _runtime_declaration():
        raise BootstrapError("config runtime trust declaration differs")
    runtime_record_pins = _distribution_record_static_audit(workspace)
    modules_before = frozenset(sys.modules)
    semantic_buffers: dict[tuple[str, str, int, str], bytes] = {}

    def _cached_authenticated_bytes(root: Path, pin: dict[str, Any], label: str) -> bytes:
        payload, observed = _authenticate_pin(root, pin, label)
        key = (
            _norm(os.fspath(root)),
            pin["path"],
            observed["bytes"],
            observed["sha256"],
        )
        previous = semantic_buffers.setdefault(key, payload)
        if previous != payload:
            raise BootstrapError(f"authenticated semantic buffer changed: {label}")
        return previous

    def _seed_semantic_buffer(root: Path, pin: dict[str, Any], payload: bytes, label: str) -> None:
        key = (_norm(os.fspath(root)), pin["path"], pin["bytes"], pin["sha256"])
        previous = semantic_buffers.setdefault(key, payload)
        if previous != payload:
            raise BootstrapError(f"authenticated semantic buffer conflict: {label}")

    context: dict[str, Any] = {
        "mode": mode,
        "workspace": workspace,
        "data_dir": data,
        "config": config,
        "science_projection": projection,
        "bootstrap_observed_pin": bootstrap_pin,
        "owner_role_pins": owner_pins,
        "all_owner_roles_authenticated": True,
        "bootstrap_documents_prevalidated": False,
        "engine_loaded": False,
        "engine_module": None,
        "modules_before_owner_load": modules_before,
        "writes_observed": 0,
        "runtime_static_audit": {
            "canonical_flags": ["-I", "-S", "-B"],
            "safe_path": True,
            "ignore_environment": True,
            "pycache_prefix_absent": True,
            "startup_sys_path_exact": True,
            "startup_meta_path_exact": True,
            "stdlib_snapshots": _STDLIB_SNAPSHOTS,
            "distribution_record_pins": runtime_record_pins,
            "earliest_audit_hook_installed": True,
            "stdlib_source_buffer_loader_installed": True,
        },
        "numerical_runtime_audit": None,
        "prevalidated_execution_documents": None,
    }

    def reverify_owner_roles() -> dict[str, dict[str, Any]]:
        _payload, live_bootstrap = _read_identity(canonical_bootstrap)
        if _pin3(BOOTSTRAP_RELATIVE, live_bootstrap) != bootstrap_pin:
            raise BootstrapError("bootstrap changed during process lifetime")
        live: dict[str, dict[str, Any]] = {}
        for role, expected in PINNED_OWNER_ROLES.items():
            _role_payload, observed = _authenticate_pin(workspace, expected, role)
            live[role] = observed
        if live != owner_pins:
            raise BootstrapError("owner role pin map changed during process lifetime")
        return live

    def authenticated_bytes_for_pin(pin: dict[str, Any], label: str) -> bytes:
        return _cached_authenticated_bytes(workspace, pin, label)

    def _authenticated_data_bytes_for_integrity(pin: dict[str, Any], label: str) -> bytes:
        return _cached_authenticated_bytes(data, pin, label)

    def authenticated_train_bytes_for_pin(pin: dict[str, Any], label: str) -> bytes:
        if pin != config["source_pins"]["train.csv"]:
            raise BootstrapError("only train.csv may cross the semantic data buffer boundary")
        return _authenticated_data_bytes_for_integrity(pin, label)

    def authenticated_json_for_pin(pin: dict[str, Any], label: str) -> Any:
        payload = authenticated_bytes_for_pin(pin, label)
        return parse_json_text(payload.decode("utf-8", errors="strict"), label)

    def strict_dynamic_json_for_pin(pin: dict[str, Any], label: str) -> Any:
        return authenticated_json_for_pin(pin, label)

    context.update(
        {
            "reverify_owner_roles": reverify_owner_roles,
            "authenticated_bytes_for_pin": authenticated_bytes_for_pin,
            "authenticated_json_for_pin": authenticated_json_for_pin,
            "authenticated_train_bytes_for_pin": authenticated_train_bytes_for_pin,
            "strict_dynamic_json_for_pin": strict_dynamic_json_for_pin,
            "parse_json_text": parse_json_text,
            "verify_numerical_runtime": _verify_loaded_numerical_origins,
        }
    )
    if mode == "check-only":
        _install_check_only_write_firewall(context)
    else:
        documents = _prevalidate_documents(
            workspace=workspace,
            data=data,
            config=config,
            bootstrap_pin=bootstrap_pin,
            owner_pins=owner_pins,
            authenticated_data_bytes_for_pin=_authenticated_data_bytes_for_integrity,
        )
        _seed_semantic_buffer(workspace, documents["qa_pin"], documents["qa_raw"], "independent QA")
        _seed_semantic_buffer(
            workspace,
            documents["auth_pin"],
            documents["auth_raw"],
            "execution authorization",
        )
        context["prevalidated_execution_documents"] = documents
        context["bootstrap_documents_prevalidated"] = True
        context["numerical_runtime_audit"] = _authenticate_numerical_runtime(workspace)
    injected = {"_P1_V6R2_BOOTSTRAP_CONTEXT": context}
    science_module = _compile_module(
        name="_p1_v6r2_authenticated_science",
        path=PINNED_OWNER_ROLES["SCIENCE"]["path"],
        payload=role_buffers["SCIENCE"],
        injected=injected,
    )
    contract_module = _compile_module(
        name="_p1_v6r2_authenticated_contract",
        path=PINNED_OWNER_ROLES["CONTRACT"]["path"],
        payload=role_buffers["CONTRACT"],
        injected=injected,
    )
    runner_injected = {
        **injected,
        "_P1_V6R2_AUTH_SCIENCE": science_module,
        "_P1_V6R2_AUTH_CONTRACT": contract_module,
    }
    if mode == "execute":
        engine_module = _compile_module(
            name="_p1_v6r2_authenticated_engine",
            path=PINNED_OWNER_ROLES["ENGINE"]["path"],
            payload=role_buffers["ENGINE"],
            injected=runner_injected,
        )
        context["engine_module"] = engine_module
        context["engine_loaded"] = True
    runner_module = _compile_module(
        name="_p1_v6r2_authenticated_runner",
        path=PINNED_OWNER_ROLES["RUNNER"]["path"],
        payload=role_buffers["RUNNER"],
        injected=runner_injected,
    )
    report = runner_module.main()
    if type(report) is not dict:
        raise BootstrapError("runner report must be an exact object")
    report["summary_sha256"] = deep_sha256(report)
    print(canonical_json_bytes(report).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
