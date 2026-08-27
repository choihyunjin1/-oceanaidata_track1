"""Noncyclic authenticated bootstrap for P1 multiscale Gen6r3.

The independently pinned external PowerShell host and launcher are the trust
root.  They authenticate and retain the interpreter/startup files before this
script can execute.  This bootstrap then installs the earliest possible audit
hook, authenticates and holds the complete runtime and owner closure, and only
then compiles subordinate modules from held byte buffers.
"""

# ruff: noqa: E402 -- import order is the enforced pre-import trust boundary.

from __future__ import annotations

import sys

if not (
    sys.flags.isolated == 1
    and sys.flags.no_site == 1
    and sys.flags.dont_write_bytecode == 1
    and sys.flags.ignore_environment == 1
    and sys.flags.safe_path is True
):
    raise RuntimeError("canonical Gen6r3 bootstrap requires python -I -S -B")


def _parse_early_mode(argv: list[str]) -> tuple[str, int | None, str | None, str | None]:
    if argv == ["--check-only"]:
        return "check-only", None, None, None
    if argv == ["--execute"]:
        return "parent", None, None, None
    if (
        len(argv) == 6
        and argv[0] == "--cell-worker"
        and argv[2] == "--session"
        and argv[4] == "--prior"
        and argv[1].isdigit()
    ):
        cell = int(argv[1])
        if 1 <= cell <= 15:
            return "cell_worker", cell, argv[3], argv[5]
    raise RuntimeError("canonical Gen6r3 bootstrap arguments differ")


_MODE, _CELL, _SESSION_ARGUMENT, _PRIOR_ARGUMENT = _parse_early_mode(sys.argv[1:])
_EARLY_AUDIT_EVENTS: list[tuple[str, tuple[object, ...]]] = []
_EARLY_WRITES = 0
_WRITE_DEPTH = 0
_PROCESS_DEPTH = 0
_TRUSTED_CTYPES_DEPTH = 0
_THIRD_PARTY_ENABLED = False
_RUNTIME_LOCKED = False
_HELD_PATHS: set[str] = set()
_BASE_PREFIX_NORM = ""
_VENV_ROOT_NORM = ""
_WORKSPACE_NORM = ""
_OUTPUT_ROOT_NORM = ""


def _early_audit_hook(event: str, arguments: tuple[object, ...]) -> None:
    global _EARLY_WRITES
    if event == "import":
        name = arguments[0] if arguments else ""
        if (
            type(name) is str
            and name.split(".", 1)[0]
            in {
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
            and not _THIRD_PARTY_ENABLED
        ):
            raise PermissionError("third-party import preceded RECORD authentication")
    if event == "open":
        target = arguments[0] if arguments else ""
        mode = arguments[1] if len(arguments) > 1 else "r"
        flags = arguments[2] if len(arguments) > 2 else 0
        if type(target) is str and target.casefold().endswith((".pyc", ".pyo")):
            raise PermissionError("bytecode read/write is forbidden")
        writable = (type(mode) is str and any(character in mode for character in "wax+")) or (
            type(flags) is int and bool(flags & (1 | 2 | 8 | 64 | 256 | 512 | 1024))
        )
        if writable and _WRITE_DEPTH == 0:
            _EARLY_WRITES += 1
            raise PermissionError("write open is outside the exclusive writer")
        if (
            _RUNTIME_LOCKED
            and not writable
            and type(target) is str
            and target
            and _is_runtime_path_for_audit(target)
            and _norm_for_audit(target) not in _HELD_PATHS
        ):
            raise PermissionError("runtime read is outside the held authenticated inventory")
    if event in {
        "os.remove",
        "os.unlink",
        "os.rename",
        "os.replace",
        "os.rmdir",
        "os.mkdir",
        "os.chmod",
        "os.truncate",
        "os.utime",
        "os.symlink",
        "os.link",
    } and _WRITE_DEPTH == 0:
        _EARLY_WRITES += 1
        raise PermissionError(f"filesystem mutation is outside the exclusive writer: {event}")
    if event == "subprocess.Popen" and _PROCESS_DEPTH == 0:
        raise PermissionError("process creation is outside the exact parent worker launcher")
    if event.startswith("socket."):
        raise PermissionError(f"network activity is forbidden: {event}")
    if event.startswith("ctypes.") and _TRUSTED_CTYPES_DEPTH == 0:
        raise PermissionError(f"ctypes escape is forbidden: {event}")
    if len(_EARLY_AUDIT_EVENTS) < 8192 and event in {"import", "open", "subprocess.Popen"}:
        _EARLY_AUDIT_EVENTS.append((event, arguments[:4]))


def _norm_for_audit(path: str) -> str:
    # os is imported immediately after the hook; the function is never called
    # before then because _RUNTIME_LOCKED starts false.
    return os.path.normcase(os.path.abspath(path))  # type: ignore[name-defined]


def _is_runtime_path_for_audit(path: str) -> bool:
    normalized = _norm_for_audit(path)
    roots = tuple(root for root in (_BASE_PREFIX_NORM, _VENV_ROOT_NORM) if root)
    return any(normalized == root or normalized.startswith(root + os.sep) for root in roots)  # type: ignore[name-defined]


sys.addaudithook(_early_audit_hook)

import _sha2
import _winapi
import msvcrt
import os
import stat

if (
    getattr(os.__spec__, "origin", None) != "frozen"
    or getattr(stat.__spec__, "origin", None) != "frozen"
    or getattr(_sha2.__spec__, "origin", None) != "built-in"
    or getattr(_winapi.__spec__, "origin", None) != "built-in"
    or getattr(msvcrt.__spec__, "origin", None) != "built-in"
):
    raise RuntimeError("earliest bootstrap modules are not frozen/builtin")

sys.dont_write_bytecode = True

_BOOTSTRAP_CREATE_FILE = _winapi.CreateFile
_BOOTSTRAP_CREATE_PROCESS = _winapi.CreateProcess
_BOOTSTRAP_OS_MKDIR = os.mkdir

_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_FILE_SHARE_READ = 0x00000001
_CREATE_NEW = 1
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_NORMAL = 0x00000080
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_REPARSE_ATTRIBUTE = 0x00000400


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


_RAW_WORKSPACE = os.environ.get("P1_WORKSPACE_ROOT")
if not _RAW_WORKSPACE or not os.path.isabs(_RAW_WORKSPACE):
    raise RuntimeError("P1_WORKSPACE_ROOT must be externally supplied and absolute")
_WORKSPACE_NORM = _norm(_RAW_WORKSPACE)
_BASE_PREFIX_NORM = _norm(sys.base_prefix)
_VENV_ROOT_NORM = _norm(os.path.join(_WORKSPACE_NORM, ".venv-p1"))
_OUTPUT_ROOT_NORM = _norm(
    os.path.join(
        _WORKSPACE_NORM,
        "artifacts",
        "p1_multiscale_cross_layer_offset_drift_unary_v6r3",
    )
)
_PYCACHE_PREFIX = _norm(
    os.path.join(
        _WORKSPACE_NORM,
        "configs",
        "experiments",
        "p1_multiscale_cross_layer_offset_drift_unary_v6r3_startup_trust.json",
    )
)
_BOOTSTRAP_PATH = _norm(
    os.path.join(
        _WORKSPACE_NORM,
        "scripts",
        "bootstrap_p1_multiscale_cross_layer_offset_drift_unary_v6r3.py",
    )
)
_VENV_PYTHON = _norm(os.path.join(_VENV_ROOT_NORM, "Scripts", "python.exe"))
_BASE_PYTHON = _norm(os.path.join(_BASE_PREFIX_NORM, "python.exe"))
_CONTROL_ROOT_EARLY = _norm(
    os.path.join(
        _WORKSPACE_NORM,
        "artifacts",
        "p1_multiscale_cross_layer_offset_drift_unary_v6r3_control",
    )
)


def _build_native_brokers(
    create_file: object,
    create_process: object,
    mkdir: object,
) -> tuple[object, object, object]:
    def broker_create_file(*args: object, **kwargs: object) -> object:
        path = args[0] if args else kwargs.get("file_name")
        access = args[1] if len(args) > 1 else kwargs.get("desired_access")
        share = args[2] if len(args) > 2 else kwargs.get("share_mode")
        disposition = args[4] if len(args) > 4 else kwargs.get("creation_disposition")
        attributes = args[5] if len(args) > 5 else kwargs.get("flags_and_attributes")
        if type(path) is not str:
            raise PermissionError("native file broker path differs")
        absolute = _norm(path)
        if access == _GENERIC_READ:
            if (
                share != _FILE_SHARE_READ
                or disposition != _OPEN_EXISTING
                or type(attributes) is not int
                or not attributes & _FILE_FLAG_OPEN_REPARSE_POINT
            ):
                raise PermissionError("native read broker flags differ")
        elif access == (_GENERIC_READ | _GENERIC_WRITE):
            contained = any(
                absolute == root or absolute.startswith(root + os.sep)
                for root in (_OUTPUT_ROOT_NORM, _CONTROL_ROOT_EARLY)
            )
            if (
                _WRITE_DEPTH != 1
                or not contained
                or share != _FILE_SHARE_READ
                or disposition != _CREATE_NEW
                or type(attributes) is not int
                or not attributes & _FILE_FLAG_OPEN_REPARSE_POINT
            ):
                raise PermissionError("native exclusive-write broker flags differ")
        else:
            raise PermissionError("native file broker access differs")
        return create_file(*args, **kwargs)  # type: ignore[operator]

    def broker_create_process(*args: object, **kwargs: object) -> object:
        return create_process(*args, **kwargs)  # type: ignore[operator]

    def broker_mkdir(path: str) -> object:
        absolute = _norm(path)
        contained = any(
            absolute == root or absolute.startswith(root + os.sep)
            for root in (_OUTPUT_ROOT_NORM, _CONTROL_ROOT_EARLY)
        )
        if _WRITE_DEPTH != 1 or not contained:
            raise PermissionError("native mkdir broker scope differs")
        return mkdir(path)  # type: ignore[operator]

    return broker_create_file, broker_create_process, broker_mkdir


_BROKER_CREATE_FILE, _BROKER_CREATE_PROCESS, _BROKER_MKDIR = _build_native_brokers(
    _BOOTSTRAP_CREATE_FILE,
    _BOOTSTRAP_CREATE_PROCESS,
    _BOOTSTRAP_OS_MKDIR,
)
del _BOOTSTRAP_CREATE_FILE, _BOOTSTRAP_CREATE_PROCESS, _BOOTSTRAP_OS_MKDIR, _build_native_brokers

if _norm(sys.executable) != _VENV_PYTHON:
    raise RuntimeError("canonical venv interpreter path differs")
if _norm(getattr(sys, "_base_executable", "")) != _BASE_PYTHON:
    raise RuntimeError("base executable binding differs")
if (
    _norm(sys.pycache_prefix or "") != _PYCACHE_PREFIX
    or not os.path.isfile(_PYCACHE_PREFIX)
    or os.path.isdir(_PYCACHE_PREFIX)
):
    raise RuntimeError("canonical regular-file pycache sentinel differs")

_EXPECTED_SYS_PATH = (
    _norm(os.path.join(_BASE_PREFIX_NORM, "python312.zip")),
    _norm(os.path.join(_BASE_PREFIX_NORM, "DLLs")),
    _norm(os.path.join(_BASE_PREFIX_NORM, "Lib")),
    _BASE_PREFIX_NORM,
)
if tuple(map(_norm, sys.path)) != _EXPECTED_SYS_PATH:
    raise RuntimeError("canonical isolated initial sys.path differs")


def _meta_path_identity(item: object) -> tuple[str, str]:
    target = item if isinstance(item, type) else type(item)
    return target.__module__, target.__qualname__


if tuple(_meta_path_identity(item) for item in sys.meta_path) != (
    ("_frozen_importlib", "BuiltinImporter"),
    ("_frozen_importlib", "FrozenImporter"),
    ("_frozen_importlib_external", "PathFinder"),
):
    raise RuntimeError("canonical initial meta_path differs")

_EXPECTED_ORIG_ARGV = [
    _BASE_PYTHON,
    "-I",
    "-S",
    "-B",
    "-X",
    f"pycache_prefix={_PYCACHE_PREFIX}",
    _BOOTSTRAP_PATH,
    *sys.argv[1:],
]
_OBSERVED_ORIG_ARGV = [
    _norm(value) if index in {0, 6} else value
    for index, value in enumerate(sys.orig_argv)
]
if _OBSERVED_ORIG_ARGV != _EXPECTED_ORIG_ARGV:
    raise RuntimeError("canonical sys.orig_argv differs")

_PRE_SCRIPT_ORIGINS = {
    "encodings": _norm(os.path.join(_BASE_PREFIX_NORM, "Lib", "encodings", "__init__.py")),
    "encodings.aliases": _norm(
        os.path.join(_BASE_PREFIX_NORM, "Lib", "encodings", "aliases.py")
    ),
    "encodings.cp949": _norm(os.path.join(_BASE_PREFIX_NORM, "Lib", "encodings", "cp949.py")),
    "encodings.utf_8": _norm(os.path.join(_BASE_PREFIX_NORM, "Lib", "encodings", "utf_8.py")),
}
for _module_name, _expected_origin in _PRE_SCRIPT_ORIGINS.items():
    _module = sys.modules.get(_module_name)
    if _module is None or _norm(getattr(getattr(_module, "__spec__", None), "origin", "")) != _expected_origin:
        raise RuntimeError(f"pre-script encoding origin differs: {_module_name}")
for _module_name in ("_codecs_kr", "_multibytecodec"):
    _module = sys.modules.get(_module_name)
    if _module is None or getattr(getattr(_module, "__spec__", None), "origin", None) != "built-in":
        raise RuntimeError(f"pre-script codec helper is not built-in: {_module_name}")


def _assert_plain_ancestors(path: str) -> None:
    absolute = _norm(path)
    drive, tail = os.path.splitdrive(absolute)
    current = drive + os.sep
    for part in [item for item in tail.split(os.sep) if item]:
        current = os.path.join(current, part)
        if not os.path.lexists(current):
            raise RuntimeError(f"required path ancestor is absent: {path}")
        info = os.lstat(current)
        if int(getattr(info, "st_file_attributes", 0)) & _REPARSE_ATTRIBUTE:
            raise RuntimeError(f"reparse point is forbidden: {current}")


_HELD: dict[str, dict[str, object]] = {}
_SOURCE_BUFFERS: dict[str, bytes] = {}


def _read_fd(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    blocks: list[bytes] = []
    while True:
        block = os.read(fd, 1 << 20)
        if not block:
            break
        blocks.append(block)
    return b"".join(blocks)


def _requires_retained_handle(path: str) -> bool:
    absolute = _norm(path)
    runtime_path = (
        absolute == _BASE_PREFIX_NORM
        or absolute.startswith(_BASE_PREFIX_NORM + os.sep)
        or absolute == _VENV_ROOT_NORM
        or absolute.startswith(_VENV_ROOT_NORM + os.sep)
    )
    startup_paths = {
        _VENV_PYTHON,
        _BASE_PYTHON,
        _norm(os.path.join(_VENV_ROOT_NORM, "pyvenv.cfg")),
        *set(_PRE_SCRIPT_ORIGINS.values()),
    }
    return (
        not runtime_path
        or absolute in startup_paths
        or absolute.casefold().endswith((".dll", ".exe", ".pyd"))
    )


def _hold_plain_file_impl(
    path: str, broker_create_file: object
) -> tuple[bytes, dict[str, object]]:
    absolute = _norm(path)
    existing = _HELD.get(absolute)
    if existing is not None:
        payload = (
            bytes(existing["payload"])
            if existing["fd"] is None
            else _read_fd(int(existing["fd"]))
        )
        if len(payload) != existing["bytes"] or _sha2.sha256(payload).hexdigest() != existing["sha256"]:
            raise RuntimeError(f"held file changed: {path}")
        return payload, existing
    _assert_plain_ancestors(absolute)
    before = os.lstat(absolute)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or int(getattr(before, "st_file_attributes", 0)) & _REPARSE_ATTRIBUTE
    ):
        raise RuntimeError(f"held path is not a unique plain file: {path}")
    raw_handle = broker_create_file(  # type: ignore[operator]
        absolute,
        _GENERIC_READ,
        _FILE_SHARE_READ,
        0,
        _OPEN_EXISTING,
        _FILE_ATTRIBUTE_NORMAL | _FILE_FLAG_OPEN_REPARSE_POINT,
        0,
    )
    try:
        fd = msvcrt.open_osfhandle(raw_handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except BaseException:
        _winapi.CloseHandle(raw_handle)
        raise
    opened = os.fstat(fd)
    payload = _read_fd(fd)
    after = os.lstat(absolute)
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_mode),
        int(before.st_nlink),
        int(getattr(before, "st_file_attributes", 0)),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_mode),
        int(after.st_nlink),
        int(getattr(after, "st_file_attributes", 0)),
    )
    handle_identity = (
        int(opened.st_dev),
        int(opened.st_ino),
        int(opened.st_size),
    )
    if (
        before_identity != after_identity
        or handle_identity != after_identity[:3]
        or after.st_nlink != 1
        or len(payload) != opened.st_size
    ):
        os.close(fd)
        raise RuntimeError(f"held file identity changed during authentication: {path}")
    identity = after_identity[:5]
    retained = _requires_retained_handle(absolute)
    if not retained:
        os.close(fd)
    record: dict[str, object] = {
        "fd": fd if retained else None,
        "path": absolute,
        "bytes": len(payload),
        "sha256": _sha2.sha256(payload).hexdigest(),
        "identity": identity,
        "payload": payload,
    }
    _HELD[absolute] = record
    _HELD_PATHS.add(absolute)
    if absolute.casefold().endswith(".py"):
        _SOURCE_BUFFERS[absolute] = payload
    return payload, record


def _bind_hold_plain_file(implementation: object, broker_create_file: object) -> object:
    def held(path: str) -> tuple[bytes, dict[str, object]]:
        return implementation(path, broker_create_file)  # type: ignore[operator, no-any-return]

    return held


_hold_plain_file = _bind_hold_plain_file(_hold_plain_file_impl, _BROKER_CREATE_FILE)
del _bind_hold_plain_file, _hold_plain_file_impl


def _assert_pin(path: str, expected_bytes: int, expected_sha256: str) -> bytes:
    payload, _record = _hold_plain_file(path)
    if len(payload) != expected_bytes or _sha2.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeError(f"file pin differs: {path}")
    return payload


_STARTUP_PINS = (
    (_PYCACHE_PREFIX, 8249, "7dcc0c4a79eb3d1d67e22a1d9889e0160c082a925f460b884b1b7e50f5d75dfc"),
    (_VENV_PYTHON, 274424, "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14"),
    (_norm(os.path.join(_VENV_ROOT_NORM, "pyvenv.cfg")), 339, "d1fb970854073922d49959ae01539088550613e316cb67f9fac858f586361174"),
    (_BASE_PYTHON, 104952, "4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a"),
    (_norm(os.path.join(_BASE_PREFIX_NORM, "python312.dll")), 6945272, "9a0e3435aaa680d868150f87ab3e388ad2eebc22f87e036155c7b4eda8cd2120"),
    (_norm(os.path.join(_BASE_PREFIX_NORM, "python3.dll")), 70376, "fb975a606e7fbf74f64260e3f60c3490b4f74a183c0926fd6ed1ac4c52ac7b1c"),
    (_norm(os.path.join(_BASE_PREFIX_NORM, "vcruntime140.dll")), 120400, "052ad6a20d375957e82aa6a3c441ea548d89be0981516ca7eb306e063d5027f4"),
    (_PRE_SCRIPT_ORIGINS["encodings"], 6058, "8b997e9f7beef09de01c34ac34191866d3ab25e17164e08f411940b070bc3e74"),
    (_PRE_SCRIPT_ORIGINS["encodings.aliases"], 16228, "1893cfb597bc5eafd38ef03ac85d8874620112514eb42660408811929cc0d6f8"),
    (_PRE_SCRIPT_ORIGINS["encodings.utf_8"], 1047, "9c54c7db8ce0722ca4ddb5f45d4e170357e37991afb3fcdc091721bf6c09257e"),
    (_PRE_SCRIPT_ORIGINS["encodings.cp949"], 1062, "da13fd6f1bd7a1d3b48aed1fc75f7516d6a33814086cf971e030625590e9dda0"),
)
for _path, _bytes, _sha in _STARTUP_PINS:
    _assert_pin(_path, _bytes, _sha)

_PYVENV_TEXT = bytes(_HELD[_norm(os.path.join(_VENV_ROOT_NORM, "pyvenv.cfg"))]["payload"]).decode(
    "utf-8", errors="strict"
)
_PYVENV_VALUES = {
    key.strip().casefold(): value.strip()
    for line in _PYVENV_TEXT.splitlines()
    if "=" in line
    for key, value in [line.split("=", 1)]
}
if (
    _norm(_PYVENV_VALUES.get("home", "")) != _BASE_PREFIX_NORM
    or _norm(_PYVENV_VALUES.get("executable", "")) != _BASE_PYTHON
    or _PYVENV_VALUES.get("version") != "3.12.10"
    or _PYVENV_VALUES.get("include-system-site-packages", "").casefold() != "false"
):
    raise RuntimeError("pyvenv.cfg/base-prefix binding differs")


def _authenticate_snapshot(root: str, expected: tuple[int, int, str], *, exclude_site: bool) -> None:
    rows: list[str] = []
    files = 0
    total = 0
    for base, directories, names in os.walk(root):
        filtered: list[str] = []
        for name in sorted(directories):
            path = os.path.join(base, name)
            if name == "__pycache__":
                continue
            relative = os.path.relpath(path, root).replace("\\", "/")
            if exclude_site and relative.split("/", 1)[0] == "site-packages":
                continue
            if int(getattr(os.lstat(path), "st_file_attributes", 0)) & _REPARSE_ATTRIBUTE:
                raise RuntimeError(f"reparse directory in runtime snapshot: {path}")
            filtered.append(name)
        directories[:] = filtered
        for name in sorted(names):
            if name.casefold().endswith((".pyc", ".pyo")):
                continue
            path = os.path.join(base, name)
            payload, record = _hold_plain_file(path)
            relative = os.path.relpath(path, root).replace("\\", "/")
            rows.append(f"{relative}\0{len(payload)}\0{record['sha256']}\n")
            files += 1
            total += len(payload)
    digest = _sha2.sha256("".join(sorted(rows)).encode("utf-8")).hexdigest()
    if (files, total, digest) != expected:
        raise RuntimeError(f"runtime snapshot differs: {root}")


_authenticate_snapshot(
    os.path.join(_BASE_PREFIX_NORM, "Lib"),
    (2397, 43971009, "81c5fbf6d2e0d268b9f8b2193a778e22393d9b8d4cded3dc7743eee659a3e519"),
    exclude_site=True,
)
_authenticate_snapshot(
    os.path.join(_BASE_PREFIX_NORM, "DLLs"),
    (42, 15330590, "f552f359861f2ce695d3c8e28af427c02571455e2b85e73bddf716fdf19b3500"),
    exclude_site=False,
)

import _frozen_importlib_external


class _AuthenticatedSourceLoader(_frozen_importlib_external.SourceFileLoader):
    def get_data(self, path: str) -> bytes:
        normalized = _norm(path)
        if normalized.casefold().endswith((".pyc", ".pyo")):
            raise OSError("bytecode loading is forbidden")
        try:
            return _SOURCE_BUFFERS[normalized]
        except KeyError as exc:
            raise ImportError(f"source is outside authenticated buffers: {path}") from exc

    def set_data(self, path: str, data: bytes, *_args: object, **_kwargs: object) -> None:
        del path, data


class _AuthenticatedExtensionLoader(_frozen_importlib_external.ExtensionFileLoader):
    def _verify(self) -> None:
        normalized = _norm(self.path)
        record = _HELD.get(normalized)
        if record is None or record["fd"] is None:
            raise ImportError(f"native extension is not held: {self.path}")
        current = os.lstat(normalized)
        identity = record["identity"]
        if (
            current.st_nlink != 1
            or int(getattr(current, "st_file_attributes", 0)) & _REPARSE_ATTRIBUTE
            or (int(current.st_dev), int(current.st_ino), int(current.st_size))
            != (identity[0], identity[1], identity[2])
        ):
            raise ImportError(f"native held identity differs: {self.path}")

    def create_module(self, spec: object) -> object:
        self._verify()
        value = super().create_module(spec)
        self._verify()
        return value

    def exec_module(self, module: object) -> None:
        self._verify()
        super().exec_module(module)
        self._verify()


_SITE_PACKAGES = _norm(os.path.join(_VENV_ROOT_NORM, "Lib", "site-packages"))


def _trusted_path_hook(path: str) -> object:
    normalized = _norm(path)
    roots = (
        _norm(os.path.join(_BASE_PREFIX_NORM, "Lib")),
        _norm(os.path.join(_BASE_PREFIX_NORM, "DLLs")),
        _BASE_PREFIX_NORM,
        _SITE_PACKAGES,
    )
    if not any(normalized == root or normalized.startswith(root + os.sep) for root in roots):
        raise ImportError
    return _frozen_importlib_external.FileFinder(
        path,
        (_AuthenticatedSourceLoader, _frozen_importlib_external.SOURCE_SUFFIXES),
        (_AuthenticatedExtensionLoader, _frozen_importlib_external.EXTENSION_SUFFIXES),
    )


sys.path_hooks[:] = [_trusted_path_hook]
sys.path_importer_cache.clear()

import base64
import csv
import json
import types


def _strict_json(payload: bytes, label: str) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise RuntimeError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise RuntimeError(f"{label} contains non-finite constant {value}")

    return json.loads(
        payload.decode("utf-8", errors="strict"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _is_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _workspace_path(relative: str) -> str:
    if (
        type(relative) is not str
        or not relative
        or "\\" in relative
        or relative.startswith("/")
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise RuntimeError(f"unsafe workspace-relative path: {relative!r}")
    value = _norm(os.path.join(_WORKSPACE_NORM, *relative.split("/")))
    if value != _WORKSPACE_NORM and not value.startswith(_WORKSPACE_NORM + os.sep):
        raise RuntimeError("workspace path escapes root")
    return value


def _authenticate_workspace_pin(pin: dict[str, object], label: str) -> bytes:
    if (
        type(pin) is not dict
        or set(pin) != {"path", "bytes", "sha256"}
        or type(pin["path"]) is not str
        or type(pin["bytes"]) is not int
        or not _is_sha(pin["sha256"])
    ):
        raise RuntimeError(f"{label} pin domain differs")
    payload = _assert_pin(_workspace_path(pin["path"]), pin["bytes"], pin["sha256"])
    return payload


_CONFIG_PIN = {
    "path": "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r3.json",
    "bytes": 13513,
    "sha256": "83eba89c97635ebc2bb38e22b7401b38ef980d89c7832cb3aab9abb490fd7cb8",
}
_BASE_CONFIG_PIN = {
    "path": "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r2.json",
    "bytes": 21292,
    "sha256": "5343b6d9a15ac7e0b2728b30f84db5974431b80070ea8519d51d9bfd8ad1dc12",
}
_CONFIG = _strict_json(_authenticate_workspace_pin(_CONFIG_PIN, "Gen6r3 config"), "Gen6r3 config")
_BASE_CONFIG = _strict_json(
    _authenticate_workspace_pin(_BASE_CONFIG_PIN, "frozen Gen6r2 config"),
    "frozen Gen6r2 config",
)
if type(_CONFIG) is not dict or type(_BASE_CONFIG) is not dict:
    raise RuntimeError("authenticated configuration object differs")


def _authenticate_distribution_records() -> dict[str, int]:
    records = _BASE_CONFIG["runtime_trust_contract"]["distribution_records"]
    if type(records) is not dict or len(records) != 12:
        raise RuntimeError("distribution RECORD declaration differs")
    members: set[str] = set()
    native = 0
    sources = 0
    bytecode_declarations = 0
    for distribution, pin in sorted(records.items()):
        record_payload = _authenticate_workspace_pin(pin, f"{distribution} RECORD")
        for row in csv.reader(record_payload.decode("utf-8", errors="strict").splitlines()):
            if len(row) != 3 or not row[0]:
                raise RuntimeError(f"{distribution} RECORD row differs")
            absolute = _norm(os.path.join(_SITE_PACKAGES, *row[0].split("/")))
            if absolute != _VENV_ROOT_NORM and not absolute.startswith(_VENV_ROOT_NORM + os.sep):
                raise RuntimeError(f"{distribution} RECORD member escapes venv")
            if absolute.casefold().endswith((".pyc", ".pyo")):
                if row[1] or row[2]:
                    raise RuntimeError(f"{distribution} RECORD bytecode row is unexpectedly bound")
                bytecode_declarations += 1
                continue
            payload, held = _hold_plain_file(absolute)
            if row[1]:
                algorithm, encoded = row[1].split("=", 1)
                if algorithm != "sha256":
                    raise RuntimeError(f"{distribution} RECORD algorithm differs")
                expected_digest = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).hex()
                if held["sha256"] != expected_digest:
                    raise RuntimeError(f"{distribution} RECORD member hash differs: {row[0]}")
            if row[2] and (not row[2].isdigit() or int(row[2]) != len(payload)):
                raise RuntimeError(f"{distribution} RECORD member size differs: {row[0]}")
            members.add(absolute)
            if absolute.casefold().endswith(".py"):
                sources += 1
            if absolute.casefold().endswith((".pyd", ".dll")):
                native += 1
    return {
        "distributions": len(records),
        "members": len(members),
        "sources": sources,
        "native": native,
        "ignored_unbound_bytecode_declarations": bytecode_declarations,
    }


_DISTRIBUTION_INVENTORY = _authenticate_distribution_records()
if _SITE_PACKAGES not in map(_norm, sys.path):
    sys.path.append(_SITE_PACKAGES)
    sys.path_importer_cache.clear()

_THIRD_PARTY_ENABLED = True
_TRUSTED_CTYPES_DEPTH += 1
try:
    import _ctypes
    import ctypes
    import mmap

    import numpy
    import pandas
    import pyarrow
    import scipy
    import sklearn
finally:
    _TRUSTED_CTYPES_DEPTH -= 1


def _verify_numerical_runtime() -> dict[str, str]:
    expected = {
        "numpy": numpy,
        "pandas": pandas,
        "pyarrow": pyarrow,
        "scipy": scipy,
        "sklearn": sklearn,
    }
    result: dict[str, str] = {}
    for name, module in expected.items():
        origin = _norm(getattr(module, "__file__", ""))
        if origin not in _HELD or not origin.startswith(_SITE_PACKAGES + os.sep):
            raise RuntimeError(f"numerical origin is not held: {name}")
        result[name] = origin
    for module in tuple(sys.modules.values()):
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        if type(origin) is str and origin not in {"built-in", "frozen"} and os.path.isabs(origin):
            normalized = _norm(origin)
            if _is_runtime_path_for_audit(normalized) and normalized not in _HELD:
                raise RuntimeError(f"loaded runtime origin is not held: {origin}")
    return result


_NUMERICAL_ORIGINS = _verify_numerical_runtime()
_RUNTIME_LOCKED = True


def _deny_winapi(*_args: object, **_kwargs: object) -> object:
    raise PermissionError("direct WinAPI entry is forbidden")


def _build_private_process_gate(create_process: object) -> tuple[object, object]:
    state: dict[str, object] = {"active": False, "command_line": None}

    def guarded(*args: object, **kwargs: object) -> object:
        command_line = args[1] if len(args) > 1 else kwargs.get("command_line")
        if state["active"] is not True or command_line != state["command_line"]:
            raise PermissionError("direct CreateProcess is forbidden")
        return create_process(*args, **kwargs)  # type: ignore[operator]

    def make_launcher(
        verify_controls: object,
        capability: object,
        session_sha256_expected: str,
    ) -> object:
        def launch(
            *, cell: int, session_sha256: str, prior_event_sha256: str
        ) -> dict[str, object]:
            global _PROCESS_DEPTH
            if (
                _MODE != "parent"
                or session_sha256 != session_sha256_expected
                or not 1 <= cell <= 15
                or not _is_sha(prior_event_sha256)
            ):
                raise PermissionError("cell worker launch request differs")
            verify_controls(capability)  # type: ignore[operator]
            command = [
                _VENV_PYTHON,
                "-I",
                "-S",
                "-B",
                "-X",
                f"pycache_prefix={_PYCACHE_PREFIX}",
                _BOOTSTRAP_PATH,
                "--cell-worker",
                str(cell),
                "--session",
                session_sha256,
                "--prior",
                prior_event_sha256,
            ]
            environment = {
                key: value
                for key, value in os.environ.items()
                if key.casefold()
                in {
                    "systemroot",
                    "windir",
                    "comspec",
                    "temp",
                    "tmp",
                    "path",
                    "p1_workspace_root",
                    "p1_data_dir",
                    "p1_powershell_host",
                }
            }
            environment.update(
                {
                    "P1_WORKSPACE_ROOT": _WORKSPACE_NORM,
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                }
            )
            command_line = subprocess.list2cmdline(command)  # type: ignore[name-defined]
            if state["active"] is not False:
                raise PermissionError("process gate is already active")
            state["active"] = True
            state["command_line"] = command_line
            _PROCESS_DEPTH += 1
            try:
                completed = subprocess.run(  # type: ignore[name-defined]
                    command,
                    cwd=_WORKSPACE_NORM,
                    env=environment,
                    shell=False,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="strict",
                    close_fds=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # type: ignore[name-defined]
                )
            finally:
                _PROCESS_DEPTH -= 1
                state["active"] = False
                state["command_line"] = None
            if completed.returncode != 0 or completed.stderr or not completed.stdout.endswith("\n"):
                raise RuntimeError(
                    f"cell worker failed closed: cell={cell} returncode={completed.returncode}"
                )
            result = _strict_json(completed.stdout.encode("utf-8"), f"cell {cell} stdout")
            if type(result) is not dict or result.get("cell") != cell:
                raise RuntimeError("cell worker stdout identity differs")
            return result

        return launch

    return guarded, make_launcher


_guarded_create_process, _MAKE_LAUNCH_CELL_WORKER = _build_private_process_gate(
    _BROKER_CREATE_PROCESS
)
del _BROKER_CREATE_PROCESS, _build_private_process_gate


_winapi.CreateFile = _deny_winapi
_winapi.CreateProcess = _guarded_create_process
_winapi.OpenProcess = _deny_winapi
_winapi.TerminateProcess = _deny_winapi
_winapi.CopyFile2 = _deny_winapi
_winapi.CreateFileMapping = _deny_winapi
_winapi.WriteFile = _deny_winapi
for _winapi_name in (
    "ConnectNamedPipe",
    "CreateJunction",
    "CreateNamedPipe",
    "DuplicateHandle",
    "ExitProcess",
    "MapViewOfFile",
    "OpenFileMapping",
    "ReadFile",
    "SetNamedPipeHandleState",
    "UnmapViewOfFile",
    "WaitNamedPipe",
):
    if hasattr(_winapi, _winapi_name):
        setattr(_winapi, _winapi_name, _deny_winapi)


def _deny_mutation(*_args: object, **_kwargs: object) -> object:
    raise PermissionError("direct mutation function is forbidden")


os.truncate = _deny_mutation
os.ftruncate = _deny_mutation
os.chmod = _deny_mutation
os.utime = _deny_mutation

for _ctypes_name in (
    "FreeLibrary",
    "LoadLibrary",
    "call_cdeclfunction",
    "call_function",
    "resize",
):
    if hasattr(_ctypes, _ctypes_name):
        setattr(_ctypes, _ctypes_name, _deny_winapi)
for _ctypes_name in (
    "CDLL",
    "LibraryLoader",
    "OleDLL",
    "PyDLL",
    "WinDLL",
    "cast",
    "cdll",
    "memmove",
    "memset",
    "oledll",
    "pydll",
    "pythonapi",
    "resize",
    "string_at",
    "windll",
    "wstring_at",
):
    if hasattr(ctypes, _ctypes_name):
        setattr(ctypes, _ctypes_name, _deny_winapi)
mmap.mmap = _deny_winapi


def _verify_real_firewall_probes() -> dict[str, bool]:
    probes = {
        "winapi_anonymous_writable_mapping": lambda: _winapi.CreateFileMapping(
            -1, None, 0x04, 0, 4096, None
        ),
        "winapi_direct_create_file": lambda: _winapi.CreateFile(
            _PYCACHE_PREFIX, _GENERIC_READ, _FILE_SHARE_READ, 0, _OPEN_EXISTING, 0, 0
        ),
        "winapi_direct_process": lambda: _winapi.CreateProcess(
            None, "cmd.exe /c exit 0", None, None, False, 0, None, None, None
        ),
        "ctypes_load_library": lambda: _ctypes.LoadLibrary("kernel32.dll", 0),
        "mmap_anonymous_writable": lambda: mmap.mmap(-1, 4096),
    }
    result: dict[str, bool] = {}
    for name, probe in probes.items():
        try:
            probe()
        except PermissionError:
            result[name] = True
        else:
            raise RuntimeError(f"real firewall probe unexpectedly succeeded: {name}")
    return result


_REAL_FIREWALL_PROBES = _verify_real_firewall_probes()

_OWNER_COMPILE_PINS = {
    "CONTRACT": {
        "path": "src/p1_qc/multiscale_cross_layer_offset_drift_contract_v6r3.py",
        "bytes": 24539,
        "sha256": "04f1325864ef983b70f1639844483b853067237d696146fe3dab934b0c7b7ef6",
    },
    "ENGINE": {
        "path": "src/p1_qc/multiscale_cross_layer_offset_drift_execution_v6r3.py",
        "bytes": 46446,
        "sha256": "7952e709788c5f398861df636680307930b00a98b5fcdfa78b9b6db8b38e08d1",
    },
    "VERIFIER": {
        "path": "src/p1_qc/multiscale_cross_layer_offset_drift_verifier_v6r3.py",
        "bytes": 10051,
        "sha256": "b016b1f767e2f2b12e79ef086b2193fceab484997531db292fc9a0162867571d",
    },
    "RUNNER": {
        "path": "scripts/run_p1_multiscale_cross_layer_offset_drift_unary_v6r3.py",
        "bytes": 1595,
        "sha256": "85f5dca139d5a98eaba12105e96ff4d3c5606ce2827fe998b384759ef22fea59",
    },
    "SCIENCE": {
        "path": "src/p1_qc/multiscale_cross_layer_offset_drift_v6r2.py",
        "bytes": 70846,
        "sha256": "4b5c74aeb54406416cda09576fac1d7e1569c6cbc46534ad92a9c2ea154a03af",
    },
    "LEGACY_ENGINE": {
        "path": "src/p1_qc/multiscale_cross_layer_offset_drift_execution_v6r2.py",
        "bytes": 63055,
        "sha256": "4d46d30cb1895a952e2925cbf6811609d776d133528dd43ff06f520a276fffc9",
    },
}
for _role, _pin in _OWNER_COMPILE_PINS.items():
    _authenticate_workspace_pin(_pin, f"owner compile role {_role}")


def _compile_module(name: str, pin: dict[str, object], injected: dict[str, object]) -> object:
    payload = _authenticate_workspace_pin(pin, name)
    module = types.ModuleType(name)
    module.__file__ = _workspace_path(pin["path"])
    module.__dict__.update(injected)
    sys.modules[name] = module
    exec(compile(payload, module.__file__, "exec", dont_inherit=True), module.__dict__)
    return module


_OUTPUT_HELD_RELATIVE: dict[str, dict[str, object]] = {}


def _output_path(relative: str) -> str:
    if (
        type(relative) is not str
        or not relative
        or "\\" in relative
        or relative.startswith("/")
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise RuntimeError("unsafe output-relative path")
    value = _norm(os.path.join(_OUTPUT_ROOT_NORM, *relative.split("/")))
    if not value.startswith(_OUTPUT_ROOT_NORM + os.sep):
        raise RuntimeError("output path escapes root")
    return value


def _rehash_record(record: dict[str, object]) -> bytes | None:
    if record["fd"] is None:
        return None
    payload = _read_fd(int(record["fd"]))
    if len(payload) != record["bytes"] or _sha2.sha256(payload).hexdigest() != record["sha256"]:
        raise RuntimeError(f"same-handle rehash differs: {record['path']}")
    return payload


def _create_exclusive_held_impl(
    path: str, payload: bytes, broker_create_file: object
) -> dict[str, object]:
    global _WRITE_DEPTH
    absolute = _norm(path)
    parent = os.path.dirname(absolute)
    _assert_plain_ancestors(parent)
    if os.path.lexists(absolute):
        raise FileExistsError(absolute)
    _WRITE_DEPTH += 1
    try:
        raw_handle = broker_create_file(  # type: ignore[operator]
            absolute,
            _GENERIC_READ | _GENERIC_WRITE,
            _FILE_SHARE_READ,
            0,
            _CREATE_NEW,
            _FILE_ATTRIBUTE_NORMAL | _FILE_FLAG_OPEN_REPARSE_POINT,
            0,
        )
        try:
            fd = msvcrt.open_osfhandle(
                raw_handle, os.O_RDWR | getattr(os, "O_BINARY", 0)
            )
        except BaseException:
            _winapi.CloseHandle(raw_handle)
            raise
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += os.write(fd, view[written:])
        os.fsync(fd)
        opened = os.fstat(fd)
        observed = _read_fd(fd)
        after = os.lstat(absolute)
        if (
            observed != payload
            or after.st_nlink != 1
            or int(getattr(after, "st_file_attributes", 0)) & _REPARSE_ATTRIBUTE
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
        ):
            os.close(fd)
            raise RuntimeError("exclusive same-handle write identity differs")
        record: dict[str, object] = {
            "fd": fd,
            "path": absolute,
            "bytes": len(payload),
            "sha256": _sha2.sha256(payload).hexdigest(),
            "identity": (
                int(opened.st_dev),
                int(opened.st_ino),
                int(opened.st_size),
                int(opened.st_mtime_ns),
                int(opened.st_mode),
            ),
            "payload": payload,
        }
        _HELD[absolute] = record
        _HELD_PATHS.add(absolute)
        return record
    finally:
        _WRITE_DEPTH -= 1


def _bind_create_exclusive_held(implementation: object, broker_create_file: object) -> object:
    def create(path: str, payload: bytes) -> dict[str, object]:
        return implementation(path, payload, broker_create_file)  # type: ignore[operator, no-any-return]

    return create


_create_exclusive_held = _bind_create_exclusive_held(
    _create_exclusive_held_impl, _BROKER_CREATE_FILE
)
del _BROKER_CREATE_FILE, _bind_create_exclusive_held, _create_exclusive_held_impl


def _pin(relative: str, record: dict[str, object]) -> dict[str, object]:
    return {"path": relative, "bytes": record["bytes"], "sha256": record["sha256"]}


def _authenticated_output_bytes(pin: dict[str, object], label: str) -> bytes:
    if type(pin) is not dict or set(pin) != {"path", "bytes", "sha256"}:
        raise RuntimeError(f"{label} output pin domain differs")
    relative = pin["path"]
    if type(relative) is not str:
        raise RuntimeError(f"{label} output path differs")
    record = _OUTPUT_HELD_RELATIVE.get(relative)
    if record is None:
        payload, record = _hold_plain_file(_output_path(relative))
        record["payload"] = payload
        _OUTPUT_HELD_RELATIVE[relative] = record
    payload = _rehash_record(record)
    if len(payload) != pin["bytes"] or record["sha256"] != pin["sha256"]:
        raise RuntimeError(f"{label} output pin differs")
    return payload


def _output_pin(relative: str) -> dict[str, object]:
    record = _OUTPUT_HELD_RELATIVE.get(relative)
    if record is None:
        payload, record = _hold_plain_file(_output_path(relative))
        record["payload"] = payload
        _OUTPUT_HELD_RELATIVE[relative] = record
    _rehash_record(record)
    return _pin(relative, record)


def _scan_output_files() -> list[str]:
    if not os.path.isdir(_OUTPUT_ROOT_NORM):
        return []
    result: list[str] = []
    for base, directories, names in os.walk(_OUTPUT_ROOT_NORM):
        for name in sorted(directories):
            path = os.path.join(base, name)
            if int(getattr(os.lstat(path), "st_file_attributes", 0)) & _REPARSE_ATTRIBUTE:
                raise RuntimeError("reparse output directory is forbidden")
        for name in sorted(names):
            path = _norm(os.path.join(base, name))
            payload, record = _hold_plain_file(path)
            relative = os.path.relpath(path, _OUTPUT_ROOT_NORM).replace("\\", "/")
            record["payload"] = payload
            _OUTPUT_HELD_RELATIVE[relative] = record
            result.append(relative)
    return sorted(result)


def _list_output_paths(prefix: str) -> list[str]:
    return [path for path in _scan_output_files() if path.startswith(prefix)]


def _output_inventory(*, final: bool) -> dict[str, object]:
    paths = _scan_output_files()
    rows: list[str] = []
    for relative in paths:
        record = _OUTPUT_HELD_RELATIVE[relative]
        payload = _rehash_record(record)
        rows.append(f"{relative}\0{len(payload)}\0{record['sha256']}\n")
    return {
        "files": len(paths),
        "paths": paths,
        "inventory_sha256": _sha2.sha256("".join(rows).encode("utf-8")).hexdigest(),
        "same_handle_final_rehashes": len(paths) if final else 0,
    }


def _make_runtime_context(mode: str) -> dict[str, object]:
    merged = json.loads(json.dumps(_BASE_CONFIG))
    merged["canonical_paths"]["output"] = _CONFIG["canonical_paths"]["output"]

    def authenticated_bytes_for_pin(pin: dict[str, object], label: str) -> bytes:
        return _authenticate_workspace_pin(pin, label)

    def authenticated_json_for_pin(pin: dict[str, object], label: str) -> object:
        return _strict_json(_authenticate_workspace_pin(pin, label), label)

    def strict_dynamic_json_for_pin(pin: dict[str, object], label: str) -> object:
        return _strict_json(_authenticated_output_bytes(pin, label), label)

    def authenticated_train_bytes_for_pin(pin: dict[str, object], label: str) -> bytes:
        data_root = os.environ.get("P1_DATA_DIR")
        if not data_root or not os.path.isabs(data_root):
            raise RuntimeError("P1_DATA_DIR must be externally supplied and absolute")
        if pin != _BASE_CONFIG["source_pins"]["train.csv"]:
            raise PermissionError("only the pinned train buffer is available")
        path = _norm(os.path.join(data_root, pin["path"]))
        return _assert_pin(path, pin["bytes"], pin["sha256"])

    return {
        "mode": mode,
        "cell": _CELL,
        "prior_event_sha256": _PRIOR_ARGUMENT,
        "workspace": _WORKSPACE_NORM,
        "config": _CONFIG,
        "base_config": _BASE_CONFIG,
        "all_owner_roles_authenticated": True,
        "bootstrap_documents_prevalidated": mode == "check-only",
        "authenticated_bytes_for_pin": authenticated_bytes_for_pin,
        "authenticated_json_for_pin": authenticated_json_for_pin,
        "strict_dynamic_json_for_pin": strict_dynamic_json_for_pin,
        "authenticated_output_bytes": _authenticated_output_bytes,
        "authenticated_train_bytes_for_pin": authenticated_train_bytes_for_pin,
        "verify_numerical_runtime": _verify_numerical_runtime,
        "output_pin": _output_pin,
        "list_output_paths": _list_output_paths,
        "output_inventory": _output_inventory,
    }


_CONTEXT = _make_runtime_context(_MODE)
_CONTRACT = _compile_module(
    "_p1_v6r3_authenticated_contract",
    _OWNER_COMPILE_PINS["CONTRACT"],
    {"_P1_V6R3_BOOTSTRAP_CONTEXT": _CONTEXT},
)

if _MODE == "check-only":
    _SUMMARY = _CONTRACT.static_preflight(require_future_state_absent=True)
    for _authority_name in (
        "_BROKER_MKDIR",
        "_MAKE_LAUNCH_CELL_WORKER",
        "_create_exclusive_held",
    ):
        globals().pop(_authority_name, None)
    _dangerous_main_names = {
        "_BOOTSTRAP_CREATE_FILE",
        "_BOOTSTRAP_CREATE_PROCESS",
        "_BOOTSTRAP_OS_MKDIR",
        "_BROKER_CREATE_FILE",
        "_BROKER_CREATE_PROCESS",
        "_BROKER_MKDIR",
        "_CAPABILITY",
        "_CAPABILITY_CALLBACKS",
        "_EXPECTED_CHILD_COMMAND_LINE",
        "_MAKE_LAUNCH_CELL_WORKER",
        "_RAW_CREATE_FILE",
        "_RAW_CREATE_PROCESS",
        "_create_exclusive_held",
    }
    if _dangerous_main_names & globals().keys():
        raise RuntimeError("dangerous bootstrap authority remains in check-only __main__")
    _SUMMARY["runtime"] = {
        "python": sys.version,
        "held_files": len(_HELD),
        "distribution_inventory": _DISTRIBUTION_INVENTORY,
        "numerical_origins": _NUMERICAL_ORIGINS,
        "pre_script_path_backed_modules": sorted(_PRE_SCRIPT_ORIGINS),
        "pre_script_builtin_modules": ["_codecs_kr", "_multibytecodec"],
        "real_firewall_probes": _REAL_FIREWALL_PROBES,
        "direct_main_authority_absent": True,
        "pycache_regular_file_sentinel_held": (
            _PYCACHE_PREFIX in _HELD
            and _HELD[_PYCACHE_PREFIX]["fd"] is not None
            and os.path.isfile(_PYCACHE_PREFIX)
        ),
    }
    _FINAL_REHASHES = sum(1 for record in _HELD.values() if _rehash_record(record) is not None)
    _SUMMARY["runtime"]["same_handle_final_rehashes"] = _FINAL_REHASHES
    if _EARLY_WRITES != 0 or not os.path.isfile(_PYCACHE_PREFIX):
        raise RuntimeError("check-only created forbidden state")
    print(_canonical(_SUMMARY).decode("utf-8"))
    raise SystemExit(0)

import secrets
import subprocess


def _bootstrap_pin() -> dict[str, object]:
    payload, record = _hold_plain_file(_BOOTSTRAP_PATH)
    return {"path": os.path.relpath(_BOOTSTRAP_PATH, _WORKSPACE_NORM).replace("\\", "/"), "bytes": len(payload), "sha256": record["sha256"]}


def _verify_external_tcb_qa(qa: dict[str, object]) -> None:
    tcb = qa.get("external_powershell_tcb")
    if type(tcb) is not dict or set(tcb) != {
        "declared_outer_tcb",
        "distribution_inventory",
        "host",
        "pycache_regular_file_sentinel",
        "stage0",
        "startup_trust",
    }:
        raise RuntimeError("independent QA PowerShell TCB domain differs")
    if tcb["host"] != {
        "path_source": "externally_injected_absolute_P1_POWERSHELL_HOST",
        "version": "7.6.4",
        "bytes": 301368,
        "sha256": "db6dd81183fe57d22e03b911ec9a30a2fd7c40542e97743615355a6fb44f458f",
    }:
        raise RuntimeError("independent QA PowerShell host pin differs")
    if tcb["distribution_inventory"] != {
        "root_source": "directory_containing_P1_POWERSHELL_HOST",
        "files": 983,
        "bytes": 296034085,
        "sha256": "fcbbc18499e682ca08a0860dcb3b5353099a2a846e9eedc50afbb0c28ed728dc",
        "handles_held_through_host_start": True,
    }:
        raise RuntimeError("independent QA PowerShell distribution pin differs")
    expected_startup = _CONFIG["owner_roles"]["STARTUP_TRUST"]
    if (
        tcb["startup_trust"] != expected_startup
        or tcb["pycache_regular_file_sentinel"] != expected_startup
        or tcb["declared_outer_tcb"]
        != [
            "fresh_independent_QA_process",
            "Windows_kernel_file_sharing",
            "user_authorized_Codex_environment",
        ]
    ):
        raise RuntimeError("independent QA startup/outer TCB binding differs")
    stage0 = tcb["stage0"]
    if type(stage0) is not dict or set(stage0) != {
        "encoded_command",
        "host_argv",
        "mode_environment",
        "reference_source",
        "utf16le_payload",
    }:
        raise RuntimeError("independent QA stage-0 domain differs")
    if (
        stage0["host_argv"]
        != ["-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand"]
        or stage0["mode_environment"] != "P1_LAUNCH_MODE=CheckOnly|Execute"
    ):
        raise RuntimeError("independent QA stage-0 invocation differs")
    source_pin = stage0["reference_source"]
    if type(source_pin) is not dict or source_pin.get("path") != _CONFIG["canonical_paths"]["powershell_stage0_reference"]:
        raise RuntimeError("independent QA stage-0 path differs")
    source = _authenticate_workspace_pin(source_pin, "PowerShell stage-0 from QA")
    source_text = source.decode("utf-8", errors="strict")
    utf16le = source_text.encode("utf-16-le", errors="strict")
    encoded = base64.b64encode(utf16le)
    if stage0["utf16le_payload"] != {
        "bytes": len(utf16le),
        "sha256": _sha2.sha256(utf16le).hexdigest(),
    } or stage0["encoded_command"] != {
        "characters": len(encoded),
        "ascii_sha256": _sha2.sha256(encoded).hexdigest(),
    }:
        raise RuntimeError("independent QA stage-0 encoding pin differs")
    launcher = qa.get("external_launcher")
    if type(launcher) is not dict:
        raise RuntimeError("independent QA launcher pin differs")
    if (
        f"$launcherStream.Length -ne {launcher.get('bytes')}" not in source_text
        or source_text.count(str(launcher.get("sha256"))) != 2
    ):
        raise RuntimeError("stage-0 source does not bind the QA launcher pin")


def _validate_control_documents() -> dict[str, object]:
    control = _CONFIG["canonical_paths"]["control"]
    qa_relative = f"{control}/pre_execution_qa.json"
    authorization_relative = f"{control}/execution_authorization.json"
    qa_payload, qa_record = _hold_plain_file(_workspace_path(qa_relative))
    auth_payload, auth_record = _hold_plain_file(_workspace_path(authorization_relative))
    qa = _strict_json(qa_payload, "Gen6r3 independent QA")
    authorization = _strict_json(auth_payload, "Gen6r3 execution authorization")
    qa_pin = _pin(qa_relative, qa_record)
    auth_pin = _pin(authorization_relative, auth_record)
    if (
        type(qa) is not dict
        or qa.get("schema_version") != "p1_multiscale_cross_layer_offset_drift_unary.v6r3.independent_qa.v1"
        or qa.get("generation") != _CONTRACT.GENERATION
        or qa.get("verdict") != "GO"
        or qa.get("p0_count") != 0
        or qa.get("p1_count") != 0
        or qa.get("open_findings") != []
        or qa.get("bootstrap") != _bootstrap_pin()
        or qa.get("config") != _CONFIG_PIN
        or qa.get("external_launcher", {}).get("path")
        != _CONFIG["canonical_paths"]["external_launcher"]
        or not _is_sha(qa.get("external_launcher", {}).get("sha256"))
    ):
        raise RuntimeError("fresh independent QA receipt semantics differ")
    _authenticate_workspace_pin(qa["external_launcher"], "external launcher from QA")
    _verify_external_tcb_qa(qa)
    if (
        type(authorization) is not dict
        or authorization.get("schema_version")
        != "p1_multiscale_cross_layer_offset_drift_unary.v6r3.execution_authorization.v1"
        or authorization.get("generation") != _CONTRACT.GENERATION
        or authorization.get("authorized") is not True
        or authorization.get("scope") != "ONE_RESEARCH_CURVE_NO_CANDIDATE_TEST_LEDGER_OR_UPLOAD"
        or authorization.get("qa_receipt") != qa_pin
        or authorization.get("bootstrap") != _bootstrap_pin()
        or authorization.get("config") != _CONFIG_PIN
        or authorization.get("external_launcher") != qa["external_launcher"]
        or authorization.get("external_powershell_tcb") != qa["external_powershell_tcb"]
    ):
        raise RuntimeError("explicit execution authorization semantics differ")
    return {
        "qa": qa,
        "qa_pin": qa_pin,
        "qa_record": qa_record,
        "authorization": authorization,
        "authorization_pin": auth_pin,
        "authorization_record": auth_record,
    }


def _mkdir_plain_impl(path: str, broker_mkdir: object) -> None:
    global _WRITE_DEPTH
    absolute = _norm(path)
    if os.path.lexists(absolute):
        info = os.lstat(absolute)
        if not stat.S_ISDIR(info.st_mode) or int(getattr(info, "st_file_attributes", 0)) & _REPARSE_ATTRIBUTE:
            raise RuntimeError("required output directory is not plain")
        return
    _assert_plain_ancestors(os.path.dirname(absolute))
    _WRITE_DEPTH += 1
    try:
        broker_mkdir(absolute)  # type: ignore[operator]
    finally:
        _WRITE_DEPTH -= 1
    info = os.lstat(absolute)
    if not stat.S_ISDIR(info.st_mode) or int(getattr(info, "st_file_attributes", 0)) & _REPARSE_ATTRIBUTE:
        raise RuntimeError("created output directory identity differs")


def _bind_mkdir_plain(implementation: object, broker_mkdir: object) -> object:
    def mkdir_plain(path: str) -> None:
        implementation(path, broker_mkdir)  # type: ignore[operator]

    return mkdir_plain


_mkdir_plain = _bind_mkdir_plain(_mkdir_plain_impl, _BROKER_MKDIR)
del _BROKER_MKDIR, _bind_mkdir_plain, _mkdir_plain_impl


def _prepare_output_directories() -> None:
    _mkdir_plain(_OUTPUT_ROOT_NORM)
    for relative in ("commitments", "metrics"):
        _mkdir_plain(_output_path(relative))
    for cell in range(1, 16):
        prefix = _output_path(f"cells/cell_{cell:02d}")
        _mkdir_plain(prefix)
        for child in ("models", "inner_predictions", "commitments"):
            _mkdir_plain(os.path.join(prefix, child))


def _parent_paths() -> tuple[str, ...]:
    result: list[str] = ["commitments/session.json"]
    for fold in _CONTRACT.FOLDS:
        result.extend((f"commitments/fold_{fold}.json", f"metrics/fold_{fold}.json"))
    result.append("commitments/predictions_complete.json")
    result.extend(f"metrics/fraction_{tag}.json" for tag in _CONTRACT.FRACTION_TAGS)
    result.extend(
        (
            "split_audit.json",
            "selective_target_audit.json",
            "metrics.json",
            "learning_curve_evidence.json",
            "result.json",
            "resource_audit.json",
            "manifest.json",
            "manifest.sha256",
            "final_seal.json",
        )
    )
    if len(result) != 22:
        raise RuntimeError("parent output sequence arithmetic differs")
    return tuple(result)


def _worker_paths(cell: int) -> tuple[str, ...]:
    prefix = f"cells/cell_{cell:02d}"
    result: list[str] = []
    for block in (1, 2, 3):
        result.extend(
            (
                f"{prefix}/models/inner_{block}.json",
                f"{prefix}/inner_predictions/block_{block}.bin",
                f"{prefix}/commitments/inner_{block}.json",
            )
        )
    result.extend(
        (
            f"{prefix}/models/outer.json",
            f"{prefix}/outer_prediction.bin",
            f"{prefix}/cell_receipt.json",
        )
    )
    return tuple(result)


def _make_capability_system(
    *,
    role: str,
    session_sha256: str,
    cell: int | None,
    controls: dict[str, object],
    lock_record: dict[str, object],
    expected_head: str,
    create_exclusive: object,
) -> tuple[object, dict[str, object]]:
    registry: dict[int, object] = {}
    state: dict[int, dict[str, object]] = {}
    constructor_key = object()

    class _OpaqueCapability:
        __slots__ = ("_identity",)

        def __init__(self, key: object) -> None:
            if key is not constructor_key:
                raise PermissionError("opaque capability constructor is bootstrap-only")
            self._identity = object()

    capability = _OpaqueCapability(constructor_key)
    registry[id(capability)] = capability
    state[id(capability)] = {
        "role": role,
        "cell": cell,
        "session_sha256": session_sha256,
        "counters": {},
        "closed": False,
        "write_paths": _parent_paths() if role == "parent" else _worker_paths(int(cell)),
        "write_index": 0,
        "expected_head": expected_head,
    }

    def guard(value: object, _entry: str) -> object:
        current = state.get(id(value))
        if registry.get(id(value)) is not value or current is None or current["closed"] is not False:
            raise PermissionError("opaque live capability required")
        return value

    def snapshot(value: object) -> dict[str, object]:
        guard(value, "snapshot")
        current = state[id(value)]
        return {
            "role": current["role"],
            "cell": current["cell"],
            "session_sha256": current["session_sha256"],
            "counters": dict(current["counters"]),
        }

    def bump(value: object, name: str, amount: int) -> int:
        guard(value, "counter")
        counters = state[id(value)]["counters"]
        counters[name] = counters.get(name, 0) + amount
        return counters[name]

    def close(value: object) -> dict[str, int]:
        guard(value, "close")
        current = state[id(value)]
        current["closed"] = True
        registry.pop(id(value), None)
        return dict(current["counters"])

    def verify_controls(value: object) -> None:
        guard(value, "control reverify")
        for key in ("qa_record", "authorization_record"):
            _rehash_record(controls[key])
        _rehash_record(lock_record)
        lock = _strict_json(_rehash_record(lock_record), "attempt lock")
        current = state[id(value)]
        if (
            type(lock) is not dict
            or lock.get("session_sha256") != current["session_sha256"]
            or lock.get("qa_receipt") != controls["qa_pin"]
            or lock.get("authorization") != controls["authorization_pin"]
        ):
            raise PermissionError("live control/lock/session binding differs")

    def writer(value: object, relative: str, payload: bytes) -> dict[str, object]:
        guard(value, "exclusive writer")
        verify_controls(value)
        current = state[id(value)]
        paths = current["write_paths"]
        index = current["write_index"]
        if index >= len(paths) or relative != paths[index]:
            raise PermissionError("exact output creation order differs")
        if current["role"] == "cell_worker" and not relative.startswith(
            f"cells/cell_{int(current['cell']):02d}/"
        ):
            raise PermissionError("worker output escaped its exact cell")
        if relative.endswith(".json") and (
            "/commitments/inner_" in relative
            or relative.endswith("/cell_receipt.json")
            or relative.startswith("commitments/")
        ):
            event = _strict_json(payload, relative)
            if (
                type(event) is not dict
                or event.get("prior_event_sha256") != current["expected_head"]
                or not _is_sha(event.get("event_sha256"))
            ):
                raise PermissionError("persisted commitment prior/head differs")
            body = dict(event)
            claimed = body.pop("event_sha256")
            if _sha2.sha256(_canonical(body)).hexdigest() != claimed:
                raise PermissionError("persisted commitment hash differs")
        record = create_exclusive(_output_path(relative), payload)  # type: ignore[operator]
        _OUTPUT_HELD_RELATIVE[relative] = record
        current["write_index"] = index + 1
        if relative.endswith(".json") and (
            "/commitments/inner_" in relative
            or relative.endswith("/cell_receipt.json")
            or relative.startswith("commitments/")
        ):
            current["expected_head"] = _strict_json(payload, relative)["event_sha256"]
        return _pin(relative, record)

    def set_expected_head(value: object, expected: str) -> None:
        guard(value, "set expected head")
        if not _is_sha(expected):
            raise PermissionError("expected chain head differs")
        state[id(value)]["expected_head"] = expected

    return capability, {
        "capability_guard": guard,
        "capability_snapshot": snapshot,
        "capability_counter_bump": bump,
        "capability_close": close,
        "exclusive_output_writer": writer,
        "verify_controls": verify_controls,
        "set_expected_head": set_expected_head,
    }


_CONTROLS = _validate_control_documents()
_CONTROL_ROOT = _workspace_path(_CONFIG["canonical_paths"]["control"])
_LOCK_PATH = _workspace_path(_CONFIG["canonical_paths"]["attempt_lock"])

if _MODE == "parent":
    if os.path.lexists(_LOCK_PATH) or os.path.lexists(_OUTPUT_ROOT_NORM):
        raise RuntimeError("one-shot Gen6r3 attempt state already exists")
    _SESSION = _sha2.sha256(
        _canonical(
            {
                "generation": _CONTRACT.GENERATION,
                "qa": _CONTROLS["qa_pin"],
                "authorization": _CONTROLS["authorization_pin"],
                "nonce": secrets.token_hex(32),
            }
        )
    ).hexdigest()
    _LOCK_BODY = {
        "schema_version": "p1_multiscale_cross_layer_offset_drift_unary.v6r3.attempt_lock.v1",
        "generation": _CONTRACT.GENERATION,
        "session_sha256": _SESSION,
        "qa_receipt": _CONTROLS["qa_pin"],
        "authorization": _CONTROLS["authorization_pin"],
        "config": _CONFIG_PIN,
        "commitment_genesis_sha256": _sha2.sha256(
            b"p1_v6r3_process_isolated_commitment_genesis"
        ).hexdigest(),
        "worker_processes": 15,
    }
    _LOCK_RECORD = _create_exclusive_held(_LOCK_PATH, _canonical(_LOCK_BODY) + b"\n")
    _prepare_output_directories()
    _INITIAL_HEAD = _LOCK_BODY["commitment_genesis_sha256"]
else:
    if not (_is_sha(_SESSION_ARGUMENT) and _is_sha(_PRIOR_ARGUMENT)):
        raise RuntimeError("worker session/prior arguments differ")
    _LOCK_PAYLOAD, _LOCK_RECORD = _hold_plain_file(_LOCK_PATH)
    _LOCK_BODY = _strict_json(_LOCK_PAYLOAD, "attempt lock")
    if (
        type(_LOCK_BODY) is not dict
        or _LOCK_BODY.get("session_sha256") != _SESSION_ARGUMENT
        or _LOCK_BODY.get("qa_receipt") != _CONTROLS["qa_pin"]
        or _LOCK_BODY.get("authorization") != _CONTROLS["authorization_pin"]
        or not os.path.isdir(_OUTPUT_ROOT_NORM)
    ):
        raise RuntimeError("worker live attempt lock differs")
    _SESSION = _SESSION_ARGUMENT
    _INITIAL_HEAD = _PRIOR_ARGUMENT

globals().pop("_mkdir_plain", None)
globals().pop("_prepare_output_directories", None)

_CONTEXT["bootstrap_documents_prevalidated"] = True
_CONTEXT["require_engine_capability"] = _CONTRACT.require_engine_capability

_LEGACY_CONTEXT = dict(_CONTEXT)
_LEGACY_CONTEXT["config"] = json.loads(json.dumps(_BASE_CONFIG))
_LEGACY_CONTEXT["config"]["canonical_paths"]["output"] = _CONFIG["canonical_paths"]["output"]
_SCIENCE = _compile_module(
    "_p1_v6r3_authenticated_frozen_science",
    _OWNER_COMPILE_PINS["SCIENCE"],
    {"_P1_V6R2_BOOTSTRAP_CONTEXT": _LEGACY_CONTEXT},
)
_LEGACY_ENGINE = _compile_module(
    "_p1_v6r3_authenticated_legacy_helpers",
    _OWNER_COMPILE_PINS["LEGACY_ENGINE"],
    {
        "_P1_V6R2_BOOTSTRAP_CONTEXT": _LEGACY_CONTEXT,
        "_P1_V6R2_AUTH_CONTRACT": _CONTRACT,
        "_P1_V6R2_AUTH_SCIENCE": _SCIENCE,
    },
)
_VERIFIER = _compile_module(
    "_p1_v6r3_authenticated_verifier",
    _OWNER_COMPILE_PINS["VERIFIER"],
    {
        "_P1_V6R3_BOOTSTRAP_CONTEXT": _CONTEXT,
        "_P1_V6R3_AUTH_CONTRACT": _CONTRACT,
    },
)
_ENGINE = _compile_module(
    "_p1_v6r3_authenticated_engine",
    _OWNER_COMPILE_PINS["ENGINE"],
    {
        "_P1_V6R3_BOOTSTRAP_CONTEXT": _CONTEXT,
        "_P1_V6R3_AUTH_CONTRACT": _CONTRACT,
        "_P1_V6R3_AUTH_SCIENCE": _SCIENCE,
        "_P1_V6R3_AUTH_LEGACY_ENGINE": _LEGACY_ENGINE,
        "_P1_V6R3_AUTH_VERIFIER": _VERIFIER,
    },
)
_RUNNER = _compile_module(
    "_p1_v6r3_authenticated_runner",
    _OWNER_COMPILE_PINS["RUNNER"],
    {
        "_P1_V6R3_BOOTSTRAP_CONTEXT": _CONTEXT,
        "_P1_V6R3_AUTH_CONTRACT": _CONTRACT,
        "_P1_V6R3_AUTH_ENGINE": _ENGINE,
    },
)


def _scan_chain_head() -> tuple[str, int]:
    session, _pin_value = _strict_json(
        _authenticated_output_bytes(_output_pin("commitments/session.json"), "session"),
        "session",
    ), None
    prior = session["prior_event_sha256"]
    claimed = session["event_sha256"]
    body = dict(session)
    body.pop("event_sha256")
    if _sha2.sha256(_canonical(body)).hexdigest() != claimed:
        raise RuntimeError("session chain hash differs")
    prior = claimed
    completed = 0
    for cell in range(1, 16):
        receipt_relative = f"cells/cell_{cell:02d}/cell_receipt.json"
        if receipt_relative not in _scan_output_files():
            break
        for block in (1, 2, 3):
            relative = f"cells/cell_{cell:02d}/commitments/inner_{block}.json"
            event = _strict_json(
                _authenticated_output_bytes(_output_pin(relative), relative), relative
            )
            body = dict(event)
            event_sha = body.pop("event_sha256")
            if event.get("prior_event_sha256") != prior or _sha2.sha256(
                _canonical(body)
            ).hexdigest() != event_sha:
                raise RuntimeError("inner persisted chain differs")
            prior = event_sha
        receipt = _strict_json(
            _authenticated_output_bytes(_output_pin(receipt_relative), receipt_relative),
            receipt_relative,
        )
        body = dict(receipt)
        event_sha = body.pop("event_sha256")
        if receipt.get("prior_event_sha256") != prior or _sha2.sha256(
            _canonical(body)
        ).hexdigest() != event_sha:
            raise RuntimeError("cell persisted chain differs")
        prior = event_sha
        completed = cell
        if cell % 5 == 0:
            fold = _CONTRACT.FOLDS[cell // 5 - 1]
            relative = f"commitments/fold_{fold}.json"
            if relative in _scan_output_files():
                event = _strict_json(
                    _authenticated_output_bytes(_output_pin(relative), relative), relative
                )
                body = dict(event)
                event_sha = body.pop("event_sha256")
                if event.get("prior_event_sha256") != prior or _sha2.sha256(
                    _canonical(body)
                ).hexdigest() != event_sha:
                    raise RuntimeError("fold persisted chain differs")
                prior = event_sha
            elif cell < 15 and f"cells/cell_{cell + 1:02d}/cell_receipt.json" in _scan_output_files():
                raise RuntimeError("next fold cell preceded fold commitment")
    return prior, completed


def _validate_persisted_chain_impl(
    *,
    expected_next_cell: int,
    expected_head: str,
    session_sha256: str,
    set_expected_head: object,
) -> None:
    if session_sha256 != _SESSION or not 1 <= expected_next_cell <= 16:
        raise RuntimeError("persisted chain session/cell request differs")
    head, completed = _scan_chain_head()
    if head != expected_head or completed not in {expected_next_cell - 1, expected_next_cell - 2}:
        raise RuntimeError("persisted chain head/cardinality differs")
    set_expected_head(head)  # type: ignore[operator]


def _is_fold_committed(fold: str, pin_sha256: str) -> bool:
    relative = f"commitments/fold_{fold}.json"
    try:
        pin = _output_pin(relative)
        event = _strict_json(_authenticated_output_bytes(pin, relative), relative)
    except (FileNotFoundError, RuntimeError):
        return False
    return pin["sha256"] == pin_sha256 and event.get("fold") == fold


def _bind_runtime_capability(
    capability: object,
    callbacks: dict[str, object],
    runner: object,
    make_launcher: object,
    validate_implementation: object,
) -> tuple[dict[str, object], object, object]:
    def set_expected_head(expected: str) -> None:
        callbacks["set_expected_head"](capability, expected)  # type: ignore[operator]

    def validate_persisted_chain(
        *, expected_next_cell: int, expected_head: str, session_sha256: str
    ) -> None:
        validate_implementation(  # type: ignore[operator]
            expected_next_cell=expected_next_cell,
            expected_head=expected_head,
            session_sha256=session_sha256,
            set_expected_head=set_expected_head,
        )

    launch_cell_worker = make_launcher(  # type: ignore[operator]
        callbacks["verify_controls"], capability, _SESSION
    )

    def execute() -> object:
        return runner.execute(capability)  # type: ignore[attr-defined, no-any-return]

    def verify() -> None:
        callbacks["verify_controls"](capability)  # type: ignore[operator]

    return (
        {
            "validate_persisted_chain": validate_persisted_chain,
            "is_fold_committed": _is_fold_committed,
            "launch_cell_worker": launch_cell_worker,
        },
        execute,
        verify,
    )


_capability, _capability_callbacks = _make_capability_system(
    role=_MODE,
    session_sha256=_SESSION,
    cell=_CELL,
    controls=_CONTROLS,
    lock_record=_LOCK_RECORD,
    expected_head=_INITIAL_HEAD,
    create_exclusive=_create_exclusive_held,
)
_CONTEXT.update(_capability_callbacks)
_runtime_context, _runtime_execute, _runtime_verify = _bind_runtime_capability(
    _capability,
    _capability_callbacks,
    _RUNNER,
    _MAKE_LAUNCH_CELL_WORKER,
    _validate_persisted_chain_impl,
)
_CONTEXT.update(_runtime_context)
for _authority_name in (
    "_MAKE_LAUNCH_CELL_WORKER",
    "_bind_runtime_capability",
    "_capability",
    "_capability_callbacks",
    "_create_exclusive_held",
    "_make_capability_system",
    "_runtime_context",
    "_validate_persisted_chain_impl",
):
    globals().pop(_authority_name, None)
_forbidden_main_names = {
    "_BOOTSTRAP_CREATE_FILE",
    "_BOOTSTRAP_CREATE_PROCESS",
    "_BOOTSTRAP_OS_MKDIR",
    "_BROKER_CREATE_FILE",
    "_BROKER_CREATE_PROCESS",
    "_BROKER_MKDIR",
    "_CAPABILITY",
    "_CAPABILITY_CALLBACKS",
    "_EXPECTED_CHILD_COMMAND_LINE",
    "_RAW_CREATE_FILE",
    "_RAW_CREATE_PROCESS",
}
if _forbidden_main_names & globals().keys():
    raise RuntimeError("dangerous bootstrap authority remains in __main__")

_RESULT = _runtime_execute()  # type: ignore[operator]
del _runtime_execute
if _MODE == "cell_worker":
    _runtime_verify()  # type: ignore[operator]
    del _runtime_verify
    print(_canonical(_RESULT).decode("utf-8"))
else:
    del _runtime_verify
    sentinel_record = _HELD.get(_PYCACHE_PREFIX)
    if (
        sentinel_record is None
        or sentinel_record["fd"] is None
        or not os.path.isfile(_PYCACHE_PREFIX)
        or _rehash_record(sentinel_record) is None
    ):
        raise RuntimeError("regular-file pycache sentinel binding changed")
    print(_canonical(_RESULT).decode("utf-8"))
