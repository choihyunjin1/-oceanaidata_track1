"""Hermetic trust bootstrap for the P2 Layer-4 r3 compatibility verifier v3."""

from __future__ import annotations

import sys as _sys

_IDENTITY = "P2_JOINT_HYDROGRAPHIC_MULTITASK_LAYER4_R3_COMPATIBILITY_BOOTSTRAP_V3"
_FORBIDDEN_ROOTS = ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "torch")
_FORBIDDEN_ENGINE = "p2_restore.joint_hydrographic_multitask_layer4_execution_r3"
_BOOT_IMPORTS = {
    "nt",
    "_winapi",
    "msvcrt",
    "_frozen_importlib",
    "_frozen_importlib_external",
    "_hashlib",
}
_EXPECTED_ABSENT_PLATFORM_PROBES = {"fcntl", "posix"}
_WRITE_FLAGS = 1 | 2 | 8 | 256 | 512 | 1024
_GENERIC_READ = 0x80000000
_OPEN_EXISTING = 3
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_MUTATION_OR_PROCESS_EVENTS = {
    "os.remove",
    "os.rename",
    "os.replace",
    "os.rmdir",
    "os.mkdir",
    "os.chmod",
    "os.chown",
    "os.truncate",
    "os.utime",
    "os.symlink",
    "os.link",
    "shutil.copyfile",
    "shutil.copymode",
    "shutil.copystat",
    "shutil.copytree",
    "shutil.move",
    "subprocess.Popen",
    "_winapi.CreateProcess",
    "os.system",
    "os.startfile",
    "socket.connect",
    "socket.bind",
}
_FIREWALL = {
    "installed": False,
    "dependency_ready": False,
    "write_delete_process_attempts": 0,
    "forbidden_import_attempts": 0,
    "unexpected_import_attempts": 0,
    "bytecode_read_attempts": 0,
}
_ALLOWED_IMPORTS: set[str] = set(_BOOT_IMPORTS)
_UNEXPECTED_IMPORT_NAMES: list[str] = []


class BootstrapV3Error(RuntimeError):
    """The v3 isolated trust bootstrap failed closed."""


def _is_forbidden_import(name: str) -> bool:
    return name == _FORBIDDEN_ENGINE or any(
        name == root or name.startswith(root + ".") for root in _FORBIDDEN_ROOTS
    )


def _audit_hook(event: str, arguments: tuple[object, ...]) -> None:
    if event == "open":
        target = arguments[0] if arguments else None
        mode = arguments[1] if len(arguments) > 1 else None
        flags = arguments[2] if len(arguments) > 2 else 0
        bytecode_read = (
            isinstance(target, str) and target.casefold().endswith((".pyc", ".pyo"))
        ) or (isinstance(target, bytes) and target.casefold().endswith((b".pyc", b".pyo")))
        if bytecode_read:
            _FIREWALL["bytecode_read_attempts"] += 1
            raise PermissionError("P2 v3 audit firewall rejected bytecode access")
        textual_write = isinstance(mode, str) and any(character in mode for character in "wax+")
        flag_write = isinstance(flags, int) and bool(flags & _WRITE_FLAGS)
        if textual_write or flag_write:
            _FIREWALL["write_delete_process_attempts"] += 1
            raise PermissionError("P2 v3 audit firewall rejected a write-capable open")
    if event in _MUTATION_OR_PROCESS_EVENTS:
        _FIREWALL["write_delete_process_attempts"] += 1
        raise PermissionError(f"P2 v3 audit firewall rejected {event}")
    if event == "_winapi.CreateFile":
        access = arguments[1] if len(arguments) > 1 else None
        creation = arguments[3] if len(arguments) > 3 else None
        if access != _GENERIC_READ or creation != _OPEN_EXISTING:
            _FIREWALL["write_delete_process_attempts"] += 1
            raise PermissionError("P2 v3 audit firewall rejected write/create WinAPI access")
    if event == "import" and arguments:
        name = str(arguments[0])
        if _is_forbidden_import(name):
            _FIREWALL["forbidden_import_attempts"] += 1
            raise ImportError(f"P2 v3 forbidden import rejected: {name}")
        root = name.split(".", 1)[0]
        if (
            name not in _ALLOWED_IMPORTS
            and name not in _EXPECTED_ABSENT_PLATFORM_PROBES
            and root not in _sys.builtin_module_names
        ):
            _FIREWALL["unexpected_import_attempts"] += 1
            _UNEXPECTED_IMPORT_NAMES.append(name)
            raise ImportError(f"P2 v3 unauthenticated import rejected: {name}")


def _assert_initial_runtime() -> None:
    flags = _sys.flags
    if (
        flags.isolated != 1
        or flags.no_site != 1
        or flags.dont_write_bytecode != 1
        or flags.ignore_environment != 1
        or flags.no_user_site != 1
        or flags.safe_path is not True
        or _sys.dont_write_bytecode is not True
        or _sys.version_info[:3] != (3, 12, 10)
    ):
        raise BootstrapV3Error("canonical python -I -S -B runtime is required")
    base = _sys.base_prefix
    expected_path = [
        base + "\\python312.zip",
        base + "\\DLLs",
        base + "\\Lib",
        base,
    ]
    if _sys.path != expected_path:
        raise BootstrapV3Error("initial isolated sys.path changed")
    expected_meta = [
        ("_frozen_importlib", "BuiltinImporter"),
        ("_frozen_importlib", "FrozenImporter"),
        ("_frozen_importlib_external", "PathFinder"),
    ]
    observed_meta = [
        (getattr(finder, "__module__", ""), getattr(finder, "__qualname__", ""))
        for finder in _sys.meta_path
    ]
    if observed_meta != expected_meta:
        raise BootstrapV3Error("initial import finder chain changed")
    expected_modules = {
        "__future__",
        "__main__",
        "_abc",
        "_codecs",
        "_codecs_kr",
        "_frozen_importlib",
        "_frozen_importlib_external",
        "_imp",
        "_io",
        "_multibytecodec",
        "_signal",
        "_thread",
        "_warnings",
        "_weakref",
        "abc",
        "builtins",
        "codecs",
        "encodings",
        "encodings.aliases",
        "encodings.cp949",
        "encodings.utf_8",
        "io",
        "marshal",
        "nt",
        "sys",
        "time",
        "winreg",
        "zipimport",
    }
    if set(_sys.modules) != expected_modules:
        raise BootstrapV3Error(
            "pre-bootstrap module set changed: "
            f"extra={sorted(set(_sys.modules) - expected_modules)}, "
            f"missing={sorted(expected_modules - set(_sys.modules))}"
        )


_assert_initial_runtime()
_sys.addaudithook(_audit_hook)
_FIREWALL["installed"] = True

import _frozen_importlib as _import_core  # noqa: E402
import _frozen_importlib_external as _import_external  # noqa: E402
import _winapi as _winapi  # noqa: E402
import msvcrt as _msvcrt  # noqa: E402
import nt as _nt  # noqa: E402

if any(
    getattr(_sys.modules[name].__spec__, "origin", None) not in {"built-in", "frozen"}
    for name in ("nt", "_winapi", "msvcrt", "_frozen_importlib", "_frozen_importlib_external")
):
    raise BootstrapV3Error("phase-zero module origin changed")


_REPARSE_ATTRIBUTE = 0x400
_FILE_SHARE_READ = 1
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_PHASE_LOCKS: dict[str, dict[str, object]] = {}
_STDLIB_BUFFERS: dict[str, bytes] = {}
_STDLIB_PATHS: dict[str, str] = {}
_SOURCE_MODULES: dict[str, tuple[str, bool]] = {}
_NATIVE_MODULES: dict[str, str] = {}
_AUTH_FINDER: object | None = None
_INITIAL_SYS_PATH = tuple(_sys.path)


def _case(path: str) -> str:
    return _nt._getfullpathname(path).casefold()


def _identity_stat(value: object) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(getattr(value, "st_file_attributes", 0)),
    )


def _full(path: str) -> str:
    return _nt._getfullpathname(path)


def _join(parent: str, child: str) -> str:
    return parent.rstrip("\\/") + "\\" + child.replace("/", "\\")


def _reject_reparse_chain(path: str, *, require_target: bool = True) -> str:
    full = _full(path)
    root, tail = _nt._path_splitroot(full)
    current = root
    missing = False
    for part in [item for item in tail.replace("/", "\\").split("\\") if item]:
        current = _join(current, part)
        try:
            info = _nt.lstat(current)
        except FileNotFoundError:
            missing = True
            continue
        if missing:
            raise BootstrapV3Error("filesystem ancestor identity is inconsistent")
        if int(getattr(info, "st_file_attributes", 0)) & _REPARSE_ATTRIBUTE:
            raise BootstrapV3Error(f"link/reparse ancestor is forbidden: {current}")
    if require_target:
        try:
            _nt.lstat(full)
        except FileNotFoundError as exc:
            raise FileNotFoundError(full) from exc
    return full


def _lock_path(path: str, *, directory: bool) -> dict[str, object]:
    full = _reject_reparse_chain(path)
    key = _case(full)
    existing = _PHASE_LOCKS.get(key)
    if existing is not None:
        if bool(existing["directory"]) != directory:
            raise BootstrapV3Error("stable handle kind changed")
        return existing
    flags = _FILE_FLAG_OPEN_REPARSE_POINT
    if directory:
        flags |= _FILE_FLAG_BACKUP_SEMANTICS
    handle = _winapi.CreateFile(
        full,
        _winapi.GENERIC_READ,
        _FILE_SHARE_READ,
        0,
        _winapi.OPEN_EXISTING,
        flags,
        0,
    )
    try:
        descriptor = _msvcrt.open_osfhandle(
            handle,
            _nt.O_RDONLY | getattr(_nt, "O_BINARY", 0),
        )
    except BaseException:
        _winapi.CloseHandle(handle)
        raise
    descriptor_stat = _nt.fstat(descriptor)
    path_stat = _nt.lstat(full)
    if (
        int(getattr(descriptor_stat, "st_file_attributes", 0)) & _REPARSE_ATTRIBUTE
        or int(getattr(path_stat, "st_file_attributes", 0)) & _REPARSE_ATTRIBUTE
    ):
        _nt.close(descriptor)
        raise BootstrapV3Error("opened stable handle resolves to a reparse point")
    if _identity_stat(descriptor_stat) != _identity_stat(path_stat):
        _nt.close(descriptor)
        raise BootstrapV3Error(
            "stable handle and path identity differ: "
            f"descriptor={_identity_stat(descriptor_stat)!r}, "
            f"path={_identity_stat(path_stat)!r}, target={full}"
        )
    raw = b""
    if not directory:
        blocks: list[bytes] = []
        while True:
            block = _nt.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        raw = b"".join(blocks)
        if len(raw) != int(descriptor_stat.st_size):
            _nt.close(descriptor)
            raise BootstrapV3Error("stable handle read was truncated")
    entry: dict[str, object] = {
        "path": full,
        "descriptor": descriptor,
        "directory": directory,
        "identity": _identity_stat(descriptor_stat),
        "raw": raw,
    }
    _PHASE_LOCKS[key] = entry
    return entry


def _right_rotate(value: int, amount: int) -> int:
    return ((value >> amount) | ((value << (32 - amount)) & 0xFFFFFFFF)) & 0xFFFFFFFF


def _pure_sha256(raw: bytes) -> str:
    constants = (
        0x428A2F98,
        0x71374491,
        0xB5C0FBCF,
        0xE9B5DBA5,
        0x3956C25B,
        0x59F111F1,
        0x923F82A4,
        0xAB1C5ED5,
        0xD807AA98,
        0x12835B01,
        0x243185BE,
        0x550C7DC3,
        0x72BE5D74,
        0x80DEB1FE,
        0x9BDC06A7,
        0xC19BF174,
        0xE49B69C1,
        0xEFBE4786,
        0x0FC19DC6,
        0x240CA1CC,
        0x2DE92C6F,
        0x4A7484AA,
        0x5CB0A9DC,
        0x76F988DA,
        0x983E5152,
        0xA831C66D,
        0xB00327C8,
        0xBF597FC7,
        0xC6E00BF3,
        0xD5A79147,
        0x06CA6351,
        0x14292967,
        0x27B70A85,
        0x2E1B2138,
        0x4D2C6DFC,
        0x53380D13,
        0x650A7354,
        0x766A0ABB,
        0x81C2C92E,
        0x92722C85,
        0xA2BFE8A1,
        0xA81A664B,
        0xC24B8B70,
        0xC76C51A3,
        0xD192E819,
        0xD6990624,
        0xF40E3585,
        0x106AA070,
        0x19A4C116,
        0x1E376C08,
        0x2748774C,
        0x34B0BCB5,
        0x391C0CB3,
        0x4ED8AA4A,
        0x5B9CCA4F,
        0x682E6FF3,
        0x748F82EE,
        0x78A5636F,
        0x84C87814,
        0x8CC70208,
        0x90BEFFFA,
        0xA4506CEB,
        0xBEF9A3F7,
        0xC67178F2,
    )
    state = [
        0x6A09E667,
        0xBB67AE85,
        0x3C6EF372,
        0xA54FF53A,
        0x510E527F,
        0x9B05688C,
        0x1F83D9AB,
        0x5BE0CD19,
    ]
    message = bytearray(raw)
    bit_length = len(message) * 8
    message.append(0x80)
    while len(message) % 64 != 56:
        message.append(0)
    message.extend(bit_length.to_bytes(8, "big"))
    for offset in range(0, len(message), 64):
        words = [
            int.from_bytes(message[index : index + 4], "big")
            for index in range(offset, offset + 64, 4)
        ]
        for index in range(16, 64):
            a = words[index - 15]
            b = words[index - 2]
            s0 = _right_rotate(a, 7) ^ _right_rotate(a, 18) ^ (a >> 3)
            s1 = _right_rotate(b, 17) ^ _right_rotate(b, 19) ^ (b >> 10)
            words.append((words[index - 16] + s0 + words[index - 7] + s1) & 0xFFFFFFFF)
        a, b, c, d, e, f, g, h = state
        for index in range(64):
            s1 = _right_rotate(e, 6) ^ _right_rotate(e, 11) ^ _right_rotate(e, 25)
            choose = (e & f) ^ ((~e) & g)
            temp1 = (h + s1 + choose + constants[index] + words[index]) & 0xFFFFFFFF
            s0 = _right_rotate(a, 2) ^ _right_rotate(a, 13) ^ _right_rotate(a, 22)
            majority = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (s0 + majority) & 0xFFFFFFFF
            h, g, f, e, d, c, b, a = (
                g,
                f,
                e,
                (d + temp1) & 0xFFFFFFFF,
                c,
                b,
                a,
                (temp1 + temp2) & 0xFFFFFFFF,
            )
        state = [
            (left + right) & 0xFFFFFFFF
            for left, right in zip(state, (a, b, c, d, e, f, g, h), strict=True)
        ]
    return "".join(f"{value:08x}" for value in state)


if (
    _pure_sha256(b"") != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    or _pure_sha256(b"abc") != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
):
    raise BootstrapV3Error("pure SHA-256 self-test failed")


def _argument_root() -> str:
    values = _sys.argv[1:]
    if values.count("--root") != 1:
        raise BootstrapV3Error("exactly one --root is required")
    index = values.index("--root")
    if index + 1 >= len(values) or values[index + 1].startswith("--"):
        raise BootstrapV3Error("--root requires a value")
    return _full(values[index + 1])


_WORKSPACE_TEXT = _reject_reparse_chain(_argument_root())
if not bool(_nt.stat(_WORKSPACE_TEXT).st_mode & 0o040000):
    raise BootstrapV3Error("workspace is not a directory")
_BASE_TEXT = _full(_sys.base_prefix)
_lock_path(_BASE_TEXT, directory=True)
_lock_path(_join(_BASE_TEXT, "DLLs"), directory=True)


def _verify_phase_pin(path: str, *, size: int, digest: str, label: str) -> bytes:
    entry = _lock_path(path, directory=False)
    raw = entry["raw"]
    if not isinstance(raw, bytes) or len(raw) != size or _pure_sha256(raw) != digest:
        raise BootstrapV3Error(f"phase-zero pin changed: {label}")
    return raw


_HASHLIB_PATH = _join(_BASE_TEXT, "DLLs/_hashlib.pyd")
_CRYPTO_PATH = _join(_BASE_TEXT, "DLLs/libcrypto-3.dll")
_verify_phase_pin(
    _HASHLIB_PATH,
    size=68976,
    digest="03d233dbace599168eaffec823de6ed7beec8c2a4ebfe9b8a3d8e042a59af3ba",
    label="_hashlib.pyd",
)
_verify_phase_pin(
    _CRYPTO_PATH,
    size=5231472,
    digest="ccfffddcd3defb8d899026298af9af43bc186130f8483d77e97c93233d5f27d7",
    label="libcrypto-3.dll",
)

import _hashlib as _trusted_hashlib  # noqa: E402

if _case(str(_trusted_hashlib.__spec__.origin)) != _case(_HASHLIB_PATH):
    raise BootstrapV3Error("authenticated _hashlib origin changed")


def _fast_sha256(raw: bytes) -> str:
    return _trusted_hashlib.openssl_sha256(raw).hexdigest()


def _inventory_stdlib() -> tuple[int, int, int, str]:
    records: list[tuple[str, str, str]] = []
    for entry in sorted(_nt.scandir(_BASE_TEXT), key=lambda item: item.name.casefold()):
        if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".dll"):
            records.append(("f", entry.name, entry.path))
    pending = [("DLLs", _join(_BASE_TEXT, "DLLs")), ("Lib", _join(_BASE_TEXT, "Lib"))]
    while pending:
        relative, current = pending.pop(0)
        directory_entry = _lock_path(current, directory=True)
        del directory_entry
        records.append(("d", relative.replace("\\", "/"), current))
        children = sorted(_nt.scandir(current), key=lambda item: item.name.casefold())
        for child in children:
            attributes = int(getattr(child.stat(follow_symlinks=False), "st_file_attributes", 0))
            if attributes & _REPARSE_ATTRIBUTE:
                raise BootstrapV3Error(f"stdlib reparse entry is forbidden: {child.path}")
            child_relative = relative + "/" + child.name
            if child.is_dir(follow_symlinks=False):
                if child.name not in {"__pycache__", "site-packages"}:
                    pending.append((child_relative, child.path))
            elif child.is_file(follow_symlinks=False) and not child.name.endswith((".pyc", ".pyo")):
                records.append(("f", child_relative, child.path))
            elif not child.is_dir(follow_symlinks=False) and not child.is_file(
                follow_symlinks=False
            ):
                raise BootstrapV3Error("stdlib contains a special filesystem entry")
    records.sort(key=lambda item: item[1])
    digest = _trusted_hashlib.openssl_sha256()
    directories = files = total = 0
    for kind, relative, path in records:
        relative = relative.replace("\\", "/")
        if kind == "d":
            directories += 1
            digest.update(("d\0" + relative + "\n").encode("utf-8"))
            continue
        locked = _lock_path(path, directory=False)
        raw = locked["raw"]
        if not isinstance(raw, bytes):
            raise BootstrapV3Error("stdlib file buffer changed type")
        file_digest = _fast_sha256(raw)
        files += 1
        total += len(raw)
        _STDLIB_BUFFERS[relative] = raw
        _STDLIB_PATHS[relative] = str(locked["path"])
        digest.update((f"f\0{relative}\0{len(raw)}\0{file_digest}\n").encode())
    return directories, files, total, digest.hexdigest()


_STDLIB_INVENTORY = _inventory_stdlib()
if _STDLIB_INVENTORY != (
    199,
    2443,
    66487423,
    "5cc5d4b2f90199292a4334a6530eaa90c288fd45723ba5290295a3803d13eeba",
):
    raise BootstrapV3Error(f"authenticated stdlib inventory changed: {_STDLIB_INVENTORY}")


for _relative in sorted(_STDLIB_BUFFERS):
    if _relative.startswith("Lib/") and _relative.endswith(".py"):
        _module_parts = _relative[4:-3].split("/")
        _is_package = _module_parts[-1] == "__init__"
        if _is_package:
            _module_parts = _module_parts[:-1]
        if _module_parts:
            _module_name = ".".join(_module_parts)
            if _module_name in _SOURCE_MODULES:
                raise BootstrapV3Error(f"duplicate stdlib source module: {_module_name}")
            _SOURCE_MODULES[_module_name] = (_relative, _is_package)
    elif _relative.startswith("DLLs/") and _relative.lower().endswith(".pyd"):
        _module_name = _relative.rsplit("/", 1)[-1][:-4]
        if _module_name in _NATIVE_MODULES:
            raise BootstrapV3Error(f"duplicate stdlib native module: {_module_name}")
        _NATIVE_MODULES[_module_name] = _relative


class _AuthenticatedSourceLoader:
    def __init__(self, fullname: str, relative: str, is_package: bool) -> None:
        self.fullname = fullname
        self.relative = relative
        self.is_package = is_package

    def create_module(self, spec: object) -> None:
        del spec
        return None

    def exec_module(self, module: object) -> None:
        raw = _STDLIB_BUFFERS[self.relative]
        path = _STDLIB_PATHS[self.relative]
        module.__file__ = path
        module.__cached__ = None
        module.__loader__ = self
        module.__package__ = self.fullname if self.is_package else self.fullname.rpartition(".")[0]
        code = compile(raw, path, "exec", dont_inherit=True, optimize=0)
        exec(code, module.__dict__)  # noqa: S102


class _AuthenticatedFinder:
    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> object | None:
        del path, target
        source = _SOURCE_MODULES.get(fullname)
        if source is not None:
            relative, is_package = source
            loader = _AuthenticatedSourceLoader(fullname, relative, is_package)
            spec = _import_core.ModuleSpec(
                fullname,
                loader,
                origin=_STDLIB_PATHS[relative],
                is_package=is_package,
            )
            spec._set_fileattr = True
            if is_package:
                spec.submodule_search_locations = [_STDLIB_PATHS[relative].rsplit("\\", 1)[0]]
            return spec
        native = _NATIVE_MODULES.get(fullname)
        if native is not None:
            origin = _STDLIB_PATHS[native]
            loader = _import_external.ExtensionFileLoader(fullname, origin)
            spec = _import_core.ModuleSpec(fullname, loader, origin=origin, is_package=False)
            spec._set_fileattr = True
            return spec
        return None


_ALLOWED_IMPORTS.update(_SOURCE_MODULES)
_ALLOWED_IMPORTS.update(_NATIVE_MODULES)
_AUTH_FINDER = _AuthenticatedFinder()
_sys.meta_path[:] = [
    _import_core.BuiltinImporter,
    _import_core.FrozenImporter,
    _AUTH_FINDER,
]
_FIREWALL["dependency_ready"] = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import types  # noqa: E402
from collections.abc import Mapping  # noqa: E402
from pathlib import Path, PurePosixPath  # noqa: E402
from typing import Any  # noqa: E402

_LOWER_SHA = re.compile(r"[0-9a-f]{64}\Z")
_EXPECTED_ROLES = ("CONFIG", "HELPER", "CLI", "TESTS")
_TRUSTED_PINS: dict[str, dict[str, Any]] = {
    "CONFIG": {
        "path": "configs/experiments/p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v3.json",
        "bytes": 8236,
        "sha256": "08bf804b0367951a43a80ffa22dc1067e3c1a1bc516da8351a64e97203a597ec",
    },
    "HELPER": {
        "path": "src/p2_restore/joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v3.py",
        "bytes": 28473,
        "sha256": "4f7d327afadac3ce9ff244ada52aca851df8fd39e089cdd310745880643eabc3",
    },
    "CLI": {
        "path": "scripts/verify_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v3.py",
        "bytes": 1843,
        "sha256": "a7feac34b70ee3160f98c8bba79aa266a8a4da0e43000edb398211ee463bcd60",
    },
    "TESTS": {
        "path": "tests/test_p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v3.py",
        "bytes": 21385,
        "sha256": "1ab2268d59b2bece36e125ce8060105befc7e3aa025ca44c914e993d861c63ce",
    },
}
_HELPER_MODULE = "p2_restore.joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v3"
_CLI_MODULE = "p2_restore.joint_hydrographic_multitask_layer4_r3_compatibility_cli_v3"


def _relative_parts(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise BootstrapV3Error("path is not canonical POSIX relative text")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise BootstrapV3Error("path is not a contained relative path")
    return pure.parts


def _workspace_path(
    relative: str,
    *,
    must_exist: bool = True,
    kind: str | None = None,
) -> Path:
    candidate = Path(_WORKSPACE_TEXT).joinpath(*_relative_parts(relative))
    lexical = Path(_reject_reparse_chain(str(candidate), require_target=must_exist))
    resolved = lexical.resolve(strict=must_exist)
    workspace = Path(_WORKSPACE_TEXT).resolve(strict=True)
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise BootstrapV3Error(f"path escaped workspace: {relative}") from exc
    if must_exist:
        if kind == "file" and not resolved.is_file():
            raise BootstrapV3Error(f"regular file required: {relative}")
        if kind == "directory" and not resolved.is_dir():
            raise BootstrapV3Error(f"directory required: {relative}")
    return resolved


def _relative_from_path(path: Path) -> str:
    resolved = Path(_reject_reparse_chain(str(path))).resolve(strict=True)
    workspace = Path(_WORKSPACE_TEXT).resolve(strict=True)
    try:
        return resolved.relative_to(workspace).as_posix()
    except ValueError as exc:
        raise BootstrapV3Error("path is outside workspace") from exc


def _pin_tuple(pin: Mapping[str, Any]) -> tuple[str, int, str]:
    if set(pin) != {"path", "bytes", "sha256"}:
        raise BootstrapV3Error("pin field set changed")
    relative = pin.get("path")
    size = pin.get("bytes")
    digest = pin.get("sha256")
    if (
        not isinstance(relative, str)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(digest, str)
        or _LOWER_SHA.fullmatch(digest) is None
    ):
        raise BootstrapV3Error("pin value changed")
    _relative_parts(relative)
    return relative, size, digest


def _unique_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BootstrapV3Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise BootstrapV3Error(f"non-finite JSON number: {value}")


def _parse_json_buffer(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapV3Error(f"invalid JSON: {label}") from exc
    if not isinstance(value, dict):
        raise BootstrapV3Error(f"JSON root must be an object: {label}")
    return value


class _StableRegistry:
    def __init__(self) -> None:
        self._pins: dict[str, dict[str, Any]] = {}
        self._tree_directories: dict[str, tuple[int, int, int, int, int]] = {}

    def lock_pin(self, pin: Mapping[str, Any], *, label: str) -> bytes:
        relative, size, digest = _pin_tuple(pin)
        exact = {"path": relative, "bytes": size, "sha256": digest}
        prior = self._pins.get(relative)
        if prior is not None and prior != exact:
            raise BootstrapV3Error(f"conflicting stable pin: {label}")
        path = _workspace_path(relative, kind="file")
        entry = _lock_path(str(path), directory=False)
        raw = entry["raw"]
        if not isinstance(raw, bytes) or len(raw) != size or _fast_sha256(raw) != digest:
            raise BootstrapV3Error(f"stable authenticated bytes changed: {label}")
        self._pins[relative] = exact
        return raw

    def lock_many(self, pins: Mapping[str, Mapping[str, Any]]) -> None:
        for relative in sorted(pins):
            pin = pins[relative]
            if pin.get("path") != relative:
                raise BootstrapV3Error("stable registry key/path mismatch")
            self.lock_pin(pin, label=f"registry {relative}")

    def bytes_for_path(self, path: Path) -> bytes:
        relative = _relative_from_path(path)
        pin = self._pins.get(relative)
        if pin is None:
            raise BootstrapV3Error(f"unregistered semantic read: {relative}")
        entry = _PHASE_LOCKS.get(_case(str(_workspace_path(relative, kind="file"))))
        if entry is None or not isinstance(entry.get("raw"), bytes):
            raise BootstrapV3Error("stable semantic buffer is unavailable")
        return entry["raw"]

    def seal_tree(
        self,
        relative_root: str,
        pins: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        root = _workspace_path(relative_root, kind="directory")
        prefix = relative_root.rstrip("/") + "/"
        expected = {path for path in pins if path.startswith(prefix)}
        observed: set[str] = set()
        entries: list[dict[str, Any]] = []
        pending = [root]
        while pending:
            current = pending.pop(0)
            locked = _lock_path(str(current), directory=True)
            current_relative = current.relative_to(root).as_posix()
            if current_relative != ".":
                entries.append({"path": current_relative, "type": "directory"})
            self._tree_directories[_relative_from_path(current)] = locked["identity"]
            children = sorted(os.scandir(current), key=lambda item: item.name)
            for child in children:
                info = child.stat(follow_symlinks=False)
                if int(getattr(info, "st_file_attributes", 0)) & _REPARSE_ATTRIBUTE:
                    raise BootstrapV3Error(f"tree reparse entry is forbidden: {child.path}")
                child_path = Path(child.path)
                if child.is_dir(follow_symlinks=False):
                    pending.append(child_path)
                elif child.is_file(follow_symlinks=False):
                    relative = _relative_from_path(child_path)
                    pin = pins.get(relative)
                    if pin is None:
                        raise BootstrapV3Error(f"tree file is not allowlisted: {relative}")
                    raw = self.lock_pin(pin, label=f"tree {relative}")
                    observed.add(relative)
                    entries.append(
                        {
                            "path": child_path.relative_to(root).as_posix(),
                            "type": "file",
                            "bytes": len(raw),
                            "sha256": _fast_sha256(raw),
                        }
                    )
                else:
                    raise BootstrapV3Error("tree contains a special entry")
        if observed != expected:
            missing = sorted(expected - observed)
            raise BootstrapV3Error(f"tree allowlist has missing files: {missing[:3]}")
        entries.sort(key=lambda item: str(item["path"]))
        files = [entry for entry in entries if entry["type"] == "file"]
        directories = [entry for entry in entries if entry["type"] == "directory"]
        payload = (
            json.dumps(
                entries,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        return {
            "directories": len(directories),
            "files": len(files),
            "file_bytes": sum(int(entry["bytes"]) for entry in files),
            "algorithm": "SHA256_CANONICAL_JSON_SORTED_RELATIVE_PATH_TYPE_BYTES_SHA256_WITH_LF",
            "sha256": _fast_sha256(payload),
        }

    def final_reverify(self) -> dict[str, Any]:
        total = 0
        for relative in sorted(self._pins):
            pin = self._pins[relative]
            path = _workspace_path(relative, kind="file")
            entry = _PHASE_LOCKS.get(_case(str(path)))
            if entry is None:
                raise BootstrapV3Error("stable file handle disappeared")
            descriptor = int(entry["descriptor"])
            if _identity_stat(_nt.fstat(descriptor)) != entry["identity"]:
                raise BootstrapV3Error("stable descriptor identity changed")
            if _identity_stat(_nt.stat(str(path))) != entry["identity"]:
                raise BootstrapV3Error("stable path identity changed")
            _nt.lseek(descriptor, 0, 0)
            digest = _trusted_hashlib.openssl_sha256()
            size = 0
            while True:
                block = _nt.read(descriptor, 1024 * 1024)
                if not block:
                    break
                size += len(block)
                digest.update(block)
            if size != pin["bytes"] or digest.hexdigest() != pin["sha256"]:
                raise BootstrapV3Error("same-handle final hash changed")
            total += size
        for relative, identity in sorted(self._tree_directories.items()):
            path = _workspace_path(relative, kind="directory")
            entry = _PHASE_LOCKS.get(_case(str(path)))
            if (
                entry is None
                or entry["identity"] != identity
                or _identity_stat(_nt.fstat(int(entry["descriptor"]))) != identity
                or _identity_stat(_nt.stat(str(path))) != identity
            ):
                raise BootstrapV3Error("held directory identity changed")
        return {
            "files": len(self._pins),
            "file_bytes": total,
            "held_directories": len(self._tree_directories),
            "same_handle_final_rehashes": len(self._pins),
            "share_write_allowed": False,
            "share_delete_allowed": False,
        }


_STABLE_REGISTRY = _StableRegistry()


def _exec_buffer(
    *,
    name: str,
    path: Path,
    raw: bytes,
    injected: Mapping[str, Any],
) -> Any:
    if name in _sys.modules:
        raise BootstrapV3Error(f"authenticated module already loaded: {name}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__dict__.update(injected)
    _sys.modules[name] = module
    try:
        code = compile(raw, str(path), "exec", dont_inherit=True, optimize=0)
        exec(code, module.__dict__)  # noqa: S102
    except BaseException:
        _sys.modules.pop(name, None)
        raise
    return module


def _loaded_forbidden() -> list[str]:
    return sorted(name for name in _sys.modules if _is_forbidden_import(name))


def _module_origin_report() -> dict[str, Any]:
    file_modules: list[dict[str, str]] = []
    for name, module in sorted(_sys.modules.items()):
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        if origin in {None, "built-in", "frozen"}:
            continue
        origin_text = str(origin)
        case = _case(origin_text)
        matches = [relative for relative, path in _STDLIB_PATHS.items() if _case(path) == case]
        if not matches:
            if name == "__main__" and case == _case(str(Path(__file__).resolve(strict=True))):
                continue
            raise BootstrapV3Error(f"loaded module origin is not authenticated: {name}")
        file_modules.append({"module": name, "origin": matches[0]})
    if any("site-packages" in item["origin"].casefold() for item in file_modules):
        raise BootstrapV3Error("site-packages module entered the runtime")
    return {
        "file_backed_modules": file_modules,
        "file_backed_module_count": len(file_modules),
        "third_party_distributions": [],
        "third_party_record_files": [],
        "numerical_distributions_imported": [],
    }


def _assert_runtime() -> None:
    _assert_flags = _sys.flags
    if (
        _assert_flags.isolated != 1
        or _assert_flags.no_site != 1
        or _assert_flags.dont_write_bytecode != 1
        or _assert_flags.ignore_environment != 1
        or _assert_flags.no_user_site != 1
        or _assert_flags.safe_path is not True
        or _sys.dont_write_bytecode is not True
        or tuple(_sys.path) != _INITIAL_SYS_PATH
        or _sys.meta_path
        != [_import_core.BuiltinImporter, _import_core.FrozenImporter, _AUTH_FINDER]
        or _loaded_forbidden()
        or _FIREWALL
        != {
            "installed": True,
            "dependency_ready": True,
            "write_delete_process_attempts": 0,
            "forbidden_import_attempts": 0,
            "unexpected_import_attempts": 0,
            "bytecode_read_attempts": 0,
        }
    ):
        raise BootstrapV3Error(
            "isolated runtime or zero-attempt firewall changed: "
            f"flags={_sys.flags!r}, meta={_sys.meta_path!r}, "
            f"forbidden={_loaded_forbidden()!r}, firewall={_FIREWALL!r}, "
            f"unexpected_names={_UNEXPECTED_IMPORT_NAMES!r}"
        )
    _module_origin_report()


def _dependency_report() -> dict[str, Any]:
    _assert_runtime()
    return {
        "python_version": _sys.version.split()[0],
        "required_flags": ["-I", "-S", "-B"],
        "sys_path_roles": ["python312.zip", "DLLs", "Lib", "BASE"],
        "meta_path": ["BuiltinImporter", "FrozenImporter", "AuthenticatedFinder"],
        "isolated": _sys.flags.isolated,
        "no_site": _sys.flags.no_site,
        "dont_write_bytecode": _sys.flags.dont_write_bytecode,
        "ignore_environment": _sys.flags.ignore_environment,
        "safe_path": _sys.flags.safe_path,
        "stdlib_inventory": {
            "directories": _STDLIB_INVENTORY[0],
            "files": _STDLIB_INVENTORY[1],
            "file_bytes": _STDLIB_INVENTORY[2],
            "sha256": _STDLIB_INVENTORY[3],
        },
        "interpreter_pin": {
            "path": ".venv-p1/Scripts/python.exe",
            "bytes": 274424,
            "sha256": "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14",
        },
        "pyvenv_cfg_pin": {
            "path": ".venv-p1/pyvenv.cfg",
            "bytes": 339,
            "sha256": "d1fb970854073922d49959ae01539088550613e316cb67f9fac858f586361174",
        },
        "module_origins": _module_origin_report(),
        "source_loader": "AUTHENTICATED_BUFFER_COMPILE_EXEC_NO_PYC",
        "native_loader": "PINNED_ORIGIN_HELD_FILE_SHARE_READ_ONLY",
        "firewall": dict(_FIREWALL),
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hermetic read-only P2 Layer-4 r3 compatibility verifier v3."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("check-only",), default="check-only")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    workspace = Path(_WORKSPACE_TEXT).resolve(strict=True)
    if arguments.root.resolve(strict=True) != workspace:
        raise BootstrapV3Error("parsed workspace changed")
    expected_python = _workspace_path(".venv-p1/Scripts/python.exe", kind="file")
    if _case(_sys.executable) != _case(str(expected_python)):
        raise BootstrapV3Error("canonical workspace interpreter is required")
    python_raw = _lock_path(str(expected_python), directory=False)["raw"]
    if (
        not isinstance(python_raw, bytes)
        or len(python_raw) != 274424
        or _fast_sha256(python_raw)
        != "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14"
    ):
        raise BootstrapV3Error("canonical interpreter pin changed")
    pyvenv = _workspace_path(".venv-p1/pyvenv.cfg", kind="file")
    pyvenv_raw = _lock_path(str(pyvenv), directory=False)["raw"]
    if (
        not isinstance(pyvenv_raw, bytes)
        or len(pyvenv_raw) != 339
        or _fast_sha256(pyvenv_raw)
        != "d1fb970854073922d49959ae01539088550613e316cb67f9fac858f586361174"
    ):
        raise BootstrapV3Error("pyvenv.cfg pin changed")
    observed: dict[str, dict[str, Any]] = {}
    buffers: dict[str, bytes] = {}
    for role in _EXPECTED_ROLES:
        pin = _TRUSTED_PINS[role]
        raw = _STABLE_REGISTRY.lock_pin(pin, label=f"v3 {role}")
        observed[role] = dict(pin)
        buffers[role] = raw
    config = _parse_json_buffer(buffers["CONFIG"], "v3 config")
    if config.get("implementation_roles") != {
        role: _TRUSTED_PINS[role]["path"] for role in _EXPECTED_ROLES
    }:
        raise BootstrapV3Error("config implementation role paths changed")
    requested = arguments.config
    if requested is not None:
        requested_path = requested if requested.is_absolute() else workspace / requested
        canonical = _workspace_path(_TRUSTED_PINS["CONFIG"]["path"], kind="file")
        if requested_path.resolve(strict=True) != canonical:
            raise BootstrapV3Error("alternate v3 config is forbidden")
    bootstrap_path = Path(__file__).resolve(strict=True)
    bootstrap_relative = bootstrap_path.relative_to(workspace).as_posix()
    bootstrap_raw = _lock_path(str(bootstrap_path), directory=False)["raw"]
    if not isinstance(bootstrap_raw, bytes):
        raise BootstrapV3Error("bootstrap source buffer changed")
    bootstrap_pin = {
        "path": bootstrap_relative,
        "bytes": len(bootstrap_raw),
        "sha256": _fast_sha256(bootstrap_raw),
    }
    context = {
        "workspace": workspace,
        "arguments": {
            "root": str(arguments.root),
            "config": str(arguments.config) if arguments.config is not None else None,
            "mode": arguments.mode,
        },
        "config": config,
        "config_raw": buffers["CONFIG"],
        "trusted_pins": _TRUSTED_PINS,
        "observed_implementation_pins": observed,
        "bootstrap_observed_pin": bootstrap_pin,
        "stable_registry": _STABLE_REGISTRY,
        "parse_json_buffer": _parse_json_buffer,
        "contained_path": lambda relative, must_exist, kind: _workspace_path(
            relative, must_exist=must_exist, kind=kind
        ),
        "relative_path": _relative_from_path,
        "path_exists": lambda relative: os.path.lexists(
            _workspace_path(relative, must_exist=False)
        ),
        "assert_runtime": _assert_runtime,
        "dependency_report": _dependency_report,
        "firewall": _FIREWALL,
    }
    _assert_runtime()
    helper = _exec_buffer(
        name=_HELPER_MODULE,
        path=_workspace_path(_TRUSTED_PINS["HELPER"]["path"], kind="file"),
        raw=buffers["HELPER"],
        injected={"_P2_V3_BOOTSTRAP_CONTEXT": context},
    )
    _assert_runtime()
    cli = _exec_buffer(
        name=_CLI_MODULE,
        path=_workspace_path(_TRUSTED_PINS["CLI"]["path"], kind="file"),
        raw=buffers["CLI"],
        injected={
            "_P2_V3_BOOTSTRAP_CONTEXT": context,
            "_P2_V3_AUTHENTICATED_HELPER": helper,
        },
    )
    report = cli.run_authenticated()
    if {
        role: {
            "path": pin["path"],
            "bytes": len(_STABLE_REGISTRY.lock_pin(pin, label=f"v3 final {role}")),
            "sha256": _fast_sha256(_STABLE_REGISTRY.lock_pin(pin, label=f"v3 final hash {role}")),
        }
        for role, pin in _TRUSTED_PINS.items()
    } != _TRUSTED_PINS:
        raise BootstrapV3Error("v3 implementation changed during verification")
    current_bootstrap = _lock_path(str(bootstrap_path), directory=False)["raw"]
    if (
        not isinstance(current_bootstrap, bytes)
        or len(current_bootstrap) != bootstrap_pin["bytes"]
        or _fast_sha256(current_bootstrap) != bootstrap_pin["sha256"]
    ):
        raise BootstrapV3Error("bootstrap changed during verification")
    _assert_runtime()
    report["authenticated_bootstrap"] = {
        "identity": _IDENTITY,
        "observed_pin": bootstrap_pin,
        "all_v3_roles_authenticated_before_execution": True,
        "authenticated_buffer_compile_exec_only": True,
        "source_file_loader_or_pyc_used_for_v3": False,
        "canonical_python_flags": ["-I", "-S", "-B"],
        "write_delete_process_attempts": _FIREWALL["write_delete_process_attempts"],
        "forbidden_import_attempts": _FIREWALL["forbidden_import_attempts"],
        "unexpected_import_attempts": _FIREWALL["unexpected_import_attempts"],
        "bytecode_read_attempts": _FIREWALL["bytecode_read_attempts"],
    }
    report.pop("summary_sha256", None)
    report["summary_sha256"] = hashlib.sha256(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
