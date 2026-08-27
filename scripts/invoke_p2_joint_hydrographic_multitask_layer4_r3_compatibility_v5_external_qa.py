"""Reference external-QA holder for the P2 Layer-4 r3 v5 check-only verifier.

An independent reviewer must authenticate these source bytes before executing them.
This process opens the complete PowerShell runtime with share-read-only handles before
it starts the exact pinned host, and supplies stage zero only as UTF-16LE
``-EncodedCommand`` bytes. It never executes stage zero or stage one by path.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import gzip
import hashlib
import json
import os
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path
from typing import NamedTuple

HOST_BYTES = 301368
HOST_SHA256 = "db6dd81183fe57d22e03b911ec9a30a2fd7c40542e97743615355a6fb44f458f"
RUNTIME_DIRECTORIES = 53
RUNTIME_FILES = 983
RUNTIME_FILE_BYTES = 296034085
RUNTIME_SHA256 = "9a197570fffc3399d9c8477ef0199e31ad950701de7b133df7c4669d42099be1"
STAGE_ZERO_BYTES = 18460
STAGE_ZERO_SHA256 = "26d08395897a12307bac4785a9f8768a0a59451e1402c484c50fc1e996f86a8f"
STAGE_ZERO_RELATIVE = (
    "scripts/launch_p2_joint_hydrographic_multitask_layer4_r3_compatibility_"
    "v5_stage0.ps1"
)

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class _FileTime(ctypes.Structure):
    _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]


class _ByHandleFileInformation(ctypes.Structure):
    _fields_ = [
        ("attributes", wintypes.DWORD),
        ("creation", _FileTime),
        ("access", _FileTime),
        ("write", _FileTime),
        ("volume_serial", wintypes.DWORD),
        ("size_high", wintypes.DWORD),
        ("size_low", wintypes.DWORD),
        ("number_of_links", wintypes.DWORD),
        ("index_high", wintypes.DWORD),
        ("index_low", wintypes.DWORD),
    ]


class _Record(NamedTuple):
    kind: str
    relative: str
    path: Path


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_create_file = _kernel32.CreateFileW
_create_file.argtypes = (
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
)
_create_file.restype = wintypes.HANDLE
_get_information = _kernel32.GetFileInformationByHandle
_get_information.argtypes = (wintypes.HANDLE, ctypes.POINTER(_ByHandleFileInformation))
_get_information.restype = wintypes.BOOL
_close_handle = _kernel32.CloseHandle
_close_handle.argtypes = (wintypes.HANDLE,)
_close_handle.restype = wintypes.BOOL


class ExternalQaError(RuntimeError):
    """The reference external-QA trust holder failed closed."""


def _pin(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _open_share_deny(path: Path, *, directory: bool) -> int:
    if not path.is_absolute() or path.is_symlink():
        raise ExternalQaError(f"non-canonical runtime path: {path}")
    flags = FILE_FLAG_OPEN_REPARSE_POINT
    if directory:
        flags |= FILE_FLAG_BACKUP_SEMANTICS
    handle = _create_file(
        str(path),
        GENERIC_READ,
        FILE_SHARE_READ,
        None,
        OPEN_EXISTING,
        flags,
        None,
    )
    if handle == INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
    information = _ByHandleFileInformation()
    if not _get_information(handle, ctypes.byref(information)):
        error = ctypes.get_last_error()
        _close_handle(handle)
        raise ctypes.WinError(error)
    if information.attributes & FILE_ATTRIBUTE_REPARSE_POINT:
        _close_handle(handle)
        raise ExternalQaError(f"runtime reparse target: {path}")
    if not directory and information.number_of_links != 1:
        _close_handle(handle)
        raise ExternalQaError(f"runtime hardlink target: {path}")
    return int(handle)


def _runtime_records(root: Path) -> list[_Record]:
    records: list[_Record] = []
    pending = [root]
    while pending:
        current = pending.pop(0)
        for child in sorted(current.iterdir(), key=lambda item: item.name):
            relative = child.relative_to(root).as_posix()
            if child.is_symlink():
                raise ExternalQaError(f"runtime symlink/reparse entry: {relative}")
            if child.is_dir():
                records.append(_Record("d", relative, child))
                pending.append(child)
            elif child.is_file():
                records.append(_Record("f", relative, child))
            else:
                raise ExternalQaError(f"runtime special entry: {relative}")
    return sorted(records, key=lambda record: record.relative)


def _hold_runtime(root: Path) -> tuple[list[int], dict[str, object]]:
    records = _runtime_records(root)
    handles = [_open_share_deny(root, directory=True)]
    digest = hashlib.sha256()
    directories = files = total = 0
    try:
        for record in records:
            is_directory = record.kind == "d"
            handles.append(_open_share_deny(record.path, directory=is_directory))
            if is_directory:
                directories += 1
                digest.update(f"d\0{record.relative}\n".encode())
            else:
                size, sha256 = _pin(record.path)
                files += 1
                total += size
                digest.update(f"f\0{record.relative}\0{size}\0{sha256}\n".encode())
        observed = {
            "directories": directories,
            "files": files,
            "file_bytes": total,
            "algorithm": (
                "SHA256_SORTED_ORDINAL_TYPE_NUL_RELATIVE_NUL_BYTES_NUL_"
                "FILE_SHA256_LF"
            ),
            "sha256": digest.hexdigest(),
        }
        expected = {
            "directories": RUNTIME_DIRECTORIES,
            "files": RUNTIME_FILES,
            "file_bytes": RUNTIME_FILE_BYTES,
            "algorithm": observed["algorithm"],
            "sha256": RUNTIME_SHA256,
        }
        if observed != expected:
            raise ExternalQaError(
                "complete PowerShell runtime inventory changed: "
                + json.dumps(observed, sort_keys=True)
            )
        return handles, observed
    except BaseException:
        for handle in reversed(handles):
            _close_handle(handle)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="External holder for the P2 v5 check-only verifier")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.root.resolve(strict=True)
    host = arguments.host.resolve(strict=True)
    if _pin(host) != (HOST_BYTES, HOST_SHA256):
        raise ExternalQaError("exact PowerShell host pin changed")
    handles, inventory = _hold_runtime(host.parent)
    try:
        stage_zero = (root / STAGE_ZERO_RELATIVE).resolve(strict=True)
        raw = stage_zero.read_bytes()
        if (len(raw), hashlib.sha256(raw).hexdigest()) != (
            STAGE_ZERO_BYTES,
            STAGE_ZERO_SHA256,
        ):
            raise ExternalQaError("stage-zero source pin changed")
        text = raw.decode("utf-8")
        if text.encode("utf-8") != raw:
            raise ExternalQaError("stage-zero UTF-8 round trip changed")
        compressed = base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")
        inline = (
            "$z=[Convert]::FromBase64String('"
            + compressed
            + "');$m=[IO.MemoryStream]::new($z,$false);"
            "$g=[IO.Compression.GZipStream]::new($m,[IO.Compression.CompressionMode]::Decompress);"
            "$o=[IO.MemoryStream]::new();$g.CopyTo($o);$g.Dispose();$m.Dispose();"
            "$b=$o.ToArray();$o.Dispose();"
            "$t=[Text.UTF8Encoding]::new($false,$true).GetString($b);"
            "& ([ScriptBlock]::Create($t))"
        )
        encoded = base64.b64encode(inline.encode("utf-16-le")).decode("ascii")
        if len(encoded) >= 30000:
            raise ExternalQaError("path-immutable encoded stage zero exceeds Windows limits")
        environment = os.environ.copy()
        environment["P2_POWERSHELL_HOST"] = str(host)
        environment["P2_V5_WORKSPACE_ROOT"] = str(root)
        environment["P2_V5_EXTERNAL_RUNTIME_INVENTORY_SHA256"] = str(inventory["sha256"])
        environment["P2_V5_STAGE_ZERO_BYTES"] = str(len(raw))
        environment["P2_V5_STAGE_ZERO_SHA256"] = hashlib.sha256(raw).hexdigest()
        result = subprocess.run(
            [
                str(host),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded,
            ],
            cwd=root,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
        if result.stderr:
            sys.stderr.buffer.write(result.stderr)
        if result.stdout:
            sys.stdout.buffer.write(result.stdout)
        return int(result.returncode)
    finally:
        for handle in reversed(handles):
            _close_handle(handle)


if __name__ == "__main__":
    raise SystemExit(main())
