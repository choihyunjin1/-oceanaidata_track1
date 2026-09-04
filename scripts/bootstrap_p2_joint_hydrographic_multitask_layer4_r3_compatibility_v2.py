"""Non-cyclic trust bootstrap for the P2 Layer-4 r3 verifier v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import types
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

sys.dont_write_bytecode = True


IDENTITY = "P2_JOINT_HYDROGRAPHIC_MULTITASK_LAYER4_R3_COMPATIBILITY_BOOTSTRAP_V2"
ANCHOR_RELATIVE = (
    "configs/experiments/"
    "p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v2_trust_anchor.json"
)
ANCHOR_BYTES = 1245
ANCHOR_SHA256 = "f99393120c31c4f4ffaf2b804c5a22d0b2a469cb8149d00036aaa72dd12fd75e"
HELPER_MODULE = "p2_restore.joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v2"
CLI_MODULE = "p2_restore.joint_hydrographic_multitask_layer4_r3_compatibility_cli_v2"
TRUSTED_ROLES = ("CONFIG", "HELPER", "CLI", "TESTS")
FORBIDDEN_ROOTS = ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "torch")
FORBIDDEN_ENGINE = "p2_restore.joint_hydrographic_multitask_layer4_execution_r3"
LOWER_SHA = re.compile(r"[0-9a-f]{64}\Z")
REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
WRITE_FLAGS = (
    getattr(os, "O_WRONLY", 1)
    | getattr(os, "O_RDWR", 2)
    | getattr(os, "O_APPEND", 8)
    | getattr(os, "O_CREAT", 256)
    | getattr(os, "O_TRUNC", 512)
)
MUTATION_EVENTS = {
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
    "os.system",
}
_FIREWALL_STATE = {
    "installed": False,
    "mutation_attempts": 0,
    "forbidden_import_attempts": 0,
}


class BootstrapV2Error(RuntimeError):
    """The v2 trust bootstrap failed closed."""


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _has_reparse(path: Path) -> bool:
    information = path.lstat()
    attributes = int(getattr(information, "st_file_attributes", 0))
    return path.is_symlink() or bool(attributes & REPARSE_ATTRIBUTE)


def _reject_full_ancestor_reparse(path: Path, *, require_target: bool) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    chain = list(reversed(lexical.parents)) + [lexical]
    missing_seen = False
    for entry in chain:
        exists = _lexists(entry)
        if exists and _has_reparse(entry):
            raise BootstrapV2Error(f"link/reparse ancestor is forbidden: {entry}")
        if not exists:
            missing_seen = True
        elif missing_seen:
            raise BootstrapV2Error("filesystem ancestor identity is inconsistent")
    if require_target and not _lexists(lexical):
        raise FileNotFoundError(lexical)
    return lexical


def _plain_workspace(root: Path) -> Path:
    lexical = _reject_full_ancestor_reparse(root, require_target=True)
    if not lexical.is_dir():
        raise BootstrapV2Error("workspace is not a plain directory")
    resolved = lexical.resolve(strict=True)
    if resolved != lexical:
        raise BootstrapV2Error("workspace lexical and resolved identities differ")
    return resolved


def _relative_parts(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise BootstrapV2Error("path is not canonical POSIX relative text")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise BootstrapV2Error("path is not a contained canonical relative path")
    return pure.parts


def contained_path(
    workspace: Path,
    relative: str,
    *,
    must_exist: bool = True,
    kind: str | None = None,
) -> Path:
    root = _plain_workspace(workspace)
    candidate = root.joinpath(*_relative_parts(relative))
    lexical = _reject_full_ancestor_reparse(candidate, require_target=must_exist)
    resolved = lexical.resolve(strict=must_exist)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise BootstrapV2Error(f"path escapes workspace: {relative}") from exc
    if must_exist:
        if kind == "file" and not resolved.is_file():
            raise BootstrapV2Error(f"regular file required: {relative}")
        if kind == "directory" and not resolved.is_dir():
            raise BootstrapV2Error(f"directory required: {relative}")
        if kind is None and not (resolved.is_file() or resolved.is_dir()):
            raise BootstrapV2Error(f"special filesystem entry rejected: {relative}")
    return resolved


def relative_plain_path(workspace: Path, path: Path) -> str:
    root = _plain_workspace(workspace)
    lexical = _reject_full_ancestor_reparse(path, require_target=True)
    resolved = lexical.resolve(strict=True)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise BootstrapV2Error("path is outside the workspace") from exc


def _pin_schema(pin: Mapping[str, Any]) -> tuple[str, int, str]:
    if set(pin) != {"path", "bytes", "sha256"}:
        raise BootstrapV2Error("trusted pin field set changed")
    relative = pin.get("path")
    size = pin.get("bytes")
    digest = pin.get("sha256")
    if (
        not isinstance(relative, str)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(digest, str)
        or LOWER_SHA.fullmatch(digest) is None
    ):
        raise BootstrapV2Error("trusted pin value changed")
    _relative_parts(relative)
    return relative, size, digest


def _identity(path: Path) -> tuple[int, int, int, int, int, int]:
    value = path.lstat()
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(getattr(value, "st_file_attributes", 0)),
    )


def _current_pin(workspace: Path, relative: str) -> dict[str, Any]:
    path = contained_path(workspace, relative, kind="file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def authenticated_bytes(workspace: Path, pin: Mapping[str, Any], *, label: str) -> bytes:
    relative, size, digest = _pin_schema(pin)
    path = contained_path(workspace, relative, kind="file")
    identity_before = _identity(path)
    with path.open("rb") as stream:
        descriptor_before = os.fstat(stream.fileno())
        raw = stream.read()
        descriptor_after = os.fstat(stream.fileno())
    identity_after = _identity(path)
    if (
        identity_before != identity_after
        or int(descriptor_before.st_dev) != int(descriptor_after.st_dev)
        or int(descriptor_before.st_ino) != int(descriptor_after.st_ino)
        or int(descriptor_before.st_size) != int(descriptor_after.st_size)
        or len(raw) != size
        or hashlib.sha256(raw).hexdigest() != digest
    ):
        raise BootstrapV2Error(f"authenticated buffer identity changed: {label}")
    if _current_pin(workspace, relative) != dict(pin):
        raise BootstrapV2Error(f"after-read file identity changed: {label}")
    return raw


def parse_json_buffer(raw: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BootstrapV2Error(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def finite_only(value: str) -> Any:
        raise BootstrapV2Error(f"non-finite JSON number in {label}: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=finite_only,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapV2Error(f"invalid JSON in {label}") from exc
    if not isinstance(value, dict):
        raise BootstrapV2Error(f"JSON root must be an object: {label}")
    return value


def authenticated_json(workspace: Path, pin: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    raw = authenticated_bytes(workspace, pin, label=label)
    return parse_json_buffer(raw, label=label)


def _loaded_forbidden() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if name == FORBIDDEN_ENGINE
        or any(name == root or name.startswith(root + ".") for root in FORBIDDEN_ROOTS)
    )


def _audit_hook(event: str, arguments: tuple[Any, ...]) -> None:
    if event == "open":
        mode = arguments[1] if len(arguments) > 1 else None
        flags = arguments[2] if len(arguments) > 2 else 0
        textual_write = isinstance(mode, str) and any(character in mode for character in "wax+")
        flag_write = isinstance(flags, int) and bool(flags & WRITE_FLAGS)
        if textual_write or flag_write:
            _FIREWALL_STATE["mutation_attempts"] += 1
            raise PermissionError("P2 v2 write audit firewall rejected open")
    if event in MUTATION_EVENTS:
        _FIREWALL_STATE["mutation_attempts"] += 1
        raise PermissionError(f"P2 v2 write audit firewall rejected {event}")
    if event == "import" and arguments:
        name = str(arguments[0])
        if name == FORBIDDEN_ENGINE or any(
            name == root or name.startswith(root + ".") for root in FORBIDDEN_ROOTS
        ):
            _FIREWALL_STATE["forbidden_import_attempts"] += 1
            raise ImportError(f"P2 v2 forbidden import rejected: {name}")


def install_firewall() -> None:
    if _FIREWALL_STATE["installed"]:
        raise BootstrapV2Error("write/import firewall is already installed")
    if sys.dont_write_bytecode is not True:
        raise BootstrapV2Error("sys.dont_write_bytecode is not enabled")
    if _loaded_forbidden():
        raise BootstrapV2Error("forbidden module was loaded before the firewall")
    sys.addaudithook(_audit_hook)
    _FIREWALL_STATE["installed"] = True


def assert_firewall() -> None:
    if (
        sys.dont_write_bytecode is not True
        or _FIREWALL_STATE["installed"] is not True
        or _FIREWALL_STATE["mutation_attempts"] != 0
        or _FIREWALL_STATE["forbidden_import_attempts"] != 0
        or _loaded_forbidden()
    ):
        raise BootstrapV2Error("default check-only zero-write/import firewall changed")


def _exec_authenticated_module(
    *,
    name: str,
    path: Path,
    raw: bytes,
    injected: Mapping[str, Any],
) -> Any:
    if name in sys.modules:
        raise BootstrapV2Error(f"authenticated module already exists: {name}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__dict__.update(injected)
    sys.modules[name] = module
    try:
        code = compile(raw, str(path), "exec", dont_inherit=True, optimize=0)
        exec(code, module.__dict__)  # noqa: S102
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _validate_anchor(anchor: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if set(anchor) != {
        "schema_version",
        "identity",
        "created_at_kst",
        "noncyclic_trust_root",
        "bootstrap_external_pin_required",
        "trusted_files",
    }:
        raise BootstrapV2Error("trust anchor field set changed")
    if (
        anchor.get("schema_version")
        != "p2_joint_hydrographic_multitask_layer4.r3_compatibility_trust_anchor.v2"
        or anchor.get("identity")
        != "P2_JOINT_HYDROGRAPHIC_MULTITASK_LAYER4_R3_COMPATIBILITY_TRUST_ANCHOR_V2"
        or anchor.get("noncyclic_trust_root") is not True
        or anchor.get("bootstrap_external_pin_required") is not True
    ):
        raise BootstrapV2Error("trust anchor identity changed")
    trusted = anchor.get("trusted_files")
    if not isinstance(trusted, Mapping) or set(trusted) != set(TRUSTED_ROLES):
        raise BootstrapV2Error("trust anchor role set changed")
    return {role: dict(trusted[role]) for role in TRUSTED_ROLES}


def verify_trusted_files(
    workspace: Path,
    anchor: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    trusted = _validate_anchor(anchor)
    observed: dict[str, dict[str, Any]] = {}
    buffers: dict[str, bytes] = {}
    for role in TRUSTED_ROLES:
        pin = trusted[role]
        raw = authenticated_bytes(workspace, pin, label=f"v2 {role}")
        observed[role] = dict(pin)
        buffers[role] = raw
    return observed, buffers


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticated read-only P2 Layer-4 r3 compatibility verifier v2."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("check-only",), default="check-only")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    workspace = _plain_workspace(arguments.root)
    install_firewall()
    anchor_pin = {
        "path": ANCHOR_RELATIVE,
        "bytes": ANCHOR_BYTES,
        "sha256": ANCHOR_SHA256,
    }
    anchor_raw = authenticated_bytes(workspace, anchor_pin, label="v2 trust anchor")
    anchor = parse_json_buffer(anchor_raw, label="v2 trust anchor")
    observed, buffers = verify_trusted_files(workspace, anchor)
    config = parse_json_buffer(buffers["CONFIG"], label="v2 config")
    bootstrap_relative = relative_plain_path(workspace, Path(__file__))
    bootstrap_observed_pin = _current_pin(workspace, bootstrap_relative)

    def context_bytes(pin: Mapping[str, Any], label: str) -> bytes:
        return authenticated_bytes(workspace, pin, label=label)

    def context_json(pin: Mapping[str, Any], label: str) -> dict[str, Any]:
        return authenticated_json(workspace, pin, label=label)

    def context_contained(relative: str, must_exist: bool, kind: str | None) -> Path:
        return contained_path(
            workspace,
            relative,
            must_exist=must_exist,
            kind=kind,
        )

    def context_relative(path: Path) -> str:
        return relative_plain_path(workspace, path)

    def reverify_trusted() -> dict[str, dict[str, Any]]:
        current, _buffers = verify_trusted_files(workspace, anchor)
        return current

    def reverify_anchor() -> dict[str, Any]:
        authenticated_bytes(workspace, anchor_pin, label="v2 trust anchor recheck")
        return dict(anchor_pin)

    context = {
        "workspace": workspace,
        "arguments": {
            "root": str(arguments.root),
            "config": str(arguments.config) if arguments.config is not None else None,
            "mode": arguments.mode,
        },
        "config": config,
        "config_raw": buffers["CONFIG"],
        "anchor": anchor,
        "anchor_pin": anchor_pin,
        "bootstrap_observed_pin": bootstrap_observed_pin,
        "observed_implementation_pins": observed,
        "authenticated_bytes": context_bytes,
        "authenticated_json": context_json,
        "parse_json_buffer": lambda raw, label: parse_json_buffer(raw, label=label),
        "contained_path": context_contained,
        "relative_plain_path": context_relative,
        "reverify_trusted_files": reverify_trusted,
        "reverify_anchor": reverify_anchor,
        "assert_firewall": assert_firewall,
        "firewall_state": _FIREWALL_STATE,
    }
    assert_firewall()
    helper_path = contained_path(workspace, observed["HELPER"]["path"], kind="file")
    helper = _exec_authenticated_module(
        name=HELPER_MODULE,
        path=helper_path,
        raw=buffers["HELPER"],
        injected={"_P2_V2_BOOTSTRAP_CONTEXT": context},
    )
    assert_firewall()
    cli_path = contained_path(workspace, observed["CLI"]["path"], kind="file")
    cli = _exec_authenticated_module(
        name=CLI_MODULE,
        path=cli_path,
        raw=buffers["CLI"],
        injected={
            "_P2_V2_BOOTSTRAP_CONTEXT": context,
            "_P2_V2_AUTHENTICATED_HELPER": helper,
        },
    )
    report = cli.run_authenticated()
    if reverify_trusted() != anchor["trusted_files"] or reverify_anchor() != anchor_pin:
        raise BootstrapV2Error("trusted implementation changed after execution")
    if _current_pin(workspace, bootstrap_relative) != bootstrap_observed_pin:
        raise BootstrapV2Error("bootstrap changed during execution")
    assert_firewall()
    report["authenticated_bootstrap"] = {
        "identity": IDENTITY,
        "trust_anchor": anchor_pin,
        "bootstrap_observed_pin": bootstrap_observed_pin,
        "all_trusted_roles_authenticated_before_execution": True,
        "authenticated_buffer_compile_exec_only": True,
        "source_file_loader_or_pyc_used": False,
        "single_buffer_json_parse": True,
        "after_read_identity_checks": True,
        "full_ancestor_reparse_checks": True,
        "sys_dont_write_bytecode": sys.dont_write_bytecode,
        "write_audit_attempts": _FIREWALL_STATE["mutation_attempts"],
        "forbidden_import_attempts": _FIREWALL_STATE["forbidden_import_attempts"],
        "forbidden_modules": _loaded_forbidden(),
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
