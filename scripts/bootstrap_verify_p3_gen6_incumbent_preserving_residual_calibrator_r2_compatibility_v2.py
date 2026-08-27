"""Noncyclic pre-import trust root for the Gen6r2 compatibility-v2 verifier.

Independent QA must pin this bootstrap itself.  It embeds immutable pins for
every subordinate v2 implementation file and every executable legacy source,
authenticates all of them before executing any module, and executes only the
authenticated in-memory buffers with compile/exec.  Python bytecode loaders are
never used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import types
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

sys.dont_write_bytecode = True

IDENTITY: Final = (
    "P3_GEN6_INCUMBENT_PRESERVING_RESIDUAL_CALIBRATOR_R2_COMPATIBILITY_V2_NONCYCLIC_TRUST_ROOT"
)
BOOTSTRAP_RELATIVE: Final = (
    "scripts/bootstrap_verify_p3_gen6_incumbent_preserving_residual_calibrator_"
    "r2_compatibility_v2.py"
)
R2_MODULE: Final = "p3_wave.gen6_incumbent_preserving_residual_calibrator_contract_r2"
V1_MODULE: Final = (
    "p3_wave.gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v1"
)
V2_MODULE: Final = (
    "p3_wave.gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v2"
)
CLI_MODULE: Final = "p3_wave.gen6_incumbent_preserving_residual_calibrator_r2_compatibility_cli_v2"
ENGINE_MODULE: Final = "p3_wave.gen6_incumbent_preserving_residual_calibrator_execution_r2"
NUMERICAL_ROOTS: Final = ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "torch")
ALLOWED_POST_AUTH_NUMERICAL_ROOTS: Final = ("numpy", "pandas", "pyarrow")
REPARSE_ATTRIBUTE: Final = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
O_BINARY: Final = getattr(os, "O_BINARY", 0)
THREAD_ENVIRONMENT: Final = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}

IMPLEMENTATION_ROLES: Final = {
    "TRUSTED_BOOTSTRAP": BOOTSTRAP_RELATIVE,
    "CONFIG": (
        "configs/experiments/"
        "p3_gen6_incumbent_preserving_residual_calibrator_"
        "v1r2_compatibility_verifier_v2.json"
    ),
    "HELPER": (
        "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v2.py"
    ),
    "CLI": (
        "scripts/verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v2.py"
    ),
    "TESTS": (
        "tests/test_p3_gen6_incumbent_preserving_residual_calibrator_"
        "r2_compatibility_verifier_v2.py"
    ),
}

# This is the machine-enforced trust map.  CONFIG is authenticated here rather
# than self-pinned, so the trust graph is acyclic.  Independent QA pins this
# bootstrap after the subordinate files have been frozen.
PINNED_SOURCES: Final = {
    "CONFIG": {
        "path": IMPLEMENTATION_ROLES["CONFIG"],
        "bytes": 11074,
        "sha256": "a80aedd91cc1ed73d638fcaa2827f73344220d49b3f2c1073458e7040c044cc1",
    },
    "HELPER": {
        "path": IMPLEMENTATION_ROLES["HELPER"],
        "bytes": 15462,
        "sha256": "054c271e2ba0d8aac8fc8f4436884b491ab7c03a397fd10aae0e0478ddcd681b",
    },
    "CLI": {
        "path": IMPLEMENTATION_ROLES["CLI"],
        "bytes": 1417,
        "sha256": "39faf8b6f6d6a1acb043ed038cce396a85a4265be04678e8a9cdbc134980df13",
    },
    "TESTS": {
        "path": IMPLEMENTATION_ROLES["TESTS"],
        "bytes": 12901,
        "sha256": "a64f7e865422d2ace395e17a3717cca1ebec9fa35a57c06c477f0983f23abb67",
    },
    "R2_CONTRACT": {
        "path": "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_contract_r2.py",
        "bytes": 73236,
        "sha256": "24245a5ed9e47c335607560bb02185259984ffae70b18edb7e8afd57a4aafe51",
    },
    "V1_CONFIG": {
        "path": (
            "configs/experiments/p3_gen6_incumbent_preserving_residual_calibrator_"
            "v1r2_compatibility_verifier_v1.json"
        ),
        "bytes": 12799,
        "sha256": "6b92a6eb67adfb042958cb518633ead4e2c70ffb1e7de35eceafccd6c6e42d2a",
    },
    "V1_HELPER": {
        "path": (
            "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_"
            "r2_compatibility_verifier_v1.py"
        ),
        "bytes": 47746,
        "sha256": "9749feb754815915a50bab2dcf6a6ed687159047874fcd9ad0cda376e9ea0375",
    },
    "V1_CLI": {
        "path": (
            "scripts/verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v1.py"
        ),
        "bytes": 2546,
        "sha256": "25a24339319b43da3d9f97b422ac07dd574bb27b10532d7cf983a21726b86912",
    },
    "V1_TESTS": {
        "path": (
            "tests/test_p3_gen6_incumbent_preserving_residual_calibrator_"
            "r2_compatibility_verifier_v1.py"
        ),
        "bytes": 12727,
        "sha256": "20b343905ad4670bfa9b970880577216560edb8bdcf048f5f30e97b1f7bbc255",
    },
    "V1_OWNER_NO_GO": {
        "path": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_"
            "v1r2_compatibility_verifier_v1_no_go/OWNER_STATIC_QA_NO_GO_20260823.json"
        ),
        "bytes": 5814,
        "sha256": "b721a6de42429754fb1b98062f54832848af92dcc58fbfb204bc5299c344b620",
    },
    "V1_TOMBSTONE": {
        "path": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_"
            "v1r2_compatibility_verifier_v1_no_go/EXECUTION_TOMBSTONE.json"
        ),
        "bytes": 3006,
        "sha256": "380830f54ff7d0c2d78e4ad11592bf8180d7fd3d2a59118d93e5944980661783",
    },
}


class BootstrapTrustError(RuntimeError):
    """A pre-import trust, containment, or immutable-state check failed."""


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _has_link_or_reparse(path: Path) -> bool:
    info = path.lstat()
    attributes = int(getattr(info, "st_file_attributes", 0))
    return path.is_symlink() or bool(attributes & REPARSE_ATTRIBUTE)


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.fspath(left)) == os.path.normcase(os.fspath(right))


def _plain_root(root: Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(root)))
    chain = list(reversed((lexical, *lexical.parents)))
    for ancestor in chain:
        if not _lexists(ancestor):
            raise BootstrapTrustError("canonical workspace ancestor is missing")
        if _has_link_or_reparse(ancestor):
            raise BootstrapTrustError("workspace ancestor link/reparse point is forbidden")
        if not ancestor.is_dir():
            raise BootstrapTrustError("canonical workspace ancestor is not a directory")
    resolved = lexical.resolve(strict=True)
    if not _same_path(resolved, lexical):
        raise BootstrapTrustError("canonical workspace resolves through an alias")
    return resolved


def _relative_parts(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise BootstrapTrustError("relative path is not canonical POSIX text")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise BootstrapTrustError("relative path is not strictly contained")
    return pure.parts


def _checked_path(
    root: Path,
    relative: str,
    *,
    must_exist: bool = True,
    kind: str | None = None,
) -> Path:
    workspace = _plain_root(root)
    candidate = workspace.joinpath(*_relative_parts(relative))
    probe = workspace
    for part in _relative_parts(relative):
        probe /= part
        if _lexists(probe) and _has_link_or_reparse(probe):
            raise BootstrapTrustError(f"link/reparse component is forbidden: {relative}")
    if must_exist and not _lexists(candidate):
        raise BootstrapTrustError(f"required pinned path is missing: {relative}")
    resolved = candidate.resolve(strict=must_exist)
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise BootstrapTrustError(f"path escapes canonical workspace: {relative}") from exc
    if must_exist:
        if kind == "file" and not resolved.is_file():
            raise BootstrapTrustError(f"regular file required: {relative}")
        if kind == "directory" and not resolved.is_dir():
            raise BootstrapTrustError(f"plain directory required: {relative}")
        if kind is None and not (resolved.is_file() or resolved.is_dir()):
            raise BootstrapTrustError(f"special filesystem object rejected: {relative}")
    return resolved


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_mode),
    )


def _read_stable_file(path: Path) -> tuple[bytes, tuple[int, int, int, int, int]]:
    if _has_link_or_reparse(path):
        raise BootstrapTrustError("authenticated source became link/reparse")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise BootstrapTrustError("authenticated source is not a regular file")
    descriptor = os.open(os.fspath(path), os.O_RDONLY | O_BINARY)
    try:
        opened = os.fstat(descriptor)
        if _stat_identity(opened) != _stat_identity(before):
            raise BootstrapTrustError("source identity changed while opening")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if _stat_identity(after) != _stat_identity(before) or _has_link_or_reparse(path):
        raise BootstrapTrustError("source identity changed while reading")
    return b"".join(chunks), _stat_identity(after)


def _validate_pin(role: str, pin: Any) -> dict[str, Any]:
    if not isinstance(pin, dict) or set(pin) != {"path", "bytes", "sha256"}:
        raise BootstrapTrustError(f"trusted pin schema changed: {role}")
    if (
        not isinstance(pin["bytes"], int)
        or pin["bytes"] <= 0
        or not isinstance(pin["sha256"], str)
        or len(pin["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in pin["sha256"])
    ):
        raise BootstrapTrustError(f"trusted pin value is malformed: {role}")
    _relative_parts(pin["path"])
    return dict(pin)


def _authenticate_all(
    root: Path,
    pins: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    dict[str, bytes],
    dict[str, tuple[int, int, int, int, int]],
]:
    workspace = _plain_root(root)
    expected = PINNED_SOURCES if pins is None else pins
    if not expected:
        raise BootstrapTrustError("trusted source pin map is empty")
    buffers: dict[str, bytes] = {}
    identities: dict[str, tuple[int, int, int, int, int]] = {}
    for role, untrusted_pin in expected.items():
        pin = _validate_pin(role, untrusted_pin)
        path = _checked_path(workspace, pin["path"], must_exist=True, kind="file")
        raw, identity = _read_stable_file(path)
        if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise BootstrapTrustError(f"authenticated source pin drift: {role}")
        buffers[role] = raw
        identities[role] = identity
    if set(buffers) != set(expected):
        raise BootstrapTrustError("not every trusted source was authenticated")
    return buffers, identities


def _reverify_all(
    root: Path,
    pins: dict[str, dict[str, Any]],
    buffers: dict[str, bytes],
    identities: dict[str, tuple[int, int, int, int, int]],
) -> None:
    current_buffers, current_identities = _authenticate_all(root, pins)
    if current_buffers != buffers or current_identities != identities:
        raise BootstrapTrustError("authenticated source file mutated after initial read")


def _loaded_numerical_roots() -> list[str]:
    return sorted(
        root
        for root in NUMERICAL_ROOTS
        if any(name == root or name.startswith(root + ".") for name in sys.modules)
    )


def _preimport_fail_closed() -> None:
    numerical = _loaded_numerical_roots()
    if numerical:
        raise BootstrapTrustError(
            f"numerical modules were imported before authentication: {numerical}"
        )
    forbidden = [
        name
        for name in (R2_MODULE, V1_MODULE, V2_MODULE, CLI_MODULE, ENGINE_MODULE)
        if name in sys.modules
    ]
    if forbidden:
        raise BootstrapTrustError(
            f"protected modules were imported before authentication: {forbidden}"
        )


def _verify_canonical_environment(workspace: Path) -> Path:
    workspace_text = os.environ.get("P3_WORKSPACE_ROOT")
    data_text = os.environ.get("P3_DATA_DIR")
    if not workspace_text or not data_text:
        raise BootstrapTrustError("P3_WORKSPACE_ROOT and P3_DATA_DIR are required")
    workspace_env = _plain_root(Path(workspace_text))
    if not _same_path(workspace_env, workspace):
        raise BootstrapTrustError("P3_WORKSPACE_ROOT differs from --root")
    data_dir = _plain_root(Path(data_text))
    for key, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(key) != expected:
            raise BootstrapTrustError(f"canonical thread environment differs: {key}")
    return data_dir


class _EngineImportGuard:
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        del path, target
        if fullname == ENGINE_MODULE or fullname.startswith(ENGINE_MODULE + "."):
            raise BootstrapTrustError("r2 execution engine import is forbidden")
        return None


def _exec_authenticated_buffer(
    *,
    module_name: str,
    source_path: Path,
    raw: bytes,
    injected: dict[str, Any] | None = None,
) -> types.ModuleType:
    module = types.ModuleType(module_name)
    module.__file__ = os.fspath(source_path)
    module.__package__ = module_name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    if injected:
        module.__dict__.update(injected)
    code = compile(raw, os.fspath(source_path), "exec", dont_inherit=True)
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trusted read-only verification of the frozen P3 Gen6r2 result."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("check-only",), default="check-only")
    return parser.parse_args(argv)


def _run_trusted(args: argparse.Namespace) -> dict[str, Any]:
    _preimport_fail_closed()
    workspace = _plain_root(args.root)
    _verify_canonical_environment(workspace)
    canonical_config = _checked_path(
        workspace, IMPLEMENTATION_ROLES["CONFIG"], must_exist=True, kind="file"
    )
    if args.config is not None:
        requested = args.config if args.config.is_absolute() else workspace / args.config
        if (
            not _same_path(requested.absolute(), canonical_config.absolute())
            or requested.resolve(strict=True) != canonical_config
        ):
            raise BootstrapTrustError("alternate compatibility-v2 config is forbidden")
    bootstrap_path = _checked_path(workspace, BOOTSTRAP_RELATIVE, must_exist=True, kind="file")
    bootstrap_raw, bootstrap_identity = _read_stable_file(bootstrap_path)
    buffers, identities = _authenticate_all(workspace)
    initial_roles = set(buffers)
    required_roles = {
        "CONFIG",
        "HELPER",
        "CLI",
        "TESTS",
        "R2_CONTRACT",
        "V1_CONFIG",
        "V1_HELPER",
        "V1_CLI",
        "V1_TESTS",
        "V1_OWNER_NO_GO",
        "V1_TOMBSTONE",
    }
    if initial_roles != required_roles:
        raise BootstrapTrustError("canonical machine-enforced source roles changed")

    reverify_phases: list[str] = []
    claimed_phases: set[str] = set()
    token = object()

    def reverify(phase: str) -> None:
        if not isinstance(phase, str) or not phase:
            raise BootstrapTrustError("reverification phase is malformed")
        _reverify_all(workspace, PINNED_SOURCES, buffers, identities)
        reverify_phases.append(phase)

    def checked_path(
        relative: str,
        *,
        must_exist: bool = True,
        kind: str | None = None,
    ) -> Path:
        return _checked_path(
            workspace,
            relative,
            must_exist=must_exist,
            kind=kind,
        )

    def claim_phase(phase: str) -> None:
        if phase != "CLI_VERIFY_ONCE" or phase in claimed_phases:
            raise BootstrapTrustError("trusted phase capability is invalid or replayed")
        claimed_phases.add(phase)

    frozen_pins = {role: MappingProxyType(dict(pin)) for role, pin in PINNED_SOURCES.items()}
    context_data: dict[str, Any] = {
        "token": token,
        "root": workspace,
        "buffers": MappingProxyType(dict(buffers)),
        "pins": MappingProxyType(frozen_pins),
        "implementation_roles": MappingProxyType(dict(IMPLEMENTATION_ROLES)),
        "reverify": reverify,
        "checked_path": checked_path,
        "claim_phase": claim_phase,
    }
    context = MappingProxyType(context_data)
    injected_modules: list[str] = []
    guard = _EngineImportGuard()
    sys.meta_path.insert(0, guard)
    run_error: BaseException | None = None
    result: dict[str, Any] | None = None
    try:
        reverify("pre_r2_contract_exec")
        r2_module = _exec_authenticated_buffer(
            module_name=R2_MODULE,
            source_path=_checked_path(
                workspace, PINNED_SOURCES["R2_CONTRACT"]["path"], kind="file"
            ),
            raw=buffers["R2_CONTRACT"],
        )
        injected_modules.append(R2_MODULE)
        context_data["r2_module"] = r2_module

        reverify("pre_v1_helper_exec")
        v1_module = _exec_authenticated_buffer(
            module_name=V1_MODULE,
            source_path=_checked_path(workspace, PINNED_SOURCES["V1_HELPER"]["path"], kind="file"),
            raw=buffers["V1_HELPER"],
        )
        injected_modules.append(V1_MODULE)
        context_data["v1_module"] = v1_module

        reverify("pre_v2_helper_exec")
        v2_module = _exec_authenticated_buffer(
            module_name=V2_MODULE,
            source_path=_checked_path(workspace, PINNED_SOURCES["HELPER"]["path"], kind="file"),
            raw=buffers["HELPER"],
            injected={
                "__trusted_bootstrap_context__": context,
                "__trusted_bootstrap_token__": token,
                "__trusted_v1_module__": v1_module,
                "__trusted_r2_module__": r2_module,
            },
        )
        injected_modules.append(V2_MODULE)
        context_data["v2_helper"] = v2_module

        reverify("pre_v2_cli_exec")
        cli_module = _exec_authenticated_buffer(
            module_name=CLI_MODULE,
            source_path=_checked_path(workspace, PINNED_SOURCES["CLI"]["path"], kind="file"),
            raw=buffers["CLI"],
            injected={
                "__trusted_bootstrap_context__": context,
                "__trusted_bootstrap_token__": token,
                "__trusted_v2_helper__": v2_module,
            },
        )
        injected_modules.append(CLI_MODULE)
        reverify("pre_cli_entry")
        requested_config = args.config
        if requested_config is not None and not requested_config.is_absolute():
            requested_config = workspace / requested_config
        result = cli_module.run(
            root=workspace,
            requested_config=requested_config,
            mode=args.mode,
        )
        reverify("post_cli_entry")
    except BaseException as exc:
        run_error = exc
    finally:
        integrity_error: BaseException | None = None
        try:
            reverify("bootstrap_finally")
            final_bootstrap_raw, final_bootstrap_identity = _read_stable_file(bootstrap_path)
            if (
                final_bootstrap_raw != bootstrap_raw
                or final_bootstrap_identity != bootstrap_identity
            ):
                raise BootstrapTrustError("trusted bootstrap mutated during verification")
            if ENGINE_MODULE in sys.modules:
                raise BootstrapTrustError("r2 execution engine was imported")
            unexpected = sorted(
                set(_loaded_numerical_roots()) - set(ALLOWED_POST_AUTH_NUMERICAL_ROOTS)
            )
            if unexpected:
                raise BootstrapTrustError(
                    f"unexpected numerical modules imported after authentication: {unexpected}"
                )
        except BaseException as exc:
            integrity_error = exc
        finally:
            if guard in sys.meta_path:
                sys.meta_path.remove(guard)
            for module_name in reversed(injected_modules):
                sys.modules.pop(module_name, None)
        if integrity_error is not None:
            raise integrity_error from run_error
    if run_error is not None:
        raise run_error
    if result is None or claimed_phases != {"CLI_VERIFY_ONCE"}:
        raise BootstrapTrustError("trusted CLI phase was not consumed exactly once")
    result["trusted_bootstrap"]["source_roles_authenticated"] = sorted(initial_roles)
    result["trusted_bootstrap"]["reverification_phases"] = reverify_phases
    result["trusted_bootstrap"]["final_file_identity_verified"] = True
    result["trusted_bootstrap"]["r2_engine_imported"] = False
    result["trusted_bootstrap"]["numerical_roots_after_authentication"] = _loaded_numerical_roots()
    return result


def main(argv: list[str] | None = None) -> int:
    result = _run_trusted(_parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
