"""Stable-byte, verifier-only compatibility check for frozen P2 Layer-4 r3.

This source is never imported through the ambient import system.  The v4
bootstrap authenticates its exact bytes, compiles that buffer, and injects a
closed trust context before execution.  Startup trust is deliberately rooted
outside Python in the independently pinned v4 launcher.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import struct
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    _BOOTSTRAP = _P2_V4_BOOTSTRAP_CONTEXT  # type: ignore[name-defined]  # noqa: F821
except NameError as exc:  # pragma: no cover - subprocess contract test
    raise RuntimeError("P2 v4 helper requires the authenticated bootstrap") from exc


IDENTITY = "P2_JOINT_HYDROGRAPHIC_MULTITASK_LAYER4_R3_COMPATIBILITY_VERIFIER_V4"
R3_MODULE = "p2_restore.joint_hydrographic_multitask_layer4_contract_r3_v4_authenticated"
R3_ENGINE_MODULE = "p2_restore.joint_hydrographic_multitask_layer4_execution_r3"
TRUSTED_ROLES = ("CONFIG", "HELPER", "CLI", "TESTS")
FORBIDDEN_ROOTS = (
    "_ctypes",
    "_overlapped",
    "_socket",
    "asyncio",
    "ctypes",
    "mmap",
    "multiprocessing",
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "sklearn",
    "socket",
    "ssl",
    "torch",
)


class CompatibilityVerifierV4Error(RuntimeError):
    """The v4 trust, stable-byte, or compatibility contract failed closed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _pin_key(pin: Mapping[str, Any]) -> tuple[str, int, str]:
    if set(pin) != {"path", "bytes", "sha256"}:
        raise CompatibilityVerifierV4Error("pin field set changed")
    path = pin.get("path")
    size = pin.get("bytes")
    digest = pin.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or "\\" in path
        or path.startswith("/")
        or ".." in Path(path).parts
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise CompatibilityVerifierV4Error("pin value changed")
    return path, size, digest


def _register_pin(
    registry: dict[str, dict[str, Any]],
    pin: Mapping[str, Any],
    *,
    label: str,
) -> None:
    path, size, digest = _pin_key(pin)
    exact = {"path": path, "bytes": size, "sha256": digest}
    prior = registry.get(path)
    if prior is not None and prior != exact:
        raise CompatibilityVerifierV4Error(f"conflicting pin for {label}: {path}")
    registry[path] = exact


def _collect_pins(value: Any, registry: dict[str, dict[str, Any]], *, label: str) -> None:
    if isinstance(value, Mapping):
        if set(value) == {"path", "bytes", "sha256"}:
            _register_pin(registry, value, label=label)
            return
        for key, child in value.items():
            _collect_pins(child, registry, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_pins(child, registry, label=f"{label}[{index}]")


def _stable_bytes(pin: Mapping[str, Any], *, label: str) -> bytes:
    return _BOOTSTRAP["stable_registry"].lock_pin(dict(pin), label=label)


def _stable_json(pin: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    return _BOOTSTRAP["parse_json_buffer"](_stable_bytes(pin, label=label), label)


def _loaded_forbidden() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if name == R3_ENGINE_MODULE
        or any(name == root or name.startswith(root + ".") for root in FORBIDDEN_ROOTS)
    )


def _validate_config() -> dict[str, Any]:
    config = _BOOTSTRAP["config"]
    if (
        config.get("schema_version")
        != "p2_joint_hydrographic_multitask_layer4.r3_compatibility_verifier.v4"
        or config.get("identity") != IDENTITY
        or config.get("verifier_only") is not True
        or config.get("check_only_default") is not True
        or config.get("append_only_successor_of_v3") is not True
        or config.get("r3_mutation_allowed") is not False
        or config.get("r3_rerun_or_resume_allowed") is not False
        or config.get("execution_authorization_or_lock_allowed") is not False
        or config.get("fit_prediction_truth_decode_or_scoring_allowed") is not False
        or config.get("compatibility_receipt_write_allowed") is not False
        or config.get("official_promotion_allowed") is not False
        or config.get("candidate_or_test_prediction_allowed") is not False
        or config.get("registry_append_allowed") is not False
        or config.get("upload_allowed") is not False
        or config.get("implementation_roles")
        != {
            "CONFIG": (
                "configs/experiments/"
                "p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v4.json"
            ),
            "HELPER": (
                "src/p2_restore/joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v4.py"
            ),
            "CLI": ("scripts/verify_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v4.py"),
            "TESTS": (
                "tests/test_p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v4.py"
            ),
        }
        or config.get("canonical_bootstrap")
        != "scripts/bootstrap_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v4.py"
        or config.get("canonical_external_launcher")
        != "scripts/launch_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v4.ps1"
        or config.get("canonical_absent_pycache_prefix")
        != (
            "artifacts/p2_joint_hydrographic_multitask_layer4_r3_compatibility_"
            "verifier_v4_absent_pycache"
        )
    ):
        raise CompatibilityVerifierV4Error("v4 static identity or prohibition changed")
    runtime = config.get("canonical_runtime_contract")
    stable = config.get("stable_semantic_read_contract")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("required_cli_flags")
        != ["-I", "-S", "-B", "-X pycache_prefix=<canonical-absent-path>"]
        or runtime.get("python_relative_to_workspace") != ".venv-p1/Scripts/python.exe"
        or runtime.get("python_pin")
        != {
            "bytes": 274424,
            "sha256": "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14",
        }
        or runtime.get("pyvenv_cfg_relative_to_workspace") != ".venv-p1/pyvenv.cfg"
        or runtime.get("pyvenv_cfg_pin")
        != {
            "bytes": 339,
            "sha256": "d1fb970854073922d49959ae01539088550613e316cb67f9fac858f586361174",
        }
        or runtime.get("base_python_pin")
        != {
            "relative_to_base": "python.exe",
            "bytes": 104952,
            "sha256": "4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a",
        }
        or runtime.get("prehook_encodings_source_count") != 4
        or runtime.get("prehook_native_module_origins")
        != {"_codecs_kr": "built-in", "_multibytecodec": "built-in"}
        or runtime.get("external_launcher_required") is not True
        or runtime.get("external_launcher_must_be_independently_pinned") is not True
        or runtime.get("external_powershell_host_env") != "P2_POWERSHELL_HOST"
        or runtime.get("external_powershell_host_exact_absolute_path_required") is not True
        or runtime.get("external_powershell_host_must_be_pinned_before_execution") is not True
        or runtime.get("external_powershell_host_pin")
        != {
            "bytes": 301368,
            "sha256": "db6dd81183fe57d22e03b911ec9a30a2fd7c40542e97743615355a6fb44f458f",
        }
        or runtime.get("external_launcher_must_be_pinned_before_execution") is not True
        or runtime.get("host_and_launcher_self_authentication_claimed") is not False
        or runtime.get("external_launcher_holds_all_startup_files_until_child_exit") is not True
        or runtime.get("external_launcher_post_child_same_handle_rehash") is not True
        or runtime.get("canonical_pycache_prefix_must_be_absent") is not True
        or runtime.get("required_sys_flags")
        != {
            "isolated": 1,
            "no_site": 1,
            "dont_write_bytecode": 1,
            "ignore_environment": 1,
            "no_user_site": 1,
            "safe_path": True,
        }
        or runtime.get("stdlib_inventory")
        != {
            "base_root_direct_files": "DLL_ONLY",
            "recursive_roots": ["DLLs", "Lib"],
            "excluded_directory_names": ["__pycache__", "site-packages"],
            "excluded_file_suffixes": [".pyc", ".pyo"],
            "directories": 199,
            "files": 2443,
            "file_bytes": 66487423,
            "algorithm": "SHA256_SORTED_TYPE_NUL_RELATIVE_NUL_BYTES_NUL_FILE_SHA256_LF",
            "sha256": "5cc5d4b2f90199292a4334a6530eaa90c288fd45723ba5290295a3803d13eeba",
        }
        or runtime.get("stdlib_source_execution") != "AUTHENTICATED_BUFFER_COMPILE_EXEC_NO_PYC"
        or runtime.get("third_party_distributions") != []
        or runtime.get("third_party_record_files") != []
        or runtime.get("site_packages_on_sys_path") is not False
        or runtime.get("post_hook_pyc_or_pyo_read_allowed") is not False
        or runtime.get("direct_winapi_create_or_process_allowed") is not False
        or runtime.get("ctypes_socket_mmap_or_spawn_allowed") is not False
        or runtime.get("write_process_network_registry_attempts_required") != 0
        or not isinstance(stable, Mapping)
        or stable.get("windows_share_mode") != "FILE_SHARE_READ_ONLY_NO_SHARE_WRITE_NO_SHARE_DELETE"
        or stable.get("strict_json_from_single_authenticated_buffer") is not True
        or stable.get("open_reparse_point_required") is not True
        or stable.get("post_open_reparse_and_identity_check_required") is not True
        or stable.get("regular_file_nlink_required") != 1
        or stable.get("npy_and_csv_from_authenticated_buffers") is not True
        or stable.get("final_same_handle_identity_and_hash_required") is not True
        or stable.get("concurrent_swap_reopen_gap_allowed") is not False
    ):
        raise CompatibilityVerifierV4Error("v4 runtime or stable-read contract changed")
    if _BOOTSTRAP["observed_implementation_pins"] != _BOOTSTRAP["trusted_pins"]:
        raise CompatibilityVerifierV4Error("v4 implementation pin set changed")
    if runtime["external_startup_file_pins"] != _BOOTSTRAP["startup_pins_public"]:
        raise CompatibilityVerifierV4Error("external startup pin set changed")
    return dict(config)


def _verify_pin_map(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for role, pin in expected.items():
        raw = _stable_bytes(pin, label=f"{label} {role}")
        observed[str(role)] = {
            "path": pin["path"],
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if observed != dict(expected):
        raise CompatibilityVerifierV4Error(f"{label} pin map changed")
    return observed


def _verify_v3_disposition(config: Mapping[str, Any]) -> dict[str, Any]:
    pins = config["v3_disposition_pins"]
    receipt = _stable_json(pins["OWNER_NO_GO"], label="v3 owner NO-GO")
    tombstone = _stable_json(pins["EXECUTION_TOMBSTONE"], label="v3 tombstone")
    expected = config["expected_v3_no_go"]
    codes = [item.get("code") for item in receipt.get("findings", [])]
    if (
        receipt.get("verdict") != expected["verdict"]
        or receipt.get("p0_count") != 0
        or receipt.get("p1_count") != 2
        or receipt.get("independent_qa_review_performed") is not True
        or receipt.get("independent_qa_receipt_created") is not False
        or receipt.get("independent_qa_receipt_path") is not None
        or receipt.get("reviewed_v3_pins") != config["v3_implementation_pins"]
        or codes != expected["required_finding_codes"]
        or tombstone.get("owner_no_go_receipt") != pins["OWNER_NO_GO"]
        or tombstone.get("v3_compatibility_pass_must_fail_closed") is not True
        or tombstone.get("execution_prohibited") is not True
        or tombstone.get("authorization_or_attempt_lock_prohibited") is not True
        or tombstone.get("independent_qa_receipt_created") is not False
    ):
        raise CompatibilityVerifierV4Error("v3 independent NO-GO lineage changed")
    v2 = receipt.get("preserved_v2_no_go_lineage")
    if (
        not isinstance(v2, Mapping)
        or v2.get("verdict") != config["expected_v2_no_go"]["verdict"]
        or v2.get("preserved_v1_verdict") != config["expected_v1_no_go"]["verdict"]
    ):
        raise CompatibilityVerifierV4Error("v2/v1 lineage inside v3 disposition changed")
    return {
        "verdict": expected["verdict"],
        "finding_codes": codes,
        "owner_no_go": dict(pins["OWNER_NO_GO"]),
        "execution_tombstone": dict(pins["EXECUTION_TOMBSTONE"]),
        "independent_qa_review_performed": True,
        "independent_qa_receipt_created": False,
        "preserved_v2_verdict": v2["verdict"],
        "preserved_v1_verdict": v2["preserved_v1_verdict"],
    }


def _verify_v2_disposition(config: Mapping[str, Any]) -> dict[str, Any]:
    pins = config["v2_disposition_pins"]
    receipt = _stable_json(pins["OWNER_NO_GO"], label="v2 owner NO-GO")
    tombstone = _stable_json(pins["EXECUTION_TOMBSTONE"], label="v2 tombstone")
    expected = config["expected_v2_no_go"]
    codes = [item.get("code") for item in receipt.get("findings", [])]
    if (
        receipt.get("verdict") != expected["verdict"]
        or receipt.get("p0_count") != 0
        or receipt.get("p1_count") != 2
        or receipt.get("independent_p2_v2_review_claimed") is not False
        or receipt.get("reviewed_v2_pins") != config["v2_implementation_pins"]
        or codes != expected["required_finding_codes"]
        or tombstone.get("owner_no_go_receipt") != pins["OWNER_NO_GO"]
        or tombstone.get("v2_compatibility_pass_must_fail_closed") is not True
        or tombstone.get("execution_prohibited") is not True
        or tombstone.get("authorization_or_attempt_lock_prohibited") is not True
    ):
        raise CompatibilityVerifierV4Error("v2 owner NO-GO lineage changed")
    v1 = receipt.get("preserved_v1_no_go_lineage")
    if (
        not isinstance(v1, Mapping)
        or v1.get("verdict") != config["expected_v1_no_go"]["verdict"]
        or v1.get("required_finding_codes") != config["expected_v1_no_go"]["required_finding_codes"]
    ):
        raise CompatibilityVerifierV4Error("v1 P1=3 lineage changed")
    return {
        "verdict": expected["verdict"],
        "finding_codes": codes,
        "owner_no_go": dict(pins["OWNER_NO_GO"]),
        "execution_tombstone": dict(pins["EXECUTION_TOMBSTONE"]),
        "preserved_v1_verdict": v1["verdict"],
    }


def _build_semantic_registry(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    pins: dict[str, dict[str, Any]] = {}
    _collect_pins(config["v3_implementation_pins"], pins, label="v3 implementation")
    _collect_pins(config["v3_disposition_pins"], pins, label="v3 disposition")
    v3_receipt = _stable_json(
        config["v3_disposition_pins"]["OWNER_NO_GO"], label="v3 lineage receipt"
    )
    _collect_pins(
        v3_receipt["preserved_v2_no_go_lineage"],
        pins,
        label="preserved v2 disposition",
    )
    _collect_pins(config["v2_implementation_pins"], pins, label="v2 implementation")
    _collect_pins(config["v2_disposition_pins"], pins, label="v2 disposition")
    v2_receipt = _stable_json(
        config["v2_disposition_pins"]["OWNER_NO_GO"], label="v2 lineage receipt"
    )
    _collect_pins(
        v2_receipt["preserved_v1_no_go_lineage"],
        pins,
        label="preserved v1 disposition",
    )
    _register_pin(pins, config["v1_contract_pin"], label="v1 contract")
    v1 = _stable_json(config["v1_contract_pin"], label="frozen v1 contract")
    if v1.get("identity") != "P2_JOINT_HYDROGRAPHIC_MULTITASK_LAYER4_R3_COMPATIBILITY_VERIFIER_V1":
        raise CompatibilityVerifierV4Error("frozen v1 identity changed")
    _collect_pins(v1, pins, label="v1")
    r3_config_pin = v1["r3_implementation_pins"]["CONFIG"]
    r3_config = _stable_json(r3_config_pin, label="r3 config")
    _collect_pins(r3_config, pins, label="r3 config")
    manifest_pin = v1["r3_core_artifact_pins"]["MANIFEST"]
    manifest = _stable_json(manifest_pin, label="r3 manifest")
    artifacts = manifest.get("artifacts")
    output_root = v1["canonical_paths"]["r3_output"].rstrip("/")
    if not isinstance(artifacts, Mapping):
        raise CompatibilityVerifierV4Error("r3 manifest artifact map changed")
    for relative, pin in artifacts.items():
        if not isinstance(relative, str) or not isinstance(pin, Mapping):
            raise CompatibilityVerifierV4Error("r3 manifest artifact pin changed")
        full = dict(pin)
        if full.get("path") != relative:
            raise CompatibilityVerifierV4Error("manifest pin path changed")
        full["path"] = f"{output_root}/{relative}"
        _register_pin(pins, full, label=f"r3 artifact {relative}")
    _BOOTSTRAP["stable_registry"].lock_many(pins)
    return v1, r3_config, pins


def _seal_trees(
    config: Mapping[str, Any],
    v1: Mapping[str, Any],
    pins: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    stable = _BOOTSTRAP["stable_registry"]
    output_inventory = stable.seal_tree(v1["canonical_paths"]["r3_output"], pins)
    control_inventory = stable.seal_tree(v1["canonical_paths"]["r3_control"], pins)
    if output_inventory != v1["r3_output_inventory"]:
        raise CompatibilityVerifierV4Error("r3 output inventory changed")
    if control_inventory != v1["r3_control_inventory"]:
        raise CompatibilityVerifierV4Error("r3 control inventory changed")
    v1_control_root = (
        "artifacts/p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v1_control"
    )
    v1_inventory = stable.seal_tree(v1_control_root, pins)
    v2_inventory = stable.seal_tree(
        "artifacts/p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v2_control",
        pins,
    )
    v3_inventory = stable.seal_tree(
        "artifacts/p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v3_control",
        pins,
    )
    if (
        v1_inventory["directories"] != 0
        or v1_inventory["files"] != 2
        or v2_inventory["directories"] != 0
        or v2_inventory["files"] != 2
        or v3_inventory["directories"] != 0
        or v3_inventory["files"] != 2
    ):
        raise CompatibilityVerifierV4Error("compatibility disposition tree changed")
    v4_control = config["canonical_paths"]["v4_control"]
    if _BOOTSTRAP["path_exists"](v4_control):
        raise CompatibilityVerifierV4Error("v4 control must remain absent before fresh QA")
    return {
        "r3_output": output_inventory,
        "r3_control": control_inventory,
        "v1_disposition": v1_inventory,
        "v2_disposition": v2_inventory,
        "v3_disposition": v3_inventory,
        "v4_control_exists": False,
    }


def _load_r3_guard(pin: Mapping[str, Any]) -> Any:
    raw = _stable_bytes(pin, label="r3 guard")
    path = _BOOTSTRAP["contained_path"](pin["path"], True, "file")
    if R3_MODULE in sys.modules:
        raise CompatibilityVerifierV4Error("authenticated r3 module already loaded")
    module = types.ModuleType(R3_MODULE)
    module.__file__ = str(path)
    module.__package__ = "p2_restore"
    module.__loader__ = None
    sys.modules[R3_MODULE] = module
    try:
        code = compile(raw, str(path), "exec", dont_inherit=True, optimize=0)
        exec(code, module.__dict__)  # noqa: S102
    except BaseException:
        sys.modules.pop(R3_MODULE, None)
        raise
    if _loaded_forbidden():
        raise CompatibilityVerifierV4Error("r3 guard imported forbidden code")
    return module


def _npy_payload_from_buffer(r3: Any, path: Path) -> tuple[bytes, int]:
    raw = _BOOTSTRAP["stable_registry"].bytes_for_path(path)
    stream = io.BytesIO(raw)
    if stream.read(6) != b"\x93NUMPY":
        raise r3.Layer4ContractError("blind prediction is not an NPY artifact")
    version = stream.read(2)
    if version == b"\x01\x00":
        length_bytes = stream.read(2)
        if len(length_bytes) != 2:
            raise r3.Layer4ContractError("blind prediction NPY header is truncated")
        header_length = struct.unpack("<H", length_bytes)[0]
        encoding = "latin1"
    elif version in {b"\x02\x00", b"\x03\x00"}:
        length_bytes = stream.read(4)
        if len(length_bytes) != 4:
            raise r3.Layer4ContractError("blind prediction NPY header is truncated")
        header_length = struct.unpack("<I", length_bytes)[0]
        encoding = "utf-8" if version == b"\x03\x00" else "latin1"
    else:
        raise r3.Layer4ContractError("blind prediction NPY version changed")
    header_bytes = stream.read(header_length)
    if len(header_bytes) != header_length:
        raise r3.Layer4ContractError("blind prediction NPY header is truncated")
    try:
        header = ast.literal_eval(header_bytes.decode(encoding).strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        raise r3.Layer4ContractError("blind prediction NPY header is invalid") from exc
    if (
        not isinstance(header, dict)
        or set(header) != {"descr", "fortran_order", "shape"}
        or header["descr"] != "<f8"
        or header["fortran_order"] is not False
        or not isinstance(header["shape"], tuple)
        or len(header["shape"]) != 1
        or not isinstance(header["shape"][0], int)
        or isinstance(header["shape"][0], bool)
        or header["shape"][0] <= 0
    ):
        raise r3.Layer4ContractError("blind prediction NPY contract changed")
    payload = stream.read()
    rows = int(header["shape"][0])
    if len(payload) != rows * 8:
        raise r3.Layer4ContractError("blind prediction NPY payload size changed")
    if not all(math.isfinite(value[0]) for value in struct.iter_unpack("<d", payload)):
        raise r3.Layer4ContractError("blind prediction NPY payload is non-finite")
    return payload, rows


def _csv_rows_from_buffer(r3: Any, path: Path) -> tuple[list[str], int]:
    raw = _BOOTSTRAP["stable_registry"].bytes_for_path(path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise r3.Layer4ContractError("sealed CSV is not UTF-8") from exc
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise r3.Layer4ContractError("sealed CSV is empty") from exc
    return header, sum(1 for _ in reader)


def _verify_fold_audits(
    fold_audits: Mapping[str, Any],
    commitment_audits: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    order = list(contract["canonical_fold_order"])
    expected_priors = contract["verified_prior_fold_commitments_by_canonical_fold"]
    forbidden = list(contract["forbidden_decode_fields"])
    if set(fold_audits) != set(order) or set(commitment_audits) != set(order):
        raise CompatibilityVerifierV4Error("fold audit exact key set changed")
    verified: list[dict[str, Any]] = []
    for index, name in enumerate(order):
        audit = fold_audits[name]
        committed = commitment_audits[name]
        if not isinstance(audit, Mapping) or dict(audit) != dict(committed):
            raise CompatibilityVerifierV4Error(f"fold audit differs from commitment: {name}")
        prior = audit.get("verified_prior_fold_commitments")
        if (
            audit.get("fold") != name
            or expected_priors.get(name) != index
            or not isinstance(prior, int)
            or isinstance(prior, bool)
            or prior != index
        ):
            raise CompatibilityVerifierV4Error(f"fold audit identity changed: {name}")
        for field in forbidden:
            value = audit.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value != 0:
                raise CompatibilityVerifierV4Error(
                    f"forbidden fold audit decode changed: {name}:{field}"
                )
        verified.append({"fold": name, "verified_prior_fold_commitments": prior})
    return {
        "canonical_fold_order": order,
        "verified": verified,
        "mapping_insertion_order_ignored": True,
        "exact_fold_commitment_audit_equality": True,
    }


def _verify_v9(anchor: Mapping[str, Any]) -> dict[str, Any]:
    pin = {key: anchor[key] for key in ("path", "bytes", "sha256")}
    raw = _stable_bytes(pin, label="v9 ledger")
    records = [
        _BOOTSTRAP["parse_json_buffer"](line, f"v9 line {index}")
        for index, line in enumerate(raw.splitlines(), start=1)
        if line.strip()
    ]
    sequences = [record.get("seq") for record in records]
    if len(records) != anchor["record_count"] or sequences != anchor["sequences"]:
        raise CompatibilityVerifierV4Error("v9 sequence changed")
    for index, record in enumerate(records):
        claimed = record.get("event_sha256")
        unsigned = {key: value for key, value in record.items() if key != "event_sha256"}
        if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != claimed:
            raise CompatibilityVerifierV4Error("v9 event hash changed")
        if index and record.get("previous_event_sha256") != records[index - 1]["event_sha256"]:
            raise CompatibilityVerifierV4Error("v9 event chain changed")
    head = records[-1]
    uploads = sum(
        int(bool(record.get("payload", {}).get("upload_performed", False))) for record in records
    )
    if (
        head.get("seq") != anchor["head_sequence"]
        or head.get("event_sha256") != anchor["head_event_sha256"]
        or uploads != 0
    ):
        raise CompatibilityVerifierV4Error("v9 head or upload state changed")
    return {
        "pin": pin,
        "sequences": sequences,
        "head_sequence": head["seq"],
        "head_event_sha256": head["event_sha256"],
        "uploads": uploads,
    }


def _run_original_verifier(
    r3: Any,
    r3_config: Mapping[str, Any],
    expected_failure: Mapping[str, Any],
) -> dict[str, Any]:
    stable = _BOOTSTRAP["stable_registry"]
    originals = {
        "strict_json_object": r3.strict_json_object,
        "sha256_file": r3.sha256_file,
        "_npy_little_endian_float64_payload": r3._npy_little_endian_float64_payload,
        "_csv_header_and_rows": r3._csv_header_and_rows,
    }

    def stable_json(path: Path) -> dict[str, Any]:
        return _BOOTSTRAP["parse_json_buffer"](stable.bytes_for_path(path), f"r3 JSON {path.name}")

    def stable_sha(path: Path) -> str:
        return hashlib.sha256(stable.bytes_for_path(path)).hexdigest()

    r3.strict_json_object = stable_json
    r3.sha256_file = stable_sha
    r3._npy_little_endian_float64_payload = lambda path: _npy_payload_from_buffer(r3, path)
    r3._csv_header_and_rows = lambda path: _csv_rows_from_buffer(r3, path)
    try:
        loaded = r3.load_canonical_config(_BOOTSTRAP["workspace"])
        if loaded != r3_config:
            raise CompatibilityVerifierV4Error("r3 config deep value changed")
        try:
            r3.verify_seal(_BOOTSTRAP["workspace"], loaded)
        except r3.Layer4ContractError as exc:
            if (
                type(exc).__name__ != expected_failure["exception_type"]
                or str(exc) != expected_failure["message"]
            ):
                raise CompatibilityVerifierV4Error(
                    "original r3 verifier has a different failure"
                ) from exc
        else:
            raise CompatibilityVerifierV4Error(
                "original r3 verifier no longer has the exact false-negative"
            )
    finally:
        for name, original in originals.items():
            setattr(r3, name, original)
        if any(getattr(r3, name) is not original for name, original in originals.items()):
            raise CompatibilityVerifierV4Error("r3 stable-byte adapters were not restored")
    return {
        "failure": dict(expected_failure),
        "authenticated_buffer_adapters": sorted(originals),
        "all_original_globals_restored": True,
    }


def _verify_result(v1: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    core = v1["r3_core_artifact_pins"]
    receipt = _stable_json(core["TRAINING_RECEIPT"], label="r3 training receipt")
    contract = v1["fold_audit_compatibility_contract"]
    if list(receipt.get("fold_blind_input_audits", {})) != contract["persisted_mapping_order"]:
        raise CompatibilityVerifierV4Error("persisted fold audit order changed")
    commitment_audits: dict[str, Any] = {}
    for role, name in (
        ("FOLD_SEP_OCT", "outer_2024_sep_oct"),
        ("FOLD_MAY_JUN", "outer_2025_may_jun"),
        ("FOLD_JUL_AUG", "outer_2025_jul_aug"),
    ):
        commitment = _stable_json(core[role], label=f"r3 fold commitment {name}")
        commitment_audits[name] = commitment.get("blind_input_audit")
    corrected = _verify_fold_audits(receipt["fold_blind_input_audits"], commitment_audits, contract)
    decision = _stable_json(core["DECISION"], label="r3 decision")
    evidence = _stable_json(core["EVIDENCE"], label="r3 evidence")
    seal = _stable_json(core["SEAL"], label="r3 seal")
    expected = v1["expected_result"]
    checks = {
        "status": decision.get("status") == seal.get("status") == expected["status"],
        "local": decision.get("local_qualification") is False
        and evidence.get("local_qualification") is False
        and seal.get("local_qualification") is False,
        "passed": decision.get("passed") is False,
        "promotion": decision.get("official_promotion") is False
        and decision.get("official_promotion_allowed") is False
        and seal.get("official_promotion_allowed") is False,
        "candidate": decision.get("candidate_generated") is False
        and seal.get("candidate_generated") is False,
        "test": decision.get("test_prediction_generated") is False
        and seal.get("test_prediction_generated") is False,
        "upload": decision.get("upload_performed") is False and seal.get("upload_count") == 0,
        "finite_metrics": all(
            math.isfinite(float(point[key]))
            for point in evidence.get("points", [])
            for key in ("incumbent", "challenger", "delta")
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise CompatibilityVerifierV4Error(f"research-only result changed: {failed}")
    return corrected, checks


def verify_static_compatibility() -> dict[str, Any]:
    """Verify the frozen r3 result through stable authenticated bytes only."""

    config = _validate_config()
    v3_disposition = _verify_v3_disposition(config)
    v2_disposition = _verify_v2_disposition(config)
    v1, r3_config, pins = _build_semantic_registry(config)
    trees_before = _seal_trees(config, v1, pins)
    v3_implementation = _verify_pin_map(config["v3_implementation_pins"], label="v3 implementation")
    v2_implementation = _verify_pin_map(config["v2_implementation_pins"], label="v2 implementation")
    r3_implementation = _verify_pin_map(v1["r3_implementation_pins"], label="r3 implementation")
    r3_controls = _verify_pin_map(v1["r3_control_pins"], label="r3 control")
    r3_core = _verify_pin_map(v1["r3_core_artifact_pins"], label="r3 core")
    v9 = _verify_v9(v1["v9_anchor"])
    r3 = _load_r3_guard(v1["r3_implementation_pins"]["GUARD"])
    original = _run_original_verifier(r3, r3_config, v1["original_verifier_expected_failure"])
    corrected, result_checks = _verify_result(v1)
    trees_after = _seal_trees(config, v1, pins)
    if trees_after != trees_before:
        raise CompatibilityVerifierV4Error("protected tree changed during verification")
    final_registry = _BOOTSTRAP["stable_registry"].final_reverify()
    _BOOTSTRAP["assert_runtime"]()
    if _loaded_forbidden():
        raise CompatibilityVerifierV4Error("forbidden module appeared during verification")
    counters = dict(config["static_counters"])
    report = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.r3_compatibility_check.v4",
        "status": (
            "PASS_OWNER_STATIC_EXTERNAL_STARTUP_ROOT_STABLE_R3_COMPATIBILITY_"
            "AWAITING_FRESH_INDEPENDENT_QA_RESEARCH_ONLY_LOCAL_FAIL"
        ),
        "identity": IDENTITY,
        "v4_implementation_pins": dict(_BOOTSTRAP["trusted_pins"]),
        "bootstrap_observed_pin": dict(_BOOTSTRAP["bootstrap_observed_pin"]),
        "external_startup_trust": _BOOTSTRAP["external_startup_report"](),
        "v3_disposition": v3_disposition,
        "v3_implementation_pins": v3_implementation,
        "v2_disposition": v2_disposition,
        "v2_implementation_pins": v2_implementation,
        "r3_implementation_pins": r3_implementation,
        "r3_control_pins": r3_controls,
        "r3_core_pins": r3_core,
        "original_r3_verifier": original,
        "corrected_fold_audit_verification": corrected,
        "result_checks": result_checks,
        "r3_control_inventory": trees_before["r3_control"],
        "r3_output_inventory": trees_before["r3_output"],
        "v9": v9,
        "stable_registry": final_registry,
        "dependency_trust": _BOOTSTRAP["dependency_report"](),
        "v4_control_exists": False,
        "pre_execution_qa_exists": False,
        "compatibility_receipt_exists": False,
        "files_written": 0,
        **counters,
    }
    report["summary_sha256"] = hashlib.sha256(_canonical_json_bytes(report)).hexdigest()
    return report


__all__ = [
    "CompatibilityVerifierV4Error",
    "IDENTITY",
    "R3_ENGINE_MODULE",
    "R3_MODULE",
    "TRUSTED_ROLES",
    "verify_static_compatibility",
]
