"""Authenticated, read-only P2 Layer-4 r3 compatibility verification.

This module is intentionally not importable through normal Python machinery.
The canonical v2 bootstrap authenticates its exact source bytes, compiles that
buffer, and injects the non-cyclic trust context before execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    _BOOTSTRAP = _P2_V2_BOOTSTRAP_CONTEXT  # type: ignore[name-defined]  # noqa: F821
except NameError as exc:  # pragma: no cover - exercised by a subprocess test
    raise RuntimeError("P2 v2 helper requires the authenticated bootstrap") from exc


IDENTITY = "P2_JOINT_HYDROGRAPHIC_MULTITASK_LAYER4_R3_COMPATIBILITY_VERIFIER_V2"
R3_MODULE = "p2_restore.joint_hydrographic_multitask_layer4_contract_r3_v2_authenticated"
R3_ENGINE_MODULE = "p2_restore.joint_hydrographic_multitask_layer4_execution_r3"
TRUSTED_ROLES = ("CONFIG", "HELPER", "CLI", "TESTS")
FORBIDDEN_NUMERICAL_ROOTS = ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "torch")


class CompatibilityVerifierV2Error(RuntimeError):
    """The authenticated v2 compatibility contract was not satisfied."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise CompatibilityVerifierV2Error(f"{label} field set changed")


def _pin_key(pin: Mapping[str, Any]) -> tuple[str, int, str]:
    if set(pin) != {"path", "bytes", "sha256"}:
        raise CompatibilityVerifierV2Error("pin field set changed")
    path = pin.get("path")
    size = pin.get("bytes")
    digest = pin.get("sha256")
    if (
        not isinstance(path, str)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise CompatibilityVerifierV2Error("pin value changed")
    return path, size, digest


def _authenticated_bytes(pin: Mapping[str, Any], *, label: str) -> bytes:
    _pin_key(pin)
    return _BOOTSTRAP["authenticated_bytes"](dict(pin), label)


def _authenticated_json(pin: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    _pin_key(pin)
    return _BOOTSTRAP["authenticated_json"](dict(pin), label)


def _contained(relative: str, *, must_exist: bool = True, kind: str | None = None) -> Path:
    return _BOOTSTRAP["contained_path"](relative, must_exist, kind)


def _loaded_forbidden() -> list[str]:
    names: list[str] = []
    for name in sys.modules:
        if name == R3_ENGINE_MODULE or any(
            name == root or name.startswith(root + ".") for root in FORBIDDEN_NUMERICAL_ROOTS
        ):
            names.append(name)
    return sorted(names)


def _validate_bootstrap_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    if sys.dont_write_bytecode is not True:
        raise CompatibilityVerifierV2Error("sys.dont_write_bytecode is not enforced")
    _BOOTSTRAP["assert_firewall"]()
    if _loaded_forbidden():
        raise CompatibilityVerifierV2Error("forbidden numerical or r3 engine module is loaded")
    config = _BOOTSTRAP["config"]
    anchor = _BOOTSTRAP["anchor"]
    _exact_keys(
        config,
        {
            "schema_version",
            "created_at_kst",
            "status",
            "problem",
            "identity",
            "verifier_only",
            "check_only_default",
            "append_only_successor_of_v1",
            "r3_mutation_allowed",
            "r3_rerun_or_resume_allowed",
            "execution_authorization_or_lock_allowed",
            "fit_prediction_truth_decode_or_scoring_allowed",
            "compatibility_receipt_write_allowed",
            "official_promotion_allowed",
            "candidate_or_test_prediction_allowed",
            "registry_append_allowed",
            "upload_allowed",
            "implementation_roles",
            "trusted_bootstrap_contract",
            "canonical_paths",
            "v1_contract_pin",
            "v1_implementation_pins",
            "v1_disposition_pins",
            "v1_independent_qa",
            "expected_result",
            "static_counters",
        },
        label="v2 config",
    )
    if (
        config.get("schema_version")
        != "p2_joint_hydrographic_multitask_layer4.r3_compatibility_verifier.v2"
        or config.get("identity") != IDENTITY
        or config.get("problem") != "P2"
        or config.get("verifier_only") is not True
        or config.get("check_only_default") is not True
        or config.get("append_only_successor_of_v1") is not True
    ):
        raise CompatibilityVerifierV2Error("v2 config identity changed")
    false_flags = (
        "r3_mutation_allowed",
        "r3_rerun_or_resume_allowed",
        "execution_authorization_or_lock_allowed",
        "fit_prediction_truth_decode_or_scoring_allowed",
        "compatibility_receipt_write_allowed",
        "official_promotion_allowed",
        "candidate_or_test_prediction_allowed",
        "registry_append_allowed",
        "upload_allowed",
    )
    if any(config.get(name) is not False for name in false_flags):
        raise CompatibilityVerifierV2Error("v2 read-only firewall changed")
    if any(value != 0 for value in config.get("static_counters", {}).values()):
        raise CompatibilityVerifierV2Error("v2 static counter changed")
    contract = config["trusted_bootstrap_contract"]
    required_true = (
        "external_fresh_qa_must_pin_bootstrap",
        "bootstrap_hardcodes_trust_anchor_bytes_and_sha256",
        "authenticate_all_roles_before_any_helper_or_cli_compile",
        "compile_and_exec_authenticated_buffers_only",
        "single_authenticated_buffer_json_parse_required",
        "after_read_file_identity_check_required",
        "full_implementation_ancestor_reparse_rejection_required",
        "sys_dont_write_bytecode_required",
        "write_audit_firewall_required",
    )
    if (
        contract.get("canonical_entrypoint_role") != "BOOTSTRAP"
        or contract.get("trust_anchor_exact_roles") != list(TRUSTED_ROLES)
        or any(contract.get(name) is not True for name in required_true)
        or contract.get("source_file_loader_or_pyc_execution_allowed") is not False
        or contract.get("numerical_or_r3_engine_import_allowed") is not False
    ):
        raise CompatibilityVerifierV2Error("trusted bootstrap contract changed")
    trusted = anchor.get("trusted_files")
    observed = _BOOTSTRAP["observed_implementation_pins"]
    if (
        not isinstance(trusted, Mapping)
        or set(trusted) != set(TRUSTED_ROLES)
        or dict(observed) != dict(trusted)
        or _BOOTSTRAP["reverify_trusted_files"]() != dict(trusted)
    ):
        raise CompatibilityVerifierV2Error("current v2 implementation pin set changed")
    expected_paths = {role: config["implementation_roles"][role] for role in TRUSTED_ROLES}
    if {role: trusted[role]["path"] for role in TRUSTED_ROLES} != expected_paths:
        raise CompatibilityVerifierV2Error("v2 implementation role path changed")
    return config, anchor


def _verify_pin_map(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for role, pin in expected.items():
        _authenticated_bytes(pin, label=f"{label} {role}")
        observed[str(role)] = dict(pin)
    return observed


def _register_pin(
    registry: dict[str, dict[str, Any]], pin: Mapping[str, Any], *, label: str
) -> None:
    path, _size, _digest = _pin_key(pin)
    _contained(path, must_exist=True, kind="file")
    normalized = dict(pin)
    prior = registry.get(path)
    if prior is not None and prior != normalized:
        raise CompatibilityVerifierV2Error(f"conflicting authenticated pin: {label}:{path}")
    registry[path] = normalized


def _collect_pins(value: Any, registry: dict[str, dict[str, Any]], *, label: str) -> None:
    if isinstance(value, Mapping):
        if {"path", "bytes", "sha256"}.issubset(value):
            candidate = {key: value[key] for key in ("path", "bytes", "sha256")}
            _register_pin(registry, candidate, label=label)
        for key, item in value.items():
            _collect_pins(item, registry, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _collect_pins(item, registry, label=f"{label}[{index}]")


def _load_r3_guard(pin: Mapping[str, Any]) -> Any:
    if R3_MODULE in sys.modules:
        raise CompatibilityVerifierV2Error("authenticated r3 guard module already exists")
    source = _authenticated_bytes(pin, label="frozen r3 guard source")
    path = _contained(str(pin["path"]), kind="file")
    module = types.ModuleType(R3_MODULE)
    module.__file__ = str(path)
    module.__package__ = "p2_restore"
    module.__loader__ = None
    sys.modules[R3_MODULE] = module
    try:
        code = compile(source, str(path), "exec", dont_inherit=True, optimize=0)
        exec(code, module.__dict__)  # noqa: S102
    except BaseException:
        sys.modules.pop(R3_MODULE, None)
        raise
    if _loaded_forbidden():
        raise CompatibilityVerifierV2Error("r3 guard loaded a forbidden module")
    return module


def _inventory(
    relative_root: str,
    registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    root = _contained(relative_root, kind="directory")
    entries: list[dict[str, Any]] = []
    for item in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        relative = item.relative_to(root).as_posix()
        workspace_relative = _BOOTSTRAP["relative_plain_path"](item)
        if item.is_dir():
            entries.append({"path": relative, "type": "directory"})
        elif item.is_file():
            pin = registry.get(workspace_relative)
            if pin is None:
                raise CompatibilityVerifierV2Error(
                    f"inventory file lacks an authenticated pin: {workspace_relative}"
                )
            raw = _authenticated_bytes(pin, label=f"inventory {workspace_relative}")
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        else:
            raise CompatibilityVerifierV2Error("inventory contains a special entry")
    files = [entry for entry in entries if entry["type"] == "file"]
    directories = [entry for entry in entries if entry["type"] == "directory"]
    payload = _canonical_json_bytes(entries) + b"\n"
    return {
        "directories": len(directories),
        "files": len(files),
        "file_bytes": sum(int(entry["bytes"]) for entry in files),
        "algorithm": "SHA256_CANONICAL_JSON_SORTED_RELATIVE_PATH_TYPE_BYTES_SHA256_WITH_LF",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _verify_disposition(config: Mapping[str, Any]) -> dict[str, Any]:
    pins = config["v1_disposition_pins"]
    receipt = _authenticated_json(pins["OWNER_NO_GO"], label="v1 owner NO-GO")
    tombstone = _authenticated_json(pins["EXECUTION_TOMBSTONE"], label="v1 execution tombstone")
    qa = config["v1_independent_qa"]
    finding_codes = [item.get("code") for item in receipt.get("findings", [])]
    if (
        receipt.get("verdict") != qa["verdict"]
        or receipt.get("p0_count") != qa["p0_count"]
        or receipt.get("p1_count") != qa["p1_count"]
        or receipt.get("reviewer") != qa["reviewer"]
        or finding_codes != qa["required_finding_codes"]
        or receipt.get("reviewed_state", {}).get("check_only_summary_sha256")
        != qa["check_only_summary_sha256"]
        or receipt.get("reviewed_v1_pins") != config["v1_implementation_pins"]
    ):
        raise CompatibilityVerifierV2Error("v1 owner NO-GO disposition changed")
    receipt_pin = tombstone.get("owner_no_go_receipt")
    if (
        receipt_pin != pins["OWNER_NO_GO"]
        or tombstone.get("execution_prohibited") is not True
        or tombstone.get("v1_compatibility_pass_must_fail_closed") is not True
        or tombstone.get("authorization_or_attempt_lock_prohibited") is not True
    ):
        raise CompatibilityVerifierV2Error("v1 execution tombstone changed")
    return {
        "owner_no_go": dict(pins["OWNER_NO_GO"]),
        "execution_tombstone": dict(pins["EXECUTION_TOMBSTONE"]),
        "verdict": qa["verdict"],
        "finding_codes": finding_codes,
    }


def _verify_fold_audits(
    fold_audits: Mapping[str, Any],
    commitment_audits: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    order = list(contract["canonical_fold_order"])
    expected_priors = contract["verified_prior_fold_commitments_by_canonical_fold"]
    forbidden = list(contract["forbidden_decode_fields"])
    if set(fold_audits) != set(order) or set(commitment_audits) != set(order):
        raise CompatibilityVerifierV2Error("fold audit exact key set changed")
    verified: list[dict[str, Any]] = []
    for index, name in enumerate(order):
        audit = fold_audits[name]
        committed = commitment_audits[name]
        if not isinstance(audit, Mapping) or dict(audit) != dict(committed):
            raise CompatibilityVerifierV2Error(f"fold audit differs from commitment: {name}")
        prior = audit.get("verified_prior_fold_commitments")
        if (
            audit.get("fold") != name
            or expected_priors.get(name) != index
            or not isinstance(prior, int)
            or isinstance(prior, bool)
            or prior != index
        ):
            raise CompatibilityVerifierV2Error(f"fold audit identity/order changed: {name}")
        for field in forbidden:
            value = audit.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value != 0:
                raise CompatibilityVerifierV2Error(
                    f"forbidden fold audit decode changed: {name}:{field}"
                )
        verified.append({"fold": name, "verified_prior_fold_commitments": prior})
    return {
        "canonical_fold_order": order,
        "verified": verified,
        "mapping_insertion_order_ignored": True,
        "exact_fold_commitment_audit_equality": True,
    }


def _verify_v9(pin: Mapping[str, Any], anchor: Mapping[str, Any]) -> dict[str, Any]:
    exact_pin = {key: pin[key] for key in ("path", "bytes", "sha256")}
    raw = _authenticated_bytes(exact_pin, label="v9 ledger")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        if line.strip():
            records.append(_BOOTSTRAP["parse_json_buffer"](line, f"v9 line {index}"))
    sequences = [record.get("seq") for record in records]
    if len(records) != anchor["record_count"] or sequences != anchor["sequences"]:
        raise CompatibilityVerifierV2Error("v9 sequence changed")
    for index, record in enumerate(records):
        claimed = record.get("event_sha256")
        unsigned = {key: value for key, value in record.items() if key != "event_sha256"}
        if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != claimed:
            raise CompatibilityVerifierV2Error("v9 event hash changed")
        if index and record.get("previous_event_sha256") != records[index - 1]["event_sha256"]:
            raise CompatibilityVerifierV2Error("v9 event chain changed")
    head = records[-1]
    uploads = sum(
        int(bool(record.get("payload", {}).get("upload_performed", False))) for record in records
    )
    if (
        head.get("seq") != anchor["head_sequence"]
        or head.get("event_sha256") != anchor["head_event_sha256"]
        or uploads != anchor["uploads"]
    ):
        raise CompatibilityVerifierV2Error("v9 head or upload state changed")
    return {
        "pin": exact_pin,
        "sequences": sequences,
        "head_sequence": head["seq"],
        "head_event_sha256": head["event_sha256"],
        "uploads": uploads,
    }


def _build_registry(
    v2_config: Mapping[str, Any], v1_config: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    _collect_pins(v2_config, registry, label="v2")
    _collect_pins(v1_config, registry, label="v1")
    r3_config_pin = v1_config["r3_implementation_pins"]["CONFIG"]
    r3_config = _authenticated_json(r3_config_pin, label="r3 canonical config")
    _collect_pins(r3_config, registry, label="r3 config")
    manifest_pin = v1_config["r3_core_artifact_pins"]["MANIFEST"]
    manifest = _authenticated_json(manifest_pin, label="r3 manifest")
    output_root = v1_config["canonical_paths"]["r3_output"].rstrip("/")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CompatibilityVerifierV2Error("r3 manifest artifact map changed")
    for relative, pin in artifacts.items():
        if not isinstance(relative, str) or not isinstance(pin, Mapping):
            raise CompatibilityVerifierV2Error("r3 manifest artifact pin changed")
        full_pin = dict(pin)
        if full_pin.get("path") != relative:
            raise CompatibilityVerifierV2Error("r3 manifest artifact path changed")
        full_pin["path"] = f"{output_root}/{relative}"
        _register_pin(registry, full_pin, label=f"r3 manifest artifact {relative}")
    return registry, r3_config, manifest


def _snapshot(
    config: Mapping[str, Any],
    v1: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    trusted = _BOOTSTRAP["reverify_trusted_files"]()
    anchor_pin = _BOOTSTRAP["reverify_anchor"]()
    v1_implementation = _verify_pin_map(config["v1_implementation_pins"], label="v1 implementation")
    r3_implementation = _verify_pin_map(v1["r3_implementation_pins"], label="r3 implementation")
    r3_controls = _verify_pin_map(v1["r3_control_pins"], label="r3 control")
    r3_core = _verify_pin_map(v1["r3_core_artifact_pins"], label="r3 core")
    disposition = _verify_pin_map(config["v1_disposition_pins"], label="v1 disposition")
    r3_control_inventory = _inventory(v1["canonical_paths"]["r3_control"], registry)
    r3_output_inventory = _inventory(v1["canonical_paths"]["r3_output"], registry)
    if r3_control_inventory != v1["r3_control_inventory"]:
        raise CompatibilityVerifierV2Error("r3 control inventory changed")
    if r3_output_inventory != v1["r3_output_inventory"]:
        raise CompatibilityVerifierV2Error("r3 output inventory changed")
    v1_control = Path(config["v1_disposition_pins"]["OWNER_NO_GO"]["path"]).parent.as_posix()
    v1_control_inventory = _inventory(v1_control, registry)
    if v1_control_inventory["directories"] != 0 or v1_control_inventory["files"] != 2:
        raise CompatibilityVerifierV2Error("v1 disposition control allowlist changed")
    v2_control = _contained(config["canonical_paths"]["v2_control"], must_exist=False, kind=None)
    if v2_control.exists():
        raise CompatibilityVerifierV2Error("v2 control must remain absent before QA")
    v9 = _verify_v9(v1["v9_anchor"], v1["v9_anchor"])
    return {
        "trusted_implementation": trusted,
        "trust_anchor": anchor_pin,
        "v1_implementation": v1_implementation,
        "v1_disposition": disposition,
        "r3_implementation": r3_implementation,
        "r3_controls": r3_controls,
        "r3_core": r3_core,
        "r3_control_inventory": r3_control_inventory,
        "r3_output_inventory": r3_output_inventory,
        "v1_control_inventory": v1_control_inventory,
        "v9": v9,
        "v2_control_exists": False,
    }


def _verify_result_semantics(
    v1: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    core = v1["r3_core_artifact_pins"]
    receipt = _authenticated_json(core["TRAINING_RECEIPT"], label="r3 training receipt")
    contract = v1["fold_audit_compatibility_contract"]
    persisted_order = list(receipt.get("fold_blind_input_audits", {}))
    if persisted_order != contract["persisted_mapping_order"]:
        raise CompatibilityVerifierV2Error("persisted fold-audit mapping order changed")
    commitment_audits: dict[str, Any] = {}
    for role, name in (
        ("FOLD_SEP_OCT", "outer_2024_sep_oct"),
        ("FOLD_MAY_JUN", "outer_2025_may_jun"),
        ("FOLD_JUL_AUG", "outer_2025_jul_aug"),
    ):
        payload = _authenticated_json(core[role], label=f"r3 fold commitment {name}")
        commitment_audits[name] = payload.get("blind_input_audit")
    corrected = _verify_fold_audits(receipt["fold_blind_input_audits"], commitment_audits, contract)
    decision = _authenticated_json(core["DECISION"], label="r3 gate decision")
    evidence = _authenticated_json(core["EVIDENCE"], label="r3 evidence")
    seal = _authenticated_json(core["SEAL"], label="r3 seal")
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
        raise CompatibilityVerifierV2Error(f"research-only result changed: {failed}")
    return corrected, checks


def verify_static_compatibility() -> dict[str, Any]:
    """Run the authenticated static verifier without any write capability."""

    config, anchor = _validate_bootstrap_contract()
    v1 = _authenticated_json(config["v1_contract_pin"], label="frozen v1 contract")
    if v1.get("identity") != "P2_JOINT_HYDROGRAPHIC_MULTITASK_LAYER4_R3_COMPATIBILITY_VERIFIER_V1":
        raise CompatibilityVerifierV2Error("frozen v1 contract identity changed")
    disposition = _verify_disposition(config)
    registry, r3_config, _manifest = _build_registry(config, v1)
    before = _snapshot(config, v1, registry)
    guard_pin = v1["r3_implementation_pins"]["GUARD"]
    r3 = _load_r3_guard(guard_pin)

    original_json = r3.strict_json_object

    def authenticated_strict_json(path: Path) -> dict[str, Any]:
        relative = _BOOTSTRAP["relative_plain_path"](path)
        pin = registry.get(relative)
        if pin is None:
            raise CompatibilityVerifierV2Error(f"r3 requested unregistered JSON: {relative}")
        return _authenticated_json(pin, label=f"r3 JSON {relative}")

    expected_failure = v1["original_verifier_expected_failure"]
    r3.strict_json_object = authenticated_strict_json
    try:
        loaded = r3.load_canonical_config(_BOOTSTRAP["workspace"])
        if loaded != r3_config:
            raise CompatibilityVerifierV2Error("authenticated r3 config deep value changed")
        try:
            r3.verify_seal(_BOOTSTRAP["workspace"], loaded)
        except r3.Layer4ContractError as exc:
            if (
                type(exc).__name__ != expected_failure["exception_type"]
                or str(exc) != expected_failure["message"]
            ):
                raise CompatibilityVerifierV2Error(
                    "original r3 verifier has a different failure"
                ) from exc
        else:
            raise CompatibilityVerifierV2Error(
                "original r3 verifier no longer has the exact pinned false-negative"
            )
    finally:
        r3.strict_json_object = original_json
        if r3.strict_json_object is not original_json:
            raise CompatibilityVerifierV2Error("r3 authenticated JSON adapter was not restored")

    corrected, result_checks = _verify_result_semantics(v1, registry)
    after = _snapshot(config, v1, registry)
    if after != before:
        raise CompatibilityVerifierV2Error("protected filesystem state changed during check")
    if _BOOTSTRAP["reverify_trusted_files"]() != anchor["trusted_files"]:
        raise CompatibilityVerifierV2Error("v2 implementation changed during check")
    _BOOTSTRAP["assert_firewall"]()
    if _loaded_forbidden():
        raise CompatibilityVerifierV2Error("forbidden module appeared during check")
    counters = dict(config["static_counters"])
    report = {
        "schema_version": ("p2_joint_hydrographic_multitask_layer4.r3_compatibility_check.v2"),
        "status": "PASS_AUTHENTICATED_R3_COMPATIBILITY_RESEARCH_ONLY_LOCAL_FAIL",
        "identity": IDENTITY,
        "trusted_implementation_pins": dict(anchor["trusted_files"]),
        "bootstrap_observed_pin": dict(_BOOTSTRAP["bootstrap_observed_pin"]),
        "trust_anchor_pin": dict(_BOOTSTRAP["anchor_pin"]),
        "v1_disposition": disposition,
        "original_r3_verifier_failure": dict(expected_failure),
        "corrected_fold_audit_verification": corrected,
        "result_checks": result_checks,
        "r3_control_inventory": before["r3_control_inventory"],
        "r3_output_inventory": before["r3_output_inventory"],
        "v9": before["v9"],
        "v2_control_exists": False,
        "pre_execution_qa_exists": False,
        "compatibility_receipt_exists": False,
        "sys_dont_write_bytecode": sys.dont_write_bytecode,
        "write_audit_attempts": _BOOTSTRAP["firewall_state"]["mutation_attempts"],
        "forbidden_modules": _loaded_forbidden(),
        "files_written": 0,
        **counters,
    }
    report["summary_sha256"] = hashlib.sha256(_canonical_json_bytes(report)).hexdigest()
    return report


__all__ = [
    "CompatibilityVerifierV2Error",
    "IDENTITY",
    "R3_ENGINE_MODULE",
    "R3_MODULE",
    "TRUSTED_ROLES",
    "verify_static_compatibility",
]
