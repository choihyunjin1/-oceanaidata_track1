"""Read-only compatibility verification for the immutable P2 Layer-4 r3 attempt."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

R3_MODULE = "p2_restore.joint_hydrographic_multitask_layer4_contract_r3"
R3_RELATIVE = "joint_hydrographic_multitask_layer4_contract_r3.py"
R3_GUARD_SHA256 = "aa553956438a25d15c8f816c67c21467dc9b6d316f676f9b6f30d98df0505573"
R3_GUARD_BYTES = 142834


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_r3_guard() -> Any:
    path = Path(__file__).resolve(strict=True).with_name(R3_RELATIVE)
    guard_bytes = path.read_bytes()
    if len(guard_bytes) != R3_GUARD_BYTES or _sha256_bytes(guard_bytes) != R3_GUARD_SHA256:
        raise RuntimeError("frozen r3 guard pin changed")
    existing = sys.modules.get(R3_MODULE)
    if existing is not None:
        if Path(existing.__file__).resolve(strict=True) != path:
            raise RuntimeError("noncanonical r3 guard is already imported")
        return existing
    spec = importlib.util.spec_from_file_location(R3_MODULE, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen r3 guard")
    module = importlib.util.module_from_spec(spec)
    sys.modules[R3_MODULE] = module
    spec.loader.exec_module(module)
    return module


r3 = _load_r3_guard()

CONFIG_RELATIVE = (
    "configs/experiments/p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v1.json"
)
CONFIG_SHA256 = "b1e30c04801bec2a575ed1cefbf6afd913da17147415176bc432f086b9e87491"
IDENTITY = "P2_JOINT_HYDROGRAPHIC_MULTITASK_LAYER4_R3_COMPATIBILITY_VERIFIER_V1"
IMPLEMENTATION_ROLES = {
    "CONFIG": CONFIG_RELATIVE,
    "HELPER": (
        "src/p2_restore/joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v1.py"
    ),
    "CLI": "scripts/verify_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v1.py",
    "TESTS": ("tests/test_p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v1.py"),
}
NUMERICAL_PREFIXES = ("numpy", "pandas", "scipy", "sklearn", "torch")


class CompatibilityVerifierError(ValueError):
    """The frozen compatibility-verification contract was not satisfied."""


def _is_linklike(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _loaded_numerical_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in NUMERICAL_PREFIXES)
    )


def _path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    workspace = root.resolve(strict=True)
    requested = Path(relative)
    if requested.is_absolute() or ".." in requested.parts:
        raise CompatibilityVerifierError(f"path is not a contained relative path: {relative}")
    lexical = workspace / requested
    current = workspace
    for part in requested.parts:
        if part in ("", "."):
            continue
        current /= part
        if _is_linklike(current):
            raise CompatibilityVerifierError(f"path contains link-like entry: {relative}")
        if not current.exists():
            break
    candidate = lexical.resolve(strict=must_exist)
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise CompatibilityVerifierError(f"path escapes workspace: {relative}") from exc
    return candidate


def _pin(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": r3.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise CompatibilityVerifierError(f"{label} keys changed")


def _strict_json_bytes(value: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise CompatibilityVerifierError(f"duplicate JSON key in {label}: {key}")
            result[key] = item
        return result

    def finite_only(item: str) -> Any:
        raise CompatibilityVerifierError(f"non-finite JSON number in {label}: {item}")

    try:
        parsed = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=finite_only,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompatibilityVerifierError(f"invalid JSON in {label}") from exc
    if not isinstance(parsed, dict):
        raise CompatibilityVerifierError(f"JSON root is not an object in {label}")
    return parsed


def _strict_pinned_json(
    root: Path,
    pin: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if set(pin) != {"path", "sha256", "bytes"}:
        raise CompatibilityVerifierError(f"{label} pin schema changed")
    workspace = root.resolve(strict=True)
    path = _path(workspace, str(pin["path"]))
    value = path.read_bytes()
    observed = {
        "path": path.relative_to(workspace).as_posix(),
        "sha256": _sha256_bytes(value),
        "bytes": len(value),
    }
    if observed != dict(pin):
        raise CompatibilityVerifierError(f"{label} pin drift during parse")
    parsed = _strict_json_bytes(value, label=label)
    if _pin(path, workspace) != observed:
        raise CompatibilityVerifierError(f"{label} changed during parse")
    return parsed


def load_config(
    root: Path,
    requested_path: Path | None = None,
    *,
    supplied_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    workspace = root.resolve(strict=True)
    canonical = _path(workspace, CONFIG_RELATIVE)
    requested = (requested_path or canonical).resolve(strict=True)
    if requested != canonical or r3.sha256_file(canonical) != CONFIG_SHA256:
        raise CompatibilityVerifierError("canonical compatibility config changed")
    config = r3.strict_json_object(canonical)
    _exact_keys(
        config,
        {
            "schema_version",
            "created_at_kst",
            "status",
            "problem",
            "identity",
            "comparison_mode",
            "verifier_only",
            "check_only_default",
            "r3_mutation_allowed",
            "r3_rerun_or_resume_allowed",
            "execution_authorization_or_lock_allowed",
            "fit_prediction_truth_decode_or_scoring_allowed",
            "official_promotion_allowed",
            "candidate_or_test_prediction_allowed",
            "upload_allowed",
            "implementation_roles",
            "canonical_paths",
            "r3_implementation_pins",
            "r3_control_pins",
            "r3_control_inventory",
            "r3_output_inventory",
            "r3_core_artifact_pins",
            "original_verifier_expected_failure",
            "fold_audit_compatibility_contract",
            "expected_result",
            "v9_anchor",
        },
        label="compatibility config",
    )
    if (
        config.get("schema_version")
        != "p2_joint_hydrographic_multitask_layer4.r3_compatibility_verifier.v1"
        or config.get("identity") != IDENTITY
        or config.get("problem") != "P2"
        or config.get("comparison_mode") != r3.MODE
        or config.get("implementation_roles") != IMPLEMENTATION_ROLES
    ):
        raise CompatibilityVerifierError("compatibility identity changed")
    true_flags = ("verifier_only", "check_only_default")
    false_flags = (
        "r3_mutation_allowed",
        "r3_rerun_or_resume_allowed",
        "execution_authorization_or_lock_allowed",
        "fit_prediction_truth_decode_or_scoring_allowed",
        "official_promotion_allowed",
        "candidate_or_test_prediction_allowed",
        "upload_allowed",
    )
    if any(config.get(key) is not True for key in true_flags) or any(
        config.get(key) is not False for key in false_flags
    ):
        raise CompatibilityVerifierError("read-only firewall changed")
    if supplied_config is not None and dict(supplied_config) != config:
        raise CompatibilityVerifierError("supplied compatibility config differs")
    return config


def implementation_pins(root: Path) -> dict[str, dict[str, Any]]:
    workspace = root.resolve(strict=True)
    return {
        role: _pin(_path(workspace, relative), workspace)
        for role, relative in IMPLEMENTATION_ROLES.items()
    }


def verify_pin_map(
    root: Path,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    workspace = root.resolve(strict=True)
    observed: dict[str, dict[str, Any]] = {}
    for role, pin in expected.items():
        if not isinstance(pin, Mapping) or set(pin) != {"path", "sha256", "bytes"}:
            raise CompatibilityVerifierError(f"{label} pin schema changed: {role}")
        current = _pin(_path(workspace, str(pin["path"])), workspace)
        if current != dict(pin):
            raise CompatibilityVerifierError(f"{label} pin drift: {role}")
        observed[str(role)] = current
    return observed


def frozen_inventory(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if _is_linklike(path):
        raise CompatibilityVerifierError("frozen inventory root is link-like")
    entries: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        if _is_linklike(item):
            raise CompatibilityVerifierError(f"frozen inventory contains link-like entry: {item}")
        try:
            item.resolve(strict=True).relative_to(resolved)
        except ValueError as exc:
            raise CompatibilityVerifierError("frozen inventory escapes its root") from exc
        relative = item.relative_to(path).as_posix()
        if item.is_dir():
            entries.append({"path": relative, "type": "directory"})
        elif item.is_file():
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "bytes": item.stat().st_size,
                    "sha256": r3.sha256_file(item),
                }
            )
        else:
            raise CompatibilityVerifierError("frozen inventory contains a special entry")
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
    files = [entry for entry in entries if entry["type"] == "file"]
    directories = [entry for entry in entries if entry["type"] == "directory"]
    return {
        "directories": len(directories),
        "files": len(files),
        "file_bytes": sum(int(entry["bytes"]) for entry in files),
        "algorithm": "SHA256_CANONICAL_JSON_SORTED_RELATIVE_PATH_TYPE_BYTES_SHA256_WITH_LF",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _verify_fold_audit_compatibility(
    fold_audits: Mapping[str, Any],
    fold_commitment_audits: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    order = list(contract["canonical_fold_order"])
    expected_priors = contract["verified_prior_fold_commitments_by_canonical_fold"]
    forbidden = list(contract["forbidden_decode_fields"])
    if set(fold_audits) != set(order) or set(fold_commitment_audits) != set(order):
        raise CompatibilityVerifierError("fold audit key set changed")
    verified: list[dict[str, Any]] = []
    for index, name in enumerate(order):
        audit = fold_audits[name]
        committed = fold_commitment_audits[name]
        if not isinstance(audit, Mapping) or dict(audit) != dict(committed):
            raise CompatibilityVerifierError(f"fold audit differs from commitment: {name}")
        prior = audit.get("verified_prior_fold_commitments")
        if (
            audit.get("fold") != name
            or expected_priors.get(name) != index
            or not isinstance(prior, int)
            or isinstance(prior, bool)
            or prior != index
        ):
            raise CompatibilityVerifierError(f"fold audit identity/order changed: {name}")
        for field in forbidden:
            value = audit.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value != 0:
                raise CompatibilityVerifierError(
                    f"forbidden fold audit decode changed: {name}:{field}"
                )
        verified.append({"fold": name, "verified_prior_fold_commitments": prior})
    return {
        "canonical_fold_order": order,
        "verified": verified,
        "mapping_insertion_order_ignored": True,
        "exact_fold_commitment_audit_equality": True,
    }


def _verify_v9(root: Path, anchor: Mapping[str, Any]) -> dict[str, Any]:
    workspace = root.resolve(strict=True)
    pin = {key: anchor[key] for key in ("path", "sha256", "bytes")}
    path = _path(workspace, str(pin["path"]))
    ledger_bytes = path.read_bytes()
    observed = {
        "path": path.relative_to(workspace).as_posix(),
        "sha256": _sha256_bytes(ledger_bytes),
        "bytes": len(ledger_bytes),
    }
    if observed != pin:
        raise CompatibilityVerifierError("v9 ledger pin changed")
    records = []
    for index, line in enumerate(ledger_bytes.splitlines(), start=1):
        if line.strip():
            records.append(_strict_json_bytes(line, label=f"v9 ledger line {index}"))
    if _pin(path, workspace) != observed:
        raise CompatibilityVerifierError("v9 ledger changed during parse")
    sequences = [record.get("seq") for record in records]
    if len(records) != anchor["record_count"] or sequences != anchor["sequences"]:
        raise CompatibilityVerifierError("v9 sequence changed")
    for index, record in enumerate(records):
        claimed = record.get("event_sha256")
        base = {key: value for key, value in record.items() if key != "event_sha256"}
        if hashlib.sha256(r3.canonical_json_bytes(base)).hexdigest() != claimed:
            raise CompatibilityVerifierError("v9 event hash changed")
        if index and record.get("previous_event_sha256") != records[index - 1]["event_sha256"]:
            raise CompatibilityVerifierError("v9 event chain changed")
    head = records[-1]
    uploads = sum(
        int(bool(record.get("payload", {}).get("upload_performed", False))) for record in records
    )
    if (
        head.get("seq") != anchor["head_sequence"]
        or head.get("event_sha256") != anchor["head_event_sha256"]
        or uploads != anchor["uploads"]
    ):
        raise CompatibilityVerifierError("v9 head or upload state changed")
    return {
        "pin": observed,
        "record_count": len(records),
        "sequences": sequences,
        "head_sequence": head["seq"],
        "head_event_type": head["event_type"],
        "head_event_sha256": head["event_sha256"],
        "uploads": uploads,
    }


def verify_static_compatibility(
    root: Path,
    *,
    requested_config: Path | None = None,
    supplied_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the one frozen r3 attempt without writing or evaluating metrics."""

    workspace = root.resolve(strict=True)
    before = _loaded_numerical_modules()
    config = load_config(
        workspace,
        requested_config,
        supplied_config=supplied_config,
    )
    implementation = verify_pin_map(
        workspace, config["r3_implementation_pins"], label="r3 implementation"
    )
    controls = verify_pin_map(workspace, config["r3_control_pins"], label="r3 control")
    core = verify_pin_map(workspace, config["r3_core_artifact_pins"], label="r3 core artifact")
    control_inventory = frozen_inventory(_path(workspace, config["canonical_paths"]["r3_control"]))
    output_inventory = frozen_inventory(_path(workspace, config["canonical_paths"]["r3_output"]))
    if control_inventory != config["r3_control_inventory"]:
        raise CompatibilityVerifierError("r3 control inventory changed")
    if output_inventory != config["r3_output_inventory"]:
        raise CompatibilityVerifierError("r3 output inventory changed")

    r3_config = r3.load_canonical_config(workspace)
    expected_failure = config["original_verifier_expected_failure"]
    try:
        r3.verify_seal(workspace, r3_config)
    except r3.Layer4ContractError as exc:
        if (
            type(exc).__name__ != expected_failure["exception_type"]
            or str(exc) != expected_failure["message"]
        ):
            raise CompatibilityVerifierError(
                "original r3 verifier has a different failure"
            ) from exc
    else:
        raise CompatibilityVerifierError(
            "original r3 verifier no longer has the pinned false-negative"
        )

    receipt = _strict_pinned_json(
        workspace,
        config["r3_core_artifact_pins"]["TRAINING_RECEIPT"],
        label="r3 training receipt",
    )
    persisted_order = list(receipt.get("fold_blind_input_audits", {}))
    contract = config["fold_audit_compatibility_contract"]
    if persisted_order != contract["persisted_mapping_order"]:
        raise CompatibilityVerifierError("pinned persisted fold-audit order changed")
    commitment_audits: dict[str, Any] = {}
    for role, name in (
        ("FOLD_SEP_OCT", "outer_2024_sep_oct"),
        ("FOLD_MAY_JUN", "outer_2025_may_jun"),
        ("FOLD_JUL_AUG", "outer_2025_jul_aug"),
    ):
        payload = _strict_pinned_json(
            workspace,
            config["r3_core_artifact_pins"][role],
            label=f"r3 fold commitment {name}",
        )
        commitment_audits[name] = payload.get("blind_input_audit")
    corrected = _verify_fold_audit_compatibility(
        receipt["fold_blind_input_audits"],
        commitment_audits,
        contract,
    )

    decision = _strict_pinned_json(
        workspace,
        config["r3_core_artifact_pins"]["DECISION"],
        label="r3 decision",
    )
    evidence = _strict_pinned_json(
        workspace,
        config["r3_core_artifact_pins"]["EVIDENCE"],
        label="r3 evidence",
    )
    seal = _strict_pinned_json(
        workspace,
        config["r3_core_artifact_pins"]["SEAL"],
        label="r3 seal",
    )
    expected = config["expected_result"]
    result_checks = {
        "status": decision.get("status") == seal.get("status") == expected["status"],
        "local": decision.get("local_qualification") is expected["local_qualification"]
        and evidence.get("local_qualification") is expected["local_qualification"]
        and seal.get("local_qualification") is expected["local_qualification"],
        "passed": decision.get("passed") is expected["passed"],
        "promotion": decision.get("official_promotion") is expected["official_promotion"]
        and decision.get("official_promotion_allowed") is False
        and seal.get("official_promotion_allowed") is False,
        "candidate": decision.get("candidate_generated") is expected["candidate_generated"]
        and seal.get("candidate_generated") is expected["candidate_generated"],
        "test": decision.get("test_prediction_generated") is expected["test_prediction_generated"]
        and seal.get("test_prediction_generated") is expected["test_prediction_generated"],
        "upload": decision.get("upload_performed") is False
        and seal.get("upload_count") == expected["uploads"],
        "finite_metrics": all(
            math.isfinite(float(point[key]))
            for point in evidence.get("points", [])
            for key in ("incumbent", "challenger", "delta")
        ),
    }
    if not all(result_checks.values()):
        raise CompatibilityVerifierError(
            f"r3 research-only result changed: {sorted(k for k, v in result_checks.items() if not v)}"
        )
    v9 = _verify_v9(workspace, config["v9_anchor"])
    after = _loaded_numerical_modules()
    compatibility_control = _path(
        workspace,
        config["canonical_paths"]["compatibility_control"],
        must_exist=False,
    )
    report = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.r3_compatibility_check.v1",
        "status": "PASS_R3_COMPATIBILITY_VERIFIER_ONLY_RESEARCH_RESULT_LOCAL_FAIL",
        "identity": IDENTITY,
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "original_r3_verifier_failure": dict(expected_failure),
        "corrected_fold_audit_verification": corrected,
        "r3_implementation_pins": implementation,
        "r3_control_pins": controls,
        "r3_core_artifact_pins": core,
        "r3_control_inventory": control_inventory,
        "r3_output_inventory": output_inventory,
        "result_checks": result_checks,
        "v9": v9,
        "compatibility_control_exists": compatibility_control.exists(),
        "compatibility_qa_receipt_exists": _path(
            workspace, config["canonical_paths"]["pre_execution_qa"], must_exist=False
        ).exists(),
        "compatibility_receipt_exists": _path(
            workspace, config["canonical_paths"]["compatibility_receipt"], must_exist=False
        ).exists(),
        "new_numerical_modules": sorted(set(after) - set(before)),
        "files_written": 0,
        "execution_authorizations_created": 0,
        "attempt_locks_created": 0,
        "model_fits": 0,
        "predictions": 0,
        "truth_scalar_decodes": 0,
        "scores_computed": 0,
        "candidate_predictions": 0,
        "test_predictions": 0,
        "uploads": 0,
    }
    report["summary_sha256"] = _sha256_bytes(r3.canonical_json_bytes(report))
    return report


__all__ = [
    "CONFIG_RELATIVE",
    "CONFIG_SHA256",
    "CompatibilityVerifierError",
    "IDENTITY",
    "IMPLEMENTATION_ROLES",
    "frozen_inventory",
    "implementation_pins",
    "load_config",
    "verify_pin_map",
    "verify_static_compatibility",
]
