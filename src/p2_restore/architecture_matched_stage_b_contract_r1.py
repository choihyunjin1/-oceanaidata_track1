"""Fail-closed contract for the P2 Stage-B parser correction r1.

The correction is a new one-shot descendant of the consumed v3 attempt.  It
reuses the exact v3 scientific contract and execution implementation, while
pinning a pure-stdlib CSV repair and a full-source selective-parser preflight.
"""

from __future__ import annotations

import copy
import hashlib
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from p2_restore import architecture_matched_stage_b_contract_v3 as base_guard
from p2_restore.architecture_matched_stage_b_csv_r1 import (
    full_pinned_source_parser_preflight,
    sha256_file,
)

CONFIG_RELATIVE = "configs/experiments/p2_architecture_matched_stage_b_parser_correction_r1.json"
CONFIG_SHA256 = "f958f986f20433564122625d6860fcbfa61a0b48bb73ab1a523710565a3e7109"
MODE = "ARCHITECTURE_MATCHED_TIME_SAFE_BASELINE"
STAGE = "P2_ARCHITECTURE_MATCHED_STAGE_B_PARSER_CORRECTION_R1"
FRACTION_ROLES = dict(base_guard.FRACTION_ROLES)
IMPLEMENTATION_ROLES = frozenset({"CONFIG", "PARSER", "GUARD", "ENGINE", "RUNNER", "TESTS"})

canonical_json_bytes = base_guard.canonical_json_bytes
contained_path = base_guard.contained_path
strict_json_object = base_guard.strict_json_object


class ParserCorrectionContractError(RuntimeError):
    """Raised when the correction lineage or one-shot contract drifts."""


def _canonical_workspace(root: Path) -> Path:
    workspace = root.resolve(strict=True)
    module_workspace = Path(__file__).resolve(strict=True).parents[2]
    if workspace != module_workspace:
        raise PermissionError("parser correction root differs from the canonical module workspace")
    if Path.cwd().resolve(strict=True) != workspace:
        raise PermissionError("parser correction must run from the canonical workspace root")
    if not (workspace / ".git").is_dir():
        raise PermissionError("canonical parser-correction workspace lacks its Git boundary")
    return workspace


def workspace_path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PermissionError("parser-correction path is not workspace relative")
    workspace = root.resolve(strict=True)
    path = (workspace / candidate).resolve(strict=must_exist)
    if path != workspace and workspace not in path.parents:
        raise PermissionError("parser-correction path escaped the canonical workspace")
    return path


def _pin(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _verify_pin(root: Path, expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = workspace_path(root, str(expected.get("path", "")))
    observed = _pin(path, root)
    if observed != dict(expected):
        raise ParserCorrectionContractError(f"{label} pin changed")
    return observed


def _read_config_file(workspace: Path) -> dict[str, Any]:
    path = workspace_path(workspace, CONFIG_RELATIVE)
    if sha256_file(path) != CONFIG_SHA256:
        raise ParserCorrectionContractError("canonical parser-correction config changed")
    return strict_json_object(path)


def _scientific_projection(base: Mapping[str, Any], correction: Mapping[str, Any]) -> dict[str, Any]:
    sections = correction["scientific_contract"]["base_v3_sections"]
    if not isinstance(sections, list) or len(sections) != len(set(sections)):
        raise ParserCorrectionContractError("scientific section registry changed")
    try:
        projected = {str(key): copy.deepcopy(base[str(key)]) for key in sections}
    except KeyError as exc:
        raise ParserCorrectionContractError("base v3 scientific section is missing") from exc
    observed = hashlib.sha256(canonical_json_bytes(projected)).hexdigest()
    if observed != correction["scientific_contract"]["canonical_sha256"]:
        raise ParserCorrectionContractError("base v3 scientific contract changed")
    return projected


def _verify_base_lineage(workspace: Path, correction: Mapping[str, Any]) -> dict[str, Any]:
    base = correction["base_v3"]
    verified = {
        key: _verify_pin(workspace, base[key], label=f"base v3 {key}")
        for key in (
            "config",
            "guard",
            "engine",
            "runner",
            "tests",
            "pre_execution_qa",
            "authorization",
            "attempt_lock",
            "failure_receipt",
            "failure_forensic_addendum",
        )
    }
    empty_output = workspace_path(workspace, base["empty_output"]["path"])
    observed_entries = list(empty_output.rglob("*"))
    if len(observed_entries) != int(base["empty_output"]["required_recursive_entry_count"]):
        raise ParserCorrectionContractError("consumed v3 empty output state changed")
    failure = strict_json_object(workspace_path(workspace, base["failure_receipt"]["path"]))
    addendum = strict_json_object(
        workspace_path(workspace, base["failure_forensic_addendum"]["path"])
    )
    if (
        failure.get("classification") != "INFRASTRUCTURE_FAILURE_NO_EVALUATION"
        or failure.get("disposition", {}).get("existing_attempt_resume_permitted") is not False
        or failure.get("execution_audit", {}).get("challenger_fit_count") != 0
        or failure.get("execution_audit", {}).get("challenger_prediction_count") != 0
        or failure.get("field_access_audit", {}).get(
            "withheld_validation_target_scalar_decode_count"
        )
        != 0
    ):
        raise ParserCorrectionContractError("base v3 failure receipt is not fail-closed")
    if (
        addendum.get("classification")
        != "SECONDARY_PREEXECUTION_INPUT_ROUTING_DEFECT_NO_EVALUATION"
        or addendum.get("discovery", {}).get("actual_v3_attempt_reached_this_defect")
        is not False
        or addendum.get("correction_scope", {}).get(
            "model_feature_fold_seed_prefix_bootstrap_postprocess_gate_changes"
        )
        != 0
    ):
        raise ParserCorrectionContractError("base v3 forensic addendum is not fail-closed")
    base_config = strict_json_object(workspace_path(workspace, base["config"]["path"]))
    _scientific_projection(base_config, correction)
    return {
        "pins": verified,
        "empty_output_recursive_entry_count": len(observed_entries),
        "failure_classification": failure["classification"],
        "failure_addendum_classification": addendum["classification"],
    }


def _validate_correction(correction: Mapping[str, Any]) -> None:
    required_top = {
        "schema_version",
        "experiment_id",
        "created_at_kst",
        "status",
        "problem",
        "comparison_mode",
        "exact_official_incumbent_comparison",
        "research_feedback_observed_from_failed_v3",
        "model_family_evaluated_by_failed_v3",
        "candidate_or_test_prediction_allowed",
        "upload_allowed",
        "base_v3",
        "scientific_contract",
        "correction_contract",
        "parser_preflight",
        "implementation_roles",
        "canonical_paths",
        "execution_policy",
        "qa_receipt_contract",
        "authorization_contract",
    }
    if set(correction) != required_top:
        raise ParserCorrectionContractError("parser-correction config keys changed")
    if (
        correction["schema_version"]
        != "p2_architecture_matched_stage_b.parser_correction.r1"
        or correction["experiment_id"]
        != "p2_architecture_matched_stage_b_parser_correction_r1"
        or correction["problem"] != "P2"
        or correction["comparison_mode"] != MODE
        or correction["exact_official_incumbent_comparison"] is not False
        or correction["research_feedback_observed_from_failed_v3"] is not False
        or correction["model_family_evaluated_by_failed_v3"] is not False
        or correction["candidate_or_test_prediction_allowed"] is not False
        or correction["upload_allowed"] is not False
    ):
        raise ParserCorrectionContractError("parser-correction identity changed")
    if set(correction["implementation_roles"]) != IMPLEMENTATION_ROLES:
        raise ParserCorrectionContractError("parser-correction implementation roles changed")
    expected_paths = {
        "config": CONFIG_RELATIVE,
        "output": "artifacts/p2_architecture_matched_stage_b_parser_correction_r1",
        "control": "artifacts/p2_architecture_matched_stage_b_parser_correction_r1_control",
        "pre_execution_qa": (
            "artifacts/p2_architecture_matched_stage_b_parser_correction_r1_control/"
            "pre_execution_qa.json"
        ),
        "authorization": (
            "artifacts/p2_architecture_matched_stage_b_parser_correction_r1_control/"
            "authorization.json"
        ),
        "attempt_lock": (
            "artifacts/p2_architecture_matched_stage_b_parser_correction_r1_control/attempt.lock"
        ),
        "run_failure_receipt": (
            "artifacts/p2_architecture_matched_stage_b_parser_correction_r1_control/"
            "run_failure_receipt.json"
        ),
    }
    if correction["canonical_paths"] != expected_paths:
        raise ParserCorrectionContractError("parser-correction canonical paths changed")
    correction_contract = correction["correction_contract"]
    if (
        correction_contract.get("unquoted_empty_scalar_result") != ""
        or correction_contract.get("public_layers") != [1, 5, 6, 7, 8]
        or correction_contract.get("target_layers") != [2, 3, 4]
        or correction_contract.get("withheld_target_scalar_decode_convert_use_required") != 0
        or correction_contract.get("base_v3_module_reused_for_all_scientific_computation")
        is not True
        or correction_contract.get("base_v3_decoder_rebound_only_to_pinned_parser_module")
        is not True
    ):
        raise ParserCorrectionContractError("parser-only correction contract changed")
    policy = correction["execution_policy"]
    if (
        policy.get("check_only_is_default") is not True
        or policy.get("full_pinned_source_parser_preflight_before_qa_authorization_or_lock")
        is not True
        or policy.get("actual_run_requires_independent_static_qa") is not True
        or policy.get("actual_run_requires_separate_append_only_authorization") is not True
        or policy.get("qa_and_authorization_verified_before_attempt_lock") is not True
        or policy.get("execution_engine_imported_after_attempt_lock") is not True
        or policy.get("output_and_files_use_o_excl_semantics") is not True
        or policy.get("post_lock_exception_writes_aggregate_only_o_excl_failure_receipt")
        is not True
        or policy.get("rerun_allowed") is not False
        or policy.get("resume_allowed") is not False
        or policy.get("candidate_or_test_prediction_allowed") is not False
        or policy.get("frozen_or_submission_mutation_allowed") is not False
        or policy.get("registry_append_allowed") is not False
        or policy.get("automatic_upload_allowed") is not False
    ):
        raise ParserCorrectionContractError("parser-correction execution policy changed")


def _synthesize_config(workspace: Path, correction: Mapping[str, Any]) -> dict[str, Any]:
    base_path = workspace_path(workspace, correction["base_v3"]["config"]["path"])
    base = strict_json_object(base_path)
    _scientific_projection(base, correction)
    synthesized = copy.deepcopy(base)
    synthesized["schema_version"] = correction["schema_version"]
    synthesized["experiment_id"] = correction["experiment_id"]
    synthesized["created_at_kst"] = correction["created_at_kst"]
    synthesized["status"] = correction["status"]
    synthesized["preregistration"] = copy.deepcopy(base["preregistration"])
    synthesized["preregistration"]["generation_id"] = STAGE
    synthesized["implementation_roles"] = copy.deepcopy(correction["implementation_roles"])
    synthesized["canonical_paths"] = copy.deepcopy(correction["canonical_paths"])
    synthesized["execution_policy"] = copy.deepcopy(correction["execution_policy"])
    synthesized["qa_receipt_contract"] = copy.deepcopy(correction["qa_receipt_contract"])
    synthesized["authorization_contract"] = copy.deepcopy(correction["authorization_contract"])
    synthesized["parser_correction"] = copy.deepcopy(correction["correction_contract"])
    synthesized["base_v3_failure_receipt"] = copy.deepcopy(
        correction["base_v3"]["failure_receipt"]
    )
    synthesized["base_v3_failure_addendum"] = copy.deepcopy(
        correction["base_v3"]["failure_forensic_addendum"]
    )
    return synthesized


def load_canonical_config(
    root: Path,
    requested_config: Path | None = None,
    *,
    supplied_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    workspace = _canonical_workspace(root)
    correction = _read_config_file(workspace)
    _validate_correction(correction)
    if requested_config is not None:
        requested = requested_config.resolve(strict=True)
        if requested != workspace_path(workspace, CONFIG_RELATIVE):
            raise PermissionError("noncanonical parser-correction config was requested")
    synthesized = _synthesize_config(workspace, correction)
    if supplied_config is not None and canonical_json_bytes(dict(supplied_config)) != canonical_json_bytes(
        synthesized
    ):
        raise PermissionError("supplied parser-correction config differs from canonical synthesis")
    return synthesized


def implementation_pins(root: Path) -> dict[str, dict[str, Any]]:
    workspace = _canonical_workspace(root)
    correction = _read_config_file(workspace)
    return {
        role: _pin(workspace_path(workspace, relative), workspace)
        for role, relative in correction["implementation_roles"].items()
    }


def stage_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    workspace = _canonical_workspace(root)
    canonical = config["canonical_paths"]
    return {
        key: workspace_path(workspace, canonical[key], must_exist=False)
        for key in (
            "output",
            "control",
            "pre_execution_qa",
            "authorization",
            "attempt_lock",
            "run_failure_receipt",
        )
    }


def verify_stage_a_reference(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    workspace = _canonical_workspace(root)
    correction = _read_config_file(workspace)
    base_config = strict_json_object(
        workspace_path(workspace, correction["base_v3"]["config"]["path"])
    )
    if config["stage_a_reference"] != base_config["stage_a_reference"]:
        raise ParserCorrectionContractError("Stage-A reference differs from base v3")
    return base_guard.verify_stage_a_reference(workspace, base_config)


def static_preflight(
    root: Path,
    data_dir: Path,
    *,
    requested_config: Path | None = None,
    supplied_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    workspace = _canonical_workspace(root)
    correction = _read_config_file(workspace)
    canonical = load_canonical_config(
        workspace,
        requested_config,
        supplied_config=supplied_config,
    )
    lineage = _verify_base_lineage(workspace, correction)
    pins = implementation_pins(workspace)
    base_report = base_guard.static_preflight(workspace, data_dir)
    parser_spec = correction["parser_preflight"]
    parser_report = full_pinned_source_parser_preflight(
        data_dir,
        source_sha256=parser_spec["observations_sha256"],
        source_bytes=int(parser_spec["observations_bytes"]),
        expected_rows=int(parser_spec["observations_rows"]),
        outer_folds=canonical["curve_protocol"]["outer_folds"],
        embargo_days=int(canonical["curve_protocol"]["embargo_days"]),
    )
    parser_sha = hashlib.sha256(canonical_json_bytes(parser_report)).hexdigest()
    paths = stage_paths(workspace, canonical)
    return {
        "schema_version": "p2_architecture_matched_stage_b.parser_correction.static_preflight.r1",
        "status": "PASS_STATIC_IMPLEMENTATION_ONLY_STAGE_A_SEALED_PARSER_CORRECTED",
        "problem": "P2",
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "canonical_config": _pin(workspace_path(workspace, CONFIG_RELATIVE), workspace),
        "implementation_pins": pins,
        "source_pins": base_report["source_pins"],
        "stage_a_reference": base_report["stage_a_reference"],
        "schema_and_keys": base_report["schema_and_keys"],
        "runtime_probe": base_report["runtime_probe"],
        "base_v3_lineage": lineage,
        "scientific_contract_sha256": correction["scientific_contract"]["canonical_sha256"],
        "scientific_contract_changes": 0,
        "parser_preflight": parser_report,
        "parser_preflight_sha256": parser_sha,
        "canonical_path_state": {key: path.exists() for key, path in paths.items()},
        "preflight_process_new_numerical_modules": base_report[
            "preflight_process_new_numerical_modules"
        ],
        "files_written": 0,
        "attempt_locks_created": 0,
        "challenger_fits": 0,
        "challenger_predictions": 0,
        "scores": 0,
        "test_predictions": 0,
        "uploads": 0,
        "resource_estimate": canonical["resource_estimate"],
    }


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o600)
    try:
        payload = canonical_json_bytes(dict(value)) + b"\n"
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def verify_pre_execution_qa(
    root: Path,
    config: Mapping[str, Any],
    *,
    parser_preflight_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    paths = stage_paths(root, config)
    if not paths["pre_execution_qa"].is_file():
        raise PermissionError("independent parser-correction QA receipt is missing")
    receipt = strict_json_object(paths["pre_execution_qa"])
    expected_keys = {
        "schema_version",
        "created_at_kst",
        "reviewer",
        "decision",
        "p0_count",
        "p1_count",
        "config",
        "implementation_pins",
        "base_v3_failure_receipt",
        "base_v3_failure_addendum",
        "parser_preflight_sha256",
        "notes",
    }
    contract = config["qa_receipt_contract"]
    checks = {
        "keys": set(receipt) == expected_keys,
        "schema": receipt.get("schema_version") == contract["schema_version"],
        "reviewer": bool(receipt.get("reviewer")),
        "decision": receipt.get("decision") == contract["decision"],
        "p0": receipt.get("p0_count") == 0,
        "p1": receipt.get("p1_count") == 0,
        "config": receipt.get("config")
        == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "implementation": receipt.get("implementation_pins") == implementation_pins(root),
        "failure": receipt.get("base_v3_failure_receipt") == config["base_v3_failure_receipt"],
        "failure_addendum": receipt.get("base_v3_failure_addendum")
        == config["base_v3_failure_addendum"],
        "parser_sha": isinstance(receipt.get("parser_preflight_sha256"), str)
        and len(receipt["parser_preflight_sha256"]) == 64,
        "notes": isinstance(receipt.get("notes"), list) and bool(receipt["notes"]),
    }
    if parser_preflight_sha256 is not None:
        checks["parser_sha_exact"] = (
            receipt.get("parser_preflight_sha256") == parser_preflight_sha256
        )
    if not all(checks.values()):
        raise PermissionError(
            "parser-correction QA failed: "
            f"{sorted(key for key, value in checks.items() if not value)}"
        )
    return receipt, sha256_file(paths["pre_execution_qa"])


def verify_execution_authorization(
    root: Path,
    config: Mapping[str, Any],
    *,
    qa_sha256: str,
    require_unconsumed: bool = True,
    require_output_absent: bool = True,
) -> tuple[dict[str, Any], str]:
    paths = stage_paths(root, config)
    if require_output_absent and paths["output"].exists():
        raise FileExistsError("append-only parser-correction output already exists")
    if require_unconsumed and paths["attempt_lock"].exists():
        raise FileExistsError("one-shot parser-correction attempt was already consumed")
    if require_unconsumed and paths["run_failure_receipt"].exists():
        raise FileExistsError("parser-correction attempt already has a failure receipt")
    if not paths["authorization"].is_file():
        raise PermissionError("separate parser-correction authorization is missing")
    authorization = strict_json_object(paths["authorization"])
    expected_keys = {
        "schema_version",
        "created_at_kst",
        "stage",
        "config",
        "authorization",
        "user_message_reference",
        "qa_receipt",
        "base_v3_failure_receipt",
        "base_v3_failure_addendum",
        "implementation_pins",
        "execution_authorized",
        "candidate_or_test_prediction_allowed",
        "upload_allowed",
    }
    contract = config["authorization_contract"]
    checks = {
        "keys": set(authorization) == expected_keys,
        "schema": authorization.get("schema_version") == contract["schema_version"],
        "stage": authorization.get("stage") == STAGE,
        "config": authorization.get("config")
        == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "phrase": authorization.get("authorization")
        == contract["authorization_phrase_prefix"] + CONFIG_SHA256,
        "user": bool(authorization.get("user_message_reference")),
        "qa": authorization.get("qa_receipt")
        == {"path": config["canonical_paths"]["pre_execution_qa"], "sha256": qa_sha256},
        "failure": authorization.get("base_v3_failure_receipt")
        == config["base_v3_failure_receipt"],
        "failure_addendum": authorization.get("base_v3_failure_addendum")
        == config["base_v3_failure_addendum"],
        "implementation": authorization.get("implementation_pins") == implementation_pins(root),
        "execution": authorization.get("execution_authorized") is True,
        "candidate": authorization.get("candidate_or_test_prediction_allowed") is False,
        "upload": authorization.get("upload_allowed") is False,
    }
    if not all(checks.values()):
        raise PermissionError(
            "parser-correction authorization failed: "
            f"{sorted(key for key, value in checks.items() if not value)}"
        )
    return authorization, sha256_file(paths["authorization"])


def _lock_payload(
    root: Path,
    config: Mapping[str, Any],
    *,
    qa_sha256: str,
    authorization_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "p2_architecture_matched_stage_b.parser_correction.attempt_lock.r1",
        "stage": STAGE,
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "qa_receipt_sha256": qa_sha256,
        "authorization_sha256": authorization_sha256,
        "base_v3_failure_receipt": config["base_v3_failure_receipt"],
        "base_v3_failure_addendum": config["base_v3_failure_addendum"],
        "implementation_pins": implementation_pins(root),
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "candidate_or_test_prediction_allowed": False,
        "rerun_allowed": False,
        "resume_allowed": False,
        "upload_allowed": False,
    }


def consume_attempt_lock(
    root: Path,
    config: Mapping[str, Any],
    *,
    qa_sha256: str,
    authorization_sha256: str,
) -> Path:
    paths = stage_paths(root, config)
    if paths["output"].exists():
        raise FileExistsError("append-only parser-correction output already exists")
    if paths["run_failure_receipt"].exists():
        raise FileExistsError("parser-correction failure receipt already exists")
    exclusive_json(
        paths["attempt_lock"],
        _lock_payload(
            root,
            config,
            qa_sha256=qa_sha256,
            authorization_sha256=authorization_sha256,
        ),
    )
    return paths["attempt_lock"]


def verify_consumed_attempt_lock(
    root: Path,
    config: Mapping[str, Any],
    *,
    qa_sha256: str,
    authorization_sha256: str,
) -> dict[str, Any]:
    path = stage_paths(root, config)["attempt_lock"]
    if not path.is_file():
        raise PermissionError("one-shot parser-correction attempt lock is missing")
    observed = strict_json_object(path)
    expected = _lock_payload(
        root,
        config,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
    )
    if observed != expected:
        raise PermissionError("parser-correction attempt lock fails full deep equality")
    return observed


def write_run_failure_receipt(
    root: Path,
    config: Mapping[str, Any],
    *,
    exception: BaseException,
) -> Path:
    """Persist a sanitized aggregate failure receipt after lock consumption."""

    workspace = _canonical_workspace(root)
    paths = stage_paths(workspace, config)
    lock = paths["attempt_lock"].resolve(strict=True)
    output_entries: list[str] = []
    if paths["output"].is_dir():
        output_entries = sorted(
            path.relative_to(paths["output"]).as_posix()
            for path in paths["output"].rglob("*")
        )
    message_digest = hashlib.sha256(str(exception).encode("utf-8")).hexdigest()
    receipt = {
        "schema_version": "p2_architecture_matched_stage_b.parser_correction.run_failure.r1",
        "stage": STAGE,
        "classification": "POST_LOCK_INFRASTRUCTURE_FAILURE_NO_AUTOMATIC_RETRY",
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "implementation_pins": implementation_pins(workspace),
        "attempt_lock": _pin(lock, workspace),
        "base_v3_failure_receipt": config["base_v3_failure_receipt"],
        "base_v3_failure_addendum": config["base_v3_failure_addendum"],
        "exception_type": type(exception).__name__,
        "exception_message_sha256": message_digest,
        "raw_exception_message_persisted": False,
        "output_directory_exists": paths["output"].is_dir(),
        "output_recursive_entry_count": len(output_entries),
        "output_registered_relative_entries": output_entries,
        "candidate_or_test_prediction_allowed": False,
        "upload_allowed": False,
        "rerun_allowed": False,
        "resume_allowed": False,
        "uploads": 0,
    }
    exclusive_json(paths["run_failure_receipt"], receipt)
    return paths["run_failure_receipt"]


@contextmanager
def _bound_base_guard() -> Iterator[None]:
    names = {
        "CONFIG_RELATIVE": CONFIG_RELATIVE,
        "CONFIG_SHA256": CONFIG_SHA256,
        "implementation_pins": implementation_pins,
        "stage_paths": stage_paths,
    }
    previous = {name: getattr(base_guard, name) for name in names}
    try:
        for name, value in names.items():
            setattr(base_guard, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(base_guard, name, value)


def verify_stage_b_seal(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    with _bound_base_guard():
        result = base_guard.verify_stage_b_seal(root, config)
    return {
        **result,
        "status": "PASS_SEALED_STAGE_B_PARSER_CORRECTION_R1",
        "base_v3_failure_receipt": config["base_v3_failure_receipt"],
        "base_v3_failure_addendum": config["base_v3_failure_addendum"],
    }


__all__ = [
    "CONFIG_RELATIVE",
    "CONFIG_SHA256",
    "FRACTION_ROLES",
    "MODE",
    "STAGE",
    "canonical_json_bytes",
    "consume_attempt_lock",
    "contained_path",
    "exclusive_json",
    "implementation_pins",
    "load_canonical_config",
    "sha256_file",
    "stage_paths",
    "static_preflight",
    "strict_json_object",
    "verify_consumed_attempt_lock",
    "verify_execution_authorization",
    "verify_pre_execution_qa",
    "verify_stage_a_reference",
    "verify_stage_b_seal",
    "write_run_failure_receipt",
]
