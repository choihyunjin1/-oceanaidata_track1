"""Held-buffer compatibility-v4 wrapper for the frozen P3 Gen6r2 result.

This source is never imported by ``PathFinder``.  An externally pinned launcher
authenticates the isolated interpreter startup and the v4 bootstrap; that
bootstrap authenticates these exact bytes, compiles them in memory, and
injects an opaque, single-use capability.  All legacy semantic reads and the
two historical compatibility adaptations are owned by the stable-buffer layer.
"""

from __future__ import annotations

if "__trusted_v4_context__" not in globals() or "__trusted_v4_token__" not in globals():
    raise RuntimeError("compatibility-v4 helper requires the trusted v4 bootstrap")

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

_CONTEXT = globals()["__trusted_v4_context__"]
_TOKEN = globals()["__trusted_v4_token__"]
if _CONTEXT.get("token") is not _TOKEN:
    raise RuntimeError("compatibility-v4 opaque bootstrap token identity differs")

IDENTITY: Final = "P3_GEN6_INCUMBENT_PRESERVING_RESIDUAL_CALIBRATOR_R2_COMPATIBILITY_VERIFIER_V4"
CONFIG_RELATIVE: Final = (
    "configs/experiments/p3_gen6_incumbent_preserving_residual_calibrator_"
    "v1r2_compatibility_verifier_v4.json"
)
BOOTSTRAP_RELATIVE: Final = (
    "scripts/bootstrap_verify_p3_gen6_incumbent_preserving_residual_calibrator_"
    "r2_compatibility_v4.py"
)
ENGINE_MODULE: Final = "p3_wave.gen6_incumbent_preserving_residual_calibrator_execution_r2"
EXPECTED_ADAPTATIONS: Final = (
    "replace_frozen_r2_verifier_prefix_expectation_with_four_source_consensus",
    "admit_exact_pinned_historical_failure_receipt_to_the_r2_control_inventory",
)
EXPECTED_FINDINGS: Final = (
    "PRE_SCRIPT_ENCODING_BYTECODE_TRUST_NOT_CLOSED",
    "STDLIB_NATIVE_AUTH_TO_LOAD_TOCTOU",
    "SEMANTIC_BYTES_NOT_ALL_PARSED_FROM_HELD_BUFFERS",
    "HARDLINK_CONTAINMENT_NOT_ENFORCED",
    "DIRECT_WINAPI_AND_NETWORK_FIREWALL_INCOMPLETE",
)
FALSE_FLAGS: Final = (
    "r2_mutation_allowed",
    "r2_rerun_or_resume_allowed",
    "qa_or_compatibility_receipt_write_allowed",
    "execution_authorization_or_attempt_lock_allowed",
    "fit_prediction_or_new_score_allowed",
    "official_promotion_allowed",
    "candidate_or_test_prediction_allowed",
    "registry_append_allowed",
    "upload_allowed",
)
_CONFIG_PARSE_COUNT = 0


class CompatibilityV4Error(RuntimeError):
    """The authenticated held-buffer v4 contract was not satisfied."""


def _duplicates_forbidden(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CompatibilityV4Error("duplicate JSON object key is forbidden")
        value[key] = item
    return value


def _reject_constant(value: str) -> Any:
    raise CompatibilityV4Error(f"non-finite JSON constant is forbidden: {value}")


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_duplicates_forbidden,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompatibilityV4Error(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CompatibilityV4Error(f"{label} must be a JSON object")
    return value


def _parse_config_once(requested_config: Path | None) -> dict[str, Any]:
    global _CONFIG_PARSE_COUNT
    if _CONFIG_PARSE_COUNT != 0:
        raise CompatibilityV4Error("v4 config may be parsed exactly once")
    canonical = _CONTEXT["canonical_path"](CONFIG_RELATIVE, "workspace")
    if requested_config is not None:
        requested = Path(requested_config)
        if not requested.is_absolute():
            requested = _CONTEXT["workspace"] / requested
        if not _CONTEXT["same_path"](requested, canonical):
            raise CompatibilityV4Error("alternate compatibility-v4 config is forbidden")
    raw = _CONTEXT["source_buffer"]("V4_CONFIG")
    pin = _CONTEXT["source_pin"]("V4_CONFIG")
    if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise CompatibilityV4Error("v4 config authenticated-buffer identity differs")
    config = _strict_json(raw, label="v4 config")
    _CONFIG_PARSE_COUNT += 1
    _validate_config(config)
    _CONTEXT["reverify"]("v4_helper_post_config_parse")
    return config


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version")
        != "p3_gen6_incumbent_preserving_residual_calibrator.r2_compatibility_verifier.v4"
        or config.get("problem") != "P3"
        or config.get("identity") != IDENTITY
        or config.get("verifier_only") is not True
        or config.get("check_only_default") is not True
        or config.get("append_only_successor_of_v3") is not True
    ):
        raise CompatibilityV4Error("v4 identity or read-only mode changed")
    if any(config.get(flag) is not False for flag in FALSE_FLAGS):
        raise CompatibilityV4Error("v4 read-only firewall changed")
    counters = config.get("static_counters")
    if not isinstance(counters, Mapping) or not counters or any(counters.values()):
        raise CompatibilityV4Error("v4 static counters changed")
    if config.get("implementation_roles") != _CONTEXT["implementation_roles"]:
        raise CompatibilityV4Error("v4 implementation roles differ from trust root")
    if config.get("authenticated_subordinate_pins") != _CONTEXT["subordinate_pins"]:
        raise CompatibilityV4Error("v4 subordinate pins differ from trust root")
    disposition = config.get("v3_disposition")
    if (
        not isinstance(disposition, Mapping)
        or disposition.get("reviewer") != "/root/p3_gen6_compat_v3_qa"
        or disposition.get("verdict") != "P0=0_P1=5_INDEPENDENT_NO_GO"
        or disposition.get("qa_receipt_present") is not False
        or disposition.get("independent_qa_receipt_sha256") is not None
        or tuple(disposition.get("finding_ids", ())) != EXPECTED_FINDINGS
    ):
        raise CompatibilityV4Error("v3 NO-GO lineage changed")
    compatibility = config.get("compatibility_contract")
    if (
        not isinstance(compatibility, Mapping)
        or tuple(compatibility.get("only_scoped_science_adaptations", ())) != EXPECTED_ADAPTATIONS
        or compatibility.get("adaptation_count") != 2
        or compatibility.get("all_other_v1_and_r2_checks_reexecuted") is not True
        or compatibility.get("independent_oof_bootstrap_gate_replay") is not True
        or compatibility.get("bootstrap_replicates_per_point") != 5000
        or compatibility.get("bootstrap_points") != 5
        or compatibility.get("research_only_no_promotion") is not True
    ):
        raise CompatibilityV4Error("v4 compatibility science contract changed")
    runtime = config.get("canonical_runtime_contract")
    if runtime != _CONTEXT["runtime_contract"]:
        raise CompatibilityV4Error("canonical runtime/dependency contract changed")
    if (
        runtime.get("external_startup_trust", {}).get("required") is not True
        or runtime.get("external_startup_trust", {}).get(
            "pycache_regular_file_sentinel_must_be_pinned_nlink1_non_reparse_and_held_share_deny"
        )
        is not True
        or runtime.get("native_extension_execution")
        != "AUTHENTICATED_HELD_BYTES_BOUND_TO_PATH_LOAD"
    ):
        raise CompatibilityV4Error("external startup or native-load trust changed")
    semantic = config.get("stable_semantic_read_contract")
    if semantic != _CONTEXT["semantic_read_contract"]:
        raise CompatibilityV4Error("stable semantic-read contract changed")
    if (
        semantic.get("hardlink_nlink_must_equal_one") is not True
        or semantic.get("parquet_file_constructor_from_buffer_only") is not True
        or semantic.get("all_json_and_jsonl_strict") is not True
    ):
        raise CompatibilityV4Error("v4 strict semantic containment changed")


def _require_absence(config: Mapping[str, Any]) -> None:
    expected = {
        "compatibility_control": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_"
            "v1r2_compatibility_verifier_v4_control"
        ),
        "pre_execution_qa": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_"
            "v1r2_compatibility_verifier_v4_control/pre_execution_qa.json"
        ),
        "compatibility_receipt": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_"
            "v1r2_compatibility_verifier_v4_control/compatibility_receipt.json"
        ),
    }
    if config.get("canonical_paths") != expected:
        raise CompatibilityV4Error("canonical v4 absence paths changed")
    for label, relative in expected.items():
        if not _CONTEXT["absent"](relative):
            raise CompatibilityV4Error(f"forbidden v4 state exists: {label}")


def _validate_legacy_result(result: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if result.get("status") != "PASS_AUTHENTICATED_R2_COMPATIBILITY_RESEARCH_ONLY_NO_PROMOTION":
        raise CompatibilityV4Error("frozen compatibility-v2 replay did not pass")
    if result.get("compatibility_adaptations") != {
        "count": 2,
        "only": list(EXPECTED_ADAPTATIONS),
        "globals_restored": True,
    }:
        raise CompatibilityV4Error("the exact two compatibility adaptations changed")
    legacy = result.get("legacy_compatibility_verification")
    if not isinstance(legacy, Mapping):
        raise CompatibilityV4Error("legacy compatibility result is missing")
    frozen = legacy.get("legacy_compatibility_verification", legacy)
    if frozen.get("status") != "PASS_R2_COMPATIBILITY_VERIFIER_RESEARCH_ONLY_NO_PROMOTION":
        raise CompatibilityV4Error("frozen v1/r2 replay status changed")
    science = frozen.get("legacy_compatibility_verification", frozen)
    if science.get("status") not in {
        "PASS_R2_COMPATIBILITY_VERIFIER_RESEARCH_ONLY_NO_PROMOTION",
        "NO_LOCAL_CURVE_QUALIFICATION_RESEARCH_ONLY_STOPPED_BEFORE_TEST",
    }:
        raise CompatibilityV4Error("sealed research-only science status changed")
    expected = config["expected_result"]
    numerical = frozen.get("independent_metric_verification", {})
    if numerical:
        gate = numerical.get("gate", {})
        if (
            numerical.get("bootstrap_replicates_total") != 25000
            or numerical.get("points_deep_equal") is not True
            or numerical.get("gate_deep_equal") is not True
            or gate.get("decision") != expected["gate_decision"]
            or gate.get("passed") is not expected["local_gate_passed"]
            or numerical.get("identity_cells") != expected["identity_cells"]
            or numerical.get("bounded_correction_cells") != expected["bounded_correction_cells"]
        ):
            raise CompatibilityV4Error("independent OOF/bootstrap/gate replay changed")
    forbidden = (
        "files_written",
        "independent_qa_receipts_created",
        "compatibility_receipts_created",
        "execution_authorizations_created",
        "attempt_locks_created",
        "model_fit_calls",
        "prediction_calls",
        "source_train_target_scalar_decodes",
        "experiment_score_calls",
        "candidate_files",
        "test_prediction_files",
        "registry_appends",
        "uploads",
    )
    if any(result.get(key, 0) != 0 for key in forbidden):
        raise CompatibilityV4Error("legacy replay reported a forbidden side effect")


def verify_trusted(
    root: Path,
    *,
    requested_config: Path | None = None,
    mode: str = "check-only",
) -> dict[str, Any]:
    """Consume the v4 capability once and replay from authenticated buffers."""

    if mode != "check-only":
        raise CompatibilityV4Error("compatibility-v4 supports check-only mode")
    workspace = Path(root)
    if not _CONTEXT["same_path"](workspace, _CONTEXT["workspace"]):
        raise CompatibilityV4Error("canonical workspace identity differs")
    if ENGINE_MODULE in sys.modules:
        raise CompatibilityV4Error("r2 execution engine was imported")
    config = _parse_config_once(requested_config)
    _require_absence(config)
    runtime_report = _CONTEXT["runtime_report"]()
    startup = runtime_report.get("external_startup_trust", {})
    if (
        startup.get("pre_script_bytecode_executed") is not False
        or startup.get("external_launcher", {}).get(
            "authenticated_and_held_by_externally_pinned_encoded_stage0"
        )
        is not True
        or startup.get("pinned_regular_file_pycache_sentinel", {}).get("held") is not True
        or startup.get("external_powershell_host", {})
        .get("distribution_inventory", {})
        .get("externally_preauthenticated_and_launcher_held")
        is not True
        or not startup.get("startup_files")
    ):
        raise CompatibilityV4Error("external startup trust report is incomplete")
    _CONTEXT["claim_phase"]("V4_CHECK_ONLY_ONCE", _TOKEN)
    _CONTEXT["reverify"]("v4_helper_pre_legacy_replay")
    result = _CONTEXT["run_legacy_replay"](_TOKEN)
    _validate_legacy_result(result, config)
    _CONTEXT["reverify"]("v4_helper_post_legacy_replay")
    _require_absence(config)
    if ENGINE_MODULE in sys.modules:
        raise CompatibilityV4Error("r2 execution engine was imported during replay")
    return {
        "schema_version": (
            "p3_gen6_incumbent_preserving_residual_calibrator.r2_compatibility_check.v4"
        ),
        "status": "PASS_HELD_BUFFER_R2_COMPATIBILITY_RESEARCH_ONLY_NO_PROMOTION",
        "identity": IDENTITY,
        "trusted_runtime": runtime_report,
        "stable_semantic_registry": _CONTEXT["semantic_report"](),
        "compatibility_adaptations": {
            "count": 2,
            "only": list(EXPECTED_ADAPTATIONS),
            "globals_restored": True,
        },
        "legacy_compatibility_verification": result,
        "v4_config_parse_count": _CONFIG_PARSE_COUNT,
        "v4_control_exists": False,
        "v4_independent_qa_receipt_exists": False,
        "v4_compatibility_receipt_exists": False,
        "files_written": 0,
        "model_fit_calls": 0,
        "prediction_calls": 0,
        "new_score_calls": 0,
        "candidate_files": 0,
        "test_prediction_files": 0,
        "registry_appends": 0,
        "uploads": 0,
    }


__all__ = [
    "BOOTSTRAP_RELATIVE",
    "CONFIG_RELATIVE",
    "CompatibilityV4Error",
    "EXPECTED_ADAPTATIONS",
    "IDENTITY",
    "verify_trusted",
]
