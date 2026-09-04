"""Held-buffer compatibility-v3 wrapper for the frozen P3 Gen6r2 result.

This source is never imported by ``PathFinder``.  The externally pinned v3
bootstrap authenticates its exact bytes, compiles those bytes in memory, and
injects an opaque, single-use capability.  All legacy semantic reads and the
two historical compatibility adaptations are owned by that bootstrap.
"""

from __future__ import annotations

if (
    "__trusted_v3_context__" not in globals()
    or "__trusted_v3_token__" not in globals()
):
    raise RuntimeError("compatibility-v3 helper requires the trusted v3 bootstrap")

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

_CONTEXT = globals()["__trusted_v3_context__"]
_TOKEN = globals()["__trusted_v3_token__"]
if _CONTEXT.get("token") is not _TOKEN:
    raise RuntimeError("compatibility-v3 opaque bootstrap token identity differs")

IDENTITY: Final = (
    "P3_GEN6_INCUMBENT_PRESERVING_RESIDUAL_CALIBRATOR_R2_COMPATIBILITY_VERIFIER_V3"
)
CONFIG_RELATIVE: Final = (
    "configs/experiments/p3_gen6_incumbent_preserving_residual_calibrator_"
    "v1r2_compatibility_verifier_v3.json"
)
BOOTSTRAP_RELATIVE: Final = (
    "scripts/bootstrap_verify_p3_gen6_incumbent_preserving_residual_calibrator_"
    "r2_compatibility_v3.py"
)
ENGINE_MODULE: Final = "p3_wave.gen6_incumbent_preserving_residual_calibrator_execution_r2"
EXPECTED_ADAPTATIONS: Final = (
    "replace_frozen_r2_verifier_prefix_expectation_with_four_source_consensus",
    "admit_exact_pinned_historical_failure_receipt_to_the_r2_control_inventory",
)
EXPECTED_FINDINGS: Final = (
    "PREIMPORT_AND_DEPENDENCY_TRUST_BOUNDARY_NOT_CLOSED",
    "PINNED_ARTIFACT_BYTES_NOT_BOUND_TO_SEMANTIC_REPLAY",
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


class CompatibilityV3Error(RuntimeError):
    """The authenticated held-buffer v3 contract was not satisfied."""


def _duplicates_forbidden(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CompatibilityV3Error("duplicate JSON object key is forbidden")
        value[key] = item
    return value


def _reject_constant(value: str) -> Any:
    raise CompatibilityV3Error(f"non-finite JSON constant is forbidden: {value}")


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_duplicates_forbidden,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompatibilityV3Error(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CompatibilityV3Error(f"{label} must be a JSON object")
    return value


def _parse_config_once(requested_config: Path | None) -> dict[str, Any]:
    global _CONFIG_PARSE_COUNT
    if _CONFIG_PARSE_COUNT != 0:
        raise CompatibilityV3Error("v3 config may be parsed exactly once")
    canonical = _CONTEXT["canonical_path"](CONFIG_RELATIVE, "workspace")
    if requested_config is not None:
        requested = Path(requested_config)
        if not requested.is_absolute():
            requested = _CONTEXT["workspace"] / requested
        if not _CONTEXT["same_path"](requested, canonical):
            raise CompatibilityV3Error("alternate compatibility-v3 config is forbidden")
    raw = _CONTEXT["source_buffer"]("V3_CONFIG")
    pin = _CONTEXT["source_pin"]("V3_CONFIG")
    if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise CompatibilityV3Error("v3 config authenticated-buffer identity differs")
    config = _strict_json(raw, label="v3 config")
    _CONFIG_PARSE_COUNT += 1
    _validate_config(config)
    _CONTEXT["reverify"]("v3_helper_post_config_parse")
    return config


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version")
        != "p3_gen6_incumbent_preserving_residual_calibrator.r2_compatibility_verifier.v3"
        or config.get("problem") != "P3"
        or config.get("identity") != IDENTITY
        or config.get("verifier_only") is not True
        or config.get("check_only_default") is not True
        or config.get("append_only_successor_of_v2") is not True
    ):
        raise CompatibilityV3Error("v3 identity or read-only mode changed")
    if any(config.get(flag) is not False for flag in FALSE_FLAGS):
        raise CompatibilityV3Error("v3 read-only firewall changed")
    counters = config.get("static_counters")
    if not isinstance(counters, Mapping) or not counters or any(counters.values()):
        raise CompatibilityV3Error("v3 static counters changed")
    if config.get("implementation_roles") != _CONTEXT["implementation_roles"]:
        raise CompatibilityV3Error("v3 implementation roles differ from trust root")
    if config.get("authenticated_subordinate_pins") != _CONTEXT["subordinate_pins"]:
        raise CompatibilityV3Error("v3 subordinate pins differ from trust root")
    disposition = config.get("v2_disposition")
    if (
        not isinstance(disposition, Mapping)
        or disposition.get("reviewer")
        != "/root/meaningful_improvement_audit/p2_stageb_blind_review"
        or disposition.get("verdict") != "P0=0_P1=2_OWNER_KNOWN_NO_GO"
        or disposition.get("independent_qa_receipt_file_exists") is not False
        or disposition.get("independent_qa_receipt_hash_exists") is not False
        or tuple(disposition.get("finding_ids", ())) != EXPECTED_FINDINGS
    ):
        raise CompatibilityV3Error("v2 NO-GO lineage changed")
    compatibility = config.get("compatibility_contract")
    if (
        not isinstance(compatibility, Mapping)
        or tuple(compatibility.get("only_scoped_science_adaptations", ()))
        != EXPECTED_ADAPTATIONS
        or compatibility.get("adaptation_count") != 2
        or compatibility.get("all_other_v1_and_r2_checks_reexecuted") is not True
        or compatibility.get("independent_oof_bootstrap_gate_replay") is not True
        or compatibility.get("bootstrap_replicates_per_point") != 5000
        or compatibility.get("bootstrap_points") != 5
        or compatibility.get("research_only_no_promotion") is not True
    ):
        raise CompatibilityV3Error("v3 compatibility science contract changed")
    runtime = config.get("canonical_runtime_contract")
    if runtime != _CONTEXT["runtime_contract"]:
        raise CompatibilityV3Error("canonical runtime/dependency contract changed")
    semantic = config.get("stable_semantic_read_contract")
    if semantic != _CONTEXT["semantic_read_contract"]:
        raise CompatibilityV3Error("stable semantic-read contract changed")


def _require_absence(config: Mapping[str, Any]) -> None:
    expected = {
        "compatibility_control": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_"
            "v1r2_compatibility_verifier_v3_control"
        ),
        "pre_execution_qa": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_"
            "v1r2_compatibility_verifier_v3_control/pre_execution_qa.json"
        ),
        "compatibility_receipt": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_"
            "v1r2_compatibility_verifier_v3_control/compatibility_receipt.json"
        ),
    }
    if config.get("canonical_paths") != expected:
        raise CompatibilityV3Error("canonical v3 absence paths changed")
    for label, relative in expected.items():
        if not _CONTEXT["absent"](relative):
            raise CompatibilityV3Error(f"forbidden v3 state exists: {label}")


def _validate_legacy_result(result: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if result.get("status") != "PASS_AUTHENTICATED_R2_COMPATIBILITY_RESEARCH_ONLY_NO_PROMOTION":
        raise CompatibilityV3Error("frozen compatibility-v2 replay did not pass")
    if result.get("compatibility_adaptations") != {
        "count": 2,
        "only": list(EXPECTED_ADAPTATIONS),
        "globals_restored": True,
    }:
        raise CompatibilityV3Error("the exact two compatibility adaptations changed")
    legacy = result.get("legacy_compatibility_verification")
    if not isinstance(legacy, Mapping):
        raise CompatibilityV3Error("legacy compatibility result is missing")
    frozen = legacy.get("legacy_compatibility_verification", legacy)
    if frozen.get("status") != "PASS_R2_COMPATIBILITY_VERIFIER_RESEARCH_ONLY_NO_PROMOTION":
        raise CompatibilityV3Error("frozen v1/r2 replay status changed")
    science = frozen.get("legacy_compatibility_verification", frozen)
    if science.get("status") not in {
        "PASS_R2_COMPATIBILITY_VERIFIER_RESEARCH_ONLY_NO_PROMOTION",
        "NO_LOCAL_CURVE_QUALIFICATION_RESEARCH_ONLY_STOPPED_BEFORE_TEST",
    }:
        raise CompatibilityV3Error("sealed research-only science status changed")
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
            or numerical.get("bounded_correction_cells")
            != expected["bounded_correction_cells"]
        ):
            raise CompatibilityV3Error("independent OOF/bootstrap/gate replay changed")
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
        raise CompatibilityV3Error("legacy replay reported a forbidden side effect")


def verify_trusted(
    root: Path,
    *,
    requested_config: Path | None = None,
    mode: str = "check-only",
) -> dict[str, Any]:
    """Consume the v3 capability once and replay from authenticated buffers."""

    if mode != "check-only":
        raise CompatibilityV3Error("compatibility-v3 supports check-only mode")
    workspace = Path(root)
    if not _CONTEXT["same_path"](workspace, _CONTEXT["workspace"]):
        raise CompatibilityV3Error("canonical workspace identity differs")
    if ENGINE_MODULE in sys.modules:
        raise CompatibilityV3Error("r2 execution engine was imported")
    config = _parse_config_once(requested_config)
    _require_absence(config)
    _CONTEXT["claim_phase"]("V3_CHECK_ONLY_ONCE", _TOKEN)
    _CONTEXT["reverify"]("v3_helper_pre_legacy_replay")
    result = _CONTEXT["run_legacy_replay"](_TOKEN)
    _validate_legacy_result(result, config)
    _CONTEXT["reverify"]("v3_helper_post_legacy_replay")
    _require_absence(config)
    if ENGINE_MODULE in sys.modules:
        raise CompatibilityV3Error("r2 execution engine was imported during replay")
    return {
        "schema_version": (
            "p3_gen6_incumbent_preserving_residual_calibrator.r2_compatibility_check.v3"
        ),
        "status": "PASS_HELD_BUFFER_R2_COMPATIBILITY_RESEARCH_ONLY_NO_PROMOTION",
        "identity": IDENTITY,
        "trusted_runtime": _CONTEXT["runtime_report"](),
        "stable_semantic_registry": _CONTEXT["semantic_report"](),
        "compatibility_adaptations": {
            "count": 2,
            "only": list(EXPECTED_ADAPTATIONS),
            "globals_restored": True,
        },
        "legacy_compatibility_verification": result,
        "v3_config_parse_count": _CONFIG_PARSE_COUNT,
        "v3_control_exists": False,
        "v3_independent_qa_receipt_exists": False,
        "v3_compatibility_receipt_exists": False,
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
    "CompatibilityV3Error",
    "EXPECTED_ADAPTATIONS",
    "IDENTITY",
    "verify_trusted",
]
