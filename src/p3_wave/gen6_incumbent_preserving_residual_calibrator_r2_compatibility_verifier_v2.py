"""Authenticated-buffer wrapper for the frozen Gen6r2 compatibility verifier.

This module is intentionally not importable through Python's import machinery.
The noncyclic v2 bootstrap must authenticate every executable source buffer,
compile/exec the exact bytes, and inject the opaque trust context first.  The
scientific verification remains the byte-frozen compatibility-v1 procedure.
"""

from __future__ import annotations

if (
    "__trusted_bootstrap_context__" not in globals()
    or "__trusted_bootstrap_token__" not in globals()
):
    raise RuntimeError("compatibility-v2 helper requires the trusted bootstrap")

import hashlib
import json
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

_TRUST_CONTEXT = globals()["__trusted_bootstrap_context__"]
_TRUST_TOKEN = globals()["__trusted_bootstrap_token__"]
if _TRUST_CONTEXT.get("token") is not _TRUST_TOKEN:
    raise RuntimeError("compatibility-v2 bootstrap token identity differs")
if globals().get("__trusted_v1_module__") is not _TRUST_CONTEXT.get("v1_module"):
    raise RuntimeError("compatibility-v2 legacy verifier identity differs")
if globals().get("__trusted_r2_module__") is not _TRUST_CONTEXT.get("r2_module"):
    raise RuntimeError("compatibility-v2 r2 contract identity differs")

v1 = globals()["__trusted_v1_module__"]
r2 = globals()["__trusted_r2_module__"]

IDENTITY: Final = "P3_GEN6_INCUMBENT_PRESERVING_RESIDUAL_CALIBRATOR_R2_COMPATIBILITY_VERIFIER_V2"
CONFIG_RELATIVE: Final = (
    "configs/experiments/"
    "p3_gen6_incumbent_preserving_residual_calibrator_v1r2_compatibility_verifier_v2.json"
)
BOOTSTRAP_RELATIVE: Final = (
    "scripts/bootstrap_verify_p3_gen6_incumbent_preserving_residual_calibrator_"
    "r2_compatibility_v2.py"
)
ENGINE_MODULE: Final = "p3_wave.gen6_incumbent_preserving_residual_calibrator_execution_r2"
EXPECTED_ADAPTATIONS: Final = (
    "replace_frozen_r2_verifier_prefix_expectation_with_four_source_consensus",
    "admit_exact_pinned_historical_failure_receipt_to_the_r2_control_inventory",
)
EXPECTED_FALSE_FLAGS: Final = (
    "r2_mutation_allowed",
    "r2_rerun_or_resume_allowed",
    "qa_receipt_or_compatibility_receipt_write_allowed",
    "execution_authorization_or_attempt_lock_allowed",
    "fit_prediction_source_truth_decode_or_experiment_scoring_allowed",
    "official_promotion_allowed",
    "candidate_or_test_prediction_allowed",
    "registry_append_allowed",
    "upload_allowed",
)
_CONFIG_PARSE_COUNT = 0


class CompatibilityV2Error(RuntimeError):
    """The authenticated compatibility-v2 contract was not satisfied."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise CompatibilityV2Error(f"{label} field set changed")


def _canonical_config_path() -> Path:
    return _TRUST_CONTEXT["checked_path"](CONFIG_RELATIVE, must_exist=True, kind="file")


def _parse_config_once(requested_config: Path | None) -> dict[str, Any]:
    global _CONFIG_PARSE_COUNT
    if _CONFIG_PARSE_COUNT != 0:
        raise CompatibilityV2Error("v2 config may be parsed exactly once")
    canonical = _canonical_config_path()
    if requested_config is not None:
        requested = Path(requested_config)
        if not requested.is_absolute():
            requested = _TRUST_CONTEXT["root"] / requested
        if (
            requested.absolute() != canonical.absolute()
            or requested.resolve(strict=True) != canonical
        ):
            raise CompatibilityV2Error("alternate compatibility-v2 config is forbidden")

    raw = _TRUST_CONTEXT["buffers"]["CONFIG"]
    pin = _TRUST_CONTEXT["pins"]["CONFIG"]
    if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise CompatibilityV2Error("authenticated v2 config buffer identity differs")
    try:
        config = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompatibilityV2Error("authenticated v2 config is not strict UTF-8 JSON") from exc
    _CONFIG_PARSE_COUNT += 1
    if not isinstance(config, dict):
        raise CompatibilityV2Error("compatibility-v2 config must be an object")
    _validate_config(config)
    _TRUST_CONTEXT["reverify"]("helper_post_config_parse")
    return config


def _validate_config(config: Mapping[str, Any]) -> None:
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
            "trusted_bootstrap",
            "implementation_roles",
            "authenticated_source_pins",
            "canonical_paths",
            "lineage",
            "compatibility_contract",
            "expected_result",
            "r2_mutation_allowed",
            "r2_rerun_or_resume_allowed",
            "qa_receipt_or_compatibility_receipt_write_allowed",
            "execution_authorization_or_attempt_lock_allowed",
            "fit_prediction_source_truth_decode_or_experiment_scoring_allowed",
            "official_promotion_allowed",
            "candidate_or_test_prediction_allowed",
            "registry_append_allowed",
            "upload_allowed",
            "static_counters",
        },
        label="compatibility-v2 config",
    )
    if (
        config.get("schema_version")
        != "p3_gen6_incumbent_preserving_residual_calibrator.r2_compatibility_verifier.v2"
        or config.get("problem") != "P3"
        or config.get("identity") != IDENTITY
        or config.get("verifier_only") is not True
        or config.get("check_only_default") is not True
    ):
        raise CompatibilityV2Error("compatibility-v2 identity or mode changed")
    if any(config.get(flag) is not False for flag in EXPECTED_FALSE_FLAGS):
        raise CompatibilityV2Error("compatibility-v2 read-only firewall changed")
    counters = config.get("static_counters")
    if (
        not isinstance(counters, Mapping)
        or not counters
        or any(value != 0 for value in counters.values())
    ):
        raise CompatibilityV2Error("compatibility-v2 static counters changed")

    bootstrap = config.get("trusted_bootstrap")
    if not isinstance(bootstrap, Mapping) or dict(bootstrap) != {
        "path": BOOTSTRAP_RELATIVE,
        "role": "NONCYCLIC_EXTERNALLY_PINNED_PRE_IMPORT_TRUST_ROOT",
        "self_hash_embedded": False,
        "independent_static_qa_required_before_attestation": True,
    }:
        raise CompatibilityV2Error("trusted bootstrap contract changed")
    if config.get("implementation_roles") != _TRUST_CONTEXT["implementation_roles"]:
        raise CompatibilityV2Error("v2 implementation role map changed")
    expected_subordinate_pins = {
        role: dict(pin) for role, pin in _TRUST_CONTEXT["pins"].items() if role != "CONFIG"
    }
    if config.get("authenticated_source_pins") != expected_subordinate_pins:
        raise CompatibilityV2Error("authenticated source pin map differs from trust root")

    compatibility = config.get("compatibility_contract")
    if not isinstance(compatibility, Mapping):
        raise CompatibilityV2Error("compatibility contract is not an object")
    if tuple(compatibility.get("only_scoped_adaptations", ())) != EXPECTED_ADAPTATIONS:
        raise CompatibilityV2Error("the exact two compatibility adaptations changed")
    if compatibility.get("adaptation_count") != 2:
        raise CompatibilityV2Error("compatibility adaptation count changed")
    if compatibility.get("all_other_v1_and_r2_checks_reexecuted") is not True:
        raise CompatibilityV2Error("full frozen verifier replay requirement changed")
    if compatibility.get("independent_oof_bootstrap_gate_replay") is not True:
        raise CompatibilityV2Error("independent numerical replay requirement changed")


def _require_absence(config: Mapping[str, Any]) -> None:
    paths = config["canonical_paths"]
    expected = {
        "compatibility_control": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_"
            "v1r2_compatibility_verifier_v2_control"
        ),
        "pre_execution_qa": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_"
            "v1r2_compatibility_verifier_v2_control/pre_execution_qa.json"
        ),
        "compatibility_receipt": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_"
            "v1r2_compatibility_verifier_v2_control/compatibility_receipt.json"
        ),
    }
    if paths != expected:
        raise CompatibilityV2Error("compatibility-v2 canonical control paths changed")
    for label, relative in expected.items():
        path = _TRUST_CONTEXT["checked_path"](relative, must_exist=False, kind=None)
        if path.exists() or path.is_symlink():
            raise CompatibilityV2Error(f"forbidden compatibility-v2 state exists: {label}")


@contextmanager
def _restoration_guard() -> Iterator[None]:
    """Fail closed and restore the two frozen globals on every exit path."""

    original_prefixes = r2.PREFIX_FRACTIONS
    original_control = r2._control_inventory
    caught: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        caught = exc
        raise
    finally:
        changed = (
            r2.PREFIX_FRACTIONS is not original_prefixes
            or r2._control_inventory is not original_control
        )
        r2._control_inventory = original_control
        r2.PREFIX_FRACTIONS = original_prefixes
        if changed:
            raise CompatibilityV2Error(
                "legacy verifier failed to restore adapted globals"
            ) from caught


def _validate_result(result: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    expected = config["expected_result"]
    if result.get("status") != "PASS_R2_COMPATIBILITY_VERIFIER_RESEARCH_ONLY_NO_PROMOTION":
        raise CompatibilityV2Error("legacy compatibility verifier did not pass")
    if result.get("frozen_r2_verifier", {}).get("status") != (
        "POST_PUBLISH_VERIFIED_EXACT_ALLOWLIST_AND_LINEAGE"
    ):
        raise CompatibilityV2Error("frozen r2 verifier replay did not pass")
    reconciliation = result.get("oof_reconciliation", {})
    if (
        reconciliation.get("truth_bytes_exact_to_sealed_gen1") is not True
        or reconciliation.get("keys_exact_to_sealed_gen1") is not True
    ):
        raise CompatibilityV2Error("independent OOF reconciliation changed")
    metrics = result.get("independent_metric_verification", {})
    gate = metrics.get("gate", {})
    if (
        metrics.get("bootstrap_replicates_total") != 25000
        or metrics.get("points_deep_equal") is not True
        or metrics.get("gate_deep_equal") is not True
        or metrics.get("central_evidence_deep_equal") is not True
        or gate.get("decision") != "RESEARCH_ONLY"
        or gate.get("passed") is not False
    ):
        raise CompatibilityV2Error("independent bootstrap or gate replay changed")
    if (
        result.get("historical_failure_receipt", {}).get("message_sha256")
        != expected["historical_failure_message_sha256"]
        or result.get("prefix_compatibility", {}).get("corrected_prefix_fractions")
        != expected["corrected_prefix_fractions"]
        or result.get("frozen_r2_verifier", {})
        .get("commitments", {})
        .get("fit_count_observed_exact")
        != expected["fit_count_observed_exact"]
        or metrics.get("identity_cells") != expected["identity_cells"]
        or metrics.get("bounded_correction_cells") != expected["bounded_correction_cells"]
    ):
        raise CompatibilityV2Error("frozen research result differs")
    for key in (
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
    ):
        if result.get(key) != 0:
            raise CompatibilityV2Error(f"forbidden compatibility-v2 counter is nonzero: {key}")


def verify_trusted(
    root: Path,
    *,
    requested_config: Path | None = None,
) -> dict[str, Any]:
    """Run the exact legacy verifier from a bootstrap-authenticated context."""

    workspace = Path(root)
    if (
        workspace.absolute() != _TRUST_CONTEXT["root"].absolute()
        or workspace.resolve(strict=True) != _TRUST_CONTEXT["root"]
    ):
        raise CompatibilityV2Error("canonical workspace identity differs")
    if sys.modules.get(ENGINE_MODULE) is not None:
        raise CompatibilityV2Error("r2 execution engine was imported")
    config = _parse_config_once(requested_config)
    _require_absence(config)
    _TRUST_CONTEXT["reverify"]("helper_pre_legacy_verifier_entry")

    original_prefixes = r2.PREFIX_FRACTIONS
    original_control = r2._control_inventory
    with _restoration_guard():
        result = v1.verify_static_compatibility(workspace)
    if (
        r2.PREFIX_FRACTIONS is not original_prefixes
        or r2._control_inventory is not original_control
    ):
        raise CompatibilityV2Error("r2 globals differ after compatibility replay")
    if sys.modules.get(ENGINE_MODULE) is not None:
        raise CompatibilityV2Error("r2 execution engine was imported during verification")
    _validate_result(result, config)
    _require_absence(config)
    _TRUST_CONTEXT["reverify"]("helper_post_legacy_verifier")

    return {
        "schema_version": (
            "p3_gen6_incumbent_preserving_residual_calibrator.r2_compatibility_check.v2"
        ),
        "status": "PASS_AUTHENTICATED_R2_COMPATIBILITY_RESEARCH_ONLY_NO_PROMOTION",
        "identity": IDENTITY,
        "trusted_bootstrap": {
            "path": BOOTSTRAP_RELATIVE,
            "noncyclic_external_trust_root": True,
            "authenticated_all_sources_before_module_execution": True,
            "authenticated_buffer_compile_exec_only": True,
            "config_parse_count": _CONFIG_PARSE_COUNT,
            "post_read_identity_reverified": True,
            "sys_dont_write_bytecode": sys.dont_write_bytecode,
        },
        "compatibility_adaptations": {
            "count": 2,
            "only": list(EXPECTED_ADAPTATIONS),
            "globals_restored": True,
        },
        "legacy_compatibility_verification": result,
        "compatibility_control_exists": False,
        "compatibility_qa_receipt_exists": False,
        "compatibility_receipt_exists": False,
        "files_written": 0,
        "independent_qa_receipts_created": 0,
        "compatibility_receipts_created": 0,
        "execution_authorizations_created": 0,
        "attempt_locks_created": 0,
        "model_fit_calls": 0,
        "prediction_calls": 0,
        "source_train_target_scalar_decodes": 0,
        "experiment_score_calls": 0,
        "candidate_files": 0,
        "test_prediction_files": 0,
        "registry_appends": 0,
        "uploads": 0,
    }


__all__ = [
    "BOOTSTRAP_RELATIVE",
    "CONFIG_RELATIVE",
    "CompatibilityV2Error",
    "EXPECTED_ADAPTATIONS",
    "IDENTITY",
    "verify_trusted",
]
