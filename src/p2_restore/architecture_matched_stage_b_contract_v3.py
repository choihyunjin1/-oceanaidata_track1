"""Fail-closed contract for the P2 architecture-matched Stage-B v3 cycle."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from p2_restore import architecture_matched_stage_a_contract_v3 as stage_a_guard

CONFIG_RELATIVE = "configs/experiments/p2_architecture_matched_stage_b_challenger_v3.json"
CONFIG_SHA256 = "dc11ea956241531a19e846bee95fbac09df68fe4615930336355478b108cc798"
MODE = "ARCHITECTURE_MATCHED_TIME_SAFE_BASELINE"
STAGE = "P2_ARCHITECTURE_MATCHED_STAGE_B_CHALLENGER_V3"
ENGINE_RELATIVE = "src/p2_restore/architecture_matched_stage_b_execution_v3.py"
ENGINE_MODULE = "p2_restore.architecture_matched_stage_b_execution_v3"
IMPLEMENTATION_ROLES = {
    "CONFIG": CONFIG_RELATIVE,
    "GUARD": "src/p2_restore/architecture_matched_stage_b_contract_v3.py",
    "ENGINE": ENGINE_RELATIVE,
    "RUNNER": "scripts/run_p2_architecture_matched_stage_b_v3.py",
    "TESTS": "tests/test_p2_architecture_matched_stage_b_v3.py",
}
STAGE_A_ROLES = {
    "CONFIG",
    "MANIFEST",
    "SEAL",
    "ARCHITECTURE_MANIFEST",
    "TRAINING_RECIPE",
    "TRAINING_RECEIPT",
    "CURVE_METRICS",
    "OOF_040",
    "OOF_055",
    "OOF_070",
    "OOF_085",
    "OOF_100",
    "QA_RECEIPT",
    "AUTHORIZATION",
    "ATTEMPT_LOCK",
}
FRACTION_ROLES = {
    "0.4": "OOF_040",
    "0.55": "OOF_055",
    "0.7": "OOF_070",
    "0.85": "OOF_085",
    "1.0": "OOF_100",
}
NUMERICAL_PREFIXES = ("lightgbm", "numpy", "pandas", "scipy", "sklearn", "torch")


class StageBContractError(ValueError):
    """Raised when the Stage-B preregistration or its sealed inputs drift."""


def sha256_file(path: Path) -> str:
    return stage_a_guard.sha256_file(path)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return stage_a_guard.canonical_json_bytes(value)


def strict_json_object(path: Path) -> dict[str, Any]:
    return stage_a_guard.strict_json_object(path)


def workspace_path(root: Path, relative: str | Path, *, must_exist: bool = True) -> Path:
    try:
        return stage_a_guard.workspace_path(root, relative, must_exist=must_exist)
    except stage_a_guard.StageAContractV3Error as exc:
        raise StageBContractError(str(exc)) from exc


def contained_path(output: Path, child: str | Path) -> Path:
    try:
        return stage_a_guard.contained_path(output, child)
    except stage_a_guard.StageAContractV3Error as exc:
        raise StageBContractError(str(exc)) from exc


def exclusive_bytes(path: Path, payload: bytes) -> None:
    stage_a_guard.exclusive_bytes(path, payload)


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    stage_a_guard.exclusive_json(path, value)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise StageBContractError(f"{name} keys changed")


def _validate_pin_map(pins: Any, *, roles: set[str], name: str) -> None:
    if not isinstance(pins, Mapping) or set(pins) != roles:
        raise StageBContractError(f"{name} roles changed")
    for role, pin in pins.items():
        if not isinstance(pin, Mapping) or set(pin) != {"path", "sha256", "bytes"}:
            raise StageBContractError(f"{name} pin shape changed: {role}")
        if (
            not isinstance(pin["path"], str)
            or not _is_sha256(pin["sha256"])
            or not isinstance(pin["bytes"], int)
            or pin["bytes"] <= 0
        ):
            raise StageBContractError(f"{name} pin is invalid: {role}")


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "p2_architecture_matched_stage_b_challenger.v3":
        raise StageBContractError("Stage-B config schema changed")
    if config.get("problem") != "P2" or config.get("comparison_mode") != MODE:
        raise StageBContractError("architecture-matched Stage B is P2-only")
    for key in (
        "exact_official_incumbent_comparison",
        "official_promotion_allowed",
        "upload_allowed",
    ):
        if config.get(key) is not False:
            raise StageBContractError("non-exact local-only status changed")
    if (
        config.get("explicitly_not_exact_official_incumbent") is not True
        or config.get("local_qualification_only") is not True
        or config.get("research_only") is not True
        or config.get("official_submission_count") != 0
    ):
        raise StageBContractError("research-only status changed")
    if config.get("preregistration") != {
        "generation_id": "P2_ARCHITECTURE_MATCHED_STAGE_B_CHALLENGER_V3",
        "created_before_first_fit": True,
        "hypothesis_count": 1,
        "score_derived_tuning": False,
        "prior_result_used_for_parameter_search": False,
    }:
        raise StageBContractError("Stage-B preregistration changed")

    hypothesis = config.get("hypothesis")
    if not isinstance(hypothesis, Mapping):
        raise StageBContractError("challenger hypothesis is missing")
    if (
        hypothesis.get("id") != "H1_SEEDED_CONDITIONAL_ANALOG_RANK2_PROFILE_MANIFOLD"
        or hypothesis.get("family") != "GPU_CONDITIONAL_ANALOG_LOCAL_AFFINE_PROFILE_MANIFOLD"
        or hypothesis.get("not_a_blend_weight_or_alpha_tweak") is not True
        or hypothesis.get("catboost_used") is not False
        or hypothesis.get("failed_generation_v1_reused") is not False
        or hypothesis.get("hyperparameter_searches") != 0
        or hypothesis.get("target_layer_values_in_features") is not False
        or hypothesis.get("joint_temp_psal_training_mask") is not True
    ):
        raise StageBContractError("structural challenger identity changed")
    model = hypothesis.get("model")
    if not isinstance(model, Mapping) or any(
        model.get(key) != expected
        for key, expected in {
            "projection_dimensions": 16,
            "nearest_neighbors": 128,
            "neighbor_backend": "CUDA_FLOAT32_BATCHED_CDIST_TOPK",
            "query_batch_size": 256,
            "local_manifold_rank": 2,
            "incomplete_three_layer_groups": "identity_keep_stage_a_seed_prediction",
        }.items()
    ):
        raise StageBContractError("conditional-manifold architecture changed")

    reference = config.get("stage_a_reference")
    if not isinstance(reference, Mapping):
        raise StageBContractError("Stage-A reference is missing")
    _validate_pin_map(reference.get("artifacts"), roles=STAGE_A_ROLES, name="Stage-A")
    if reference.get("seal_verified_before_challenger_import_fit_or_score") is not True:
        raise StageBContractError("seal-before-challenger rule changed")
    if reference.get("expected") != {
        "complete": True,
        "all_five_prefixes_sealed": True,
        "cells": 45,
        "deep_training_jobs": 720,
        "router_training_jobs": 180,
        "challenger_fit_or_score_count_before_seal": 0,
        "submission_predictions": 0,
        "uploads": 0,
        "full_reference_rmse_c": 1.0109798870010898,
    }:
        raise StageBContractError("Stage-A expected state changed")
    source = config.get("source_pins")
    if not isinstance(source, Mapping) or len(source) != 12:
        raise StageBContractError("source pin roles changed")
    _validate_pin_map(source, roles=set(source), name="source")
    if config.get("implementation_roles") != IMPLEMENTATION_ROLES:
        raise StageBContractError("Stage-B implementation roles changed")

    expected_paths = {
        "config": CONFIG_RELATIVE,
        "output": "artifacts/p2_architecture_matched_stage_b_v3",
        "control": "artifacts/p2_architecture_matched_stage_b_v3_control",
        "pre_execution_qa": (
            "artifacts/p2_architecture_matched_stage_b_v3_control/pre_execution_qa.json"
        ),
        "authorization": (
            "artifacts/p2_architecture_matched_stage_b_v3_control/authorization.json"
        ),
        "attempt_lock": ("artifacts/p2_architecture_matched_stage_b_v3_control/attempt.lock"),
    }
    if config.get("canonical_paths") != expected_paths:
        raise StageBContractError("canonical Stage-B paths changed")

    protocol = config.get("curve_protocol")
    if not isinstance(protocol, Mapping):
        raise StageBContractError("curve protocol is missing")
    if (
        protocol.get("prefix_fractions") != [0.4, 0.55, 0.7, 0.85, 1.0]
        or protocol.get("seed_ids") != [20260823, 20260824, 20260825]
        or protocol.get("seed_aggregation") != "PREDICTION_MEAN_THEN_METRIC"
        or protocol.get("bootstrap_replicates") != 5000
        or protocol.get("bootstrap_cluster") != "KST_day"
        or protocol.get("embargo_days") != 7
        or protocol.get("reference_fresh_refit_each_prefix") is not True
        or protocol.get("challenger_fresh_refit_each_prefix") is not True
        or protocol.get("same_fold_keys_metric_postprocess") is not True
        or protocol.get("reference_100_percent_oof_sealed_before_challenger_scoring") is not True
        or protocol.get("all_45_blind_predictions_committed_before_validation_truth_read_or_merge")
        is not True
        or protocol.get("prediction_commitment_write")
        != "O_EXCL_AGGREGATE_SHA_COUNT_KEY_ORDER_ONLY"
        or protocol.get("validation_truth_load_boundary")
        != ("AFTER_O_EXCL_COMMITMENT_RELOAD_DEEP_EQUAL_THEN_REVERIFY_AFTER_TRUTH_LOAD_BEFORE_MERGE")
        or protocol.get("row_level_outputs_allowed") is not False
    ):
        raise StageBContractError("learning-curve protocol changed")
    folds = protocol.get("outer_folds")
    if not isinstance(folds, list) or [fold.get("name") for fold in folds] != [
        "outer_2024_sep_oct",
        "outer_2025_may_jun",
        "outer_2025_jul_aug",
    ]:
        raise StageBContractError("outer fold order changed")
    policy = config.get("execution_policy")
    required_policy = {
        "check_only_is_default": True,
        "stage_a_seal_verified_before_challenger_engine_import": True,
        "actual_run_requires_independent_static_qa": True,
        "actual_run_requires_separate_append_only_authorization": True,
        "qa_and_authorization_verified_before_attempt_lock": True,
        "challenger_engine_imported_after_attempt_lock": True,
        "direct_engine_entry_reruns_fresh_preflight": True,
        "fold_local_training_targets_loaded_without_validation_truth_values": True,
        "precommitment_raw_observation_bytes_limited_to_integrity_hash_and_selective_key_routing": True,
        "withheld_current_fold_target_scalar_fields_decoded_or_converted_before_commitment": False,
        "validation_target_scalar_attachment_after_commitment_only": True,
        "all_blind_predictions_before_validation_truth_read_or_merge": True,
        "prediction_commitment_reloaded_before_truth_and_reverified_after_truth_load_before_merge": True,
        "output_and_files_use_o_excl_semantics": True,
        "rerun_allowed": False,
        "resume_allowed": False,
        "candidate_or_test_prediction_allowed": False,
        "frozen_or_submission_mutation_allowed": False,
        "registry_append_allowed": False,
        "automatic_upload_allowed": False,
    }
    if policy != required_policy:
        raise StageBContractError("Stage-B execution policy changed")

    gates = config.get("metric_and_gates")
    if not isinstance(gates, Mapping) or any(
        gates.get(key) != expected
        for key, expected in {
            "official_layer_counts": {"2": 8713, "3": 8712, "4": 8636},
            "late_fractions_all_improve": [0.7, 0.85, 1.0],
            "full_and_one_other_late_ci90_exclude_zero": True,
            "full_delta_candidate_minus_reference_at_most_c": -0.03,
            "minimum_improved_folds": 2,
            "critical_slices": ["layer_2", "layer_3", "layer_4", "2024_sep_oct"],
            "maximum_each_critical_slice_regression_c": 0.0075,
            "all_leakage_checks_required": True,
            "all_reproducibility_checks_required": True,
        }.items()
    ):
        raise StageBContractError("numeric gates changed")
    output = config.get("output_contract")
    if (
        not isinstance(output, Mapping)
        or output.get("aggregate_only") is not True
        or output.get("full_fit_performed") is not False
        or output.get("candidate_generated") is not False
        or output.get("test_prediction_generated") is not False
        or output.get("upload_performed") is not False
        or set(output.get("artifacts", {}))
        != {
            "prediction_commitment",
            "learning_curve_evidence",
            "gate_decision",
            "training_receipt",
            "manifest",
            "seal",
        }
    ):
        raise StageBContractError("aggregate-only output firewall changed")
    qa = config.get("qa_receipt_contract")
    if (
        not isinstance(qa, Mapping)
        or qa.get("decision") != "GO_AUTHORIZE_P2_STAGE_B_CHALLENGER_V3"
        or qa.get("required_p0_count") != 0
        or qa.get("required_p1_count") != 0
        or qa.get("must_pin_exact_roles") != list(IMPLEMENTATION_ROLES)
    ):
        raise StageBContractError("independent QA contract changed")


def load_canonical_config(
    root: Path,
    requested_path: Path | None = None,
    *,
    supplied_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    canonical = workspace_path(root, CONFIG_RELATIVE)
    requested = (requested_path or canonical).resolve(strict=True)
    if requested != canonical:
        raise StageBContractError("only the canonical Stage-B v3 config path is accepted")
    if sha256_file(canonical) != CONFIG_SHA256:
        raise StageBContractError("canonical Stage-B v3 config SHA mismatch")
    config = strict_json_object(canonical)
    validate_config(config)
    if supplied_config is not None and dict(supplied_config) != config:
        raise StageBContractError("supplied Stage-B config fails full deep equality")
    return config


def _pin(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def implementation_pins(root: Path) -> dict[str, dict[str, Any]]:
    workspace = root.resolve(strict=True)
    return {
        role: _pin(workspace_path(workspace, relative), workspace)
        for role, relative in IMPLEMENTATION_ROLES.items()
    }


def verify_pin_map(
    root: Path,
    pins: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    workspace = root.resolve(strict=True)
    result: dict[str, dict[str, Any]] = {}
    for role, expected in pins.items():
        path = workspace_path(workspace, str(expected["path"]))
        observed = _pin(path, workspace)
        if observed != dict(expected):
            raise StageBContractError(f"{label} pin drift: {role}")
        result[str(role)] = observed
    return result


def stage_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    paths = config["canonical_paths"]
    return {
        key: workspace_path(root, paths[key], must_exist=False)
        for key in ("output", "control", "pre_execution_qa", "authorization", "attempt_lock")
    }


def verify_stage_a_reference(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Deeply bind the completed v3 reference before challenger code is imported."""

    workspace = root.resolve(strict=True)
    pins = verify_pin_map(
        workspace,
        config["stage_a_reference"]["artifacts"],
        label="Stage-A reference",
    )
    stage_a_config = stage_a_guard.load_canonical_config(workspace)
    stage_a_guard.verify_stage_a_seal(workspace, stage_a_config)
    seal = strict_json_object(workspace_path(workspace, pins["SEAL"]["path"]))
    manifest = strict_json_object(workspace_path(workspace, pins["MANIFEST"]["path"]))
    metrics = strict_json_object(workspace_path(workspace, pins["CURVE_METRICS"]["path"]))
    receipt = strict_json_object(workspace_path(workspace, pins["TRAINING_RECEIPT"]["path"]))
    expected = config["stage_a_reference"]["expected"]
    checks = {
        "seal_schema": seal.get("schema_version") == "p2_architecture_matched_reference.seal.v3",
        "seal_complete": seal.get("complete") is expected["complete"],
        "seal_prefixes": seal.get("all_five_prefixes_sealed")
        is expected["all_five_prefixes_sealed"],
        "seal_challenger_zero": seal.get("challenger_import_fit_or_score_count_before_seal")
        == expected["challenger_fit_or_score_count_before_seal"],
        "seal_non_exact": seal.get("exact_official_incumbent_comparison") is False,
        "seal_no_promotion": seal.get("official_promotion_allowed") is False,
        "seal_no_upload": seal.get("upload_count") == expected["uploads"],
        "seal_manifest": seal.get("manifest")
        == {
            "path": "manifest.json",
            "sha256": pins["MANIFEST"]["sha256"],
            "bytes": pins["MANIFEST"]["bytes"],
        },
        "manifest_schema": manifest.get("schema_version")
        == "p2_architecture_matched_reference.manifest.v3",
        "manifest_no_challenger": manifest.get("challenger_import_fit_or_score_count") == 0,
        "manifest_no_upload": manifest.get("uploads") == 0,
        "receipt_cells": len(receipt.get("cells", [])) == expected["cells"],
        "receipt_deep_jobs": receipt.get("plan", {}).get("deep_training_jobs")
        == expected["deep_training_jobs"],
        "receipt_router_jobs": receipt.get("plan", {}).get("router_training_jobs")
        == expected["router_training_jobs"],
        "receipt_no_submission": receipt.get("plan", {}).get("submission_predictions")
        == expected["submission_predictions"],
        "metrics_full": abs(
            float(metrics.get("points", [{}])[-1].get("prediction_mean_metric", float("nan")))
            - expected["full_reference_rmse_c"]
        )
        <= 1e-15,
    }
    sealed_oof = seal.get("reference_oof_by_fraction")
    if not isinstance(sealed_oof, Mapping) or set(sealed_oof) != set(FRACTION_ROLES):
        checks["seal_oof_roles"] = False
    else:
        checks["seal_oof_roles"] = all(
            sealed_oof[fraction]
            == {
                "path": Path(pins[role]["path"]).name,
                "sha256": pins[role]["sha256"],
                "bytes": pins[role]["bytes"],
            }
            for fraction, role in FRACTION_ROLES.items()
        )
    if not all(checks.values()):
        raise StageBContractError(
            "Stage-A v3 binding failed: "
            f"{sorted(key for key, value in checks.items() if not value)}"
        )
    return {
        "status": "PASS_EXACT_STAGE_A_V3_REFERENCE_BINDING",
        "pins": pins,
        "full_reference_rmse_c": expected["full_reference_rmse_c"],
        "cells": expected["cells"],
        "deep_training_jobs": expected["deep_training_jobs"],
        "router_training_jobs": expected["router_training_jobs"],
        "challenger_fit_or_score_count_before_seal": 0,
        "uploads": 0,
    }


def _loaded_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in NUMERICAL_PREFIXES)
    )


def static_preflight(
    root: Path,
    data_dir: Path,
    *,
    requested_config: Path | None = None,
    supplied_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    workspace = root.resolve(strict=True)
    before = set(_loaded_modules())
    config = load_canonical_config(
        workspace,
        requested_config,
        supplied_config=supplied_config,
    )
    source = verify_pin_map(workspace, config["source_pins"], label="source")
    reference = verify_stage_a_reference(workspace, config)
    stage_a_preflight = stage_a_guard.static_preflight(workspace, data_dir)
    after = set(_loaded_modules())
    paths = stage_paths(workspace, config)
    return {
        "schema_version": "p2_architecture_matched_stage_b.static_preflight.v3",
        "status": "PASS_STATIC_IMPLEMENTATION_ONLY_STAGE_A_SEALED",
        "problem": "P2",
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "canonical_config": {
            "path": CONFIG_RELATIVE,
            "sha256": CONFIG_SHA256,
            "bytes": workspace_path(workspace, CONFIG_RELATIVE).stat().st_size,
        },
        "implementation_pins": implementation_pins(workspace),
        "source_pins": source,
        "stage_a_reference": reference,
        "schema_and_keys": stage_a_preflight["schema_and_keys"],
        "runtime_probe": stage_a_preflight["runtime_probe"],
        "stage_a_sealed_before_challenger_import_fit_or_score": True,
        "preflight_process_loaded_numerical_modules": sorted(after),
        "preflight_process_new_numerical_modules": sorted(after - before),
        "challenger_engine_imported": ENGINE_MODULE in sys.modules,
        "canonical_path_state": {key: path.exists() for key, path in paths.items()},
        "files_written": 0,
        "attempt_locks_created": 0,
        "challenger_fits": 0,
        "challenger_predictions": 0,
        "test_predictions": 0,
        "uploads": 0,
        "resource_estimate": config["resource_estimate"],
    }


def _qa_keys() -> set[str]:
    return {
        "schema_version",
        "created_at_kst",
        "decision",
        "p0_count",
        "p1_count",
        "config",
        "stage_a_reference_pins",
        "source_pins",
        "implementation_pins",
        "reviewer",
        "notes",
    }


def verify_pre_execution_qa(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    path = stage_paths(root, config)["pre_execution_qa"]
    if not path.is_file():
        raise PermissionError("independent Stage-B static QA receipt is missing")
    receipt = strict_json_object(path)
    _require_exact_keys(receipt, _qa_keys(), name="Stage-B QA receipt")
    contract = config["qa_receipt_contract"]
    checks = {
        "schema": receipt.get("schema_version") == contract["schema_version"],
        "decision": receipt.get("decision") == contract["decision"],
        "p0": receipt.get("p0_count") == 0,
        "p1": receipt.get("p1_count") == 0,
        "config": receipt.get("config") == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "reference": receipt.get("stage_a_reference_pins")
        == config["stage_a_reference"]["artifacts"],
        "source": receipt.get("source_pins") == config["source_pins"],
        "implementation": receipt.get("implementation_pins") == implementation_pins(root),
        "roles": set(receipt.get("implementation_pins", {})) == set(IMPLEMENTATION_ROLES),
        "reviewer": bool(receipt.get("reviewer")),
        "notes": isinstance(receipt.get("notes"), list),
    }
    if not all(checks.values()):
        raise PermissionError(
            f"Stage-B QA failed: {sorted(key for key, value in checks.items() if not value)}"
        )
    return receipt, sha256_file(path)


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
        raise FileExistsError("append-only Stage-B output already exists")
    if require_unconsumed and paths["attempt_lock"].exists():
        raise FileExistsError("one-shot Stage-B attempt was already consumed")
    if not paths["authorization"].is_file():
        raise PermissionError("separate Stage-B authorization is missing")
    authorization = strict_json_object(paths["authorization"])
    expected_keys = {
        "schema_version",
        "created_at_kst",
        "stage",
        "config",
        "authorization",
        "user_message_reference",
        "qa_receipt",
        "stage_a_seal",
        "implementation_pins",
        "execution_authorized",
        "candidate_or_test_prediction_allowed",
        "upload_allowed",
    }
    _require_exact_keys(authorization, expected_keys, name="Stage-B authorization")
    contract = config["authorization_contract"]
    checks = {
        "schema": authorization.get("schema_version") == contract["schema_version"],
        "stage": authorization.get("stage") == STAGE,
        "config": authorization.get("config") == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "phrase": authorization.get("authorization")
        == contract["authorization_phrase_prefix"] + CONFIG_SHA256,
        "user_reference": bool(authorization.get("user_message_reference")),
        "qa": authorization.get("qa_receipt")
        == {
            "path": config["canonical_paths"]["pre_execution_qa"],
            "sha256": qa_sha256,
        },
        "stage_a": authorization.get("stage_a_seal")
        == config["stage_a_reference"]["artifacts"]["SEAL"],
        "implementation": authorization.get("implementation_pins") == implementation_pins(root),
        "execution": authorization.get("execution_authorized") is True,
        "candidate": authorization.get("candidate_or_test_prediction_allowed") is False,
        "upload": authorization.get("upload_allowed") is False,
    }
    if not all(checks.values()):
        raise PermissionError(
            "Stage-B authorization failed: "
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
        "schema_version": "p2_architecture_matched_stage_b.attempt_lock.v3",
        "stage": STAGE,
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "qa_receipt_sha256": qa_sha256,
        "authorization_sha256": authorization_sha256,
        "stage_a_seal": config["stage_a_reference"]["artifacts"]["SEAL"],
        "implementation_pins": implementation_pins(root),
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "candidate_or_test_prediction_allowed": False,
        "rerun_allowed": False,
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
        raise FileExistsError("append-only Stage-B output already exists")
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
        raise PermissionError("one-shot Stage-B attempt lock is missing")
    observed = strict_json_object(path)
    expected = _lock_payload(
        root,
        config,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
    )
    if observed != expected:
        raise PermissionError("Stage-B attempt lock fails full deep equality")
    return observed


def verify_stage_b_seal(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    output = stage_paths(root, config)["output"].resolve(strict=True)
    artifacts = config["output_contract"]["artifacts"]
    paths = {role: contained_path(output, relative) for role, relative in artifacts.items()}
    if any(not path.is_file() for path in paths.values()):
        raise StageBContractError("a required Stage-B artifact is missing")
    expected_children = {path.name for path in paths.values()}
    observed_children = {path.name for path in output.iterdir()}
    if observed_children != expected_children or any(
        not path.is_file() for path in output.iterdir()
    ):
        raise StageBContractError("Stage-B output contains an unregistered artifact")
    manifest = strict_json_object(paths["manifest"])
    seal = strict_json_object(paths["seal"])
    commitment = strict_json_object(paths["prediction_commitment"])
    evidence = strict_json_object(paths["learning_curve_evidence"])
    decision = strict_json_object(paths["gate_decision"])
    receipt = strict_json_object(paths["training_receipt"])
    leakage_checks = evidence.get("leakage_checks")
    reproducibility_checks = evidence.get("reproducibility_checks")
    fold_blind_audits = receipt.get("fold_blind_input_audits")
    digest_roles = (
        "key_order_sha256",
        "prediction_values_sha256",
        "cell_prediction_sha256",
        "combined_prediction_commitment_sha256",
    )
    commitment_digests_valid = all(_is_sha256(commitment.get(key)) for key in digest_roles)
    commitment_combined_valid = False
    if commitment_digests_valid:
        commitment_combined_valid = (
            hashlib.sha256(
                bytes.fromhex(commitment["key_order_sha256"])
                + bytes.fromhex(commitment["prediction_values_sha256"])
                + bytes.fromhex(commitment["cell_prediction_sha256"])
            ).hexdigest()
            == commitment["combined_prediction_commitment_sha256"]
        )
    checks = {
        "manifest_schema": manifest.get("schema_version")
        == "p2_architecture_matched_stage_b.manifest.v3",
        "manifest_config": manifest.get("config")
        == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "manifest_implementation": manifest.get("implementation_pins") == implementation_pins(root),
        "manifest_aggregate": manifest.get("aggregate_only") is True,
        "manifest_no_candidate": manifest.get("candidate_generated") is False,
        "manifest_no_test": manifest.get("test_prediction_generated") is False,
        "manifest_no_upload": manifest.get("uploads") == 0,
        "manifest_artifact_roles": set(manifest.get("artifacts", {}))
        == {
            "prediction_commitment",
            "learning_curve_evidence",
            "gate_decision",
            "training_receipt",
        },
        "manifest_commitment": manifest.get("prediction_commitment")
        == _pin(paths["prediction_commitment"], output),
        "commitment_schema": commitment.get("schema_version")
        == "p2_architecture_matched_stage_b.prediction_commitment.v3",
        "commitment_exact_keys": set(commitment)
        == {
            "schema_version",
            "stage",
            "config",
            "implementation_pins",
            "stage_a_seal",
            "aggregate_only",
            "row_level_predictions_persisted",
            "truth_columns_present",
            "validation_truth_read_or_merged_before_commitment",
            "prefix_fractions_in_order",
            "key_columns_in_order",
            "prediction_columns_in_order",
            "rows_by_fraction",
            "total_rows",
            "cell_prediction_count",
            "key_order_sha256",
            "prediction_values_sha256",
            "cell_prediction_sha256",
            "combined_prediction_commitment_sha256",
            "candidate_generated",
            "test_prediction_generated",
            "uploads",
        },
        "commitment_stage": commitment.get("stage")
        == "ALL_45_BLIND_PREDICTIONS_BEFORE_VALIDATION_TRUTH",
        "commitment_config": commitment.get("config")
        == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "commitment_implementation": commitment.get("implementation_pins")
        == implementation_pins(root),
        "commitment_stage_a": commitment.get("stage_a_seal")
        == config["stage_a_reference"]["artifacts"]["SEAL"],
        "commitment_aggregate": commitment.get("aggregate_only") is True,
        "commitment_no_rows": commitment.get("row_level_predictions_persisted") is False,
        "commitment_no_truth": commitment.get("truth_columns_present") is False,
        "commitment_before_truth": commitment.get(
            "validation_truth_read_or_merged_before_commitment"
        )
        is False,
        "commitment_cells": commitment.get("cell_prediction_count") == 45,
        "commitment_rows": commitment.get("total_rows") == 390780,
        "commitment_rows_by_fraction": commitment.get("rows_by_fraction")
        == {"0.4": 78156, "0.55": 78156, "0.7": 78156, "0.85": 78156, "1.0": 78156},
        "commitment_fractions": commitment.get("prefix_fractions_in_order")
        == [0.4, 0.55, 0.7, 0.85, 1.0],
        "commitment_key_order": commitment.get("key_columns_in_order")
        == ["fraction", "fold", "station", "layer", "time"],
        "commitment_prediction_order": commitment.get("prediction_columns_in_order")
        == [
            "challenger_seed_20260823",
            "challenger_seed_20260824",
            "challenger_seed_20260825",
            "challenger_mean",
        ],
        "commitment_digests": commitment_digests_valid,
        "commitment_combined_digest": commitment_combined_valid,
        "commitment_no_candidate": commitment.get("candidate_generated") is False,
        "commitment_no_test": commitment.get("test_prediction_generated") is False,
        "commitment_no_upload": commitment.get("uploads") == 0,
        "evidence_schema": evidence.get("schema_version")
        == "p2_architecture_matched_stage_b.learning_curve_evidence.v3",
        "evidence_commitment": evidence.get("prediction_commitment")
        == _pin(paths["prediction_commitment"], output),
        "evidence_leakage_checks": isinstance(leakage_checks, Mapping)
        and bool(leakage_checks)
        and all(value is True for value in leakage_checks.values()),
        "evidence_reproducibility_checks": isinstance(reproducibility_checks, Mapping)
        and bool(reproducibility_checks)
        and all(value is True for value in reproducibility_checks.values()),
        "evidence_firewall": evidence.get("output_firewall")
        == {
            "aggregate_only": True,
            "full_fit_performed": False,
            "candidate_generated": False,
            "test_prediction_generated": False,
            "upload_performed": False,
            "official_submission_count": 0,
        },
        "decision_schema": decision.get("schema_version")
        == "p2_architecture_matched_stage_b.gate_decision.v3",
        "decision_never_promotes": decision.get("passed") is False,
        "decision_non_exact": decision.get("exact_official_incumbent_comparison") is False,
        "decision_no_candidate": decision.get("candidate_generated") is False,
        "decision_no_test": decision.get("test_prediction_generated") is False,
        "decision_no_upload": decision.get("upload_performed") is False,
        "receipt_schema": receipt.get("schema_version")
        == "p2_architecture_matched_stage_b.training_receipt.v3",
        "receipt_cells": len(receipt.get("cells", [])) == 45,
        "receipt_commitment": {
            key: receipt.get("prediction_commitment", {}).get(key)
            for key in ("path", "sha256", "bytes")
        }
        == _pin(paths["prediction_commitment"], output),
        "receipt_commitment_digest": receipt.get("prediction_commitment", {}).get(
            "combined_prediction_commitment_sha256"
        )
        == commitment.get("combined_prediction_commitment_sha256"),
        "receipt_commitment_before_truth": receipt.get("prediction_commitment", {}).get(
            "persisted_before_validation_truth_access"
        )
        is True,
        "receipt_commitment_after_truth": receipt.get("prediction_commitment", {}).get(
            "reverified_after_truth_load_before_merge"
        )
        is True,
        "receipt_fold_blind_audits": isinstance(fold_blind_audits, Mapping)
        and set(fold_blind_audits)
        == {"outer_2024_sep_oct", "outer_2025_may_jun", "outer_2025_jul_aug"}
        and all(
            audit.get("withheld_target_scalar_fields_decoded_or_converted") == 0
            and audit.get("validation_target_temp_psal_strings_converted") == 0
            and audit.get("validation_truth_columns_read_by_challenger") == 0
            and audit.get("raw_source_bytes_preflight_hashed_for_integrity_only") is True
            for audit in fold_blind_audits.values()
        ),
        "receipt_truth_access": receipt.get("validation_truth_access_audit", {}).get(
            "hidden_test_target_scalars_converted"
        )
        == 0
        and receipt.get("validation_truth_access_audit", {}).get(
            "nonvalidation_target_scalars_converted"
        )
        == 0,
        "receipt_blind_first": receipt.get("guard_summary", {}).get(
            "all_45_predictions_before_validation_truth_access"
        )
        is True,
        "receipt_commitment_reverified": receipt.get("guard_summary", {}).get(
            "commitment_reverified_after_truth_load_before_merge"
        )
        is True,
        "receipt_no_withheld_decode": receipt.get("guard_summary", {}).get(
            "withheld_current_fold_target_scalars_never_decoded_or_converted"
        )
        is True,
        "receipt_no_row_predictions": receipt.get("guard_summary", {}).get(
            "row_level_predictions_persisted"
        )
        is False,
        "receipt_fits": receipt.get("guard_summary", {}).get("challenger_fit_count") == 45,
        "receipt_no_full_fit": receipt.get("guard_summary", {}).get("full_fit_count") == 0,
        "receipt_no_candidate": receipt.get("guard_summary", {}).get("candidate_count") == 0,
        "receipt_no_test": receipt.get("guard_summary", {}).get("test_prediction_count") == 0,
        "receipt_no_upload": receipt.get("guard_summary", {}).get("upload_count") == 0,
        "seal_schema": seal.get("schema_version") == "p2_architecture_matched_stage_b.seal.v3",
        "seal_complete": seal.get("complete") is True,
        "seal_manifest": seal.get("manifest")
        == {
            "path": "manifest.json",
            "sha256": sha256_file(paths["manifest"]),
            "bytes": paths["manifest"].stat().st_size,
        },
        "seal_commitment": seal.get("prediction_commitment")
        == _pin(paths["prediction_commitment"], output),
        "seal_no_candidate": seal.get("candidate_generated") is False,
        "seal_no_test": seal.get("test_prediction_generated") is False,
        "seal_no_upload": seal.get("upload_count") == 0,
    }
    if not all(checks.values()):
        raise StageBContractError(
            f"Stage-B seal failed: {sorted(key for key, value in checks.items() if not value)}"
        )
    for role in (
        "prediction_commitment",
        "learning_curve_evidence",
        "gate_decision",
        "training_receipt",
    ):
        expected = manifest.get("artifacts", {}).get(role)
        if not isinstance(expected, Mapping):
            raise StageBContractError(f"Stage-B manifest lacks artifact: {role}")
        if _pin(paths[role], output) != dict(expected):
            raise StageBContractError(f"Stage-B artifact drift: {role}")
    return {
        "status": "PASS_SEALED_STAGE_B_V3",
        "manifest_sha256": sha256_file(paths["manifest"]),
        "seal_sha256": sha256_file(paths["seal"]),
        "local_qualification": bool(seal.get("local_qualification")),
        "candidate_generated": False,
        "test_prediction_generated": False,
        "uploads": 0,
    }


__all__ = [
    "CONFIG_RELATIVE",
    "CONFIG_SHA256",
    "ENGINE_MODULE",
    "ENGINE_RELATIVE",
    "FRACTION_ROLES",
    "IMPLEMENTATION_ROLES",
    "MODE",
    "STAGE",
    "StageBContractError",
    "canonical_json_bytes",
    "consume_attempt_lock",
    "contained_path",
    "exclusive_bytes",
    "exclusive_json",
    "implementation_pins",
    "load_canonical_config",
    "sha256_file",
    "stage_paths",
    "static_preflight",
    "strict_json_object",
    "validate_config",
    "verify_consumed_attempt_lock",
    "verify_execution_authorization",
    "verify_pin_map",
    "verify_pre_execution_qa",
    "verify_stage_a_reference",
    "verify_stage_b_seal",
    "workspace_path",
]
