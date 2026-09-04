"""Fail-closed contract for executable P2 architecture-matched Stage A v2.

This module is deliberately limited to JSON/SHA/path/schema guards.  It never
imports numerical training or prediction modules.  The runner may import the
execution engine only after independent QA, authorization, and an O_EXCL
attempt lock have all succeeded.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from p2_restore import architecture_matched_prefix_refit as design_v1

MODE = "ARCHITECTURE_MATCHED_TIME_SAFE_BASELINE"
STAGE = "STAGE_A_REFERENCE"
CONFIG_RELATIVE = "configs/experiments/p2_architecture_matched_stage_a_execution_v2.json"
CONFIG_SHA256 = "cfdd5432d6513d69c63307068dd15ddd0d347ffea175769a523fd613b1083386"
GUARD_RELATIVE = "src/p2_restore/architecture_matched_stage_a_contract_v2.py"
ENGINE_RELATIVE = "src/p2_restore/architecture_matched_stage_a_execution_v2.py"
RUNNER_RELATIVE = "scripts/run_p2_architecture_matched_reference_v2.py"
TESTS_RELATIVE = "tests/test_p2_architecture_matched_stage_a_execution_v2.py"
PREFIX_FRACTIONS = (0.4, 0.55, 0.7, 0.85, 1.0)
PIPELINE_SEEDS = (20260823, 20260824, 20260825)
OUTER_FOLDS = (
    "outer_2024_sep_oct",
    "outer_2025_may_jun",
    "outer_2025_jul_aug",
)
CONTRIBUTORS = (
    "router_400",
    "depth_query_bitcn",
    "lsti_style",
    "timemixerpp_style",
    "moment_units_scratch",
)
IMPLEMENTATION_ROLES = {
    "CONFIG": CONFIG_RELATIVE,
    "GUARD": GUARD_RELATIVE,
    "ENGINE": ENGINE_RELATIVE,
    "RUNNER": RUNNER_RELATIVE,
    "TESTS": TESTS_RELATIVE,
}


class StageAContractError(ValueError):
    """Raised before execution if an immutable Stage-A invariant differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_constant(value: str) -> None:
    raise StageAContractError(f"non-finite JSON constant is forbidden: {value}")


def _pairs_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StageAContractError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def strict_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageAContractError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise StageAContractError(f"expected a JSON object: {path}")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def workspace_path(root: Path, relative: str | Path, *, must_exist: bool = True) -> Path:
    workspace = root.resolve(strict=True)
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise StageAContractError("path must be workspace-relative and non-traversing")
    resolved = (workspace / candidate).resolve(strict=must_exist)
    if not resolved.is_relative_to(workspace):
        raise StageAContractError("path escapes workspace")
    return resolved


def contained_path(output_root: Path, child: str | Path) -> Path:
    base = output_root.resolve(strict=False)
    candidate = Path(child)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise StageAContractError("output child path is unsafe")
    target = (base / candidate).resolve(strict=False)
    if target == base or not target.is_relative_to(base):
        raise StageAContractError("output child escapes canonical output")
    return target


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise StageAContractError(f"{name} keys changed")


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "p2_architecture_matched_stage_a_execution.v2":
        raise StageAContractError("config schema changed")
    if config.get("problem") != "P2" or config.get("comparison_mode") != MODE:
        raise StageAContractError("architecture-matched mode is P2-only")
    required_false = (
        "exact_official_incumbent_comparison",
        "official_promotion_allowed",
        "upload_allowed",
    )
    if any(config.get(key) is not False for key in required_false):
        raise StageAContractError("non-exact/research-only status changed")
    if config.get("explicitly_not_exact_official_incumbent") is not True:
        raise StageAContractError("explicit non-exact label is required")
    if config.get("local_qualification_only") is not True or config.get("research_only") is not True:
        raise StageAContractError("local-only status changed")
    if config.get("official_submission_count") != 0:
        raise StageAContractError("official submission state changed")
    if config.get("preregistration") != {
        "generation_id": "P2_ARCHITECTURE_MATCHED_STAGE_A_EXECUTION_V2",
        "created_before_first_fit": True,
        "score_derived_tuning": False,
        "challenger_hypothesis_count": 0,
        "challenger_import_fit_or_score_count": 0,
    }:
        raise StageAContractError("Stage-A preregistration changed")

    expected_paths = {
        "config": CONFIG_RELATIVE,
        "output": "artifacts/p2_architecture_matched_reference_v2",
        "control": "artifacts/p2_architecture_matched_reference_v2_control",
        "pre_execution_qa": (
            "artifacts/p2_architecture_matched_reference_v2_control/pre_execution_qa.json"
        ),
        "authorization": (
            "artifacts/p2_architecture_matched_reference_v2_control/authorization.json"
        ),
        "attempt_lock": (
            "artifacts/p2_architecture_matched_reference_v2_control/attempt.lock"
        ),
    }
    if config.get("canonical_paths") != expected_paths:
        raise StageAContractError("canonical paths changed")

    policy = config.get("execution_policy")
    if not isinstance(policy, Mapping):
        raise StageAContractError("execution policy is missing")
    for key in (
        "check_only_is_default",
        "check_only_must_not_import_training_or_prediction_modules",
        "actual_run_requires_independent_qa_receipt",
        "actual_run_requires_separate_append_only_authorization",
        "qa_and_authorization_verified_before_attempt_lock",
        "execution_engine_imported_after_attempt_lock",
        "output_directory_created_with_o_excl_semantics",
        "all_materialized_files_created_o_excl",
    ):
        if policy.get(key) is not True:
            raise StageAContractError(f"required execution guard changed: {key}")
    for key in (
        "rerun_allowed",
        "resume_allowed",
        "frozen_mutation_allowed",
        "submission_mutation_allowed",
        "registry_append_allowed",
        "automatic_upload_allowed",
    ):
        if policy.get(key) is not False:
            raise StageAContractError(f"forbidden execution capability changed: {key}")

    if config.get("runtime_contract") != {
        "python": "3.12.10",
        "numpy": "2.3.5",
        "pandas": "3.0.1",
        "scipy": "1.18.0",
        "scikit_learn": "1.9.0",
        "lightgbm": "4.7.0",
        "torch": "2.13.0+cu130",
        "torch_cuda": "13.0",
        "cuda_available": True,
        "gpu_name_contains": "NVIDIA GeForce RTX 5090",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }:
        raise StageAContractError("runtime reproducibility contract changed")

    architecture = config.get("architecture_reference")
    if not isinstance(architecture, Mapping):
        raise StageAContractError("architecture reference is missing")
    if architecture.get("contributors_in_order") != list(CONTRIBUTORS):
        raise StageAContractError("contributor order changed")
    if architecture.get("router_layer_arms") != {"2": "phase", "3": "phase", "4": "state"}:
        raise StageAContractError("router arms changed")
    if architecture.get("gate_route_layers") != [2, 4]:
        raise StageAContractError("gate route layers changed")
    if architecture.get("layer_extrapolation_factors") != {
        "2": 10.0,
        "3": 0.0,
        "4": 2.0,
    }:
        raise StageAContractError("extrapolation factors changed")
    pins = architecture.get("existing_source_pins")
    if not isinstance(pins, Mapping) or not pins or any(not _is_sha256(v) for v in pins.values()):
        raise StageAContractError("architecture source pins are invalid")

    recipe = config.get("training_recipe")
    if not isinstance(recipe, Mapping):
        raise StageAContractError("training recipe is missing")
    if recipe.get("complete_pipeline_seed_ids") != list(PIPELINE_SEEDS):
        raise StageAContractError("pipeline seeds changed")
    if recipe.get("prefix_fractions") != list(PREFIX_FRACTIONS):
        raise StageAContractError("prefix fractions changed")
    if recipe.get("embargo_days") != 7:
        raise StageAContractError("embargo changed")
    folds = recipe.get("outer_folds")
    if not isinstance(folds, Sequence) or [fold.get("name") for fold in folds] != list(OUTER_FOLDS):
        raise StageAContractError("outer fold order changed")
    inner = recipe.get("inner_oof")
    if not isinstance(inner, Mapping) or inner.get("validation_fraction_edges") != [
        0.55,
        0.7,
        0.85,
        1.0,
    ]:
        raise StageAContractError("inner fold edges changed")
    if inner.get("future_or_outer_target_labels_allowed") is not False:
        raise StageAContractError("future target labels became allowed")
    epoch = recipe.get("epoch_selection")
    if not isinstance(epoch, Mapping) or epoch.get("epoch_grid") != [12, 20, 28, 36, 44, 52]:
        raise StageAContractError("epoch grid changed")
    deep = recipe.get("deep_training")
    expected_components = {
        "depth_query_bitcn",
        "lsti_style",
        "timemixerpp_style",
        "moment_units_scratch",
    }
    if not isinstance(deep, Mapping) or set(deep.get("components", {})) != expected_components:
        raise StageAContractError("deep component recipe changed")
    meta = recipe.get("meta_training")
    if (
        not isinstance(meta, Mapping)
        or meta.get("contributors_in_order") != list(CONTRIBUTORS)
        or meta.get("gate_regularization") != 10.0
        or meta.get("frozen_stack_or_gate_reuse_allowed") is not False
    ):
        raise StageAContractError("prefix-local meta recipe changed")
    joint = recipe.get("joint_target_mask")
    if (
        not isinstance(joint, Mapping)
        or joint.get("columns") != ["temp", "psal"]
        or joint.get("layers") != [2, 3, 4]
        or joint.get("target_layer_values_in_model_inputs") is not False
        or joint.get(
            "applied_before_normalizer_feature_fit_router_fit_deep_fit_meta_fit_and_metric"
        )
        is not True
    ):
        raise StageAContractError("joint target-mask contract changed")

    reference = config.get("stage_a_reference_contract")
    if not isinstance(reference, Mapping):
        raise StageAContractError("Stage-A reference contract is missing")
    if reference.get(
        "generate_all_five_prefix_oof_before_any_challenger_import_fit_or_score"
    ) is not True or reference.get("row_predictions_may_contain_targets") is not False:
        raise StageAContractError("seal-before-challenger contract changed")
    expected_columns = [
        "fold",
        "station",
        "layer",
        "time",
        *[f"seed_{seed}" for seed in PIPELINE_SEEDS],
        "prediction_mean",
    ]
    if reference.get("reference_oof_columns") != expected_columns:
        raise StageAContractError("target-free OOF schema changed")

    qa = config.get("qa_receipt_contract")
    if (
        not isinstance(qa, Mapping)
        or qa.get("decision") != "GO_AUTHORIZE_P2_STAGE_A_EXECUTION_V2"
        or qa.get("required_p0_count") != 0
        or qa.get("required_p1_count") != 0
        or qa.get("must_pin_exact_roles") != list(IMPLEMENTATION_ROLES)
    ):
        raise StageAContractError("independent QA contract changed")
    promotion = config.get("official_promotion_contract")
    if (
        not isinstance(promotion, Mapping)
        or promotion.get("actual_immutable_incumbent_csv")
        != {
            "path": "output/2026-08-20/ready/P2_submission.csv",
            "sha256": "1c959f818737850fd7fa9c6609ba3ae49dc9a470a269f7313119d840df1736bf",
        }
        or promotion.get("stage_a_architecture_matched_curve_can_promote") is not False
        or promotion.get("requires_distinct_official_paired_ab_receipts") is not True
        or promotion.get("requires_same_official_scoring_version_and_split") is not True
        or promotion.get("minimum_raw_integrated_rmse_improvement_c") != 0.03
        or promotion.get("explicit_user_approval_required_for_each_upload") is not True
        or promotion.get("team_wide_daily_upload_limit") != 3
        or promotion.get("official_uploads_now") != 0
    ):
        raise StageAContractError("official promotion firewall changed")


def load_canonical_config(
    root: Path,
    requested_path: Path | None = None,
    *,
    supplied_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    canonical = workspace_path(root, CONFIG_RELATIVE)
    requested = (requested_path or canonical).resolve(strict=True)
    if requested != canonical:
        raise StageAContractError("only the canonical v2 config path is accepted")
    if sha256_file(canonical) != CONFIG_SHA256:
        raise StageAContractError("canonical v2 config SHA mismatch")
    config = strict_json_object(canonical)
    validate_config(config)
    if supplied_config is not None and dict(supplied_config) != config:
        raise StageAContractError("supplied config fails full deep equality")
    return config


def _verify_pin_group(root: Path, pins: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for relative, expected in pins.items():
        path = workspace_path(root, relative)
        observed = sha256_file(path)
        if observed != expected:
            raise StageAContractError(f"{role} SHA mismatch: {relative}")
        verified[str(relative)] = {"sha256": observed, "bytes": path.stat().st_size}
    return verified


def implementation_pins(root: Path) -> dict[str, dict[str, Any]]:
    return {
        role: {
            "path": relative,
            "sha256": sha256_file(workspace_path(root, relative)),
            "bytes": workspace_path(root, relative).stat().st_size,
        }
        for role, relative in IMPLEMENTATION_ROLES.items()
    }


def stage_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    canonical = config["canonical_paths"]
    return {
        key: workspace_path(root, canonical[key], must_exist=False)
        for key in ("output", "control", "pre_execution_qa", "authorization", "attempt_lock")
    }


def static_preflight(
    root: Path,
    data_dir: Path,
    *,
    requested_config: Path | None = None,
    supplied_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_canonical_config(
        root,
        requested_config,
        supplied_config=supplied_config,
    )
    predecessor_entries = config["predecessor_pins"]
    predecessor = _verify_pin_group(
        root,
        {
            entry["path"]: entry["sha256"]
            for entry in predecessor_entries.values()
        },
        role="predecessor",
    )
    source = _verify_pin_group(
        root,
        config["architecture_reference"]["existing_source_pins"],
        role="architecture source",
    )
    design = design_v1.load_canonical_config(root)
    deployed = design_v1.verify_deployed_graph(root, design)
    keys = design_v1.inspect_schema_and_keys(data_dir, config)
    paths = stage_paths(root, config)
    return {
        "schema_version": "p2_architecture_matched_stage_a_execution.static_preflight.v2",
        "status": "PASS_STATIC_IMPLEMENTATION_ONLY",
        "problem": "P2",
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "canonical_config": {
            "path": CONFIG_RELATIVE,
            "sha256": CONFIG_SHA256,
            "bytes": workspace_path(root, CONFIG_RELATIVE).stat().st_size,
        },
        "implementation_pins": implementation_pins(root),
        "predecessor_pins": predecessor,
        "architecture_source_pins": source,
        "deployed_v1_graph_verified": deployed,
        "schema_and_keys": keys,
        "canonical_path_state": {key: path.exists() for key, path in paths.items()},
        "training_modules_imported": False,
        "prediction_modules_imported": False,
        "files_written": 0,
        "attempt_locks_created": 0,
        "fits": 0,
        "predictions": 0,
        "uploads": 0,
        "resource_estimate": config["resource_estimate"],
    }


def exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short exclusive write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    exclusive_bytes(path, canonical_json_bytes(value) + b"\n")


def _qa_expected_keys() -> set[str]:
    return {
        "schema_version",
        "created_at_kst",
        "decision",
        "p0_count",
        "p1_count",
        "config",
        "predecessor_pins",
        "implementation_pins",
        "reviewer",
        "notes",
    }


def verify_pre_execution_qa(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    path = stage_paths(root, config)["pre_execution_qa"]
    if not path.is_file():
        raise PermissionError("independent pre-execution QA receipt is missing")
    receipt = strict_json_object(path)
    _require_exact_keys(receipt, _qa_expected_keys(), name="QA receipt")
    contract = config["qa_receipt_contract"]
    checks = {
        "schema": receipt.get("schema_version") == contract["schema_version"],
        "decision": receipt.get("decision") == contract["decision"],
        "p0": receipt.get("p0_count") == 0,
        "p1": receipt.get("p1_count") == 0,
        "config": receipt.get("config")
        == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "predecessor": receipt.get("predecessor_pins") == config["predecessor_pins"],
        "implementation": receipt.get("implementation_pins") == implementation_pins(root),
        "reviewer": bool(receipt.get("reviewer")),
        "notes": isinstance(receipt.get("notes"), list),
    }
    if not all(checks.values()):
        raise PermissionError(
            "pre-execution QA failed: "
            f"{sorted(key for key, value in checks.items() if not value)}"
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
        raise FileExistsError("canonical append-only Stage-A output already exists")
    if require_unconsumed and paths["attempt_lock"].exists():
        raise FileExistsError("canonical one-shot Stage-A attempt was already consumed")
    if not paths["authorization"].is_file():
        raise PermissionError("separate append-only Stage-A authorization is missing")
    authorization = strict_json_object(paths["authorization"])
    expected_keys = {
        "schema_version",
        "created_at_kst",
        "stage",
        "config",
        "authorization",
        "user_message_reference",
        "qa_receipt",
        "implementation_pins",
        "execution_authorized",
        "upload_allowed",
    }
    _require_exact_keys(authorization, expected_keys, name="authorization")
    contract = config["authorization_contract"]
    expected_phrase = contract["authorization_phrase_prefix"] + CONFIG_SHA256
    qa_path = config["canonical_paths"]["pre_execution_qa"]
    checks = {
        "schema": authorization.get("schema_version") == contract["schema_version"],
        "stage": authorization.get("stage") == STAGE,
        "config": authorization.get("config")
        == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "phrase": authorization.get("authorization") == expected_phrase,
        "user_reference": bool(authorization.get("user_message_reference")),
        "qa": authorization.get("qa_receipt")
        == {"path": qa_path, "sha256": qa_sha256},
        "implementation": authorization.get("implementation_pins")
        == implementation_pins(root),
        "execution": authorization.get("execution_authorized") is True,
        "upload": authorization.get("upload_allowed") is False,
    }
    if not all(checks.values()):
        raise PermissionError(
            "execution authorization failed: "
            f"{sorted(key for key, value in checks.items() if not value)}"
        )
    return authorization, sha256_file(paths["authorization"])


def _attempt_lock_payload(
    root: Path,
    *,
    qa_sha256: str,
    authorization_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "p2_architecture_matched_stage_a_execution.attempt_lock.v2",
        "stage": STAGE,
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "qa_receipt_sha256": qa_sha256,
        "authorization_sha256": authorization_sha256,
        "implementation_pins": implementation_pins(root),
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
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
        raise FileExistsError("canonical append-only Stage-A output already exists")
    payload = _attempt_lock_payload(
        root,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
    )
    exclusive_json(paths["attempt_lock"], payload)
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
        raise PermissionError("canonical one-shot Stage-A attempt lock is missing")
    observed = strict_json_object(path)
    expected = _attempt_lock_payload(
        root,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
    )
    if observed != expected:
        raise PermissionError("consumed Stage-A attempt lock fails full deep equality")
    return observed


def _oof_paths(config: Mapping[str, Any]) -> dict[str, str]:
    artifacts = config["stage_a_reference_contract"]["artifacts"]
    return {
        "0.4": artifacts["reference_oof_040"],
        "0.55": artifacts["reference_oof_055"],
        "0.7": artifacts["reference_oof_070"],
        "0.85": artifacts["reference_oof_085"],
        "1.0": artifacts["reference_oof_100"],
    }


def verify_stage_a_seal(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    output = stage_paths(root, config)["output"].resolve(strict=True)
    artifacts = config["stage_a_reference_contract"]["artifacts"]
    paths = {name: contained_path(output, relative) for name, relative in artifacts.items()}
    if any(not path.is_file() for path in paths.values()):
        raise StageAContractError("a required Stage-A artifact is missing")
    manifest = strict_json_object(paths["manifest"])
    seal = strict_json_object(paths["seal"])
    expected_oof = _oof_paths(config)
    if manifest.get("schema_version") != "p2_architecture_matched_reference.manifest.v2":
        raise StageAContractError("Stage-A manifest schema changed")
    expected_manifest_keys = {
        "schema_version",
        "created_at_kst",
        "append_only",
        "problem",
        "comparison_mode",
        "exact_official_incumbent_comparison",
        "config",
        "implementation_pins",
        "runtime",
        "data_source_pins",
        "architecture_manifest_sha256",
        "training_recipe_sha256",
        "artifacts",
        "challenger_import_fit_or_score_count",
        "official_promotion_allowed",
        "uploads",
    }
    _require_exact_keys(manifest, expected_manifest_keys, name="Stage-A manifest")
    runtime = manifest.get("runtime")
    runtime_contract = config["runtime_contract"]
    runtime_exact_keys = {
        key: runtime_contract[key]
        for key in (
            "python",
            "numpy",
            "pandas",
            "scipy",
            "scikit_learn",
            "lightgbm",
            "torch",
            "torch_cuda",
            "cuda_available",
            "cudnn_benchmark",
            "cudnn_deterministic",
        )
    }
    runtime_valid = (
        isinstance(runtime, Mapping)
        and all(runtime.get(key) == value for key, value in runtime_exact_keys.items())
        and isinstance(runtime.get("gpu_name"), str)
        and runtime_contract["gpu_name_contains"] in runtime["gpu_name"]
    )
    manifest_checks = {
        "append_only": manifest.get("append_only") is True,
        "problem": manifest.get("problem") == "P2",
        "mode": manifest.get("comparison_mode") == MODE,
        "non_exact": manifest.get("exact_official_incumbent_comparison") is False,
        "config": manifest.get("config")
        == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "implementation": manifest.get("implementation_pins") == implementation_pins(root),
        "runtime": runtime_valid,
        "architecture": manifest.get("architecture_manifest_sha256")
        == canonical_mapping_sha256(config["architecture_reference"]),
        "recipe": manifest.get("training_recipe_sha256")
        == canonical_mapping_sha256(config["training_recipe"]),
        "challenger": manifest.get("challenger_import_fit_or_score_count") == 0,
        "promotion": manifest.get("official_promotion_allowed") is False,
        "uploads": manifest.get("uploads") == 0,
    }
    if not all(manifest_checks.values()):
        raise StageAContractError(
            "Stage-A manifest invariant failed: "
            f"{sorted(key for key, value in manifest_checks.items() if not value)}"
        )
    data_pins = manifest.get("data_source_pins")
    expected_data = config["data_contract"]["source_pins"]
    if not isinstance(data_pins, Mapping) or set(data_pins) != set(expected_data):
        raise StageAContractError("manifest data-source pin set changed")
    for name, pin in data_pins.items():
        if (
            not isinstance(pin, Mapping)
            or set(pin) != {"sha256", "bytes"}
            or pin.get("sha256") != expected_data[name]
            or not isinstance(pin.get("bytes"), int)
            or pin["bytes"] <= 0
        ):
            raise StageAContractError(f"manifest data-source pin is invalid: {name}")
    if seal.get("schema_version") != "p2_architecture_matched_reference.seal.v2":
        raise StageAContractError("Stage-A seal schema changed")
    pins = manifest.get("artifacts")
    if not isinstance(pins, Mapping) or set(pins) != set(artifacts) - {"manifest", "seal"}:
        raise StageAContractError("manifest artifact pin set changed")
    verified: dict[str, dict[str, Any]] = {}
    for role, pin in pins.items():
        if not isinstance(pin, Mapping) or set(pin) != {"path", "sha256", "bytes"}:
            raise StageAContractError("invalid manifest artifact pin")
        path = contained_path(output, str(pin["path"]))
        if (
            path != paths[role]
            or not _is_sha256(pin["sha256"])
            or path.stat().st_size != pin["bytes"]
            or sha256_file(path) != pin["sha256"]
        ):
            raise StageAContractError(f"Stage-A artifact pin mismatch: {role}")
        verified[role] = dict(pin)
    expected_header = config["stage_a_reference_contract"]["reference_oof_columns"]
    for fraction, relative in expected_oof.items():
        path = contained_path(output, relative)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), None)
        if header != expected_header or "truth" in header or "target" in header:
            raise StageAContractError(f"target-free OOF header changed: {fraction}")
    if strict_json_object(paths["architecture_manifest"]) != config["architecture_reference"]:
        raise StageAContractError("architecture manifest differs from preregistration")
    if strict_json_object(paths["training_recipe"]) != config["training_recipe"]:
        raise StageAContractError("training recipe differs from preregistration")
    curve = strict_json_object(paths["reference_curve_metrics"])
    if (
        curve.get("schema_version")
        != "p2_architecture_matched_reference.curve_metrics.v2"
        or curve.get("problem") != "P2"
        or curve.get("comparison_mode") != MODE
        or curve.get("exact_official_incumbent_comparison") is not False
        or curve.get("seed_aggregation") != "PREDICTION_MEAN_THEN_METRIC"
        or curve.get("metric") != config["training_recipe"]["metric"]
        or curve.get("local_qualification_only") is not True
        or curve.get("official_promotion_allowed") is not False
        or curve.get("uploads") != 0
    ):
        raise StageAContractError("reference curve metric contract changed")
    points = curve.get("points")
    if (
        not isinstance(points, list)
        or any(not isinstance(point, Mapping) for point in points)
        or [point.get("fraction") for point in points] != list(PREFIX_FRACTIONS)
    ):
        raise StageAContractError("reference curve must contain five ordered prefix points")
    for point in points:
        seed_metrics = point.get("seed_metrics")
        fold_metrics = point.get("fold_metrics")
        layer_metrics = point.get("layer_metrics")
        numeric = [
            point.get("prediction_mean_metric"),
            *(seed_metrics if isinstance(seed_metrics, list) else []),
            *(fold_metrics.values() if isinstance(fold_metrics, Mapping) else []),
            *(layer_metrics.values() if isinstance(layer_metrics, Mapping) else []),
        ]
        if (
            not isinstance(point.get("rows"), int)
            or point["rows"] <= 0
            or not isinstance(seed_metrics, list)
            or len(seed_metrics) != 3
            or not isinstance(fold_metrics, Mapping)
            or set(fold_metrics) != set(OUTER_FOLDS)
            or not isinstance(layer_metrics, Mapping)
            or set(layer_metrics) != {"2", "3", "4"}
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in numeric
            )
        ):
            raise StageAContractError("reference curve point is incomplete or non-finite")
    receipt = strict_json_object(paths["training_receipt"])
    guard_summary = receipt.get("guard_summary")
    if (
        receipt.get("schema_version")
        != "p2_architecture_matched_reference.training_receipt.v2"
        or receipt.get("config") != {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256}
        or receipt.get("runtime") != runtime
        or not isinstance(receipt.get("cells"), list)
        or len(receipt["cells"]) != 45
        or receipt.get("plan", {}).get("deep_training_jobs") != 720
        or receipt.get("plan", {}).get("router_training_jobs") != 180
        or not isinstance(guard_summary, Mapping)
        or guard_summary.get("joint_temp_psal_mask_applied_before_all_label_use") is not True
        or guard_summary.get("outer_and_future_target_labels_used_for_fit") is not False
        or guard_summary.get("frozen_stack_or_gate_reused") is not False
        or guard_summary.get("all_five_prefixes_completed_before_seal") is not True
        or guard_summary.get("challenger_import_fit_or_score_count") != 0
        or guard_summary.get("full_fit_count") != 0
        or guard_summary.get("submission_prediction_count") != 0
        or guard_summary.get("upload_count") != 0
    ):
        raise StageAContractError("training receipt safety contract changed")
    expected_seal = {
        "schema_version": "p2_architecture_matched_reference.seal.v2",
        "complete": True,
        "all_five_prefixes_sealed": True,
        "challenger_import_fit_or_score_count_before_seal": 0,
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "official_promotion_allowed": False,
        "upload_count": 0,
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "manifest": {
            "path": artifacts["manifest"],
            "sha256": sha256_file(paths["manifest"]),
            "bytes": paths["manifest"].stat().st_size,
        },
        "reference_oof_by_fraction": {
            fraction: verified[role]
            for fraction, role in zip(
                ("0.4", "0.55", "0.7", "0.85", "1.0"),
                (
                    "reference_oof_040",
                    "reference_oof_055",
                    "reference_oof_070",
                    "reference_oof_085",
                    "reference_oof_100",
                ),
                strict=True,
            )
        },
    }
    if seal != expected_seal:
        raise StageAContractError("Stage-A seal fails full deep equality")
    return {
        "seal_path": paths["seal"].relative_to(root.resolve(strict=True)).as_posix(),
        "seal_sha256": sha256_file(paths["seal"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "all_five_prefixes_verified": True,
        "exact_official_incumbent_comparison": False,
        "official_promotion_allowed": False,
        "uploads": 0,
    }


__all__ = [
    "CONFIG_RELATIVE",
    "CONFIG_SHA256",
    "CONTRIBUTORS",
    "ENGINE_RELATIVE",
    "IMPLEMENTATION_ROLES",
    "MODE",
    "OUTER_FOLDS",
    "PIPELINE_SEEDS",
    "PREFIX_FRACTIONS",
    "STAGE",
    "StageAContractError",
    "canonical_json_bytes",
    "canonical_mapping_sha256",
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
    "verify_execution_authorization",
    "verify_consumed_attempt_lock",
    "verify_pre_execution_qa",
    "verify_stage_a_seal",
    "workspace_path",
]
