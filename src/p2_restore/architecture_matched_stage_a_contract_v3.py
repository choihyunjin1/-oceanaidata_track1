"""Fail-closed contract for P2 architecture-matched Stage-A execution v3.

V3 is an append-only correction over the immutable v2 implementation.  It
adds the two runtime source pins omitted by v2 and makes the import boundary
observable.  The command-line check runs this module in an isolated process;
an authorized run imports it in-process before consuming the attempt lock.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from p2_restore import architecture_matched_stage_a_contract_v2 as base_v2

CONFIG_RELATIVE = "configs/experiments/p2_architecture_matched_stage_a_execution_v3.json"
CONFIG_SHA256 = "482a646fc36c7639befbd1cea8fde91b358d54e931dad8e82cbb36026c98c649"
BASE_CONFIG_RELATIVE = base_v2.CONFIG_RELATIVE
BASE_CONFIG_SHA256 = base_v2.CONFIG_SHA256
MODE = "ARCHITECTURE_MATCHED_TIME_SAFE_BASELINE"
STAGE = "P2_ARCHITECTURE_MATCHED_REFERENCE_STAGE_A_V3"
ENGINE_RELATIVE = "src/p2_restore/architecture_matched_stage_a_execution_v3.py"
ENGINE_MODULE = "p2_restore.architecture_matched_stage_a_execution_v3"

IMPLEMENTATION_ROLES = {
    "CONFIG": CONFIG_RELATIVE,
    "GUARD": "src/p2_restore/architecture_matched_stage_a_contract_v3.py",
    "ENGINE": ENGINE_RELATIVE,
    "RUNNER": "scripts/run_p2_architecture_matched_reference_v3.py",
    "TESTS": "tests/test_p2_architecture_matched_stage_a_execution_v3.py",
}
IMMUTABLE_V2_ROLES = {
    "CONFIG",
    "GUARD",
    "ENGINE",
    "RUNNER",
    "TESTS",
}
ADDITIONAL_SOURCE_ROLES = {"MODEL_MODULE", "PACKAGE_INIT"}
NUMERICAL_MODULE_PREFIXES = (
    "lightgbm",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
)


class StageAContractV3Error(ValueError):
    """Raised when an immutable v3 execution condition is not satisfied."""


def sha256_file(path: Path) -> str:
    return base_v2.sha256_file(path)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return base_v2.canonical_json_bytes(value)


def canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    return base_v2.canonical_mapping_sha256(value)


def strict_json_object(path: Path) -> dict[str, Any]:
    return base_v2.strict_json_object(path)


def workspace_path(root: Path, relative: str | Path, *, must_exist: bool = True) -> Path:
    try:
        return base_v2.workspace_path(root, relative, must_exist=must_exist)
    except base_v2.StageAContractError as exc:
        raise StageAContractV3Error(str(exc)) from exc


def contained_path(output_root: Path, child: str | Path) -> Path:
    try:
        return base_v2.contained_path(output_root, child)
    except base_v2.StageAContractError as exc:
        raise StageAContractV3Error(str(exc)) from exc


def exclusive_bytes(path: Path, payload: bytes) -> None:
    base_v2.exclusive_bytes(path, payload)


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    base_v2.exclusive_json(path, value)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise StageAContractV3Error(f"{name} keys changed")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_pin_map(
    pins: Any,
    *,
    roles: set[str],
    name: str,
) -> None:
    if not isinstance(pins, Mapping) or set(pins) != roles:
        raise StageAContractV3Error(f"{name} roles changed")
    for role, pin in pins.items():
        if not isinstance(pin, Mapping) or set(pin) != {"path", "sha256", "bytes"}:
            raise StageAContractV3Error(f"{name} pin shape changed: {role}")
        if (
            not isinstance(pin["path"], str)
            or not _is_sha256(pin["sha256"])
            or not isinstance(pin["bytes"], int)
            or pin["bytes"] <= 0
        ):
            raise StageAContractV3Error(f"{name} pin invalid: {role}")


def validate_overlay(overlay: Mapping[str, Any]) -> None:
    if overlay.get("schema_version") != "p2_architecture_matched_stage_a_execution.v3":
        raise StageAContractV3Error("config schema changed")
    if overlay.get("problem") != "P2" or overlay.get("comparison_mode") != MODE:
        raise StageAContractV3Error("architecture-matched mode is P2-only")
    required_false = (
        "exact_official_incumbent_comparison",
        "official_promotion_allowed",
        "upload_allowed",
    )
    if any(overlay.get(key) is not False for key in required_false):
        raise StageAContractV3Error("non-exact/research-only status changed")
    if (
        overlay.get("explicitly_not_exact_official_incumbent") is not True
        or overlay.get("local_qualification_only") is not True
        or overlay.get("research_only") is not True
        or overlay.get("official_submission_count") != 0
    ):
        raise StageAContractV3Error("local-only status changed")
    if overlay.get("preregistration") != {
        "generation_id": "P2_ARCHITECTURE_MATCHED_STAGE_A_EXECUTION_V3",
        "created_before_first_fit": True,
        "score_derived_tuning": False,
        "challenger_hypothesis_count": 0,
        "challenger_import_fit_or_score_count": 0,
    }:
        raise StageAContractV3Error("Stage-A v3 preregistration changed")

    _validate_pin_map(
        overlay.get("immutable_v2_implementation_pins"),
        roles=IMMUTABLE_V2_ROLES,
        name="immutable v2 implementation",
    )
    _validate_pin_map(
        overlay.get("additional_runtime_source_pins"),
        roles=ADDITIONAL_SOURCE_ROLES,
        name="additional runtime source",
    )
    if overlay.get("implementation_roles") != IMPLEMENTATION_ROLES:
        raise StageAContractV3Error("v3 implementation roles changed")

    expected_paths = {
        "config": CONFIG_RELATIVE,
        "output": "artifacts/p2_architecture_matched_reference_v3",
        "control": "artifacts/p2_architecture_matched_reference_v3_control",
        "pre_execution_qa": (
            "artifacts/p2_architecture_matched_reference_v3_control/pre_execution_qa.json"
        ),
        "authorization": (
            "artifacts/p2_architecture_matched_reference_v3_control/authorization.json"
        ),
        "attempt_lock": (
            "artifacts/p2_architecture_matched_reference_v3_control/attempt.lock"
        ),
    }
    if overlay.get("canonical_paths") != expected_paths:
        raise StageAContractV3Error("canonical v3 paths changed")

    policy = overlay.get("execution_policy")
    if not isinstance(policy, Mapping):
        raise StageAContractV3Error("execution policy is missing")
    for key in (
        "check_only_is_default",
        "check_only_preflight_runs_in_isolated_child",
        "check_only_parent_must_not_import_engine_or_numerical_training_modules",
        "child_import_accounting_must_be_truthful",
        "direct_engine_entry_must_ignore_caller_preflight_and_rerun_canonical_preflight",
        "direct_engine_entry_rechecks_runtime_in_current_process",
        "all_architecture_deployed_source_runtime_pins_before_output_or_fit",
        "actual_run_requires_independent_qa_receipt",
        "actual_run_requires_separate_append_only_authorization",
        "qa_and_authorization_verified_before_attempt_lock",
        "execution_engine_imported_after_attempt_lock",
        "output_directory_created_with_o_excl_semantics",
        "all_materialized_files_created_o_excl",
    ):
        if policy.get(key) is not True:
            raise StageAContractV3Error(f"required execution guard changed: {key}")
    for key in (
        "rerun_allowed",
        "resume_allowed",
        "frozen_mutation_allowed",
        "submission_mutation_allowed",
        "registry_append_allowed",
        "automatic_upload_allowed",
    ):
        if policy.get(key) is not False:
            raise StageAContractV3Error(f"forbidden execution capability changed: {key}")

    compiled = overlay.get("compiled_recipe")
    if compiled != {
        "base": "FULL_DEEP_EQUAL_V2_RECIPE_AND_ARCHITECTURE",
        "base_config_sha256": BASE_CONFIG_SHA256,
        "override_scope": [
            "identity_and_schema_labels",
            "canonical_v3_output_and_control_paths",
            "v3_qa_authorization_and_seal_schemas",
            "additional_model_and_package_source_pins",
            "fresh_preflight_and_runtime_guards",
        ],
        "training_or_metric_change_from_v2": False,
    }:
        raise StageAContractV3Error("compiled v2 recipe inheritance changed")
    qa = overlay.get("qa_receipt_contract")
    if qa != {
        "schema_version": "p2_architecture_matched_stage_a_execution.pre_execution_qa.v3",
        "decision": "GO_AUTHORIZE_P2_STAGE_A_EXECUTION_V3",
        "required_p0_count": 0,
        "required_p1_count": 0,
        "must_pin_exact_roles": list(IMPLEMENTATION_ROLES),
    }:
        raise StageAContractV3Error("independent QA contract changed")
    authorization = overlay.get("authorization_contract")
    if authorization != {
        "schema_version": "p2_architecture_matched_stage_a_execution.authorization.v3",
        "authorization_phrase_prefix": "AUTHORIZE_P2_STAGE_A_EXECUTION_V3:",
        "must_pin_qa_receipt_sha256": True,
        "must_deep_equal_current_implementation_pins": True,
        "execution_authorized": True,
        "upload_allowed": False,
    }:
        raise StageAContractV3Error("authorization contract changed")


def _compile_config(base_config: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    compiled = copy.deepcopy(dict(base_config))
    for key in (
        "schema_version",
        "experiment_id",
        "created_at_kst",
        "status",
        "problem",
        "comparison_mode",
        "exact_official_incumbent_comparison",
        "explicitly_not_exact_official_incumbent",
        "local_qualification_only",
        "official_promotion_allowed",
        "research_only",
        "upload_allowed",
        "official_submission_count",
        "preregistration",
        "canonical_paths",
        "execution_policy",
        "qa_receipt_contract",
        "authorization_contract",
        "resource_estimate",
    ):
        compiled[key] = copy.deepcopy(overlay[key])
    compiled["immutable_v2_implementation_pins"] = copy.deepcopy(
        overlay["immutable_v2_implementation_pins"]
    )
    compiled["additional_runtime_source_pins"] = copy.deepcopy(
        overlay["additional_runtime_source_pins"]
    )
    compiled["implementation_roles"] = copy.deepcopy(overlay["implementation_roles"])
    compiled["compiled_recipe"] = copy.deepcopy(overlay["compiled_recipe"])
    return compiled


def load_canonical_overlay(
    root: Path,
    requested_path: Path | None = None,
) -> dict[str, Any]:
    canonical = workspace_path(root, CONFIG_RELATIVE)
    requested = (requested_path or canonical).resolve(strict=True)
    if requested != canonical:
        raise StageAContractV3Error("only the canonical v3 config path is accepted")
    if sha256_file(canonical) != CONFIG_SHA256:
        raise StageAContractV3Error("canonical v3 config SHA mismatch")
    overlay = strict_json_object(canonical)
    validate_overlay(overlay)
    return overlay


def load_canonical_config(
    root: Path,
    requested_path: Path | None = None,
    *,
    supplied_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    overlay = load_canonical_overlay(root, requested_path)
    base_config = base_v2.load_canonical_config(root)
    compiled = _compile_config(base_config, overlay)
    if supplied_config is not None and dict(supplied_config) != compiled:
        raise StageAContractV3Error("supplied compiled config fails full deep equality")
    return compiled


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


def _verify_exact_pin_map(
    root: Path,
    pins: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    workspace = root.resolve(strict=True)
    verified: dict[str, dict[str, Any]] = {}
    for role, expected in pins.items():
        path = workspace_path(workspace, str(expected["path"]))
        actual = _pin(path, workspace)
        if actual != dict(expected):
            raise StageAContractV3Error(f"{label} drift: {role}")
        verified[str(role)] = actual
    return verified


def stage_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    canonical = config["canonical_paths"]
    return {
        key: workspace_path(root, canonical[key], must_exist=False)
        for key in ("output", "control", "pre_execution_qa", "authorization", "attempt_lock")
    }


def _loaded_modules(prefixes: tuple[str, ...] = NUMERICAL_MODULE_PREFIXES) -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    )


def probe_runtime_isolated(root: Path) -> dict[str, Any]:
    """Verify the exact v2 runtime in a disposable process and report its imports."""

    workspace = root.resolve(strict=True)
    script = """
import json
import sys
from pathlib import Path
from p2_restore import architecture_matched_stage_a_contract_v2 as guard
from p2_restore import architecture_matched_stage_a_execution_v2 as engine
root = Path(sys.argv[1]).resolve(strict=True)
config = guard.load_canonical_config(root)
engine.set_deterministic_seed(engine.PIPELINE_SEEDS[0])
runtime = engine._verify_runtime(config)
prefixes = ("lightgbm", "numpy", "pandas", "scipy", "sklearn", "torch")
loaded = sorted(name for name in sys.modules if any(name == p or name.startswith(p + ".") for p in prefixes))
print(json.dumps({
    "status": "PASS_ISOLATED_RUNTIME",
    "runtime": runtime,
    "loaded_numerical_modules": loaded,
    "v2_execution_engine_imported_in_isolated_probe": True,
}, sort_keys=True))
"""
    environment = os.environ.copy()
    source = str((workspace / "src").resolve(strict=True))
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not existing else os.pathsep.join((source, existing))
    completed = subprocess.run(
        [sys.executable, "-c", script, str(workspace)],
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"isolated Stage-A runtime preflight failed: {detail}")
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("isolated runtime probe returned invalid JSON") from exc
    if result.get("status") != "PASS_ISOLATED_RUNTIME":
        raise RuntimeError("isolated runtime probe did not pass")
    return result


def static_preflight(
    root: Path,
    data_dir: Path,
    *,
    requested_config: Path | None = None,
    supplied_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freshly verify every configuration, graph, source, data-key, and runtime pin."""

    workspace = root.resolve(strict=True)
    imports_before = set(_loaded_modules())
    config = load_canonical_config(
        workspace,
        requested_config,
        supplied_config=supplied_config,
    )
    overlay = load_canonical_overlay(workspace, requested_config)
    immutable_v2 = _verify_exact_pin_map(
        workspace,
        overlay["immutable_v2_implementation_pins"],
        label="immutable v2 implementation",
    )
    additional_sources = _verify_exact_pin_map(
        workspace,
        overlay["additional_runtime_source_pins"],
        label="additional runtime source",
    )
    base_report = base_v2.static_preflight(workspace, data_dir)
    runtime_probe = probe_runtime_isolated(workspace)
    imports_after = set(_loaded_modules())
    paths = stage_paths(workspace, config)
    return {
        "schema_version": "p2_architecture_matched_stage_a_execution.static_preflight.v3",
        "status": "PASS_STATIC_IMPLEMENTATION_ONLY",
        "problem": "P2",
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "canonical_config": {
            "path": CONFIG_RELATIVE,
            "sha256": CONFIG_SHA256,
            "bytes": workspace_path(workspace, CONFIG_RELATIVE).stat().st_size,
        },
        "implementation_pins": implementation_pins(workspace),
        "immutable_v2_implementation_pins": immutable_v2,
        "additional_runtime_source_pins": additional_sources,
        "architecture_source_pins": base_report["architecture_source_pins"],
        "predecessor_pins": base_report["predecessor_pins"],
        "deployed_v1_graph_verified": base_report["deployed_v1_graph_verified"],
        "schema_and_keys": base_report["schema_and_keys"],
        "runtime_probe": runtime_probe,
        "preflight_process_loaded_numerical_modules": sorted(imports_after),
        "preflight_process_new_numerical_modules": sorted(imports_after - imports_before),
        "runtime_probe_isolated": True,
        "canonical_path_state": {key: path.exists() for key, path in paths.items()},
        "files_written": 0,
        "attempt_locks_created": 0,
        "fits": 0,
        "predictions": 0,
        "uploads": 0,
        "resource_estimate": config["resource_estimate"],
    }


def _qa_expected_keys() -> set[str]:
    return {
        "schema_version",
        "created_at_kst",
        "decision",
        "p0_count",
        "p1_count",
        "config",
        "immutable_v2_implementation_pins",
        "additional_runtime_source_pins",
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
        raise PermissionError("independent v3 pre-execution QA receipt is missing")
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
        "immutable_v2": receipt.get("immutable_v2_implementation_pins")
        == config["immutable_v2_implementation_pins"],
        "additional_sources": receipt.get("additional_runtime_source_pins")
        == config["additional_runtime_source_pins"],
        "implementation": receipt.get("implementation_pins") == implementation_pins(root),
        "roles": set(receipt.get("implementation_pins", {})) == set(IMPLEMENTATION_ROLES),
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
        raise FileExistsError("canonical append-only Stage-A v3 output already exists")
    if require_unconsumed and paths["attempt_lock"].exists():
        raise FileExistsError("canonical one-shot Stage-A v3 attempt was already consumed")
    if not paths["authorization"].is_file():
        raise PermissionError("separate append-only Stage-A v3 authorization is missing")
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
    checks = {
        "schema": authorization.get("schema_version") == contract["schema_version"],
        "stage": authorization.get("stage") == STAGE,
        "config": authorization.get("config")
        == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "phrase": authorization.get("authorization") == expected_phrase,
        "user_reference": bool(authorization.get("user_message_reference")),
        "qa": authorization.get("qa_receipt")
        == {
            "path": config["canonical_paths"]["pre_execution_qa"],
            "sha256": qa_sha256,
        },
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
        "schema_version": "p2_architecture_matched_stage_a_execution.attempt_lock.v3",
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
        raise FileExistsError("canonical append-only Stage-A v3 output already exists")
    exclusive_json(
        paths["attempt_lock"],
        _attempt_lock_payload(
            root,
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
        raise PermissionError("canonical one-shot Stage-A v3 attempt lock is missing")
    observed = strict_json_object(path)
    expected = _attempt_lock_payload(
        root,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
    )
    if observed != expected:
        raise PermissionError("consumed Stage-A v3 attempt lock fails full deep equality")
    return observed


def verify_stage_a_seal(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    output = stage_paths(root, config)["output"].resolve(strict=True)
    artifacts = config["stage_a_reference_contract"]["artifacts"]
    paths = {role: contained_path(output, relative) for role, relative in artifacts.items()}
    if any(not path.is_file() for path in paths.values()):
        raise StageAContractV3Error("a required Stage-A v3 artifact is missing")
    manifest = strict_json_object(paths["manifest"])
    seal = strict_json_object(paths["seal"])
    if manifest.get("schema_version") != "p2_architecture_matched_reference.manifest.v3":
        raise StageAContractV3Error("Stage-A v3 manifest schema changed")
    if seal.get("schema_version") != "p2_architecture_matched_reference.seal.v3":
        raise StageAContractV3Error("Stage-A v3 seal schema changed")
    checks = {
        "manifest_config": manifest.get("config")
        == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "manifest_implementation": manifest.get("implementation_pins")
        == implementation_pins(root),
        "manifest_non_exact": manifest.get("exact_official_incumbent_comparison") is False,
        "manifest_no_promotion": manifest.get("official_promotion_allowed") is False,
        "manifest_no_upload": manifest.get("uploads") == 0,
        "seal_complete": seal.get("complete") is True,
        "seal_all_prefixes": seal.get("all_five_prefixes_sealed") is True,
        "seal_non_exact": seal.get("exact_official_incumbent_comparison") is False,
        "seal_no_promotion": seal.get("official_promotion_allowed") is False,
        "seal_no_upload": seal.get("upload_count") == 0,
        "seal_config": seal.get("config")
        == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "seal_manifest": seal.get("manifest")
        == {
            "path": paths["manifest"].relative_to(output).as_posix(),
            "sha256": sha256_file(paths["manifest"]),
            "bytes": paths["manifest"].stat().st_size,
        },
    }
    if not all(checks.values()):
        raise StageAContractV3Error(
            "Stage-A v3 seal failed: "
            f"{sorted(key for key, value in checks.items() if not value)}"
        )
    for role, pin in manifest.get("artifacts", {}).items():
        if role in {"manifest", "seal"} or not isinstance(pin, Mapping):
            raise StageAContractV3Error("invalid manifest artifact pin")
        path = contained_path(output, str(pin.get("path")))
        if _pin(path, output) != dict(pin):
            raise StageAContractV3Error(f"Stage-A artifact drift: {role}")
    return {
        "status": "PASS_SEALED_ARCHITECTURE_MATCHED_REFERENCE_V3",
        "manifest_sha256": sha256_file(paths["manifest"]),
        "seal_sha256": sha256_file(paths["seal"]),
        "exact_official_incumbent_comparison": False,
        "official_promotion_allowed": False,
        "uploads": 0,
    }


__all__ = [
    "ADDITIONAL_SOURCE_ROLES",
    "BASE_CONFIG_RELATIVE",
    "BASE_CONFIG_SHA256",
    "CONFIG_RELATIVE",
    "CONFIG_SHA256",
    "ENGINE_MODULE",
    "ENGINE_RELATIVE",
    "IMPLEMENTATION_ROLES",
    "MODE",
    "NUMERICAL_MODULE_PREFIXES",
    "STAGE",
    "StageAContractV3Error",
    "canonical_json_bytes",
    "canonical_mapping_sha256",
    "consume_attempt_lock",
    "contained_path",
    "exclusive_bytes",
    "exclusive_json",
    "implementation_pins",
    "load_canonical_config",
    "load_canonical_overlay",
    "probe_runtime_isolated",
    "sha256_file",
    "stage_paths",
    "static_preflight",
    "strict_json_object",
    "validate_overlay",
    "verify_consumed_attempt_lock",
    "verify_execution_authorization",
    "verify_pre_execution_qa",
    "verify_stage_a_seal",
    "workspace_path",
]
