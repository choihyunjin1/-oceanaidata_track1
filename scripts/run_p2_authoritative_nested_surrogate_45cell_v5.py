#!/usr/bin/env python
"""Seal or run the launch-bound P2 v5 authoritative 45-cell contract.

V5 inherits the sealed v4 scientific contract and atomic crash-resume engine.
Its only changes are an exact authorization schema requirement and a literal,
approved absolute observations directory in the one clean-start/resume command.
The runner never opens or enumerates official P2 test/sample/submission paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts import run_p2_authoritative_nested_surrogate_45cell_v1 as base_runner  # noqa: E402
from scripts import run_p2_authoritative_nested_surrogate_45cell_v3 as v3_runner  # noqa: E402
from scripts import run_p2_authoritative_nested_surrogate_45cell_v4 as v4_runner  # noqa: E402
from p2_restore.authoritative_nested_surrogate_conformance import COMPONENTS  # noqa: E402
from p2_restore.authoritative_nested_surrogate_execution import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    temporary_tiny_fixture,
    verify_authorization,
    verify_preexecution_seal,
)
from p2_restore.authoritative_nested_surrogate_execution_v5 import (  # noqa: E402
    ExecutionBindingV5,
    SemanticPreflightOutcomeV5,
    TERMINAL_STATUS,
    execute_authorized_curve_v5,
    inspect_actual_namespace_read_only,
    run_resumable_execution_v5,
    semantic_preflight_actual_data_v5,
)

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/p2_authoritative_nested_surrogate_execution_20260825_v5.json"
)
DEFAULT_READY_OUTPUT = (
    PROJECT_ROOT / "artifacts/p2_authoritative_nested_surrogate_execution_ready_20260825_v5"
)
EXPECTED_CONFIG_SHA256 = "a954511ed0a01ef08d7c3762d0444bad9f00d8bb59b084cd0be15f178b74d2a0"
MODULE_RELATIVE = "src/p2_restore/authoritative_nested_surrogate_execution_v5.py"
RUNNER_RELATIVE = "scripts/run_p2_authoritative_nested_surrogate_45cell_v5.py"
REQUIRED_AUTHORIZATION_SCHEMA_VERSION = (
    "p2_authoritative_nested_surrogate_execution_authorization.v5"
)
APPROVED_DATA_DIRECTORY = Path(
    r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path.name}")
    return value


def _write_json_exclusive(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_text_exclusive(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _resolve_repo_path(relative: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve(strict=True)
    path.relative_to(PROJECT_ROOT.resolve(strict=True))
    return path


def _load_recipe_v5(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    overlay = _read_json(path)
    base_pin = dict(overlay["base_recipe"])
    base_path = _resolve_repo_path(str(base_pin["path"]))
    _require(sha256_file(base_path) == base_pin["sha256"], "v5 base recipe changed")
    base, _ = v4_runner._load_recipe_v4(base_path)
    return base_runner._deep_merge(base, overlay), base_pin


def _load_config_v5(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    overlay = _read_json(path)
    base_pin = dict(overlay["base_config"])
    base_path = _resolve_repo_path(str(base_pin["path"]))
    _require(sha256_file(base_path) == base_pin["sha256"], "v5 base config changed")
    base, _ = v4_runner._load_config_v4(base_path)
    return base_runner._deep_merge(base, overlay), base_pin


def _verify_static(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve(strict=True)
    config_sha = sha256_file(config_path)
    if EXPECTED_CONFIG_SHA256 != "TO_BE_SEALED_V5":
        _require(config_sha == EXPECTED_CONFIG_SHA256, "v5 execution config changed")
    overlay = _read_json(config_path)
    base_path = _resolve_repo_path(str(overlay["base_config"]["path"]))
    base_config, base_recipe, base_static = v4_runner._verify_static(base_path)
    config, base_pin = _load_config_v5(config_path)
    _require(
        config["schema_version"] == "p2_authoritative_nested_surrogate_execution.v5",
        "v5 execution schema changed",
    )
    _require(
        config["status"] == "SEALED_EXECUTION_READY_DRY_RUN_NO_ACTUAL_FIT",
        "v5 execution config is not sealed",
    )
    _require(
        all(value is False for value in config["permissions"].values()),
        "v5 readiness grants a forbidden permission",
    )
    _require(
        config["required_authorization"]["schema_version"]
        == REQUIRED_AUTHORIZATION_SCHEMA_VERSION,
        "v5 required authorization schema changed",
    )
    resolved = config["resolved_data_dir_contract"]
    _require(
        resolved["absolute_path"] == str(APPROVED_DATA_DIRECTORY),
        "v5 approved data directory changed",
    )
    _require(resolved["observations_filename"] == "observations.csv", "v5 data file changed")
    _require(resolved["observations_sha256"] == config["data_contract"]["sha256"], "v5 data SHA differs")
    _require(int(resolved["observations_bytes"]) == int(config["data_contract"]["bytes"]), "v5 data bytes differ")
    _require(resolved["directory_enumeration_allowed"] is False, "v5 data enumeration enabled")
    _require(resolved["official_test_sample_submission_reads_allowed"] is False, "v5 official reads enabled")

    parent = config["parent_contract"]
    pins = {
        parent["path"]: parent["sha256"],
        parent["contract_seal_path"]: parent["contract_seal_sha256"],
        **config["source_pins"],
        **config["v5_evidence_pins"],
    }
    verified: dict[str, Any] = {}
    for relative, expected in pins.items():
        current = _resolve_repo_path(str(relative))
        actual = sha256_file(current)
        _require(actual == expected, f"v5 pinned source/evidence changed: {relative}")
        verified[str(relative)] = {"sha256": actual, "bytes": current.stat().st_size}

    recipe, recipe_base_pin = _load_recipe_v5(_resolve_repo_path(parent["path"]))
    nested = recipe["authoritative_nested_surrogate_recipe"]
    base_nested = base_recipe["authoritative_nested_surrogate_recipe"]
    _require(recipe["training_authorized"] is False, "v5 parent recipe authorizes fit")
    _require(
        nested["identity"] == "P2_AUTHORITATIVE_NESTED_CAUSAL_SURROGATE_V5_LAUNCH_BOUND",
        "v5 recipe identity changed",
    )
    for key in (
        "component_hyperparameters",
        "meta_refit",
        "postprocess",
        "family_views",
        "metrics",
        "execution_graph",
    ):
        _require(config[key] == base_config[key], f"v5 science surface changed: {key}")
    for key in (
        "outer_fold_contract",
        "chronological_prefix_contract",
        "nested_component_oof_contract",
        "deep_public_context_contract",
        "complete_pipeline_seed_contract",
        "epoch_and_meta_contract",
        "postprocess_contract",
    ):
        _require(nested[key] == base_nested[key], f"v5 recipe science surface changed: {key}")
    resume = config["checkpoint_resume"]
    _require(resume["maximum_resume_attempts_after_initial"] == 2, "v5 resume budget changed")
    _require(resume["maximum_total_attempts"] == 3, "v5 total attempt budget changed")
    _require(resume["same_exact_command_clean_or_resume"] is True, "v5 command mode changed")
    _require(resume["v1_v2_v3_v4_job_reuse_allowed"] is False, "foreign jobs allowed")
    _require(
        config["output"]["actual_directory"]
        == "artifacts/p2_authoritative_nested_surrogate_actual_20260825_v5",
        "v5 actual namespace changed",
    )
    _require(
        config["metrics"]["expected_evaluation_rows_per_fraction"] == 78156,
        "v5 outer population changed",
    )
    command_receipt = _command_namespace(str(config["exact_command"]), config=config)
    return config, recipe, {
        "status": "PASS_V5_STATIC_EXACT_AUTH_SCHEMA_LITERAL_DATA_DIR_AND_UNCHANGED_SCIENCE",
        "config_sha256": config_sha,
        "base_config": base_pin,
        "base_recipe": recipe_base_pin,
        "base_v4_static_status": base_static["status"],
        "verified_input_count": len(verified),
        "verified_inputs": verified,
        "recipe_identity": nested["identity"],
        "required_authorization_schema_version": REQUIRED_AUTHORIZATION_SCHEMA_VERSION,
        "resolved_data_directory": resolved["absolute_path"],
        "observations_sha256": resolved["observations_sha256"],
        "observations_bytes": int(resolved["observations_bytes"]),
        "exact_command_sha256": hashlib.sha256(str(config["exact_command"]).encode()).hexdigest(),
        "exact_command_namespace": command_receipt["status"],
        "component_hyperparameters_sha256": canonical_sha256(config["component_hyperparameters"]),
        "meta_refit_sha256": canonical_sha256(config["meta_refit"]),
        "postprocess_sha256": canonical_sha256(config["postprocess"]),
        "family_views_sha256": canonical_sha256(config["family_views"]),
        "metrics_sha256": canonical_sha256(config["metrics"]),
        "execution_graph_sha256": canonical_sha256(config["execution_graph"]),
        "supervised_ledger_and_splits_unchanged": True,
        "outer_windows_fractions_seeds_unchanged": True,
        "models_hyperparameters_meta_postprocess_metrics_unchanged": True,
        "outer_evaluation_rows_per_fraction": 78156,
        "v4_resume_engine_byte_pinned": True,
        "only_launch_authorization_and_data_dir_binding_changed": True,
    }


def _verify_resolved_data_dir(data_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Bind one direct observations path without listing any sibling entry."""

    contract = config["resolved_data_dir_contract"]
    approved = Path(str(contract["absolute_path"])).resolve(strict=True)
    provided = data_dir.expanduser().resolve(strict=True)
    _require(provided == approved, "v5 data directory differs from sealed absolute path")
    _require(provided == APPROVED_DATA_DIRECTORY.resolve(strict=True), "v5 approved directory constant differs")
    observation = (provided / str(contract["observations_filename"])).resolve(strict=True)
    try:
        observation.relative_to(provided)
    except ValueError as error:
        raise ValueError("v5 observations.csv escaped approved directory") from error
    _require(observation.is_file() and not observation.is_symlink(), "v5 observations path is unsafe")
    _require(observation.stat().st_size == int(contract["observations_bytes"]), "v5 observation bytes changed")
    _require(sha256_file(observation) == contract["observations_sha256"], "v5 observation hash changed")
    return {
        "schema_version": "p2_authoritative_resolved_data_dir_receipt.v5",
        "status": "PASS_LITERAL_APPROVED_DATA_DIR_OBSERVATIONS_ONLY",
        "resolved_data_directory": str(provided),
        "files_directly_addressed": ["observations.csv"],
        "directory_entries_enumerated": 0,
        "observations_sha256": contract["observations_sha256"],
        "observations_bytes": int(contract["observations_bytes"]),
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
    }


def _verify_semantic_gates(config: dict[str, Any], receipt: dict[str, Any]) -> None:
    v4_runner._verify_semantic_gates(config, receipt)
    _require(
        receipt["scientific_preflight_v4_sha256"]
        == config["semantic_preflight_v5"]["expected_scientific_v4_sha256"],
        "v5 inherited v4 semantic digest changed",
    )
    revision = receipt["orchestration_revision_v5"]
    _require(revision["science_surface_changed"] is False, "v5 semantic wrapper changed science")
    _require(revision["v4_resume_engine_byte_pinned"] is True, "v5 resume engine pin missing")
    _require(receipt["outer_evaluation_rows_per_fraction"] == 78156, "v5 outer population changed")
    preimage = dict(receipt)
    claimed = str(preimage.pop("semantic_receipt_sha256"))
    _require(canonical_sha256(preimage) == claimed, "v5 semantic receipt self-hash changed")


def _command_namespace(command: str, *, config: dict[str, Any]) -> dict[str, Any]:
    literal_data_fragment = f'--data-dir "{APPROVED_DATA_DIRECTORY}"'
    fragments = (
        "run_p2_authoritative_nested_surrogate_45cell_v5.py",
        "p2_authoritative_nested_surrogate_execution_20260825_v5.json",
        literal_data_fragment,
        "p2_authoritative_nested_surrogate_execution_ready_20260825_v5\\preexecution_seal.json",
        "p2_authoritative_nested_surrogate_execution_ready_20260825_v5\\EXECUTION_AUTHORIZATION.json",
    )
    _require(all(fragment in command for fragment in fragments), "v5 command namespace/data dir is incomplete")
    _require("$env:P2_DATA_DIR" not in command and "%P2_DATA_DIR%" not in command, "v5 command depends on data-dir environment")
    for foreign in ("actual_20260825_v1", "actual_20260825_v2", "actual_20260825_v3", "actual_20260825_v4"):
        _require(foreign not in command, "v5 command references a foreign actual namespace")
    _require(
        config["required_authorization"]["schema_version"] == REQUIRED_AUTHORIZATION_SCHEMA_VERSION,
        "v5 command authorization contract schema changed",
    )
    return {
        "schema_version": "p2_authoritative_exact_command_receipt.v5",
        "status": "PASS_SAME_EXACT_COMMAND_V5_LITERAL_APPROVED_DATA_DIR",
        "required_fragments": list(fragments),
        "required_authorization_schema_version": REQUIRED_AUTHORIZATION_SCHEMA_VERSION,
        "literal_resolved_data_directory": str(APPROVED_DATA_DIRECTORY),
        "environment_data_dir_dependency": False,
        "foreign_actual_namespace_present": False,
        "actual_model_fits": 0,
    }


class _FixtureDeath(BaseException):
    pass


def _resume_control_fixture() -> dict[str, Any]:
    """Synthetic v5 control check; no production fit, prediction, or score."""

    with tempfile.TemporaryDirectory(prefix="p2_v5_resume_fixture_") as raw:
        actual = Path(raw) / "fixture_actual_v5"
        binding = ExecutionBindingV5(
            namespace=actual.name,
            execution_contract_sha256="1" * 64,
            parent_recipe_sha256="2" * 64,
            preexecution_seal_sha256="3" * 64,
            semantic_preflight_sha256="4" * 64,
            exact_command_sha256="5" * 64,
            authorization_sha256="6" * 64,
            module_sha256="7" * 64,
            runner_sha256="8" * 64,
        )

        def preflight() -> SemanticPreflightOutcomeV5:
            return SemanticPreflightOutcomeV5("4" * 64, None)

        try:
            run_resumable_execution_v5(
                actual_dir=actual,
                binding=binding,
                semantic_preflight=preflight,
                execute_curve=lambda _context, _contract: (_ for _ in ()).throw(
                    _FixtureDeath("simulated abrupt process death")
                ),
            )
        except _FixtureDeath:
            pass
        interrupted = inspect_actual_namespace_read_only(actual, binding=binding)
        final = run_resumable_execution_v5(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=preflight,
            execute_curve=lambda _context, _contract: {
                "status": TERMINAL_STATUS,
                "submission_files_generated": 0,
                "uploads": 0,
            },
        )
        closed = inspect_actual_namespace_read_only(actual, binding=binding)
        _require(interrupted["total_attempts_started"] == 1, "fixture initial count changed")
        _require(final["total_attempts_started"] == 2, "fixture resume count changed")
        _require(closed["status"] == "TERMINAL_COMPLETE_NO_RERUN", "fixture did not close")
        return {
            "status": "PASS_V5_SYNTHETIC_CRASH_SAME_COMMAND_RESUME_ATOMIC_TERMINAL",
            "control_engine_schema_version": "v4-byte-pinned",
            "initial_attempts": 1,
            "resume_attempts": 1,
            "maximum_resume_attempts": 2,
            "terminal_result_sha256": final["result_sha256"],
            "actual_model_fits": 0,
            "actual_predictions": 0,
            "actual_scores": 0,
        }


def _report_ko(*, semantic: dict[str, Any], command: str) -> str:
    ledger = semantic["supervised_common_ledger"]
    minima = semantic["minimum_support_across_all_scopes"]
    return f"""# P2 authoritative nested surrogate v5 실행 준비 보고서

## 결론

판정은 `EXECUTION_READY_NOT_AUTHORIZED`이다. v4의 과학 계약과 원자적 crash-resume 엔진은 바이트 고정했다. v5에서 바뀐 것은 독립 QA가 지적한 두 launch binding뿐이다. 승인은 정확히 `{REQUIRED_AUTHORIZATION_SCHEMA_VERSION}`만 허용하고, 동일 clean-start/resume 명령은 승인된 observations 절대 경로를 literal `--data-dir`로 포함한다.

실데이터 production parser와 동일 scientific adapter로 15개 prefix/45개 inner를 다시 검사했고 fit/prediction/score는 0/0/0이다. 공통 supervised ledger는 {ledger['common_supervised_time_count']:,}시각이며 SHA-256은 `{ledger['ordered_time_sha256']}`다.

## 실행 가능성

- 최소 router train/validation support: 층당 {minima['router_train_rows_per_layer']:,}/{minima['router_validation_rows_per_layer']:,}행
- 최소 deep train/validation support: 층당 {minima['deep_train_rows_per_layer']:,}/{minima['deep_validation_rows_per_layer']:,}행
- 최소 deep supervised chunk: inner {minima['deep_supervised_chunks']:,}, full {minima['full_deep_supervised_chunks']:,}
- 최소 meta OOF support: 층당 {minima['meta_oof_rows_per_layer']:,}행
- 비율당 outer 평가 모집단: {semantic['outer_evaluation_rows_per_fraction']:,}행

## v5 launch 및 resume 계약

실행 전 승인 JSON의 `schema_version`은 정확히 `{REQUIRED_AUTHORIZATION_SCHEMA_VERSION}`이어야 한다. 데이터 경로는 `{APPROVED_DATA_DIRECTORY}`로 봉인되며 직접 지정한 `observations.csv` 외 sibling은 열거나 열거하지 않는다. v4 engine의 clean preflight, exclusive lock, immutable start, strict JobStore audit, initial 1회+resume 최대 2회, deterministic fail-closed, atomic result/receipt 및 finalization-only recovery는 그대로다. v1-v4 job/cell 재사용과 결과 기반 재실행·튜닝은 금지된다.

## 별도 승인 후 clean start와 crash resume에 동일하게 쓰는 명령

```powershell
{command}
```

이번 준비에서는 authorization과 actual namespace를 만들지 않았다. 공식 P2 test/sample/submission 및 submission candidate를 읽지 않았고, submission 생성·업로드와 P3 변경도 0회다.
"""


def _manifest(output_dir: Path, config_path: Path, static: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "manifest.json":
            outputs[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {
        "schema_version": "p2_authoritative_nested_surrogate_execution_ready_manifest.v5",
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "status": "EXECUTION_READY_NOT_AUTHORIZED",
        "config": {
            "path": config_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "sha256": static["config_sha256"],
        },
        "module": {"path": MODULE_RELATIVE, "sha256": sha256_file(_resolve_repo_path(MODULE_RELATIVE))},
        "runner": {"path": RUNNER_RELATIVE, "sha256": sha256_file(Path(__file__).resolve())},
        "required_authorization": {"schema_version": REQUIRED_AUTHORIZATION_SCHEMA_VERSION},
        "resolved_data_dir_contract": {
            "absolute_path": static["resolved_data_directory"],
            "observations_sha256": static["observations_sha256"],
            "observations_bytes": static["observations_bytes"],
            "directory_entries_enumerated": 0,
        },
        "exact_command_sha256": static["exact_command_sha256"],
        "outputs": outputs,
        "actual_namespace_created": False,
        "authorization_receipt_created": False,
        "actual_model_fits": 0,
        "actual_scores": 0,
        "actual_predictions": 0,
        "official_test_sample_submission_reads": 0,
        "submission_files_generated": 0,
        "uploads": 0,
        "p3_process_mutations": 0,
    }


def seal_readiness(config_path: Path, *, data_dir: Path, output_dir: Path) -> dict[str, Any]:
    config, recipe, static = _verify_static(config_path)
    resolved_receipt = _verify_resolved_data_dir(data_dir, config)
    observations, data_receipt = v3_runner._load_observations(data_dir, config)
    plans, semantic = semantic_preflight_actual_data_v5(observations, recipe=recipe, config=config)
    _verify_semantic_gates(config, semantic)
    seeds = [
        int(value)
        for value in recipe["authoritative_nested_surrogate_recipe"]["complete_pipeline_seed_contract"]["seeds"]
    ]
    plan_receipt = base_runner._seed_plan_receipt(list(plans), seeds)
    tiny = temporary_tiny_fixture(plans[0])
    resume_fixture = _resume_control_fixture()
    model_shape = base_runner._model_shape_receipt(config)
    atomic_publish = base_runner._atomic_publish_fixture()
    resource = base_runner._resource_receipt(config)
    command = str(config["exact_command"])
    command_receipt = _command_namespace(command, config=config)
    module_sha = sha256_file(_resolve_repo_path(MODULE_RELATIVE))
    runner_sha = sha256_file(Path(__file__).resolve())
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), "v5 readiness output already exists")
    actual_dir = (PROJECT_ROOT / config["output"]["actual_directory"]).resolve()
    _require(not actual_dir.exists(), "v5 actual namespace already exists before readiness")
    output_dir.mkdir(parents=True, exist_ok=False)
    semantic_sha = str(semantic["semantic_receipt_sha256"])
    required_auth = {"schema_version": REQUIRED_AUTHORIZATION_SCHEMA_VERSION}
    preexecution = {
        "schema_version": "p2_authoritative_nested_surrogate_preexecution_seal.v5",
        "status": "EXECUTION_READY_NOT_AUTHORIZED",
        "config_sha256": static["config_sha256"],
        "parent_recipe_sha256": config["parent_contract"]["sha256"],
        "module_sha256": module_sha,
        "runner_sha256": runner_sha,
        "v4_resume_engine_sha256": config["source_pins"]["src/p2_restore/authoritative_nested_surrogate_execution_v4.py"],
        "exact_command_sha256": hashlib.sha256(command.encode()).hexdigest(),
        "semantic_preflight_sha256": semantic_sha,
        "scientific_preflight_v4_sha256": semantic["scientific_preflight_v4_sha256"],
        "observations_sha256": config["data_contract"]["sha256"],
        "observations_bytes": int(config["data_contract"]["bytes"]),
        "resolved_data_directory": str(APPROVED_DATA_DIRECTORY),
        "exact_command_contains_literal_data_dir": True,
        "required_authorization": required_auth,
        "supervised_common_ledger_sha256": semantic["supervised_common_ledger"]["ordered_time_sha256"],
        "component_hyperparameters_sha256": static["component_hyperparameters_sha256"],
        "meta_refit_sha256": static["meta_refit_sha256"],
        "postprocess_sha256": static["postprocess_sha256"],
        "family_views_sha256": static["family_views_sha256"],
        "metrics_sha256": static["metrics_sha256"],
        "execution_graph_sha256": static["execution_graph_sha256"],
        "semantic_preflight_before_clean_actual_namespace": True,
        "resume_read_only_audit_before_lock": True,
        "resume_strict_audit_and_semantic_preflight_under_lock_before_fit": True,
        "same_exact_command_clean_or_resume": True,
        "maximum_resume_attempts_after_initial": 2,
        "maximum_total_attempts": 3,
        "job_store_contract_sha256_is_this_seal_file_sha256": True,
        "v1_v2_v3_v4_resume_allowed": False,
        "result_based_rerun_or_tuning_allowed": False,
        "terminal_result_atomic_and_self_bound": True,
        "terminal_receipt_pins_exact_result_hash": True,
        "required_authorization_v5": {
            "schema_version": REQUIRED_AUTHORIZATION_SCHEMA_VERSION,
            "same_contract_crash_resume_authorized": True,
            "maximum_automatic_resume_attempts_after_initial": 2,
            "maximum_total_attempts": 3,
            "result_based_rerun_or_tuning_authorized": False,
            "v1_v2_v3_v4_job_or_cell_reuse_authorized": False,
            "single_scientific_contract_only": True,
            "single_execution_only": False,
        },
        "actual_model_fits": 0,
        "actual_scores": 0,
        "actual_predictions": 0,
        "authorization_receipt_created": False,
        "actual_namespace_created": False,
    }
    execution_plan = {
        "schema_version": "p2_authoritative_nested_surrogate_45cell_plan.v5",
        "status": "PASS_DATA_EXECUTABLE_RESUMABLE_PLAN_FIT_NOT_AUTHORIZED",
        **plan_receipt,
        "prefix_plans": [plan.summary() for plan in plans],
        "complete_pipeline_seeds": seeds,
        "component_order": list(COMPONENTS),
        "split_population": "supervised_common_ledger_v3_unchanged",
        "semantic_preflight_sha256": semantic_sha,
        "outer_evaluation_rows_per_fraction": 78156,
        "actual_fit_authorized": False,
    }
    qa = {
        "schema_version": "p2_authoritative_nested_surrogate_execution_ready_qa.v5",
        "status": "PASS_EXECUTION_READY_NOT_AUTHORIZED",
        "static": static["status"],
        "semantic_preflight": semantic["status"],
        "semantic_preflight_sha256": semantic_sha,
        "required_authorization": required_auth,
        "resolved_data_directory": str(APPROVED_DATA_DIRECTORY),
        "exact_command_contains_literal_data_dir": True,
        "environment_data_dir_dependency": False,
        "observations_sha256": resolved_receipt["observations_sha256"],
        "observations_bytes": resolved_receipt["observations_bytes"],
        "directory_entries_enumerated": 0,
        "outer_prefix_scopes_checked": semantic["outer_prefix_scopes_checked"],
        "inner_scopes_checked": semantic["inner_scopes_checked"],
        "minimum_support": semantic["minimum_support_across_all_scopes"],
        "outer_evaluation_rows_per_fraction": semantic["outer_evaluation_rows_per_fraction"],
        "tiny_full_cell_and_job_resume": tiny["status"],
        "v5_process_crash_resume_atomic_terminal": resume_fixture["status"],
        "v4_resume_engine_byte_pinned": True,
        "deep_cpu_forward": model_shape["status"],
        "evaluated_oof_atomic_publish": atomic_publish["status"],
        "exact_command_namespace": command_receipt["status"],
        "maximum_resume_attempts": 2,
        "deterministic_failure_auto_resume_allowed": False,
        "terminal_result_rerun_allowed": False,
        "v1_v2_v3_v4_job_or_cell_reuse_allowed": False,
        "actual_namespace_created": False,
        "actual_model_fits": 0,
        "actual_scores": 0,
        "actual_predictions": 0,
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
        "submission_files_generated": 0,
        "uploads": 0,
        "p3_process_mutations": 0,
        "authorization_receipt_created": False,
    }
    _write_json_exclusive(output_dir / "static_verification.json", static)
    _write_json_exclusive(output_dir / "resolved_data_dir_receipt.json", resolved_receipt)
    _write_json_exclusive(output_dir / "observations_access_receipt.json", data_receipt)
    _write_json_exclusive(output_dir / "actual_data_semantic_preflight.json", semantic)
    _write_json_exclusive(output_dir / "execution_plan.json", execution_plan)
    _write_json_exclusive(output_dir / "tiny_fixture_receipt.json", tiny)
    _write_json_exclusive(output_dir / "resume_control_fixture_receipt.json", resume_fixture)
    _write_json_exclusive(output_dir / "deep_model_shape_receipt.json", model_shape)
    _write_json_exclusive(output_dir / "atomic_publish_receipt.json", atomic_publish)
    _write_json_exclusive(output_dir / "exact_command_namespace_receipt.json", command_receipt)
    _write_json_exclusive(output_dir / "resource_estimate.json", resource)
    _write_json_exclusive(output_dir / "preexecution_seal.json", preexecution)
    _write_json_exclusive(output_dir / "qa.json", qa)
    _write_text_exclusive(output_dir / "REPORT_KO.md", _report_ko(semantic=semantic, command=command))
    manifest = _manifest(output_dir, config_path, static)
    _write_json_exclusive(output_dir / "manifest.json", manifest)
    return {
        "status": "EXECUTION_READY_NOT_AUTHORIZED",
        "output_dir": str(output_dir),
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "preexecution_seal_sha256": sha256_file(output_dir / "preexecution_seal.json"),
        "semantic_preflight_sha256": semantic_sha,
        "qa_sha256": sha256_file(output_dir / "qa.json"),
        "required_authorization_schema_version": REQUIRED_AUTHORIZATION_SCHEMA_VERSION,
        "actual_namespace_created": False,
        "actual_model_fits": 0,
    }


def _verify_runtime_policy() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _require(os.environ.get(name) == "4", f"{name} must be exactly 4")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "single GPU 0 is required")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(torch.cuda.device_count() == 1, "exactly one visible GPU is required")
    torch.set_num_threads(4)


def _verify_authorization_v5(
    authorization_path: Path, *, preexecution_seal_sha256: str, exact_command: str
) -> dict[str, Any]:
    value = verify_authorization(
        authorization_path,
        preexecution_seal_sha256=preexecution_seal_sha256,
        exact_command=exact_command,
    )
    _require(
        value.get("schema_version") == REQUIRED_AUTHORIZATION_SCHEMA_VERSION,
        "v5 authorization schema_version is not the exact required value",
    )
    _require(value.get("same_contract_crash_resume_authorized") is True, "v5 crash resume is not explicitly authorized")
    _require(value.get("maximum_automatic_resume_attempts_after_initial") == 2, "v5 authorization resume budget differs")
    _require(value.get("maximum_total_attempts") == 3, "v5 authorization total attempt budget differs")
    _require(value.get("result_based_rerun_or_tuning_authorized") is False, "v5 authorization permits result rerun/tuning")
    _require(value.get("v1_v2_v3_v4_job_or_cell_reuse_authorized") is False, "v5 authorization permits foreign JobStore reuse")
    _require(value.get("single_scientific_contract_only") is True, "v5 authorization does not bind one scientific contract")
    _require(value.get("single_execution_only") is False, "v5 authorization contradicts bounded crash resume")
    return value


def execute_actual(
    config_path: Path,
    *,
    data_dir: Path,
    preexecution_seal_path: Path,
    authorization_path: Path,
) -> dict[str, Any]:
    config, recipe, static = _verify_static(config_path)
    _verify_resolved_data_dir(data_dir, config)
    _verify_runtime_policy()
    command = str(config["exact_command"])
    seal_path = preexecution_seal_path.resolve(strict=True)
    authorization_path = authorization_path.resolve(strict=True)
    module_sha = sha256_file(_resolve_repo_path(MODULE_RELATIVE))
    runner_sha = sha256_file(Path(__file__).resolve())
    seal = verify_preexecution_seal(
        seal_path,
        config_sha256=static["config_sha256"],
        module_sha256=module_sha,
        runner_sha256=runner_sha,
        exact_command=command,
    )
    _require(seal.get("schema_version") == "p2_authoritative_nested_surrogate_preexecution_seal.v5", "v5 preexecution seal schema changed")
    _require(seal.get("required_authorization", {}).get("schema_version") == REQUIRED_AUTHORIZATION_SCHEMA_VERSION, "v5 seal authorization schema binding changed")
    _require(seal.get("resolved_data_directory") == str(APPROVED_DATA_DIRECTORY), "v5 seal data directory changed")
    _verify_authorization_v5(
        authorization_path,
        preexecution_seal_sha256=sha256_file(seal_path),
        exact_command=command,
    )
    actual_dir = (PROJECT_ROOT / config["output"]["actual_directory"]).resolve()
    binding = ExecutionBindingV5(
        namespace=actual_dir.name,
        execution_contract_sha256=static["config_sha256"],
        parent_recipe_sha256=config["parent_contract"]["sha256"],
        preexecution_seal_sha256=sha256_file(seal_path),
        semantic_preflight_sha256=seal["semantic_preflight_sha256"],
        exact_command_sha256=hashlib.sha256(command.encode()).hexdigest(),
        authorization_sha256=sha256_file(authorization_path),
        module_sha256=module_sha,
        runner_sha256=runner_sha,
    )

    def semantic() -> SemanticPreflightOutcomeV5:
        _verify_resolved_data_dir(data_dir, config)
        observations, _ = v3_runner._load_observations(data_dir, config)
        plans, receipt = semantic_preflight_actual_data_v5(observations, recipe=recipe, config=config)
        _verify_semantic_gates(config, receipt)
        _require(receipt["semantic_receipt_sha256"] == seal["semantic_preflight_sha256"], "runtime v5 semantic preflight differs from sealed readiness")
        return SemanticPreflightOutcomeV5(str(receipt["semantic_receipt_sha256"]), (observations, plans))

    def execute(context: Any, contract_sha256: str) -> dict[str, Any]:
        observations, plans = context
        return execute_authorized_curve_v5(
            observations=observations,
            plans=plans,
            parent_recipe=recipe,
            config=config,
            output_dir=actual_dir,
            contract_sha256=contract_sha256,
        )

    result = run_resumable_execution_v5(
        actual_dir=actual_dir,
        binding=binding,
        semantic_preflight=semantic,
        execute_curve=execute,
    )
    return {**result, "actual_output_dir": str(actual_dir), "submission_files_generated": 0, "uploads": 0}


def _default_data_dir(config_path: Path) -> Path:
    overlay = _read_json(config_path.resolve(strict=True))
    contract = overlay.get("resolved_data_dir_contract")
    _require(isinstance(contract, dict), "v5 config lacks resolved data-dir contract")
    value = contract.get("absolute_path")
    _require(value == str(APPROVED_DATA_DIRECTORY), "v5 config default data directory changed")
    return Path(str(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_READY_OUTPUT)
    parser.add_argument("--seal-readiness", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preexecution-seal", type=Path)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    _require(not (args.seal_readiness and args.execute), "choose one v5 execution mode")
    data_dir = args.data_dir
    if (args.execute or args.seal_readiness) and data_dir is None:
        data_dir = _default_data_dir(args.config)
    if args.execute:
        assert data_dir is not None
        _require(args.preexecution_seal is not None, "v5 preexecution seal is required")
        _require(args.authorization is not None, "v5 authorization receipt is required")
        result = execute_actual(
            args.config,
            data_dir=data_dir,
            preexecution_seal_path=args.preexecution_seal,
            authorization_path=args.authorization,
        )
    elif args.seal_readiness:
        assert data_dir is not None
        result = seal_readiness(args.config, data_dir=data_dir, output_dir=args.output_dir)
    else:
        config, _, static = _verify_static(args.config)
        result = {
            "status": "PASS_V5_STATIC_ONLY_NO_FIT",
            "config_sha256": static["config_sha256"],
            "exact_command_sha256": hashlib.sha256(str(config["exact_command"]).encode()).hexdigest(),
            "required_authorization_schema_version": REQUIRED_AUTHORIZATION_SCHEMA_VERSION,
            "resolved_data_directory": str(APPROVED_DATA_DIRECTORY),
            "actual_model_fits": 0,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
