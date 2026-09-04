#!/usr/bin/env python
"""Seal or run the crash-resumable P2 v4 authoritative 45-cell contract.

V4 inherits every scientific surface from sealed v3.  It adds only a new,
isolated execution namespace with an immutable start receipt, strict JobStore
audits, two same-command crash resumes, deterministic fail-closed receipts,
and atomic terminal publication.  There is no official test/sample/submission
or submission-generation path in this runner.
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
from p2_restore.authoritative_nested_surrogate_conformance import COMPONENTS  # noqa: E402
from p2_restore.authoritative_nested_surrogate_execution import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    temporary_tiny_fixture,
    verify_authorization,
    verify_preexecution_seal,
)
from p2_restore.authoritative_nested_surrogate_execution_v4 import (  # noqa: E402
    ExecutionBindingV4,
    SemanticPreflightOutcomeV4,
    TERMINAL_STATUS,
    execute_authorized_curve_v4,
    inspect_actual_namespace_read_only,
    run_resumable_execution_v4,
    semantic_preflight_actual_data_v4,
)

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/p2_authoritative_nested_surrogate_execution_20260825_v4.json"
)
DEFAULT_READY_OUTPUT = (
    PROJECT_ROOT / "artifacts/p2_authoritative_nested_surrogate_execution_ready_20260825_v4"
)
EXPECTED_CONFIG_SHA256 = "b7533ba1e435c25de9d0cf6e10b8febbed2637f4677a95b813ac6de3f5f9d466"
MODULE_RELATIVE = "src/p2_restore/authoritative_nested_surrogate_execution_v4.py"
RUNNER_RELATIVE = "scripts/run_p2_authoritative_nested_surrogate_45cell_v4.py"


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


def _load_recipe_v4(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    overlay = _read_json(path)
    base_pin = dict(overlay["base_recipe"])
    base_path = _resolve_repo_path(str(base_pin["path"]))
    _require(sha256_file(base_path) == base_pin["sha256"], "v4 base recipe changed")
    base, _ = v3_runner._load_recipe(base_path)
    return base_runner._deep_merge(base, overlay), base_pin


def _load_config_v4(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    overlay = _read_json(path)
    base_pin = dict(overlay["base_config"])
    base_path = _resolve_repo_path(str(base_pin["path"]))
    _require(sha256_file(base_path) == base_pin["sha256"], "v4 base config changed")
    base, _ = v3_runner._load_config_v3(base_path)
    return base_runner._deep_merge(base, overlay), base_pin


def _verify_static(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve(strict=True)
    config_sha = sha256_file(config_path)
    if EXPECTED_CONFIG_SHA256 != "TO_BE_SEALED_V4":
        _require(config_sha == EXPECTED_CONFIG_SHA256, "v4 execution config changed")
    overlay = _read_json(config_path)
    base_path = _resolve_repo_path(str(overlay["base_config"]["path"]))
    base_config, base_recipe, base_static = v3_runner._verify_static(base_path)
    config, base_pin = _load_config_v4(config_path)
    _require(
        config["status"] == "SEALED_EXECUTION_READY_DRY_RUN_NO_ACTUAL_FIT",
        "v4 execution config is not sealed",
    )
    _require(
        all(value is False for value in config["permissions"].values()),
        "v4 readiness grants a forbidden permission",
    )
    parent = config["parent_contract"]
    pins = {
        parent["path"]: parent["sha256"],
        parent["contract_seal_path"]: parent["contract_seal_sha256"],
        parent["decisive_recon_preregistration_path"]: parent[
            "decisive_recon_preregistration_sha256"
        ],
        parent["common_protocol_path"]: parent["common_protocol_sha256"],
        config["completed_conformance"]["config_path"]: config["completed_conformance"][
            "config_sha256"
        ],
        config["completed_conformance"]["manifest_path"]: config[
            "completed_conformance"
        ]["manifest_sha256"],
        config["completed_conformance"]["qa_path"]: config["completed_conformance"][
            "qa_sha256"
        ],
        config["supersedes"]["external_independent_qa"]["path"]: config[
            "supersedes"
        ]["external_independent_qa"]["sha256"],
        **config["source_pins"],
        **config["hyperparameter_evidence_pins"],
    }
    verified: dict[str, Any] = {}
    for relative, expected in pins.items():
        current = _resolve_repo_path(str(relative))
        actual = sha256_file(current)
        _require(actual == expected, f"v4 pinned source/evidence changed: {relative}")
        verified[str(relative)] = {"sha256": actual, "bytes": current.stat().st_size}
    recipe, recipe_base_pin = _load_recipe_v4(_resolve_repo_path(parent["path"]))
    nested = recipe["authoritative_nested_surrogate_recipe"]
    base_nested = base_recipe["authoritative_nested_surrogate_recipe"]
    _require(recipe["training_authorized"] is False, "v4 parent recipe authorizes fit")
    _require(
        nested["identity"] == "P2_AUTHORITATIVE_NESTED_CAUSAL_SURROGATE_V4_RESUMABLE",
        "v4 recipe identity changed",
    )
    for key in (
        "component_hyperparameters",
        "meta_refit",
        "postprocess",
        "family_views",
        "metrics",
        "execution_graph",
    ):
        _require(config[key] == base_config[key], f"v4 science surface changed: {key}")
    for key in (
        "outer_fold_contract",
        "chronological_prefix_contract",
        "nested_component_oof_contract",
        "deep_public_context_contract",
        "complete_pipeline_seed_contract",
        "epoch_and_meta_contract",
        "postprocess_contract",
    ):
        _require(nested[key] == base_nested[key], f"v4 recipe science surface changed: {key}")
    resume = config["checkpoint_resume"]
    _require(resume["maximum_resume_attempts_after_initial"] == 2, "v4 resume budget changed")
    _require(resume["maximum_total_attempts"] == 3, "v4 total attempt budget changed")
    _require(resume["same_exact_command_clean_or_resume"] is True, "v4 command mode changed")
    _require(resume["v1_v2_v3_job_reuse_allowed"] is False, "foreign jobs allowed")
    _require(
        config["output"]["actual_directory"]
        == "artifacts/p2_authoritative_nested_surrogate_actual_20260825_v4",
        "v4 actual namespace changed",
    )
    _require(
        config["metrics"]["expected_evaluation_rows_per_fraction"] == 78156,
        "v4 outer population changed",
    )
    return config, recipe, {
        "status": "PASS_V4_STATIC_PINS_AND_BYTE_IDENTICAL_SCIENCE_SURFACE",
        "config_sha256": config_sha,
        "base_config": base_pin,
        "base_recipe": recipe_base_pin,
        "base_v3_static_status": base_static["status"],
        "verified_input_count": len(verified),
        "verified_inputs": verified,
        "recipe_identity": nested["identity"],
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
        "only_crash_resume_control_plane_changed": True,
    }


def _verify_semantic_gates(config: dict[str, Any], receipt: dict[str, Any]) -> None:
    v3_runner._verify_semantic_gates(config, receipt)
    _require(
        receipt["operational_revision"]["science_surface_changed"] is False,
        "v4 semantic wrapper changed science",
    )
    _require(
        receipt["scientific_preflight_v3_sha256"]
        == config["semantic_preflight_v4"]["expected_scientific_v3_sha256"],
        "v4 inherited v3 semantic digest changed",
    )
    _require(
        receipt["outer_evaluation_rows_per_fraction"] == 78156,
        "v4 semantic outer population changed",
    )


def _command_namespace(command: str) -> dict[str, Any]:
    fragments = (
        "run_p2_authoritative_nested_surrogate_45cell_v4.py",
        "p2_authoritative_nested_surrogate_execution_20260825_v4.json",
        "p2_authoritative_nested_surrogate_execution_ready_20260825_v4\\preexecution_seal.json",
        "p2_authoritative_nested_surrogate_execution_ready_20260825_v4\\EXECUTION_AUTHORIZATION.json",
    )
    _require(all(fragment in command for fragment in fragments), "v4 command namespace is incomplete")
    for foreign in (
        "actual_20260825_v1",
        "actual_20260825_v2",
        "actual_20260825_v3",
    ):
        _require(foreign not in command, "v4 command references a foreign actual namespace")
    return {
        "status": "PASS_SAME_EXACT_COMMAND_CLEAN_OR_RESUME_V4_ONLY",
        "required_fragments": list(fragments),
        "foreign_actual_namespace_present": False,
        "actual_model_fits": 0,
    }


class _FixtureDeath(BaseException):
    pass


def _resume_control_fixture() -> dict[str, Any]:
    """Synthetic control-plane check; no production model, prediction, or score."""

    with tempfile.TemporaryDirectory(prefix="p2_v4_resume_fixture_") as raw:
        actual = Path(raw) / "fixture_actual_v4"
        binding = ExecutionBindingV4(
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

        def preflight() -> SemanticPreflightOutcomeV4:
            return SemanticPreflightOutcomeV4("4" * 64, None)

        try:
            run_resumable_execution_v4(
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
        final = run_resumable_execution_v4(
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
            "status": "PASS_SYNTHETIC_CRASH_SAME_COMMAND_RESUME_ATOMIC_TERMINAL",
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
    return f"""# P2 authoritative nested surrogate v4 실행 준비 보고서

## 결론

판정은 `EXECUTION_READY_NOT_AUTHORIZED`이다. v3의 supervised common ledger·분할·모델·하이퍼파라미터·meta·후처리·지표·78,156행 모집단은 바꾸지 않았다. v3 독립 QA가 지적한 한 가지 실행 결함, 즉 실제 디렉터리가 생긴 뒤 동일 명령으로 JobStore를 재개할 수 없던 제어 흐름만 v4의 별도 namespace에서 고쳤다.

실데이터 production parser와 정확한 scientific adapter로 15개 prefix/45개 inner를 다시 검사했고 fit/prediction/score는 0/0/0이다. 공통 supervised ledger는 {ledger['common_supervised_time_count']:,}시각이며 SHA-256은 `{ledger['ordered_time_sha256']}`다.

## 실행 가능성

- 최소 router train/validation support: 층당 {minima['router_train_rows_per_layer']:,}/{minima['router_validation_rows_per_layer']:,}행
- 최소 deep train/validation support: 층당 {minima['deep_train_rows_per_layer']:,}/{minima['deep_validation_rows_per_layer']:,}행
- 최소 deep supervised chunk: inner {minima['deep_supervised_chunks']:,}, full {minima['full_deep_supervised_chunks']:,}
- 최소 meta OOF support: 층당 {minima['meta_oof_rows_per_layer']:,}행
- 비율당 outer 평가 모집단: {semantic['outer_evaluation_rows_per_fraction']:,}행

## v4 crash-resume 계약

clean start는 actual namespace 생성 전에 semantic preflight를 통과해야 한다. resume은 기존 namespace를 read-only 검사한 뒤 nonblocking lock을 얻고, lock 아래에서 모든 job/cell manifest의 contract·file hash·bytes와 result 부재를 재검사한다. initial 1회 뒤 자동 resume은 최대 2회다. 명시적 transient 또는 terminal receipt가 없는 abrupt stop만 resume할 수 있고, graceful deterministic failure는 append-only failure receipt로 닫힌다.

`result.json`은 execution/seal/semantic/attempt count를 자체 포함하고 unique fsynced partial에서 atomic rename된다. terminal receipt는 result SHA-256과 bytes를 고정한다. result rename 직후 중단된 경우에는 model 실행이나 resume budget 소모 없이 receipt만 finalization한다. 이미 유효한 terminal result가 있으면 재실행은 금지된다.

## 별도 승인 후 clean start와 crash resume에 동일하게 쓰는 명령

```powershell
{command}
```

이번 준비에서는 authorization을 만들지 않았다. 공식 P2 test/sample/submission 및 submission candidate를 읽지 않았고, submission 생성·업로드와 P3 변경도 0회다.
"""


def _manifest(output_dir: Path, config_path: Path, static: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "manifest.json":
            outputs[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {
        "schema_version": "p2_authoritative_nested_surrogate_execution_ready_manifest.v4",
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "status": "EXECUTION_READY_NOT_AUTHORIZED",
        "config": {
            "path": config_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "sha256": static["config_sha256"],
        },
        "module": {"path": MODULE_RELATIVE, "sha256": sha256_file(_resolve_repo_path(MODULE_RELATIVE))},
        "runner": {"path": RUNNER_RELATIVE, "sha256": sha256_file(Path(__file__).resolve())},
        "outputs": outputs,
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
    observations, data_receipt = v3_runner._load_observations(data_dir, config)
    plans, semantic = semantic_preflight_actual_data_v4(
        observations, recipe=recipe, config=config
    )
    _verify_semantic_gates(config, semantic)
    seeds = [
        int(value)
        for value in recipe["authoritative_nested_surrogate_recipe"][
            "complete_pipeline_seed_contract"
        ]["seeds"]
    ]
    plan_receipt = base_runner._seed_plan_receipt(list(plans), seeds)
    tiny = temporary_tiny_fixture(plans[0])
    resume_fixture = _resume_control_fixture()
    model_shape = base_runner._model_shape_receipt(config)
    atomic_publish = base_runner._atomic_publish_fixture()
    resource = base_runner._resource_receipt(config)
    command = str(config["exact_command"])
    command_receipt = _command_namespace(command)
    module_sha = sha256_file(_resolve_repo_path(MODULE_RELATIVE))
    runner_sha = sha256_file(Path(__file__).resolve())
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), "v4 readiness output already exists")
    output_dir.mkdir(parents=True, exist_ok=False)
    semantic_sha = str(semantic["semantic_receipt_sha256"])
    preexecution = {
        "schema_version": "p2_authoritative_nested_surrogate_preexecution_seal.v4",
        "status": "EXECUTION_READY_NOT_AUTHORIZED",
        "config_sha256": static["config_sha256"],
        "parent_recipe_sha256": config["parent_contract"]["sha256"],
        "module_sha256": module_sha,
        "runner_sha256": runner_sha,
        "exact_command_sha256": hashlib.sha256(command.encode()).hexdigest(),
        "semantic_preflight_sha256": semantic_sha,
        "scientific_preflight_v3_sha256": semantic["scientific_preflight_v3_sha256"],
        "observations_sha256": config["data_contract"]["sha256"],
        "supervised_common_ledger_sha256": semantic["supervised_common_ledger"][
            "ordered_time_sha256"
        ],
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
        "v1_v2_v3_resume_allowed": False,
        "terminal_result_atomic_and_self_bound": True,
        "terminal_receipt_pins_exact_result_hash": True,
        "required_authorization_v4": {
            "same_contract_crash_resume_authorized": True,
            "maximum_automatic_resume_attempts_after_initial": 2,
            "maximum_total_attempts": 3,
            "result_based_rerun_or_tuning_authorized": False,
            "v1_v2_v3_job_or_cell_reuse_authorized": False,
            "single_scientific_contract_only": True,
            "single_execution_only": False
        },
        "actual_model_fits": 0,
        "actual_scores": 0,
        "actual_predictions": 0,
        "authorization_receipt_created": False,
    }
    execution_plan = {
        "schema_version": "p2_authoritative_nested_surrogate_45cell_plan.v4",
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
        "schema_version": "p2_authoritative_nested_surrogate_execution_ready_qa.v4",
        "status": "PASS_EXECUTION_READY_NOT_AUTHORIZED",
        "static": static["status"],
        "semantic_preflight": semantic["status"],
        "semantic_preflight_sha256": semantic_sha,
        "outer_prefix_scopes_checked": semantic["outer_prefix_scopes_checked"],
        "inner_scopes_checked": semantic["inner_scopes_checked"],
        "minimum_support": semantic["minimum_support_across_all_scopes"],
        "outer_evaluation_rows_per_fraction": semantic[
            "outer_evaluation_rows_per_fraction"
        ],
        "tiny_full_cell_and_job_resume": tiny["status"],
        "v4_process_crash_resume_atomic_terminal": resume_fixture["status"],
        "deep_cpu_forward": model_shape["status"],
        "evaluated_oof_atomic_publish": atomic_publish["status"],
        "exact_command_namespace": command_receipt["status"],
        "maximum_resume_attempts": 2,
        "deterministic_failure_auto_resume_allowed": False,
        "terminal_result_rerun_allowed": False,
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
    _write_text_exclusive(
        output_dir / "REPORT_KO.md", _report_ko(semantic=semantic, command=command)
    )
    manifest = _manifest(output_dir, config_path, static)
    _write_json_exclusive(output_dir / "manifest.json", manifest)
    return {
        "status": "EXECUTION_READY_NOT_AUTHORIZED",
        "output_dir": str(output_dir),
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "preexecution_seal_sha256": sha256_file(output_dir / "preexecution_seal.json"),
        "semantic_preflight_sha256": semantic_sha,
        "qa_sha256": sha256_file(output_dir / "qa.json"),
        "actual_model_fits": 0,
    }


def _verify_runtime_policy() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _require(os.environ.get(name) == "4", f"{name} must be exactly 4")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "single GPU 0 is required")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(torch.cuda.device_count() == 1, "exactly one visible GPU is required")
    torch.set_num_threads(4)


def _verify_authorization_v4(
    authorization_path: Path, *, preexecution_seal_sha256: str, exact_command: str
) -> dict[str, Any]:
    value = verify_authorization(
        authorization_path,
        preexecution_seal_sha256=preexecution_seal_sha256,
        exact_command=exact_command,
    )
    _require(
        value.get("same_contract_crash_resume_authorized") is True,
        "v4 same-contract crash resume is not explicitly authorized",
    )
    _require(
        value.get("maximum_automatic_resume_attempts_after_initial") == 2,
        "v4 authorization resume budget differs",
    )
    _require(
        value.get("maximum_total_attempts") == 3,
        "v4 authorization total attempt budget differs",
    )
    _require(
        value.get("result_based_rerun_or_tuning_authorized") is False,
        "v4 authorization permits result-based rerun or tuning",
    )
    _require(
        value.get("v1_v2_v3_job_or_cell_reuse_authorized") is False,
        "v4 authorization permits foreign JobStore reuse",
    )
    _require(
        value.get("single_scientific_contract_only") is True,
        "v4 authorization does not bind one scientific contract",
    )
    _require(
        value.get("single_execution_only") in {None, False},
        "legacy single_execution_only contradicts bounded crash resume",
    )
    return value


def execute_actual(
    config_path: Path,
    *,
    data_dir: Path,
    preexecution_seal_path: Path,
    authorization_path: Path,
) -> dict[str, Any]:
    config, recipe, static = _verify_static(config_path)
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
    _require(seal["schema_version"].endswith(".v4"), "v4 preexecution seal schema changed")
    _verify_authorization_v4(
        authorization_path,
        preexecution_seal_sha256=sha256_file(seal_path),
        exact_command=command,
    )
    actual_dir = (PROJECT_ROOT / config["output"]["actual_directory"]).resolve()
    binding = ExecutionBindingV4(
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

    def semantic() -> SemanticPreflightOutcomeV4:
        observations, _ = v3_runner._load_observations(data_dir, config)
        plans, receipt = semantic_preflight_actual_data_v4(
            observations, recipe=recipe, config=config
        )
        _verify_semantic_gates(config, receipt)
        _require(
            receipt["semantic_receipt_sha256"] == seal["semantic_preflight_sha256"],
            "runtime v4 semantic preflight differs from sealed readiness",
        )
        return SemanticPreflightOutcomeV4(
            str(receipt["semantic_receipt_sha256"]), (observations, plans)
        )

    def execute(context: Any, contract_sha256: str) -> dict[str, Any]:
        observations, plans = context
        return execute_authorized_curve_v4(
            observations=observations,
            plans=plans,
            parent_recipe=recipe,
            config=config,
            output_dir=actual_dir,
            contract_sha256=contract_sha256,
        )

    result = run_resumable_execution_v4(
        actual_dir=actual_dir,
        binding=binding,
        semantic_preflight=semantic,
        execute_curve=execute,
    )
    return {
        **result,
        "actual_output_dir": str(actual_dir),
        "submission_files_generated": 0,
        "uploads": 0,
    }


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
    _require(not (args.seal_readiness and args.execute), "choose one v4 execution mode")
    data_dir = args.data_dir
    if (args.execute or args.seal_readiness) and data_dir is None:
        raw = os.environ.get("P2_DATA_DIR")
        _require(bool(raw), "set P2_DATA_DIR or pass --data-dir")
        data_dir = Path(str(raw))
    if args.execute:
        assert data_dir is not None
        _require(args.preexecution_seal is not None, "v4 preexecution seal is required")
        _require(args.authorization is not None, "v4 authorization receipt is required")
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
            "status": "PASS_V4_STATIC_ONLY_NO_FIT",
            "config_sha256": static["config_sha256"],
            "exact_command_sha256": hashlib.sha256(str(config["exact_command"]).encode()).hexdigest(),
            "actual_model_fits": 0,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
