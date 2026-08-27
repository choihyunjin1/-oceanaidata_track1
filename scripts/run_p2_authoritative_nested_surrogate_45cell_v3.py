#!/usr/bin/env python
"""Seal or execute the repaired P2 v3 authoritative 45-cell comparison.

Readiness is based on a zero-fit semantic preflight over the pinned production
``observations.csv``.  In execute mode the same preflight must reproduce its
sealed digest before the actual directory, lock, partial, or model fit exists.
This runner has no official test/sample/submission or submission-generation
path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts import run_p2_authoritative_nested_surrogate_45cell_v1 as base_runner  # noqa: E402
from p2_restore.authoritative_nested_surrogate_conformance import COMPONENTS  # noqa: E402
from p2_restore.authoritative_nested_surrogate_execution import (  # noqa: E402
    canonical_sha256,
    process_lock,
    sha256_file,
    temporary_tiny_fixture,
    verify_authorization,
    verify_preexecution_seal,
)
from p2_restore.authoritative_nested_surrogate_execution_v3 import (  # noqa: E402
    execute_authorized_curve_v3,
    semantic_preflight_actual_data,
)

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/p2_authoritative_nested_surrogate_execution_20260825_v3.json"
)
DEFAULT_READY_OUTPUT = (
    PROJECT_ROOT / "artifacts/p2_authoritative_nested_surrogate_execution_ready_20260825_v3"
)
EXPECTED_CONFIG_SHA256 = "7eba50355d747a0306df5973447eba9c29ae0d240ba7def3887660b827d1d682"
MODULE_RELATIVE = "src/p2_restore/authoritative_nested_surrogate_execution_v3.py"
RUNNER_RELATIVE = "scripts/run_p2_authoritative_nested_surrogate_45cell_v3.py"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_exclusive(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as handle:
        handle.write(payload)


def _write_text_exclusive(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _resolve_repo_path(relative: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve(strict=True)
    path.relative_to(PROJECT_ROOT.resolve(strict=True))
    return path


def _load_recipe(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    overlay = _read_json(path)
    base_pin = dict(overlay["base_recipe"])
    base_path = _resolve_repo_path(str(base_pin["path"]))
    _require(sha256_file(base_path) == base_pin["sha256"], "v3 base recipe changed")
    merged = base_runner._deep_merge(_read_json(base_path), overlay)
    return merged, base_pin


def _load_config_v3(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    overlay = _read_json(path)
    base_pin = dict(overlay["base_config"])
    base_path = _resolve_repo_path(str(base_pin["path"]))
    _require(sha256_file(base_path) == base_pin["sha256"], "v3 base config changed")
    base, _ = base_runner._load_config(base_path)
    return base_runner._deep_merge(base, overlay), base_pin


def _verify_static(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve(strict=True)
    config_sha = sha256_file(config_path)
    if EXPECTED_CONFIG_SHA256 != "TO_BE_SEALED_V3":
        _require(config_sha == EXPECTED_CONFIG_SHA256, "v3 execution config changed")
    config, base_pin = _load_config_v3(config_path)
    _require(
        config["status"] == "SEALED_EXECUTION_READY_DRY_RUN_NO_ACTUAL_FIT",
        "v3 execution config is not sealed",
    )
    _require(
        all(value is False for value in config["permissions"].values()),
        "v3 readiness grants a forbidden permission",
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
        **config["source_pins"],
        **config["hyperparameter_evidence_pins"],
    }
    verified: dict[str, Any] = {}
    for relative, expected in pins.items():
        current = _resolve_repo_path(str(relative))
        actual = sha256_file(current)
        _require(actual == expected, f"v3 pinned source/evidence changed: {relative}")
        verified[str(relative)] = {"sha256": actual, "bytes": current.stat().st_size}
    recipe, recipe_base_pin = _load_recipe(_resolve_repo_path(parent["path"]))
    nested = recipe["authoritative_nested_surrogate_recipe"]
    _require(recipe["training_authorized"] is False, "v3 parent recipe authorizes fit")
    _require(
        nested["identity"] == "P2_AUTHORITATIVE_NESTED_CAUSAL_SURROGATE_V3",
        "v3 recipe identity changed",
    )
    _require(
        nested["chronological_prefix_contract"]["unit"]
        == "unique supervised-eligible common-ledger timestamp",
        "v3 supervised fraction denominator changed",
    )
    base_config_path = _resolve_repo_path(str(base_pin["path"]))
    base_config, _ = base_runner._load_config(base_config_path)
    for key in (
        "component_hyperparameters",
        "meta_refit",
        "postprocess",
        "family_views",
        "metrics",
        "execution_graph",
    ):
        _require(config[key] == base_config[key], f"v3 immutable execution surface changed: {key}")
    base_recipe = _read_json(_resolve_repo_path(str(recipe_base_pin["path"])))
    base_nested = base_recipe["authoritative_nested_surrogate_recipe"]
    _require(
        nested["outer_fold_contract"]["folds"] == base_nested["outer_fold_contract"]["folds"],
        "v3 outer windows changed",
    )
    _require(
        nested["chronological_prefix_contract"]["fractions"]
        == base_nested["chronological_prefix_contract"]["fractions"],
        "v3 fractions changed",
    )
    _require(
        nested["complete_pipeline_seed_contract"]["seeds"]
        == base_nested["complete_pipeline_seed_contract"]["seeds"],
        "v3 complete seeds changed",
    )
    _require(
        config["execution_graph"]["top_level_component_jobs"] == 900,
        "v3 component DAG changed",
    )
    return config, recipe, {
        "status": "PASS_V3_STATIC_PINS_AND_IMMUTABLE_SCIENCE_SURFACE",
        "config_sha256": config_sha,
        "base_config": base_pin,
        "base_recipe": recipe_base_pin,
        "verified_input_count": len(verified),
        "verified_inputs": verified,
        "recipe_identity": nested["identity"],
        "component_hyperparameters_sha256": canonical_sha256(
            config["component_hyperparameters"]
        ),
        "meta_refit_sha256": canonical_sha256(config["meta_refit"]),
        "postprocess_sha256": canonical_sha256(config["postprocess"]),
        "family_views_sha256": canonical_sha256(config["family_views"]),
        "metrics_sha256": canonical_sha256(config["metrics"]),
        "outer_windows_unchanged": True,
        "fractions_unchanged": True,
        "complete_pipeline_seeds_unchanged": True,
        "model_components_and_hyperparameters_unchanged": True,
        "postprocess_and_metrics_unchanged": True,
    }


def _load_observations(data_dir: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = base_runner._observation_path(data_dir, config)
    expected_columns = list(config["data_contract"]["semantic_preflight_columns"])
    frame = pd.read_csv(path)
    _require(list(frame.columns) == expected_columns, "v3 observation schema changed")
    _require(
        not frame[["station", "year", "layer", "time"]].isna().any().any(),
        "v3 observation keys are missing",
    )
    return frame, {
        "status": "PASS_PINNED_OBSERVATIONS_ONLY",
        "files_opened": ["observations.csv"],
        "columns_read": expected_columns,
        "rows": len(frame),
        "observations_sha256": config["data_contract"]["sha256"],
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
    }


def _verify_semantic_gates(config: dict[str, Any], receipt: dict[str, Any]) -> None:
    gates = config["semantic_preflight_gates"]
    ledger = receipt["supervised_common_ledger"]
    minima = receipt["minimum_support_across_all_scopes"]
    expected = config["supervised_population_v3"]
    _require(receipt["status"] == "PASS_DATA_EXECUTABLE_ZERO_FIT", "v3 preflight failed")
    _require(
        ledger["common_supervised_time_count"] == expected["expected_common_supervised_time_count"],
        "v3 common-ledger count changed",
    )
    _require(
        ledger["ordered_time_sha256"] == expected["expected_ordered_time_sha256"],
        "v3 common-ledger digest changed",
    )
    checks = {
        "router_train_rows_per_layer": "minimum_router_train_rows_per_layer",
        "router_validation_rows_per_layer": "minimum_router_validation_rows_per_layer",
        "deep_train_rows_per_layer": "minimum_deep_train_rows_per_layer",
        "deep_validation_rows_per_layer": "minimum_deep_validation_rows_per_layer",
        "deep_supervised_chunks": "minimum_deep_supervised_chunks",
        "meta_oof_rows_per_layer": "minimum_meta_oof_rows_per_layer",
        "full_deep_supervised_chunks": "minimum_full_deep_supervised_chunks",
    }
    for observed, threshold in checks.items():
        _require(int(minima[observed]) >= int(gates[threshold]), f"v3 support gate failed: {observed}")
    state_threshold = int(gates["minimum_router_state_partition_rows"])
    _require(
        int(minima["router_mixed_partition_rows"]) >= state_threshold
        and int(minima["router_stratified_partition_rows"]) >= state_threshold,
        "v3 router state support gate failed",
    )
    _require(
        receipt["outer_evaluation_rows_per_fraction"]
        == int(gates["expected_outer_evaluation_rows_per_fraction"]),
        "v3 outer population gate failed",
    )
    _require(
        (receipt["component_model_fits"], receipt["predictions_materialized"], receipt["scores_computed"])
        == (0, 0, 0),
        "v3 semantic preflight performed forbidden work",
    )
    preimage = dict(receipt)
    claimed = preimage.pop("semantic_receipt_sha256")
    _require(canonical_sha256(preimage) == claimed, "v3 semantic receipt self-hash changed")


def _command_namespace(command: str) -> dict[str, Any]:
    fragments = (
        "run_p2_authoritative_nested_surrogate_45cell_v3.py",
        "p2_authoritative_nested_surrogate_execution_20260825_v3.json",
        "p2_authoritative_nested_surrogate_execution_ready_20260825_v3\\preexecution_seal.json",
        "p2_authoritative_nested_surrogate_execution_ready_20260825_v3\\EXECUTION_AUTHORIZATION.json",
    )
    _require(all(fragment in command for fragment in fragments), "v3 command namespace is incomplete")
    _require("actual_20260825_v2" not in command, "v3 command references v2 actual namespace")
    return {
        "status": "PASS_EXACT_COMMAND_PINS_V3_ONLY",
        "required_fragments": list(fragments),
        "v1_or_v2_actual_namespace_present": False,
        "actual_model_fits": 0,
    }


def _report_ko(
    *,
    semantic: dict[str, Any],
    command: str,
) -> str:
    ledger = semantic["supervised_common_ledger"]
    minima = semantic["minimum_support_across_all_scopes"]
    return f"""# P2 authoritative nested surrogate v3 실행 준비 보고서

## 결론

판정은 `EXECUTION_READY_NOT_AUTHORIZED`이다. v2의 실제 실패 원인이었던 public-only 시각 기반 split을 제거하고, 목표층 2·3·4의 TEMP·PSAL이 모두 유한하면서 router/deep 양쪽이 실행 가능한 {ledger['common_supervised_time_count']:,}개 공통 시각 ledger로 5개 비율과 45개 inner 범위를 다시 사전등록했다. 실제 production parser와 정확한 v3 backend adapter로 전 범위를 검사했으며 model fit/prediction/score는 0/0/0이다.

## 실행 가능성의 실제 증거

- 공통 ledger: {ledger['first_kst']} ~ {ledger['last_kst']}, SHA-256 `{ledger['ordered_time_sha256']}`
- 45개 inner 최소 router train/validation support: 층당 {minima['router_train_rows_per_layer']:,} / {minima['router_validation_rows_per_layer']:,}행
- 45개 inner 최소 deep train/validation support: 층당 {minima['deep_train_rows_per_layer']:,} / {minima['deep_validation_rows_per_layer']:,}행
- 최소 deep supervised chunk: {minima['deep_supervised_chunks']:,}개; full-prefix 최소 {minima['full_deep_supervised_chunks']:,}개
- 최소 router mixed/stratified state partition: {minima['router_mixed_partition_rows']:,} / {minima['router_stratified_partition_rows']:,}행
- 최소 meta OOF support: 층당 {minima['meta_oof_rows_per_layer']:,}행
- 고정 outer 평가 모집단: 비율당 {semantic['outer_evaluation_rows_per_fraction']:,}행

## v3에서 고친 범위

비율의 분모와 inner split은 supervised common ledger만 사용한다. deep 모델은 라벨 시각만 이어 붙이지 않고 train cutoff와 validation 구간의 연속 public-covariate 시각을 유지한다. 다만 target mask는 등록된 train/validation ledger와 교차해 그 밖의 target 값은 NaN으로 만든다. router와 deep validation ordered keys, deep chunk, router state partition, OOF/meta 층별 support를 모두 실제 backend 경로에서 확인했다.

outer window, 5개 fraction, 3개 seed, 5개 model component, 모든 hyperparameter, meta, postprocess, family view, metric은 v2와 동일하다. v1/v2 job·cell·partial은 v3에서 재사용하지 않는다.

## 별도 승인 후에만 사용할 명령

```powershell
{command}
```

이 명령은 v3 preexecution seal과 별도 authorization이 모두 맞고, 동일 semantic receipt가 실제 directory/lock/partial 생성 전에 재현될 때만 학습을 시작한다. 이번 준비에서는 authorization을 만들지 않았다.

공식 P2 test/sample/submission 및 submission candidate는 읽지 않았고, submission 생성·업로드와 P3 변경도 0회다.
"""


def _manifest(output_dir: Path, config_path: Path, static: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "manifest.json":
            outputs[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {
        "schema_version": "p2_authoritative_nested_surrogate_execution_ready_manifest.v3",
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
    observations, data_receipt = _load_observations(data_dir, config)
    plans, semantic = semantic_preflight_actual_data(
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
    model_shape = base_runner._model_shape_receipt(config)
    atomic_publish = base_runner._atomic_publish_fixture()
    resource = base_runner._resource_receipt(config)
    command = str(config["exact_command"])
    command_receipt = _command_namespace(command)
    module_sha = sha256_file(_resolve_repo_path(MODULE_RELATIVE))
    runner_sha = sha256_file(Path(__file__).resolve())
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), "v3 readiness output already exists")
    output_dir.mkdir(parents=True, exist_ok=False)
    semantic_sha = str(semantic["semantic_receipt_sha256"])
    preexecution = {
        "schema_version": "p2_authoritative_nested_surrogate_preexecution_seal.v3",
        "status": "EXECUTION_READY_NOT_AUTHORIZED",
        "config_sha256": static["config_sha256"],
        "module_sha256": module_sha,
        "runner_sha256": runner_sha,
        "exact_command_sha256": hashlib.sha256(command.encode()).hexdigest(),
        "semantic_preflight_sha256": semantic_sha,
        "observations_sha256": config["data_contract"]["sha256"],
        "supervised_common_ledger_sha256": semantic["supervised_common_ledger"][
            "ordered_time_sha256"
        ],
        "parent_contract_sha256": config["parent_contract"]["sha256"],
        "component_hyperparameters_sha256": static["component_hyperparameters_sha256"],
        "meta_refit_sha256": static["meta_refit_sha256"],
        "postprocess_sha256": static["postprocess_sha256"],
        "family_views_sha256": static["family_views_sha256"],
        "metrics_sha256": static["metrics_sha256"],
        "semantic_preflight_before_any_actual_side_effect": True,
        "v1_or_v2_resume_allowed": False,
        "top_level_component_jobs_if_authorized": 900,
        "underlying_base_estimator_fits_if_authorized": 1440,
        "actual_model_fits": 0,
        "actual_scores": 0,
        "actual_predictions": 0,
        "authorization_receipt_created": False,
    }
    execution_plan = {
        "schema_version": "p2_authoritative_nested_surrogate_45cell_plan.v3",
        "status": "PASS_DATA_EXECUTABLE_PLAN_FIT_NOT_AUTHORIZED",
        **plan_receipt,
        "prefix_plans": [plan.summary() for plan in plans],
        "complete_pipeline_seeds": seeds,
        "component_order": list(COMPONENTS),
        "split_population": "supervised_common_ledger_v3",
        "semantic_preflight_sha256": semantic_sha,
        "actual_fit_authorized": False,
    }
    qa = {
        "schema_version": "p2_authoritative_nested_surrogate_execution_ready_qa.v3",
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
        "tiny_full_cell_and_resume": tiny["status"],
        "deep_cpu_forward": model_shape["status"],
        "evaluated_oof_atomic_publish": atomic_publish["status"],
        "exact_command_namespace": command_receipt["status"],
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
    module_sha = sha256_file(_resolve_repo_path(MODULE_RELATIVE))
    runner_sha = sha256_file(Path(__file__).resolve())
    seal = verify_preexecution_seal(
        seal_path,
        config_sha256=static["config_sha256"],
        module_sha256=module_sha,
        runner_sha256=runner_sha,
        exact_command=command,
    )
    verify_authorization(
        authorization_path.resolve(strict=True),
        preexecution_seal_sha256=sha256_file(seal_path),
        exact_command=command,
    )
    observations, _ = _load_observations(data_dir, config)
    # Mandatory transaction phase 1: exact actual-data adapters, no actual dir.
    plans, semantic = semantic_preflight_actual_data(
        observations, recipe=recipe, config=config
    )
    _verify_semantic_gates(config, semantic)
    _require(
        semantic["semantic_receipt_sha256"] == seal["semantic_preflight_sha256"],
        "runtime semantic preflight differs from sealed readiness",
    )
    actual_dir = (PROJECT_ROOT / config["output"]["actual_directory"]).resolve()
    _require(not actual_dir.exists(), "v3 actual namespace already exists before one-shot start")
    # Transaction phase 2 begins only here, after the full semantic preflight.
    actual_dir.mkdir(parents=True, exist_ok=False)
    with process_lock(actual_dir / "execution.lock"):
        result = execute_authorized_curve_v3(
            observations=observations,
            plans=plans,
            parent_recipe=recipe,
            config=config,
            output_dir=actual_dir,
            contract_sha256=sha256_file(seal_path),
        )
        _write_json_exclusive(actual_dir / "result.json", result)
    return {
        "status": result["status"],
        "actual_output_dir": str(actual_dir),
        "result_sha256": sha256_file(actual_dir / "result.json"),
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
    _require(not (args.seal_readiness and args.execute), "choose one v3 execution mode")
    data_dir = args.data_dir
    if (args.execute or args.seal_readiness) and data_dir is None:
        raw = os.environ.get("P2_DATA_DIR")
        _require(bool(raw), "set P2_DATA_DIR or pass --data-dir")
        data_dir = Path(str(raw))
    if args.execute:
        assert data_dir is not None
        _require(args.preexecution_seal is not None, "v3 preexecution seal is required")
        _require(args.authorization is not None, "v3 authorization receipt is required")
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
            "status": "PASS_V3_STATIC_ONLY_NO_FIT",
            "config_sha256": static["config_sha256"],
            "exact_command_sha256": hashlib.sha256(str(config["exact_command"]).encode()).hexdigest(),
            "actual_model_fits": 0,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
