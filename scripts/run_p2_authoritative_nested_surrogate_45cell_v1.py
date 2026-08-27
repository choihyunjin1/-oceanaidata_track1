#!/usr/bin/env python
"""Seal or execute the P2 authoritative nested-surrogate 45-cell DAG.

Readiness mode reads only station/layer/time from observations.csv and performs
no model fit.  Actual mode is fail-closed behind a separately supplied approval
receipt bound to the command, config, source, and preexecution seal hashes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p2_restore.authoritative_nested_surrogate_conformance import (  # noqa: E402
    COMPONENTS,
    build_all_prefix_plans,
    child_seed,
)
from p2_restore.authoritative_nested_surrogate_execution import (  # noqa: E402
    atomic_write_or_verify,
    canonical_sha256,
    execute_authorized_curve,
    process_lock,
    sha256_file,
    temporary_tiny_fixture,
    verify_authorization,
    verify_preexecution_seal,
)
from p2_restore.deep_models import build_model, count_parameters  # noqa: E402

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/p2_authoritative_nested_surrogate_execution_20260825_v2.json"
)
DEFAULT_READY_OUTPUT = (
    PROJECT_ROOT / "artifacts/p2_authoritative_nested_surrogate_execution_ready_20260825_v2"
)
EXPECTED_CONFIG_SHA256 = "b05fe56730ef8116b0aa6b914823dedfbb878595190d5d7a9366987eb07685b4"
MODULE_RELATIVE = "src/p2_restore/authoritative_nested_surrogate_execution.py"
RUNNER_RELATIVE = "scripts/run_p2_authoritative_nested_surrogate_45cell_v1.py"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _monotonic_timestamp(previous: datetime | None) -> tuple[datetime, str]:
    current = datetime.now().astimezone()
    if previous is not None and current <= previous:
        current = previous + timedelta(microseconds=1)
    return current, current.isoformat()


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
    try:
        path.relative_to(PROJECT_ROOT.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"pinned path escaped repository: {relative}") from error
    return path


def _load_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    overlay = _read_json(config_path)
    base_pin = overlay.get("base_config")
    if base_pin is None:
        return overlay, None
    base_path = _resolve_repo_path(str(base_pin["path"]))
    _require(sha256_file(base_path) == base_pin["sha256"], "base execution config changed")
    return _deep_merge(_read_json(base_path), overlay), dict(base_pin)


def _verify_static(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve(strict=True)
    config_sha = sha256_file(config_path)
    if EXPECTED_CONFIG_SHA256 not in {"TO_BE_SEALED", "TO_BE_SEALED_V2"}:
        _require(config_sha == EXPECTED_CONFIG_SHA256, "execution config changed")
    config, base_pin = _load_config(config_path)
    _require(
        config["status"] == "SEALED_EXECUTION_READY_DRY_RUN_NO_ACTUAL_FIT",
        "execution config is not sealed",
    )
    forbidden_permissions = config["permissions"]
    _require(
        all(value is False for value in forbidden_permissions.values()),
        "readiness config grants a forbidden permission",
    )
    parent = config["parent_contract"]
    conformance = config["completed_conformance"]
    pins = {
        parent["path"]: parent["sha256"],
        parent["contract_seal_path"]: parent["contract_seal_sha256"],
        parent["decisive_recon_preregistration_path"]: parent[
            "decisive_recon_preregistration_sha256"
        ],
        parent["common_protocol_path"]: parent["common_protocol_sha256"],
        conformance["config_path"]: conformance["config_sha256"],
        conformance["manifest_path"]: conformance["manifest_sha256"],
        conformance["qa_path"]: conformance["qa_sha256"],
        **config["source_pins"],
        **config["hyperparameter_evidence_pins"],
    }
    verified: dict[str, Any] = {}
    if base_pin is not None:
        base_path = _resolve_repo_path(str(base_pin["path"]))
        verified[str(base_pin["path"])] = {
            "sha256": str(base_pin["sha256"]),
            "bytes": base_path.stat().st_size,
        }
    for relative, expected in pins.items():
        path = _resolve_repo_path(relative)
        actual = sha256_file(path)
        _require(actual == expected, f"pinned source/evidence changed: {relative}")
        verified[relative] = {"sha256": actual, "bytes": path.stat().st_size}
    parent_recipe = _read_json(_resolve_repo_path(parent["path"]))
    _require(parent_recipe["training_authorized"] is False, "parent authorizes fit")
    identity = parent_recipe["authoritative_nested_surrogate_recipe"]["identity"]
    _require(identity == "P2_AUTHORITATIVE_NESTED_CAUSAL_SURROGATE_V1", "identity changed")
    graph = config["execution_graph"]
    _require(
        (
            graph["top_level_component_jobs"],
            graph["underlying_deep_fits"],
            graph["underlying_lightgbm_fits"],
            graph["underlying_base_estimator_fits"],
            graph["meta_optimizations"],
        )
        == (900, 720, 720, 1440, 405),
        "execution DAG counts changed",
    )
    return config, {
        "status": "PASS_STATIC_SOURCE_CONFIG_AND_PARENT_PINS",
        "config_sha256": config_sha,
        "verified_input_count": len(verified),
        "verified_inputs": verified,
        "parent_recipe_identity": identity,
        "parent_training_authorized": False,
        "component_hyperparameters_sha256": canonical_sha256(config["component_hyperparameters"]),
        "postprocess_sha256": canonical_sha256(config["postprocess"]),
        "family_views_sha256": canonical_sha256(config["family_views"]),
        "metrics_sha256": canonical_sha256(config["metrics"]),
        "base_config": base_pin,
        "supersedes": config.get("supersedes"),
    }


def _observation_path(data_dir: Path, config: dict[str, Any]) -> Path:
    directory = data_dir.expanduser().resolve(strict=True)
    path = (directory / config["data_contract"]["allowed_filename"]).resolve(strict=True)
    try:
        path.relative_to(directory)
    except ValueError as error:
        raise ValueError("observations.csv escaped P2_DATA_DIR") from error
    contract = config["data_contract"]
    _require(path.stat().st_size == int(contract["bytes"]), "observation size changed")
    _require(sha256_file(path) == contract["sha256"], "observation hash changed")
    return path


def _load_metadata(data_dir: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _observation_path(data_dir, config)
    columns = list(config["data_contract"]["dry_run_columns"])
    frame = pd.read_csv(
        path,
        usecols=columns,
        dtype={"station": "string", "layer": "int16", "time": "string"},
    )
    _require(list(frame.columns) == columns, "metadata column order changed")
    _require(not frame.isna().any().any(), "metadata contains missing keys")
    _require(not frame.duplicated(columns).any(), "metadata contains duplicate keys")
    parsed = pd.to_datetime(frame["time"], utc=True, errors="raise")
    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="<u8")
    return frame, {
        "status": "PASS_TRAIN_METADATA_ONLY",
        "access_scope": "OBSERVATIONS_STATION_LAYER_TIME_ONLY",
        "files_opened": ["observations.csv"],
        "columns_read": columns,
        "value_columns_read": [],
        "rows": len(frame),
        "unique_time_count": int(parsed.nunique()),
        "station_count": int(frame["station"].nunique()),
        "layer_count": int(frame["layer"].nunique()),
        "first_time_kst": parsed.min().tz_convert("Asia/Seoul").isoformat(),
        "last_time_kst": parsed.max().tz_convert("Asia/Seoul").isoformat(),
        "metadata_key_hash_sha256": hashlib.sha256(
            np.ascontiguousarray(hashed).tobytes()
        ).hexdigest(),
        "observations_sha256": config["data_contract"]["sha256"],
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
    }


def _seed_plan_receipt(plans: list[Any], seeds: list[int]) -> dict[str, Any]:
    ledger: list[dict[str, Any]] = []
    for plan in plans:
        for complete_seed in seeds:
            for component in COMPONENTS:
                for phase in ("inner_1", "inner_2", "inner_3", "full"):
                    ledger.append(
                        {
                            "scope_id": plan.scope_id,
                            "complete_seed": int(complete_seed),
                            "component": component,
                            "phase": phase,
                            "child_seed": child_seed(
                                complete_seed,
                                component,
                                plan.outer_fold,
                                plan.fraction,
                                phase,
                            ),
                        }
                    )
    values = [int(row["child_seed"]) for row in ledger]
    _require(len(plans) == 15 and len(seeds) == 3, "15/45 plan changed")
    _require(len(ledger) == 900, "top-level component job count changed")
    _require(len(set(values)) == 900, "child seed collision")
    return {
        "status": "PASS_15_CELL_45_SEEDED_CELL_900_TOP_LEVEL_JOB_PLAN",
        "outer_prefix_cells": len(plans),
        "seeded_cells": len(plans) * len(seeds),
        "top_level_component_jobs": len(ledger),
        "unique_child_seeds": len(set(values)),
        "child_seed_ledger_sha256": canonical_sha256(ledger),
        "ledger_values_emitted": False,
    }


def _model_shape_receipt(config: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(20260825)
    entries: dict[str, Any] = {}
    with torch.inference_mode():
        for name, expected in config["component_hyperparameters"]["deep"].items():
            if not isinstance(expected, dict):
                continue
            model = build_model(name, int(expected["input_channels"])).cpu().eval()
            actual_parameters = count_parameters(model)
            _require(
                actual_parameters == int(expected["parameter_count"]),
                f"{name} parameter count changed",
            )
            output = model(torch.zeros(1, 12, int(expected["input_channels"])))
            _require(tuple(output.shape) == (1, 12, 3), f"{name} output shape changed")
            entries[name] = {
                "input_shape": [1, 12, int(expected["input_channels"])],
                "output_shape": list(output.shape),
                "parameter_count": actual_parameters,
                "fit_calls": 0,
            }
            del model, output
    _require(len(entries) == 4, "deep component set changed")
    return {
        "status": "PASS_FOUR_CPU_TINY_FORWARDS_NO_FIT",
        "components": entries,
        "actual_model_fits": 0,
    }


def _atomic_publish_fixture() -> dict[str, Any]:
    payload = b"synthetic-evaluated-oof-payload"
    with tempfile.TemporaryDirectory(prefix="p2_atomic_publish_") as directory:
        root = Path(directory)
        target = root / "evaluated_oof_100.parquet"
        stale = root / ".evaluated_oof_100.parquet.partial.simulated-crash"
        with stale.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        committed = atomic_write_or_verify(target, payload)
        _require(stale.is_file(), "simulated crash partial was not preserved")
        _require(target.read_bytes() == payload, "atomic fixture final differs")
        reused = atomic_write_or_verify(target, payload)
        partials = sorted(root.glob(".evaluated_oof_100.parquet.partial.*"))
        _require(partials == [stale], "successful publication left a new partial")
    return {
        "status": "PASS_STALE_PARTIAL_PRESERVED_ATOMIC_COMMIT_AND_VERIFIED_RESUME",
        "first_publication": committed,
        "second_publication": reused,
        "stale_failed_partial_preserved": True,
        "new_successful_partial_consumed_by_rename": True,
        "actual_model_fits": 0,
        "actual_scores": 0,
        "actual_predictions": 0,
    }


def _superseded_v1_fail_closed_receipt(
    config: dict[str, Any],
    static: dict[str, Any],
    *,
    current_module_sha256: str,
    current_runner_sha256: str,
) -> dict[str, Any]:
    base_pin = static["base_config"]
    _require(base_pin is not None, "v2 base config pin is missing")
    base = _read_json(_resolve_repo_path(str(base_pin["path"])))
    superseded = config["supersedes"]
    old_seal_path = _resolve_repo_path(
        str(Path(superseded["readiness_directory"]) / "preexecution_seal.json")
    )
    _require(
        sha256_file(old_seal_path) == superseded["preexecution_seal_sha256"],
        "superseded v1 seal bytes changed",
    )
    failure = ""
    try:
        verify_preexecution_seal(
            old_seal_path,
            config_sha256=str(base_pin["sha256"]),
            module_sha256=current_module_sha256,
            runner_sha256=current_runner_sha256,
            exact_command=str(base["exact_command"]),
        )
    except ValueError as error:
        failure = str(error)
    _require(
        failure in {"preexecution module pin changed", "preexecution runner pin changed"},
        "superseded v1 seal did not fail closed on current sources",
    )
    return {
        "status": "PASS_V1_SEAL_FAILS_CLOSED_ON_CURRENT_SOURCE_HASH",
        "base_config_sha256": str(base_pin["sha256"]),
        "superseded_preexecution_seal_sha256": superseded[
            "preexecution_seal_sha256"
        ],
        "failure_reason": failure,
        "current_module_sha256": current_module_sha256,
        "current_runner_sha256": current_runner_sha256,
        "authorization_usable": False,
        "actual_model_fits": 0,
    }


def _v2_command_namespace_receipt(command: str) -> dict[str, Any]:
    required_arguments = {
        "config": (
            '--config "configs\\experiments\\'
            'p2_authoritative_nested_surrogate_execution_20260825_v2.json"'
        ),
        "preexecution_seal": (
            '--preexecution-seal "artifacts\\'
            'p2_authoritative_nested_surrogate_execution_ready_20260825_v2\\'
            'preexecution_seal.json"'
        ),
        "authorization": (
            '--authorization "artifacts\\'
            'p2_authoritative_nested_surrogate_execution_ready_20260825_v2\\'
            'EXECUTION_AUTHORIZATION.json"'
        ),
    }
    for name, fragment in required_arguments.items():
        _require(fragment in command, f"exact command does not pin v2 {name} namespace")
    _require(
        "p2_authoritative_nested_surrogate_execution_ready_20260825_v1\\"
        not in command,
        "exact command still references the superseded v1 readiness namespace",
    )
    return {
        "status": "PASS_EXACT_COMMAND_PINS_V2_CONFIG_SEAL_AND_AUTHORIZATION",
        "required_arguments": required_arguments,
        "superseded_v1_readiness_namespace_present": False,
        "actual_model_fits": 0,
        "actual_scores": 0,
        "actual_predictions": 0,
    }


def _gpu_snapshot() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        fields = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
        return {
            "available": True,
            "name": fields[0],
            "memory_total_mib": int(fields[1]),
            "memory_free_mib_at_dry_run": int(fields[2]),
            "utilization_percent_at_dry_run": int(fields[3]),
        }
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return {"available": False}


def _resource_receipt(config: dict[str, Any]) -> dict[str, Any]:
    estimate = config["resource_estimate"]
    disk = shutil.disk_usage(PROJECT_ROOT)
    return {
        **estimate,
        "execution_graph": config["execution_graph"],
        "host_snapshot": {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "gpu": _gpu_snapshot(),
            "repository_volume_free_gib": disk.free / (1024**3),
        },
        "coexistence_policy": "single GPU, serial component jobs, four CPU threads; P3 unchanged",
        "actual_45_cell_benchmark_performed": False,
    }


def _report_ko(
    *,
    metadata: dict[str, Any],
    plan: dict[str, Any],
    resource: dict[str, Any],
    command: str,
) -> str:
    return f"""# P2 authoritative nested surrogate 45-cell 실행 준비 보고서

## 결론

판정은 `EXECUTION_READY_NOT_AUTHORIZED`이다. 복합 router_400, 네 deep contributor의 정확한 inner checkpoint 선택, 3-inner OOF 기반 fresh stack/gate, full-prefix refit, 동일 후처리·family view·metric, exclusive resume를 하나의 실행기에 고정했다. 다만 실제 학습 승인은 없으므로 45-cell fit은 0회이며 실행은 시작하지 않았다.

이 문서와 v2 seal이 현재 authoritative readiness다. 이전 v1 readiness seal은 final evaluated OOF의 crash-safe publication이 부족해 `SUPERSEDED_ROBUSTNESS_CAVEAT`이며 실행 승인에 사용할 수 없다.

## 이번 봉인에서 바로잡은 기술 차이

- router_400은 단일 LightGBM이 아니라 base·phase·mixed·stratified 4개 400-round LightGBM으로 구성된다. layer 2/3은 phase, layer 4는 state arm을 사용한다.
- 기존 `train_fold`의 1e-6 개선 임계값을 사용하지 않는다. 새 실행기는 모든 등록 checkpoint에서 정확한 `(RMSE, epoch)` 최소값을 선택하여 exact tie에서만 earliest epoch가 이긴다.
- inner fold 세 개의 component OOF에서 현재 scope 전용 layer별 NNLS stack과 10-feature soft gate(reg=10)를 새로 적합하고, 세 best epoch의 중간 정수로 각 deep component를 full-prefix refit한다.
- checkpoint는 full inference 호환 필수 keys와 `epochs`를 포함한다. 완결 job만 manifest/hash/size 확인 후 재사용하며 partial은 보존하되 재사용하지 않는다.
- 최종 `evaluated_oof_*.parquet`도 unique partial에 쓴 뒤 file fsync와 atomic rename으로만 게시한다. 기존 final은 hash/size가 정확할 때만 재사용하며, crash/race partial은 감사용으로 보존하고 final로 간주하지 않는다.

## Metadata dry-run 및 실행 그래프

- observations metadata: {metadata["rows"]:,}행, 고유 시각 {metadata["unique_time_count"]:,}개; 읽은 값 열 0개
- outer-prefix cell {plan["outer_prefix_cells"]}개, seeded cell {plan["seeded_cells"]}개
- top-level component jobs {plan["top_level_component_jobs"]}개
- 실제 승인 후 underlying base-estimator fits: 1,440회(deep 720 + LightGBM 720)
- meta optimizations: 405회
- 이번 실행의 actual model fits/scores/predictions: 0/0/0

## 자원 추정

- single RTX 5090 planning range: {resource["single_rtx5090_wall_hours_low"]:.0f}~{resource["single_rtx5090_wall_hours_high"]:.0f}시간
- GPU peak {resource["gpu_peak_memory_gib_low"]}~{resource["gpu_peak_memory_gib_high"]} GiB, host RAM {resource["host_ram_gib_low"]}~{resource["host_ram_gib_high"]} GiB
- peak/retained storage {resource["peak_storage_gib_low"]}~{resource["peak_storage_gib_high"]} / {resource["retained_storage_gib_low"]}~{resource["retained_storage_gib_high"]} GiB
- 4~8시간은 actual 45-cell benchmark가 아니다. 과거 deep 20-checkpoint 272초를 720 deep jobs로 기계 확장한 2.72시간, 5,000-round router 이력의 400-round 환산 약 0.45시간, meta/prediction/I/O/P3 공존 여유를 합친 범위다.

## 별도 승인 후에만 사용할 단일 명령

```powershell
{command}
```

이 명령도 exact preexecution seal과 별도 `EXECUTION_AUTHORIZATION.json`이 모두 일치해야 진행한다. 승인 파일은 이번 준비 단계에서 생성하지 않았다.

## 경계

공식 P2 test/sample/submission, submission candidate, Public 점수는 읽거나 선택에 사용하지 않았다. submission 생성·업로드와 P3 프로세스 변경도 0회다. 이 surface는 새 authoritative local surrogate이며 exact official incumbent라고 주장하지 않는다.
"""


def _manifest(output_dir: Path, config_path: Path, static: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "manifest.json":
            outputs[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return {
        "schema_version": "p2_authoritative_nested_surrogate_execution_ready_manifest.v2",
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "status": "EXECUTION_READY_NOT_AUTHORIZED",
        "config": {
            "path": config_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "sha256": static["config_sha256"],
        },
        "module": {
            "path": MODULE_RELATIVE,
            "sha256": sha256_file(_resolve_repo_path(MODULE_RELATIVE)),
        },
        "runner": {"path": RUNNER_RELATIVE, "sha256": sha256_file(Path(__file__).resolve())},
        "supersedes": static.get("supersedes"),
        "outputs": outputs,
        "actual_model_fits": 0,
        "actual_scores": 0,
        "actual_predictions": 0,
        "official_test_sample_submission_reads": 0,
        "public_score_selection_or_tuning": False,
        "submission_files_generated": 0,
        "uploads": 0,
        "p3_process_mutations": 0,
    }


def seal_readiness(
    config_path: Path,
    *,
    data_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config, static = _verify_static(config_path)
    metadata, metadata_receipt = _load_metadata(data_dir, config)
    parent_recipe = _read_json(_resolve_repo_path(config["parent_contract"]["path"]))
    plans = build_all_prefix_plans(metadata, parent_recipe)
    seeds = [
        int(value)
        for value in parent_recipe["authoritative_nested_surrogate_recipe"][
            "complete_pipeline_seed_contract"
        ]["seeds"]
    ]
    plan_receipt = _seed_plan_receipt(plans, seeds)
    output_dir = output_dir.resolve()
    tiny = temporary_tiny_fixture(plans[0])
    model_shape = _model_shape_receipt(config)
    atomic_publish = _atomic_publish_fixture()
    resource = _resource_receipt(config)
    module_sha = sha256_file(_resolve_repo_path(MODULE_RELATIVE))
    runner_sha = sha256_file(Path(__file__).resolve())
    command = str(config["exact_command"])
    command_namespace = _v2_command_namespace_receipt(command)
    superseded_v1 = _superseded_v1_fail_closed_receipt(
        config,
        static,
        current_module_sha256=module_sha,
        current_runner_sha256=runner_sha,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    preexecution = {
        "schema_version": "p2_authoritative_nested_surrogate_preexecution_seal.v2",
        "status": "EXECUTION_READY_NOT_AUTHORIZED",
        "config_sha256": static["config_sha256"],
        "module_sha256": module_sha,
        "runner_sha256": runner_sha,
        "exact_command_sha256": hashlib.sha256(command.encode()).hexdigest(),
        "component_hyperparameters_sha256": static["component_hyperparameters_sha256"],
        "parent_contract_sha256": config["parent_contract"]["sha256"],
        "completed_conformance_manifest_sha256": config["completed_conformance"]["manifest_sha256"],
        "top_level_component_jobs_if_authorized": 900,
        "underlying_base_estimator_fits_if_authorized": 1440,
        "meta_optimizations_if_authorized": 405,
        "actual_model_fits": 0,
        "actual_scores": 0,
        "actual_predictions": 0,
        "authorization_receipt_created": False,
        "evaluated_oof_publication": "UNIQUE_PARTIAL_FSYNC_ATOMIC_RENAME_HASH_VERIFIED_RESUME",
        "failed_partial_policy": "PRESERVE_FOR_AUDIT_NEVER_TREAT_AS_FINAL",
        "supersedes": static["supersedes"],
    }
    execution_plan = {
        "schema_version": "p2_authoritative_nested_surrogate_45cell_plan.v2",
        "status": "PASS_PLAN_FIT_NOT_AUTHORIZED",
        **plan_receipt,
        "prefix_plans": [plan.summary() for plan in plans],
        "complete_pipeline_seeds": seeds,
        "component_order": list(COMPONENTS),
        "underlying_deep_fits": 720,
        "underlying_lightgbm_fits": 720,
        "underlying_base_estimator_fits": 1440,
        "meta_optimizations": 405,
        "actual_fit_authorized": False,
    }
    receipt_order = [
        ("static_verification", static),
        ("train_metadata", metadata_receipt),
        ("execution_plan", execution_plan),
        ("tiny_fixture", tiny),
        ("deep_model_shape", model_shape),
        ("atomic_publish", atomic_publish),
        ("exact_command_namespace", command_namespace),
        ("superseded_v1_fail_closed", superseded_v1),
        ("resource_estimate", resource),
        ("preexecution_seal", preexecution),
    ]
    previous: datetime | None = None
    receipt_timestamps: dict[str, str] = {}
    for receipt_id, receipt in receipt_order:
        previous, timestamp = _monotonic_timestamp(previous)
        receipt["receipt_id"] = receipt_id
        receipt["created_at_kst"] = timestamp
        receipt_timestamps[receipt_id] = timestamp
    previous, qa_timestamp = _monotonic_timestamp(previous)
    receipt_timestamps["qa"] = qa_timestamp
    parsed_receipt_timestamps = [
        datetime.fromisoformat(value) for value in receipt_timestamps.values()
    ]
    config_created = datetime.fromisoformat(str(config["created_at_kst"]))
    timestamps_monotonic = all(
        left < right
        for left, right in zip(
            parsed_receipt_timestamps,
            parsed_receipt_timestamps[1:],
            strict=False,
        )
    )
    _require(timestamps_monotonic, "readiness receipt timestamps are not monotonic")
    _require(
        config_created < parsed_receipt_timestamps[0],
        "execution config timestamp is not earlier than receipts",
    )
    qa = {
        "schema_version": "p2_authoritative_nested_surrogate_execution_ready_qa.v2",
        "receipt_id": "qa",
        "created_at_kst": qa_timestamp,
        "status": "PASS_EXECUTION_READY_NOT_AUTHORIZED",
        "static_pins": static["status"],
        "metadata": metadata_receipt["status"],
        "tiny_full_cell_and_resume": tiny["status"],
        "deep_cpu_forward": model_shape["status"],
        "evaluated_oof_atomic_publish": atomic_publish["status"],
        "exact_command_namespace": command_namespace["status"],
        "superseded_v1_fail_closed": superseded_v1["status"],
        "outer_prefix_cells": 15,
        "seeded_cells": 45,
        "top_level_component_jobs": 900,
        "underlying_base_estimator_fits_if_authorized": 1440,
        "meta_optimizations_if_authorized": 405,
        "actual_model_fits": 0,
        "actual_scores": 0,
        "actual_predictions": 0,
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
        "official_public_score_used_for_selection_or_tuning": False,
        "submission_files_generated": 0,
        "uploads": 0,
        "p3_process_mutations": 0,
        "exact_official_incumbent_claimed": False,
        "authorization_receipt_created": False,
        "previous_v1_readiness_superseded": True,
        "config_created_before_receipts": True,
        "receipt_timestamps_monotonic_strict": True,
        "receipt_timestamp_order": list(receipt_timestamps),
        "receipt_timestamps": receipt_timestamps,
    }
    _write_json_exclusive(output_dir / "static_verification.json", static)
    _write_json_exclusive(output_dir / "train_metadata_receipt.json", metadata_receipt)
    _write_json_exclusive(output_dir / "execution_plan.json", execution_plan)
    _write_json_exclusive(output_dir / "tiny_fixture_receipt.json", tiny)
    _write_json_exclusive(output_dir / "deep_model_shape_receipt.json", model_shape)
    _write_json_exclusive(output_dir / "atomic_publish_receipt.json", atomic_publish)
    _write_json_exclusive(output_dir / "exact_command_namespace_receipt.json", command_namespace)
    _write_json_exclusive(output_dir / "superseded_v1_fail_closed_receipt.json", superseded_v1)
    _write_json_exclusive(output_dir / "resource_estimate.json", resource)
    _write_json_exclusive(output_dir / "preexecution_seal.json", preexecution)
    _write_json_exclusive(output_dir / "qa.json", qa)
    _write_text_exclusive(
        output_dir / "REPORT_KO.md",
        _report_ko(
            metadata=metadata_receipt,
            plan=plan_receipt,
            resource=resource,
            command=command,
        ),
    )
    manifest = _manifest(output_dir, config_path, static)
    _write_json_exclusive(output_dir / "manifest.json", manifest)
    return {
        "status": "EXECUTION_READY_NOT_AUTHORIZED",
        "output_dir": str(output_dir),
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "preexecution_seal_sha256": sha256_file(output_dir / "preexecution_seal.json"),
        "qa_sha256": sha256_file(output_dir / "qa.json"),
        "actual_model_fits": 0,
    }


def _verify_runtime_policy(config: dict[str, Any]) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
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
    config, static = _verify_static(config_path)
    _verify_runtime_policy(config)
    command = str(config["exact_command"])
    module_sha = sha256_file(_resolve_repo_path(MODULE_RELATIVE))
    runner_sha = sha256_file(Path(__file__).resolve())
    verify_preexecution_seal(
        preexecution_seal_path.resolve(strict=True),
        config_sha256=static["config_sha256"],
        module_sha256=module_sha,
        runner_sha256=runner_sha,
        exact_command=command,
    )
    verify_authorization(
        authorization_path.resolve(strict=True),
        preexecution_seal_sha256=sha256_file(preexecution_seal_path),
        exact_command=command,
    )
    observation_path = _observation_path(data_dir, config)
    actual_dir = (PROJECT_ROOT / config["output"]["actual_directory"]).resolve()
    actual_dir.mkdir(parents=True, exist_ok=True)
    with process_lock(actual_dir / "execution.lock"):
        observations = pd.read_csv(observation_path)
        parent_recipe = _read_json(_resolve_repo_path(config["parent_contract"]["path"]))
        result = execute_authorized_curve(
            observations=observations,
            parent_recipe=parent_recipe,
            config=config,
            output_dir=actual_dir,
            contract_sha256=sha256_file(preexecution_seal_path),
        )
        result_path = actual_dir / "result.json"
        if result_path.exists():
            _require(_read_json(result_path) == result, "resumed result changed")
        else:
            _write_json_exclusive(result_path, result)
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
    _require(not (args.seal_readiness and args.execute), "choose one execution mode")
    data_dir = args.data_dir
    if (args.execute or args.seal_readiness) and data_dir is None:
        raw = os.environ.get("P2_DATA_DIR")
        _require(bool(raw), "set P2_DATA_DIR or pass --data-dir")
        data_dir = Path(str(raw))
    if args.execute:
        assert data_dir is not None
        _require(args.preexecution_seal is not None, "preexecution seal is required")
        _require(args.authorization is not None, "authorization receipt is required")
        result = execute_actual(
            args.config,
            data_dir=data_dir,
            preexecution_seal_path=args.preexecution_seal,
            authorization_path=args.authorization,
        )
    elif args.seal_readiness:
        assert data_dir is not None
        result = seal_readiness(
            args.config,
            data_dir=data_dir,
            output_dir=args.output_dir,
        )
    else:
        config, static = _verify_static(args.config)
        result = {
            "status": "PASS_STATIC_ONLY_NO_FIT",
            "config_sha256": static["config_sha256"],
            "component_hyperparameters_sha256": static["component_hyperparameters_sha256"],
            "exact_command_sha256": hashlib.sha256(
                str(config["exact_command"]).encode()
            ).hexdigest(),
            "actual_model_fits": 0,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
