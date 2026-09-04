#!/usr/bin/env python
"""Validate the sealed P2 nested-surrogate implementation without fitting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p2_restore.authoritative_nested_surrogate_conformance import (  # noqa: E402
    COMPONENTS,
    DEEP_COMPONENTS,
    KEY_COLUMNS,
    TARGET_LAYERS,
    adapt_panel_for_full_prefix,
    adapt_panel_for_inner_fold,
    build_all_prefix_plans,
    build_epoch_refit_receipt,
    build_prefix_plan,
    build_seeded_execution_plan,
    child_seed,
    fit_prefix_local_meta,
    joint_mask_target_observations,
    merge_component_oof,
    source_api_conformance,
)
from p2_restore.deep_data import P2Panel  # noqa: E402
from p2_restore.regime_gate import STATE_FEATURES  # noqa: E402

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/p2_authoritative_nested_surrogate_conformance_20260825_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts/p2_authoritative_nested_surrogate_conformance_20260825_v1"
)
EXPECTED_CONFIG_SHA256 = "7f84b707bf7059e947a9145f7df4fbbab762739db1b1e7d2d95feaede14a28b9"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"pinned path escaped repository: {relative}") from error
    return path


def _verify_static(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve(strict=True)
    config_sha = _sha256(config_path)
    if EXPECTED_CONFIG_SHA256 != "TO_BE_SEALED" and config_sha != EXPECTED_CONFIG_SHA256:
        raise ValueError("conformance preregistration config hash changed")
    config = _read_json(config_path)
    if config["status"] != "SEALED_BEFORE_CONFORMANCE_RESULTS_NO_45_CELL_FIT":
        raise ValueError("conformance preregistration is not sealed")
    if config["actual_45_cell_fit_authorized"]:
        raise ValueError("conformance config unexpectedly authorizes training")
    parent = config["parent_contract"]
    pins = {
        parent["config_path"]: parent["config_sha256"],
        parent["artifact_contract_seal_path"]: parent[
            "artifact_contract_seal_sha256"
        ],
        parent["decisive_recon_preregistration_path"]: parent[
            "decisive_recon_preregistration_sha256"
        ],
        **config["source_pins"],
        config["resource_estimation_reference"]["historical_result_path"]: config[
            "resource_estimation_reference"
        ]["historical_result_sha256"],
    }
    verified: dict[str, Any] = {}
    for relative, expected in pins.items():
        path = _resolve(PROJECT_ROOT, relative)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"pinned input changed: {relative}")
        verified[relative] = {"sha256": actual, "bytes": path.stat().st_size}
    parent_recipe = _read_json(_resolve(PROJECT_ROOT, parent["config_path"]))
    if parent_recipe["training_authorized"]:
        raise ValueError("sealed parent unexpectedly authorizes training")
    if (
        parent_recipe["authoritative_nested_surrogate_recipe"]["identity"]
        != "P2_AUTHORITATIVE_NESTED_CAUSAL_SURROGATE_V1"
    ):
        raise ValueError("parent recipe identity changed")
    return config, {
        "status": "PASS",
        "config_sha256": config_sha,
        "verified_input_count": len(verified),
        "verified_inputs": verified,
        "parent_contract_unchanged": True,
        "parent_training_authorized": False,
    }


def _load_train_metadata(data_dir: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = data_dir.expanduser().resolve(strict=True)
    observations = (directory / config["train_metadata_contract"]["filename"]).resolve(
        strict=True
    )
    try:
        observations.relative_to(directory)
    except ValueError as error:
        raise ValueError("observations.csv escaped the supplied P2 data directory") from error
    contract = config["train_metadata_contract"]
    if observations.stat().st_size != int(contract["bytes"]):
        raise ValueError("observations.csv byte size changed")
    if _sha256(observations) != contract["sha256"]:
        raise ValueError("observations.csv SHA-256 changed")
    columns = list(contract["columns_read"])
    metadata = pd.read_csv(
        observations,
        usecols=columns,
        dtype={"station": "string", "layer": "int16", "time": "string"},
    )
    if list(metadata.columns) != columns:
        raise ValueError("metadata column order changed")
    if metadata.isna().any().any() or metadata.duplicated(columns).any():
        raise ValueError("train metadata keys are invalid")
    parsed = pd.to_datetime(metadata["time"], utc=True, errors="raise")
    key_hash = pd.util.hash_pandas_object(metadata, index=False).to_numpy(dtype="<u8")
    metadata_digest = hashlib.sha256(np.ascontiguousarray(key_hash).tobytes()).hexdigest()
    return metadata, {
        "access_scope": "TRAIN_STATION_LAYER_TIME_METADATA_ONLY",
        "files_opened": ["observations.csv"],
        "columns_read": columns,
        "value_columns_read": [],
        "rows": len(metadata),
        "duplicate_keys": 0,
        "station_count": int(metadata["station"].nunique()),
        "layer_count": int(metadata["layer"].nunique()),
        "unique_time_count": int(parsed.nunique()),
        "first_time_kst": parsed.min().tz_convert("Asia/Seoul").isoformat(),
        "last_time_kst": parsed.max().tz_convert("Asia/Seoul").isoformat(),
        "metadata_key_hash_sha256": metadata_digest,
        "observations_sha256": contract["sha256"],
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
    }


def _synthetic_metadata() -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=120, freq="12h", tz="Asia/Seoul")
    return pd.DataFrame(
        [("SYNTHETIC", layer, time.isoformat()) for time in times for layer in range(1, 9)],
        columns=["station", "layer", "time"],
    )


def _synthetic_panel(times: pd.DatetimeIndex) -> P2Panel:
    rows = len(times)
    inputs = np.column_stack(
        [
            np.sin(np.arange(rows) / 9.0),
            np.cos(np.arange(rows) / 13.0),
            np.ones(rows),
        ]
    ).astype(np.float32)
    baseline = np.column_stack(
        [15.0 + np.arange(rows) * 0.001 + offset for offset in (0.0, 0.2, 0.4)]
    )
    target = baseline + np.column_stack(
        [np.sin(np.arange(rows) / divisor) * 0.1 for divisor in (7.0, 9.0, 11.0)]
    )
    delta = times.to_series().diff().dt.total_seconds().div(60).to_numpy()
    segment = (
        np.cumsum(np.r_[True, ~np.isclose(delta[1:], 10)]).astype(np.int32) - 1
    )
    return P2Panel(
        times=times.tz_convert("UTC"),
        inputs=inputs,
        input_names=("public_x", "public_y", "public_mask"),
        baseline=baseline,
        target=target,
        target_mask=np.ones((rows, 3), dtype=bool),
        segment_ids=segment,
    )


def _synthetic_oof(plan: Any) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for inner in plan.inner_folds:
        for time_number, time in enumerate(inner.validation_times):
            for layer in TARGET_LAYERS:
                truth = 12.0 + 0.15 * layer + 0.002 * time_number + np.sin(time_number / 7) * 0.05
                rows.append(
                    {
                        "inner_fold": inner.inner_fold,
                        "station": "SYNTHETIC",
                        "layer": layer,
                        "time": time.isoformat(),
                        "truth": float(truth),
                    }
                )
    keys = pd.DataFrame(rows)
    result: dict[str, pd.DataFrame] = {}
    x = np.arange(len(keys), dtype=float)
    for number, component in enumerate(COMPONENTS):
        current = keys.copy()
        current["prediction"] = (
            current["truth"].to_numpy(float)
            + 0.012 * (number - 2)
            + np.sin(x / (6.0 + number)) * (0.025 + 0.004 * number)
        )
        result[component] = current
    return result, keys.loc[:, list(KEY_COLUMNS)]


def _synthetic_features(oof: pd.DataFrame) -> pd.DataFrame:
    result = oof.copy()
    x = np.arange(len(result), dtype=float)
    for number, feature in enumerate(STATE_FEATURES):
        result[feature] = np.sin(x / (5.0 + number)) + np.cos(x / (11.0 + number))
    return result


def _synthetic_receipts() -> dict[str, Any]:
    metadata = _synthetic_metadata()
    plan = build_prefix_plan(
        metadata,
        outer_fold="synthetic_outer",
        validation_start_kst="2024-04-01T00:00:00+09:00",
        validation_stop_kst="2024-05-01T00:00:00+09:00",
        fraction=1.0,
    )
    times = pd.DatetimeIndex(pd.to_datetime(metadata["time"], utc=True).unique()).sort_values()
    observations = pd.DataFrame(
        [
            {
                "station": "SYNTHETIC",
                "layer": layer,
                "time": time.isoformat(),
                "temp": 10.0 + layer,
                "psal": 30.0 + layer / 10.0,
            }
            for time in times
            for layer in range(1, 9)
        ]
    )
    _, joint_mask = joint_mask_target_observations(observations, plan.prefix_times)
    panel = _synthetic_panel(times)
    _, inner_adapter = adapt_panel_for_inner_fold(panel, plan.inner_folds[0])
    _, full_adapter = adapt_panel_for_full_prefix(panel, plan)
    component_frames, expected_keys = _synthetic_oof(plan)
    merged, ledger = merge_component_oof(component_frames, expected_keys=expected_keys)
    prediction_columns = tuple(f"pred_{component}" for component in COMPONENTS)
    meta = fit_prefix_local_meta(
        _synthetic_features(merged),
        scope_id=plan.scope_id,
        prediction_columns=prediction_columns,
        gate_regularization=10.0,
    )
    histories = {
        component: {
            "inner_1": [
                {"epoch": 1, "rmse": 0.8},
                {"epoch": 4, "rmse": 0.5},
                {"epoch": 6, "rmse": 0.5},
            ],
            "inner_2": [{"epoch": 2, "rmse": 0.7}, {"epoch": 8, "rmse": 0.4}],
            "inner_3": [{"epoch": 3, "rmse": 0.6}, {"epoch": 12, "rmse": 0.3}],
        }
        for component in DEEP_COMPONENTS
    }
    epoch = build_epoch_refit_receipt(histories)
    return {
        "status": "PASS",
        "synthetic_values_only": True,
        "plan": plan.summary(),
        "joint_target_mask": joint_mask,
        "deep_inner_adapter": inner_adapter,
        "deep_full_adapter": full_adapter,
        "component_oof_ledger": ledger,
        "epoch_full_refit": epoch,
        "prefix_local_meta_refit": meta,
    }


def _child_seed_receipt(
    seeded_cells: list[dict[str, Any]],
) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for cell in seeded_cells:
        for component in COMPONENTS:
            for phase in ("inner_1", "inner_2", "inner_3", "full"):
                values.append(
                    {
                        "cell_id": cell["cell_id"],
                        "component": component,
                        "phase": phase,
                        "child_seed": child_seed(
                            cell["complete_pipeline_seed"],
                            component,
                            cell["outer_fold"],
                            cell["prefix_fraction"],
                            phase,
                        ),
                    }
                )
    seeds = [item["child_seed"] for item in values]
    if len(values) != 900 or len(set(seeds)) != len(seeds):
        raise ValueError("child seed fan-out is incomplete or collided")
    return {
        "status": "PASS",
        "complete_pipeline_cell_count": len(seeded_cells),
        "component_count": len(COMPONENTS),
        "phase_count": 4,
        "child_seed_count": len(values),
        "unique_child_seed_count": len(set(seeds)),
        "child_seed_ledger_sha256": _canonical_sha256(values),
        "ledger_values_emitted": False,
    }


def _gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=10
        )
        fields = [part.strip() for part in completed.stdout.strip().splitlines()[0].split(",")]
        return {
            "available": True,
            "name": fields[0],
            "memory_total_mib": int(fields[1]),
            "memory_free_mib_at_dry_run": int(fields[2]),
            "utilization_percent_at_dry_run": int(fields[3]),
        }
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return {"available": False}


def _resource_estimate(config: dict[str, Any]) -> dict[str, Any]:
    reference = config["resource_estimation_reference"]
    seconds_per_historical_checkpoint = float(reference["historical_elapsed_seconds"]) / int(
        reference["historical_checkpoint_count"]
    )
    deep_fits = int(config["dry_run_contract"]["expected_deep_fit_count_if_later_authorized"])
    mechanical_hours = seconds_per_historical_checkpoint * deep_fits / 3600.0
    disk = shutil.disk_usage(PROJECT_ROOT)
    return {
        "status": "ESTIMATE_NOT_A_45_CELL_BENCHMARK",
        "fit_count_if_separately_authorized": {
            "total": 900,
            "deep": 720,
            "router": 180,
            "derivation": "45 seeded cells x (3 inner + 1 full) x (4 deep + 1 router)",
        },
        "historical_reference": {
            "elapsed_seconds": float(reference["historical_elapsed_seconds"]),
            "checkpoint_count": int(reference["historical_checkpoint_count"]),
            "checkpoint_bytes": int(reference["historical_checkpoint_bytes"]),
            "seconds_per_checkpoint_crude": seconds_per_historical_checkpoint,
            "mechanical_deep_projection_hours": mechanical_hours,
        },
        "planning_estimate": {
            "single_rtx5090_wall_hours_low": 4.0,
            "single_rtx5090_wall_hours_high": 7.0,
            "gpu_peak_memory_gib_low": 8,
            "gpu_peak_memory_gib_high": 12,
            "cpu_threads": 4,
            "host_ram_gib_low": 16,
            "host_ram_gib_high": 32,
            "peak_storage_gib_low": 12,
            "peak_storage_gib_high": 18,
            "retained_storage_gib_low": 6,
            "retained_storage_gib_high": 10,
            "coexistence_note": "Single-GPU serial deep fits and four CPU threads preserve headroom for the unrelated P3 ERA5 transfer process.",
        },
        "host_snapshot": {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "gpu": _gpu_snapshot(),
            "repository_volume_free_gib": disk.free / (1024**3),
        },
    }


def _korean_report(
    metadata: dict[str, Any],
    resource: dict[str, Any],
    qa: dict[str, Any],
    command: str,
) -> str:
    estimate = resource["planning_estimate"]
    return f"""# P2 authoritative nested surrogate 구현 conformance 보고서

## 결론

세 pending 차원(prefix mask, prefix-local component OOF, epoch/meta-refit)의 구현 conformance는 모두 통과했다. 기술 준비 상태는 `GO_TECHNICAL_CONFORMANCE`이지만, 부모 계약이 실제 45-cell 학습을 허가하지 않으므로 현재 실행 판정은 `BLOCKED_AUTHORIZATION_ONLY`다. 기존 공식 incumbent의 exact 재현이라는 주장은 하지 않는다.

## 확인된 구현 의미

- 15개 outer-fold/prefix와 45개 complete-seed 셀이 원래 봉인된 fold, 7일 embargo, 5개 fraction, 3개 seed에서만 생성됐다.
- 목표층 2/3/4의 TEMP/PSAL은 허용 prefix 밖에서 함께 마스킹된다. deep inner adapter는 미래·embargo 행 자체를 panel에서 제거하므로 기존 complement split이 등록된 train timestamps와 정확히 같다.
- 5개 component OOF는 동일 ordered key/truth digest를 강제한다. stack은 layer별 NNLS 뒤 sum-one 정규화, gate는 같은 현재 scope OOF와 고정 public-state features에서 새로 적합한다.
- epoch는 각 inner fold의 최저 RMSE checkpoint(동률이면 가장 이른 epoch)를 선택하고 세 정수의 중앙값으로 full-prefix를 새로 적합하도록 고정했다.
- 공식 test/sample/submission, submission candidate, Public 점수는 읽지 않았고, 새 fit·prediction·upload는 0회다.

## 실제 train metadata dry-run

- observations metadata 행: {metadata['rows']:,}
- 고유 시각: {metadata['unique_time_count']:,}
- 읽은 열: `{', '.join(metadata['columns_read'])}`; 값 열은 0개
- 생성 계획: 15 outer-prefix cells, 45 seeded cells, child seed 900개(충돌 0)

## 향후 별도 승인 시 자원 추정

- 총 fit 900회: deep 720회, router 180회
- 단일 RTX 5090 예상 wall time: {estimate['single_rtx5090_wall_hours_low']:.0f}~{estimate['single_rtx5090_wall_hours_high']:.0f}시간
- GPU peak: {estimate['gpu_peak_memory_gib_low']}~{estimate['gpu_peak_memory_gib_high']} GiB, CPU 4 threads, RAM {estimate['host_ram_gib_low']}~{estimate['host_ram_gib_high']} GiB
- peak storage: {estimate['peak_storage_gib_low']}~{estimate['peak_storage_gib_high']} GiB, retained {estimate['retained_storage_gib_low']}~{estimate['retained_storage_gib_high']} GiB
- 이 시간은 실제 45-cell benchmark가 아니라 과거 20-checkpoint/272초 실행과 orchestration·prefix 크기·P3 공존 여유를 반영한 planning range다.

## 정확한 현재 단일 명령

```powershell
{command}
```

이 명령은 conformance와 metadata dry-run만 수행하며 실제 45-cell fit을 시작하지 않는다. 실제 학습 명령은 별도 승인과 실행 runner가 봉인되기 전에는 발행하지 않는다.

## QA

- QA status: `{qa['status']}`
- 부모 계약 변경: 0
- actual model fits: 0
- official/test/submission/Public 기반 선택: 0
- P3 process mutations: 0
"""


def run(
    config_path: Path = DEFAULT_CONFIG,
    *,
    execute: bool,
    data_dir: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config, static = _verify_static(config_path)
    api = source_api_conformance()
    if not execute:
        return {
            "status": "PASS_STATIC_CHECK_ONLY",
            "config_sha256": static["config_sha256"],
            "parent_contract_unchanged": True,
            "source_api_conformance": api["status"],
            "new_model_fits": 0,
        }
    if data_dir is None:
        raw = os.environ.get("P2_DATA_DIR")
        if not raw:
            raise FileNotFoundError("set P2_DATA_DIR or pass --data-dir")
        data_dir = Path(raw)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now().astimezone()
    _write_json(
        output_dir / "preexecution_seal.json",
        {
            "schema_version": "p2_authoritative_nested_surrogate_conformance_seal.v1",
            "created_at_kst": started.isoformat(),
            "config_sha256": static["config_sha256"],
            "parent_contract_sha256": config["parent_contract"]["config_sha256"],
            "parent_contract_mutations": 0,
            "actual_45_cell_fit_authorized": False,
            "new_model_fits_before_dry_run": 0,
            "new_scores_before_dry_run": 0,
            "official_public_score_use": "PROHIBITED",
        },
    )
    metadata, metadata_receipt = _load_train_metadata(data_dir, config)
    parent_recipe = _read_json(
        _resolve(PROJECT_ROOT, config["parent_contract"]["config_path"])
    )
    plans = build_all_prefix_plans(metadata, parent_recipe)
    seeds = parent_recipe["authoritative_nested_surrogate_recipe"][
        "complete_pipeline_seed_contract"
    ]["seeds"]
    seeded_cells = build_seeded_execution_plan(plans, seeds)
    seed_receipt = _child_seed_receipt(seeded_cells)
    synthetic = _synthetic_receipts()
    resource = _resource_estimate(config)
    plan_summary = {
        "schema_version": "p2_authoritative_nested_surrogate_execution_plan.v1",
        "status": "PASS_DRY_RUN_FIT_NOT_AUTHORIZED",
        "parent_recipe_identity": "P2_AUTHORITATIVE_NESTED_CAUSAL_SURROGATE_V1",
        "exact_official_incumbent_claimed": False,
        "outer_prefix_cell_count": len(plans),
        "seeded_cell_count": len(seeded_cells),
        "prefix_plans": [plan.summary() for plan in plans],
        "seeded_cells": seeded_cells,
        "actual_fit_authorized": False,
    }
    conformance = {
        "schema_version": "p2_authoritative_nested_surrogate_conformance_result.v1",
        "status": "GO_TECHNICAL_CONFORMANCE_BLOCKED_AUTHORIZATION_ONLY",
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "claim_boundary": "Authoritative future train-only surrogate comparator; never exact official incumbent reproduction.",
        "dimensions": {
            "prefix mask semantics": "PASS_IMPLEMENTED_SYNTHETIC_AND_METADATA_PLAN",
            "component OOF availability": "PASS_IMPLEMENTED_SYNTHETIC_LEDGER",
            "epoch and meta-refit semantics": "PASS_IMPLEMENTED_SYNTHETIC_REFIT_RECEIPTS",
        },
        "source_api_conformance": api,
        "train_metadata_access": metadata_receipt,
        "execution_plan": {
            "outer_prefix_cells": len(plans),
            "seeded_cells": len(seeded_cells),
            "child_seeds": seed_receipt["child_seed_count"],
            "child_seed_collisions": 0,
        },
        "technical_go": True,
        "actual_45_cell_execution": "BLOCKED_AUTHORIZATION_ONLY",
        "actual_45_cell_runner_emitted": False,
        "new_model_fits": 0,
        "new_scores": 0,
    }
    qa = {
        "schema_version": "p2_authoritative_nested_surrogate_conformance_qa.v1",
        "status": "PASS_TECHNICAL_CONFORMANCE_FIT_BLOCKED",
        "parent_contract_hash_pass": True,
        "parent_contract_mutations": 0,
        "pending_dimension_count": 3,
        "pending_dimensions_passed": 3,
        "outer_prefix_cell_count": len(plans),
        "seeded_pipeline_cell_count": len(seeded_cells),
        "child_seed_count": seed_receipt["child_seed_count"],
        "child_seed_collisions": 0,
        "synthetic_receipt_status": synthetic["status"],
        "metadata_only_columns": metadata_receipt["columns_read"],
        "metadata_value_columns_read": 0,
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
        "submission_files_generated": 0,
        "official_public_score_used_for_selection": False,
        "official_public_score_used_for_tuning": False,
        "new_model_fits": 0,
        "new_predictions": 0,
        "uploads": 0,
        "p3_era5_process_mutations": 0,
        "actual_45_cell_execution_authorized": False,
    }
    command = config["portable_single_command"]
    outputs = {
        "static_verification.json": static,
        "train_metadata_receipt.json": metadata_receipt,
        "execution_plan.json": plan_summary,
        "child_seed_receipt.json": seed_receipt,
        "synthetic_conformance_receipts.json": synthetic,
        "resource_estimate.json": resource,
        "conformance_result.json": conformance,
        "qa.json": qa,
    }
    for filename, value in outputs.items():
        _write_json(output_dir / filename, value)
    (output_dir / "REPORT_KO.md").write_text(
        _korean_report(metadata_receipt, resource, qa, command),
        encoding="utf-8",
        newline="\n",
    )
    manifest_outputs: dict[str, Any] = {}
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "manifest.json":
            manifest_outputs[path.name] = {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
    manifest = {
        "schema_version": "p2_authoritative_nested_surrogate_conformance_manifest.v1",
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "config": {
            "path": config_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "sha256": static["config_sha256"],
        },
        "parent_contract": config["parent_contract"],
        "conformance_source": {
            "path": "src/p2_restore/authoritative_nested_surrogate_conformance.py",
            "sha256": _sha256(
                PROJECT_ROOT
                / "src/p2_restore/authoritative_nested_surrogate_conformance.py"
            ),
        },
        "runner": {
            "path": "scripts/validate_p2_authoritative_nested_surrogate_conformance_v1.py",
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "outputs": manifest_outputs,
        "actual_45_cell_fits": 0,
        "official_test_sample_submission_reads": 0,
        "public_score_selection": False,
        "p3_process_mutations": 0,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {
        "status": conformance["status"],
        "output_dir": str(output_dir),
        "manifest_sha256": _sha256(output_dir / "manifest.json"),
        "qa_sha256": _sha256(output_dir / "qa.json"),
        "report_sha256": _sha256(output_dir / "REPORT_KO.md"),
        "outer_prefix_cells": len(plans),
        "seeded_cells": len(seeded_cells),
        "new_model_fits": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run(
        args.config,
        execute=args.execute,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
