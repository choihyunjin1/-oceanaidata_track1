"""Run the sealed one-shot P3 selection-matched masked-history experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from catboost.utils import get_gpu_device_count

from p3_wave.selection_matched_masked_ssl_20260830_v1 import (
    LEADS,
    STATIONS,
    apply_paired_prequential_reference,
    comparison_metrics,
    evaluate_promotion_gate,
    extract_history_sequences,
    fit_candidate_fold,
    paired_case_bootstrap,
    recipe_summary,
    saved_catboost_component_predictions,
    summarize_validation_histories,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_selection_matched_masked_ssl_20260830_v1"
CONFIG_RELATIVE = f"configs/experiments/{EXPERIMENT_ID}.json"
MODULE_RELATIVE = "src/p3_wave/selection_matched_masked_ssl_20260830_v1.py"
TEST_RELATIVE = f"tests/test_{EXPERIMENT_ID}.py"
OUTPUT_RELATIVE = f"artifacts/{EXPERIMENT_ID}"
LOCK_RELATIVE = f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
FAILURE_RELATIVE = f"artifacts/{EXPERIMENT_ID}.FAILED.json"
STAGE0_RUNNER_RELATIVE = "scripts/run_p3_selection_matched_cohort_preflight_20260830_v1.py"
EXPECTED_CONFIG_SHA256 = "f907f9ce4d6946697a55383c0d2d675e28f60ad18934f968dff949efefee1f66"


class ContractError(ValueError):
    """Raised before the one-shot lock when a sealed contract differs."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load_config(path: Path | None = None) -> dict[str, Any]:
    canonical = (ROOT / CONFIG_RELATIVE).resolve(strict=True)
    requested = (path or canonical).resolve(strict=True)
    _require(requested == canonical, "non-canonical config path is forbidden")
    _require(sha256_file(canonical) == EXPECTED_CONFIG_SHA256, "config SHA changed")
    config = json.loads(canonical.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    _require(
        config["schema_version"] == "p3.selection_matched_masked_ssl.preregistration.v1",
        "schema version changed",
    )
    _require(config["experiment_id"] == EXPERIMENT_ID, "experiment id changed")
    paths = config["canonical_paths"]
    _require(paths["config"] == CONFIG_RELATIVE, "canonical config path changed")
    _require(paths["runner"] == f"scripts/run_{EXPERIMENT_ID}.py", "runner path changed")
    _require(paths["implementation"] == MODULE_RELATIVE, "module path changed")
    _require(paths["focused_tests"] == TEST_RELATIVE, "test path changed")
    _require(paths["output"] == OUTPUT_RELATIVE, "output path changed")
    _require(paths["attempt_lock"] == LOCK_RELATIVE, "lock path changed")
    _require(paths["failure_receipt"] == FAILURE_RELATIVE, "failure path changed")
    boundary = config["data_boundary"]
    _require(
        boundary["allowed_source_basenames"]
        == ["README.md", "train_wave.csv", "train_atmos.csv"],
        "source allowlist changed",
    )
    _require(boundary["load_p3_data_allowed"] is False, "broad P3 loader enabled")
    _require(boundary["official_or_context_value_reads"] == 0, "official reads enabled")
    _require(boundary["csv_output_count"] == 0, "CSV output enabled")
    cohort = config["cohort_contract"]
    _require(cohort["canonical_grid_minutes"] == 10, "grid cadence changed")
    _require(cohort["canonical_dense_anchor_minutes"] == 60, "anchor cadence changed")
    _require(cohort["history_hours"] == 48, "history changed")
    _require(cohort["history_rows_including_anchor"] == 289, "history rows changed")
    _require(tuple(cohort["official_leads_hours"]) == LEADS, "official leads changed")
    _require(cohort["current_hs_min_inclusive_m"] == 1.5, "lower Hs changed")
    _require(cohort["current_hs_max_exclusive_m"] == 2.2, "upper Hs changed")
    _require(cohort["rise_lookback_hours"] == 12, "rise lookback changed")
    _require(cohort["rise_min_exclusive_m"] == 0.2, "rise threshold changed")
    _require(cohort["station_global_validation_gap_hours"] == 78, "gap changed")
    _require(cohort["train_cutoff_hours_before_window_start"] == 78, "cutoff changed")
    windows = config["forward_windows"]
    _require(
        [(item["name"], item["expected_dense_train_count"], item["expected_validation_count"])
         for item in windows]
        == [
            ("2024_h2_storm", 740, 41),
            ("winter_transition", 1135, 65),
            ("2025_h1", 1737, 51),
        ],
        "forward windows or support counts changed",
    )
    representation = config["representation"]
    _require(representation["input_channels"] == 24, "SSL input dimension changed")
    _require(representation["model"]["channels"] == [16, 24, 24], "encoder changed")
    _require(representation["model"]["aggregated_embedding_dimension"] == 72, "embedding changed")
    masked = representation["masked_training"]
    _require(masked["window_steps"] == 289, "SSL window changed")
    _require(masked["stride_steps"] == 289, "SSL stride changed")
    _require(masked["mask_fraction"] == 0.25, "mask fraction changed")
    _require(masked["mask_block_steps"] == 36, "mask block changed")
    _require(masked["maximum_epochs"] == 8, "epoch budget changed")
    _require(masked["fold_seeds"] == [20260830, 20260831, 20260832], "seeds changed")
    _require(
        masked["outer_validation_histories_used_for_ssl_fit_or_early_stop"] is False,
        "outer validation SSL exposure enabled",
    )
    head = config["robust_residual_head"]
    _require(head["family"] == "sklearn.linear_model.HuberRegressor", "head changed")
    _require(head["one_shared_head_per_fold"] is True, "head count changed")
    _require(head["hyperparameter_search"] is False, "head search enabled")
    reference = config["paired_incumbent_reference"]
    _require(reference["catboost_refit_count"] == 0, "CatBoost refit enabled")
    _require(reference["fixed_router_fit_count"] == 2, "router budget changed")
    _require(reference["router"]["hyperparameter_search"] is False, "router search enabled")
    budget = config["fit_and_runtime_budget"]
    _require(
        (
            budget["masked_encoder_fits"],
            budget["huber_head_fits"],
            budget["reference_router_fits"],
            budget["catboost_fits"],
            budget["total_fit_calls"],
        )
        == (3, 3, 2, 0, 8),
        "fit budget changed",
    )
    sensor = config["sensor_and_extreme_policy"]
    _require(sensor["source_or_canonical_rows_deleted"] == 0, "row deletion enabled")
    _require(sensor["sensor_flags_used_for_membership_or_weighting"] is False, "flags used")
    _require(sensor["high_wave_and_rapid_rise_rows_deleted"] == 0, "extremes deleted")
    _require(all(config["prohibitions"].values()), "a prohibited action was enabled")
    closed = config["closed_family_boundary"]
    _require(closed["hierarchical_residual_basis_dense72"]["reopened"] is False, "dense72 reopened")
    _require(closed["generic_nhits_reopened"] is False, "generic N-HiTS reopened")
    _require(closed["confirmed_catboost_reopened"] is False, "CatBoost lane reopened")


def _implementation_snapshot() -> dict[str, str]:
    paths = {
        "config": ROOT / CONFIG_RELATIVE,
        "runner": ROOT / f"scripts/run_{EXPERIMENT_ID}.py",
        "implementation": ROOT / MODULE_RELATIVE,
        "focused_tests": ROOT / TEST_RELATIVE,
    }
    for path in paths.values():
        _require(path.is_file(), f"implementation file missing: {path.name}")
    return {name: sha256_file(path) for name, path in paths.items()}


def _dependency_paths(config: dict[str, Any]) -> dict[str, Path]:
    reference = config["paired_incumbent_reference"]
    paths = {
        "stage0_receipt": ROOT / config["stage0_authorization"]["receipt"],
        "reused_masked_ssl": ROOT / config["representation"]["reused_module"],
        "dense72_closed_evidence": ROOT
        / config["closed_family_boundary"]["hierarchical_residual_basis_dense72"]["evidence"],
        "champion_feature_columns": ROOT / reference["feature_columns"]["path"],
        "champion_metrics": ROOT / reference["historical_metrics"]["path"],
        "stage0_runner": ROOT / STAGE0_RUNNER_RELATIVE,
    }
    model_root = ROOT / "artifacts/p3_corrected_repeated_forward_catboost_v2/models/folds"
    for fold, files in reference["fold_model_sha256"].items():
        for filename in files:
            paths[f"champion_model/{fold}/{filename}"] = model_root / fold / filename
    return paths


def _expected_dependency_hashes(config: dict[str, Any]) -> dict[str, str]:
    reference = config["paired_incumbent_reference"]
    expected = {
        "stage0_receipt": config["stage0_authorization"]["receipt_sha256"],
        "reused_masked_ssl": config["representation"]["reused_module_sha256"],
        "dense72_closed_evidence": config["closed_family_boundary"]
        ["hierarchical_residual_basis_dense72"]["evidence_sha256"],
        "champion_feature_columns": reference["feature_columns"]["sha256"],
        "champion_metrics": reference["historical_metrics"]["sha256"],
        "stage0_runner": "cc9bae8199226b90631691f4025bdc5cc6b1f5d5e3dd637038207957e21a58df",
    }
    for fold, files in reference["fold_model_sha256"].items():
        for filename, digest in files.items():
            expected[f"champion_model/{fold}/{filename}"] = digest
    return expected


def _verify_dependencies(config: dict[str, Any]) -> dict[str, str]:
    paths = _dependency_paths(config)
    expected = _expected_dependency_hashes(config)
    _require(set(paths) == set(expected), "dependency key set changed")
    observed: dict[str, str] = {}
    for name, path in paths.items():
        _require(path.is_file(), f"sealed dependency missing: {name}")
        observed[name] = sha256_file(path)
        _require(observed[name] == expected[name], f"sealed dependency hash changed: {name}")
    return observed


def _validate_prior_evidence(config: dict[str, Any]) -> dict[str, Any]:
    stage0 = json.loads((ROOT / config["stage0_authorization"]["receipt"]).read_text("utf-8"))
    authorization = config["stage0_authorization"]
    _require(stage0["experiment_id"] == authorization["experiment_id"], "Stage-0 id changed")
    _require(stage0["status"] == authorization["required_status"], "Stage-0 status changed")
    _require(
        stage0["gates"]["overall_preflight_pass"]
        is authorization["required_overall_preflight_pass"],
        "Stage-0 authorization is not PASS",
    )
    _require(
        stage0["support"]["selection_matched_dense_count"]
        == authorization["selection_matched_dense_count"],
        "Stage-0 dense count changed",
    )
    _require(
        stage0["support"]["validation_union_independent_count"]
        == authorization["validation_union_independent_count"],
        "Stage-0 validation count changed",
    )
    _require(stage0["sensor_error_flags"]["rows_deleted_or_masked"] == 0, "Stage-0 deleted rows")
    _require(
        stage0["sensor_error_flags"]["flags_used_for_cohort_membership"] is False,
        "Stage-0 sensor flags changed membership",
    )
    dense = json.loads(
        (
            ROOT
            / config["closed_family_boundary"]["hierarchical_residual_basis_dense72"][
                "evidence"
            ]
        ).read_text("utf-8")
    )
    _require(abs(float(dense["points"][-1]["challenger"]) - 0.8472434865201071) < 1e-12, "dense72 evidence changed")
    _require(
        abs(float(dense["points"][-1]["challenger"] - dense["points"][-1]["incumbent"]) - 0.067295)
        < 1.0e-6,
        "dense72 full delta changed",
    )
    metrics = json.loads(
        (ROOT / config["paired_incumbent_reference"]["historical_metrics"]["path"]).read_text(
            "utf-8"
        )
    )
    historical = config["paired_incumbent_reference"]["historical_metrics"]
    _require(metrics["split_audit"]["validation_case_count"] == 181, "champion cases changed")
    _require(
        abs(float(metrics["metrics"]["final"]["rmse"]) - historical["original_surface_rmse_m"])
        < 1e-12,
        "champion historical RMSE changed",
    )
    return {
        "stage0_status": stage0["status"],
        "stage0_overall_pass": stage0["gates"]["overall_preflight_pass"],
        "stage0_validation_cases": stage0["support"]["validation_union_independent_count"],
        "closed_dense72_fit_count": 45,
        "closed_dense72_full_delta_m": 0.067295,
        "champion_original_surface_cases": 181,
        "champion_original_surface_rmse_m": metrics["metrics"]["final"]["rmse"],
    }


def contract_only() -> dict[str, Any]:
    config = load_config()
    output = ROOT / OUTPUT_RELATIVE
    lock = ROOT / LOCK_RELATIVE
    failure = ROOT / FAILURE_RELATIVE
    _require(not output.exists(), "canonical output already exists")
    _require(not lock.exists(), "one-shot attempt is already consumed")
    _require(not failure.exists(), "failure receipt already exists")
    dependencies = _verify_dependencies(config)
    prior = _validate_prior_evidence(config)
    device = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "catboost_gpu_device_count": int(get_gpu_device_count()),
    }
    _require(device["torch_cuda_available"], "CUDA is unavailable")
    _require(device["torch_cuda_device_count"] == 1, "exactly one Torch GPU is required")
    _require(device["catboost_gpu_device_count"] == 1, "exactly one CatBoost GPU is required")
    return {
        "status": "STATIC_CONTRACT_PASS_ZERO_SOURCE_READ_ZERO_FIT",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "implementation_sha256": _implementation_snapshot(),
        "dependency_sha256": dependencies,
        "prior_evidence": prior,
        "device": device,
        "recipe": recipe_summary(config),
        "source_rows_read": 0,
        "model_fit_count": 0,
    }


def _write_exclusive_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    return sha256_file(path)


def _consume_attempt(
    config: dict[str, Any], implementation: dict[str, str], dependencies: dict[str, str]
) -> dict[str, Any]:
    payload = {
        "created_at_utc": _now(),
        "status": "ATTEMPT_CONSUMED_ONE_SHOT",
        "experiment_id": EXPERIMENT_ID,
        "canonical_config_sha256": EXPECTED_CONFIG_SHA256,
        "implementation_sha256": implementation,
        "dependency_sha256": dependencies,
        "o_excl": True,
        "rerun_forbidden": True,
        "technical_failure_auto_retry": False,
    }
    lock = ROOT / config["canonical_paths"]["attempt_lock"]
    digest = _write_exclusive_json(lock, payload)
    return {**payload, "sha256": digest}


def _load_stage0_runner() -> Any:
    path = ROOT / STAGE0_RUNNER_RELATIVE
    spec = importlib.util.spec_from_file_location("p3_stage0_train_only_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError("failed to load sealed Stage-0 helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _minimum_station_gap_hours(frame: pd.DataFrame) -> float:
    values: list[float] = []
    for _, group in frame.groupby("station", sort=True, observed=True):
        delta = (
            group.sort_values("anchor_time")["anchor_time"]
            .diff()
            .dropna()
            .dt.total_seconds()
            .div(3600.0)
        )
        values.extend(float(value) for value in delta)
    return min(values) if values else float("inf")


def _build_folds(
    matched: pd.DataFrame, config: dict[str, Any], stage0: Any
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    window_mask = np.zeros(len(matched), dtype=bool)
    for window in config["forward_windows"]:
        start = pd.Timestamp(window["validation_start_utc"])
        end = pd.Timestamp(window["validation_end_utc"])
        window_mask |= matched["anchor_time"].ge(start) & matched["anchor_time"].lt(end)
    validation = stage0.select_station_global_independent(
        matched.loc[window_mask].copy(),
        gap_hours=int(config["cohort_contract"]["station_global_validation_gap_hours"]),
    )
    validation["fold"] = ""
    folds: list[dict[str, Any]] = []
    for window in config["forward_windows"]:
        start = pd.Timestamp(window["validation_start_utc"])
        end = pd.Timestamp(window["validation_end_utc"])
        current = validation["anchor_time"].ge(start) & validation["anchor_time"].lt(end)
        validation.loc[current, "fold"] = window["name"]
        cutoff = start - pd.Timedelta(
            hours=int(config["cohort_contract"]["train_cutoff_hours_before_window_start"])
        )
        train = matched.loc[matched["anchor_time"].lt(cutoff)].copy()
        valid = validation.loc[current].copy()
        _require(len(train) == window["expected_dense_train_count"], f"train count changed: {window['name']}")
        _require(len(valid) == window["expected_validation_count"], f"validation count changed: {window['name']}")
        _require(train["anchor_time"].max() < cutoff, f"train cutoff failed: {window['name']}")
        _require(
            not set(train["anchor_id"]).intersection(valid["anchor_id"]),
            f"train/validation anchor overlap: {window['name']}",
        )
        folds.append(
            {
                "name": window["name"],
                "train": train,
                "validation": valid,
                "cutoff": cutoff,
            }
        )
    _require(validation["fold"].ne("").all(), "a validation anchor lacks a fold")
    _require(len(validation) == config["stage0_authorization"]["validation_union_independent_count"], "validation union count changed")
    _require(_minimum_station_gap_hours(validation) >= 78.0, "validation gap below 78h")
    return validation, folds


def _source_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {
        name: sha256_file(paths[name])
        for name in ("README.md", "train_wave.csv", "train_atmos.csv")
    }


def _run_after_lock(
    *,
    p3_dir: Path,
    config: dict[str, Any],
    attempt: dict[str, Any],
    implementation: dict[str, str],
    dependency_before: dict[str, str],
) -> dict[str, Any]:
    started = time.perf_counter()
    stage0 = _load_stage0_runner()
    stage0_config = stage0.load_config(ROOT / "configs/experiments/p3_selection_matched_cohort_preflight_20260830_v1.json")
    source_paths = stage0.resolve_train_only_source_paths(p3_dir)
    _require(
        set(source_paths) == {"root", "README.md", "train_wave.csv", "train_atmos.csv"},
        "source path resolver exceeded the three-file allowlist",
    )
    source_before = _source_hashes(source_paths)
    _require(source_before == config["expected_source_sha256"], "source hashes changed")
    wave, atmos, source_receipt = stage0.load_train_only_sources(source_paths, stage0_config)
    sensor = stage0.sensor_error_flag_aggregates(wave, atmos, stage0_config)
    _require(sensor["rows_deleted_or_masked"] == 0, "sensor diagnostics deleted rows")
    grid, anchors = stage0.build_canonical_train_only_surface(wave, atmos)
    anchors, footprint = stage0.enrich_and_check_anchor_footprints(grid, anchors)
    matched = stage0.build_selection_matched_cohort(anchors, stage0_config)
    _require(len(anchors) == config["stage0_authorization"]["canonical_anchor_count"], "canonical anchor count changed")
    _require(len(matched) == config["stage0_authorization"]["selection_matched_dense_count"], "matched dense count changed")
    validation, folds = _build_folds(matched, config, stage0)
    all_histories = extract_history_sequences(grid, matched)
    history_lookup = {
        int(anchor_id): number
        for number, anchor_id in enumerate(matched["anchor_id"].to_numpy(dtype=np.int64))
    }
    candidate_frames: list[pd.DataFrame] = []
    fold_receipts: list[dict[str, Any]] = []
    seeds = config["representation"]["masked_training"]["fold_seeds"]
    for number, fold in enumerate(folds):
        print(
            json.dumps(
                {
                    "phase": "fit_masked_ssl_then_huber",
                    "fold": fold["name"],
                    "number": number + 1,
                    "of": len(folds),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        train_index = np.asarray(
            [history_lookup[int(value)] for value in fold["train"]["anchor_id"]], dtype=int
        )
        valid_index = np.asarray(
            [history_lookup[int(value)] for value in fold["validation"]["anchor_id"]], dtype=int
        )
        fitted = fit_candidate_fold(
            all_histories[train_index],
            all_histories[valid_index],
            fold["train"].reset_index(drop=True),
            fold["validation"].reset_index(drop=True),
            fold=fold["name"],
            representation_config=config["representation"],
            head_config=config["robust_residual_head"],
            seed=int(seeds[number]),
            device="cuda:0",
        )
        candidate_frames.append(fitted.frame)
        fold_receipts.append(fitted.receipt)
        print(json.dumps(fitted.receipt, ensure_ascii=False), flush=True)

    candidate = pd.concat(candidate_frames, ignore_index=True)
    validation_for_reference = validation.copy().reset_index(drop=True)
    validation_index = np.asarray(
        [history_lookup[int(value)] for value in validation_for_reference["anchor_id"]], dtype=int
    )
    validation_features = summarize_validation_histories(
        all_histories[validation_index], validation_for_reference
    )
    feature_spec = config["paired_incumbent_reference"]["feature_columns"]
    feature_columns = json.loads((ROOT / feature_spec["path"]).read_text("utf-8"))["columns"]
    _require(len(feature_columns) == feature_spec["expected_count"], "champion feature count changed")
    _require(set(feature_columns).issubset(validation_features.columns), "champion features unavailable")
    fold_order = tuple(item["name"] for item in config["forward_windows"])
    model_root = ROOT / "artifacts/p3_corrected_repeated_forward_catboost_v2/models/folds"
    reference_started = time.perf_counter()
    components = saved_catboost_component_predictions(
        validation_features,
        validation_for_reference,
        fold_order=fold_order,
        feature_columns=feature_columns,
        model_root=model_root,
    )
    reference, reference_receipt = apply_paired_prequential_reference(
        components,
        validation_features,
        validation_for_reference,
        fold_order=fold_order,
        reference_config=config["paired_incumbent_reference"],
    )
    reference_receipt["elapsed_seconds"] = float(time.perf_counter() - reference_started)
    comparison = candidate.merge(
        reference[["fold", "anchor_id", "station", "lead_h", "incumbent_prediction", "persistence"]],
        on=["fold", "anchor_id", "station", "lead_h"],
        how="left",
        validate="one_to_one",
    )
    _require(len(comparison) == len(validation) * len(LEADS), "comparison row count changed")
    metric = comparison_metrics(comparison)
    bootstrap_spec = config["evaluation"]["paired_case_bootstrap"]
    bootstrap = paired_case_bootstrap(
        comparison,
        replicates=int(bootstrap_spec["replicates"]),
        seed=int(bootstrap_spec["seed"]),
    )
    source_after = _source_hashes(source_paths)
    dependency_after = _verify_dependencies(config)
    elapsed = float(time.perf_counter() - started)
    actual_fits = {
        "masked_encoder_fits": int(sum(item["ssl"]["fit_count"] for item in fold_receipts)),
        "huber_head_fits": int(sum(item["huber"]["fit_count"] for item in fold_receipts)),
        "reference_router_fits": int(reference_receipt["fixed_router_fit_count"]),
        "catboost_fits": int(reference_receipt["catboost_fit_count"]),
    }
    actual_fits["total_fit_calls"] = int(sum(actual_fits.values()))
    optimizer_steps = int(
        sum(
            item["ssl"]["epochs_completed"]
            * int(np.ceil(item["train_cases"] / config["representation"]["masked_training"]["batch_size"]))
            for item in fold_receipts
        )
    )
    finite_predictions = bool(
        np.isfinite(
            comparison[["candidate_prediction", "incumbent_prediction", "persistence"]].to_numpy(
                dtype=float
            )
        ).all()
    )
    integrity = {
        "stage0_authorization_remains_pass": True,
        "source_allowlist_exact_three_files": True,
        "source_hashes_unchanged_during_run": source_before == source_after,
        "dependency_hashes_unchanged_during_run": dependency_before == dependency_after,
        "canonical_footprint_checks_pass": all(
            value is True
            for key, value in footprint.items()
            if key not in {"official_leads_hours", "maximum_target_horizon_hours"}
        ),
        "validation_station_global_gap_at_least_78h": _minimum_station_gap_hours(validation)
        >= 78.0,
        "all_three_fold_counts_exact": all(
            len(item["train"]) == window["expected_dense_train_count"]
            and len(item["validation"]) == window["expected_validation_count"]
            for item, window in zip(folds, config["forward_windows"], strict=True)
        ),
        "outer_validation_histories_exposed_to_ssl_fit_or_early_stop_zero": all(
            item["ssl"]["outer_validation_cases_exposed_to_ssl_fit_or_early_stop"] == 0
            for item in fold_receipts
        ),
        "sensor_flags_diagnostic_only_no_row_deletion": sensor["rows_deleted_or_masked"] == 0,
        "high_wave_and_rapid_rise_row_deletion_zero": True,
        "huber_convergence_warning_zero": all(
            item["huber"]["convergence_warning_count"] == 0 for item in fold_receipts
        ),
        "actual_fit_budget_exact": actual_fits
        == {
            "masked_encoder_fits": 3,
            "huber_head_fits": 3,
            "reference_router_fits": 2,
            "catboost_fits": 0,
            "total_fit_calls": 8,
        },
        "optimizer_steps_within_preregistered_cap": optimizer_steps
        <= config["fit_and_runtime_budget"]["maximum_encoder_optimizer_steps"],
        "finite_aggregate_predictions": finite_predictions,
        "candidate_predictions_in_0_to_30m": bool(
            comparison["candidate_prediction"].between(0.0, 30.0).all()
        ),
        "no_official_test_sample_baseline_score_submission_or_context_reads": True,
        "csv_output_count_zero": True,
        "upload_attempt_count_zero": True,
        "wall_clock_within_preregistered_budget": elapsed
        <= float(config["fit_and_runtime_budget"]["wall_clock_budget_seconds"]),
    }
    gate = evaluate_promotion_gate(
        metric,
        bootstrap,
        gate_config=config["evaluation"]["promotion_gate"],
        integrity_checks=integrity,
    )
    result: dict[str, Any] = {
        "schema_version": "p3.selection_matched_masked_ssl.result.v1",
        "created_at_utc": _now(),
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_TRAIN_ONLY_ONE_SHOT",
        "decision": gate["decision"],
        "one_shot_attempt": attempt,
        "data_access": {
            "explicit_p3_dir_used": True,
            "opened_source_basenames": ["README.md", "train_wave.csv", "train_atmos.csv"],
            "official_test_rows_read": 0,
            "official_context_rows_read": 0,
            "sample_rows_read": 0,
            "baseline_rows_read": 0,
            "score_rows_read": 0,
            "submission_rows_read": 0,
            "hidden_or_answer_rows_read": 0,
            "csv_output_count": 0,
            "upload_attempt_count": 0,
            "source_rows_modified_or_deleted": 0,
        },
        "source_receipt": source_receipt,
        "cohort": {
            "canonical_anchor_count": int(len(anchors)),
            "selection_matched_dense_count": int(len(matched)),
            "validation_union_independent_count": int(len(validation)),
            "validation_by_window": {
                item["name"]: int(len(item["validation"])) for item in folds
            },
            "validation_by_station": {
                station: int(validation["station"].astype(str).eq(station).sum())
                for station in STATIONS
            },
            "station_global_minimum_gap_hours": float(_minimum_station_gap_hours(validation)),
            "fold_train_dense_count": {
                item["name"]: int(len(item["train"])) for item in folds
            },
        },
        "sensor_error_flags": {
            **sensor,
            "flags_used_for_cohort_membership_or_weighting": False,
            "high_wave_and_rapid_rise_rows_deleted": 0,
            "extreme_storm_rows_deleted": 0,
        },
        "fit_budget": {
            "preregistered": config["fit_and_runtime_budget"],
            "actual": actual_fits,
            "actual_encoder_optimizer_steps": optimizer_steps,
            "fold_receipts": fold_receipts,
            "paired_reference": reference_receipt,
            "total_elapsed_seconds": elapsed,
        },
        "paired_comparison": {
            "surface_note": "All metrics use the same 157 Stage-0 selection-matched historical anchors and six official leads.",
            "original_champion_headline_is_not_paired_here": config[
                "paired_incumbent_reference"
            ]["historical_metrics"],
            "metrics": metric,
            "paired_case_bootstrap": bootstrap,
        },
        "promotion_gate": gate,
        "integrity_checks": integrity,
        "closed_family_boundary": config["closed_family_boundary"],
        "provenance": {
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "implementation_sha256": implementation,
            "dependency_sha256_before": dependency_before,
            "dependency_sha256_after": dependency_after,
            "source_sha256_before": source_before,
            "source_sha256_after": source_after,
        },
        "execution": {
            "raw_prediction_rows_persisted": 0,
            "aggregate_json_files_created": 1,
            "model_files_created": 0,
            "catboost_models_loaded_for_inference": 6,
            "catboost_models_fit": 0,
            "result_based_tuning_or_retry": False,
            "technical_failure_auto_retry": False,
            "submission_or_upload_attempted": False,
        },
    }
    result["seal"] = {
        "algorithm": "sha256",
        "payload_without_seal_sha256": _payload_sha256(result),
    }
    stage_parent = ROOT / "tmp"
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f"{EXPERIMENT_ID}_", dir=stage_parent))
    result_path = stage / "result.json"
    result_sha = _write_exclusive_json(result_path, result)
    output = ROOT / OUTPUT_RELATIVE
    _require(not output.exists(), "canonical output appeared before finalization")
    stage.replace(output)
    return {
        "status": result["status"],
        "decision": result["decision"],
        "artifact": OUTPUT_RELATIVE,
        "result_sha256": result_sha,
        "overall": metric["overall"],
        "fit_count": actual_fits,
        "elapsed_seconds": elapsed,
    }


def run_experiment(p3_dir: Path) -> dict[str, Any]:
    config = load_config()
    output = ROOT / OUTPUT_RELATIVE
    lock = ROOT / LOCK_RELATIVE
    failure = ROOT / FAILURE_RELATIVE
    _require(not output.exists(), "canonical output already exists")
    _require(not lock.exists(), "one-shot attempt is already consumed")
    _require(not failure.exists(), "failure receipt already exists")
    dependencies = _verify_dependencies(config)
    _validate_prior_evidence(config)
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(torch.cuda.device_count() == 1, "exactly one Torch GPU is required")
    _require(get_gpu_device_count() == 1, "exactly one CatBoost GPU is required")
    implementation = _implementation_snapshot()
    attempt = _consume_attempt(config, implementation, dependencies)
    started = time.perf_counter()
    try:
        return _run_after_lock(
            p3_dir=p3_dir,
            config=config,
            attempt=attempt,
            implementation=implementation,
            dependency_before=dependencies,
        )
    except BaseException as exc:
        failure_payload = {
            "created_at_utc": _now(),
            "status": "TECHNICAL_FAILURE_ATTEMPT_CONSUMED_NO_AUTO_RETRY",
            "experiment_id": EXPERIMENT_ID,
            "attempt_lock_sha256": attempt["sha256"],
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "elapsed_seconds": float(time.perf_counter() - started),
            "rerun_forbidden": True,
        }
        try:
            _write_exclusive_json(failure, failure_payload)
        except BaseException:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p3-dir", type=Path)
    parser.add_argument("--contract-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.contract_only:
        if args.p3_dir is not None:
            raise SystemExit("--contract-only must not receive --p3-dir")
        result = contract_only()
    else:
        if args.p3_dir is None:
            raise SystemExit("actual one-shot run requires explicit --p3-dir")
        result = run_experiment(args.p3_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
