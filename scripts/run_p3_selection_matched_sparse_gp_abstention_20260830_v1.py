"""Run the sealed train-only P3 Bayesian RFF abstention experiment exactly once."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p3_wave.p3_selection_matched_sparse_gp_abstention_20260830_v1 import (
    LEADS,
    STATIONS,
    classify_evidence,
    comparison_metrics,
    contiguous_anchor_day_block_bootstrap,
    fit_predict_fold,
)
from p3_wave.selection_matched_masked_ssl_20260830_v1 import (
    apply_paired_prequential_reference,
    extract_history_sequences,
    saved_catboost_component_predictions,
    summarize_validation_histories,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_selection_matched_sparse_gp_abstention_20260830_v1"
CONFIG_RELATIVE = f"configs/experiments/{EXPERIMENT_ID}.json"
MODULE_RELATIVE = f"src/p3_wave/{EXPERIMENT_ID}.py"
RUNNER_RELATIVE = f"scripts/run_{EXPERIMENT_ID}.py"
TEST_RELATIVE = f"tests/test_{EXPERIMENT_ID}.py"
QA_RUNNER_RELATIVE = f"scripts/qa_{EXPERIMENT_ID}.py"
POLICY_RELATIVE = "configs/goals/metric_aligned_gate_recalibration_20260830_v1.json"
STAGE0_RUNNER_RELATIVE = "scripts/run_p3_selection_matched_cohort_preflight_20260830_v1.py"
STAGE0_CONFIG_RELATIVE = "configs/experiments/p3_selection_matched_cohort_preflight_20260830_v1.json"
HISTORY_HELPER_RELATIVE = "src/p3_wave/selection_matched_masked_ssl_20260830_v1.py"
EXPECTED_CONFIG_SHA256 = "d941e3516bc295e8e10c7dffbd4df4ccee0ed276d98a0b67f8e7875764717611"


class ContractError(ValueError):
    """Raised when a sealed pre-fit contract differs from preregistration."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


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


def _write_exclusive_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


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
        config["schema_version"]
        == "p3.selection_matched_sparse_gp_abstention.preregistration.20260830.v1",
        "schema version changed",
    )
    _require(config["experiment_id"] == EXPERIMENT_ID, "experiment id changed")
    paths = config["canonical_paths"]
    expected_paths = {
        "config": CONFIG_RELATIVE,
        "runner": RUNNER_RELATIVE,
        "implementation": MODULE_RELATIVE,
        "focused_tests": TEST_RELATIVE,
        "qa_runner": QA_RUNNER_RELATIVE,
        "result": f"reports/{EXPERIMENT_ID}/result.json",
        "attempt_lock": f"reports/{EXPERIMENT_ID}/ATTEMPT_LOCK.json",
        "failure_receipt": f"reports/{EXPERIMENT_ID}/FAILED.json",
        "qa_result": f"reports/{EXPERIMENT_ID}/independent-qa.json",
    }
    _require(paths == expected_paths, "canonical output paths changed")
    governing = config["governing_policy"]
    _require(governing["path"] == POLICY_RELATIVE, "governing policy path changed")
    _require(
        governing["sha256"]
        == "48f200b6f239ae19d2feed281864c34efbfd6ade7651f1a8290beff1caac0f65",
        "governing policy hash changed",
    )
    boundary = config["data_boundary"]
    _require(
        boundary["allowed_source_basenames"]
        == ["README.md", "train_wave.csv", "train_atmos.csv"],
        "source allowlist changed",
    )
    _require(boundary["explicit_p3_dir_required"] is True, "explicit P3 path disabled")
    _require(boundary["directory_listing_allowed"] is False, "directory listing enabled")
    _require(boundary["load_p3_data_allowed"] is False, "broad P3 loader enabled")
    _require(
        boundary["official_test_context_index_sample_baseline_score_submission_hidden_reads"]
        == 0,
        "official-interface read enabled",
    )
    cohort = config["cohort_contract"]
    _require(cohort["history_hours"] == 48, "history changed")
    _require(cohort["history_rows_including_anchor"] == 289, "history rows changed")
    _require(tuple(cohort["official_leads_hours"]) == LEADS, "lead set changed")
    _require(cohort["current_hs_min_inclusive_m"] == 1.5, "lower Hs changed")
    _require(cohort["current_hs_max_exclusive_m"] == 2.2, "upper Hs changed")
    _require(cohort["rise_lookback_hours"] == 12, "rise lookback changed")
    _require(cohort["rise_min_exclusive_m"] == 0.2, "rise threshold changed")
    _require(cohort["station_global_validation_gap_hours"] == 78, "spacing changed")
    _require(cohort["train_cutoff_hours_before_window_start"] == 78, "cutoff changed")
    windows = [
        (item["name"], item["expected_dense_train_count"], item["expected_validation_count"])
        for item in config["forward_windows"]
    ]
    _require(
        windows
        == [
            ("2024_h2_storm", 740, 41),
            ("winter_transition", 1135, 65),
            ("2025_h1", 1737, 51),
        ],
        "frozen folds changed",
    )
    recipe = config["candidate_recipe"]
    _require(
        recipe["family"] == "fixed_bayesian_random_fourier_multi_lead_residual_mean",
        "candidate family changed",
    )
    _require(recipe["random_feature_count"] == 64, "random feature count changed")
    _require(recipe["fold_seeds"] == [20260830, 20260831, 20260832], "seeds changed")
    _require(recipe["ridge_precision"] == 25.0, "ridge changed")
    _require(recipe["posterior_mean_interval"] == 0.9, "interval changed")
    _require(recipe["maximum_absolute_correction_m"] == 0.1, "cap changed")
    _require(recipe["candidate_total_fit_count"] == 3, "fit count changed")
    _require(recipe["hyperparameter_search_count"] == 0, "HPO enabled")
    _require(recipe["outlier_hard_deletion_count"] == 0, "row deletion enabled")
    _require(
        recipe["sensor_flags_used_for_membership_or_weighting"] is False,
        "sensor flags changed membership or weighting",
    )
    reference = config["paired_incumbent_reference"]
    _require(reference["catboost_refit_count"] == 0, "CatBoost refit enabled")
    _require(reference["fixed_router_fit_count"] == 2, "router fit count changed")
    _require(reference["router"]["hyperparameter_search"] is False, "router HPO enabled")
    uncertainty = config["uncertainty"]
    _require(uncertainty["episode_id_column"] is None, "undeclared episode ID added")
    _require(uncertainty["fallback_block_length_anchor_days"] == 3, "block length changed")
    _require(uncertainty["replicates"] == 5000, "bootstrap count changed")
    decision = config["decision_policy"]
    _require(decision["primary_metric"] == "pooled all-row six-lead Hs RMSE_m", "primary changed")
    _require(decision["fatal_hard_gates_only"] is True, "fatal hierarchy changed")
    _require(decision["legacy_minimum_0_01m_applied"] is False, "legacy margin restored")
    _require(decision["legacy_two_of_three_windows_applied"] is False, "window veto restored")
    _require(decision["legacy_worst_lead_0_02m_cap_applied"] is False, "lead veto restored")
    execution = config["execution_contract"]
    _require(execution["maximum_executions"] == 1, "one-shot limit changed")
    _require(execution["candidate_fit_count"] == 3, "candidate fit budget changed")
    _require(execution["candidate_hyperparameter_search_count"] == 0, "HPO budget changed")
    _require(execution["raw_predictions_persisted"] == 0, "raw output enabled")
    _require(execution["outlier_hard_deletion_count"] == 0, "deletion budget changed")
    _require(all(config["prohibitions"].values()), "a prohibited action was enabled")


def _implementation_snapshot() -> dict[str, str]:
    paths = {
        "config": ROOT / CONFIG_RELATIVE,
        "runner": ROOT / RUNNER_RELATIVE,
        "implementation": ROOT / MODULE_RELATIVE,
        "focused_tests": ROOT / TEST_RELATIVE,
        "qa_runner": ROOT / QA_RUNNER_RELATIVE,
    }
    for path in paths.values():
        _require(path.is_file(), f"implementation file missing: {path.name}")
    return {name: sha256_file(path) for name, path in paths.items()}


def _dependency_paths(config: dict[str, Any]) -> dict[str, Path]:
    reference = config["paired_incumbent_reference"]
    paths = {
        "governing_policy": ROOT / config["governing_policy"]["path"],
        "stage0_receipt": ROOT / config["stage0_authorization"]["receipt"],
        "stage0_runner": ROOT / config["dependency_sha256"]["stage0_runner"]["path"],
        "stage0_config": ROOT / config["dependency_sha256"]["stage0_config"]["path"],
        "history_and_reference_helpers": ROOT
        / config["dependency_sha256"]["history_and_reference_helpers"]["path"],
        "champion_feature_columns": ROOT / reference["feature_columns"]["path"],
    }
    model_root = ROOT / "artifacts/p3_corrected_repeated_forward_catboost_v2/models/folds"
    for fold, files in reference["fold_model_sha256"].items():
        for filename in files:
            paths[f"champion_model/{fold}/{filename}"] = model_root / fold / filename
    return paths


def _expected_dependency_hashes(config: dict[str, Any]) -> dict[str, str]:
    reference = config["paired_incumbent_reference"]
    expected = {
        "governing_policy": config["governing_policy"]["sha256"],
        "stage0_receipt": config["stage0_authorization"]["receipt_sha256"],
        "stage0_runner": config["dependency_sha256"]["stage0_runner"]["sha256"],
        "stage0_config": config["dependency_sha256"]["stage0_config"]["sha256"],
        "history_and_reference_helpers": config["dependency_sha256"]
        ["history_and_reference_helpers"]["sha256"],
        "champion_feature_columns": reference["feature_columns"]["sha256"],
    }
    for fold, files in reference["fold_model_sha256"].items():
        for filename, digest in files.items():
            expected[f"champion_model/{fold}/{filename}"] = digest
    return expected


def _verify_dependencies(config: dict[str, Any]) -> dict[str, str]:
    paths = _dependency_paths(config)
    expected = _expected_dependency_hashes(config)
    _require(set(paths) == set(expected), "dependency inventory changed")
    observed: dict[str, str] = {}
    for name, path in paths.items():
        _require(path.is_file(), f"sealed dependency missing: {name}")
        observed[name] = sha256_file(path)
        _require(observed[name] == expected[name], f"dependency hash changed: {name}")
    return observed


def _validate_authorization(config: dict[str, Any]) -> dict[str, Any]:
    policy = json.loads((ROOT / POLICY_RELATIVE).read_text(encoding="utf-8"))
    _require(
        policy["status"] == config["governing_policy"]["required_status"],
        "governing policy status changed",
    )
    _require(
        policy["problem_units"]["P3"]["paired_unit"]
        == "storm episode where available, otherwise contiguous anchor-day block with all six leads intact",
        "P3 dependence unit changed",
    )
    stage0 = json.loads(
        (ROOT / config["stage0_authorization"]["receipt"]).read_text(encoding="utf-8")
    )
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
        "Stage-0 matched support changed",
    )
    _require(
        stage0["support"]["validation_union_independent_count"]
        == authorization["validation_union_independent_count"],
        "Stage-0 validation support changed",
    )
    _require(stage0["sensor_error_flags"]["rows_deleted_or_masked"] == 0, "Stage-0 deleted rows")
    _require(
        stage0["sensor_error_flags"]["flags_used_for_cohort_membership"] is False,
        "Stage-0 sensor flags changed membership",
    )
    return {
        "policy_status": policy["status"],
        "stage0_status": stage0["status"],
        "stage0_overall_preflight_pass": stage0["gates"]["overall_preflight_pass"],
        "stage0_selection_matched_dense_count": stage0["support"][
            "selection_matched_dense_count"
        ],
        "stage0_validation_union_independent_count": stage0["support"][
            "validation_union_independent_count"
        ],
    }


def contract_only() -> dict[str, Any]:
    config = load_config()
    for key in ("result", "attempt_lock", "failure_receipt", "qa_result"):
        _require(not (ROOT / config["canonical_paths"][key]).exists(), f"{key} already exists")
    dependencies = _verify_dependencies(config)
    authorization = _validate_authorization(config)
    return {
        "status": "STATIC_CONTRACT_PASS_ZERO_SOURCE_READ_ZERO_FIT",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "implementation_sha256": _implementation_snapshot(),
        "dependency_sha256": dependencies,
        "authorization": authorization,
        "source_rows_read": 0,
        "model_fit_count": 0,
        "hyperparameter_search_count": 0,
    }


def _consume_attempt(
    config: dict[str, Any], implementation: dict[str, str], dependencies: dict[str, str]
) -> dict[str, Any]:
    payload = {
        "schema_version": "p3.one_shot_attempt_lock.20260830.v1",
        "created_at_utc": _now(),
        "status": "ATTEMPT_CONSUMED_ONE_SHOT",
        "experiment_id": EXPERIMENT_ID,
        "canonical_config_sha256": EXPECTED_CONFIG_SHA256,
        "implementation_sha256": implementation,
        "dependency_sha256": dependencies,
        "candidate_fit_budget": 3,
        "hyperparameter_search_budget": 0,
        "o_excl": True,
        "rerun_forbidden": True,
        "technical_failure_auto_retry": False,
    }
    digest = _write_exclusive_json(ROOT / config["canonical_paths"]["attempt_lock"], payload)
    return {**payload, "sha256": digest}


def _load_stage0_runner() -> Any:
    path = ROOT / STAGE0_RUNNER_RELATIVE
    spec = importlib.util.spec_from_file_location("p3_stage0_train_only_helpers_gp", path)
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


def _frame_contract_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        digest.update(("|".join(str(value) for value in row) + "\n").encode("utf-8"))
    return digest.hexdigest()


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
                "train": train.reset_index(drop=True),
                "validation": valid.reset_index(drop=True),
                "cutoff": cutoff,
            }
        )
    _require(validation["fold"].ne("").all(), "a validation anchor lacks a fold")
    _require(
        len(validation) == config["stage0_authorization"]["validation_union_independent_count"],
        "validation union count changed",
    )
    _require(_minimum_station_gap_hours(validation) >= 78.0, "validation gap below 78h")
    return validation.reset_index(drop=True), folds


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
    stage0_config = stage0.load_config(ROOT / STAGE0_CONFIG_RELATIVE)
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
    _require(sensor["flags_used_for_cohort_membership"] is False, "sensor flags changed membership")
    grid, base_anchors = stage0.build_canonical_train_only_surface(wave, atmos)
    anchors, footprint = stage0.enrich_and_check_anchor_footprints(grid, base_anchors)
    matched = stage0.build_selection_matched_cohort(anchors, stage0_config)
    _require(len(anchors) == config["stage0_authorization"]["canonical_anchor_count"], "anchor count changed")
    _require(len(matched) == config["stage0_authorization"]["selection_matched_dense_count"], "matched count changed")
    validation, folds = _build_folds(matched, config, stage0)

    histories = extract_history_sequences(grid, matched)
    history_lookup = {
        int(anchor_id): position
        for position, anchor_id in enumerate(matched["anchor_id"].to_numpy(dtype=np.int64))
    }
    features = summarize_validation_histories(histories, matched)
    feature_spec = config["paired_incumbent_reference"]["feature_columns"]
    feature_columns = json.loads((ROOT / feature_spec["path"]).read_text(encoding="utf-8"))["columns"]
    _require(len(feature_columns) == feature_spec["expected_count"], "feature count changed")
    _require(set(feature_columns).issubset(features.columns), "past-only features unavailable")

    validation_index = np.asarray(
        [history_lookup[int(value)] for value in validation["anchor_id"]], dtype=int
    )
    validation_features = features.iloc[validation_index].reset_index(drop=True)
    fold_order = tuple(item["name"] for item in config["forward_windows"])
    model_root = ROOT / "artifacts/p3_corrected_repeated_forward_catboost_v2/models/folds"
    reference_started = time.perf_counter()
    components = saved_catboost_component_predictions(
        validation_features,
        validation,
        fold_order=fold_order,
        feature_columns=feature_columns,
        model_root=model_root,
    )
    reference, reference_receipt = apply_paired_prequential_reference(
        components,
        validation_features,
        validation,
        fold_order=fold_order,
        reference_config=config["paired_incumbent_reference"],
    )
    reference_receipt["elapsed_seconds"] = float(time.perf_counter() - reference_started)

    recipe = {**config["candidate_recipe"], "feature_columns": feature_columns}
    candidate_frames: list[pd.DataFrame] = []
    fold_receipts: list[dict[str, Any]] = []
    for number, fold in enumerate(folds):
        print(
            json.dumps(
                {
                    "phase": "fit_fixed_bayesian_rff_residual_head",
                    "fold": fold["name"],
                    "number": number + 1,
                    "of": len(folds),
                }
            ),
            flush=True,
        )
        train_index = np.asarray(
            [history_lookup[int(value)] for value in fold["train"]["anchor_id"]], dtype=int
        )
        valid_index = np.asarray(
            [history_lookup[int(value)] for value in fold["validation"]["anchor_id"]], dtype=int
        )
        fold_reference = reference.loc[
            reference["fold"].astype(str).eq(fold["name"]),
            ["anchor_id", "lead_h", "incumbent_prediction"],
        ].copy()
        predicted = fit_predict_fold(
            features.iloc[train_index].reset_index(drop=True),
            features.iloc[valid_index].reset_index(drop=True),
            fold["train"],
            fold["validation"],
            fold_reference,
            recipe=recipe,
            seed=int(recipe["fold_seeds"][number]),
        )
        candidate_frames.append(predicted.frame)
        fold_receipts.append(predicted.receipt)
        print(json.dumps(predicted.receipt, ensure_ascii=False), flush=True)

    candidate = pd.concat(candidate_frames, ignore_index=True)
    comparison = candidate.merge(
        reference[
            ["fold", "anchor_id", "station", "lead_h", "incumbent_prediction", "persistence"]
        ],
        on=["fold", "anchor_id", "station", "lead_h"],
        how="left",
        validate="one_to_one",
    )
    expected_rows = len(validation) * len(LEADS)
    _require(len(comparison) == expected_rows, "comparison row count changed")
    _require(
        comparison[["incumbent_prediction", "persistence"]].notna().all().all(),
        "paired incumbent alignment failed",
    )
    metrics = comparison_metrics(comparison)
    uncertainty = config["uncertainty"]
    bootstrap = contiguous_anchor_day_block_bootstrap(
        comparison,
        replicates=int(uncertainty["replicates"]),
        seed=int(uncertainty["seed"]),
        block_length_days=int(uncertainty["fallback_block_length_anchor_days"]),
    )
    source_after = _source_hashes(source_paths)
    dependency_after = _verify_dependencies(config)
    elapsed = float(time.perf_counter() - started)
    candidate_fit_count = int(sum(item["fit"]["fit_count"] for item in fold_receipts))
    hpo_count = int(
        sum(item["fit"]["hyperparameter_search_count"] for item in fold_receipts)
    )
    exact_incumbent_rows = int(
        sum(item["prediction"]["exact_incumbent_rows"] for item in fold_receipts)
    )
    active_rows = int(
        sum(item["prediction"]["active_correction_rows"] for item in fold_receipts)
    )
    keys = ["fold", "anchor_id", "station", "lead_h"]
    candidate_key_hash = _frame_contract_sha256(candidate.sort_values(keys), keys)
    reference_key_hash = _frame_contract_sha256(reference.sort_values(keys), keys)
    finite_columns = [
        "target_hs",
        "candidate_prediction",
        "incumbent_prediction",
        "persistence",
        "correction_mean_m",
        "posterior_mean_sd_m",
        "correction_applied_m",
    ]
    fatal_integrity = {
        "governing_policy_and_preregistered_schema_match": True,
        "source_allowlist_exact_three_training_files": True,
        "source_hashes_unchanged_during_run": source_before == source_after,
        "dependency_hashes_unchanged_during_run": dependency_before == dependency_after,
        "canonical_anchor_footprints_pass": all(
            value is True
            for key, value in footprint.items()
            if key not in {"official_leads_hours", "maximum_target_horizon_hours"}
        ),
        "selection_population_and_all_fold_counts_exact": bool(
            len(anchors) == 8121
            and len(matched) == 2131
            and len(validation) == 157
            and [len(item["validation"]) for item in folds] == [41, 65, 51]
        ),
        "strict_past_only_train_cutoffs_and_no_anchor_overlap": all(
            item["train"]["anchor_time"].max() < item["cutoff"]
            and not set(item["train"]["anchor_id"]).intersection(item["validation"]["anchor_id"])
            for item in folds
        ),
        "station_global_validation_gap_at_least_78h": _minimum_station_gap_hours(validation)
        >= 78.0,
        "metric_truth_keys_rows_order_and_comparator_lineage_match": bool(
            len(candidate) == expected_rows
            and len(reference) == expected_rows
            and candidate_key_hash == reference_key_hash
            and not comparison.duplicated(keys).any()
        ),
        "finite_predictions_and_statistics": bool(
            np.isfinite(comparison[finite_columns].to_numpy(dtype=float)).all()
            and np.isfinite(bootstrap["benefit_ci90_m"]).all()
        ),
        "prediction_schema_domain_hs_0_to_30m": bool(
            comparison["candidate_prediction"].between(0.0, 30.0).all()
            and comparison["incumbent_prediction"].between(0.0, 30.0).all()
        ),
        "candidate_fit_count_three_hpo_zero": bool(
            candidate_fit_count == config["execution_contract"]["candidate_fit_count"]
            and hpo_count == 0
        ),
        "incumbent_lineage_zero_catboost_refits_two_fixed_router_fits": bool(
            reference_receipt["catboost_fit_count"] == 0
            and reference_receipt["fixed_router_fit_count"] == 2
        ),
        "sensor_flags_diagnostic_only_and_outlier_deletion_zero": bool(
            sensor["rows_deleted_or_masked"] == 0
            and sensor["flags_used_for_cohort_membership"] is False
            and all(item["fit"]["rows_deleted"] == 0 for item in fold_receipts)
            and all(item["prediction"]["rows_deleted"] == 0 for item in fold_receipts)
        ),
        "official_interface_reads_csv_upload_and_raw_output_zero": True,
        "one_shot_no_result_driven_retry_or_mutation": True,
    }
    benefit_point = float(metrics["overall"]["benefit_incumbent_minus_candidate_rmse_m"])
    evidence_state = classify_evidence(
        benefit_point=benefit_point,
        benefit_ci90=bootstrap["benefit_ci90_m"],
        fatal_integrity_checks=fatal_integrity,
    )
    numerical_warnings = {
        "any_validation_ood_cases": bool(
            any(item["prediction"]["design"]["ood_rows"] > 0 for item in fold_receipts)
        ),
        "any_posterior_abstention": bool(exact_incumbent_rows > 0),
        "no_active_corrections": bool(active_rows == 0),
        "precision_condition_number_above_1e8": bool(
            any(item["fit"]["precision_condition_number"] > 1.0e8 for item in fold_receipts)
        ),
        "wall_clock_above_preregistered_budget": bool(
            elapsed > float(config["execution_contract"]["maximum_wall_seconds"])
        ),
    }
    result: dict[str, Any] = {
        "schema_version": "p3.selection_matched_sparse_gp_abstention.result.20260830.v1",
        "created_at_utc": _now(),
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_TRAIN_ONLY_ONE_SHOT_RESEARCH_ONLY",
        "evidence_state": evidence_state,
        "official_action_authorized": False,
        "one_shot_attempt": attempt,
        "data_access": {
            "explicit_p3_dir_used": True,
            "opened_source_basenames": ["README.md", "train_wave.csv", "train_atmos.csv"],
            "forbidden_source_basenames_opened": [],
            "official_test_rows_read": 0,
            "official_context_rows_read": 0,
            "test_index_rows_read": 0,
            "sample_rows_read": 0,
            "baseline_rows_read": 0,
            "score_rows_read": 0,
            "submission_rows_read_or_written": 0,
            "hidden_or_answer_rows_read": 0,
            "csv_output_count": 0,
            "upload_attempt_count": 0,
            "raw_prediction_rows_persisted": 0,
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
            "fold_train_dense_count": {
                item["name"]: int(len(item["train"])) for item in folds
            },
            "station_global_minimum_gap_hours": float(_minimum_station_gap_hours(validation)),
            "validation_key_sha256": candidate_key_hash,
        },
        "sensor_error_flags": {
            **sensor,
            "flags_used_for_cohort_membership_or_weighting": False,
            "high_wave_and_rapid_rise_rows_deleted": 0,
            "extreme_storm_rows_deleted": 0,
        },
        "fit_budget": {
            "preregistered_candidate_fits": 3,
            "actual_candidate_fits": candidate_fit_count,
            "hyperparameter_search_count": hpo_count,
            "reference_router_fits": int(reference_receipt["fixed_router_fit_count"]),
            "catboost_refits": int(reference_receipt["catboost_fit_count"]),
            "saved_catboost_models_loaded_for_inference": int(
                reference_receipt["catboost_model_load_count"]
            ),
            "fold_receipts": fold_receipts,
            "paired_reference": reference_receipt,
            "total_elapsed_seconds": elapsed,
        },
        "abstention": {
            "active_correction_rows": active_rows,
            "exact_incumbent_rows": exact_incumbent_rows,
            "total_rows": int(len(comparison)),
            "active_fraction": float(active_rows / len(comparison)),
            "exact_incumbent_fraction": float(exact_incumbent_rows / len(comparison)),
            "maximum_absolute_correction_m": float(
                comparison["correction_applied_m"].abs().max()
            ),
        },
        "primary_evaluation": {
            "metric": "exact pooled all-row six-lead Hs RMSE_m",
            "benefit_definition": "paired_incumbent_RMSE_minus_candidate_RMSE",
            "metrics": metrics,
            "dependence_aware_interval": bootstrap,
            "decision_basis": {
                "pooled_directional_benefit_primary": True,
                "dependence_aware_uncertainty_primary": True,
                "arbitrary_numeric_magnitude_tier_applied": False,
                "legacy_minimum_0_01m_applied": False,
                "legacy_two_of_three_windows_applied": False,
                "legacy_worst_lead_0_02m_cap_applied": False,
                "window_station_lead_results_diagnostic_only": True,
            },
        },
        "validity": {
            "fatal_hard_gates": fatal_integrity,
            "all_fatal_hard_gates_pass": bool(all(fatal_integrity.values())),
            "nonfatal_numerical_and_transport_warnings": numerical_warnings,
            "warnings_do_not_override_primary_performance_state": True,
        },
        "provenance": {
            "governing_policy_path": POLICY_RELATIVE,
            "governing_policy_sha256": config["governing_policy"]["sha256"],
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "implementation_sha256": implementation,
            "dependency_sha256_before": dependency_before,
            "dependency_sha256_after": dependency_after,
            "source_sha256_before": source_before,
            "source_sha256_after": source_after,
            "candidate_key_sha256": candidate_key_hash,
            "incumbent_key_sha256": reference_key_hash,
        },
        "execution": {
            "result_based_tuning_or_retry": False,
            "technical_failure_auto_retry": False,
            "raw_prediction_rows_persisted": 0,
            "aggregate_json_files_created_by_runner": 2,
            "model_files_created": 0,
            "csv_output_count": 0,
            "submission_or_upload_attempted": False,
        },
    }
    result["seal"] = {
        "algorithm": "sha256",
        "payload_without_seal_sha256": _payload_sha256(result),
    }
    result_sha = _write_exclusive_json(ROOT / config["canonical_paths"]["result"], result)
    return {
        "status": result["status"],
        "evidence_state": evidence_state,
        "result": config["canonical_paths"]["result"],
        "result_sha256": result_sha,
        "candidate_fits": candidate_fit_count,
        "hyperparameter_search_count": hpo_count,
        "elapsed_seconds": elapsed,
        "primary": metrics["overall"],
        "benefit_ci90_m": bootstrap["benefit_ci90_m"],
        "active_correction_rows": active_rows,
    }


def run_experiment(p3_dir: Path) -> dict[str, Any]:
    config = load_config()
    for key in ("result", "attempt_lock", "failure_receipt", "qa_result"):
        _require(not (ROOT / config["canonical_paths"][key]).exists(), f"{key} already exists")
    dependencies = _verify_dependencies(config)
    _validate_authorization(config)
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
        failure = {
            "schema_version": "p3.one_shot_failure.20260830.v1",
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
            _write_exclusive_json(ROOT / config["canonical_paths"]["failure_receipt"], failure)
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
