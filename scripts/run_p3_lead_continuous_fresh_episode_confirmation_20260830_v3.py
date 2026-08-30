"""Exactly-once train-only P3 lead-continuous fresh-episode confirmation.

The runner opens only README.md, train_wave.csv, and train_atmos.csv from an
explicit P3 directory.  It reconstructs one timestamp-disjoint post-H1 anchor,
seals saved-reference and frozen-recipe predictions before using that anchor's
future targets, and reports aggregate paired six-lead RMSE only.  It never reads
or writes an official/test/submission artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402

from p3_wave.p3_lead_continuous_fresh_episode_confirmation_20260830_v3 import (  # noqa: E402
    LEADS,
    array_sha256,
    build_comparison_frame,
    classify_terminal,
    comparison_metrics,
    frame_contract_sha256,
    predict_frozen_reference,
    select_fresh_surface,
    uncertainty_or_insufficient,
)
from p3_wave.selection_matched_masked_ssl_20260830_v1 import (  # noqa: E402
    extract_history_sequences,
    summarize_validation_histories,
)

EXPERIMENT_ID = "p3_lead_continuous_fresh_episode_confirmation_20260830_v3"
DEFAULT_CONFIG = (
    ROOT
    / "configs/experiments/p3_lead_continuous_fresh_episode_confirmation_20260830_v3.json"
)


class ContractError(RuntimeError):
    """Raised when a sealed exactly-once contract differs."""


def _now_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _write_exclusive_json(path: Path, payload: Any) -> str:
    data = _canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        view = memoryview(data)
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("exclusive JSON write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(data).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load dependency: {path.relative_to(ROOT).as_posix()}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_config(payload)
    return payload


def validate_config(config: dict[str, Any]) -> None:
    _require(config["experiment_id"] == EXPERIMENT_ID, "experiment id changed")
    _require(
        config["status"] == "PREREGISTERED_EXACTLY_ONCE_FRESH_SINGLE_EPISODE",
        "preregistration status changed",
    )
    boundary = config["data_boundary"]
    _require(
        boundary["allowed_source_basenames"]
        == ["README.md", "train_wave.csv", "train_atmos.csv"],
        "source allowlist changed",
    )
    _require(boundary["load_p3_data_allowed"] is False, "load_p3_data enabled")
    _require(boundary["csv_output_count"] == 0, "CSV output enabled")
    _require(boundary["upload_count"] == 0, "upload enabled")
    surface = config["fresh_surface"]
    _require(surface["historical_three_window_fresh_case_count"] == 0, "prior audit changed")
    _require(surface["same_station_exposure_separation_hours"] == 78, "gap changed")
    _require(surface["expected_fresh_independent_case_count"] == 1, "fresh count changed")
    _require(surface["target_values_used_to_choose_surface"] == 0, "target selection enabled")
    recipe = config["candidate_recipe"]
    _require(recipe["ridge_alpha"] == 16.0, "candidate ridge changed")
    _require(recipe["candidate_fit_count"] == 1, "candidate fit count changed")
    _require(recipe["hyperparameter_search_count"] == 0, "candidate HPO enabled")
    _require(recipe["result_based_tuning"] is False, "result tuning enabled")
    reference = config["paired_incumbent"]
    _require(reference["catboost_fit_count"] == 0, "CatBoost refit enabled")
    _require(reference["router_fit_count"] == 0, "router refit enabled")
    _require(reference["long_lead_persistence_shrink_weight"] == 0.2, "shrink changed")
    uncertainty = config["uncertainty"]
    _require(uncertainty["available_independent_blocks_expected"] == 1, "support changed")
    _require(uncertainty["minimum_independent_blocks_for_interval"] == 2, "CI gate changed")
    execution = config["execution_contract"]
    _require(execution["maximum_executions"] == 1, "exactly-once changed")
    _require(execution["technical_failure_auto_retry"] is False, "auto-retry enabled")
    _require(execution["result_based_retry"] is False, "result retry enabled")


def _dependency_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "governing_policy": ROOT / config["governing_policy"]["path"],
        "stage0_runner": ROOT / config["canonical_builder"]["stage0_runner"]["path"],
        "stage0_config": ROOT / config["canonical_builder"]["stage0_config"]["path"],
        "lead_runner": ROOT / config["candidate_recipe"]["source_runner"]["path"],
        "lead_metrics": ROOT / config["candidate_recipe"]["source_metrics"]["path"],
        "surface_audit": ROOT / config["exposure_registry"]["prior_surface_audit"]["path"],
        "feature_columns": ROOT / config["paired_incumbent"]["feature_columns"]["path"],
        "single_model": ROOT / config["paired_incumbent"]["single_model"]["path"],
        "multi_model": ROOT / config["paired_incumbent"]["multi_model"]["path"],
        "router_model": ROOT / config["paired_incumbent"]["router_model"]["path"],
        "all20_anchor_map": ROOT / config["exposure_registry"]["all20_anchor_map"]["path"],
        "legacy_anchor_map": ROOT / config["exposure_registry"]["legacy_anchor_map"]["path"],
    }
    for index, item in enumerate(config["exposure_registry"]["surfaces"]):
        paths[f"exposure_surface_{index}"] = ROOT / item["path"]
    return paths


def _expected_dependency_hashes(config: dict[str, Any]) -> dict[str, str]:
    expected = {
        "governing_policy": config["governing_policy"]["sha256"],
        "stage0_runner": config["canonical_builder"]["stage0_runner"]["sha256"],
        "stage0_config": config["canonical_builder"]["stage0_config"]["sha256"],
        "lead_runner": config["candidate_recipe"]["source_runner"]["sha256"],
        "lead_metrics": config["candidate_recipe"]["source_metrics"]["sha256"],
        "surface_audit": config["exposure_registry"]["prior_surface_audit"]["sha256"],
        "feature_columns": config["paired_incumbent"]["feature_columns"]["sha256"],
        "single_model": config["paired_incumbent"]["single_model"]["sha256"],
        "multi_model": config["paired_incumbent"]["multi_model"]["sha256"],
        "router_model": config["paired_incumbent"]["router_model"]["sha256"],
        "all20_anchor_map": config["exposure_registry"]["all20_anchor_map"]["sha256"],
        "legacy_anchor_map": config["exposure_registry"]["legacy_anchor_map"]["sha256"],
    }
    for index, item in enumerate(config["exposure_registry"]["surfaces"]):
        expected[f"exposure_surface_{index}"] = item["sha256"]
    return expected


def verify_dependencies(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    paths = _dependency_paths(config)
    expected = _expected_dependency_hashes(config)
    receipt: dict[str, dict[str, Any]] = {}
    for role, path in paths.items():
        _require(path.is_file(), f"dependency missing: {role}")
        observed = sha256_file(path)
        _require(observed == expected[role], f"dependency SHA changed: {role}")
        receipt[role] = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": observed,
        }
    policy = json.loads(paths["governing_policy"].read_text(encoding="utf-8"))
    _require(
        policy["status"] == config["governing_policy"]["required_status"],
        "governing policy status changed",
    )
    return receipt


def contract_only(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    dependencies = verify_dependencies(config)
    return {
        "status": "CONTRACT_ONLY_PASS_NO_DATA_ACCESS",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(config_path),
        "dependency_count": int(len(dependencies)),
        "model_fit_count": 0,
        "official_or_test_rows_read": 0,
        "output_files_created": 0,
    }


def _git_state() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "head": head.stdout.strip() if head.returncode == 0 else "unknown",
        "dirty": bool(dirty),
        "dirty_entry_count": int(len(dirty)),
    }


def _environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "catboost", "joblib", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _load_exposure_times(
    config: dict[str, Any], stage0: Any, anchors: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    registry = config["exposure_registry"]
    all20 = pd.read_parquet(
        ROOT / registry["all20_anchor_map"]["path"],
        columns=["anchor_id", "station", "anchor_time"],
    )
    legacy = pd.read_parquet(
        ROOT / registry["legacy_anchor_map"]["path"],
        columns=["anchor_id", "station", "anchor_time"],
    )
    maps = {"all20": all20, "legacy": legacy}
    parts: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    for index, item in enumerate(registry["surfaces"]):
        keys = pd.read_parquet(
            ROOT / item["path"], columns=["anchor_id", "station"]
        ).drop_duplicates(["anchor_id", "station"])
        mapped = keys.merge(
            maps[item["anchor_map"]],
            on=["anchor_id", "station"],
            how="left",
            validate="one_to_one",
        )
        _require(mapped["anchor_time"].notna().all(), f"exposure time map failed: {index}")
        parts.append(mapped[["station", "anchor_time"]])
        counts[f"surface_{index}"] = int(len(mapped))

    stage0_config = stage0.load_config(
        ROOT / config["canonical_builder"]["stage0_config"]["path"]
    )
    matched = stage0.build_selection_matched_cohort(anchors, stage0_config)
    window_mask = pd.Series(False, index=matched.index)
    for window in stage0_config["forward_windows"]:
        start = pd.Timestamp(window["validation_start_utc"])
        end = pd.Timestamp(window["validation_end_utc"])
        window_mask |= matched["anchor_time"].ge(start) & matched["anchor_time"].lt(end)
    selection = stage0.select_station_global_independent(
        matched.loc[window_mask].copy(), gap_hours=78
    )
    _require(
        len(selection) == registry["selection_matched_validation_case_count"],
        "selection-matched exposed surface count changed",
    )
    parts.append(selection[["station", "anchor_time"]])
    counts["selection_matched"] = int(len(selection))
    union = pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)
    return union, {
        "source_surface_case_counts": counts,
        "time_union_count": int(len(union)),
        "time_union_sha256": frame_contract_sha256(union, ["station", "anchor_time"]),
        "columns_read_from_surface_artifacts": ["anchor_id", "station"],
        "columns_read_from_anchor_maps": ["anchor_id", "station", "anchor_time"],
        "target_value_columns_read": 0,
        "prediction_value_columns_read": 0,
    }


def _safe_features(
    grid: pd.DataFrame, fresh: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    safe_columns = [
        column for column in fresh.columns if not str(column).startswith("target_")
    ]
    safe = fresh.loc[:, safe_columns].copy()
    histories = extract_history_sequences(grid, safe)
    features = summarize_validation_histories(histories, safe)
    _require(set(feature_columns).issubset(features.columns), "frozen feature columns unavailable")
    _require(
        not any(str(column).startswith("target_") for column in features.columns),
        "target column entered prediction features",
    )
    return features


def _fresh_reference_and_candidate(
    config: dict[str, Any],
    features: pd.DataFrame,
    fresh: pd.DataFrame,
    lead_runner: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reference = config["paired_incumbent"]
    feature_columns = json.loads(
        (ROOT / reference["feature_columns"]["path"]).read_text(encoding="utf-8")
    )["columns"]
    _require(len(feature_columns) == reference["feature_columns"]["expected_count"], "feature count changed")
    safe_columns = [
        column for column in fresh.columns if not str(column).startswith("target_")
    ]
    safe = fresh.loc[:, safe_columns].copy()

    single = CatBoostRegressor()
    single.load_model(ROOT / reference["single_model"]["path"])
    multi = CatBoostRegressor()
    multi.load_model(ROOT / reference["multi_model"]["path"])
    router = joblib.load(ROOT / reference["router_model"]["path"])
    incumbent, reference_receipt = predict_frozen_reference(
        features,
        safe,
        feature_columns=feature_columns,
        single_predict=lambda frame: np.asarray(
            single.predict(
                frame, thread_count=int(config["execution_contract"]["cpu_threads"])
            ),
            dtype=float,
        ),
        multi_predict=lambda frame: np.asarray(
            multi.predict(
                frame, thread_count=int(config["execution_contract"]["cpu_threads"])
            ),
            dtype=float,
        ),
        router=router,
        shrink_weight=float(reference["long_lead_persistence_shrink_weight"]),
    )

    fit_started = time.perf_counter()
    history, history_audit = lead_runner._load_surface(ROOT)
    model = lead_runner._fit_model(history)
    fit_seconds = float(time.perf_counter() - fit_started)
    case_features = features.set_index("anchor_id")
    ids = safe["anchor_id"].to_numpy(dtype=np.int64)
    candidate_frame = pd.DataFrame(
        {
            "lead_h": np.tile(np.asarray(LEADS, dtype=int), len(safe)),
            "persistence": np.repeat(safe["current_hs"].to_numpy(dtype=float), len(LEADS)),
            "final_prediction": incumbent,
        }
    )
    for name in lead_runner.REGIME_FEATURES:
        candidate_frame[name] = np.repeat(
            case_features.loc[ids, name].to_numpy(dtype=float), len(LEADS)
        )
    candidate, prediction_audit = lead_runner._predict_model(model, candidate_frame)
    model_hash = array_sha256(
        model.medians,
        model.robust_scales,
        model.basis_scales,
        model.coefficients,
    )
    receipt = {
        "reference": reference_receipt,
        "candidate": {
            "fit_count": 1,
            "hyperparameter_search_count": 0,
            "history_cases": int(history[["anchor_id", "station"]].drop_duplicates().shape[0]),
            "history_rows": int(len(history)),
            "history_minimum_gap_h": history_audit["minimum_gap_h"],
            "fit_elapsed_seconds": fit_seconds,
            "model_state_sha256": model_hash,
            "prediction_audit": prediction_audit,
            "prediction_sha256": array_sha256(candidate),
            "fresh_target_values_used_for_fit_or_prediction": 0,
        },
    }
    return incumbent, candidate, receipt


def _implementation_hashes(config_path: Path) -> dict[str, str]:
    paths = {
        "config": config_path,
        "runner": Path(__file__).resolve(),
        "implementation": ROOT
        / "src/p3_wave/p3_lead_continuous_fresh_episode_confirmation_20260830_v3.py",
        "focused_tests": ROOT
        / "tests/test_p3_lead_continuous_fresh_episode_confirmation_20260830_v3.py",
        "qa_runner": ROOT
        / "scripts/qa_p3_lead_continuous_fresh_episode_confirmation_20260830_v3.py",
    }
    receipt: dict[str, str] = {}
    for name, path in paths.items():
        _require(path.is_file(), f"implementation file missing: {name}")
        receipt[name] = sha256_file(path)
    return receipt


def _consume_attempt(output: Path, config_path: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"exactly-once output already exists: {output.relative_to(ROOT)}")
    output.mkdir(parents=True, exist_ok=False)
    lock = {
        "schema_version": "p3.lead_continuous_fresh_episode_confirmation.attempt_lock.v3",
        "experiment_id": EXPERIMENT_ID,
        "status": "ATTEMPT_CONSUMED_NO_RETRY",
        "created_at_kst": _now_kst(),
        "config_sha256": sha256_file(config_path),
        "maximum_executions": 1,
        "result_based_retry": False,
        "technical_failure_auto_retry": False,
    }
    _write_exclusive_json(output / "ATTEMPT_LOCK.json", lock)
    return lock


def run_experiment(*, p3_dir: Path, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_path)
    output = ROOT / config["canonical_paths"]["output_dir"]
    dependencies = verify_dependencies(config)
    lock = _consume_attempt(output, config_path)
    target_evaluation_started_at: str | None = None
    try:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = str(config["execution_contract"]["cpu_threads"])
        stage0 = _load_script(
            "p3_stage0_fresh_v3",
            ROOT / config["canonical_builder"]["stage0_runner"]["path"],
        )
        lead_runner = _load_script(
            "p3_lead_continuous_source_v1",
            ROOT / config["candidate_recipe"]["source_runner"]["path"],
        )
        stage0_config = stage0.load_config(
            ROOT / config["canonical_builder"]["stage0_config"]["path"]
        )
        source_paths = stage0.resolve_train_only_source_paths(p3_dir)
        wave, atmos, source_receipt = stage0.load_train_only_sources(
            source_paths, stage0_config
        )
        for name, expected in config["expected_source_sha256"].items():
            _require(source_receipt[name]["sha256"] == expected, f"source SHA changed: {name}")
        grid, base_anchors = stage0.build_canonical_train_only_surface(wave, atmos)
        anchors, footprint = stage0.enrich_and_check_anchor_footprints(grid, base_anchors)
        _require(
            len(anchors) == config["canonical_builder"]["expected_canonical_anchor_count"],
            "canonical anchor count changed",
        )
        exposed, exposure_audit = _load_exposure_times(config, stage0, anchors)
        surface = config["fresh_surface"]
        fresh, freshness = select_fresh_surface(
            anchors,
            exposed,
            start=pd.Timestamp(surface["start_utc_inclusive"]),
            end=pd.Timestamp(surface["end_utc_exclusive"]),
            separation_hours=float(surface["same_station_exposure_separation_hours"]),
        )
        _require(
            freshness["window_anchor_count"] == surface["expected_canonical_window_anchor_count"],
            "post-H1 window count changed",
        )
        _require(
            freshness["fresh_dense_count"] == surface["expected_fresh_dense_count"],
            "fresh dense count changed",
        )
        _require(
            len(fresh) == surface["expected_fresh_independent_case_count"],
            "fresh independent count changed",
        )
        _require(freshness["fresh_by_station"] == surface["expected_fresh_by_station"], "fresh station footprint changed")
        fresh_surface_sha256 = frame_contract_sha256(
            fresh, ["anchor_id", "station", "anchor_time"]
        )

        feature_columns = json.loads(
            (ROOT / config["paired_incumbent"]["feature_columns"]["path"]).read_text(
                encoding="utf-8"
            )
        )["columns"]
        features = _safe_features(grid, fresh, feature_columns)
        incumbent, candidate, fit_receipt = _fresh_reference_and_candidate(
            config, features, fresh, lead_runner
        )
        blind_created_at = _now_kst()
        blind_seal = {
            "schema_version": "p3.lead_continuous_fresh_episode_confirmation.blind_seal.v3",
            "experiment_id": EXPERIMENT_ID,
            "created_at_kst": blind_created_at,
            "fresh_surface_sha256": fresh_surface_sha256,
            "case_count": int(len(fresh)),
            "row_count": int(len(candidate)),
            "leads_h": list(LEADS),
            "incumbent_prediction_sha256": array_sha256(incumbent),
            "candidate_prediction_sha256": array_sha256(candidate),
            "joint_prediction_sha256": array_sha256(incumbent, candidate),
            "raw_prediction_rows_persisted": 0,
            "fresh_target_values_used_before_seal": 0,
        }
        blind_file_sha256 = _write_exclusive_json(
            output / "blind_prediction_seal.json", blind_seal
        )

        target_evaluation_started_at = _now_kst()
        comparison = build_comparison_frame(fresh, incumbent, candidate)
        metrics = comparison_metrics(comparison)
        uncertainty = uncertainty_or_insufficient(
            comparison,
            minimum_blocks=int(config["uncertainty"]["minimum_independent_blocks_for_interval"]),
        )
        implementation = _implementation_hashes(config_path)
        integrity = {
            "governing_policy_and_all_dependency_hashes_exact": bool(dependencies),
            "source_allowlist_and_hashes_exact": set(source_receipt)
            == {"README.md", "train_wave.csv", "train_atmos.csv"},
            "canonical_anchor_count_and_footprints_exact": bool(
                len(anchors) == 8121
                and footprint["official_six_targets_match_grid_and_are_finite"]
                and footprint["history_48h_elapsed_before_every_anchor"]
            ),
            "fresh_surface_metadata_only_and_expected_one_case": bool(
                len(fresh) == 1
                and freshness["target_value_columns_used_for_selection"] == 0
                and freshness["prediction_value_columns_used_for_selection"] == 0
                and freshness["minimum_gap_to_exposed_h"] >= 78.0
            ),
            "blind_prediction_seal_precedes_target_evaluation": bool(
                blind_created_at < target_evaluation_started_at
                and (output / "blind_prediction_seal.json").is_file()
            ),
            "six_leads_intact_and_paired_rows_exact": bool(
                len(comparison) == 6
                and tuple(comparison.sort_values("lead_h")["lead_h"].astype(int)) == LEADS
            ),
            "finite_prediction_domain_0_to_30m": bool(
                np.isfinite(incumbent).all()
                and np.isfinite(candidate).all()
                and np.all((incumbent >= 0.0) & (incumbent <= 30.0))
                and np.all((candidate >= 0.0) & (candidate <= 30.0))
            ),
            "exact_frozen_recipe_one_fit_zero_hpo": bool(
                fit_receipt["candidate"]["fit_count"] == 1
                and fit_receipt["candidate"]["hyperparameter_search_count"] == 0
                and fit_receipt["reference"]["catboost_fit_count"] == 0
                and fit_receipt["reference"]["router_fit_count"] == 0
            ),
            "no_official_submission_or_raw_prediction_output": True,
            "exactly_once_attempt_lock_present": bool(
                lock["status"] == "ATTEMPT_CONSUMED_NO_RETRY"
                and (output / "ATTEMPT_LOCK.json").is_file()
            ),
        }
        terminal = classify_terminal(
            integrity_checks=integrity, uncertainty=uncertainty
        )
        payload: dict[str, Any] = {
            "schema_version": "p3.lead_continuous_fresh_episode_confirmation.result.20260830.v3",
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_EXACTLY_ONCE_COMPLETE",
            "created_at_kst": _now_kst(),
            "terminal_evidence_state": terminal,
            "submission_readiness": "NOT_READY_INSUFFICIENT_FRESH_SUPPORT",
            "official_action_authorized": False,
            "primary": metrics["overall"],
            "uncertainty": uncertainty,
            "diagnostics": {
                "by_station": metrics["by_station"],
                "by_lead": metrics["by_lead"],
                "diagnostic_only_not_vetoes": True,
            },
            "freshness": {
                **freshness,
                "surface_sha256": fresh_surface_sha256,
                "exposure_audit": exposure_audit,
                "historical_three_window_fresh_cases": 0,
                "post_h1_fresh_cases": int(len(fresh)),
                "claim_limit": surface["claim_limit"],
            },
            "fit_and_prediction": fit_receipt,
            "blind_seal": {
                "path": config["canonical_paths"]["blind_seal"],
                "file_sha256": blind_file_sha256,
                "created_at_kst": blind_created_at,
                "target_evaluation_started_at_kst": target_evaluation_started_at,
                "joint_prediction_sha256": blind_seal["joint_prediction_sha256"],
            },
            "integrity_checks": integrity,
            "execution": {
                "candidate_fit_count": 1,
                "catboost_fit_count": 0,
                "router_fit_count": 0,
                "hyperparameter_search_count": 0,
                "bootstrap_replicates_executed": uncertainty["bootstrap_replicates_executed"],
                "runtime_seconds": float(time.perf_counter() - started),
                "technical_failure_auto_retry": False,
                "result_based_retry_or_tuning": False,
                "outlier_hard_deletion_count": 0,
                "csv_output_count": 0,
                "upload_count": 0,
            },
            "data_access": {
                "opened_source_basenames": ["README.md", "train_wave.csv", "train_atmos.csv"],
                "official_test_context_index_sample_baseline_score_submission_hidden_rows_read": 0,
                "raw_prediction_rows_persisted": 0,
                "source_rows_modified_or_deleted": 0,
            },
            "provenance": {
                "source": source_receipt,
                "dependencies": dependencies,
                "implementation_sha256": implementation,
                "config_sha256": sha256_file(config_path),
                "git": _git_state(),
                "environment": _environment(),
            },
        }
        payload["seal"] = {
            "algorithm": "sha256",
            "payload_without_seal_sha256": _payload_sha256(payload),
        }
        result_sha256 = _write_exclusive_json(output / "result.json", payload)
        return {
            "status": payload["status"],
            "terminal_evidence_state": terminal,
            "result": config["canonical_paths"]["result"],
            "result_sha256": result_sha256,
            "runtime_seconds": payload["execution"]["runtime_seconds"],
            "candidate_fit_count": 1,
            "fresh_cases": int(len(fresh)),
            "benefit_m": metrics["overall"]["benefit_incumbent_minus_candidate_rmse_m"],
        }
    except Exception as error:
        failure = {
            "schema_version": "p3.lead_continuous_fresh_episode_confirmation.failure.v3",
            "experiment_id": EXPERIMENT_ID,
            "status": "INVALID_TERMINAL_TECHNICAL_FAILURE_NO_RETRY",
            "created_at_kst": _now_kst(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "target_evaluation_started_at_kst": target_evaluation_started_at,
            "traceback": traceback.format_exc(),
            "retry_allowed": False,
        }
        try:
            _write_exclusive_json(output / "FAILED.json", failure)
        finally:
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p3-dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--contract-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.contract_only:
        print(json.dumps(contract_only(args.config), ensure_ascii=False, sort_keys=True))
        return 0
    if args.p3_dir is None:
        raise SystemExit("--p3-dir is required for the exactly-once run")
    result = run_experiment(p3_dir=args.p3_dir, config_path=args.config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
