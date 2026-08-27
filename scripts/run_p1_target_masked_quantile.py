"""Run the preregistered P1 target-masked quantile one-shot experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from p1_qc.config import P1QCConfig, load_config  # noqa: E402
from p1_qc.data import load_train_test  # noqa: E402
from p1_qc.experiment import (  # noqa: E402
    environment_summary,
    git_sha,
    sha256_file,
    write_json,
)
from p1_qc.metrics import evaluate_predictions, group_row_shares  # noqa: E402
from p1_qc.pipeline import (  # noqa: E402
    TabularEncoder,
    _best_iteration,
    _fit_model,
    _inner_calibration_indices,
    _iteration_parameter,
    _model_parameters,
    _threads,
    apply_postprocess,
    load_or_build_features,
    resolve_data_dir,
    tune_postprocess,
)
from p1_qc.rules import detect_plateaus, detect_singleton_spikes  # noqa: E402
from p1_qc.splits import outer_folds  # noqa: E402
from p1_qc.target_masked_quantile import (  # noqa: E402
    QUANTILE_SCORE_COLUMNS,
    QuantileModelConfig,
    append_score_matrix,
    build_quantile_scores,
    build_target_masked_design,
    cross_fitted_quantiles,
    design_contract_hash,
    fit_predict_quantiles,
    place_quantiles,
    synthetic_offset_smoke,
)
from p1_qc.validation import (  # noqa: E402
    normal_station_layer_day_fp,
    paired_block_bootstrap,
)

EXPERIMENT_ID = "p1_target_masked_quantile_v1"
KEY_COLUMNS = ("station", "year", "layer", "time")
BACKEND = "xgboost"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiments/p1_target_masked_quantile_v1.json"),
    )
    parser.add_argument("--p1-config", type=Path, default=Path("configs/p1.toml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _relative_increase(candidate: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else None
    return (candidate - baseline) / baseline


class StatusWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started_wall = datetime.now().astimezone()
        self.started_monotonic = time.monotonic()

    def update(
        self,
        progress: float,
        phase: str,
        detail: str,
        *,
        status: str = "running",
    ) -> None:
        progress = float(np.clip(progress, 0.0, 100.0))
        elapsed = time.monotonic() - self.started_monotonic
        if status in {"complete", "failed"}:
            eta = "완료" if status == "complete" else "중단"
        elif progress > 0.5:
            total = elapsed / (progress / 100.0)
            eta_time = datetime.now().astimezone() + timedelta(seconds=max(0.0, total - elapsed))
            eta = eta_time.strftime("%Y-%m-%d %H:%M:%S KST")
        else:
            eta = "계산 중"
        write_json(
            self.path,
            {
                "title": "P1 target-masked quantile one-shot",
                "status": status,
                "progress": round(progress, 2),
                "phase": phase,
                "detail": detail,
                "eta": eta,
                "started_at": self.started_wall.isoformat(),
                "elapsed_seconds": round(elapsed, 3),
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )


def _quantile_config(contract: Mapping[str, Any]) -> QuantileModelConfig:
    section = contract["quantile_model"]
    if not isinstance(section, Mapping):
        raise TypeError("quantile_model must be a mapping")
    return QuantileModelConfig(
        alphas=tuple(float(value) for value in section["alphas"]),  # type: ignore[arg-type]
        n_estimators=int(section["n_estimators"]),
        learning_rate=float(section["learning_rate"]),
        num_leaves=int(section["num_leaves"]),
        min_child_samples=int(section["min_child_samples"]),
        reg_alpha=float(section["reg_alpha"]),
        reg_lambda=float(section["reg_lambda"]),
        crossfit_folds=int(section["crossfit_folds"]),
        threads=int(section["threads"]),
        seed=int(section["seed"]),
    )


def _validate_contract(
    contract: Mapping[str, Any],
    experiment_config: Path,
    p1_config: Path,
) -> None:
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected experiment_id")
    authorization = contract.get("authorization")
    if not isinstance(authorization, Mapping) or any(
        bool(value) for value in authorization.values()
    ):
        raise ValueError("all mutation/upload/external authorization flags must be false")
    reference = contract["reference"]
    if not isinstance(reference, Mapping):
        raise TypeError("reference must be a mapping")
    if sha256_file(p1_config) != str(reference["p1_config_sha256"]):
        raise RuntimeError("configs/p1.toml differs from the frozen reference hash")
    reference_oof = PROJECT_ROOT / str(reference["oof_relative_path"])
    if sha256_file(reference_oof) != str(reference["oof_sha256"]):
        raise RuntimeError("frozen reference OOF hash mismatch")
    candidate = contract["xgboost_candidate"]
    if not isinstance(candidate, Mapping):
        raise TypeError("xgboost_candidate must be a mapping")
    if tuple(candidate["added_score_columns"]) != QUANTILE_SCORE_COLUMNS:
        raise RuntimeError("experiment score columns differ from the code contract")
    if bool(candidate["augmentation"]) or bool(candidate["hyperparameter_search"]):
        raise RuntimeError("augmentation and hyperparameter search must remain disabled")
    if not experiment_config.is_file():
        raise FileNotFoundError(experiment_config)


def _git_dirty_summary() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return {"available": False}
    lines = [line for line in output.splitlines() if line]
    return {
        "available": True,
        "dirty": bool(lines),
        "changed_path_count": len(lines),
    }


def _score_crossfit_scope(
    train: pd.DataFrame,
    design: Any,
    target_temp: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    quantile_config: QuantileModelConfig,
    *,
    seed_offset: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    full_quantiles, audit = cross_fitted_quantiles(
        train,
        design,
        target_temp,
        labels,
        indices,
        config=quantile_config,
        seed_offset=seed_offset,
    )
    local_frame = train.iloc[indices].copy()
    scores = build_quantile_scores(
        local_frame,
        full_quantiles[indices],
        cadence_minutes=10,
    )
    return scores, audit


def _score_holdout(
    train: pd.DataFrame,
    design: Any,
    target_temp: np.ndarray,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    predict_indices: np.ndarray,
    quantile_config: QuantileModelConfig,
    *,
    seed_offset: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction, audit = fit_predict_quantiles(
        design,
        target_temp,
        labels,
        fit_indices,
        predict_indices,
        config=quantile_config,
        seed_offset=seed_offset,
    )
    local_frame = train.iloc[predict_indices].copy()
    aligned = place_quantiles(len(local_frame), np.arange(len(local_frame)), prediction)
    scores = build_quantile_scores(local_frame, aligned, cadence_minutes=10)
    return scores, audit


def _candidate_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    p1_config: P1QCConfig,
    quantile_config: QuantileModelConfig,
    status: StatusWriter,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Complete every outer prediction before the reference OOF labels are read."""

    status.update(10.0, "base_features", "기존 frozen offline feature cache 확인")
    bundle = load_or_build_features(train, p1_config, kind="train", use_cache=True)
    status.update(17.0, "target_masked_design", "자기 수온 차단 조건부 입력 구성")
    design = build_target_masked_design(train)
    folds = outer_folds(
        train,
        config=p1_config.splits,
        cadence_minutes=p1_config.data.cadence_minutes,
        group_columns=p1_config.data.group_columns,
    )
    target_temp = pd.to_numeric(train["temp"], errors="coerce").to_numpy(dtype=float)
    labels = train["label"].to_numpy(dtype=np.int8)
    parameters = _model_parameters(p1_config, BACKEND)
    iteration_key = _iteration_parameter(BACKEND)
    configured_iterations = int(parameters.get(iteration_key, 700))
    validation = p1_config.raw.get("validation", {})
    calibration_days = (
        int(validation.get("calibration_days", 60)) if isinstance(validation, Mapping) else 60
    )
    parts: list[pd.DataFrame] = []
    fold_audits: list[dict[str, Any]] = []

    for fold_number, fold in enumerate(folds):
        fold_base = 20.0 + fold_number * 23.0
        status.update(
            fold_base,
            f"{fold.name}:inner_quantiles",
            "inner-fit 정상행 교차적합 q05/q50/q95",
        )
        inner_fit, calibration = _inner_calibration_indices(
            train,
            fold,
            calibration_days=calibration_days,
            purge_days=p1_config.splits.purge_days,
        )
        inner_scores, inner_crossfit_audit = _score_crossfit_scope(
            train,
            design,
            target_temp,
            labels,
            inner_fit,
            quantile_config,
            seed_offset=fold_number * 1000,
        )
        calibration_scores, calibration_quantile_audit = _score_holdout(
            train,
            design,
            target_temp,
            labels,
            inner_fit,
            calibration,
            quantile_config,
            seed_offset=fold_number * 1000 + 100,
        )
        status.update(
            fold_base + 6.0,
            f"{fold.name}:inner_xgb",
            "기존 XGBoost에 고정 quantile score 추가·inner nuisance 선택",
        )
        inner_encoder = TabularEncoder().fit(bundle, inner_fit)
        inner_matrix = append_score_matrix(inner_encoder.transform(bundle, inner_fit), inner_scores)
        calibration_matrix = append_score_matrix(
            inner_encoder.transform(bundle, calibration), calibration_scores
        )
        inner_target = labels[inner_fit]
        calibration_target = labels[calibration]
        selection_model = _fit_model(
            BACKEND,
            parameters,
            p1_config.seed + fold_number,
            _threads(p1_config),
            inner_matrix,
            inner_target,
            evaluation=(calibration_matrix, calibration_target),
        )
        best_iterations = _best_iteration(selection_model, configured_iterations)
        calibration_probability = selection_model.predict_proba(calibration_matrix)[:, 1]
        calibration_frame = train.iloc[calibration].copy()
        selected_postprocess, _, inner_postprocess_audit = tune_postprocess(
            calibration_frame,
            calibration_probability,
            calibration_target,
            detect_plateaus(calibration_frame).to_numpy(),
            detect_singleton_spikes(calibration_frame).to_numpy(),
            p1_config,
        )

        status.update(
            fold_base + 11.0,
            f"{fold.name}:outer_train_quantiles",
            "outer-train 정상행 교차적합 quantile score 구성",
        )
        outer_scores, outer_crossfit_audit = _score_crossfit_scope(
            train,
            design,
            target_temp,
            labels,
            fold.train_idx,
            quantile_config,
            seed_offset=fold_number * 1000 + 200,
        )
        validation_scores, validation_quantile_audit = _score_holdout(
            train,
            design,
            target_temp,
            labels,
            fold.train_idx,
            fold.val_idx,
            quantile_config,
            seed_offset=fold_number * 1000 + 300,
        )
        status.update(
            fold_base + 17.0,
            f"{fold.name}:outer_xgb",
            "outer XGBoost 학습 및 label-blind validation 예측",
        )
        encoder = TabularEncoder().fit(bundle, fold.train_idx)
        outer_train_matrix = append_score_matrix(
            encoder.transform(bundle, fold.train_idx), outer_scores
        )
        validation_matrix = append_score_matrix(
            encoder.transform(bundle, fold.val_idx), validation_scores
        )
        outer_parameters = dict(parameters)
        outer_parameters[iteration_key] = best_iterations
        model = _fit_model(
            BACKEND,
            outer_parameters,
            p1_config.seed + fold_number,
            _threads(p1_config),
            outer_train_matrix,
            labels[fold.train_idx],
        )
        probability = model.predict_proba(validation_matrix)[:, 1]
        validation_frame = train.iloc[fold.val_idx].copy()
        plateau = detect_plateaus(validation_frame).to_numpy()
        spike = detect_singleton_spikes(validation_frame).to_numpy()
        prediction = apply_postprocess(
            validation_frame,
            probability,
            plateau,
            spike,
            selected_postprocess,
        )
        output = validation_frame.loc[:, KEY_COLUMNS].copy()
        output["fold"] = fold.name
        output["candidate_probability"] = probability.astype(np.float32)
        output["candidate_prediction"] = prediction.astype(np.int8)
        output["plateau"] = plateau.astype(bool)
        output["spike_candidate"] = spike.astype(bool)
        for column in QUANTILE_SCORE_COLUMNS:
            output[column] = pd.to_numeric(validation_scores[column], errors="coerce").to_numpy(
                dtype=np.float32
            )
        parts.append(output)
        fold_audits.append(
            {
                "fold": fold.name,
                "train_rows": int(len(fold.train_idx)),
                "validation_rows": int(len(fold.val_idx)),
                "inner_fit_rows": int(len(inner_fit)),
                "inner_calibration_rows": int(len(calibration)),
                "best_iterations": int(best_iterations),
                "selected_postprocess": selected_postprocess,
                "inner_postprocess_audit": inner_postprocess_audit,
                "inner_crossfit": inner_crossfit_audit,
                "inner_holdout_quantiles": calibration_quantile_audit,
                "outer_crossfit": outer_crossfit_audit,
                "outer_holdout_quantiles": validation_quantile_audit,
            }
        )
        status.update(
            fold_base + 22.0,
            f"{fold.name}:complete",
            f"{fold_number + 1}/3 outer prediction 완료; outer truth 미열람",
        )

    predictions = pd.concat(parts, ignore_index=True)
    if len(predictions) != sum(len(fold.val_idx) for fold in folds):
        raise RuntimeError("candidate outer prediction row count mismatch")
    if predictions.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError("candidate outer predictions contain duplicate keys")
    return predictions, {
        "folds": fold_audits,
        "all_outer_predictions_completed_before_reference_label_access": True,
        "outer_labels_used_for_model_or_postprocess_selection": False,
        "test_rows_used_only_for_station_layer_share_weights": len(test),
        "conditional_design_contract_hash": design_contract_hash(),
        "conditional_design_columns": list(design.feature_columns),
        "conditional_model_reads_own_temp": False,
        "conditional_model_reads_depth": False,
    }


def _evaluate(
    predictions: pd.DataFrame,
    reference_oof: Path,
    test: pd.DataFrame,
    contract: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference = pd.read_parquet(reference_oof)
    required = {*KEY_COLUMNS, "fold", "label", "anomaly_type", "prediction", "probability"}
    missing = sorted(required.difference(reference.columns))
    if missing:
        raise RuntimeError(f"reference OOF is missing columns: {missing}")
    baseline = reference.loc[
        :,
        [*KEY_COLUMNS, "fold", "label", "anomaly_type", "prediction", "probability"],
    ].rename(
        columns={
            "prediction": "baseline_prediction",
            "probability": "baseline_probability",
        }
    )
    complete = predictions.merge(
        baseline,
        on=[*KEY_COLUMNS, "fold"],
        how="inner",
        validate="one_to_one",
    )
    if len(complete) != len(reference) or len(complete) != len(predictions):
        raise RuntimeError("candidate/reference OOF keys do not align exactly")
    truth = complete["label"].to_numpy(dtype=np.int8)
    candidate = complete["candidate_prediction"].to_numpy(dtype=np.int8)
    incumbent = complete["baseline_prediction"].to_numpy(dtype=np.int8)
    test_shares = group_row_shares(test)
    candidate_report = evaluate_predictions(
        truth,
        candidate,
        complete,
        group_weights=test_shares,
        anomaly_type=complete["anomaly_type"],
    )
    baseline_report = evaluate_predictions(
        truth,
        incumbent,
        complete,
        group_weights=test_shares,
        anomaly_type=complete["anomaly_type"],
    )
    expected = float(contract["reference"]["outer_test_share_weighted_f1"])
    if abs(baseline_report.weighted.f1 - expected) > 1.0e-12:
        raise RuntimeError("frozen baseline weighted F1 did not reproduce")

    bootstrap_section = contract["outer_evaluation"]
    bootstrap = paired_block_bootstrap(
        truth,
        candidate,
        incumbent,
        complete,
        replicates=int(bootstrap_section["bootstrap_replicates"]),
        seed=int(bootstrap_section["bootstrap_seed"]),
        normal_day_timezone="Asia/Seoul",
    )
    normal_fp = normal_station_layer_day_fp(truth, candidate, incumbent, complete)
    candidate_fp = float(normal_fp["candidate"]["false_positive_rows_per_normal_station_layer_day"])
    baseline_fp = float(normal_fp["baseline"]["false_positive_rows_per_normal_station_layer_day"])
    fp_relative = _relative_increase(candidate_fp, baseline_fp)

    candidate_groups = candidate_report.groups.set_index(["station", "layer"])
    baseline_groups = baseline_report.groups.set_index(["station", "layer"])
    group_rows: list[dict[str, Any]] = []
    for key in candidate_groups.index.union(baseline_groups.index):
        candidate_f1 = float(candidate_groups.loc[key, "f1"])
        baseline_f1 = float(baseline_groups.loc[key, "f1"])
        group_rows.append(
            {
                "station": str(key[0]),
                "layer": int(key[1]),
                "candidate_f1": candidate_f1,
                "baseline_f1": baseline_f1,
                "candidate_minus_baseline_f1": candidate_f1 - baseline_f1,
            }
        )
    worst_group_delta = min((row["candidate_minus_baseline_f1"] for row in group_rows), default=0.0)

    fold_rows: list[dict[str, Any]] = []
    for fold_name, part in complete.groupby("fold", sort=False, observed=True):
        fold_truth = part["label"].to_numpy(dtype=np.int8)
        fold_candidate = evaluate_predictions(
            fold_truth,
            part["candidate_prediction"].to_numpy(dtype=np.int8),
            part,
            group_weights=test_shares,
            anomaly_type=part["anomaly_type"],
        )
        fold_baseline = evaluate_predictions(
            fold_truth,
            part["baseline_prediction"].to_numpy(dtype=np.int8),
            part,
            group_weights=test_shares,
            anomaly_type=part["anomaly_type"],
        )
        fold_rows.append(
            {
                "fold": str(fold_name),
                "candidate_weighted_f1": fold_candidate.weighted.f1,
                "baseline_weighted_f1": fold_baseline.weighted.f1,
                "candidate_minus_baseline_weighted_f1": (
                    fold_candidate.weighted.f1 - fold_baseline.weighted.f1
                ),
            }
        )

    gate_contract = bootstrap_section["promotion_gate"]
    weighted_delta = candidate_report.weighted.f1 - baseline_report.weighted.f1
    fp_pass = (
        candidate_fp == 0.0
        if baseline_fp == 0.0
        else bool(
            fp_relative is not None
            and fp_relative < float(gate_contract["normal_fp_day_relative_increase_lt"])
        )
    )
    gates = {
        "weighted_f1_delta": float(weighted_delta),
        "weighted_f1_pass": bool(
            weighted_delta >= float(gate_contract["test_share_weighted_f1_delta_min"])
        ),
        "bootstrap_ci90_lower": float(bootstrap["difference_ci90"][0]),
        "bootstrap_pass": bool(
            bootstrap["difference_ci90"][0]
            > float(gate_contract["paired_bootstrap_90pct_lower_bound_gt"])
        ),
        "normal_fp_day_relative_increase": fp_relative,
        "normal_fp_day_pass": fp_pass,
        "worst_station_layer_f1_delta": float(worst_group_delta),
        "worst_station_layer_pass": bool(
            worst_group_delta >= float(gate_contract["station_layer_f1_delta_min"])
        ),
    }
    gates["promotion_passed"] = bool(
        gates["weighted_f1_pass"]
        and gates["bootstrap_pass"]
        and gates["normal_fp_day_pass"]
        and gates["worst_station_layer_pass"]
    )
    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "candidate": candidate_report.to_dict(),
        "frozen_baseline": baseline_report.to_dict(),
        "paired_block_bootstrap": bootstrap,
        "normal_station_layer_day_fp": normal_fp,
        "station_layer_comparison": group_rows,
        "fold_comparison": fold_rows,
        "gates": gates,
        "decision": "promoted" if gates["promotion_passed"] else "failed_gate",
        "outer_is_independent_holdout": False,
        "official_hidden_test_used": False,
    }
    return complete, metrics


def _create_exposure_lock(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_config = (PROJECT_ROOT / args.experiment_config).resolve(strict=True)
    p1_config_path = (PROJECT_ROOT / args.p1_config).resolve(strict=True)
    contract = _load_json(experiment_config)
    _validate_contract(contract, experiment_config, p1_config_path)
    quantile_config = _quantile_config(contract)
    if args.smoke_only:
        report = synthetic_offset_smoke(
            QuantileModelConfig(
                n_estimators=30,
                min_child_samples=8,
                crossfit_folds=2,
                threads=1,
                seed=quantile_config.seed,
            )
        )
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    artifact_dir = PROJECT_ROOT / str(contract["artifacts"]["output_relative_dir"])
    status_path = PROJECT_ROOT / str(contract["artifacts"]["status_relative_path"])
    lock_path = PROJECT_ROOT / str(contract["artifacts"]["outer_lock_relative_path"])
    if artifact_dir.exists() or lock_path.exists():
        raise FileExistsError(
            "one-shot artifact/lock already exists; refusing to repeat the outer experiment"
        )
    artifact_dir.mkdir(parents=True, exist_ok=False)
    status = StatusWriter(status_path)
    started = datetime.now().astimezone()
    manifest: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "status": "running",
        "started_at": started.isoformat(),
        "experiment_config": {
            "relative_path": str(experiment_config.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(experiment_config),
        },
        "p1_config": {
            "relative_path": str(p1_config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(p1_config_path),
        },
        "git_sha": git_sha(),
        "git_worktree": _git_dirty_summary(),
        "environment": environment_summary(),
        "authorization": contract["authorization"],
        "submission_created": False,
        "external_observation_values_used": False,
    }
    write_json(artifact_dir / "manifest.json", manifest)
    try:
        status.update(1.0, "smoke", "누출 계약 및 합성 offset smoke 실행")
        smoke = synthetic_offset_smoke(
            QuantileModelConfig(
                n_estimators=30,
                min_child_samples=8,
                crossfit_folds=2,
                threads=1,
                seed=quantile_config.seed,
            )
        )
        write_json(artifact_dir / "smoke.json", smoke)
        status.update(4.0, "load", "원본 SHA 확인 후 train/test 읽기")
        p1_config = load_config(p1_config_path, env={})
        data_dir = resolve_data_dir(p1_config, args.data_dir)
        train, test = load_train_test(data_dir, audit=True, strict=True)
        expected_data = contract["data"]
        if train.attrs.get("source_sha256") != expected_data["train_sha256"]:
            raise RuntimeError("train.csv hash differs from the preregistration")
        if test.attrs.get("source_sha256") != expected_data["test_sha256"]:
            raise RuntimeError("test.csv hash differs from the preregistration")
        manifest["inputs"] = {
            "train": {
                "bytes": int((data_dir / "train.csv").stat().st_size),
                "sha256": train.attrs["source_sha256"],
            },
            "test": {
                "bytes": int((data_dir / "test.csv").stat().st_size),
                "sha256": test.attrs["source_sha256"],
            },
            "reference_oof": {
                "sha256": contract["reference"]["oof_sha256"],
            },
        }
        write_json(artifact_dir / "manifest.json", manifest)

        predictions, prediction_audit = _candidate_predictions(
            train,
            test,
            p1_config,
            quantile_config,
            status,
        )
        status.update(89.0, "persist_predictions", "label-free outer 예측 저장 및 SHA 고정")
        prediction_path = artifact_dir / "oof_predictions_label_blind.parquet"
        predictions.to_parquet(prediction_path, index=False, compression="zstd")
        write_json(artifact_dir / "prediction_audit.json", prediction_audit)
        prediction_sha = sha256_file(prediction_path)
        write_json(
            artifact_dir / "predictions_complete.json",
            {
                "rows": len(predictions),
                "sha256": prediction_sha,
                "outer_labels_accessed": False,
                "completed_at": datetime.now().astimezone().isoformat(),
            },
        )
        _create_exposure_lock(
            lock_path,
            {
                "experiment_id": EXPERIMENT_ID,
                "event": "outer_evaluation_started",
                "prediction_sha256": prediction_sha,
                "experiment_config_sha256": sha256_file(experiment_config),
                "created_at": datetime.now().astimezone().isoformat(),
                "outer_evaluation_count": 1,
            },
        )

        status.update(93.0, "outer_evaluation", "고정 예측에 outer truth를 1회 결합·gate 계산")
        reference_path = PROJECT_ROOT / str(contract["reference"]["oof_relative_path"])
        complete, metrics = _evaluate(predictions, reference_path, test, contract)
        evaluated_path = artifact_dir / "oof_evaluated.parquet"
        complete.to_parquet(evaluated_path, index=False, compression="zstd")
        metrics_path = write_json(artifact_dir / "metrics.json", metrics)
        if sha256_file(reference_path) != contract["reference"]["oof_sha256"]:
            raise RuntimeError("frozen reference OOF changed during the experiment")

        finished = datetime.now().astimezone()
        manifest.update(
            {
                "status": "complete",
                "finished_at": finished.isoformat(),
                "elapsed_seconds": (finished - started).total_seconds(),
                "decision": metrics["decision"],
                "gates": metrics["gates"],
                "artifacts": {
                    "smoke.json": sha256_file(artifact_dir / "smoke.json"),
                    "prediction_audit.json": sha256_file(artifact_dir / "prediction_audit.json"),
                    "oof_predictions_label_blind.parquet": prediction_sha,
                    "oof_evaluated.parquet": sha256_file(evaluated_path),
                    "metrics.json": sha256_file(metrics_path),
                    "outer_exposure.lock": sha256_file(lock_path),
                },
                "code_sha256": {
                    "src/p1_qc/target_masked_quantile.py": sha256_file(
                        PROJECT_ROOT / "src/p1_qc/target_masked_quantile.py"
                    ),
                    "scripts/run_p1_target_masked_quantile.py": sha256_file(
                        PROJECT_ROOT / "scripts/run_p1_target_masked_quantile.py"
                    ),
                    "tests/test_target_masked_quantile.py": sha256_file(
                        PROJECT_ROOT / "tests/test_target_masked_quantile.py"
                    ),
                },
                "frozen_reference_oof_sha256_after": sha256_file(reference_path),
            }
        )
        write_json(artifact_dir / "manifest.json", manifest)
        gates = metrics["gates"]
        status.update(
            100.0,
            "complete",
            (
                f"{metrics['decision']} · weighted Δ {gates['weighted_f1_delta']:+.6f} · "
                f"CI90 lower {gates['bootstrap_ci90_lower']:+.6f} · 업로드 없음"
            ),
            status="complete",
        )
        print(json.dumps({"decision": metrics["decision"], "gates": gates}, indent=2))
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now().astimezone().isoformat()
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        write_json(artifact_dir / "manifest.json", manifest)
        status.update(100.0, "failed", f"{type(exc).__name__}: {exc}", status="failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
