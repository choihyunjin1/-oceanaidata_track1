"""Run the preregistered P1 dual-flank block-inpainting one-shot experiment.

The historical gate is evaluated first.  The frozen outer OOF labels are only
opened if every historical promotion condition passes; otherwise this model
family is closed with ``outer_evaluation_count == 0``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from p1_qc.block_inpaint import (  # noqa: E402
    SCORE_COLUMNS,
    BlockInpaintConfig,
    apply_additive_gate,
    assert_mask_invariance,
    assert_target_safe_contract,
    build_safe_design,
    contract_hash,
    coverage_audit,
    enumerate_blocks,
    fit_additive_gate,
    fit_covariate_scaler,
    prepare_series,
    score_inpaint_model,
    train_inpaint_model,
)
from p1_qc.config import P1QCConfig, load_config  # noqa: E402
from p1_qc.data import load_train_test  # noqa: E402
from p1_qc.experiment import environment_summary, git_sha, sha256_file, write_json  # noqa: E402
from p1_qc.metrics import evaluate_predictions, group_row_shares  # noqa: E402
from p1_qc.pipeline import (  # noqa: E402
    TabularEncoder,
    _fit_model,
    _inner_calibration_indices,
    _model_parameters,
    _threads,
    apply_postprocess,
    load_or_build_features,
    resolve_data_dir,
)
from p1_qc.rules import detect_plateaus, detect_singleton_spikes  # noqa: E402
from p1_qc.splits import outer_folds  # noqa: E402
from p1_qc.validation import normal_station_layer_day_fp, paired_block_bootstrap  # noqa: E402

EXPERIMENT_ID = "p1_block_inpaint_v1"
KEY_COLUMNS = ("station", "year", "layer", "time")
BACKEND = "xgboost"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiments/p1_block_inpaint_v1.json"),
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


class StatusWriter:
    """Atomically refresh the local gauge using real KST wall time."""

    def __init__(self, path: Path) -> None:
        self.path = path
        existing: dict[str, Any] = {}
        if path.is_file():
            try:
                existing = _load_json(path)
            except (OSError, ValueError, TypeError):
                existing = {}
        raw_start = existing.get("started_at")
        try:
            self.started_wall = datetime.fromisoformat(str(raw_start))
        except (TypeError, ValueError):
            self.started_wall = datetime.now().astimezone()

    def update(
        self,
        progress: float,
        phase: str,
        detail: str,
        *,
        status: str = "running",
    ) -> None:
        now = datetime.now().astimezone()
        elapsed = max(0.0, (now - self.started_wall).total_seconds())
        progress = float(np.clip(progress, 0.0, 100.0))
        if status == "complete":
            eta = "완료"
        elif status == "failed":
            eta = "중단"
        elif progress > 0.5:
            total = elapsed / (progress / 100.0)
            eta = (now + timedelta(seconds=max(0.0, total - elapsed))).strftime(
                "%Y-%m-%d %H:%M:%S KST"
            )
        else:
            eta = "계산 중"
        write_json(
            self.path,
            {
                "title": "P1 dual-flank block inpaint v1",
                "status": status,
                "phase": phase,
                "progress": round(progress, 2),
                "detail": detail,
                "started_at": self.started_wall.isoformat(),
                "updated_at": now.isoformat(),
                "elapsed_seconds": round(elapsed, 3),
                "eta": eta,
            },
        )


def _config_from_contract(contract: Mapping[str, Any]) -> BlockInpaintConfig:
    model = contract["model"]
    block = contract["block_contract"]
    if not isinstance(model, Mapping) or not isinstance(block, Mapping):
        raise TypeError("model and block_contract must be mappings")
    return BlockInpaintConfig(
        cadence_minutes=int(block["cadence_minutes"]),
        mask_hours=tuple(int(value) for value in block["mask_hours"]),  # type: ignore[arg-type]
        stride_hours=int(block["stride_hours"]),
        left_flank_hours=int(block["left_flank_hours"]),
        right_flank_hours=int(block["right_flank_hours"]),
        hidden_size=int(model["hidden_size"]),
        decoder_size=int(model["decoder_size"]),
        dropout=float(model["dropout"]),
        batch_size=int(model["batch_size"]),
        max_epochs=int(model["max_epochs"]),
        patience=int(model["patience"]),
        learning_rate=float(model["learning_rate"]),
        weight_decay=float(model["weight_decay"]),
        consistency_weight=float(model["consistency_weight"]),
        train_windows_per_mask_length=int(model["train_windows_per_mask_length"]),
        validation_fraction=float(model["validation_fraction"]),
        use_bfloat16=bool(model["use_bfloat16"]),
        seed=int(model["seed"]),
    )


def _validate_contract(
    contract: Mapping[str, Any], experiment_config: Path, p1_config: Path
) -> None:
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("unexpected experiment id")
    authorization = contract.get("authorization")
    if not isinstance(authorization, Mapping) or any(
        bool(value) for value in authorization.values()
    ):
        raise RuntimeError("all mutation, upload, and external-data authorizations must be false")
    reference = contract["reference"]
    if not isinstance(reference, Mapping):
        raise TypeError("reference must be a mapping")
    frozen_files = {
        "model": (reference["model_relative_path"], reference["model_sha256"]),
        "oof": (reference["oof_relative_path"], reference["oof_sha256"]),
        "submission": (
            reference["submission_relative_path"],
            reference["submission_sha256"],
        ),
        "selection": (
            reference["selection_relative_path"],
            reference["selection_sha256"],
        ),
        "metrics": (reference["metrics_relative_path"], reference["metrics_sha256"]),
    }
    for name, (relative, expected) in frozen_files.items():
        actual = sha256_file(PROJECT_ROOT / str(relative))
        if actual != str(expected):
            raise RuntimeError(f"frozen {name} SHA mismatch: {actual}")
    if sha256_file(p1_config) != str(reference["p1_config_sha256"]):
        raise RuntimeError("frozen P1 config SHA mismatch")
    candidate = contract["additive_gate"]
    if not isinstance(candidate, Mapping):
        raise TypeError("additive_gate must be a mapping")
    if tuple(candidate["score_columns"]) != SCORE_COLUMNS:
        raise RuntimeError("score-column contract changed")
    if bool(candidate["hyperparameter_search"]):
        raise RuntimeError("hyperparameter search is forbidden")
    if contract["model"]["hyperparameter_search"]:  # type: ignore[index]
        raise RuntimeError("model search is forbidden")
    if int(contract["block_contract"]["purge_days"]) != 7:  # type: ignore[index]
        raise RuntimeError("purge contract changed")
    _config_from_contract(contract)
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
    return {"available": True, "dirty": bool(lines), "changed_path_count": len(lines)}


def _synthetic_frame(rows_per_layer: int = 2200) -> pd.DataFrame:
    time_index = pd.date_range("2024-01-01", periods=rows_per_layer, freq="10min", tz="Asia/Seoul")
    phase = np.arange(rows_per_layer, dtype=np.float64)
    parts: list[pd.DataFrame] = []
    for layer, depth, shift in ((1, 5.0, 0.0), (2, 20.0, -0.25)):
        parts.append(
            pd.DataFrame(
                {
                    "station": "SYN-ORS",
                    "year": 2024,
                    "layer": layer,
                    "time": time_index.astype(str),
                    "temp": 15.0
                    + shift
                    + 0.65 * np.sin(2.0 * np.pi * phase / 74.52)
                    + 0.15 * np.sin(2.0 * np.pi * phase / (24.0 * 6.0)),
                    "psal": 32.0 + 0.08 * np.cos(2.0 * np.pi * phase / 144.0),
                    "depth": depth,
                    "label": np.zeros(rows_per_layer, dtype=np.int8),
                    "anomaly_type": "",
                }
            )
        )
    frame = pd.concat(parts, ignore_index=True)
    event_time = time_index[1500:1800].astype(str)
    event = frame["layer"].eq(1) & frame["time"].isin(event_time)
    frame.loc[event, "temp"] += np.linspace(2.5, 4.0, int(event.sum()))
    frame.loc[event, "label"] = 1
    frame.loc[event, "anomaly_type"] = "offset+drift"
    return frame


def _synthetic_smoke(config: BlockInpaintConfig, artifact_dir: Path) -> dict[str, Any]:
    frame = _synthetic_frame()
    design = build_safe_design(frame)
    assert_target_safe_contract(frame, design)
    parsed = pd.to_datetime(frame["time"], utc=True, format="mixed")
    unique_times = np.sort(parsed.unique())
    fit_cut = unique_times[1200]
    eval_cut = unique_times[1250]
    fit_idx = np.flatnonzero(parsed.lt(fit_cut).to_numpy())
    eval_idx = np.flatnonzero(parsed.ge(eval_cut).to_numpy())
    scaler = fit_covariate_scaler(design, fit_idx)
    prepared_fit = prepare_series(frame, design, scaler, fit_idx)
    smoke_config = replace(
        config,
        batch_size=16,
        max_epochs=2,
        patience=1,
        train_windows_per_mask_length=24,
        validation_fraction=0.2,
        use_bfloat16=False,
    )
    fit_result = train_inpaint_model(
        prepared_fit,
        smoke_config,
        checkpoint_path=artifact_dir / "synthetic_model.pt",
    )
    label_free = frame.drop(columns=["label", "anomaly_type"])
    prepared_eval = prepare_series(label_free, design, scaler, eval_idx)
    specs = enumerate_blocks(prepared_eval, smoke_config, normal_only=False)
    if not specs:
        raise RuntimeError("synthetic smoke has no eligible evaluation block")
    assert_mask_invariance(prepared_eval, specs[len(specs) // 2], smoke_config)
    scores, audit = score_inpaint_model(fit_result, prepared_eval)
    truth = frame.iloc[eval_idx]["label"].to_numpy(dtype=np.int8)
    layer = frame.iloc[eval_idx]["layer"].to_numpy(dtype=np.int64)
    residual = scores[SCORE_COLUMNS[1]].to_numpy(dtype=np.float64)
    event = (truth == 1) & (layer == 1) & np.isfinite(residual)
    normal = (truth == 0) & (layer == 1) & np.isfinite(residual)
    event_median = float(np.median(residual[event]))
    normal_median = float(np.median(residual[normal]))
    separation = event_median / max(normal_median, 1.0e-6)
    passed = bool(event.sum() >= 100 and normal.sum() >= 100 and separation > 1.5)
    report = {
        "passed": passed,
        "mask_invariance_passed": True,
        "event_covered_rows": int(event.sum()),
        "normal_covered_rows": int(normal.sum()),
        "event_abs_residual_median": event_median,
        "normal_abs_residual_median": normal_median,
        "median_separation_ratio": separation,
        "training": {
            "best_epoch": fit_result.best_epoch,
            "best_validation_loss": fit_result.best_validation_loss,
            "train_windows": fit_result.train_windows,
            "validation_windows": fit_result.validation_windows,
        },
        "scoring": audit,
    }
    if not passed:
        raise RuntimeError(f"synthetic block-inpaint smoke failed: {report}")
    return report


def _real_coverage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: BlockInpaintConfig,
) -> tuple[Any, dict[str, Any]]:
    train_design = build_safe_design(train)
    assert_target_safe_contract(train, train_design)
    normal = np.flatnonzero(train["label"].to_numpy(dtype=np.int8) == 0)
    scaler = fit_covariate_scaler(train_design, normal)
    prepared_train = prepare_series(train, train_design, scaler, np.arange(len(train)))
    covered_train, train_audit = coverage_audit(prepared_train, config)
    eligible = enumerate_blocks(prepared_train, config, normal_only=False)
    if not eligible:
        raise RuntimeError("real train has no eligible block")
    for spec in eligible[:: max(1, len(eligible) // 8)][:8]:
        assert_mask_invariance(prepared_train, spec, config)

    test_design = build_safe_design(test)
    assert_target_safe_contract(test, test_design)
    prepared_test = prepare_series(test, test_design, scaler, np.arange(len(test)))
    covered_test, test_audit = coverage_audit(prepared_test, config)
    if not np.isfinite(prepared_test.covariates).all():
        raise RuntimeError("test fallback produced non-finite covariates")
    g_mask = test["station"].astype(str).eq("G-ORS").to_numpy()
    report = {
        "passed": bool(covered_train.any() and covered_test.any()),
        "train": train_audit,
        "test": test_audit,
        "train_covered_rows_recomputed": int(covered_train.sum()),
        "test_covered_rows_recomputed": int(covered_test.sum()),
        "g_ors_rows": int(g_mask.sum()),
        "g_ors_depth_all_missing": bool(test.loc[g_mask, "depth"].isna().all()),
        "g_ors_fallback_covariates_finite": bool(
            np.isfinite(prepared_test.covariates[prepared_test.stations == "G-ORS"]).all()
        ),
        "mask_invariance_samples": min(8, len(eligible)),
    }
    if not report["passed"] or not report["g_ors_fallback_covariates_finite"]:
        raise RuntimeError(f"real coverage/fallback audit failed: {report}")
    return train_design, report


def _time_indices(frame: pd.DataFrame, start: str | None, end: str) -> np.ndarray:
    parsed = pd.to_datetime(frame["time"], utc=True, format="mixed")
    mask = parsed.le(pd.Timestamp(end).tz_convert("UTC"))
    if start is not None:
        mask &= parsed.ge(pd.Timestamp(start).tz_convert("UTC"))
    return np.flatnonzero(mask.to_numpy())


def _base_probabilities(
    train: pd.DataFrame,
    bundle: Any,
    p1_config: P1QCConfig,
    fit_idx: np.ndarray,
    predict_indices: Sequence[np.ndarray],
    *,
    seed: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    encoder = TabularEncoder().fit(bundle, fit_idx)
    fit_matrix = encoder.transform(bundle, fit_idx)
    truth = train.iloc[fit_idx]["label"].to_numpy(dtype=np.int8)
    model = _fit_model(
        BACKEND,
        _model_parameters(p1_config, BACKEND),
        seed,
        _threads(p1_config),
        fit_matrix,
        truth,
    )
    probabilities = [
        model.predict_proba(encoder.transform(bundle, indices))[:, 1] for indices in predict_indices
    ]
    return probabilities, {
        "fit_rows": int(len(fit_idx)),
        "fit_positive_rows": int(truth.sum()),
        "features": int(fit_matrix.shape[1]),
        "trees": int(_model_parameters(p1_config, BACKEND).get("n_estimators", 700)),
    }


def _inpaint_scores(
    frame: pd.DataFrame,
    label_free_frame: pd.DataFrame,
    design: Any,
    fit_idx: np.ndarray,
    score_indices: Sequence[np.ndarray],
    config: BlockInpaintConfig,
    checkpoint: Path,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    scaler = fit_covariate_scaler(design, fit_idx)
    prepared_fit = prepare_series(frame, design, scaler, fit_idx)
    result = train_inpaint_model(
        prepared_fit,
        config,
        checkpoint_path=checkpoint,
    )
    outputs: list[pd.DataFrame] = []
    scoring: list[dict[str, Any]] = []
    for indices in score_indices:
        prepared = prepare_series(label_free_frame, design, scaler, indices)
        scores, audit = score_inpaint_model(result, prepared)
        outputs.append(scores)
        scoring.append(audit)
    audit = {
        "training": {
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "train_windows": result.train_windows,
            "validation_windows": result.validation_windows,
            "windows_by_length": result.windows_by_length,
            "device": result.device,
        },
        "scoring": scoring,
    }
    return outputs, audit


def _gate_config(contract: Mapping[str, Any]) -> dict[str, Any]:
    section = contract["additive_gate"]
    return {
        "l2_penalty": float(section["l2_penalty"]),  # type: ignore[index]
        "maximum_iterations": int(section["maximum_iterations"]),  # type: ignore[index]
        "bounds": tuple(
            (float(pair[0]), float(pair[1]))
            for pair in section["coefficient_bounds"]  # type: ignore[index]
        ),
    }


def _evaluate_gate(
    evaluated: pd.DataFrame,
    test: pd.DataFrame,
    gate_contract: Mapping[str, Any],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    truth = evaluated["label"].to_numpy(dtype=np.int8)
    candidate = evaluated["candidate_prediction"].to_numpy(dtype=np.int8)
    baseline = evaluated["baseline_prediction"].to_numpy(dtype=np.int8)
    weights = group_row_shares(test)
    candidate_report = evaluate_predictions(
        truth,
        candidate,
        evaluated,
        group_weights=weights,
        anomaly_type=evaluated["anomaly_type"],
    )
    baseline_report = evaluate_predictions(
        truth,
        baseline,
        evaluated,
        group_weights=weights,
        anomaly_type=evaluated["anomaly_type"],
    )
    bootstrap = paired_block_bootstrap(
        truth,
        candidate,
        baseline,
        evaluated,
        replicates=replicates,
        seed=seed,
        normal_day_timezone="Asia/Seoul",
    )
    fp = normal_station_layer_day_fp(truth, candidate, baseline, evaluated)
    candidate_fp = float(fp["candidate"]["false_positive_rows_per_normal_station_layer_day"])
    baseline_fp = float(fp["baseline"]["false_positive_rows_per_normal_station_layer_day"])
    if baseline_fp == 0.0:
        fp_relative = 0.0 if candidate_fp == 0.0 else None
        fp_pass = candidate_fp == 0.0
    else:
        fp_relative = (candidate_fp - baseline_fp) / baseline_fp
        fp_pass = fp_relative < float(gate_contract["normal_fp_day_relative_increase_lt"])

    candidate_groups = candidate_report.groups.set_index(["station", "layer"])
    baseline_groups = baseline_report.groups.set_index(["station", "layer"])
    group_comparison: list[dict[str, Any]] = []
    for key in candidate_groups.index.union(baseline_groups.index):
        candidate_f1 = float(candidate_groups.loc[key, "f1"])
        baseline_f1 = float(baseline_groups.loc[key, "f1"])
        group_comparison.append(
            {
                "station": str(key[0]),
                "layer": int(key[1]),
                "candidate_f1": candidate_f1,
                "baseline_f1": baseline_f1,
                "delta_f1": candidate_f1 - baseline_f1,
            }
        )
    worst = min((row["delta_f1"] for row in group_comparison), default=0.0)
    weighted_delta = candidate_report.weighted.f1 - baseline_report.weighted.f1
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
        "normal_fp_day_pass": bool(fp_pass),
        "worst_station_layer_f1_delta": float(worst),
        "worst_station_layer_pass": bool(
            worst >= float(gate_contract["station_layer_f1_delta_min"])
        ),
    }
    gates["promotion_passed"] = bool(
        gates["weighted_f1_pass"]
        and gates["bootstrap_pass"]
        and gates["normal_fp_day_pass"]
        and gates["worst_station_layer_pass"]
    )
    folds: list[dict[str, Any]] = []
    for name, part in evaluated.groupby("fold", sort=False, observed=True):
        fold_truth = part["label"].to_numpy(dtype=np.int8)
        fold_candidate = evaluate_predictions(
            fold_truth,
            part["candidate_prediction"].to_numpy(dtype=np.int8),
            part,
            group_weights=weights,
            anomaly_type=part["anomaly_type"],
        )
        fold_baseline = evaluate_predictions(
            fold_truth,
            part["baseline_prediction"].to_numpy(dtype=np.int8),
            part,
            group_weights=weights,
            anomaly_type=part["anomaly_type"],
        )
        folds.append(
            {
                "fold": str(name),
                "candidate_weighted_f1": fold_candidate.weighted.f1,
                "baseline_weighted_f1": fold_baseline.weighted.f1,
                "delta_weighted_f1": fold_candidate.weighted.f1 - fold_baseline.weighted.f1,
            }
        )
    return {
        "candidate": candidate_report.to_dict(),
        "baseline": baseline_report.to_dict(),
        "paired_block_bootstrap": bootstrap,
        "normal_station_layer_day_fp": fp,
        "station_layer_comparison": group_comparison,
        "fold_comparison": folds,
        "gates": gates,
        "decision": "passed" if gates["promotion_passed"] else "failed_gate",
    }


def _historical_gate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    design: Any,
    bundle: Any,
    p1_config: P1QCConfig,
    config: BlockInpaintConfig,
    contract: Mapping[str, Any],
    artifact_dir: Path,
    status: StatusWriter,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    section = contract["historical_gate"]
    folds = section["folds"]  # type: ignore[index]
    fixed_postprocess = contract["reference"]["fixed_postprocess"]  # type: ignore[index]
    label_free = train.drop(columns=["label", "anomaly_type"])
    parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for number, fold in enumerate(folds):
        name = str(fold["name"])
        progress = 20.0 + number * 17.0
        fit_idx = _time_indices(train, None, str(fold["fit_end"]))
        calibration_idx = _time_indices(
            train, str(fold["calibration_start"]), str(fold["calibration_end"])
        )
        validation_idx = _time_indices(
            train, str(fold["validation_start"]), str(fold["validation_end"])
        )
        fit_end = pd.Timestamp(fold["fit_end"])
        cal_start = pd.Timestamp(fold["calibration_start"])
        cal_end = pd.Timestamp(fold["calibration_end"])
        val_start = pd.Timestamp(fold["validation_start"])
        if cal_start - fit_end <= pd.Timedelta(days=7) or val_start - cal_end <= pd.Timedelta(
            days=7
        ):
            raise RuntimeError(f"{name} violates the strict seven-day purge")
        if min(len(fit_idx), len(calibration_idx), len(validation_idx)) == 0:
            raise RuntimeError(f"{name} contains an empty scope")
        if (
            min(
                int(train.iloc[fit_idx]["label"].sum()),
                int(train.iloc[calibration_idx]["label"].sum()),
                int(train.iloc[validation_idx]["label"].sum()),
            )
            == 0
        ):
            raise RuntimeError(f"{name} lacks positive support")

        status.update(progress, f"{name}:base", "고정 700-tree XGBoost fit/cal/validation 예측")
        base_probabilities, base_audit = _base_probabilities(
            train,
            bundle,
            p1_config,
            fit_idx,
            [calibration_idx, validation_idx],
            seed=p1_config.seed,
        )
        status.update(
            progress + 5.0,
            f"{name}:inpaint",
            "정상 fit window만으로 dual-flank GRU 학습·잔차 산출",
        )
        score_parts, inpaint_audit = _inpaint_scores(
            train,
            label_free,
            design,
            fit_idx,
            [calibration_idx, validation_idx],
            config,
            artifact_dir / "historical" / name / "model.pt",
        )
        calibration_probability, validation_probability = base_probabilities
        calibration_scores, validation_scores = score_parts
        gate = fit_additive_gate(
            calibration_probability,
            calibration_scores,
            train.iloc[calibration_idx]["label"].to_numpy(dtype=np.int8),
            **_gate_config(contract),
        )
        candidate_probability = apply_additive_gate(validation_probability, validation_scores, gate)
        validation_frame = train.iloc[validation_idx].copy()
        plateau = detect_plateaus(validation_frame).to_numpy()
        spike = detect_singleton_spikes(validation_frame).to_numpy()
        baseline_prediction = apply_postprocess(
            validation_frame,
            validation_probability,
            plateau,
            spike,
            fixed_postprocess,
        )
        candidate_prediction = apply_postprocess(
            validation_frame,
            candidate_probability,
            plateau,
            spike,
            fixed_postprocess,
        )
        output = validation_frame.loc[:, [*KEY_COLUMNS, "label", "anomaly_type"]].copy()
        output["fold"] = name
        output["baseline_probability"] = validation_probability.astype(np.float32)
        output["candidate_probability"] = candidate_probability.astype(np.float32)
        output["baseline_prediction"] = baseline_prediction.astype(np.int8)
        output["candidate_prediction"] = candidate_prediction.astype(np.int8)
        for column in SCORE_COLUMNS:
            output[column] = validation_scores[column].to_numpy(dtype=np.float32)
        output["bmi_coverage_count"] = validation_scores["bmi_coverage_count"].to_numpy(
            dtype=np.int32
        )
        parts.append(output)
        audits.append(
            {
                "fold": name,
                "fit_rows": int(len(fit_idx)),
                "calibration_rows": int(len(calibration_idx)),
                "validation_rows": int(len(validation_idx)),
                "base": base_audit,
                "inpaint": inpaint_audit,
                "gate": {
                    "coefficients": gate.coefficients.tolist(),
                    "scales": gate.scales.tolist(),
                    "objective": gate.objective,
                    "iterations": gate.iterations,
                    "success": gate.success,
                },
                "fixed_postprocess": dict(fixed_postprocess),
            }
        )
        status.update(progress + 15.0, f"{name}:complete", f"historical {number + 1}/3 완료")
    evaluated = pd.concat(parts, ignore_index=True)
    result = _evaluate_gate(
        evaluated,
        test,
        section["promotion_gate"],  # type: ignore[index]
        replicates=int(section["bootstrap_replicates"]),  # type: ignore[index]
        seed=int(section["bootstrap_seed"]),  # type: ignore[index]
    )
    result["fold_audits"] = audits
    result["outer_evaluation_count"] = 0
    result["fixed_postprocess_no_grid_search"] = True
    return evaluated, result


def _fold_postprocess(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    reference = contract["reference"]
    metrics = _load_json(PROJECT_ROOT / str(reference["metrics_relative_path"]))  # type: ignore[index]
    result: dict[str, dict[str, Any]] = {}
    for fold in metrics["folds"]:
        result[str(fold["fold"])] = dict(fold["selected_postprocess"])
    return result


def _outer_predictions(
    train: pd.DataFrame,
    design: Any,
    bundle: Any,
    p1_config: P1QCConfig,
    config: BlockInpaintConfig,
    contract: Mapping[str, Any],
    artifact_dir: Path,
    status: StatusWriter,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    reference_path = PROJECT_ROOT / str(contract["reference"]["oof_relative_path"])  # type: ignore[index]
    reference = pd.read_parquet(
        reference_path,
        columns=[*KEY_COLUMNS, "fold", "probability", "prediction", "plateau", "spike_candidate"],
    ).rename(
        columns={
            "probability": "baseline_probability",
            "prediction": "baseline_prediction",
        }
    )
    folds = outer_folds(
        train,
        config=p1_config.splits,
        cadence_minutes=p1_config.data.cadence_minutes,
        group_columns=p1_config.data.group_columns,
    )
    validation = p1_config.raw.get("validation", {})
    calibration_days = int(validation.get("calibration_days", 60))
    label_free = train.drop(columns=["label", "anomaly_type"])
    postprocess = _fold_postprocess(contract)
    parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for number, fold in enumerate(folds):
        status.update(78.0 + number * 6.0, f"{fold.name}:outer", "label-blind outer 후보 생성")
        inner_fit, calibration = _inner_calibration_indices(
            train,
            fold,
            calibration_days=calibration_days,
            purge_days=p1_config.splits.purge_days,
        )
        [calibration_probability], base_audit = _base_probabilities(
            train,
            bundle,
            p1_config,
            inner_fit,
            [calibration],
            seed=p1_config.seed + number,
        )
        [calibration_scores], inner_inpaint = _inpaint_scores(
            train,
            label_free,
            design,
            inner_fit,
            [calibration],
            config,
            artifact_dir / "outer" / fold.name / "inner_model.pt",
        )
        gate = fit_additive_gate(
            calibration_probability,
            calibration_scores,
            train.iloc[calibration]["label"].to_numpy(dtype=np.int8),
            **_gate_config(contract),
        )
        [validation_scores], outer_inpaint = _inpaint_scores(
            train,
            label_free,
            design,
            fold.train_idx,
            [fold.val_idx],
            config,
            artifact_dir / "outer" / fold.name / "outer_model.pt",
        )
        key_frame = train.iloc[fold.val_idx].loc[:, KEY_COLUMNS].copy()
        baseline_part = reference.loc[reference["fold"].eq(fold.name)].copy()
        aligned = key_frame.merge(
            baseline_part,
            on=list(KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        if len(aligned) != len(key_frame):
            raise RuntimeError(f"{fold.name} frozen OOF key mismatch")
        candidate_probability = apply_additive_gate(
            aligned["baseline_probability"].to_numpy(dtype=np.float64),
            validation_scores,
            gate,
        )
        validation_frame = train.iloc[fold.val_idx].drop(columns=["label", "anomaly_type"]).copy()
        candidate_prediction = apply_postprocess(
            validation_frame,
            candidate_probability,
            aligned["plateau"].to_numpy(dtype=bool),
            aligned["spike_candidate"].to_numpy(dtype=bool),
            postprocess[fold.name],
        )
        aligned["candidate_probability"] = candidate_probability.astype(np.float32)
        aligned["candidate_prediction"] = candidate_prediction.astype(np.int8)
        for column in SCORE_COLUMNS:
            aligned[column] = validation_scores[column].to_numpy(dtype=np.float32)
        aligned["bmi_coverage_count"] = validation_scores["bmi_coverage_count"].to_numpy(
            dtype=np.int32
        )
        parts.append(aligned)
        audits.append(
            {
                "fold": fold.name,
                "inner_fit_rows": int(len(inner_fit)),
                "calibration_rows": int(len(calibration)),
                "outer_train_rows": int(len(fold.train_idx)),
                "validation_rows": int(len(fold.val_idx)),
                "base": base_audit,
                "inner_inpaint": inner_inpaint,
                "outer_inpaint": outer_inpaint,
                "gate": {
                    "coefficients": gate.coefficients.tolist(),
                    "scales": gate.scales.tolist(),
                    "success": gate.success,
                },
                "postprocess": postprocess[fold.name],
            }
        )
    predictions = pd.concat(parts, ignore_index=True)
    if predictions.duplicated([*KEY_COLUMNS, "fold"]).any():
        raise RuntimeError("outer candidate keys are not unique")
    return predictions, audits


def _create_exposure_lock(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _outer_evaluate(
    predictions: pd.DataFrame,
    test: pd.DataFrame,
    contract: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference_path = PROJECT_ROOT / str(contract["reference"]["oof_relative_path"])  # type: ignore[index]
    labels = pd.read_parquet(
        reference_path,
        columns=[*KEY_COLUMNS, "fold", "label", "anomaly_type"],
    )
    evaluated = predictions.merge(
        labels,
        on=[*KEY_COLUMNS, "fold"],
        how="inner",
        validate="one_to_one",
    )
    if len(evaluated) != len(predictions) or len(evaluated) != len(labels):
        raise RuntimeError("outer label join mismatch")
    section = contract["outer_evaluation"]
    result = _evaluate_gate(
        evaluated,
        test,
        section["promotion_gate"],  # type: ignore[index]
        replicates=int(section["bootstrap_replicates"]),  # type: ignore[index]
        seed=int(section["bootstrap_seed"]),  # type: ignore[index]
    )
    expected = float(contract["reference"]["outer_test_share_weighted_f1"])  # type: ignore[index]
    if abs(float(result["baseline"]["weighted"]["f1"]) - expected) > 1.0e-12:
        raise RuntimeError("frozen outer baseline weighted F1 failed to reproduce")
    result["outer_evaluation_count"] = 1
    result["outer_is_independent_holdout"] = False
    result["official_hidden_test_used"] = False
    return evaluated, result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_config = (PROJECT_ROOT / args.experiment_config).resolve(strict=True)
    p1_config_path = (PROJECT_ROOT / args.p1_config).resolve(strict=True)
    contract = _load_json(experiment_config)
    _validate_contract(contract, experiment_config, p1_config_path)
    config = _config_from_contract(contract)
    status_path = PROJECT_ROOT / str(contract["artifacts"]["status_relative_path"])
    status = StatusWriter(status_path)

    if args.smoke_only:
        temporary = PROJECT_ROOT / "artifacts" / "p1_block_inpaint_smoke_tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        report = _synthetic_smoke(config, temporary)
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    artifact_dir = PROJECT_ROOT / str(contract["artifacts"]["output_relative_dir"])
    lock_path = PROJECT_ROOT / str(contract["artifacts"]["outer_lock_relative_path"])
    if artifact_dir.exists() or lock_path.exists():
        raise FileExistsError("one-shot artifact or outer lock already exists")
    artifact_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now().astimezone()
    manifest: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "status": "running",
        "started_at": started.isoformat(),
        "git_sha": git_sha(),
        "git_worktree": _git_dirty_summary(),
        "environment": environment_summary(),
        "authorization": contract["authorization"],
        "external_observation_values_used": False,
        "submission_created": False,
        "outer_evaluation_count": 0,
        "contract_hash": contract_hash(config),
        "experiment_config_sha256": sha256_file(experiment_config),
        "p1_config_sha256": sha256_file(p1_config_path),
        "frozen_before": {
            "model": contract["reference"]["model_sha256"],
            "oof": contract["reference"]["oof_sha256"],
            "submission": contract["reference"]["submission_sha256"],
        },
    }
    write_json(artifact_dir / "manifest.json", manifest)
    try:
        status.update(25.0, "synthetic_smoke", "누출 불변성 및 합성 offset+drift 분리 smoke")
        smoke = _synthetic_smoke(config, artifact_dir)
        write_json(artifact_dir / "synthetic_smoke.json", smoke)

        status.update(30.0, "load", "P1 원본 SHA/audit 검증 후 train/test 읽기")
        p1_config = load_config(p1_config_path, env={})
        data_dir = resolve_data_dir(p1_config, args.data_dir)
        train, test = load_train_test(data_dir, audit=True, strict=True)
        if train.attrs.get("source_sha256") != contract["data"]["train_sha256"]:
            raise RuntimeError("train.csv SHA differs from preregistration")
        if test.attrs.get("source_sha256") != contract["data"]["test_sha256"]:
            raise RuntimeError("test.csv SHA differs from preregistration")
        manifest["inputs"] = {
            "train_sha256": train.attrs["source_sha256"],
            "test_sha256": test.attrs["source_sha256"],
            "train_rows": len(train),
            "test_rows": len(test),
        }
        write_json(artifact_dir / "manifest.json", manifest)

        status.update(34.0, "real_coverage", "실데이터 gap/coverage/G-depth fallback 감사")
        design, coverage = _real_coverage(train, test, config)
        write_json(artifact_dir / "real_coverage.json", coverage)

        status.update(38.0, "base_features", "동결 XGBoost offline feature cache 확인")
        bundle = load_or_build_features(train, p1_config, kind="train", use_cache=True)

        status.update(40.0, "historical_gate", "3개 과거 rolling-origin gate 시작")
        historical_predictions, historical = _historical_gate(
            train,
            test,
            design,
            bundle,
            p1_config,
            config,
            contract,
            artifact_dir,
            status,
        )
        historical_path = artifact_dir / "historical_evaluated.parquet"
        historical_predictions.to_parquet(historical_path, index=False, compression="zstd")
        historical_metrics_path = write_json(artifact_dir / "historical_metrics.json", historical)
        manifest["historical_decision"] = historical["decision"]
        manifest["historical_gates"] = historical["gates"]
        manifest["historical_metrics_sha256"] = sha256_file(historical_metrics_path)
        manifest["historical_predictions_sha256"] = sha256_file(historical_path)
        write_json(artifact_dir / "manifest.json", manifest)

        if not historical["gates"]["promotion_passed"]:
            finished = datetime.now().astimezone()
            manifest.update(
                {
                    "status": "complete",
                    "decision": "failed_historical_gate",
                    "family_closed": True,
                    "outer_evaluation_count": 0,
                    "finished_at": finished.isoformat(),
                    "elapsed_seconds": (finished - started).total_seconds(),
                }
            )
            write_json(artifact_dir / "manifest.json", manifest)
            gates = historical["gates"]
            status.update(
                100.0,
                "complete",
                (
                    f"historical gate 실패 · weighted Δ {gates['weighted_f1_delta']:+.6f} · "
                    f"CI90 lower {gates['bootstrap_ci90_lower']:+.6f} · outer 0회"
                ),
                status="complete",
            )
            print(json.dumps({"decision": manifest["decision"], "gates": gates}, indent=2))
            return 0

        status.update(76.0, "outer_prepare", "historical 통과; outer 후보를 label-blind 생성")
        predictions, outer_audit = _outer_predictions(
            train,
            design,
            bundle,
            p1_config,
            config,
            contract,
            artifact_dir,
            status,
        )
        prediction_path = artifact_dir / "outer_predictions_label_blind.parquet"
        predictions.to_parquet(prediction_path, index=False, compression="zstd")
        write_json(artifact_dir / "outer_prediction_audit.json", outer_audit)
        prediction_sha = sha256_file(prediction_path)
        _create_exposure_lock(
            lock_path,
            {
                "experiment_id": EXPERIMENT_ID,
                "created_at": datetime.now().astimezone().isoformat(),
                "outer_evaluation_count": 1,
                "prediction_sha256": prediction_sha,
                "outer_labels_accessed_before_lock": False,
            },
        )
        status.update(96.0, "outer_evaluation", "고정 예측에 frozen OOF label 1회 결합")
        outer_evaluated, outer = _outer_evaluate(predictions, test, contract)
        outer_path = artifact_dir / "outer_evaluated.parquet"
        outer_evaluated.to_parquet(outer_path, index=False, compression="zstd")
        outer_metrics_path = write_json(artifact_dir / "outer_metrics.json", outer)
        finished = datetime.now().astimezone()
        manifest.update(
            {
                "status": "complete",
                "decision": (
                    "promoted" if outer["gates"]["promotion_passed"] else "failed_outer_gate"
                ),
                "family_closed": not outer["gates"]["promotion_passed"],
                "outer_evaluation_count": 1,
                "finished_at": finished.isoformat(),
                "elapsed_seconds": (finished - started).total_seconds(),
                "outer_prediction_sha256": prediction_sha,
                "outer_metrics_sha256": sha256_file(outer_metrics_path),
                "outer_evaluated_sha256": sha256_file(outer_path),
            }
        )
        write_json(artifact_dir / "manifest.json", manifest)
        gates = outer["gates"]
        status.update(
            100.0,
            "complete",
            (
                f"{manifest['decision']} · weighted Δ {gates['weighted_f1_delta']:+.6f} · "
                f"CI90 lower {gates['bootstrap_ci90_lower']:+.6f} · 업로드 없음"
            ),
            status="complete",
        )
        print(json.dumps({"decision": manifest["decision"], "gates": gates}, indent=2))
        return 0
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now().astimezone().isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_json(artifact_dir / "manifest.json", manifest)
        status.update(100.0, "failed", f"{type(exc).__name__}: {exc}", status="failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
