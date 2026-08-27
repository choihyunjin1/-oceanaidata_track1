"""Fail-closed inner-only comparison for the fixed matched-filter feature family."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import P1QCConfig
from .experiment import sha256_file
from .features import FeatureBundle
from .matched_filter import MATCHED_FILTER_FEATURES
from .metrics import EvaluationReport, evaluate_predictions, group_row_shares
from .pipeline import (
    TabularEncoder,
    _fit_model,
    _model_parameters,
    _threads,
    apply_postprocess,
)
from .rules import detect_plateaus, detect_singleton_spikes
from .validation import normal_station_layer_day_fp, paired_block_bootstrap

FIXED_POSTPROCESS = {
    "high_threshold": 0.2,
    "low_threshold": 0.1,
    "close_gap_rows": 0,
    "minimum_positive_run": 12,
}
EXPECTED_MODULE_SHA = "79e934945679c240aa903fe5618341f623788f1672482837a03f8700da26327f"


@dataclass(frozen=True)
class InnerBlock:
    name: str
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end_exclusive: pd.Timestamp


@dataclass(frozen=True)
class InnerComparisonResult:
    oof: pd.DataFrame
    metrics: dict[str, Any]
    passed: bool


def load_and_validate_contract(path: str | Path, *, project_root: Path) -> dict[str, Any]:
    contract_path = Path(path)
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0-inner-only":
        raise ValueError("matched-filter contract schema drift")
    authorization = payload.get("authorization", {})
    expected_authorization = {
        "inner_comparison": True,
        "outer_one_shot": False,
        "test_prediction": False,
        "submission": False,
        "commit": False,
        "push": False,
    }
    if authorization != expected_authorization:
        raise ValueError("matched-filter authorization drift")
    added = payload.get("single_change", {}).get("added_features")
    if tuple(added or ()) != MATCHED_FILTER_FEATURES:
        raise ValueError("matched-filter feature contract drift")
    if payload.get("postprocess", {}) != {
        **FIXED_POSTPROCESS,
        "selection": "frozen_from_incumbent",
    }:
        raise ValueError("matched-filter postprocess contract drift")
    module = project_root / "src" / "p1_qc" / "matched_filter.py"
    actual_module_hash = sha256_file(module)
    declared_module_hash = payload.get("hashes", {}).get("matched_filter_module_sha256")
    if actual_module_hash != EXPECTED_MODULE_SHA or declared_module_hash != EXPECTED_MODULE_SHA:
        raise ValueError("matched-filter implementation hash drift")
    return payload


def blocks_from_contract(payload: dict[str, Any]) -> tuple[InnerBlock, ...]:
    blocks = tuple(
        InnerBlock(
            str(raw["name"]),
            pd.Timestamp(raw["train_end_kst"]).tz_convert("UTC"),
            pd.Timestamp(raw["validation_start_kst"]).tz_convert("UTC"),
            pd.Timestamp(raw["validation_end_exclusive_kst"]).tz_convert("UTC"),
        )
        for raw in payload["inner_blocks"]
    )
    if tuple(block.name for block in blocks) != ("I1", "I2", "I3"):
        raise ValueError("inner block names/order drift")
    for block in blocks:
        if block.validation_start - block.train_end < pd.Timedelta(days=7):
            raise ValueError(f"inner block {block.name} has insufficient purge")
    if blocks[-1].validation_end_exclusive > pd.Timestamp("2025-03-01T00:00:00+09:00").tz_convert(
        "UTC"
    ):
        raise ValueError("inner comparison exceeds the predeclared development scope")
    return blocks


def _indices(frame: pd.DataFrame, block: InnerBlock) -> tuple[np.ndarray, np.ndarray]:
    time = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    train = np.flatnonzero(time.le(block.train_end).to_numpy())
    validation = np.flatnonzero(
        (time.ge(block.validation_start) & time.lt(block.validation_end_exclusive)).to_numpy()
    )
    if len(train) == 0 or len(validation) == 0:
        raise ValueError(f"inner block {block.name} is empty")
    return train, validation


def _predict_arm(
    bundle: FeatureBundle,
    train: pd.DataFrame,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: P1QCConfig,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    encoder = TabularEncoder().fit(bundle, fit_indices)
    x_fit = encoder.transform(bundle, fit_indices)
    y_fit = train.iloc[fit_indices]["label"].to_numpy(dtype=np.int8)
    x_validation = encoder.transform(bundle, validation_indices)
    parameters = _model_parameters(config, "xgboost")
    parameters["n_estimators"] = 700
    model = _fit_model(
        "xgboost",
        parameters,
        seed,
        _threads(config),
        x_fit,
        y_fit,
    )
    probability = model.predict_proba(x_validation)[:, 1]
    validation_frame = train.iloc[validation_indices].copy()
    plateau = detect_plateaus(validation_frame).to_numpy()
    spike = detect_singleton_spikes(validation_frame).to_numpy()
    prediction = apply_postprocess(
        validation_frame,
        probability,
        plateau,
        spike,
        FIXED_POSTPROCESS,
    )
    return probability.astype(np.float32), prediction.astype(np.int8)


def _group_delta(candidate: EvaluationReport, baseline: EvaluationReport) -> dict[str, float]:
    left = candidate.groups.set_index(["station", "layer"])["f1"]
    right = baseline.groups.set_index(["station", "layer"])["f1"]
    aligned = left.subtract(right, fill_value=0.0)
    return {f"{station}|{layer}": float(value) for (station, layer), value in aligned.items()}


def run_inner_comparison(
    train: pd.DataFrame,
    test: pd.DataFrame,
    baseline_bundle: FeatureBundle,
    candidate_bundle: FeatureBundle,
    config: P1QCConfig,
    contract: dict[str, Any],
) -> InnerComparisonResult:
    if len(train) != len(baseline_bundle.frame) or len(train) != len(candidate_bundle.frame):
        raise ValueError("feature bundle rows do not align with train")
    expected_added = tuple(contract["single_change"]["added_features"])
    if candidate_bundle.feature_columns != (*baseline_bundle.feature_columns, *expected_added):
        raise ValueError("candidate arm must differ by exactly four matched-filter features")
    forbidden = {"label", "anomaly_type"}
    if forbidden.intersection(candidate_bundle.feature_columns):
        raise ValueError("target columns are exposed as model features")

    shares = group_row_shares(test)
    parts: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for ordinal, block in enumerate(blocks_from_contract(contract)):
        fit_indices, validation_indices = _indices(train, block)
        seed = int(contract["model"]["base_seed"]) + ordinal
        baseline_probability, baseline_prediction = _predict_arm(
            baseline_bundle,
            train,
            fit_indices,
            validation_indices,
            config,
            seed=seed,
        )
        candidate_probability, candidate_prediction = _predict_arm(
            candidate_bundle,
            train,
            fit_indices,
            validation_indices,
            config,
            seed=seed,
        )
        validation_frame = train.iloc[validation_indices].copy()
        truth = validation_frame["label"].to_numpy(dtype=np.int8)
        baseline_report = evaluate_predictions(
            truth,
            baseline_prediction,
            validation_frame,
            group_weights=shares,
            anomaly_type=validation_frame["anomaly_type"],
        )
        candidate_report = evaluate_predictions(
            truth,
            candidate_prediction,
            validation_frame,
            group_weights=shares,
            anomaly_type=validation_frame["anomaly_type"],
        )
        fold_metrics.append(
            {
                "block": block.name,
                "fit_rows": len(fit_indices),
                "validation_rows": len(validation_indices),
                "baseline": baseline_report.to_dict(),
                "candidate": candidate_report.to_dict(),
                "micro_f1_delta": candidate_report.micro.f1 - baseline_report.micro.f1,
                "weighted_f1_delta": candidate_report.weighted.f1 - baseline_report.weighted.f1,
            }
        )
        part = validation_frame.loc[:, ["station", "year", "layer", "time", "label"]].copy()
        part["anomaly_type"] = validation_frame["anomaly_type"].fillna("").to_numpy()
        part["baseline_probability"] = baseline_probability
        part["candidate_probability"] = candidate_probability
        part["baseline_prediction"] = baseline_prediction
        part["candidate_prediction"] = candidate_prediction
        part["block"] = block.name
        parts.append(part)

    oof = pd.concat(parts, ignore_index=True)
    truth = oof["label"].to_numpy(dtype=np.int8)
    baseline_prediction = oof["baseline_prediction"].to_numpy(dtype=np.int8)
    candidate_prediction = oof["candidate_prediction"].to_numpy(dtype=np.int8)
    baseline_report = evaluate_predictions(
        truth,
        baseline_prediction,
        oof,
        group_weights=shares,
        anomaly_type=oof["anomaly_type"],
    )
    candidate_report = evaluate_predictions(
        truth,
        candidate_prediction,
        oof,
        group_weights=shares,
        anomaly_type=oof["anomaly_type"],
    )
    bootstrap = paired_block_bootstrap(
        truth,
        candidate_prediction,
        baseline_prediction,
        oof,
        replicates=2000,
        seed=int(contract["model"]["base_seed"]),
        normal_day_timezone="Asia/Seoul",
    )
    false_positive = normal_station_layer_day_fp(
        truth,
        candidate_prediction,
        baseline_prediction,
        oof,
    )
    group_delta = _group_delta(candidate_report, baseline_report)
    gates = contract["promotion_gates"]
    offset_delta = candidate_report.type_recall["offset"] - baseline_report.type_recall["offset"]
    drift_delta = candidate_report.type_recall["drift"] - baseline_report.type_recall["drift"]
    baseline_fp_rate = false_positive["baseline"][
        "false_positive_rows_per_normal_station_layer_day"
    ]
    candidate_fp_rate = false_positive["candidate"][
        "false_positive_rows_per_normal_station_layer_day"
    ]
    relative_fp_increase = (
        (candidate_fp_rate - baseline_fp_rate) / baseline_fp_rate
        if baseline_fp_rate and baseline_fp_rate > 0
        else (0.0 if candidate_fp_rate == 0 else float("inf"))
    )
    checks = {
        "weighted_f1_delta": candidate_report.weighted.f1 - baseline_report.weighted.f1
        >= gates["minimum_aggregate_weighted_f1_delta"],
        "micro_f1_delta": candidate_report.micro.f1 - baseline_report.micro.f1
        >= gates["minimum_aggregate_micro_f1_delta"],
        "bootstrap_ci90_lower": bootstrap["difference_ci90"][0]
        > gates["minimum_bootstrap_ci90_lower"],
        "bootstrap_probability": bootstrap["probability_improved"]
        >= gates["minimum_probability_improved"],
        "nondegrading_blocks": sum(item["weighted_f1_delta"] >= 0 for item in fold_metrics)
        >= gates["minimum_nondegrading_blocks"],
        "worst_group": min(group_delta.values()) >= -gates["maximum_worst_group_f1_drop"],
        "normal_fp_rate": relative_fp_increase
        < gates["maximum_normal_fp_per_day_relative_increase"],
        "slow_type_mean_recall": (offset_delta + drift_delta) / 2
        >= gates["minimum_mean_offset_drift_recall_delta"],
        "slow_type_no_material_drop": min(offset_delta, drift_delta)
        >= -gates["maximum_individual_offset_or_drift_recall_drop"],
    }
    metrics = {
        "experiment_id": contract["experiment_id"],
        "scope": "inner_only_development",
        "folds": fold_metrics,
        "aggregate": {
            "baseline": baseline_report.to_dict(),
            "candidate": candidate_report.to_dict(),
            "micro_f1_delta": candidate_report.micro.f1 - baseline_report.micro.f1,
            "weighted_f1_delta": candidate_report.weighted.f1 - baseline_report.weighted.f1,
            "offset_recall_delta": offset_delta,
            "drift_recall_delta": drift_delta,
            "group_f1_delta": group_delta,
            "worst_group_f1_delta": min(group_delta.values()),
            "normal_fp_relative_increase": relative_fp_increase,
        },
        "bootstrap": bootstrap,
        "normal_fp": false_positive,
        "gate_checks": checks,
        "passed": all(checks.values()),
        "outer_accessed": False,
    }
    return InnerComparisonResult(oof=oof, metrics=metrics, passed=all(checks.values()))


__all__ = [
    "FIXED_POSTPROCESS",
    "InnerBlock",
    "InnerComparisonResult",
    "blocks_from_contract",
    "load_and_validate_contract",
    "run_inner_comparison",
]
