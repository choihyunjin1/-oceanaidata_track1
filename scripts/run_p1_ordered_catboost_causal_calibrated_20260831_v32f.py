"""Exactly-once prefix-calibrated Ordered-CatBoost historical P1 test."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import Pool

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_p1_ordered_catboost_eventday_20260831_v32a as base  # noqa: E402

from p1_qc.config import load_config  # noqa: E402
from p1_qc.splits import Fold, outer_folds  # noqa: E402

EXPERIMENT_ID = "p1_ordered_catboost_causal_calibrated_20260831_v32f"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
ATTEMPT_LOCK = ARTIFACT_DIR / "attempt_lock.json"
RESULT_PATH = ARTIFACT_DIR / "result.json"
OOF_PATH = ARTIFACT_DIR / "historical_oof.parquet"
KEYS = ["station", "year", "layer", "time"]
FOLDS = ["2025_q2", "2025_q3", "2025_q4"]


class ContractError(RuntimeError):
    """Raised when a sealed v32f contract is violated."""


def threshold_grid(config: dict[str, Any]) -> np.ndarray:
    calibration = config["calibration"]
    start = float(calibration["threshold_grid_start"])
    end = float(calibration["threshold_grid_end"])
    step = float(calibration["threshold_grid_step"])
    count = int(round((end - start) / step)) + 1
    grid = np.round(start + np.arange(count, dtype=np.float64) * step, 2)
    if len(grid) != 91 or grid[0] != 0.05 or grid[-1] != 0.95:
        raise ContractError("threshold grid differs from preregistration")
    return grid


def select_threshold(truth: np.ndarray, probability: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError("invalid calibration vectors")
    scored = [(float(base.metric(y, p >= threshold)["f1"]), float(threshold)) for threshold in grid]
    best_score = max(score for score, _ in scored)
    tied = [threshold for score, threshold in scored if abs(score - best_score) <= 1e-12]
    selected = min(tied, key=lambda threshold: (abs(threshold - 0.5), threshold))
    return float(selected), float(best_score)


def split_fit_calibration(
    train: pd.DataFrame,
    fold: Fold,
    *,
    calibration_days: int,
    purge_days: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, str | int]]:
    parsed = pd.to_datetime(train["time"], errors="raise", utc=True, format="mixed")
    calibration_end = fold.train_end
    calibration_start = calibration_end - pd.Timedelta(days=calibration_days)
    fit_end = calibration_start - pd.Timedelta(days=purge_days)
    eligible = np.zeros(len(train), dtype=bool)
    eligible[fold.train_idx] = True
    fit = eligible & parsed.le(fit_end).to_numpy()
    calibration = (
        eligible
        & parsed.gt(calibration_start).to_numpy()
        & parsed.le(calibration_end).to_numpy()
    )
    fit_idx = np.flatnonzero(fit)
    calibration_idx = np.flatnonzero(calibration)
    if not len(fit_idx) or not len(calibration_idx):
        raise ContractError(f"empty fit/calibration split: {fold.name}")
    if np.intersect1d(fit_idx, calibration_idx).size:
        raise ContractError(f"fit/calibration overlap: {fold.name}")
    if parsed.iloc[fit_idx].max() > fit_end or parsed.iloc[calibration_idx].min() <= calibration_start:
        raise ContractError(f"fit/calibration chronology failed: {fold.name}")
    audit: dict[str, str | int] = {
        "fit_rows": int(len(fit_idx)),
        "calibration_rows": int(len(calibration_idx)),
        "fit_max_time_utc": parsed.iloc[fit_idx].max().isoformat(),
        "calibration_min_time_utc": parsed.iloc[calibration_idx].min().isoformat(),
        "calibration_max_time_utc": parsed.iloc[calibration_idx].max().isoformat(),
        "internal_purge_days": int(purge_days),
    }
    return fit_idx, calibration_idx, audit


def execute() -> dict[str, Any]:
    started_wall = base.now_kst()
    started = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if started_wall >= datetime.fromisoformat(config["deadline_kst"]):
        raise ContractError("deadline already passed")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    base.write_json_new(
        ATTEMPT_LOCK,
        {
            "experiment_id": EXPERIMENT_ID,
            "started_kst": started_wall.isoformat(),
            "config_sha256": base.sha256_file(CONFIG_PATH),
            "exactly_once": True,
        },
    )
    raw_data_dir = os.environ.get("P1_DATA_DIR")
    if not raw_data_dir:
        raise ContractError("P1_DATA_DIR is required")
    data_dir = Path(raw_data_dir).expanduser().resolve(strict=True)
    train = base.load_train(data_dir, config["inputs"]["train_csv_sha256"])
    features, categorical = base.load_features(config, len(train))
    p1_config = load_config(ROOT / "configs/p1.toml", env={"P1_DATA_DIR": str(data_dir)})
    folds = outer_folds(train, config=p1_config.splits, cadence_minutes=10)
    reference_path = ROOT / config["inputs"]["tabular_reference_oof"]
    if base.sha256_file(reference_path) != config["inputs"]["tabular_reference_oof_sha256"]:
        raise ContractError("historical OOF hash mismatch")
    oof = pd.read_parquet(
        reference_path,
        columns=[*KEYS, "fold", "label", "deployment_prediction"],
    )
    oof["candidate_probability"] = np.nan
    oof["candidate_prediction"] = -1
    oof["calibrated_threshold"] = np.nan
    row_lookup = train.reset_index(names="train_position")[[*KEYS, "train_position"]]
    aligned = oof[[*KEYS, "fold"]].merge(
        row_lookup, on=KEYS, how="left", validate="one_to_one", sort=False
    )
    if aligned["train_position"].isna().any():
        raise ContractError("OOF alignment failed")
    grid = threshold_grid(config)
    fit_records: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds):
        if fold.name != FOLDS[fold_index]:
            raise ContractError("unexpected fold order")
        if time.perf_counter() - started >= float(config["maximum_runtime_seconds"]):
            raise TimeoutError("runtime cap reached")
        fit_idx, calibration_idx, split_audit = split_fit_calibration(
            train,
            fold,
            calibration_days=int(config["calibration"]["trailing_days"]),
            purge_days=int(config["calibration"]["internal_feature_purge_days"]),
        )
        val_rows = np.flatnonzero(oof["fold"].eq(fold.name).to_numpy())
        val_idx = aligned.loc[val_rows, "train_position"].to_numpy(np.int64)
        if set(val_idx.tolist()) != set(fold.val_idx.tolist()):
            raise ContractError(f"outer membership mismatch: {fold.name}")
        weights = base.event_day_weight(
            train.iloc[fit_idx], train.loc[fit_idx, "label"].to_numpy(np.int8)
        )
        model = base.model_from_config(config)
        fit_pool = Pool(
            features.iloc[fit_idx],
            label=train.loc[fit_idx, "label"].to_numpy(np.int8),
            weight=weights,
            cat_features=categorical,
        )
        calibration_pool = Pool(features.iloc[calibration_idx], cat_features=categorical)
        validation_pool = Pool(features.iloc[val_idx], cat_features=categorical)
        fit_started = time.perf_counter()
        model.fit(fit_pool)
        calibration_probability = model.predict_proba(calibration_pool)[:, 1]
        threshold, calibration_f1 = select_threshold(
            train.loc[calibration_idx, "label"].to_numpy(np.int8),
            calibration_probability,
            grid,
        )
        probability = model.predict_proba(validation_pool)[:, 1]
        prediction = (probability >= threshold).astype(np.int8)
        oof.loc[val_rows, "candidate_probability"] = probability
        oof.loc[val_rows, "candidate_prediction"] = prediction
        oof.loc[val_rows, "calibrated_threshold"] = threshold
        fit_records.append(
            {
                "fold": fold.name,
                **split_audit,
                "validation_rows": int(len(val_idx)),
                "fit_seconds": float(time.perf_counter() - fit_started),
                "threshold": threshold,
                "calibration_f1": calibration_f1,
                "calibration_positive_rate": float((calibration_probability >= threshold).mean()),
                "outer_positive_rate": float(prediction.mean()),
                "iterations": int(model.tree_count_),
            }
        )
        print(json.dumps({"fold_complete": fold.name, **fit_records[-1]}, sort_keys=True), flush=True)
    if oof[["candidate_probability", "calibrated_threshold"]].isna().any().any():
        raise ContractError("candidate OOF is incomplete")
    candidate = oof["candidate_prediction"].to_numpy(np.int8)
    if not np.isin(candidate, [0, 1]).all():
        raise ContractError("candidate OOF is nonbinary")
    if not np.array_equal(
        oof["label"].to_numpy(np.int8),
        train.loc[aligned["train_position"], "label"].to_numpy(np.int8),
    ):
        raise ContractError("historical labels differ after alignment")
    # Outer predictions are frozen before either reference is loaded.
    tabular, e150_prediction = base.aligned_references(oof)
    truth = oof["label"].to_numpy(np.int8)
    by_fold: dict[str, Any] = {}
    replicates = int(config["validation"]["bootstrap_replicates"])
    for fold_index, fold in enumerate(FOLDS):
        mask = oof["fold"].eq(fold).to_numpy()
        candidate_metric = base.metric(truth[mask], candidate[mask])
        tabular_metric = base.metric(truth[mask], tabular[mask])
        e150_metric = base.metric(truth[mask], e150_prediction[mask])
        by_fold[fold] = {
            "candidate": candidate_metric,
            "tabular_reference": tabular_metric,
            "e150_reference": e150_metric,
            "delta_f1_vs_tabular": float(candidate_metric["f1"] - tabular_metric["f1"]),
            "delta_f1_vs_e150": float(candidate_metric["f1"] - e150_metric["f1"]),
            "bootstrap_vs_tabular": base.paired_bootstrap(
                truth[mask],
                candidate[mask],
                tabular[mask],
                oof.loc[mask, ["station", "layer", "time"]].reset_index(drop=True),
                replicates=replicates,
                seed=int(config["seed"]) + fold_index,
            ),
        }
    metadata = oof[["station", "layer", "time"]].reset_index(drop=True)
    pooled_candidate = base.metric(truth, candidate)
    pooled_tabular = base.metric(truth, tabular)
    pooled_bootstrap = base.paired_bootstrap(
        truth,
        candidate,
        tabular,
        metadata,
        replicates=replicates,
        seed=int(config["seed"]) + 10,
    )
    q34 = oof["fold"].isin(["2025_q3", "2025_q4"]).to_numpy()
    q34_candidate = base.metric(truth[q34], candidate[q34])
    q34_e150 = base.metric(truth[q34], e150_prediction[q34])
    q34_bootstrap = base.paired_bootstrap(
        truth[q34],
        candidate[q34],
        e150_prediction[q34],
        metadata.loc[q34].reset_index(drop=True),
        replicates=replicates,
        seed=int(config["seed"]) + 20,
    )
    runtime = float(time.perf_counter() - started)
    gates = {
        "all_q2_q3_q4_delta_f1_vs_tabular_nonnegative": all(
            record["delta_f1_vs_tabular"] >= 0.0 for record in by_fold.values()
        ),
        "pooled_delta_f1_vs_tabular_positive": pooled_candidate["f1"] > pooled_tabular["f1"],
        "pooled_bootstrap_ci90_low_vs_tabular_positive": pooled_bootstrap["difference_ci90"][0] > 0.0,
        "q3_q4_delta_f1_vs_e150_positive": q34_candidate["f1"] > q34_e150["f1"],
        "q3_q4_bootstrap_ci90_low_vs_e150_positive": q34_bootstrap["difference_ci90"][0] > 0.0,
        "runtime_at_most_seconds": runtime <= float(config["maximum_runtime_seconds"]),
        "official_accesses_equal_zero": True,
    }
    delta_e150 = float(q34_candidate["f1"] - q34_e150["f1"])
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "PROMOTE_MATERIALIZER_PREP_ONLY" if all(gates.values()) else "NO_GO_INTERNAL_GATE",
        "independent_candidate": True,
        "candidate_uses_incumbent_as_input": False,
        "candidate_uses_official_values": False,
        "fit_count": len(fit_records),
        "fit_records": fit_records,
        "thresholds": {record["fold"]: record["threshold"] for record in fit_records},
        "runtime_seconds": runtime,
        "by_fold": by_fold,
        "pooled_vs_tabular": {
            "candidate": pooled_candidate,
            "reference": pooled_tabular,
            "delta_f1": float(pooled_candidate["f1"] - pooled_tabular["f1"]),
            "bootstrap": pooled_bootstrap,
        },
        "q3_q4_vs_e150": {
            "candidate": q34_candidate,
            "reference": q34_e150,
            "delta_f1": delta_e150,
            "bootstrap": q34_bootstrap,
        },
        "public_score_translation": {
            "assumption": "empirical local linear slope only; not a guarantee",
            "best_public_f1": base.PUBLIC_BEST_F1,
            "best_public_points": base.PUBLIC_BEST_POINTS,
            "expected_points_center": base.PUBLIC_BEST_POINTS + base.PUBLIC_SCORE_SLOPE * delta_e150,
            "expected_points_ci90": [
                base.PUBLIC_BEST_POINTS
                + base.PUBLIC_SCORE_SLOPE * q34_bootstrap["difference_ci90"][0],
                base.PUBLIC_BEST_POINTS
                + base.PUBLIC_SCORE_SLOPE * q34_bootstrap["difference_ci90"][1],
            ],
        },
        "gates": gates,
        "official_access": {
            "test_rows": 0,
            "sample_rows": 0,
            "submission_rows": 0,
            "hidden_label_rows": 0,
            "uploads": 0,
        },
        "input_hashes": {
            "train_csv": config["inputs"]["train_csv_sha256"],
            "feature_cache": config["inputs"]["feature_cache_sha256"],
            "tabular_reference_oof": config["inputs"]["tabular_reference_oof_sha256"],
            "config": base.sha256_file(CONFIG_PATH),
        },
        "completed_kst": base.now_kst().isoformat(),
    }
    oof.to_parquet(OOF_PATH, index=False, compression="zstd")
    result["historical_oof"] = {
        "path": str(OOF_PATH),
        "rows": int(len(oof)),
        "sha256": base.sha256_file(OOF_PATH),
    }
    base.write_json_new(RESULT_PATH, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "sealed": True}, sort_keys=True))
        return 0
    try:
        result = execute()
    except Exception as error:
        failure = {
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(error).__name__,
            "error": str(error),
            "official_access": {
                "test_rows": 0,
                "sample_rows": 0,
                "submission_rows": 0,
                "hidden_label_rows": 0,
                "uploads": 0,
            },
            "completed_kst": base.now_kst().isoformat(),
        }
        if not RESULT_PATH.exists():
            base.write_json_new(RESULT_PATH, failure)
        raise
    print(json.dumps({"status": result["status"], "runtime_seconds": result["runtime_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
