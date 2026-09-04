"""Exactly-once P1 Public-transport repair training and forward validation.

Historical OOF rows are evaluated before any official covariate is opened.  The
only allowed deployment mutation is an add-only G-ORS/layer-1 extension of the
frozen Public champion.  Official CSVs are materialized only after the frozen
internal, seed, day-block bootstrap, worst-block, and transport-calibration
gates all pass.  Hidden truth and upload are out of scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for directory in (SCRIPTS, SRC):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_full_internal_submission_cycle_20260831_v2 as prior_cycle  # noqa: E402
import run_p1_parallel_candidate_cycle_20260831_v4 as prior_p1  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v5"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
CALIBRATION_PATH = ROOT / "reports/public_transport_calibration_20260831_v1/calibration.json"
AUTHORITATIVE_RESULT = (
    ROOT
    / "reports/parallel_internal_pass_registry_20260831_v1"
    / "official-submission-results-20260831.json"
)
PRIOR_RESULT = ROOT / "artifacts/p1_parallel_candidate_cycle_20260831_v4/result.json"
DELIVERY = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260831_P1_PUBLIC_TRANSPORT_REPAIR_CYCLE_V5"
)
P1_KEYS = ["station", "year", "layer", "time"]
FOLD_ORDER = ["2025_q2", "2025_q3", "2025_q4"]
FAMILIES = ("logistic", "hist_gradient_boosting")
CANDIDATE_NAMES = {
    "logistic": "P1_1_GORS_TRANSPORT_LOGIT_LCB",
    "hist_gradient_boosting": "P1_2_GORS_TRANSPORT_HGB_LCB",
    "consensus": "P1_3_GORS_TRANSPORT_CONSENSUS_LCB",
}
MODEL_FEATURES = [
    "probability_base",
    "probability_peer",
    "deployment_prediction_base",
    "deployment_prediction_peer",
    "e150_probability",
    "e150_boundary_start",
    "e150_boundary_end",
    "pmax",
    "pmean",
    "probability_disagreement",
    "since_anchor",
    "signal_run_length",
    "probability_base_past6_mean",
    "probability_base_past6_max",
    "probability_base_lag1",
    "probability_peer_past6_mean",
    "probability_peer_past6_max",
    "probability_peer_lag1",
    "e150_probability_past6_mean",
    "e150_probability_past6_max",
    "e150_probability_lag1",
    "pmax_past6_mean",
    "pmax_past6_max",
    "pmax_lag1",
]


class ContractError(RuntimeError):
    """Raised when a frozen experimental contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def update_progress(payload: dict[str, Any]) -> None:
    write_json(ARTIFACT / "progress.json", payload, exclusive=False)


def load_contract() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    p1_calibration = calibration["gates"]["P1"]
    policy = config["decision_policy"]
    comparisons = {
        "transport_residual": np.isclose(
            policy["worst_observed_public_transport_residual_points"],
            p1_calibration["worst_observed_transport_residual_points"],
            rtol=0.0,
            atol=1e-15,
        ),
        "minimum_calibrated_delta": np.isclose(
            policy["minimum_calibrated_expected_point_delta_inclusive"],
            p1_calibration["minimum_calibrated_expected_points_delta"],
            rtol=0.0,
            atol=1e-15,
        ),
        "minimum_raw_delta": np.isclose(
            policy["minimum_raw_expected_point_delta_inclusive"],
            p1_calibration["minimum_uncalibrated_expected_points_delta"],
            rtol=0.0,
            atol=1e-15,
        ),
    }
    if not all(comparisons.values()):
        raise ContractError(f"root calibration mismatch: {comparisons}")
    if not AUTHORITATIVE_RESULT.is_file() or not PRIOR_RESULT.is_file():
        raise ContractError("authoritative P1 evidence is missing")
    return config


def source_training_mask(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["layer"].eq(1).to_numpy()
        & frame["e150_prediction"].eq(0).to_numpy()
    )


def calibration_eligibility(frame: pd.DataFrame) -> np.ndarray:
    return source_training_mask(frame) & frame["pmax"].ge(0.01).to_numpy()


def deployment_eligibility(frame: pd.DataFrame) -> np.ndarray:
    return calibration_eligibility(frame) & frame["station"].eq("G-ORS").to_numpy()


def chronological_inner_split(
    frame: pd.DataFrame,
    outer_train_mask: np.ndarray,
    *,
    calibration_days: int,
    purge_hours: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    timestamp = pd.to_datetime(frame["time"], utc=True)
    train_time = timestamp[outer_train_mask]
    if train_time.empty:
        raise ContractError("empty outer training interval")
    calibration_start = train_time.max() - pd.Timedelta(days=calibration_days)
    fit_end = calibration_start - pd.Timedelta(hours=purge_hours)
    fit_mask = outer_train_mask & timestamp.lt(fit_end).to_numpy()
    calibration_mask = outer_train_mask & timestamp.ge(calibration_start).to_numpy()
    if fit_mask.sum() < 1000 or calibration_mask.sum() < 1000:
        raise ContractError("inner chronological split is too small")
    return fit_mask, calibration_mask, {
        "fit_end_exclusive": fit_end.isoformat(),
        "calibration_start_inclusive": calibration_start.isoformat(),
        "outer_train_end": train_time.max().isoformat(),
    }


def day_subsample_mask(
    frame: pd.DataFrame,
    base_mask: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    local_day = (
        pd.to_datetime(frame["time"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )
    days = np.unique(local_day[base_mask])
    rng = np.random.default_rng(seed)
    count = max(1, int(np.ceil(len(days) * fraction)))
    selected = rng.choice(days, size=count, replace=False)
    return base_mask & np.isin(local_day, selected)


def build_model(family: str, seed: int, config: dict[str, Any]) -> Any:
    spec = next(item for item in config["candidates"] if item["family"] == family)
    if family == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(spec["C"]),
                class_weight="balanced",
                max_iter=int(spec["max_iter"]),
                random_state=seed,
            ),
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=float(spec["learning_rate"]),
            max_iter=int(spec["max_iter"]),
            max_leaf_nodes=int(spec["max_leaf_nodes"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            l2_regularization=float(spec["l2_regularization"]),
            random_state=seed,
        )
    raise ContractError(f"unknown model family: {family}")


def fit_seed_models(
    family: str,
    frame: pd.DataFrame,
    fit_mask: np.ndarray,
    config: dict[str, Any],
) -> list[Any]:
    x = frame[MODEL_FEATURES].to_numpy(np.float64)
    y = frame["label_base"].to_numpy(np.int8)
    source_mask = fit_mask & source_training_mask(frame)
    models = []
    for seed in config["validation"]["seeds"]:
        sampled = day_subsample_mask(
            frame,
            source_mask,
            float(config["validation"]["training_day_subsample_fraction"]),
            int(seed),
        )
        if sampled.sum() < 100 or np.unique(y[sampled]).size != 2:
            raise ContractError(f"{family} seed {seed}: insufficient two-class rows")
        model = build_model(family, int(seed), config)
        if family == "hist_gradient_boosting":
            weights = compute_sample_weight(class_weight="balanced", y=y[sampled])
            model.fit(x[sampled], y[sampled], sample_weight=weights)
        else:
            model.fit(x[sampled], y[sampled])
        models.append(model)
    return models


def model_scores(models: list[Any], x: np.ndarray) -> np.ndarray:
    return np.vstack([model.predict_proba(x)[:, 1] for model in models])


def select_threshold(
    frame: pd.DataFrame,
    calibration_mask: np.ndarray,
    ensemble_score: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    truth = frame["label_base"].to_numpy(np.int8)
    anchor = frame["e150_prediction"].to_numpy(np.int8)
    eligible = calibration_mask & calibration_eligibility(frame)
    reference_f1 = float(f1_score(truth[calibration_mask], anchor[calibration_mask]))
    best: tuple[float, float, float, int, int, int] | None = None
    for threshold in config["threshold_selection"]["score_grid"]:
        added = eligible & (ensemble_score >= float(threshold))
        additions = int(added.sum())
        if additions < int(config["threshold_selection"]["minimum_calibration_additions"]):
            continue
        tp = int((added & (truth == 1)).sum())
        fp = additions - tp
        precision = tp / additions
        if precision < float(config["threshold_selection"]["minimum_calibration_precision"]):
            continue
        prediction = anchor.copy()
        prediction[added] = 1
        candidate_f1 = float(f1_score(truth[calibration_mask], prediction[calibration_mask]))
        record = (candidate_f1 - reference_f1, precision, float(threshold), additions, tp, fp)
        if best is None or record[:3] > best[:3]:
            best = record
    if best is None:
        return {
            "threshold": 1.000001,
            "reference_f1": reference_f1,
            "candidate_f1": reference_f1,
            "delta_f1": 0.0,
            "additions": 0,
            "true_positive_additions": 0,
            "false_positive_additions": 0,
            "precision": None,
            "status": "NO_ELIGIBLE_THRESHOLD",
        }
    delta, precision, threshold, additions, tp, fp = best
    return {
        "threshold": threshold,
        "reference_f1": reference_f1,
        "candidate_f1": reference_f1 + delta,
        "delta_f1": delta,
        "additions": additions,
        "true_positive_additions": tp,
        "false_positive_additions": fp,
        "precision": precision,
        "status": "SELECTED_ON_INNER_FUTURE_BLOCK",
    }


def f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)


def day_block_bootstrap(
    frame: pd.DataFrame,
    evaluated: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    truth = frame["label_base"].to_numpy(np.int8)
    local_day = (
        pd.to_datetime(frame["time"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
    )
    groups = pd.DataFrame(
        {
            "fold": frame["fold"],
            "day": local_day,
            "truth": truth,
            "anchor": anchor,
            "candidate": candidate,
            "evaluated": evaluated,
        }
    )
    records = []
    for _, group in groups.loc[groups["evaluated"]].groupby(["fold", "day"], sort=True):
        y = group["truth"].to_numpy(np.int8)
        a = group["anchor"].to_numpy(np.int8)
        c = group["candidate"].to_numpy(np.int8)
        records.append(
            [
                int(((a == 1) & (y == 1)).sum()),
                int(((a == 1) & (y == 0)).sum()),
                int(((a == 0) & (y == 1)).sum()),
                int(((c == 1) & (y == 1)).sum()),
                int(((c == 1) & (y == 0)).sum()),
                int(((c == 0) & (y == 1)).sum()),
            ]
        )
    counts = np.asarray(records, dtype=np.int64)
    rng = np.random.default_rng(int(config["validation"]["bootstrap_seed"]))
    replicates = int(config["validation"]["bootstrap_replicates"])
    deltas = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        selected = rng.integers(0, len(counts), size=len(counts))
        total = counts[selected].sum(axis=0)
        baseline_f1 = f1_from_counts(total[0:1], total[1:2], total[2:3])[0]
        candidate_f1 = f1_from_counts(total[3:4], total[4:5], total[5:6])[0]
        deltas[replicate] = candidate_f1 - baseline_f1
    return {
        "method": "KST calendar-day block bootstrap over fold-day blocks",
        "blocks": int(len(counts)),
        "replicates": replicates,
        "mean_delta_f1": float(deltas.mean()),
        "ci90_low": float(np.quantile(deltas, 0.05)),
        "ci90_high": float(np.quantile(deltas, 0.95)),
        "probability_improved": float(np.mean(deltas > 0.0)),
    }


def score_record(
    name: str,
    frame: pd.DataFrame,
    candidate: np.ndarray,
    seed_candidates: list[np.ndarray],
    thresholds: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    truth = frame["label_base"].to_numpy(np.int8)
    anchor = frame["e150_prediction"].to_numpy(np.int8)
    evaluated = frame["fold"].isin(FOLD_ORDER[1:]).to_numpy()
    reference_f1 = float(f1_score(truth[evaluated], anchor[evaluated]))
    candidate_f1 = float(f1_score(truth[evaluated], candidate[evaluated]))
    delta_f1 = candidate_f1 - reference_f1
    by_fold = {}
    for fold in FOLD_ORDER[1:]:
        mask = frame["fold"].eq(fold).to_numpy()
        additions = mask & (candidate == 1) & (anchor == 0)
        fold_reference = float(f1_score(truth[mask], anchor[mask]))
        fold_candidate = float(f1_score(truth[mask], candidate[mask]))
        by_fold[fold] = {
            "rows": int(mask.sum()),
            "additions": int(additions.sum()),
            "true_positive_additions": int((additions & (truth == 1)).sum()),
            "false_positive_additions": int((additions & (truth == 0)).sum()),
            "anchor_removals": 0,
            "reference_f1": fold_reference,
            "candidate_f1": fold_candidate,
            "delta_f1": fold_candidate - fold_reference,
        }
    additions = evaluated & (candidate == 1) & (anchor == 0)
    seed_deltas = []
    for seed_prediction in seed_candidates:
        seed_f1 = float(f1_score(truth[evaluated], seed_prediction[evaluated]))
        seed_deltas.append(seed_f1 - reference_f1)
    bootstrap = day_block_bootstrap(frame, evaluated, anchor, candidate, config)
    policy = config["decision_policy"]
    raw_points = delta_f1 * float(policy["score_points_per_f1"])
    calibrated_points = raw_points + float(
        policy["worst_observed_public_transport_residual_points"]
    )
    gates = {
        "positive_additions": int(additions.sum()) > 0,
        "anchor_removals_zero": True,
        "raw_expected_points_delta_at_least_calibrated_floor": raw_points
        >= float(policy["minimum_raw_expected_point_delta_inclusive"]),
        "calibrated_expected_points_delta_at_least_0_01": calibrated_points
        >= float(policy["minimum_calibrated_expected_point_delta_inclusive"]),
        "worst_forward_block_noncatastrophic": min(
            item["delta_f1"] for item in by_fold.values()
        )
        >= float(policy["worst_forward_block_delta_f1_minimum"]),
        "bootstrap_probability_stable": bootstrap["probability_improved"]
        >= float(policy["bootstrap_probability_improved_minimum"]),
        "bootstrap_lower_bound_noncatastrophic": bootstrap["ci90_low"]
        >= float(policy["bootstrap_ci90_low_minimum"]),
        "all_seed_deltas_nonnegative": min(seed_deltas)
        >= float(policy["seed_delta_f1_minimum"]),
    }
    return {
        "name": name,
        "past_only": True,
        "add_only": True,
        "forward_test_rows": int(evaluated.sum()),
        "additions": int(additions.sum()),
        "true_positive_additions": int((additions & (truth == 1)).sum()),
        "false_positive_additions": int((additions & (truth == 0)).sum()),
        "anchor_removals": 0,
        "reference_f1": reference_f1,
        "candidate_f1": candidate_f1,
        "delta_f1": delta_f1,
        "raw_expected_points_delta": raw_points,
        "worst_observed_transport_residual_points": float(
            policy["worst_observed_public_transport_residual_points"]
        ),
        "calibrated_conservative_expected_points_delta": calibrated_points,
        "thresholds": thresholds,
        "seed_delta_f1": seed_deltas,
        "by_fold": by_fold,
        "day_block_bootstrap": bootstrap,
        "gates": gates,
        "strict_internal_pass": bool(all(gates.values())),
    }


def historical_evaluation(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    n_rows = len(frame)
    anchor = frame["e150_prediction"].to_numpy(np.int8)
    x = frame[MODEL_FEATURES].to_numpy(np.float64)
    ensemble_predictions = {family: anchor.copy() for family in FAMILIES}
    seed_predictions = {
        family: [anchor.copy() for _ in config["validation"]["seeds"]]
        for family in FAMILIES
    }
    threshold_records: dict[str, dict[str, Any]] = {family: {} for family in FAMILIES}
    fit_count = 0
    for outer_index, outer in enumerate(config["validation"]["outer_forward_tests"], start=1):
        outer_train = frame["fold"].isin(outer["train_folds"]).to_numpy()
        outer_test = frame["fold"].eq(outer["test_fold"]).to_numpy()
        fit_mask, calibration_mask, split_receipt = chronological_inner_split(
            frame,
            outer_train,
            calibration_days=int(config["validation"]["inner_calibration_days"]),
            purge_hours=int(config["validation"]["purge_hours"]),
        )
        family_scores: dict[str, np.ndarray] = {}
        family_seed_scores: dict[str, np.ndarray] = {}
        for family in FAMILIES:
            models = fit_seed_models(family, frame, fit_mask, config)
            fit_count += len(models)
            scores = model_scores(models, x)
            ensemble_score = scores.mean(axis=0)
            selected = select_threshold(
                frame, calibration_mask, ensemble_score, config
            )
            threshold = float(selected["threshold"])
            eligible_test = outer_test & deployment_eligibility(frame)
            additions = eligible_test & (ensemble_score >= threshold)
            ensemble_predictions[family][outer_test] = np.maximum(
                anchor[outer_test], additions[outer_test]
            )
            for seed_index in range(len(models)):
                seed_additions = eligible_test & (scores[seed_index] >= threshold)
                seed_predictions[family][seed_index][outer_test] = np.maximum(
                    anchor[outer_test], seed_additions[outer_test]
                )
            family_scores[family] = ensemble_score
            family_seed_scores[family] = scores
            threshold_records[family][outer["test_fold"]] = {
                **selected,
                "inner_split": split_receipt,
                "outer_train_folds": outer["train_folds"],
                "outer_test_fold": outer["test_fold"],
                "outer_test_target_eligible_rows": int(eligible_test.sum()),
            }
        update_progress(
            {
                "experiment_id": EXPERIMENT_ID,
                "phase": "historical_forward_validation",
                "completed_outer_tests": outer_index,
                "total_outer_tests": len(config["validation"]["outer_forward_tests"]),
                "fit_count": fit_count,
                "performance_withheld_until_terminal": True,
            }
        )
    consensus = anchor.copy()
    consensus_seeds = [anchor.copy() for _ in config["validation"]["seeds"]]
    for fold in FOLD_ORDER[1:]:
        fold_mask = frame["fold"].eq(fold).to_numpy()
        both = (
            (ensemble_predictions["logistic"] == 1)
            & (ensemble_predictions["hist_gradient_boosting"] == 1)
            & (anchor == 0)
        )
        consensus[fold_mask] = np.maximum(anchor[fold_mask], both[fold_mask])
        for seed_index in range(len(consensus_seeds)):
            seed_both = (
                (seed_predictions["logistic"][seed_index] == 1)
                & (seed_predictions["hist_gradient_boosting"][seed_index] == 1)
                & (anchor == 0)
            )
            consensus_seeds[seed_index][fold_mask] = np.maximum(
                anchor[fold_mask], seed_both[fold_mask]
            )
    records = [
        score_record(
            CANDIDATE_NAMES[family],
            frame,
            ensemble_predictions[family],
            seed_predictions[family],
            threshold_records[family],
            config,
        )
        for family in FAMILIES
    ]
    records.append(
        score_record(
            CANDIDATE_NAMES["consensus"],
            frame,
            consensus,
            consensus_seeds,
            {"component_thresholds": threshold_records},
            config,
        )
    )
    predictions = {
        "ensemble": ensemble_predictions,
        "seed": seed_predictions,
        "consensus": consensus,
        "consensus_seed": consensus_seeds,
    }
    if any(len(value) != n_rows for value in ensemble_predictions.values()):
        raise ContractError("historical prediction length mismatch")
    return records, predictions, fit_count


def final_models_and_thresholds(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, list[Any]], dict[str, dict[str, Any]], int]:
    all_history = np.ones(len(frame), dtype=bool)
    fit_mask, calibration_mask, split_receipt = chronological_inner_split(
        frame,
        all_history,
        calibration_days=int(config["validation"]["inner_calibration_days"]),
        purge_hours=int(config["validation"]["purge_hours"]),
    )
    x = frame[MODEL_FEATURES].to_numpy(np.float64)
    models_by_family = {}
    thresholds = {}
    fit_count = 0
    for family in FAMILIES:
        models = fit_seed_models(family, frame, fit_mask, config)
        fit_count += len(models)
        scores = model_scores(models, x).mean(axis=0)
        thresholds[family] = {
            **select_threshold(frame, calibration_mask, scores, config),
            "inner_split": split_receipt,
        }
        models_by_family[family] = models
    return models_by_family, thresholds, fit_count


def materialize_passes(
    frame: pd.DataFrame,
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    passes = [record for record in records if record["strict_internal_pass"]]
    if not passes:
        return [], {"official_covariate_reads": 0, "paths": []}, 0
    if DELIVERY.exists():
        raise FileExistsError(f"delivery path already exists: {DELIVERY}")
    models_by_family, thresholds, deployment_model_fits = final_models_and_thresholds(
        frame, config
    )
    raw_test, official = prior_p1.official_frame(MODEL_FEATURES)
    official_x = official[MODEL_FEATURES].to_numpy(np.float64)
    anchor = official["e150_prediction"].to_numpy(np.int8)
    eligible = deployment_eligibility(official)
    family_additions = {}
    for family in FAMILIES:
        scores = model_scores(models_by_family[family], official_x).mean(axis=0)
        family_additions[family] = eligible & (
            scores >= float(thresholds[family]["threshold"])
        )
    candidate_additions = {
        CANDIDATE_NAMES["logistic"]: family_additions["logistic"],
        CANDIDATE_NAMES["hist_gradient_boosting"]: family_additions[
            "hist_gradient_boosting"
        ],
        CANDIDATE_NAMES["consensus"]: family_additions["logistic"]
        & family_additions["hist_gradient_boosting"],
    }
    outputs = []
    for record in passes[: int(config["official_policy"]["maximum_materialized_candidates"])]:
        additions = candidate_additions[record["name"]]
        label = np.maximum(anchor, additions).astype(np.int8)
        submission = raw_test[P1_KEYS].copy()
        submission["label"] = label
        if (
            len(submission) != 169_011
            or submission.duplicated(P1_KEYS).any()
            or not set(submission["label"].unique()).issubset({0, 1})
            or int(((anchor == 1) & (label == 0)).sum()) != 0
        ):
            raise ContractError(f"official submission contract failed: {record['name']}")
        path = DELIVERY / record["name"] / "P1_submission.csv"
        path.parent.mkdir(parents=True, exist_ok=False)
        submission.to_csv(path, index=False, lineterminator="\n")
        outputs.append(
            {
                "name": record["name"],
                "path": str(path),
                "rows": int(len(submission)),
                "sha256": sha256_file(path),
                "positive_rows": int(label.sum()),
                "additions_vs_champion": int((additions & (anchor == 0)).sum()),
                "anchor_removals": 0,
                "final_thresholds": thresholds,
                "upload_performed": False,
            }
        )
    return (
        outputs,
        {
            "official_covariate_reads": 1,
            "paths": [
                str(prior_p1.P1_DATA / "test.csv"),
                str(prior_p1.P1_CHAMPION),
                str(prior_p1.P1_E150_DEPLOY),
            ],
            "hidden_truth_reads": 0,
            "uploads": 0,
        },
        deployment_model_fits + 2,
    )


def independent_qa(result: dict[str, Any]) -> dict[str, Any]:
    policy = result["decision_policy"]
    checks: dict[str, bool] = {
        "three_candidates_max": 1 <= len(result["candidates"]) <= 3,
        "historical_fit_budget": result["historical_fit_count"] <= 12,
        "total_fit_budget": result["fit_count"] <= 20,
        "all_past_only": all(item["past_only"] for item in result["candidates"]),
        "all_add_only": all(item["add_only"] for item in result["candidates"]),
        "all_anchor_removals_zero": all(
            item["anchor_removals"] == 0 for item in result["candidates"]
        ),
        "calibration_inclusive_gate_exact": np.isclose(
            policy["minimum_calibrated_expected_point_delta_inclusive"], 0.01
        ),
        "raw_gate_exact": np.isclose(
            policy["minimum_raw_expected_point_delta_inclusive"],
            0.015383691373120248,
        ),
        "transport_residual_exact": np.isclose(
            policy["worst_observed_public_transport_residual_points"],
            -0.005383691373120247,
        ),
        "only_strict_passes_materialized": {item["name"] for item in result["outputs"]}
        == {
            item["name"]
            for item in result["candidates"]
            if item["strict_internal_pass"]
        },
        "hidden_truth_reads_zero": result["operations"]["hidden_truth_reads"] == 0,
        "uploads_zero": result["operations"]["uploads"] == 0,
        "prior_tie_not_counted": result["prior_tie_disposition"]
        == "EXCLUDED_FROM_PASS_COUNT",
    }
    for output in result["outputs"]:
        checks[f"{output['name']}_exists"] = Path(output["path"]).is_file()
        checks[f"{output['name']}_hash"] = sha256_file(Path(output["path"])) == output[
            "sha256"
        ]
        checks[f"{output['name']}_rows"] = output["rows"] == 169_011
        checks[f"{output['name']}_anchor_removals"] = output["anchor_removals"] == 0
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def validate_only() -> dict[str, Any]:
    config = load_contract()
    return {
        "status": "VALID",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "candidate_count": len(config["candidates"]),
        "historical_fit_budget": config["fit_budget"]["historical_maximum"],
        "total_fit_budget": config["fit_budget"]["total_maximum"],
    }


def execute() -> dict[str, Any]:
    if ARTIFACT.exists() or REPORT.exists():
        raise FileExistsError("exactly-once artifact/report path already exists")
    config = load_contract()
    ARTIFACT.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    started = time.perf_counter()
    write_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "pid": os.getpid(),
            "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "historical_fit_budget": 12,
            "total_fit_budget": 20,
        },
    )
    update_progress(
        {
            "experiment_id": EXPERIMENT_ID,
            "phase": "loading_historical_oof_only",
            "fit_count": 0,
            "performance_withheld_until_terminal": True,
        }
    )
    historical, _ = prior_cycle.p1_frame()
    historical, actual_features = prior_p1.add_causal_features(historical)
    missing = sorted(set(MODEL_FEATURES) - set(actual_features))
    if missing:
        raise ContractError(f"frozen model features missing: {missing}")
    records, _, historical_fits = historical_evaluation(historical, config)
    update_progress(
        {
            "experiment_id": EXPERIMENT_ID,
            "phase": "internal_gate_complete",
            "fit_count": historical_fits,
            "strict_pass_count": sum(item["strict_internal_pass"] for item in records),
            "performance_withheld_until_terminal": True,
        }
    )
    outputs, official_access, deployment_fits = materialize_passes(
        historical, records, config
    )
    result: dict[str, Any] = {
        "schema_version": "p1.public_transport_repair_cycle.20260831.v5",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_NOT_UPLOADED",
        "runtime_seconds": time.perf_counter() - started,
        "decision_policy": config["decision_policy"],
        "validation_contract": config["validation"],
        "transport_scope": config["transport_scope"],
        "candidate_count": len(records),
        "pass_count": sum(item["strict_internal_pass"] for item in records),
        "prior_tie_disposition": "EXCLUDED_FROM_PASS_COUNT",
        "candidates": records,
        "outputs": outputs,
        "historical_fit_count": historical_fits,
        "deployment_fit_count": deployment_fits,
        "fit_count": historical_fits + deployment_fits,
        "hashes": {
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "root_calibration_sha256": sha256_file(CALIBRATION_PATH),
            "authoritative_official_results_sha256": sha256_file(AUTHORITATIVE_RESULT),
            "prior_v4_result_sha256": sha256_file(PRIOR_RESULT),
        },
        "operations": {
            **official_access,
            "hidden_truth_reads": 0,
            "uploads": 0,
        },
        "source_provenance": [
            str(AUTHORITATIVE_RESULT),
            str(CALIBRATION_PATH),
            str(PRIOR_RESULT),
            "historical P1 Q2/Q3/Q4 OOF only before internal PASS",
        ],
    }
    result["independent_qa"] = independent_qa(result)
    write_json(ARTIFACT / "result.json", result)
    write_json(REPORT / "independent-qa.json", result["independent_qa"])
    if outputs:
        write_json(DELIVERY / "SET_MANIFEST.json", result)
    update_progress(
        {
            "experiment_id": EXPERIMENT_ID,
            "phase": "terminal",
            "status": result["status"],
            "fit_count": result["fit_count"],
            "pass_count": result["pass_count"],
            "outputs": len(outputs),
            "performance_available_in_result": True,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_only(), ensure_ascii=False, indent=2))
        return 0
    try:
        result = execute()
    except Exception as exc:
        if ARTIFACT.exists():
            failure = {
                "experiment_id": EXPERIMENT_ID,
                "status": "TERMINAL_TECHNICAL_FAILURE",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "automatic_restart_allowed": False,
                "hidden_truth_reads": 0,
                "uploads": 0,
            }
            write_json(ARTIFACT / "terminal_failure.json", failure)
            update_progress(failure)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
