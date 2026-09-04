"""Audit a deployment-parity, consensus type veto for P1 MSTCN additions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_mstcn_segment_precision_router_retroaudit_20260829_v1 as base  # noqa: E402

EXPERIMENT_ID = "p1_mstcn_deployable_type_veto_stability_20260829_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
OUTPUT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
CATEGORICAL = ["station", "layer"]
NUMERIC = [
    "log_length",
    "row_mean",
    "row_min",
    "row_max",
    "row_q10",
    "row_median",
    "row_q90",
    "row_std",
    "boundary_start_mean",
    "boundary_start_max",
    "boundary_end_mean",
    "boundary_end_max",
    *base.TYPE_NUMERIC,
]
FEATURES = [*NUMERIC, *CATEGORICAL]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def truth_event_groups(bundle: base.FoldBundle) -> np.ndarray:
    frame = bundle.frame
    labels = bundle.labels.astype(bool)
    parsed = pd.to_datetime(frame["time"], utc=True)
    continuation = (
        pd.Series(labels).shift(fill_value=False).to_numpy()
        & labels
        & frame["station"].astype(str).eq(frame["station"].astype(str).shift()).to_numpy()
        & frame["layer"].eq(frame["layer"].shift()).to_numpy()
        & parsed.diff().eq(pd.Timedelta(minutes=10)).to_numpy()
    )
    event_id = np.cumsum(labels & ~continuation)
    groups: list[str] = []
    for index, positions in enumerate(bundle.segment_indices):
        overlap = sorted(set(int(value) for value in event_id[positions][labels[positions]]))
        groups.append(
            "event:" + "+".join(str(value) for value in overlap)
            if overlap
            else f"negative-segment:{index}"
        )
    return np.asarray(groups, dtype=object)


def marginal_utility(bundle: base.FoldBundle) -> np.ndarray:
    incumbent = base.metric(bundle.labels, bundle.incumbent)
    values: list[float] = []
    for positions in bundle.segment_indices:
        candidate = bundle.incumbent.copy()
        candidate[positions] = 1
        values.append(float(base.metric(bundle.labels, candidate)["f1"] - incumbent["f1"]))
    return np.asarray(values, dtype=np.float64)


def _transformer() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        remainder="drop",
    )


def classifier(config: dict) -> Pipeline:
    specification = config["models"]["beneficial_classifier"]
    return Pipeline(
        [
            ("transform", _transformer()),
            (
                "model",
                LogisticRegression(
                    C=float(specification["C"]),
                    class_weight="balanced",
                    solver=str(specification["solver"]),
                    max_iter=int(specification["maximum_iterations"]),
                    random_state=int(config["validation"]["bootstrap_seed"]),
                ),
            ),
        ]
    )


def regressor(config: dict) -> Pipeline:
    specification = config["models"]["marginal_utility_regressor"]
    return Pipeline(
        [
            ("transform", _transformer()),
            ("model", Ridge(alpha=float(specification["alpha"]))),
        ]
    )


def candidate_from_acceptance(
    bundle: base.FoldBundle, acceptance: np.ndarray
) -> tuple[np.ndarray, dict[str, float | int]]:
    prediction = bundle.incumbent.copy()
    for accepted, positions in zip(acceptance, bundle.segment_indices, strict=True):
        if accepted:
            prediction[positions] = 1
    addition = (prediction == 1) & (bundle.incumbent == 0)
    return prediction, {
        "accepted_segments": int(np.sum(acceptance)),
        "accepted_rows": int(np.sum(addition)),
        "accepted_true_positive_rows": int(np.sum(bundle.labels[addition])),
        "accepted_false_positive_rows": int(np.sum(1 - bundle.labels[addition])),
        "accepted_row_precision": (
            float(np.mean(bundle.labels[addition])) if np.any(addition) else 1.0
        ),
    }


def grouped_cross_validation(
    training: base.FoldBundle, config: dict
) -> tuple[dict, np.ndarray]:
    frame = training.segments
    targets = frame["beneficial"].to_numpy(np.int8)
    utility = marginal_utility(training)
    groups = truth_event_groups(training)
    probability = np.full(len(frame), np.nan, dtype=np.float64)
    predicted_utility = np.full(len(frame), np.nan, dtype=np.float64)
    skipped_groups = 0
    for group in np.unique(groups):
        validation = groups == group
        fit = ~validation
        if np.unique(targets[fit]).size < 2:
            skipped_groups += 1
            continue
        cls = classifier(config)
        reg = regressor(config)
        cls.fit(frame.loc[fit, FEATURES], targets[fit])
        reg.fit(frame.loc[fit, FEATURES], utility[fit])
        probability[validation] = cls.predict_proba(frame.loc[validation, FEATURES])[:, 1]
        predicted_utility[validation] = reg.predict(frame.loc[validation, FEATURES])
    complete = np.isfinite(probability) & np.isfinite(predicted_utility)
    require(np.all(complete), f"incomplete grouped CV predictions; skipped={skipped_groups}")
    cls_accept = probability >= float(
        config["models"]["beneficial_classifier"]["acceptance_probability"]
    )
    reg_accept = predicted_utility > float(
        config["models"]["marginal_utility_regressor"]["acceptance_utility"]
    )
    consensus = cls_accept & reg_accept
    candidate, additions = candidate_from_acceptance(training, consensus)
    incumbent = base.metric(training.labels, training.incumbent)
    scored = base.metric(training.labels, candidate)
    return (
        {
            "groups": int(np.unique(groups).size),
            "segments": int(len(frame)),
            "beneficial_segments": int(targets.sum()),
            "beneficial_roc_auc": float(roc_auc_score(targets, probability)),
            "beneficial_balanced_accuracy": float(
                balanced_accuracy_score(targets, cls_accept.astype(np.int8))
            ),
            "consensus": additions,
            "incumbent": incumbent,
            "candidate": scored,
            "delta_f1_vs_incumbent": float(scored["f1"] - incumbent["f1"]),
        },
        consensus,
    )


def fit_and_evaluate_q4(
    training: base.FoldBundle, evaluation: base.FoldBundle, config: dict
) -> tuple[dict, np.ndarray]:
    targets = training.segments["beneficial"].to_numpy(np.int8)
    utility = marginal_utility(training)
    cls = classifier(config)
    reg = regressor(config)
    cls.fit(training.segments[FEATURES], targets)
    reg.fit(training.segments[FEATURES], utility)
    probability = cls.predict_proba(evaluation.segments[FEATURES])[:, 1]
    predicted_utility = reg.predict(evaluation.segments[FEATURES])
    cls_accept = probability >= float(
        config["models"]["beneficial_classifier"]["acceptance_probability"]
    )
    reg_accept = predicted_utility > float(
        config["models"]["marginal_utility_regressor"]["acceptance_utility"]
    )
    consensus = cls_accept & reg_accept
    candidate, additions = candidate_from_acceptance(evaluation, consensus)
    incumbent = base.metric(evaluation.labels, evaluation.incumbent)
    raw = base.metric(evaluation.labels, evaluation.raw_candidate)
    scored = base.metric(evaluation.labels, candidate)
    return (
        {
            "training_segments": int(len(training.segments)),
            "evaluation_segments": int(len(evaluation.segments)),
            "classifier_accepts": int(cls_accept.sum()),
            "regressor_accepts": int(reg_accept.sum()),
            "consensus": additions,
            "incumbent": incumbent,
            "raw_e150": raw,
            "candidate": scored,
            "delta_f1_vs_incumbent": float(scored["f1"] - incumbent["f1"]),
            "delta_f1_vs_raw_e150": float(scored["f1"] - raw["f1"]),
            "classifier_probability_summary": {
                "minimum": float(np.min(probability)),
                "median": float(np.median(probability)),
                "maximum": float(np.max(probability)),
            },
            "predicted_utility_summary": {
                "minimum": float(np.min(predicted_utility)),
                "median": float(np.median(predicted_utility)),
                "maximum": float(np.max(predicted_utility)),
            },
        },
        consensus,
    )


def bootstrap_q4(
    training: base.FoldBundle, evaluation: base.FoldBundle, config: dict
) -> dict:
    replicates = int(config["validation"]["bootstrap_replicates"])
    rng = np.random.default_rng(int(config["validation"]["bootstrap_seed"]))
    groups = truth_event_groups(training)
    unique_groups = np.unique(groups)
    targets = training.segments["beneficial"].to_numpy(np.int8)
    utility = marginal_utility(training)
    probability = np.zeros((replicates, len(evaluation.segments)), dtype=np.float64)
    predicted_utility = np.zeros_like(probability)
    completed = 0
    attempts = 0
    while completed < replicates:
        attempts += 1
        require(attempts <= replicates * 20, "bootstrap class balance could not be satisfied")
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_positions = np.concatenate(
            [np.flatnonzero(groups == group) for group in sampled_groups]
        )
        if np.unique(targets[sampled_positions]).size < 2:
            continue
        cls = classifier(config)
        reg = regressor(config)
        cls.fit(training.segments.iloc[sampled_positions][FEATURES], targets[sampled_positions])
        reg.fit(training.segments.iloc[sampled_positions][FEATURES], utility[sampled_positions])
        probability[completed] = cls.predict_proba(evaluation.segments[FEATURES])[:, 1]
        predicted_utility[completed] = reg.predict(evaluation.segments[FEATURES])
        completed += 1
    cls_accept = probability >= float(
        config["models"]["beneficial_classifier"]["acceptance_probability"]
    )
    reg_accept = predicted_utility > float(
        config["models"]["marginal_utility_regressor"]["acceptance_utility"]
    )
    consensus = cls_accept & reg_accept
    frequencies = np.mean(consensus, axis=0)
    return {
        "replicates": replicates,
        "resampling_attempts": attempts,
        "evaluation_segments": int(len(evaluation.segments)),
        "zero_acceptance_replicates": int(np.sum(np.sum(consensus, axis=1) == 0)),
        "zero_acceptance_fraction": float(np.mean(np.sum(consensus, axis=1) == 0)),
        "maximum_segment_consensus_acceptance_frequency": float(np.max(frequencies)),
        "segment_consensus_acceptance_frequency_quantiles": [
            float(value) for value in np.quantile(frequencies, [0.0, 0.25, 0.5, 0.75, 1.0])
        ],
        "classifier_probability_q95_max_across_segments": float(
            np.max(np.quantile(probability, 0.95, axis=0))
        ),
        "utility_q95_max_across_segments": float(
            np.max(np.quantile(predicted_utility, 0.95, axis=0))
        ),
    }


def execute() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    require(config["experiment_id"] == EXPERIMENT_ID, "config id")
    require(config["official_test_reads_allowed"] == 0, "official read contract")
    require(config["submission_creation_allowed"] is False, "submission contract")
    require("incumbent_probability_mean" not in FEATURES, "deployment feature leak")
    require("incumbent_probability_max" not in FEATURES, "deployment feature leak")
    bundles = base.load_bundles()
    q3 = bundles[str(config["training_fold"])]
    q4 = bundles[str(config["evaluation_fold"])]
    require(q3.type_probability is not None and q4.type_probability is not None, "type features")
    q3_cv, _q3_acceptance = grouped_cross_validation(q3, config)
    q4_result, _q4_acceptance = fit_and_evaluate_q4(q3, q4, config)
    stability = bootstrap_q4(q3, q4, config)
    threshold = float(
        config["diagnostic_gate"]["q4_max_consensus_bootstrap_acceptance_frequency"]
    )
    gate_checks = {
        "q3_consensus_nonnegative_vs_incumbent": q3_cv["delta_f1_vs_incumbent"] >= 0.0,
        "q4_consensus_exact_incumbent_fallback": q4_result["candidate"]
        == q4_result["incumbent"],
        "q4_consensus_improves_raw_e150": q4_result["delta_f1_vs_raw_e150"] > 0.0,
        "q4_bootstrap_acceptance_frequency": stability[
            "maximum_segment_consensus_acceptance_frequency"
        ]
        <= threshold,
        "deployment_feature_parity": "incumbent_probability_mean" not in FEATURES
        and "incumbent_probability_max" not in FEATURES,
        "no_official_test_or_submission_access": True,
    }
    passed = all(gate_checks.values())
    return {
        "schema_version": "p1.mstcn_deployable_type_veto_stability.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "status": "PASS_SHADOW_AUDIT_ELIGIBLE" if passed else "NO_GO_STABILITY_GATE",
        "passed_all_diagnostic_gates": passed,
        "feature_contract": {
            "numeric": NUMERIC,
            "categorical": CATEGORICAL,
            "incumbent_probability_features": 0,
        },
        "q3_grouped_cross_validation": q3_cv,
        "q4_once": q4_result,
        "q4_group_bootstrap": stability,
        "gate_checks": gate_checks,
        "input_hashes": {
            "config": sha256(CONFIG_PATH),
            "prior_runner": sha256(Path(base.__file__)),
            **{fold: sha256(path) for fold, path in base.FOLD_PATHS.items()},
        },
        "operation_counters": {
            "official_test_rows_read": 0,
            "submission_files_created": 0,
            "uploads": 0,
            "fixed_model_fits": 2,
            "group_cv_model_fits": int(2 * q3_cv["groups"]),
            "bootstrap_model_fits": int(2 * stability["replicates"]),
        },
        "claim_limit": "Retrospective stability only; passing permits a label-free shadow audit, not promotion or upload.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(CONFIG_PATH.exists(), "missing config")
    if not args.execute:
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "READY_STABILITY_ONLY",
                    "official_test_rows_read": 0,
                    "submission_files_created": 0,
                },
                indent=2,
            )
        )
        return
    result = execute()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / "result.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed_all_diagnostic_gates": result["passed_all_diagnostic_gates"],
                "q3": result["q3_grouped_cross_validation"],
                "q4": result["q4_once"],
                "stability": result["q4_group_bootstrap"],
                "gate_checks": result["gate_checks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
