"""Exactly-once P1 calibrated bagged transport-repair cycle.

This successor to v5 replaces arbitrary abstention tolerances with the exact
add-only marginal F1 rule: add a calibrated candidate only when p > F1_ref/2.
All calibration is performed on a later inner historical block separated by a
24-hour purge.  Official covariates remain unopened until every frozen gate
passes; hidden truth and upload are out of scope.
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v5 as base  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v6"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
DELIVERY = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260831_P1_PUBLIC_TRANSPORT_REPAIR_CYCLE_V6"
)
FAMILIES = ("row", "group_shrunk")
NAMES = {
    "row": "P1_1_CALIBRATED_BAGGED_ET_ROW",
    "group_shrunk": "P1_2_CALIBRATED_BAGGED_ET_GROUP_SHRUNK",
    "consensus": "P1_3_CALIBRATED_BAGGED_ET_CONSENSUS",
}


class ContractError(RuntimeError):
    """Frozen contract violation."""


def native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        json.dump(native(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def progress(payload: dict[str, Any]) -> None:
    write_json(ARTIFACT / "progress.json", payload, exclusive=False)


def load_contract() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    calibration = json.loads(base.CALIBRATION_PATH.read_text(encoding="utf-8"))["gates"]["P1"]
    policy = config["decision_policy"]
    checks = [
        np.isclose(
            policy["worst_observed_public_transport_residual_points"],
            calibration["worst_observed_transport_residual_points"],
            atol=1e-15,
        ),
        np.isclose(
            policy["minimum_calibrated_expected_point_delta_inclusive"],
            calibration["minimum_calibrated_expected_points_delta"],
            atol=1e-15,
        ),
        np.isclose(
            policy["minimum_raw_expected_point_delta_inclusive"],
            calibration["minimum_uncalibrated_expected_points_delta"],
            atol=1e-15,
        ),
    ]
    if not all(checks):
        raise ContractError("root Public-transport calibration mismatch")
    return config


def station_group_shrinkage_weights(
    frame: pd.DataFrame,
    selected: np.ndarray,
    y: np.ndarray,
    strength: float,
) -> np.ndarray:
    selected_frame = frame.loc[selected, ["station", "layer"]]
    class_weight = compute_sample_weight("balanced", y[selected])
    group_size = selected_frame.groupby(["station", "layer"], observed=True)[
        "station"
    ].transform("size").to_numpy(float)
    median_size = float(np.median(np.unique(group_size)))
    group_factor = np.clip(median_size / group_size, 0.5, 2.0)
    shrunk = (1.0 - strength) + strength * group_factor
    weight = class_weight * shrunk
    return weight / float(np.mean(weight))


def fit_calibrated_ensemble(
    family: str,
    frame: pd.DataFrame,
    fit_mask: np.ndarray,
    calibration_mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[tuple[Any, IsotonicRegression]], dict[str, Any], int]:
    x = frame[base.MODEL_FEATURES].to_numpy(np.float64)
    y = frame["label_base"].to_numpy(np.int8)
    source_fit = fit_mask & base.source_training_mask(frame)
    calibration = calibration_mask & base.calibration_eligibility(frame)
    if calibration.sum() < 100 or np.unique(y[calibration]).size != 2:
        raise ContractError(f"{family}: insufficient inner calibration support")
    spec = config["model"]
    ensemble: list[tuple[Any, IsotonicRegression]] = []
    calibrated_calibration = []
    for seed in config["validation"]["seeds"]:
        sampled = base.day_subsample_mask(
            frame,
            source_fit,
            float(config["validation"]["training_day_subsample_fraction"]),
            int(seed),
        )
        if sampled.sum() < 100 or np.unique(y[sampled]).size != 2:
            raise ContractError(f"{family} seed {seed}: insufficient two-class fit rows")
        model = ExtraTreesClassifier(
            n_estimators=int(spec["n_estimators"]),
            max_depth=int(spec["max_depth"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            max_features=float(spec["max_features"]),
            random_state=int(seed),
            n_jobs=int(spec["n_jobs_per_seed"]),
        )
        if family == "row":
            weights = compute_sample_weight("balanced", y[sampled])
        else:
            weights = station_group_shrinkage_weights(
                frame,
                sampled,
                y,
                float(config["transport_scope"]["station_layer_group_weight_shrinkage"]),
            )
        model.fit(x[sampled], y[sampled], sample_weight=weights)
        raw_calibration = model.predict_proba(x[calibration])[:, 1]
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(raw_calibration, y[calibration])
        calibrated_calibration.append(calibrator.predict(raw_calibration))
        ensemble.append((model, calibrator))
    calibrated_mean = np.mean(calibrated_calibration, axis=0)
    reference_f1 = float(f1_score(y[calibration_mask], frame.loc[calibration_mask, "e150_prediction"]))
    threshold = reference_f1 / 2.0
    additions = calibrated_mean > threshold
    tp = int((additions & (y[calibration] == 1)).sum())
    fp = int(additions.sum()) - tp
    receipt = {
        "calibration_rows": int(calibration.sum()),
        "reference_f1": reference_f1,
        "theoretical_add_only_threshold_strict": threshold,
        "rule": "calibrated_probability > inner_reference_f1 / 2",
        "calibration_additions": int(additions.sum()),
        "calibration_true_positive_additions": tp,
        "calibration_false_positive_additions": fp,
        "calibration_precision_observed_not_gated": (
            tp / int(additions.sum()) if additions.any() else None
        ),
    }
    return ensemble, receipt, len(ensemble) * 2


def calibrated_scores(
    ensemble: list[tuple[Any, IsotonicRegression]],
    x: np.ndarray,
) -> np.ndarray:
    return np.vstack(
        [
            calibrator.predict(model.predict_proba(x)[:, 1])
            for model, calibrator in ensemble
        ]
    )


def evaluate(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    x = frame[base.MODEL_FEATURES].to_numpy(np.float64)
    anchor = frame["e150_prediction"].to_numpy(np.int8)
    ensemble_predictions = {family: anchor.copy() for family in FAMILIES}
    seed_predictions = {
        family: [anchor.copy() for _ in config["validation"]["seeds"]]
        for family in FAMILIES
    }
    receipts: dict[str, dict[str, Any]] = {family: {} for family in FAMILIES}
    fit_count = 0
    for outer_index, outer in enumerate(config["validation"]["outer_forward_tests"], start=1):
        outer_train = frame["fold"].isin(outer["train_folds"]).to_numpy()
        outer_test = frame["fold"].eq(outer["test_fold"]).to_numpy()
        fit_mask, calibration_mask, split = base.chronological_inner_split(
            frame,
            outer_train,
            calibration_days=int(config["validation"]["inner_calibration_days"]),
            purge_hours=int(config["validation"]["purge_hours"]),
        )
        for family in FAMILIES:
            ensemble, receipt, fits = fit_calibrated_ensemble(
                family, frame, fit_mask, calibration_mask, config
            )
            fit_count += fits
            scores = calibrated_scores(ensemble, x)
            threshold = float(receipt["theoretical_add_only_threshold_strict"])
            eligible = outer_test & base.deployment_eligibility(frame)
            ensemble_additions = eligible & (scores.mean(axis=0) > threshold)
            ensemble_predictions[family][outer_test] = np.maximum(
                anchor[outer_test], ensemble_additions[outer_test]
            )
            for seed_index in range(len(ensemble)):
                additions = eligible & (scores[seed_index] > threshold)
                seed_predictions[family][seed_index][outer_test] = np.maximum(
                    anchor[outer_test], additions[outer_test]
                )
            receipts[family][outer["test_fold"]] = {
                **receipt,
                "inner_split": split,
                "outer_train_folds": outer["train_folds"],
                "outer_test_fold": outer["test_fold"],
                "outer_target_eligible_rows": int(eligible.sum()),
            }
        progress(
            {
                "experiment_id": EXPERIMENT_ID,
                "phase": "historical_forward_validation",
                "completed_outer_tests": outer_index,
                "total_outer_tests": 2,
                "fit_count": fit_count,
                "performance_withheld_until_terminal": True,
            }
        )
    consensus = anchor.copy()
    consensus_seeds = [anchor.copy() for _ in config["validation"]["seeds"]]
    evaluated = frame["fold"].isin(base.FOLD_ORDER[1:]).to_numpy()
    consensus[evaluated] = np.maximum(
        anchor[evaluated],
        (
            (ensemble_predictions["row"][evaluated] == 1)
            & (ensemble_predictions["group_shrunk"][evaluated] == 1)
            & (anchor[evaluated] == 0)
        ),
    )
    for index in range(len(consensus_seeds)):
        consensus_seeds[index][evaluated] = np.maximum(
            anchor[evaluated],
            (
                (seed_predictions["row"][index][evaluated] == 1)
                & (seed_predictions["group_shrunk"][index][evaluated] == 1)
                & (anchor[evaluated] == 0)
            ),
        )
    records = [
        base.score_record(
            NAMES[family],
            frame,
            ensemble_predictions[family],
            seed_predictions[family],
            receipts[family],
            config,
        )
        for family in FAMILIES
    ]
    records.append(
        base.score_record(
            NAMES["consensus"],
            frame,
            consensus,
            consensus_seeds,
            {"component_receipts": receipts},
            config,
        )
    )
    return native(records), fit_count


def final_ensembles(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, list[tuple[Any, IsotonicRegression]]], dict[str, Any], int]:
    all_history = np.ones(len(frame), dtype=bool)
    fit_mask, calibration_mask, split = base.chronological_inner_split(
        frame,
        all_history,
        calibration_days=int(config["validation"]["inner_calibration_days"]),
        purge_hours=int(config["validation"]["purge_hours"]),
    )
    ensembles = {}
    receipts = {}
    fits = 0
    for family in FAMILIES:
        ensemble, receipt, family_fits = fit_calibrated_ensemble(
            family, frame, fit_mask, calibration_mask, config
        )
        fits += family_fits
        ensembles[family] = ensemble
        receipts[family] = {**receipt, "inner_split": split}
    return ensembles, receipts, fits


def materialize(
    frame: pd.DataFrame,
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    passes = [record for record in records if record["strict_internal_pass"]]
    if not passes:
        return [], {"official_covariate_reads": 0, "paths": []}, 0
    if DELIVERY.exists():
        raise FileExistsError(f"delivery path already exists: {DELIVERY}")
    ensembles, receipts, deployment_fits = final_ensembles(frame, config)
    raw_test, official = base.prior_p1.official_frame(base.MODEL_FEATURES)
    x = official[base.MODEL_FEATURES].to_numpy(np.float64)
    anchor = official["e150_prediction"].to_numpy(np.int8)
    eligible = base.deployment_eligibility(official)
    additions = {}
    for family in FAMILIES:
        probability = calibrated_scores(ensembles[family], x).mean(axis=0)
        additions[family] = eligible & (
            probability > float(receipts[family]["theoretical_add_only_threshold_strict"])
        )
    candidate_additions = {
        NAMES["row"]: additions["row"],
        NAMES["group_shrunk"]: additions["group_shrunk"],
        NAMES["consensus"]: additions["row"] & additions["group_shrunk"],
    }
    outputs = []
    for record in passes:
        added = candidate_additions[record["name"]]
        label = np.maximum(anchor, added).astype(np.int8)
        submission = raw_test[base.P1_KEYS].copy()
        submission["label"] = label
        if (
            len(submission) != 169_011
            or submission.duplicated(base.P1_KEYS).any()
            or int(((anchor == 1) & (label == 0)).sum()) != 0
        ):
            raise ContractError(f"submission contract failed: {record['name']}")
        path = DELIVERY / record["name"] / "P1_submission.csv"
        path.parent.mkdir(parents=True, exist_ok=False)
        submission.to_csv(path, index=False, lineterminator="\n")
        outputs.append(
            {
                "name": record["name"],
                "path": str(path),
                "rows": 169_011,
                "sha256": sha256_file(path),
                "positive_rows": int(label.sum()),
                "additions_vs_champion": int((added & (anchor == 0)).sum()),
                "anchor_removals": 0,
                "final_calibration_receipts": receipts,
                "upload_performed": False,
            }
        )
    return (
        outputs,
        {
            "official_covariate_reads": 1,
            "paths": [
                str(base.prior_p1.P1_DATA / "test.csv"),
                str(base.prior_p1.P1_CHAMPION),
                str(base.prior_p1.P1_E150_DEPLOY),
            ],
        },
        deployment_fits + 2,
    )


def independent_qa(result: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "candidate_count_1_to_3": 1 <= len(result["candidates"]) <= 3,
        "historical_fit_budget": result["historical_fit_count"] <= 24,
        "total_fit_budget": result["fit_count"] <= 38,
        "all_add_only": all(item["add_only"] for item in result["candidates"]),
        "all_anchor_removals_zero": all(
            item["anchor_removals"] == 0 for item in result["candidates"]
        ),
        "theoretical_threshold_used": all(
            "threshold" not in json.dumps(item["thresholds"])
            or "theoretical_add_only_threshold_strict" in json.dumps(item["thresholds"])
            for item in result["candidates"]
        ),
        "only_passes_materialized": {item["name"] for item in result["outputs"]}
        == {
            item["name"]
            for item in result["candidates"]
            if item["strict_internal_pass"]
        },
        "calibrated_gate_exact": np.isclose(
            result["decision_policy"]["minimum_calibrated_expected_point_delta_inclusive"],
            0.01,
        ),
        "raw_gate_exact": np.isclose(
            result["decision_policy"]["minimum_raw_expected_point_delta_inclusive"],
            0.015383691373120248,
        ),
        "official_read_only_after_pass": (
            bool(result["outputs"])
            == bool(result["operations"]["official_covariate_reads"])
        ),
        "hidden_truth_reads_zero": result["operations"]["hidden_truth_reads"] == 0,
        "uploads_zero": result["operations"]["uploads"] == 0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": native(checks)}


def validate_only() -> dict[str, Any]:
    config = load_contract()
    return {
        "status": "VALID",
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "candidate_count": len(config["candidates"]),
        "historical_fit_budget": config["fit_budget"]["historical_total_maximum"],
        "total_fit_budget": config["fit_budget"]["total_maximum"],
    }


def execute() -> dict[str, Any]:
    if ARTIFACT.exists() or REPORT.exists():
        raise FileExistsError("exactly-once v6 output path already exists")
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
            "v5_recovery_sha256": sha256_file(
                ROOT / "reports/p1_public_transport_repair_cycle_20260831_v5/recovered-result.json"
            ),
        },
    )
    progress(
        {
            "experiment_id": EXPERIMENT_ID,
            "phase": "loading_historical_oof_only",
            "fit_count": 0,
            "performance_withheld_until_terminal": True,
        }
    )
    historical, _ = base.prior_cycle.p1_frame()
    historical, actual_features = base.prior_p1.add_causal_features(historical)
    if sorted(set(base.MODEL_FEATURES) - set(actual_features)):
        raise ContractError("frozen causal feature mismatch")
    records, historical_fits = evaluate(historical, config)
    progress(
        {
            "experiment_id": EXPERIMENT_ID,
            "phase": "internal_gate_complete",
            "fit_count": historical_fits,
            "pass_count": sum(item["strict_internal_pass"] for item in records),
            "performance_withheld_until_terminal": True,
        }
    )
    outputs, official_access, deployment_fits = materialize(historical, records, config)
    result = {
        "schema_version": "p1.public_transport_repair_cycle.20260831.v6",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_NOT_UPLOADED",
        "runtime_seconds": time.perf_counter() - started,
        "decision_policy": config["decision_policy"],
        "design_change_from_v5": config["design_change_from_v5"],
        "validation_contract": config["validation"],
        "candidates": records,
        "pass_count": sum(item["strict_internal_pass"] for item in records),
        "outputs": outputs,
        "historical_fit_count": historical_fits,
        "deployment_fit_count": deployment_fits,
        "fit_count": historical_fits + deployment_fits,
        "operations": {
            **official_access,
            "hidden_truth_reads": 0,
            "uploads": 0,
        },
        "hashes": {
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "root_calibration_sha256": sha256_file(base.CALIBRATION_PATH),
            "v5_recovered_result_sha256": sha256_file(
                ROOT / "reports/p1_public_transport_repair_cycle_20260831_v5/recovered-result.json"
            ),
        },
    }
    result["independent_qa"] = independent_qa(result)
    write_json(ARTIFACT / "result.json", result)
    write_json(REPORT / "independent-qa.json", result["independent_qa"])
    if outputs:
        write_json(DELIVERY / "SET_MANIFEST.json", result)
    progress(
        {
            "experiment_id": EXPERIMENT_ID,
            "phase": "terminal",
            "status": result["status"],
            "fit_count": result["fit_count"],
            "pass_count": result["pass_count"],
            "outputs": len(outputs),
        }
    )
    return native(result)


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
            write_json(
                ARTIFACT / "terminal_failure.json",
                {
                    "status": "TERMINAL_TECHNICAL_FAILURE",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "automatic_restart_allowed": False,
                    "hidden_truth_reads": 0,
                    "uploads": 0,
                },
            )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
