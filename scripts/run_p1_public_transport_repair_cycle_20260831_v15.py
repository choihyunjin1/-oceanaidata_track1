"""Two-fit nested causal additive-spline residual experiment for P1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import SplineTransformer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_p1_public_transport_repair_cycle_20260831_v13 as base  # noqa: E402

from p1_qc.config import FeatureConfig  # noqa: E402
from p1_qc.features import build_features  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v15"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
TRAIN_PATH = ROOT / "데이터셋 원본/데이터셋_P1/P1_qc_anomaly/train.csv"
FOLDS = base.FOLDS
SOURCE_FEATURES = {
    "abs_temp_robust_z_6h": ("temp_robust_z_6h", True),
    "abs_temp_robust_z_24h": ("temp_robust_z_24h", True),
    "abs_temp_robust_z_72h": ("temp_robust_z_72h", True),
    "temp_abs_median_resid_6h": ("temp_abs_median_resid_6h", False),
    "temp_abs_median_resid_24h": ("temp_abs_median_resid_24h", False),
    "temp_abs_median_resid_72h": ("temp_abs_median_resid_72h", False),
    "temp_abs_peer_residual": ("temp_abs_peer_residual", False),
    "reference_abs_resid_7d": ("reference_abs_resid_7d", False),
    "reference_abs_resid_14d": ("reference_abs_resid_14d", False),
    "abs_reference_slope_1h_7d": ("reference_slope_1h_7d", True),
}


class ContractError(RuntimeError):
    """Frozen v15 contract violation."""


@dataclass(frozen=True)
class AdditiveModel:
    median: np.ndarray
    scale: np.ndarray
    spline: SplineTransformer
    classifier: LogisticRegression

    def transform(self, values: np.ndarray) -> np.ndarray:
        missing = ~np.isfinite(values)
        filled = np.where(missing, self.median, values)
        scaled = np.clip((filled - self.median) / self.scale, -8.0, 8.0)
        spline_values = self.spline.transform(scaled)
        return np.column_stack([spline_values, missing.astype(np.float64)])

    def predict_probability(self, values: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self.transform(values))[:, 1]


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_contract() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    calibration = json.loads(base.CALIBRATION_PATH.read_text(encoding="utf-8"))
    family = config["transport_family"]
    policy = config["decision_policy"]
    smooth_penalty = max(
        float(item["adverse_penalty_points"])
        for item in calibration["observed_pairs"]
        if item["tier_id"] == "SMOOTH_LEARNED_PROFILE"
    )
    model = config["model"]
    checks = {
        "calibration_sha": family["calibration_sha256"] == base.sha256_file(base.CALIBRATION_PATH),
        "family": family["family_id"] == "P1_SMOOTH_ADDITIVE_RESIDUAL_PROFILE",
        "tier": family["tier_id"] == "SMOOTH_LEARNED_PROFILE",
        "penalty": np.isclose(policy["transport_penalty_points"], smooth_penalty, atol=1e-15),
        "raw": np.isclose(
            policy["minimum_raw_expected_point_delta_inclusive"],
            smooth_penalty + calibration["minimum_calibrated_expected_points_delta"],
            atol=1e-15,
        ),
        "features": config["features"]["names"] == list(SOURCE_FEATURES),
        "spline": model["spline_degree"] == 3 and model["spline_n_knots"] == 5,
        "logistic": model["logistic_c"] == 0.1 and model["logistic_solver"] == "lbfgs",
        "threshold": model["probability_threshold_inclusive"] == 0.9,
        "fits": config["fit_budget"]["maximum"] == 2,
        "no_tuning": config["validation"]["outer_result_based_tuning"] is False,
    }
    if not all(checks.values()):
        raise ContractError(f"v15 contract mismatch: {checks}")
    return config


def build_causal_feature_frame(config: dict[str, Any]) -> pd.DataFrame:
    raw = pd.read_csv(
        TRAIN_PATH,
        usecols=["station", "year", "layer", "time", "temp", "psal", "depth"],
    )
    feature_config = FeatureConfig(
        mode="causal",
        rolling_hours=tuple(int(value) for value in config["features"]["rolling_hours"]),
        long_windows_days=tuple(int(value) for value in config["features"]["long_windows_days"]),
    )
    bundle = build_features(raw, config=feature_config, mode="causal")
    output = raw[["station", "year", "layer", "time"]].copy()
    for output_name, (source_name, absolute) in SOURCE_FEATURES.items():
        values = pd.to_numeric(bundle.frame[source_name], errors="coerce")
        output[output_name] = values.abs() if absolute else values
    output["time"] = pd.to_datetime(output["time"], utc=True)
    if output.duplicated(["station", "year", "layer", "time"]).any():
        raise ContractError("causal feature keys are not unique")
    return output


def load_nested_frame(config: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    anchor_frame = pd.read_parquet(base.ANCHOR_PATH)
    anchor = anchor_frame["current_router_prediction"].to_numpy(np.int8)
    historical, _ = base.attach_truth(anchor_frame, anchor)
    feature_frame = build_causal_feature_frame(config)
    keys = ["station", "year", "layer", "time"]
    merged = historical.merge(feature_frame, on=keys, how="left", validate="one_to_one")
    if len(merged) != len(historical):
        raise ContractError("causal features do not align to OOF rows")
    if not np.array_equal(
        merged["current_router_prediction"].to_numpy(np.int8),
        anchor,
    ):
        raise ContractError("feature join changed anchor order")
    return merged, anchor


def prefix_singleton_weights(frame: pd.DataFrame, labels: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, int]:
    positions = np.flatnonzero(train_mask)
    work = frame.loc[train_mask, ["station", "layer", "time"]].copy()
    work["label"] = labels[positions]
    work["position"] = np.arange(len(work), dtype=np.int64)
    work["time"] = pd.to_datetime(work["time"], utc=True)
    weights = np.ones(len(work), dtype=np.float64)
    singleton_count = 0
    cadence = pd.Timedelta(minutes=10)
    for _, group in work.groupby(["station", "layer"], sort=False, observed=True):
        ordered = group.sort_values("time", kind="stable")
        y = ordered["label"].to_numpy(np.int8)
        time_values = ordered["time"].to_numpy()
        singleton = np.zeros(len(ordered), dtype=bool)
        if len(ordered) >= 3:
            exact_before = time_values[1:-1] - time_values[:-2] == cadence
            exact_after = time_values[2:] - time_values[1:-1] == cadence
            singleton[1:-1] = (
                (y[1:-1] == 1)
                & (y[:-2] == 0)
                & (y[2:] == 0)
                & exact_before
                & exact_after
            )
        selected = ordered["position"].to_numpy(np.int64)[singleton]
        weights[selected] = 0.5
        singleton_count += int(singleton.sum())
    return weights, singleton_count


def fit_additive(values: np.ndarray, labels: np.ndarray, weights: np.ndarray, config: dict[str, Any]) -> AdditiveModel:
    median = np.nanmedian(values, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    mad = np.nanmedian(np.abs(values - median), axis=0)
    scale = np.maximum(1.4826 * np.where(np.isfinite(mad), mad, 0.0), 1e-6)
    missing = ~np.isfinite(values)
    filled = np.where(missing, median, values)
    scaled = np.clip((filled - median) / scale, -8.0, 8.0)
    model = config["model"]
    spline = SplineTransformer(
        n_knots=int(model["spline_n_knots"]),
        degree=int(model["spline_degree"]),
        knots=str(model["spline_knots"]),
        include_bias=bool(model["spline_include_bias"]),
    )
    spline_values = spline.fit_transform(scaled)
    design = np.column_stack([spline_values, missing.astype(np.float64)])
    classifier = LogisticRegression(
        penalty=str(model["logistic_penalty"]),
        C=float(model["logistic_c"]),
        solver=str(model["logistic_solver"]),
        max_iter=int(model["logistic_max_iter"]),
        tol=float(model["logistic_tolerance"]),
    )
    classifier.fit(design, labels, sample_weight=weights)
    return AdditiveModel(median=median, scale=scale, spline=spline, classifier=classifier)


def train_nested_predictions(
    frame: pd.DataFrame,
    anchor: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    truth = frame["label_base"].to_numpy(np.int8)
    values = frame[config["features"]["names"]].to_numpy(np.float64)
    probability = np.zeros(len(frame), dtype=np.float64)
    candidate = anchor.copy()
    records: list[dict[str, Any]] = []
    threshold = float(config["model"]["probability_threshold_inclusive"])
    for fit_number, spec in enumerate(config["validation"]["nested_fits"], start=1):
        train_scope = frame["fold"].isin(spec["train_folds"]).to_numpy()
        train_mask = train_scope & (anchor == 0)
        validation_mask = frame["fold"].eq(spec["validation_fold"]).to_numpy()
        weights, singleton_count = prefix_singleton_weights(frame, truth, train_mask)
        model = fit_additive(values[train_mask], truth[train_mask], weights, config)
        validation_probability = model.predict_probability(values[validation_mask])
        probability[validation_mask] = validation_probability
        additions = validation_mask & (anchor == 0)
        proposed = np.zeros(len(frame), dtype=bool)
        proposed[additions] = validation_probability[anchor[validation_mask] == 0] >= threshold
        candidate[proposed] = 1
        records.append(
            {
                "fit_number": fit_number,
                "train_folds": list(spec["train_folds"]),
                "validation_fold": spec["validation_fold"],
                "train_rows": int(train_mask.sum()),
                "train_positives": int(truth[train_mask].sum()),
                "singleton_positive_rows_weighted_0_5": singleton_count,
                "validation_rows": int(validation_mask.sum()),
                "validation_anchor_negative_rows": int((validation_mask & (anchor == 0)).sum()),
                "sealed_additions": int(proposed.sum()),
                "probability_sha256": sha256_array(validation_probability.astype(np.float64)),
                "candidate_bits_sha256": base.sha256_bool(candidate[validation_mask]),
                "outer_target_reads_before_prediction_seal": 0,
                "model_coefficients_sha256": sha256_array(model.classifier.coef_.astype(np.float64)),
                "scale_median_sha256": sha256_array(model.median.astype(np.float64)),
                "scale_mad_sha256": sha256_array(model.scale.astype(np.float64)),
            }
        )
    return candidate, probability, records


def evaluate(frame: pd.DataFrame, anchor: np.ndarray, candidate: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    truth = frame["label_base"].to_numpy(np.int8)
    evaluated = frame["fold"].isin(FOLDS[1:]).to_numpy()
    additions = evaluated & (candidate == 1) & (anchor == 0)
    removals = evaluated & (candidate == 0) & (anchor == 1)
    reference = float(f1_score(truth[evaluated], anchor[evaluated]))
    value = float(f1_score(truth[evaluated], candidate[evaluated]))
    delta = value - reference
    by_fold: dict[str, dict[str, Any]] = {}
    for fold in FOLDS:
        mask = frame["fold"].eq(fold).to_numpy()
        add = mask & (candidate == 1) & (anchor == 0)
        tp = int((add & (truth == 1)).sum())
        fp = int((add & (truth == 0)).sum())
        base_score = float(f1_score(truth[mask], anchor[mask]))
        candidate_score = float(f1_score(truth[mask], candidate[mask]))
        by_fold[fold] = {
            "rows": int(mask.sum()),
            "reference_f1": base_score,
            "candidate_f1": candidate_score,
            "delta_f1": candidate_score - base_score,
            "additions": tp + fp,
            "true_positive_additions": tp,
            "false_positive_additions": fp,
            "additions_precision": tp / (tp + fp) if tp + fp else None,
            "anchor_f1_divided_by_2": base_score / 2.0,
        }
    bootstrap = base.base.day_bootstrap(
        frame,
        anchor,
        candidate,
        {
            "validation": {
                "bootstrap_replicates": config["validation"]["bootstrap_replicates"],
                "bootstrap_seed": config["validation"]["bootstrap_seed"],
            }
        },
    )
    tp = int((additions & (truth == 1)).sum())
    fp = int((additions & (truth == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    local_day = pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    outer = frame.loc[evaluated, ["station", "layer", "fold"]].copy()
    outer["day"] = local_day[evaluated].to_numpy()
    outer["addition"] = additions[evaluated]
    day_rows = outer.groupby("day", observed=True).size()
    day_additions = outer.groupby("day", observed=True)["addition"].sum()
    maximum_day_changed_fraction = float((day_additions / day_rows).max()) if len(day_rows) else 0.0
    changed = outer.loc[outer["addition"], ["station", "layer", "fold"]]
    concentration = changed.groupby(["station", "layer", "fold"], observed=True).size()
    maximum_concentration = float(concentration.max() / len(changed)) if len(changed) else 0.0
    station_layer: dict[str, float] = {}
    evaluated_positions = np.flatnonzero(evaluated)
    local = frame.loc[evaluated, ["station", "layer"]].reset_index(drop=True)
    for (station, layer), positions in local.groupby(["station", "layer"], observed=True).indices.items():
        index = evaluated_positions[np.asarray(positions, dtype=np.int64)]
        station_layer[f"{station}|{layer}"] = float(
            f1_score(truth[index], candidate[index]) - f1_score(truth[index], anchor[index])
        )
    policy = config["decision_policy"]
    raw_points = delta * float(policy["score_points_per_f1"])
    calibrated = raw_points - float(policy["transport_penalty_points"])
    gates = {
        "positive_additions": int(additions.sum()) > 0,
        "anchor_removals_zero": int(removals.sum()) == 0,
        "q3_q4_each_nonnegative": min(by_fold[fold]["delta_f1"] for fold in FOLDS[1:]) >= 0.0,
        "pooled_delta_positive": delta > 0.0,
        "addition_precision_above_anchor_f1_half": precision is not None and precision > reference / 2.0,
        "day_block_ci90_low_strictly_positive": bootstrap["ci90_low"] > 0.0,
        "bootstrap_probability_improved_at_least_0_8": bootstrap["probability_improved"]
        >= float(policy["bootstrap_probability_improved_minimum_inclusive"]),
        "raw_expected_points_at_least_0_131682092": raw_points
        >= float(policy["minimum_raw_expected_point_delta_inclusive"]),
        "calibrated_expected_points_at_least_0_01": calibrated
        >= float(policy["minimum_calibrated_expected_point_delta_inclusive"]),
        "changed_fraction_at_most_0_005": float(additions.sum() / evaluated.sum())
        <= float(config["safety"]["maximum_changed_fraction"]),
        "each_kst_day_changed_fraction_at_most_0_005": maximum_day_changed_fraction
        <= float(config["safety"]["maximum_changed_fraction_any_kst_day"]),
        "station_layer_quarter_concentration_at_most_0_5": maximum_concentration
        <= float(config["safety"]["maximum_addition_concentration_any_station_layer_quarter"]),
        "each_supported_station_layer_nonnegative": min(station_layer.values())
        >= float(config["safety"]["minimum_each_supported_station_layer_delta_f1"]),
    }
    return {
        "name": "P1_1_CAUSAL_ADDITIVE_SPLINE_RESIDUAL",
        "fit_count": 2,
        "reference_f1": reference,
        "candidate_f1": value,
        "delta_f1": delta,
        "raw_expected_points_delta": raw_points,
        "transport_penalty_points": float(policy["transport_penalty_points"]),
        "calibrated_conservative_expected_points_delta": calibrated,
        "additions": int(additions.sum()),
        "true_positive_additions": tp,
        "false_positive_additions": fp,
        "additions_precision": precision,
        "anchor_f1_divided_by_2": reference / 2.0,
        "anchor_removals": int(removals.sum()),
        "changed_fraction": float(additions.sum() / evaluated.sum()),
        "maximum_kst_day_changed_fraction": maximum_day_changed_fraction,
        "maximum_addition_concentration_station_layer_quarter": maximum_concentration,
        "station_layer_delta_f1": station_layer,
        "by_fold": by_fold,
        "day_block_bootstrap": bootstrap,
        "gates": gates,
        "strict_internal_pass": bool(all(gates.values())),
    }


def independent_qa(result: dict[str, Any]) -> dict[str, Any]:
    candidate = result["candidate"]
    checks = {
        "exactly_two_nested_fits": result["fit_count"] == 2 and len(result["nested_fit_receipts"]) == 2,
        "ten_features": len(result["feature_contract"]["names"]) == 10,
        "causal_features": result["feature_contract"]["source_mode"] == "causal",
        "outer_targets_zero_before_seals": all(
            item["outer_target_reads_before_prediction_seal"] == 0 for item in result["nested_fit_receipts"]
        ),
        "threshold_fixed_0_9": result["model_contract"]["probability_threshold_inclusive"] == 0.9,
        "add_only": candidate["anchor_removals"] == 0,
        "official_reads_zero": result["operations"]["official_covariate_reads"] == 0,
        "hidden_reads_zero": result["operations"]["hidden_truth_reads"] == 0,
        "csv_zero": result["operations"]["submission_csv_created"] == 0,
        "uploads_zero": result["operations"]["uploads"] == 0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def validate_only() -> dict[str, Any]:
    load_contract()
    for path in (TRAIN_PATH, base.ANCHOR_PATH, base.CALIBRATION_PATH, base.AUTHORITATIVE_PATH):
        if not path.is_file():
            raise ContractError(f"missing frozen input: {path}")
    return {
        "status": "VALID",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": base.sha256_file(CONFIG_PATH),
        "runner_sha256": base.sha256_file(Path(__file__)),
        "candidate_count": 1,
        "fit_budget": 2,
    }


def execute() -> dict[str, Any]:
    if ARTIFACT.exists():
        raise FileExistsError("exactly-once artifact path exists")
    config = load_contract()
    ARTIFACT.mkdir(parents=True)
    started = time.perf_counter()
    base.write_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "pid": os.getpid(),
            "config_sha256": base.sha256_file(CONFIG_PATH),
            "runner_sha256": base.sha256_file(Path(__file__)),
            "fit_budget": 2,
            "official_covariate_reads": 0,
        },
    )
    base.write_json(ARTIFACT / "progress.json", {"phase": "causal_feature_build", "fit_count": 0}, exclusive=False)
    frame, anchor = load_nested_frame(config)
    candidate, probability, fit_receipts = train_nested_predictions(frame, anchor, config)
    prediction_path = ARTIFACT / "sealed_nested_predictions.npz"
    np.savez_compressed(prediction_path, candidate=candidate, probability=probability)
    prediction_seal = {
        "candidate_sha256": base.sha256_bool(candidate),
        "probability_sha256": sha256_array(probability.astype(np.float64)),
        "npz_sha256": base.sha256_file(prediction_path),
        "q3_outer_target_reads_before_seal": 0,
        "q4_outer_target_reads_before_seal": 0,
    }
    base.write_json(ARTIFACT / "prediction_seal.json", prediction_seal)
    base.write_json(ARTIFACT / "progress.json", {"phase": "prediction_sealed", "fit_count": 2}, exclusive=False)
    record = evaluate(frame, anchor, candidate, config)
    result: dict[str, Any] = {
        "schema_version": "p1.public_transport_repair_cycle.20260831.v15.result",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_INTERNAL_ONLY",
        "runtime_seconds": time.perf_counter() - started,
        "transport_family": config["transport_family"],
        "feature_contract": config["features"],
        "model_contract": config["model"],
        "safety_contract": config["safety"],
        "decision_policy": config["decision_policy"],
        "candidate_count": 1,
        "pass_count": int(record["strict_internal_pass"]),
        "fit_count": 2,
        "candidate": record,
        "nested_fit_receipts": fit_receipts,
        "prediction_seal": prediction_seal,
        "outputs": [],
        "adaptive_surface_disclaimer": "This is newly preregistered development evidence on reused historical Q3/Q4 surfaces, not an independent confirmation.",
        "operations": {
            "historical_train_csv_reads": 1,
            "official_covariate_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "hashes": {
            "config_sha256": base.sha256_file(CONFIG_PATH),
            "runner_sha256": base.sha256_file(Path(__file__)),
            "historical_train_sha256": base.sha256_file(TRAIN_PATH),
            "feature_builder_sha256": base.sha256_file(ROOT / "src/p1_qc/features.py"),
            "anchor_sha256": base.sha256_file(base.ANCHOR_PATH),
            "root_calibration_sha256": base.sha256_file(base.CALIBRATION_PATH),
            "authoritative_results_sha256": base.sha256_file(base.AUTHORITATIVE_PATH),
        },
    }
    result["independent_qa"] = independent_qa(result)
    base.write_json(ARTIFACT / "result.json", result)
    base.write_json(REPORT / "independent-qa.json", result["independent_qa"])
    base.write_json(
        ARTIFACT / "progress.json",
        {"phase": "terminal", "fit_count": 2, "pass_count": result["pass_count"]},
        exclusive=False,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.validate_only == args.execute:
        parser.error("choose exactly one mode")
    try:
        payload = validate_only() if args.validate_only else execute()
        print(json.dumps(base.native(payload), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        payload = {
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "official_covariate_reads": 0,
            "hidden_truth_reads": 0,
            "uploads": 0,
        }
        if ARTIFACT.exists() and not (ARTIFACT / "terminal_failure.json").exists():
            base.write_json(ARTIFACT / "terminal_failure.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
