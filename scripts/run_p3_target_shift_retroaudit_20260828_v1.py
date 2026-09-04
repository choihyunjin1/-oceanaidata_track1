from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


KEYS = ["anchor_id", "station", "lead_h"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    denom = float(np.square(weights).sum())
    return 0.0 if denom == 0.0 else total * total / denom


def weighted_rmse(target: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(target) & np.isfinite(prediction) & np.isfinite(weights) & (weights >= 0)
    if not np.any(valid) or float(weights[valid].sum()) <= 0.0:
        return math.nan
    return float(np.sqrt(np.average(np.square(prediction[valid] - target[valid]), weights=weights[valid])))


def _feature_matrix(frame: pd.DataFrame, numeric_columns: list[str], medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    numeric = frame[numeric_columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    if medians is None:
        medians = np.nanmedian(numeric, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
    missing = ~np.isfinite(numeric)
    numeric = np.where(missing, medians[None, :], numeric)
    station = pd.get_dummies(frame["station"], prefix="station").reindex(
        columns=["station_G-ORS", "station_I-ORS", "station_S-ORS"], fill_value=0
    )
    return np.column_stack([numeric, missing.astype(float), station.to_numpy(dtype=float)]), medians


def cross_fitted_domain_weights(
    source: pd.DataFrame,
    target: pd.DataFrame,
    numeric_columns: list[str],
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    combined = pd.concat([source, target], ignore_index=True)
    x, _ = _feature_matrix(combined, numeric_columns)
    y = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]

    splitter = RepeatedStratifiedKFold(
        n_splits=int(settings["folds"]),
        n_repeats=int(settings["repeats"]),
        random_state=int(settings["seed"]),
    )
    probability_sum = np.zeros(len(combined), dtype=float)
    prediction_count = np.zeros(len(combined), dtype=int)
    fold_auc: list[float] = []

    for train_idx, validation_idx in splitter.split(x, y):
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        x_train = scaler.fit_transform(imputer.fit_transform(x[train_idx]))
        x_validation = scaler.transform(imputer.transform(x[validation_idx]))
        model = LogisticRegression(
            C=float(settings["c"]),
            max_iter=int(settings["maximum_iterations"]),
            solver="lbfgs",
            random_state=int(settings["seed"]),
        )
        model.fit(x_train, y[train_idx])
        prediction = model.predict_proba(x_validation)[:, 1]
        probability_sum[validation_idx] += prediction
        prediction_count[validation_idx] += 1
        fold_auc.append(float(roc_auc_score(y[validation_idx], prediction)))

    if not np.all(prediction_count == int(settings["repeats"])):
        raise RuntimeError("cross-fitting did not predict every row exactly once per repeat")
    probability = probability_sum / prediction_count
    overall_auc = float(roc_auc_score(y, probability))
    raw_source_weight = probability[: len(source)] / np.clip(1.0 - probability[: len(source)], 1e-6, None)
    raw_source_weight *= len(source) / len(target)
    clipped = np.clip(
        raw_source_weight,
        float(settings["weight_clip_lower"]),
        float(settings["weight_clip_upper"]),
    )
    clipped /= clipped.mean()

    result = source[["anchor_id", "station"]].copy()
    result["domain_target_probability"] = probability[: len(source)]
    result["propensity_weight_raw"] = raw_source_weight
    result["propensity_weight"] = clipped

    diagnostics = {
        "source_cases": int(len(source)),
        "target_cases": int(len(target)),
        "overall_cross_fitted_auc": overall_auc,
        "fold_auc_mean": float(np.mean(fold_auc)),
        "fold_auc_std": float(np.std(fold_auc, ddof=1)),
        "brier_score": float(brier_score_loss(y, probability)),
        "source_weight_min": float(clipped.min()),
        "source_weight_median": float(np.median(clipped)),
        "source_weight_max": float(clipped.max()),
        "source_weight_effective_sample_size": effective_sample_size(clipped),
        "source_weight_clipped_low_count": int(np.sum(raw_source_weight < float(settings["weight_clip_lower"]))),
        "source_weight_clipped_high_count": int(np.sum(raw_source_weight > float(settings["weight_clip_upper"]))),
    }
    return result, diagnostics, x


def nearest_neighbor_weights(x: np.ndarray, source_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputer.fit_transform(x))
    source = scaled[:source_count]
    target = scaled[source_count:]
    squared_distance = (
        np.square(target).sum(axis=1)[:, None]
        + np.square(source).sum(axis=1)[None, :]
        - 2.0 * target @ source.T
    )
    squared_distance = np.maximum(squared_distance, 0.0)
    nearest = np.argmin(squared_distance, axis=1)
    counts = np.bincount(nearest, minlength=source_count).astype(float)
    weights = counts * source_count / len(target)
    return weights, {
        "matched_source_cases": int(np.sum(counts > 0)),
        "unmatched_source_cases": int(np.sum(counts == 0)),
        "effective_sample_size": effective_sample_size(weights),
        "target_nearest_distance_mean": float(np.sqrt(np.min(squared_distance, axis=1)).mean()),
        "target_nearest_distance_q90": float(np.quantile(np.sqrt(np.min(squared_distance, axis=1)), 0.90)),
    }


def standardized_differences(source: pd.DataFrame, target: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in columns:
        source_values = pd.to_numeric(source[column], errors="coerce").to_numpy(dtype=float)
        target_values = pd.to_numeric(target[column], errors="coerce").to_numpy(dtype=float)
        source_values = source_values[np.isfinite(source_values)]
        target_values = target_values[np.isfinite(target_values)]
        pooled = math.sqrt((float(np.var(source_values)) + float(np.var(target_values))) / 2.0)
        smd = 0.0 if pooled == 0.0 else (float(np.mean(target_values)) - float(np.mean(source_values))) / pooled
        rows.append(
            {
                "feature": column,
                "source_mean": float(np.mean(source_values)),
                "target_mean": float(np.mean(target_values)),
                "standardized_mean_difference": float(smd),
                "absolute_smd": float(abs(smd)),
                "source_missing_fraction": float(1.0 - len(source_values) / len(source)),
                "target_missing_fraction": float(1.0 - len(target_values) / len(target)),
            }
        )
    return sorted(rows, key=lambda row: row["absolute_smd"], reverse=True)


def biased_mmd_permutation(x: np.ndarray, source_count: int, settings: dict[str, Any]) -> dict[str, Any]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    z = scaler.fit_transform(imputer.fit_transform(x))
    squared = (
        np.square(z).sum(axis=1)[:, None]
        + np.square(z).sum(axis=1)[None, :]
        - 2.0 * z @ z.T
    )
    squared = np.maximum(squared, 0.0)
    positive = squared[np.triu_indices_from(squared, k=1)]
    positive = positive[positive > 0]
    median_squared = float(np.median(positive)) if len(positive) else 1.0
    kernel = np.exp(-squared / max(median_squared, 1e-12))

    def statistic(source_idx: np.ndarray, target_idx: np.ndarray) -> float:
        return float(
            kernel[np.ix_(source_idx, source_idx)].mean()
            + kernel[np.ix_(target_idx, target_idx)].mean()
            - 2.0 * kernel[np.ix_(source_idx, target_idx)].mean()
        )

    observed_source = np.arange(source_count)
    observed_target = np.arange(source_count, len(z))
    observed = statistic(observed_source, observed_target)
    rng = np.random.default_rng(int(settings["seed"]))
    exceed = 0
    all_idx = np.arange(len(z))
    for _ in range(int(settings["permutations"])):
        permuted = rng.permutation(all_idx)
        value = statistic(permuted[:source_count], permuted[source_count:])
        exceed += int(value >= observed)
    return {
        "biased_mmd2": observed,
        "rbf_median_squared_distance": median_squared,
        "permutations": int(settings["permutations"]),
        "permutation_p_value": float((exceed + 1) / (int(settings["permutations"]) + 1)),
    }


def load_candidate(spec: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_parquet(spec["path"])
    for column, value in spec.get("filters", {}).items():
        frame = frame.loc[frame[column] == value]
    prediction_column = spec["prediction_column"]
    required = set(KEYS + [prediction_column])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"candidate {spec['name']} missing columns: {missing}")
    result = frame[KEYS + [prediction_column]].rename(columns={prediction_column: "candidate_prediction"})
    if result.duplicated(KEYS).any():
        raise ValueError(f"candidate {spec['name']} contains duplicate keys")
    return result


def bootstrap_delta(
    case_errors: pd.DataFrame,
    weight_column: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    rng = np.random.default_rng(int(settings["seed"]))
    candidate_mse = case_errors["candidate_mse"].to_numpy(dtype=float)
    champion_mse = case_errors["champion_mse"].to_numpy(dtype=float)
    weights = case_errors[weight_column].to_numpy(dtype=float)
    deltas = np.empty(int(settings["replicates"]), dtype=float)
    for idx in range(len(deltas)):
        sample = rng.integers(0, len(case_errors), size=len(case_errors))
        selected_weight = weights[sample]
        if selected_weight.sum() <= 0:
            deltas[idx] = math.nan
            continue
        candidate_rmse = math.sqrt(float(np.average(candidate_mse[sample], weights=selected_weight)))
        champion_rmse = math.sqrt(float(np.average(champion_mse[sample], weights=selected_weight)))
        deltas[idx] = candidate_rmse - champion_rmse
    deltas = deltas[np.isfinite(deltas)]
    alpha = (1.0 - float(settings["confidence"])) / 2.0
    return {
        "replicates": int(len(deltas)),
        "ci_lower": float(np.quantile(deltas, alpha)),
        "median": float(np.median(deltas)),
        "ci_upper": float(np.quantile(deltas, 1.0 - alpha)),
        "probability_improved": float(np.mean(deltas < 0.0)),
    }


def candidate_metrics(
    champion_rows: pd.DataFrame,
    candidate: pd.DataFrame,
    case_weights: pd.DataFrame,
    bootstrap_settings: dict[str, Any],
) -> dict[str, Any]:
    merged = champion_rows.merge(candidate, on=KEYS, how="inner", validate="one_to_one")
    merged = merged.merge(case_weights, on=["anchor_id", "station"], how="left", validate="many_to_one")
    merged["uniform_weight"] = 1.0
    merged["target_hs"] = merged["target_hs"].astype(float)
    merged["champion_prediction"] = merged["champion_prediction"].astype(float)
    merged["candidate_prediction"] = merged["candidate_prediction"].astype(float)
    merged["candidate_sqerr"] = np.square(merged["candidate_prediction"] - merged["target_hs"])
    merged["champion_sqerr"] = np.square(merged["champion_prediction"] - merged["target_hs"])

    output: dict[str, Any] = {
        "cases": int(merged["anchor_id"].nunique()),
        "rows": int(len(merged)),
        "coverage_fraction": float(merged["anchor_id"].nunique() / champion_rows["anchor_id"].nunique()),
    }
    for label, weight_column in [
        ("unweighted", "uniform_weight"),
        ("propensity_weighted", "propensity_weight"),
        ("nearest_neighbor_weighted", "nearest_neighbor_weight"),
    ]:
        row_weights = merged[weight_column].to_numpy(dtype=float)
        candidate_rmse = weighted_rmse(merged["target_hs"], merged["candidate_prediction"], row_weights)
        champion_rmse = weighted_rmse(merged["target_hs"], merged["champion_prediction"], row_weights)
        by_station: dict[str, Any] = {}
        for station, station_rows in merged.groupby("station", sort=True):
            station_weights = station_rows[weight_column].to_numpy(dtype=float)
            by_station[str(station)] = {
                "candidate_rmse": weighted_rmse(station_rows["target_hs"], station_rows["candidate_prediction"], station_weights),
                "champion_rmse": weighted_rmse(station_rows["target_hs"], station_rows["champion_prediction"], station_weights),
            }
            by_station[str(station)]["delta_rmse"] = (
                by_station[str(station)]["candidate_rmse"] - by_station[str(station)]["champion_rmse"]
            )
        output[label] = {
            "candidate_rmse": candidate_rmse,
            "champion_rmse": champion_rmse,
            "delta_rmse": candidate_rmse - champion_rmse,
            "by_station": by_station,
        }

    case_errors = (
        merged.groupby(["anchor_id", "station"], as_index=False)
        .agg(
            candidate_mse=("candidate_sqerr", "mean"),
            champion_mse=("champion_sqerr", "mean"),
            propensity_weight=("propensity_weight", "first"),
            nearest_neighbor_weight=("nearest_neighbor_weight", "first"),
        )
    )
    output["propensity_weighted"]["bootstrap_delta_vs_champion"] = bootstrap_delta(
        case_errors, "propensity_weight", bootstrap_settings
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/p3_target_shift_retroaudit_20260828_v1.json",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    champion_replay = pd.read_parquet(config["champion_replay"])
    champion_cases = champion_replay[["anchor_id", "station"]].drop_duplicates()
    if champion_cases.duplicated(["anchor_id", "station"]).any():
        raise RuntimeError("champion case keys are not unique")

    train_features = pd.read_parquet(
        config["source_features"],
        columns=["anchor_id", "station"] + config["feature_columns"],
    )
    test_features = pd.read_parquet(
        config["target_features"],
        columns=["case_id", "station"] + config["feature_columns"],
    )
    source = champion_cases.merge(train_features, on=["anchor_id", "station"], how="left", validate="one_to_one")
    target = test_features.copy()
    if len(source) != 181 or len(target) != 200:
        raise RuntimeError(f"unexpected source/target case count: {len(source)}/{len(target)}")

    case_weights, domain_diagnostics, combined_x = cross_fitted_domain_weights(
        source, target, config["feature_columns"], config["domain_classifier"]
    )
    nn_weights, nn_diagnostics = nearest_neighbor_weights(combined_x, len(source))
    case_weights["nearest_neighbor_weight"] = nn_weights
    case_weights.to_parquet(output_dir / "case_domain_weights.parquet", index=False)

    feature_shift = standardized_differences(source, target, config["feature_columns"])
    station_shift = []
    for station in ["G-ORS", "I-ORS", "S-ORS"]:
        station_shift.append(
            {
                "station": station,
                "source_fraction": float(np.mean(source["station"] == station)),
                "target_fraction": float(np.mean(target["station"] == station)),
            }
        )
    mmd = biased_mmd_permutation(combined_x, len(source), config["mmd"])

    anchor_metadata = pd.read_parquet(config["anchor_metadata"])
    target_long = anchor_metadata.melt(
        id_vars=["anchor_id", "station", "current_hs"],
        value_vars=["target_3", "target_6", "target_9", "target_12", "target_18", "target_24"],
        var_name="lead_name",
        value_name="target_hs",
    )
    target_long["lead_h"] = target_long["lead_name"].str.replace("target_", "", regex=False).astype(int)
    champion_rows = champion_replay[KEYS + ["champion_prediction"]].merge(
        target_long[KEYS + ["target_hs", "current_hs"]], on=KEYS, how="left", validate="one_to_one"
    )
    if champion_rows["target_hs"].isna().any():
        raise RuntimeError("historical target join produced missing labels")

    metrics: dict[str, Any] = {}
    persistence = champion_rows[KEYS + ["current_hs"]].rename(columns={"current_hs": "candidate_prediction"})
    metrics["persistence"] = candidate_metrics(
        champion_rows, persistence, case_weights, config["bootstrap"]
    )
    for candidate_spec in config["candidates"]:
        metrics[candidate_spec["name"]] = candidate_metrics(
            champion_rows,
            load_candidate(candidate_spec),
            case_weights,
            config["bootstrap"],
        )

    rising_source = int(np.sum(source["hs_delta_12h"] > 0.2))
    rising_target = int(np.sum(target["hs_delta_12h"] > 0.2))
    result = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED_RESEARCH_ONLY_NO_SUBMISSION",
        "research_question": config["objective"],
        "simple_rising_check": {
            "threshold_hs_delta_12h": 0.2,
            "source_count": rising_source,
            "source_fraction": rising_source / len(source),
            "target_count": rising_target,
            "target_fraction": rising_target / len(target),
        },
        "domain_classifier": domain_diagnostics,
        "nearest_neighbor": nn_diagnostics,
        "mmd": mmd,
        "largest_feature_shifts": feature_shift[:10],
        "station_shift": station_shift,
        "candidate_metrics": metrics,
        "interpretation_contract": {
            "domain_auc_near_0p5": "source and target are not linearly distinguishable on the fixed past-only physical feature set",
            "domain_auc_high": "simple rising-share matching is insufficient; use weighted and support-aware evaluation",
            "importance_weighting_assumption": "P(target_hs|past_features) is stable across historical and official domains; this is untestable without official labels",
            "weight_reliability_guard": "effective sample size below 60 cases is considered too concentrated for promotion",
            "candidate_gate": "negative propensity-weighted delta, CI90 upper below zero, at least two stations non-degrading, and coverage >= 0.95",
        },
        "scope_guards": config["scope_guards"],
        "inputs": {
            "runner": {"path": str(Path(__file__)).replace("\\", "/"), "sha256": sha256_file(Path(__file__))},
            "config": {"path": str(config_path).replace("\\", "/"), "sha256": sha256_file(config_path)},
            "source_features": {"path": config["source_features"], "sha256": sha256_file(Path(config["source_features"]))},
            "target_features": {"path": config["target_features"], "sha256": sha256_file(Path(config["target_features"]))},
            "champion_replay": {"path": config["champion_replay"], "sha256": sha256_file(Path(config["champion_replay"]))},
        },
        "output_files": {
            "case_domain_weights": "case_domain_weights.parquet",
            "result": "result.json",
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
