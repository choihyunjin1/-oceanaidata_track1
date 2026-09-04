from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LEADS = [3, 6, 9, 12, 18, 24]
CASE_KEYS = ["anchor_id", "station"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(actual) - np.asarray(predicted)))))


def add_station_indicators(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    numeric = frame[features].astype(float).copy()
    for station in ["G-ORS", "I-ORS", "S-ORS"]:
        numeric[f"station__{station}"] = (frame["station"] == station).astype(float)
    return numeric


def fit_predict_ridge(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    params: dict[str, Any],
) -> np.ndarray:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(params["alpha"]))),
        ]
    )
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_valid), dtype=float)


def fit_predict_catboost(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    params: dict[str, Any],
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(x_train)
    valid_values = imputer.transform(x_valid)
    predictions = []
    for lead_index in range(y_train.shape[1]):
        model = CatBoostRegressor(
            iterations=int(params["iterations"]),
            learning_rate=float(params["learning_rate"]),
            depth=int(params["depth"]),
            l2_leaf_reg=float(params["l2_leaf_reg"]),
            loss_function="RMSE",
            random_seed=int(params["random_seed"] + lead_index),
            thread_count=int(params["thread_count"]),
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(train_values, y_train[:, lead_index])
        predictions.append(model.predict(valid_values))
    return np.column_stack(predictions).astype(float)


def nested_select(
    x: pd.DataFrame,
    residual_target: np.ndarray,
    actual: np.ndarray,
    champion: np.ndarray,
    folds: np.ndarray,
    outer_train: np.ndarray,
    parameter_grid: list[dict[str, Any]],
    residual_scales: list[float],
    fit_predict: Callable[[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]], np.ndarray],
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    inner_folds = sorted(set(folds[outer_train].tolist()))
    if len(inner_folds) < 2:
        raise RuntimeError("nested selection requires at least two inner folds")
    rows: list[dict[str, Any]] = []
    best: tuple[float, str, float, dict[str, Any]] | None = None
    for params in parameter_grid:
        inner_prediction = np.full_like(residual_target, np.nan, dtype=float)
        for inner_fold in inner_folds:
            inner_valid = outer_train & (folds == inner_fold)
            inner_train = outer_train & (folds != inner_fold)
            if not inner_train.any() or not inner_valid.any():
                raise RuntimeError(f"empty nested split for {inner_fold}")
            inner_prediction[inner_valid] = fit_predict(
                x.loc[inner_train], residual_target[inner_train], x.loc[inner_valid], params
            )
        if np.isnan(inner_prediction[outer_train]).any():
            raise RuntimeError("nested predictions are incomplete")
        for scale in residual_scales:
            candidate = champion[outer_train] + float(scale) * inner_prediction[outer_train]
            score = rmse(actual[outer_train], candidate)
            row = {"params": params, "residual_scale": float(scale), "inner_rmse": score}
            rows.append(row)
            ordering = (score, json.dumps(params, sort_keys=True), float(scale), params)
            if best is None or ordering[:3] < best[:3]:
                best = ordering
    if best is None:
        raise RuntimeError("no nested model selected")
    return best[3], best[2], rows


def bootstrap_case_delta(
    candidate_error: np.ndarray,
    champion_error: np.ndarray,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    if candidate_error.ndim != 2 or candidate_error.shape != champion_error.shape:
        raise ValueError("bootstrap errors must be matching case-by-lead matrices")
    candidate_case_mse = np.mean(np.square(candidate_error), axis=1)
    champion_case_mse = np.mean(np.square(champion_error), axis=1)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(candidate_case_mse), size=(replicates, len(candidate_case_mse)))
    candidate_rmse = np.sqrt(np.mean(candidate_case_mse[sampled], axis=1))
    champion_rmse = np.sqrt(np.mean(champion_case_mse[sampled], axis=1))
    delta = candidate_rmse - champion_rmse
    tail = (1.0 - confidence) / 2.0
    return {
        "replicates": int(replicates),
        "ci_lower": float(np.quantile(delta, tail)),
        "median": float(np.median(delta)),
        "ci_upper": float(np.quantile(delta, 1.0 - tail)),
        "probability_improved": float(np.mean(delta < 0.0)),
    }


def group_metrics(frame: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for value, rows in frame.groupby(column, sort=True):
        champion_score = rmse(rows["target_hs"].to_numpy(), rows["champion_prediction"].to_numpy())
        candidate_score = rmse(rows["target_hs"].to_numpy(), rows["candidate_prediction"].to_numpy())
        result[str(value)] = {
            "champion_rmse": champion_score,
            "candidate_rmse": candidate_score,
            "delta_rmse": candidate_score - champion_score,
        }
    return result


def evaluate_model(frame: pd.DataFrame, bootstrap: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    champion_score = rmse(frame["target_hs"].to_numpy(), frame["champion_prediction"].to_numpy())
    candidate_score = rmse(frame["target_hs"].to_numpy(), frame["candidate_prediction"].to_numpy())
    by_station = group_metrics(frame, "station")
    by_fold = group_metrics(frame, "fold")
    by_lead = group_metrics(frame, "lead_h")
    long_rows = frame[frame["lead_h"].isin([18, 24])]
    long_champion = rmse(long_rows["target_hs"].to_numpy(), long_rows["champion_prediction"].to_numpy())
    long_candidate = rmse(long_rows["target_hs"].to_numpy(), long_rows["candidate_prediction"].to_numpy())
    ordered = frame.sort_values(CASE_KEYS + ["lead_h"])
    cases = ordered[CASE_KEYS].drop_duplicates().reset_index(drop=True)
    candidate_error = []
    champion_error = []
    for case in cases.itertuples(index=False):
        rows = ordered[(ordered["anchor_id"] == case.anchor_id) & (ordered["station"] == case.station)]
        if rows["lead_h"].tolist() != LEADS:
            raise RuntimeError("case does not contain the fixed six leads")
        candidate_error.append((rows["candidate_prediction"] - rows["target_hs"]).to_numpy())
        champion_error.append((rows["champion_prediction"] - rows["target_hs"]).to_numpy())
    bootstrap_result = bootstrap_case_delta(
        np.asarray(candidate_error),
        np.asarray(champion_error),
        int(bootstrap["replicates"]),
        float(bootstrap["confidence"]),
        int(bootstrap["seed"]),
    )
    station_deltas = [metrics["delta_rmse"] for metrics in by_station.values()]
    fold_deltas = [metrics["delta_rmse"] for metrics in by_fold.values()]
    gates = {
        "overall_delta": candidate_score - champion_score < float(gate["overall_delta_below"]),
        "bootstrap_ci90_upper": bootstrap_result["ci_upper"] < float(gate["bootstrap_ci90_upper_below"]),
        "station_consistency": (
            sum(delta <= 0.0 for delta in station_deltas) >= int(gate["minimum_non_degrading_stations"])
            and max(station_deltas) <= float(gate["maximum_station_degradation_rmse_m"])
        ),
        "fold_consistency": (
            sum(delta <= 0.0 for delta in fold_deltas) >= int(gate["minimum_non_degrading_folds"])
            and max(fold_deltas) <= float(gate["maximum_fold_degradation_rmse_m"])
        ),
        "long_lead": long_candidate - long_champion < float(gate["long_lead_delta_below"]),
    }
    return {
        "cases": int(len(cases)),
        "rows": int(len(frame)),
        "champion_rmse": champion_score,
        "candidate_rmse": candidate_score,
        "delta_rmse": candidate_score - champion_score,
        "long_lead_18_24": {
            "champion_rmse": long_champion,
            "candidate_rmse": long_candidate,
            "delta_rmse": long_candidate - long_champion,
        },
        "by_station": by_station,
        "by_fold": by_fold,
        "by_lead": by_lead,
        "bootstrap_delta_vs_champion": bootstrap_result,
        "gates": gates,
        "promotion_gate_pass": bool(all(gates.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/p3_atmosphere_complete_residual_pilot_20260828_v1.json",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features = config["wave_features"] + config["wind_features"]
    feature_frame = pd.read_parquet(
        config["source_features"], columns=CASE_KEYS + all_features
    )
    replay = pd.read_parquet(config["champion_replay"])
    case_fold = replay[["fold"] + CASE_KEYS].drop_duplicates()
    if case_fold.duplicated(CASE_KEYS).any():
        raise RuntimeError("champion cases map to multiple outer folds")
    champion_wide = replay.pivot(index=CASE_KEYS, columns="lead_h", values="champion_prediction")
    champion_wide = champion_wide.reindex(columns=LEADS)
    champion_wide.columns = [f"champion_{lead}" for lead in LEADS]
    champion_wide = champion_wide.reset_index()
    anchors = pd.read_parquet(
        config["anchor_metadata"],
        columns=CASE_KEYS + [f"target_{lead}" for lead in LEADS],
    )
    cases = case_fold.merge(feature_frame, on=CASE_KEYS, how="left", validate="one_to_one")
    cases = cases.merge(champion_wide, on=CASE_KEYS, how="left", validate="one_to_one")
    cases = cases.merge(anchors, on=CASE_KEYS, how="left", validate="one_to_one")
    if len(cases) != 181:
        raise RuntimeError(f"expected 181 champion cases, got {len(cases)}")
    complete_mask = cases[config["wind_features"]].notna().all(axis=1)
    cases = cases.loc[complete_mask].reset_index(drop=True)
    if len(cases) != 124:
        raise RuntimeError(f"expanded wind-lag completeness contract expected 124 cases, got {len(cases)}")

    actual = cases[[f"target_{lead}" for lead in LEADS]].to_numpy(dtype=float)
    champion = cases[[f"champion_{lead}" for lead in LEADS]].to_numpy(dtype=float)
    residual_target = actual - champion
    folds = cases["fold"].astype(str).to_numpy()

    ridge_grid = [{"alpha": alpha} for alpha in config["ridge"]["alphas"]]
    catboost_grid = [
        {
            "iterations": config["catboost"]["iterations"],
            "learning_rate": config["catboost"]["learning_rate"],
            "depth": depth,
            "l2_leaf_reg": l2,
            "random_seed": config["catboost"]["random_seed"],
            "thread_count": config["catboost"]["thread_count"],
        }
        for depth in config["catboost"]["depths"]
        for l2 in config["catboost"]["l2_leaf_regs"]
    ]
    families: list[tuple[str, list[str], list[dict[str, Any]], list[float], Callable[..., np.ndarray]]] = [
        ("ridge_wave_only", config["wave_features"], ridge_grid, config["ridge"]["residual_scales"], fit_predict_ridge),
        ("ridge_wind_only", config["wind_features"], ridge_grid, config["ridge"]["residual_scales"], fit_predict_ridge),
        ("ridge_wave_wind", all_features, ridge_grid, config["ridge"]["residual_scales"], fit_predict_ridge),
    ]
    if config["catboost"]["enabled"]:
        families.append(
            ("catboost_wave_wind", all_features, catboost_grid, config["catboost"]["residual_scales"], fit_predict_catboost)
        )

    prediction_frames = []
    selection_log: dict[str, list[dict[str, Any]]] = {name: [] for name, *_ in families}
    for name, feature_columns, parameter_grid, scales, fit_predict in families:
        x = add_station_indicators(cases, feature_columns)
        residual_prediction = np.full_like(residual_target, np.nan, dtype=float)
        selected_scale = np.full(len(cases), np.nan, dtype=float)
        for outer_fold in sorted(set(folds.tolist())):
            outer_valid = folds == outer_fold
            outer_train = ~outer_valid
            selected_params, scale, inner_rows = nested_select(
                x,
                residual_target,
                actual,
                champion,
                folds,
                outer_train,
                parameter_grid,
                scales,
                fit_predict,
            )
            raw_prediction = fit_predict(
                x.loc[outer_train], residual_target[outer_train], x.loc[outer_valid], selected_params
            )
            residual_prediction[outer_valid] = raw_prediction
            selected_scale[outer_valid] = scale
            selection_log[name].append(
                {
                    "outer_fold": outer_fold,
                    "train_cases": int(outer_train.sum()),
                    "valid_cases": int(outer_valid.sum()),
                    "selected_params": selected_params,
                    "selected_residual_scale": float(scale),
                    "inner_trials": inner_rows,
                }
            )
        if np.isnan(residual_prediction).any() or np.isnan(selected_scale).any():
            raise RuntimeError(f"incomplete outer prediction for {name}")
        for case_index, case in cases.iterrows():
            for lead_index, lead in enumerate(LEADS):
                prediction_frames.append(
                    {
                        "model": name,
                        "fold": case["fold"],
                        "anchor_id": case["anchor_id"],
                        "station": case["station"],
                        "lead_h": lead,
                        "target_hs": actual[case_index, lead_index],
                        "champion_prediction": champion[case_index, lead_index],
                        "raw_residual_prediction": residual_prediction[case_index, lead_index],
                        "selected_residual_scale": selected_scale[case_index],
                        "candidate_prediction": champion[case_index, lead_index]
                        + selected_scale[case_index] * residual_prediction[case_index, lead_index],
                    }
                )

    predictions = pd.DataFrame(prediction_frames)
    predictions.to_parquet(output_dir / "outer_predictions.parquet", index=False)
    metrics = {
        name: evaluate_model(
            predictions[predictions["model"] == name].copy(),
            config["bootstrap"],
            config["promotion_gate"],
        )
        for name, *_ in families
    }
    ordered_models = sorted(metrics, key=lambda name: metrics[name]["delta_rmse"])
    best_model = ordered_models[0]
    any_pass = any(model_metrics["promotion_gate_pass"] for model_metrics in metrics.values())
    result = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "status": "STRUCTURAL_SIGNAL_DETECTED_NO_SUBMISSION" if any_pass else "NO_GO_CLOSE_WIND_RESIDUAL_DIRECTION",
        "research_only": True,
        "complete_case_contract": {
            "all_champion_cases": 181,
            "expanded_wind_lag_complete_cases": int(len(cases)),
            "by_fold_station": {
                f"{fold}|{station}": int(len(group))
                for (fold, station), group in cases.groupby(["fold", "station"], sort=True)
            },
        },
        "metrics": metrics,
        "research_ranking": ordered_models,
        "best_research_model": best_model,
        "promotion_gate_passed_by": [name for name, values in metrics.items() if values["promotion_gate_pass"]],
        "selection_log": selection_log,
        "interpretation_contract": {
            "outer_predictions_only": True,
            "hyperparameters_and_residual_scale_selected_inside_outer_training_folds": True,
            "research_ranking_is_not_a_deployment_selection": True,
            "official_probe_required_before_transport_claim": True,
            "candidate_csv_creation": "prohibited in this pilot",
        },
        "scope_guards": config["scope_guards"],
        "inputs": {
            "runner": {"path": str(Path(__file__)).replace("\\", "/"), "sha256": sha256_file(Path(__file__))},
            "config": {"path": str(config_path).replace("\\", "/"), "sha256": sha256_file(config_path)},
            "source_features": {"path": config["source_features"], "sha256": sha256_file(Path(config["source_features"]))},
            "anchor_metadata": {"path": config["anchor_metadata"], "sha256": sha256_file(Path(config["anchor_metadata"]))},
            "champion_replay": {"path": config["champion_replay"], "sha256": sha256_file(Path(config["champion_replay"]))},
        },
        "outputs": {
            "outer_predictions": "outer_predictions.parquet",
            "result": "result.json"
        }
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "complete_cases": len(cases),
        "best_research_model": best_model,
        "promotion_gate_passed_by": result["promotion_gate_passed_by"],
        "model_deltas": {name: values["delta_rmse"] for name, values in metrics.items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
