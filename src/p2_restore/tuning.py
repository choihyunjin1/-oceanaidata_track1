"""Bounded nested LightGBM tuning for the P2 lean-M2 arm."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from p2_restore.features import TARGET_LAYERS, FeatureTable
from p2_restore.model import P2Model, fit_model
from p2_restore.research import STABILITY_BLOCKS

Structure = Literal["shared", "layerwise"]


@dataclass
class TunedLeanModel:
    structure: Structure
    estimators: dict[str, object]
    feature_columns: tuple[str, ...]

    def predict(self, table: FeatureTable) -> np.ndarray:
        if table.feature_columns != self.feature_columns:
            raise ValueError("tuned P2 feature schema differs from fitted model")
        baseline = table.frame["baseline"].to_numpy(float)
        if self.structure == "shared":
            residual = self.estimators["shared"].predict(table.frame.loc[:, self.feature_columns])
        else:
            residual = np.full(len(table.frame), np.nan, dtype=float)
            layers = table.frame["layer"].to_numpy(int)
            for layer in TARGET_LAYERS:
                keep = layers == layer
                residual[keep] = self.estimators[str(layer)].predict(
                    table.frame.loc[keep, self.feature_columns]
                )
        if not np.isfinite(residual).all():
            raise ValueError("tuned P2 model produced non-finite residuals")
        return np.clip(baseline + residual, -5.0, 45.0)


@dataclass
class P2TunedBlendModel:
    base_model: P2Model
    tuned_lean_model: TunedLeanModel
    weight: float = 0.5

    def predict(self, base: FeatureTable, lean: FeatureTable) -> np.ndarray:
        if self.weight != 0.5:
            raise ValueError("P2 tuned blend weight is frozen at 0.5")
        return 0.5 * self.base_model.predict(base) + 0.5 * self.tuned_lean_model.predict(lean)


def time_mask(table: FeatureTable, start: str, stop: str) -> np.ndarray:
    time = pd.to_datetime(table.frame["time"], utc=True)
    left = pd.Timestamp(start, tz="Asia/Seoul").tz_convert("UTC")
    right = pd.Timestamp(stop, tz="Asia/Seoul").tz_convert("UTC")
    return (time.ge(left) & time.lt(right)).to_numpy()


def _current_layerwise_predict(
    table: FeatureTable, train_rows: np.ndarray, predict_rows: np.ndarray, *, seed: int
) -> np.ndarray:
    validation = FeatureTable(
        table.frame.loc[predict_rows].reset_index(drop=True), table.feature_columns
    )
    output = np.full(len(validation.frame), np.nan, dtype=float)
    train_layers = table.frame["layer"].to_numpy(int)
    validation_layers = validation.frame["layer"].to_numpy(int)
    for layer in TARGET_LAYERS:
        fitted = fit_model(table, train_rows & (train_layers == layer), seed=seed + layer)
        keep = validation_layers == layer
        subset = FeatureTable(
            validation.frame.loc[keep].reset_index(drop=True), validation.feature_columns
        )
        output[keep] = fitted.predict(subset)
    return output


def screen_structures(
    base: FeatureTable,
    lean: FeatureTable,
    development_blocks: dict[str, dict[str, list[str]]],
) -> dict[str, object]:
    """Choose shared versus layerwise using only development score months."""

    truth_parts: list[np.ndarray] = []
    shared_parts: list[np.ndarray] = []
    layerwise_parts: list[np.ndarray] = []
    blocks: dict[str, object] = {}
    for number, (name, windows) in enumerate(development_blocks.items()):
        early = time_mask(base, *windows["early_stop"])
        score = time_mask(base, *windows["score"])
        train = ~(early | score)
        seed = 20260816 + number
        base_score = FeatureTable(
            base.frame.loc[score].reset_index(drop=True), base.feature_columns
        )
        lean_score = FeatureTable(
            lean.frame.loc[score].reset_index(drop=True), lean.feature_columns
        )
        base_prediction = fit_model(base, train, seed=seed).predict(base_score)
        shared_prediction = fit_model(lean, train, seed=seed).predict(lean_score)
        layerwise_prediction = _current_layerwise_predict(lean, train, score, seed=seed)
        truth = base_score.frame["target"].to_numpy(float)
        shared_blend = 0.5 * base_prediction + 0.5 * shared_prediction
        layerwise_blend = 0.5 * base_prediction + 0.5 * layerwise_prediction

        def rmse(prediction: np.ndarray, expected: np.ndarray = truth) -> float:
            return float(np.sqrt(np.mean((prediction - expected) ** 2)))

        blocks[name] = {
            "rows": int(score.sum()),
            "shared_rmse": rmse(shared_blend),
            "layerwise_rmse": rmse(layerwise_blend),
        }
        truth_parts.append(truth)
        shared_parts.append(shared_blend)
        layerwise_parts.append(layerwise_blend)
    truth = np.concatenate(truth_parts)
    shared = np.concatenate(shared_parts)
    layerwise = np.concatenate(layerwise_parts)
    shared_rmse = float(np.sqrt(np.mean((shared - truth) ** 2)))
    layerwise_rmse = float(np.sqrt(np.mean((layerwise - truth) ** 2)))
    winner: Structure = "layerwise" if layerwise_rmse < shared_rmse else "shared"
    return {
        "blocks": blocks,
        "aggregate": {
            "shared_rmse": shared_rmse,
            "layerwise_rmse": layerwise_rmse,
            "delta_layerwise_minus_shared": layerwise_rmse - shared_rmse,
        },
        "winner": winner,
    }


def _estimator(parameters: dict[str, object], *, seed: int, iterations: int):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="regression_l2",
        n_estimators=iterations,
        learning_rate=float(parameters["learning_rate"]),
        num_leaves=int(parameters["num_leaves"]),
        max_depth=int(parameters["max_depth"]),
        min_child_samples=int(parameters["min_child_samples"]),
        colsample_bytree=float(parameters["feature_fraction"]),
        subsample=float(parameters["bagging_fraction"]),
        subsample_freq=1,
        reg_alpha=float(parameters["reg_alpha"]),
        reg_lambda=float(parameters["reg_lambda"]),
        min_split_gain=float(parameters["min_split_gain"]),
        max_bin=int(parameters["max_bin"]),
        random_state=seed,
        n_jobs=8,
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )


def fit_tuned_lean(
    table: FeatureTable,
    train_rows: np.ndarray,
    *,
    structure: Structure,
    parameters: dict[str, object],
    seed: int,
    iterations: dict[str, int] | None = None,
    early_stop_rows: np.ndarray | None = None,
    early_stopping_rounds: int = 200,
    early_stopping_min_delta: float = 0.00001,
) -> tuple[TunedLeanModel, dict[str, int]]:
    """Fit one shared or three layerwise models with optional early stopping."""

    from lightgbm import early_stopping

    train_rows = np.asarray(train_rows, dtype=bool)
    if len(train_rows) != len(table.frame) or not train_rows.any():
        raise ValueError("invalid tuned-model training mask")
    if early_stop_rows is not None:
        early_stop_rows = np.asarray(early_stop_rows, dtype=bool)
        if len(early_stop_rows) != len(table.frame) or not early_stop_rows.any():
            raise ValueError("invalid early-stopping mask")
        if np.any(train_rows & early_stop_rows):
            raise ValueError("training and early-stopping rows overlap")
    keys = ("shared",) if structure == "shared" else tuple(str(layer) for layer in TARGET_LAYERS)
    layers = table.frame["layer"].to_numpy(int)
    fitted: dict[str, object] = {}
    best: dict[str, int] = {}
    for key in keys:
        layer_mask = (
            np.ones(len(table.frame), dtype=bool) if key == "shared" else layers == int(key)
        )
        fit_mask = train_rows & layer_mask
        requested = 5000 if iterations is None else int(iterations[key])
        estimator = _estimator(
            parameters, seed=seed + (0 if key == "shared" else int(key)), iterations=requested
        )
        fit_kwargs: dict[str, object] = {}
        if early_stop_rows is not None:
            evaluation = early_stop_rows & layer_mask
            if not evaluation.any():
                raise ValueError(f"no early-stopping rows for model {key}")
            fit_kwargs = {
                "eval_set": [
                    (
                        table.frame.loc[evaluation, table.feature_columns],
                        table.frame.loc[evaluation, "residual"],
                    )
                ],
                "eval_metric": "rmse",
                "callbacks": [
                    early_stopping(
                        early_stopping_rounds,
                        first_metric_only=True,
                        verbose=False,
                        min_delta=early_stopping_min_delta,
                    )
                ],
            }
        estimator.fit(
            table.frame.loc[fit_mask, table.feature_columns],
            table.frame.loc[fit_mask, "residual"],
            **fit_kwargs,
        )
        fitted[key] = estimator
        best[key] = int(estimator.best_iteration_ or requested)
    return TunedLeanModel(structure, fitted, table.feature_columns), best


def freeze_best_iterations(fold_iterations: dict[str, list[int]]) -> dict[str, int]:
    frozen: dict[str, int] = {}
    for key, values in fold_iterations.items():
        valid = [int(value) for value in values if int(value) > 0]
        if not valid:
            raise ValueError(f"no positive best iteration for {key}")
        frozen[key] = int(np.median(valid).round())
    return frozen


def optimize_parameters(
    base: FeatureTable,
    lean: FeatureTable,
    development_blocks: dict[str, dict[str, list[str]]],
    *,
    structure: Structure,
    trials: int = 40,
    progress: Callable[[int, int, float], None] | None = None,
) -> dict[str, object]:
    """Tune one preselected structure on development blocks only."""

    import optuna

    prepared: list[dict[str, object]] = []
    for number, (name, windows) in enumerate(development_blocks.items()):
        early = time_mask(base, *windows["early_stop"])
        score = time_mask(base, *windows["score"])
        train = ~(early | score)
        seed = 20260816 + number
        base_score = FeatureTable(
            base.frame.loc[score].reset_index(drop=True), base.feature_columns
        )
        prepared.append(
            {
                "name": name,
                "train": train,
                "early": early,
                "score": score,
                "seed": seed,
                "base_prediction": fit_model(base, train, seed=seed).predict(base_score),
                "truth": base_score.frame["target"].to_numpy(float),
            }
        )

    def objective(trial: optuna.Trial) -> float:
        parameters = {
            "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.08, log=True),
            "num_leaves": trial.suggest_categorical("num_leaves", [15, 31, 63, 127]),
            "max_depth": trial.suggest_categorical("max_depth", [5, 7, 9, -1]),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 800, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.65, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0001, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),
            "max_bin": trial.suggest_categorical("max_bin", [127, 255, 511]),
        }
        truths: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        iteration_history: dict[str, list[int]] = {
            key: [] for key in (("shared",) if structure == "shared" else ("2", "3", "4"))
        }
        fold_metrics: dict[str, float] = {}
        for step, fold in enumerate(prepared):
            model, iterations = fit_tuned_lean(
                lean,
                fold["train"],
                structure=structure,
                parameters=parameters,
                seed=fold["seed"],
                early_stop_rows=fold["early"],
            )
            score_table = FeatureTable(
                lean.frame.loc[fold["score"]].reset_index(drop=True), lean.feature_columns
            )
            tuned_prediction = model.predict(score_table)
            blend = 0.5 * fold["base_prediction"] + 0.5 * tuned_prediction
            truth = fold["truth"]
            truths.append(truth)
            predictions.append(blend)
            for key, value in iterations.items():
                iteration_history[key].append(value)
            combined_truth = np.concatenate(truths)
            combined_prediction = np.concatenate(predictions)
            cumulative = float(np.sqrt(np.mean((combined_prediction - combined_truth) ** 2)))
            fold_metrics[fold["name"]] = float(np.sqrt(np.mean((blend - truth) ** 2)))
            trial.report(cumulative, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_iterations", iteration_history)
        trial.set_user_attr("fold_rmse", fold_metrics)
        return cumulative

    sampler = optuna.samplers.TPESampler(seed=20260816, multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if progress is not None:
            progress(len(study.trials), trials, float(study.best_value))

    study.optimize(objective, n_trials=trials, callbacks=[callback], gc_after_trial=True)
    best_trial = study.best_trial
    fold_iterations = best_trial.user_attrs["fold_iterations"]
    frozen_iterations = freeze_best_iterations(fold_iterations)
    trial_records = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
        }
        for trial in study.trials
    ]
    return {
        "structure": structure,
        "best_value": float(best_trial.value),
        "best_parameters": best_trial.params,
        "fold_iterations": fold_iterations,
        "frozen_iterations": frozen_iterations,
        "best_fold_rmse": best_trial.user_attrs["fold_rmse"],
        "trials": trial_records,
    }


def evaluate_guard_blocks(
    base: FeatureTable,
    lean: FeatureTable,
    guard_blocks: dict[str, list[str]],
    *,
    structure: Structure,
    parameters: dict[str, object],
    iterations: dict[str, int],
) -> tuple[dict[str, object], pd.DataFrame]:
    """Apply the frozen tuned model exactly once to the guard blocks."""

    reports: dict[str, object] = {}
    parts: list[pd.DataFrame] = []
    stability_names = list(STABILITY_BLOCKS)
    for name, window in guard_blocks.items():
        validation = time_mask(base, *window)
        train = ~validation
        seed = 20260816 + stability_names.index(name)
        base_validation = FeatureTable(
            base.frame.loc[validation].reset_index(drop=True), base.feature_columns
        )
        lean_validation = FeatureTable(
            lean.frame.loc[validation].reset_index(drop=True), lean.feature_columns
        )
        base_prediction = fit_model(base, train, seed=seed).predict(base_validation)
        current_lean = fit_model(lean, train, seed=seed).predict(lean_validation)
        tuned_model, _ = fit_tuned_lean(
            lean,
            train,
            structure=structure,
            parameters=parameters,
            iterations=iterations,
            seed=seed,
        )
        tuned_lean = tuned_model.predict(lean_validation)
        current = 0.5 * base_prediction + 0.5 * current_lean
        candidate = 0.5 * base_prediction + 0.5 * tuned_lean
        truth = base_validation.frame["target"].to_numpy(float)
        layer = base_validation.frame["layer"].to_numpy(int)

        def metric(
            prediction: np.ndarray,
            expected: np.ndarray = truth,
            target_layer: np.ndarray = layer,
        ) -> dict[str, object]:
            error = prediction - expected
            return {
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
                "by_layer": {
                    str(target): float(np.sqrt(np.mean(error[target_layer == target] ** 2)))
                    for target in TARGET_LAYERS
                },
            }

        reports[name] = {
            "rows": int(validation.sum()),
            "current_blend50": metric(current),
            "tuned_blend50": metric(candidate),
        }
        local_time = pd.to_datetime(base_validation.frame["time"], utc=True).dt.tz_convert(
            "Asia/Seoul"
        )
        parts.append(
            pd.DataFrame(
                {
                    "time": local_time,
                    "layer": layer,
                    "truth": truth,
                    "current_blend50": current,
                    "tuned_blend50": candidate,
                }
            )
        )
    oof = pd.concat(parts, ignore_index=True)
    oof["day"] = oof["time"].dt.floor("D").astype(str)
    oof["month"] = oof["time"].dt.strftime("%Y-%m")
    return reports, oof


def fit_final_tuned_blend(
    base: FeatureTable,
    lean: FeatureTable,
    *,
    structure: Structure,
    parameters: dict[str, object],
    iterations: dict[str, int],
) -> P2TunedBlendModel:
    rows = np.ones(len(base.frame), dtype=bool)
    base_model = fit_model(base, seed=20260816)
    tuned_model, _ = fit_tuned_lean(
        lean,
        rows,
        structure=structure,
        parameters=parameters,
        iterations=iterations,
        seed=20260816,
    )
    return P2TunedBlendModel(base_model, tuned_model)
