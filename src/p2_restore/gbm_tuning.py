"""Nested, convergence-aware tuning for the three leading P2 GBM families."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from p2_restore.features import TARGET_LAYERS, FeatureTable
from p2_restore.gbm_tournament import (
    GBMArmSpec,
    GBMResidualModel,
    _features,
    _rmse,
    _validate_table,
)
from p2_restore.model import VALIDATION_BLOCKS

TUNING_FAMILIES = ("catboost_layerwise", "catboost_pooled", "lgbm_dart")
INNER_BLOCKS = {
    "2024_sep_oct": ("2024-07-01", "2024-09-01"),
    "2025_jul_aug": ("2025-05-01", "2025-07-01"),
    "2025_nov_dec": ("2025-07-01", "2025-09-01"),
}
CATBOOST_MAX_ITERATIONS = 3_000
CATBOOST_EARLY_STOPPING = 150


def _family_spec(family: str) -> GBMArmSpec:
    if family == "catboost_layerwise":
        return GBMArmSpec(family, "catboost", layerwise=True)
    if family == "catboost_pooled":
        return GBMArmSpec(family, "catboost", categorical_layer=True)
    if family == "lgbm_dart":
        return GBMArmSpec(family, "lightgbm_dart")
    raise ValueError(f"unsupported tuning family: {family}")


def _time_mask(time: pd.Series, start: str, stop: str) -> np.ndarray:
    left = pd.Timestamp(start, tz="Asia/Seoul").tz_convert("UTC")
    right = pd.Timestamp(stop, tz="Asia/Seoul").tz_convert("UTC")
    return (time.ge(left) & time.lt(right)).to_numpy()


def nested_masks(table: FeatureTable) -> dict[str, dict[str, np.ndarray]]:
    """Return outer and inner masks without using target values."""

    _validate_table(table)
    time = pd.to_datetime(table.frame["time"], utc=True)
    result: dict[str, dict[str, np.ndarray]] = {}
    for outer, (start, stop) in VALIDATION_BLOCKS.items():
        outer_validation = _time_mask(time, start, stop)
        inner_start, inner_stop = INNER_BLOCKS[outer]
        inner_validation = _time_mask(time, inner_start, inner_stop) & ~outer_validation
        inner_fit = ~outer_validation & ~inner_validation
        if not outer_validation.any() or not inner_validation.any() or not inner_fit.any():
            raise ValueError(f"nested tuning split is empty: {outer}")
        if (outer_validation & inner_validation).any() or (inner_fit & outer_validation).any():
            raise AssertionError("nested tuning masks overlap")
        result[outer] = {
            "outer_validation": outer_validation,
            "inner_validation": inner_validation,
            "inner_fit": inner_fit,
            "outer_fit": ~outer_validation,
        }
    return result


def sample_parameters(family: str, trial) -> dict[str, object]:
    """Sample the fixed search space for one family."""

    if family.startswith("catboost"):
        bootstrap = trial.suggest_categorical("bootstrap_type", ["MVS", "Bayesian", "Bernoulli"])
        parameters: dict[str, object] = {
            "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.12, log=True),
            "depth": trial.suggest_int("depth", 5, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 40.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 3.0, log=True),
            "rsm": trial.suggest_float("rsm", 0.55, 1.0),
            "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
            "bootstrap_type": bootstrap,
        }
        if bootstrap == "Bayesian":
            parameters["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 2.0)
        else:
            parameters["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        return parameters
    if family == "lgbm_dart":
        return {
            "n_estimators": trial.suggest_categorical(
                "n_estimators", [200, 400, 600, 800, 1_200, 1_600, 2_400, 3_000]
            ),
            "learning_rate": trial.suggest_float("learning_rate", 0.006, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 5, 7, 9, 12]),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 500, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.55, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [0, 1, 5]),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 30.0, log=True),
            "drop_rate": trial.suggest_float("drop_rate", 0.03, 0.25),
            "skip_drop": trial.suggest_float("skip_drop", 0.2, 0.85),
            "max_drop": trial.suggest_int("max_drop", 20, 100),
        }
    raise ValueError(f"unsupported tuning family: {family}")


def _catboost_estimator(parameters: dict[str, object], *, seed: int, iterations: int, threads: int):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=int(iterations),
        random_seed=seed,
        task_type="CPU",
        thread_count=threads,
        verbose=False,
        allow_writing_files=False,
        border_count=128,
        **parameters,
    )


def _dart_estimator(parameters: dict[str, object], *, seed: int, threads: int):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="regression_l2",
        boosting_type="dart",
        random_state=seed,
        n_jobs=threads,
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
        uniform_drop=False,
        drop_seed=seed + 17,
        **parameters,
    )


@dataclass
class TunedFit:
    model: GBMResidualModel
    iterations: int | dict[str, int]


def fit_tuned_model(
    table: FeatureTable,
    family: str,
    parameters: dict[str, object],
    training_rows: np.ndarray,
    *,
    validation_rows: np.ndarray | None = None,
    iterations: int | dict[str, int] | None = None,
    seed: int = 20260816,
    threads: int = 2,
) -> TunedFit:
    """Fit a tuned family, using validation only for CatBoost convergence."""

    _validate_table(table)
    training = np.asarray(training_rows, dtype=bool)
    validation = None if validation_rows is None else np.asarray(validation_rows, dtype=bool)
    if training.shape != (len(table.frame),) or not training.any():
        raise ValueError("tuned GBM training mask is invalid")
    if validation is not None:
        if (
            validation.shape != training.shape
            or not validation.any()
            or (training & validation).any()
        ):
            raise ValueError("tuned GBM validation mask is invalid")
    spec = _family_spec(family)
    inputs = _features(table, spec)
    target = table.frame["residual"].to_numpy(float)
    estimators: dict[int | str, object] = {}
    best_iterations: dict[str, int] = {}
    if family.startswith("catboost"):
        if spec.layerwise:
            for layer in TARGET_LAYERS:
                fit_rows = training & table.frame["layer"].eq(layer).to_numpy()
                if validation is None:
                    layer_iterations = int(dict(iterations or {})[str(layer)])
                else:
                    layer_iterations = CATBOOST_MAX_ITERATIONS
                estimator = _catboost_estimator(
                    parameters,
                    seed=seed + layer,
                    iterations=layer_iterations,
                    threads=threads,
                )
                kwargs: dict[str, object] = {}
                if validation is not None:
                    validation_layer = validation & table.frame["layer"].eq(layer).to_numpy()
                    kwargs = {
                        "eval_set": (inputs.loc[validation_layer], target[validation_layer]),
                        "early_stopping_rounds": CATBOOST_EARLY_STOPPING,
                        "use_best_model": True,
                    }
                estimator.fit(inputs.loc[fit_rows], target[fit_rows], **kwargs)
                estimators[layer] = estimator
                best_iterations[str(layer)] = (
                    int(estimator.get_best_iteration()) + 1
                    if validation is not None
                    else layer_iterations
                )
        else:
            pooled_iterations = (
                CATBOOST_MAX_ITERATIONS if validation is not None else int(iterations or 0)
            )
            if pooled_iterations < 1:
                raise ValueError("pooled CatBoost iterations are invalid")
            estimator = _catboost_estimator(
                parameters, seed=seed, iterations=pooled_iterations, threads=threads
            )
            kwargs = {"cat_features": ["layer_cat"]}
            if validation is not None:
                kwargs.update(
                    {
                        "eval_set": (inputs.loc[validation], target[validation]),
                        "early_stopping_rounds": CATBOOST_EARLY_STOPPING,
                        "use_best_model": True,
                    }
                )
            estimator.fit(inputs.loc[training], target[training], **kwargs)
            estimators["pooled"] = estimator
            best_iterations["pooled"] = (
                int(estimator.get_best_iteration()) + 1
                if validation is not None
                else pooled_iterations
            )
    else:
        if validation is not None and iterations is not None:
            raise ValueError("DART does not use a separate convergence override")
        estimator = _dart_estimator(parameters, seed=seed, threads=threads)
        estimator.fit(inputs.loc[training], target[training])
        estimators["pooled"] = estimator
        best_iterations["pooled"] = int(parameters["n_estimators"])
    model = GBMResidualModel(spec, table.feature_columns, estimators)
    collapsed: int | dict[str, int] = (
        best_iterations if spec.layerwise else best_iterations["pooled"]
    )
    return TunedFit(model, collapsed)


def _subset(table: FeatureTable, rows: np.ndarray) -> FeatureTable:
    return FeatureTable(table.frame.loc[rows].reset_index(drop=True), table.feature_columns)


def consensus_iterations(family: str, values: list[int | dict[str, int]]) -> int | dict[str, int]:
    if not values:
        raise ValueError("iteration consensus is empty")
    if family == "catboost_layerwise":
        return {
            str(layer): int(np.median([dict(value)[str(layer)] for value in values]))
            for layer in TARGET_LAYERS
        }
    return int(np.median([int(value) for value in values]))


def tune_family(
    table: FeatureTable,
    family: str,
    storage: Path,
    *,
    trials: int = 36,
    threads: int = 2,
    seed: int = 20260816,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> tuple[dict[str, object], pd.DataFrame, GBMResidualModel]:
    """Tune independently inside each outer fold, then evaluate it exactly once."""

    import optuna

    if family not in TUNING_FAMILIES or trials < 3 or trials % 3:
        raise ValueError("family or trial budget is invalid")
    masks = nested_masks(table)
    storage.parent.mkdir(parents=True, exist_ok=True)
    trials_per_outer = trials // len(masks)
    selected: dict[str, dict[str, object]] = {}
    studies = {}
    for fold_number, (outer, fold_masks) in enumerate(masks.items()):
        study = optuna.create_study(
            study_name=f"{family}__{outer}__nested_v2",
            storage=f"sqlite:///{storage.as_posix()}",
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed + fold_number * 100, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
            load_if_exists=True,
        )

        def objective(
            trial,
            *,
            _fold_number: int = fold_number,
            _outer: str = outer,
            _fold_masks: dict[str, np.ndarray] = fold_masks,
        ) -> float:
            parameters = sample_parameters(family, trial)
            if progress:
                progress(
                    {
                        "phase": "search",
                        "trial": _fold_number * trials_per_outer + trial.number + 1,
                        "trials": trials,
                        "fold": _outer,
                        "fold_number": _fold_number + 1,
                    }
                )
            fitted = fit_tuned_model(
                table,
                family,
                parameters,
                _fold_masks["inner_fit"],
                validation_rows=_fold_masks["inner_validation"],
                seed=seed + _fold_number * 100,
                threads=threads,
            )
            validation_table = _subset(table, _fold_masks["inner_validation"])
            prediction = fitted.model.predict(validation_table)
            truth = table.frame.loc[_fold_masks["inner_validation"], "target"].to_numpy(float)
            score = _rmse(truth, prediction)
            trial.set_user_attr("iterations", fitted.iterations)
            trial.report(score, 0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return score

        # The preregistered budget counts every suggested trial, including pruned trials.
        remaining = max(0, trials_per_outer - len(study.trials))
        if remaining:
            study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=True)
        best = study.best_trial
        selected[outer] = {
            "parameters": dict(best.params),
            "iterations": best.user_attrs["iterations"],
            "best_trial": int(best.number),
            "best_inner_rmse": float(best.value),
        }
        studies[outer] = study

    parts: list[pd.DataFrame] = []
    outer_metrics: dict[str, object] = {}
    for fold_number, (outer, fold_masks) in enumerate(masks.items()):
        chosen = selected[outer]
        selected_iterations = chosen["iterations"]
        if progress:
            progress(
                {
                    "phase": "outer",
                    "trial": trials,
                    "trials": trials,
                    "fold": outer,
                    "fold_number": fold_number + 1,
                }
            )
        fitted = fit_tuned_model(
            table,
            family,
            dict(chosen["parameters"]),
            fold_masks["outer_fit"],
            iterations=selected_iterations if family.startswith("catboost") else None,
            seed=seed + fold_number * 100,
            threads=threads,
        )
        outer_table = _subset(table, fold_masks["outer_validation"])
        prediction = fitted.model.predict(outer_table)
        truth = table.frame.loc[fold_masks["outer_validation"], "target"].to_numpy(float)
        layer = table.frame.loc[fold_masks["outer_validation"], "layer"].to_numpy(int)
        outer_metrics[outer] = {
            "rows": len(truth),
            "rmse": _rmse(truth, prediction),
            "iterations": selected_iterations,
            "by_layer_rmse": {
                str(target): _rmse(truth[layer == target], prediction[layer == target])
                for target in TARGET_LAYERS
            },
        }
        parts.append(
            pd.DataFrame(
                {
                    "time": pd.to_datetime(
                        table.frame.loc[fold_masks["outer_validation"], "time"], utc=True
                    )
                    .dt.tz_convert("Asia/Seoul")
                    .to_numpy(),
                    "layer": layer,
                    "truth": truth,
                    "block": outer,
                    "prediction": prediction,
                }
            )
        )
    oof = pd.concat(parts, ignore_index=True).sort_values(["time", "layer"]).reset_index(drop=True)
    # The final parameter set is selected without outer scores: use the inner
    # choice whose validation period is immediately before the hidden season.
    final_parameter_source = "2025_nov_dec"
    final_parameters = dict(selected[final_parameter_source]["parameters"])
    iteration_values = [value["iterations"] for value in selected.values()]
    final_iterations = (
        consensus_iterations(family, iteration_values)
        if family.startswith("catboost")
        else int(final_parameters["n_estimators"])
    )
    full_model = fit_tuned_model(
        table,
        family,
        final_parameters,
        np.ones(len(table.frame), dtype=bool),
        iterations=final_iterations if family.startswith("catboost") else None,
        seed=seed,
        threads=threads,
    ).model
    truth = oof["truth"].to_numpy(float)
    prediction = oof["prediction"].to_numpy(float)
    summary = {
        "family": family,
        "requested_trials": trials,
        "trials_per_outer": trials_per_outer,
        "study_trials": sum(len(study.trials) for study in studies.values()),
        "complete_trials": sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            for study in studies.values()
            for trial in study.trials
        ),
        "pruned_trials": sum(
            trial.state == optuna.trial.TrialState.PRUNED
            for study in studies.values()
            for trial in study.trials
        ),
        "best_trial_by_outer_fold": {
            outer: value["best_trial"] for outer, value in selected.items()
        },
        "best_inner_rmse": float(
            np.mean([value["best_inner_rmse"] for value in selected.values()])
        ),
        "best_inner_rmse_by_outer_fold": {
            outer: value["best_inner_rmse"] for outer, value in selected.items()
        },
        "best_parameters_by_outer_fold": {
            outer: value["parameters"] for outer, value in selected.items()
        },
        "best_iterations_by_outer_fold": {
            outer: value["iterations"] for outer, value in selected.items()
        },
        "final_parameter_selection": "2025_nov_dec inner choice; outer scores unused",
        "final_parameter_source": final_parameter_source,
        "best_parameters": final_parameters,
        "full_fit_iterations": final_iterations,
        "outer_rows": len(oof),
        "outer_rmse": _rmse(truth, prediction),
        "outer_by_layer_rmse": {
            str(layer): _rmse(
                truth[oof["layer"].to_numpy(int) == layer],
                prediction[oof["layer"].to_numpy(int) == layer],
            )
            for layer in TARGET_LAYERS
        },
        "outer_blocks": outer_metrics,
    }
    return summary, oof, full_model
