"""Fixed-budget GBM-family comparison for P2 profile restoration.

The module deliberately keeps feature engineering and validation blocks fixed.
It is a structure screen, not a hyperparameter search: every arm uses 400
boosting iterations and predicts the residual over the public-depth linear
interpolation baseline.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from p2_restore.features import TARGET_LAYERS, FeatureTable
from p2_restore.model import VALIDATION_BLOCKS

KEY_COLUMNS = ("time", "layer", "block")


@dataclass(frozen=True)
class GBMArmSpec:
    name: str
    backend: str
    iterations: int = 400
    layerwise: bool = False
    categorical_layer: bool = False


GBM_ARM_SPECS = (
    GBMArmSpec("lgbm_gbdt", "lightgbm"),
    GBMArmSpec("lgbm_extra_trees", "lightgbm_extra_trees"),
    GBMArmSpec("lgbm_dart", "lightgbm_dart"),
    GBMArmSpec("xgboost_hist", "xgboost"),
    GBMArmSpec("catboost_pooled", "catboost", categorical_layer=True),
    GBMArmSpec("catboost_layerwise", "catboost", layerwise=True),
)


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or not np.isfinite(truth).all():
        raise ValueError("RMSE inputs are invalid")
    if not np.isfinite(prediction).all():
        raise ValueError("prediction contains non-finite values")
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _validate_table(table: FeatureTable) -> None:
    required = {"station", "layer", "time", "baseline"}
    if missing := required.difference(table.frame.columns):
        raise ValueError(f"feature table is missing columns: {sorted(missing)}")
    if not table.feature_columns or len(set(table.feature_columns)) != len(table.feature_columns):
        raise ValueError("feature schema is empty or duplicated")
    if missing := set(table.feature_columns).difference(table.frame.columns):
        raise ValueError(f"feature columns are absent from frame: {sorted(missing)}")
    forbidden = {"target", "residual", "temp_2", "temp_3", "temp_4", "psal_2", "psal_3", "psal_4"}
    if forbidden.intersection(table.feature_columns):
        raise ValueError("target-layer values or labels leaked into model features")
    layers = set(table.frame["layer"].astype(int).unique())
    if not layers.issubset(set(TARGET_LAYERS)):
        raise ValueError("feature table includes a non-target layer")


def _features(table: FeatureTable, spec: GBMArmSpec) -> pd.DataFrame:
    values = table.frame.loc[:, table.feature_columns].copy()
    if spec.categorical_layer:
        values["layer_cat"] = table.frame["layer"].astype(int).astype(str).to_numpy()
    return values


def _make_estimator(spec: GBMArmSpec, *, seed: int):
    common_lgbm = {
        "objective": "regression_l2",
        "n_estimators": spec.iterations,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": 7,
        "min_child_samples": 200,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.2,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": 8,
        "verbosity": -1,
        "deterministic": True,
        "force_row_wise": True,
    }
    if spec.backend.startswith("lightgbm"):
        from lightgbm import LGBMRegressor

        if spec.backend == "lightgbm_extra_trees":
            common_lgbm.update(
                {
                    "extra_trees": True,
                    "subsample_freq": 1,
                    "extra_seed": seed + 17,
                    "bagging_seed": seed + 29,
                    "feature_fraction_seed": seed + 43,
                }
            )
        elif spec.backend == "lightgbm_dart":
            common_lgbm.update(
                {
                    "boosting_type": "dart",
                    "drop_rate": 0.1,
                    "skip_drop": 0.5,
                    "max_drop": 50,
                    "uniform_drop": False,
                    "drop_seed": seed + 17,
                }
            )
        return LGBMRegressor(**common_lgbm)
    if spec.backend == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=spec.iterations,
            learning_rate=0.04,
            max_depth=7,
            min_child_weight=20.0,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=1.0,
            tree_method="hist",
            max_bin=255,
            random_state=seed,
            n_jobs=8,
            verbosity=0,
        )
    if spec.backend == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            loss_function="RMSE",
            iterations=spec.iterations,
            learning_rate=0.04,
            depth=8,
            l2_leaf_reg=3.0,
            random_strength=0.5,
            rsm=0.85,
            bootstrap_type="MVS",
            random_seed=seed,
            task_type="CPU",
            thread_count=8,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"unknown GBM backend: {spec.backend}")


@dataclass
class GBMResidualModel:
    spec: GBMArmSpec
    feature_columns: tuple[str, ...]
    estimators: dict[int | str, object]

    def predict(self, table: FeatureTable) -> np.ndarray:
        _validate_table(table)
        if table.feature_columns != self.feature_columns:
            raise ValueError("P2 feature schema differs from fitted GBM model")
        inputs = _features(table, self.spec)
        residual = np.full(len(table.frame), np.nan, dtype=float)
        if self.spec.layerwise:
            for layer in TARGET_LAYERS:
                selected = table.frame["layer"].to_numpy(int) == layer
                residual[selected] = np.asarray(
                    self.estimators[layer].predict(inputs.loc[selected]), dtype=float
                )
        else:
            residual = np.asarray(self.estimators["pooled"].predict(inputs), dtype=float)
        if not np.isfinite(residual).all():
            raise ValueError("GBM residual prediction contains non-finite values")
        return np.clip(table.frame["baseline"].to_numpy(float) + residual, -5.0, 45.0)


def fit_gbm_model(
    table: FeatureTable,
    spec: GBMArmSpec,
    rows: np.ndarray | None = None,
    *,
    seed: int = 20260816,
) -> GBMResidualModel:
    _validate_table(table)
    selected = (
        np.ones(len(table.frame), dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
    )
    if selected.shape != (len(table.frame),) or not selected.any():
        raise ValueError("GBM training mask is invalid")
    if "residual" not in table.frame:
        raise ValueError("training table has no residual target")
    inputs = _features(table, spec)
    targets = table.frame["residual"].to_numpy(float)
    estimators: dict[int | str, object] = {}
    if spec.layerwise:
        for layer in TARGET_LAYERS:
            keep = selected & table.frame["layer"].eq(layer).to_numpy()
            if not keep.any():
                raise ValueError(f"layerwise GBM has no training rows for layer {layer}")
            estimator = _make_estimator(spec, seed=seed + layer)
            estimator.fit(inputs.loc[keep], targets[keep])
            estimators[layer] = estimator
    else:
        estimator = _make_estimator(spec, seed=seed)
        fit_kwargs: dict[str, object] = {}
        if spec.categorical_layer:
            fit_kwargs["cat_features"] = ["layer_cat"]
        estimator.fit(inputs.loc[selected], targets[selected], **fit_kwargs)
        estimators["pooled"] = estimator
    return GBMResidualModel(spec, table.feature_columns, estimators)


def _subset(table: FeatureTable, selected: np.ndarray) -> FeatureTable:
    return FeatureTable(table.frame.loc[selected].reset_index(drop=True), table.feature_columns)


def run_blocked_arm(
    table: FeatureTable,
    spec: GBMArmSpec,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Run the exact three P2 target-proxy blocks for one fixed arm."""

    _validate_table(table)
    time = pd.to_datetime(table.frame["time"], utc=True)
    parts: list[pd.DataFrame] = []
    reports: dict[str, object] = {}
    for number, (block, (start, stop)) in enumerate(VALIDATION_BLOCKS.items()):
        left = pd.Timestamp(start, tz="Asia/Seoul").tz_convert("UTC")
        right = pd.Timestamp(stop, tz="Asia/Seoul").tz_convert("UTC")
        validation = (time.ge(left) & time.lt(right)).to_numpy()
        if not validation.any():
            raise ValueError(f"validation block has no rows: {block}")
        if progress:
            progress(number + 1, len(VALIDATION_BLOCKS), block)
        model = fit_gbm_model(table, spec, ~validation, seed=20260816 + number * 100)
        prediction = model.predict(_subset(table, validation))
        truth = table.frame.loc[validation, "target"].to_numpy(float)
        baseline = table.frame.loc[validation, "baseline"].to_numpy(float)
        layer = table.frame.loc[validation, "layer"].to_numpy(int)
        reports[block] = {
            "rows": int(validation.sum()),
            "rmse": _rmse(truth, prediction),
            "baseline_rmse": _rmse(truth, baseline),
            "by_layer_rmse": {
                str(target): _rmse(truth[layer == target], prediction[layer == target])
                for target in TARGET_LAYERS
            },
        }
        parts.append(
            pd.DataFrame(
                {
                    "time": pd.to_datetime(table.frame.loc[validation, "time"], utc=True)
                    .dt.tz_convert("Asia/Seoul")
                    .to_numpy(),
                    "layer": layer,
                    "truth": truth,
                    "baseline": baseline,
                    "block": block,
                    "prediction": prediction,
                }
            )
        )
    oof = pd.concat(parts, ignore_index=True).sort_values(["time", "layer"]).reset_index(drop=True)
    truth = oof["truth"].to_numpy(float)
    prediction = oof["prediction"].to_numpy(float)
    return (
        {
            "arm": spec.name,
            "backend": spec.backend,
            "iterations": spec.iterations,
            "layerwise": spec.layerwise,
            "categorical_layer": spec.categorical_layer,
            "rows": len(oof),
            "rmse": _rmse(truth, prediction),
            "by_layer_rmse": {
                str(layer): _rmse(
                    truth[oof["layer"].to_numpy(int) == layer],
                    prediction[oof["layer"].to_numpy(int) == layer],
                )
                for layer in TARGET_LAYERS
            },
            "blocks": reports,
        },
        oof,
    )


def align_with_deep_stack(deep: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Align a GBM OOF with the frozen deep-stack OOF and verify its grain."""

    required_deep = {"time", "layer", "truth", "block", "prediction", "lobo_prediction"}
    required_candidate = {"time", "layer", "truth", "block", "prediction"}
    if missing := required_deep.difference(deep.columns):
        raise ValueError(f"deep stack OOF is missing columns: {sorted(missing)}")
    if missing := required_candidate.difference(candidate.columns):
        raise ValueError(f"candidate OOF is missing columns: {sorted(missing)}")
    left = deep.copy()
    right = candidate.copy()
    left["time"] = pd.to_datetime(left["time"], utc=True)
    right["time"] = pd.to_datetime(right["time"], utc=True)
    left = left.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    right = right.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    if left[list(KEY_COLUMNS)].duplicated().any() or right[list(KEY_COLUMNS)].duplicated().any():
        raise ValueError("OOF keys are duplicated")
    if not left[[*KEY_COLUMNS, "truth"]].equals(right[[*KEY_COLUMNS, "truth"]]):
        raise ValueError("candidate and deep-stack OOF grain or truth differs")
    return left.rename(
        columns={"prediction": "deep_prediction", "lobo_prediction": "deep_lobo_prediction"}
    ).assign(gbm_prediction=right["prediction"].to_numpy(float))


def fit_pair_weight(truth: np.ndarray, reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return the constrained least-squares candidate weight in [0, 1]."""

    truth = np.asarray(truth, dtype=float)
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    direction = candidate - reference
    denominator = float(direction @ direction)
    if denominator <= 1e-15:
        return 0.0
    weight = float(direction @ (truth - reference) / denominator)
    return float(np.clip(weight, 0.0, 1.0))


def blend_by_layer(
    frame: pd.DataFrame,
    weights: dict[str, float],
    *,
    reference_column: str,
    candidate_column: str = "gbm_prediction",
) -> np.ndarray:
    prediction = np.full(len(frame), np.nan, dtype=float)
    for layer in TARGET_LAYERS:
        selected = frame["layer"].to_numpy(int) == layer
        weight = float(weights[str(layer)])
        prediction[selected] = (1.0 - weight) * frame.loc[selected, reference_column].to_numpy(
            float
        ) + weight * frame.loc[selected, candidate_column].to_numpy(float)
    if not np.isfinite(prediction).all():
        raise ValueError("layer blend left rows unpredicted")
    return prediction


def evaluate_deep_pair(frame: pd.DataFrame) -> dict[str, object]:
    """Evaluate fitted and leave-one-block-out two-model convex blends."""

    truth = frame["truth"].to_numpy(float)
    fitted_weights: dict[str, float] = {}
    for layer in TARGET_LAYERS:
        selected = frame["layer"].to_numpy(int) == layer
        fitted_weights[str(layer)] = fit_pair_weight(
            truth[selected],
            frame.loc[selected, "deep_prediction"].to_numpy(float),
            frame.loc[selected, "gbm_prediction"].to_numpy(float),
        )
    fitted = blend_by_layer(frame, fitted_weights, reference_column="deep_prediction")
    lobo = np.full(len(frame), np.nan, dtype=float)
    lobo_weights: dict[str, dict[str, float]] = {}
    for held in VALIDATION_BLOCKS:
        lobo_weights[held] = {}
        for layer in TARGET_LAYERS:
            training = frame["block"].ne(held).to_numpy() & frame["layer"].eq(layer).to_numpy()
            testing = frame["block"].eq(held).to_numpy() & frame["layer"].eq(layer).to_numpy()
            weight = fit_pair_weight(
                truth[training],
                frame.loc[training, "deep_lobo_prediction"].to_numpy(float),
                frame.loc[training, "gbm_prediction"].to_numpy(float),
            )
            lobo_weights[held][str(layer)] = weight
            lobo[testing] = (1.0 - weight) * frame.loc[testing, "deep_lobo_prediction"].to_numpy(
                float
            ) + weight * frame.loc[testing, "gbm_prediction"].to_numpy(float)
    if not np.isfinite(lobo).all():
        raise AssertionError("LOBO pair blend left rows unpredicted")
    deep = frame["deep_prediction"].to_numpy(float)
    deep_lobo = frame["deep_lobo_prediction"].to_numpy(float)
    return {
        "deep_rmse": _rmse(truth, deep),
        "deep_lobo_rmse": _rmse(truth, deep_lobo),
        "fitted_blend_rmse": _rmse(truth, fitted),
        "fitted_delta_vs_deep": _rmse(truth, fitted) - _rmse(truth, deep),
        "lobo_blend_rmse": _rmse(truth, lobo),
        "lobo_delta_vs_deep_lobo": _rmse(truth, lobo) - _rmse(truth, deep_lobo),
        "fitted_weights_by_layer": fitted_weights,
        "lobo_weights": lobo_weights,
        "fitted_prediction": fitted,
        "lobo_prediction": lobo,
    }


def paired_day_bootstrap(
    frame: pd.DataFrame,
    candidate: np.ndarray,
    *,
    reference_column: str = "deep_prediction",
    replicates: int = 2000,
    seed: int = 20260816,
) -> dict[str, object]:
    """KST-day paired bootstrap for a candidate-minus-reference RMSE delta."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    times = pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Asia/Seoul")
    days = times.dt.strftime("%Y-%m-%d").to_numpy()
    unique = np.unique(days)
    blocks = [np.flatnonzero(days == value) for value in unique]
    truth = frame["truth"].to_numpy(float)
    reference = frame[reference_column].to_numpy(float)
    candidate = np.asarray(candidate, dtype=float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for index in range(replicates):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        rows = np.concatenate([blocks[position] for position in chosen])
        deltas[index] = _rmse(truth[rows], candidate[rows]) - _rmse(truth[rows], reference[rows])
    return {
        "replicates": replicates,
        "kst_days": len(unique),
        "delta_rmse": _rmse(truth, candidate) - _rmse(truth, reference),
        "ci90_low": float(np.quantile(deltas, 0.05)),
        "ci90_high": float(np.quantile(deltas, 0.95)),
        "probability_improved": float(np.mean(deltas < 0)),
    }


def arm_names(specs: Sequence[GBMArmSpec] = GBM_ARM_SPECS) -> list[str]:
    return [spec.name for spec in specs]
