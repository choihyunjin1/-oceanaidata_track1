"""Deterministic sklearn-style wrappers for optional tabular boosters."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Backend = Literal["lightgbm", "xgboost", "catboost"]


class OptionalDependencyError(ImportError):
    """Raised when a requested optional model backend is not installed."""


@dataclass(frozen=True)
class TabularModelConfig:
    backend: Backend = "lightgbm"
    seed: int = 20260813
    n_jobs: int = 1
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend not in {"lightgbm", "xgboost", "catboost"}:
            raise ValueError("backend must be 'lightgbm', 'xgboost', or 'catboost'")
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("n_jobs must be -1 or a positive integer")


def _optional_import(module_name: str, install_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - exact branch depends on environment
        raise OptionalDependencyError(
            f"{install_name} is required for this backend; install the pinned "
            "CPU stack from requirements.txt"
        ) from exc


def lightgbm_parameters(config: TabularModelConfig) -> dict[str, Any]:
    """Return deterministic LightGBM defaults with explicit seed fan-out."""

    defaults: dict[str, Any] = {
        "objective": "binary",
        "n_estimators": 400,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": config.seed,
        "n_jobs": config.n_jobs,
        "verbosity": -1,
        "deterministic": True,
        "force_row_wise": True,
        "feature_fraction_seed": config.seed,
        "bagging_seed": config.seed,
        "data_random_seed": config.seed,
        "extra_seed": config.seed,
    }
    defaults.update(dict(config.parameters))
    return defaults


def xgboost_parameters(config: TabularModelConfig) -> dict[str, Any]:
    """Return deterministic CPU-hist XGBoost defaults.

    GPU histogram kernels can introduce small run-to-run floating-point
    differences, so the reproducible default is deliberately CPU ``hist``.
    Callers may override ``device`` for exploratory GPU runs and must record it.
    """

    defaults: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 400,
        "learning_rate": 0.04,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "device": "cpu",
        "random_state": config.seed,
        "seed": config.seed,
        "n_jobs": config.n_jobs,
        "verbosity": 0,
    }
    defaults.update(dict(config.parameters))
    return defaults


def catboost_parameters(config: TabularModelConfig) -> dict[str, Any]:
    """Return deterministic CatBoost CPU defaults.

    CatBoost GPU training is not bitwise deterministic, so ``task_type='CPU'``
    is the repeatable default.  Experiments may explicitly override it with
    ``parameters={'task_type': 'GPU', ...}`` and must record that choice.
    """

    defaults: dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "iterations": 400,
        "learning_rate": 0.04,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "random_seed": config.seed,
        "random_strength": 0.0,
        "bootstrap_type": "No",
        "task_type": "CPU",
        "thread_count": config.n_jobs,
        "allow_writing_files": False,
        "verbose": False,
    }
    defaults.update(dict(config.parameters))
    return defaults


def build_lightgbm_classifier(
    config: TabularModelConfig | None = None,
) -> Any:
    """Build a pinned-parameter ``lightgbm.LGBMClassifier`` lazily."""

    config = config or TabularModelConfig(backend="lightgbm")
    if config.backend != "lightgbm":
        raise ValueError("config.backend must be 'lightgbm'")
    lightgbm = _optional_import("lightgbm", "lightgbm")
    return lightgbm.LGBMClassifier(**lightgbm_parameters(config))


def build_xgboost_classifier(
    config: TabularModelConfig | None = None,
) -> Any:
    """Build a pinned-parameter ``xgboost.XGBClassifier`` lazily."""

    config = config or TabularModelConfig(backend="xgboost")
    if config.backend != "xgboost":
        raise ValueError("config.backend must be 'xgboost'")
    xgboost = _optional_import("xgboost", "xgboost")
    return xgboost.XGBClassifier(**xgboost_parameters(config))


def build_catboost_classifier(
    config: TabularModelConfig | None = None,
) -> Any:
    """Build a pinned-parameter ``catboost.CatBoostClassifier`` lazily."""

    config = config or TabularModelConfig(backend="catboost")
    if config.backend != "catboost":
        raise ValueError("config.backend must be 'catboost'")
    catboost = _optional_import("catboost", "catboost")
    return catboost.CatBoostClassifier(**catboost_parameters(config))


class DeterministicTabularClassifier:
    """Small common facade over LightGBM, XGBoost, and CatBoost classifiers.

    The optional backend is imported only when :meth:`fit` (or ``model``) is
    first used, so feature/rule-only workflows do not require either package.
    """

    def __init__(self, config: TabularModelConfig | None = None) -> None:
        self.config = config or TabularModelConfig()
        self._model: Any | None = None
        self.feature_names_in_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def _build(self) -> Any:
        if self.config.backend == "lightgbm":
            return build_lightgbm_classifier(self.config)
        if self.config.backend == "xgboost":
            return build_xgboost_classifier(self.config)
        return build_catboost_classifier(self.config)

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._build()
        return self._model

    def fit(
        self,
        features: Any,
        target: Sequence[int] | np.ndarray,
        *,
        sample_weight: Sequence[float] | np.ndarray | None = None,
        eval_set: Sequence[tuple[Any, Sequence[int] | np.ndarray]] | None = None,
        **fit_parameters: Any,
    ) -> DeterministicTabularClassifier:
        target_array = np.asarray(target)
        if target_array.ndim != 1:
            raise ValueError("target must be one-dimensional")
        if len(features) != len(target_array):
            raise ValueError("features and target must have equal row counts")
        shape = getattr(features, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError("features must be two-dimensional")
        if not np.isin(target_array, [0, 1]).all():
            raise ValueError("target must be binary 0/1")
        if sample_weight is not None:
            weight = np.asarray(sample_weight, dtype=float)
            if weight.shape != target_array.shape or not np.isfinite(weight).all():
                raise ValueError("sample_weight must be finite and match target")
            fit_parameters["sample_weight"] = weight
        if eval_set is not None:
            evaluation = list(eval_set)
            if self.config.backend == "lightgbm":
                # LightGBM 4.7 keeps eval_set for compatibility but deprecates
                # it in favour of the explicit eval_X/eval_y pair.
                fit_parameters["eval_X"] = tuple(item[0] for item in evaluation)
                fit_parameters["eval_y"] = tuple(item[1] for item in evaluation)
            else:
                fit_parameters["eval_set"] = evaluation
        if self.config.backend == "xgboost":
            fit_parameters.setdefault("verbose", False)

        self.model.fit(features, target_array.astype(np.int8), **fit_parameters)
        self.n_features_in_ = int(shape[1])
        columns = getattr(features, "columns", None)
        if columns is not None:
            self.feature_names_in_ = np.asarray([str(column) for column in columns])
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit must be called before predict_proba")
        probability = np.asarray(self._model.predict_proba(features), dtype=float)
        if probability.ndim == 1:
            positive = probability
        elif probability.ndim == 2 and probability.shape[1] == 2:
            positive = probability[:, 1]
        elif probability.ndim == 2 and probability.shape[1] == 1:
            positive = probability[:, 0]
        else:
            raise RuntimeError(f"unexpected predict_proba shape: {probability.shape}")
        if not np.isfinite(positive).all():
            raise RuntimeError("backend produced non-finite probabilities")
        positive = np.clip(positive, 0.0, 1.0)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, features: Any, *, threshold: float = 0.5) -> np.ndarray:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        return (self.predict_proba(features)[:, 1] >= threshold).astype(np.int8)

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit must be called before reading feature importances")
        importance = getattr(self._model, "feature_importances_", None)
        if importance is None:
            raise AttributeError("backend does not expose feature_importances_")
        return np.asarray(importance, dtype=float)


def make_tabular_classifier(
    backend: Backend,
    *,
    seed: int = 20260813,
    n_jobs: int = 1,
    parameters: Mapping[str, Any] | None = None,
) -> DeterministicTabularClassifier:
    """Convenience constructor used by experiment configs and tests."""

    return DeterministicTabularClassifier(
        TabularModelConfig(
            backend=backend,
            seed=seed,
            n_jobs=n_jobs,
            parameters={} if parameters is None else parameters,
        )
    )
