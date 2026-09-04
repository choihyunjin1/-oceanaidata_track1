"""Leakage-safe supervised stress evaluations for temporal and station shift.

Stress holdouts are diagnostics, not tuning sets.  This module deliberately
requires an already selected model iteration and post-processing mapping, fits
the :class:`~p1_qc.pipeline.TabularEncoder` on stress-training rows only, and
never passes holdout labels to the model wrapper or an early-stopping callback.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from p1_qc.config import P1QCConfig
from p1_qc.features import FeatureBundle
from p1_qc.metrics import (
    EvaluationReport,
    binary_counts,
    evaluate_predictions,
    group_row_shares,
)
from p1_qc.models_tabular import Backend, make_tabular_classifier
from p1_qc.postprocess import PostprocessConfig
from p1_qc.rules import detect_plateaus, detect_singleton_spikes
from p1_qc.splits import Fold, group_holdout_fold, year_transfer_fold

ClassifierFactory = Callable[..., Any]


@dataclass(frozen=True)
class StressResult:
    """Predictions, fixed selection, metrics, and leakage diagnostics."""

    name: str
    backend: str
    train_idx: np.ndarray
    holdout_idx: np.ndarray
    probability: np.ndarray
    prediction: np.ndarray
    model_parameters: Mapping[str, Any]
    postprocess_selection: Mapping[str, Any]
    metrics: EvaluationReport
    preprocessing: Mapping[str, Any]
    fallback: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_idx", np.asarray(self.train_idx, dtype=np.int64))
        object.__setattr__(self, "holdout_idx", np.asarray(self.holdout_idx, dtype=np.int64))
        object.__setattr__(self, "probability", np.asarray(self.probability, dtype=float))
        object.__setattr__(self, "prediction", np.asarray(self.prediction, dtype=np.int8))

    def to_dict(self) -> dict[str, Any]:
        """Return an experiment-record payload without row-level predictions."""

        return {
            "name": self.name,
            "backend": self.backend,
            "train_rows": len(self.train_idx),
            "holdout_rows": len(self.holdout_idx),
            "model_parameters": dict(self.model_parameters),
            "postprocess_selection": dict(self.postprocess_selection),
            "metrics": self.metrics.to_dict(),
            "preprocessing": dict(self.preprocessing),
            "fallback": dict(self.fallback),
        }


_ITERATION_PARAMETER: dict[str, str] = {
    "lightgbm": "n_estimators",
    "xgboost": "n_estimators",
    "catboost": "iterations",
}
_POSTPROCESS_KEYS = (
    "high_threshold",
    "low_threshold",
    "close_gap_rows",
    "minimum_positive_run",
)


def _threads(config: P1QCConfig) -> int:
    project = config.raw.get("project", {})
    if isinstance(project, Mapping):
        value = int(project.get("threads", 1))
        if value == -1 or value > 0:
            return value
    return 1


def _year_transfer_purge(config: P1QCConfig, bundle: FeatureBundle) -> pd.Timedelta:
    feature_mode = str(bundle.frame.attrs.get("feature_mode", config.features.mode)).lower()
    if feature_mode not in {"causal", "offline"}:
        raise ValueError(f"unknown FeatureBundle mode: {feature_mode}")
    configured = pd.Timedelta(days=config.splits.purge_days)
    if feature_mode == "causal":
        return configured
    rolling_future = (
        pd.Timedelta(hours=max(config.features.rolling_hours) / 2)
        if config.features.rolling_hours
        else pd.Timedelta(0)
    )
    long_future = (
        pd.Timedelta(days=max(config.features.long_windows_days) / 2)
        if config.features.long_windows_days
        else pd.Timedelta(0)
    )
    # Centered windows can see half their span into the future.  A stress
    # training row is admitted only when even that future context ends before
    # the holdout.  The configured embargo remains a lower bound.
    return max(configured, rolling_future, long_future)


def _selected_parameters(
    config: P1QCConfig,
    backend: str,
    selected_model_parameters: Mapping[str, Any] | None,
    selected_iteration: int | None,
) -> dict[str, Any]:
    if backend not in _ITERATION_PARAMETER:
        raise ValueError("backend must be 'lightgbm', 'xgboost', or 'catboost'")
    if selected_model_parameters is None:
        models = config.raw.get("models", {})
        section = models.get(backend, {}) if isinstance(models, Mapping) else {}
        if not isinstance(section, Mapping):
            raise TypeError(f"config models.{backend} must be a mapping")
        parameters = dict(section)
    else:
        parameters = dict(selected_model_parameters)
    iteration_key = _ITERATION_PARAMETER[backend]
    if selected_iteration is not None:
        if selected_iteration < 1:
            raise ValueError("selected_iteration must be positive")
        parameters[iteration_key] = int(selected_iteration)
    if iteration_key not in parameters:
        raise ValueError(
            f"stress evaluation requires preselected {iteration_key}; "
            "holdout iteration tuning is forbidden"
        )
    try:
        iteration_value = int(parameters[iteration_key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{iteration_key} must be a selected positive integer") from exc
    if iteration_value < 1:
        raise ValueError(f"{iteration_key} must be a selected positive integer")
    parameters[iteration_key] = iteration_value
    return parameters


def _postprocess_mapping(
    selection: Mapping[str, Any] | PostprocessConfig,
) -> dict[str, Any]:
    values = asdict(selection) if isinstance(selection, PostprocessConfig) else dict(selection)
    missing = [key for key in _POSTPROCESS_KEYS if key not in values]
    if missing:
        raise ValueError(
            f"postprocess_selection is not preselected; missing scalar values {missing}"
        )
    result = {
        "high_threshold": float(values["high_threshold"]),
        "low_threshold": float(values["low_threshold"]),
        "close_gap_rows": int(values["close_gap_rows"]),
        "minimum_positive_run": int(values["minimum_positive_run"]),
    }
    # Reuse PostprocessConfig validation without tuning any value.
    PostprocessConfig(**result)
    return result


def _class_weights(target: np.ndarray) -> np.ndarray:
    positive = max(1, int(target.sum()))
    negative = max(1, len(target) - positive)
    return np.where(target == 1, np.sqrt(negative / positive), 1.0).astype(np.float32)


def _positive_probability(model: Any, features: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=float)
    if raw.ndim == 1:
        probability = raw
    elif raw.ndim == 2 and raw.shape[1] == 2:
        probability = raw[:, 1]
    elif raw.ndim == 2 and raw.shape[1] == 1:
        probability = raw[:, 0]
    else:
        raise RuntimeError(f"unexpected predict_proba shape: {raw.shape}")
    if not np.isfinite(probability).all():
        raise RuntimeError("stress model produced non-finite probabilities")
    if ((probability < 0) | (probability > 1)).any():
        raise RuntimeError("stress model probabilities must lie in [0, 1]")
    return probability


def _validate_inputs(train: pd.DataFrame, bundle: FeatureBundle, fold: Fold) -> None:
    if len(train) != len(bundle.frame):
        raise ValueError("train and FeatureBundle row counts differ")
    if not train.index.equals(bundle.frame.index):
        raise ValueError("train and FeatureBundle indices/order differ")
    required = {"station", "year", "layer", "time", "label", "anomaly_type", "temp"}
    missing = sorted(required.difference(train.columns))
    if missing:
        raise KeyError(f"stress train frame is missing columns: {missing}")
    train_idx = np.asarray(fold.train_idx, dtype=np.int64)
    holdout_idx = np.asarray(fold.val_idx, dtype=np.int64)
    if not len(train_idx) or not len(holdout_idx):
        raise ValueError("stress fold must have non-empty train and holdout rows")
    if train_idx.min() < 0 or holdout_idx.min() < 0:
        raise IndexError("stress indices cannot be negative")
    if train_idx.max() >= len(train) or holdout_idx.max() >= len(train):
        raise IndexError("stress indices exceed the frame")
    if np.intersect1d(train_idx, holdout_idx).size:
        raise ValueError("stress train and holdout rows overlap")
    target = pd.to_numeric(train.iloc[train_idx]["label"], errors="coerce")
    holdout_target = pd.to_numeric(train.iloc[holdout_idx]["label"], errors="coerce")
    if not target.isin([0, 1]).all() or not holdout_target.isin([0, 1]).all():
        raise ValueError("stress labels must be finite binary 0/1")
    if target.nunique() < 2:
        raise ValueError("stress training rows must contain both label classes")


def _preprocessing_diagnostics(
    bundle: FeatureBundle,
    encoder: Any,
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
) -> dict[str, Any]:
    category_maps = encoder.category_maps
    if category_maps is None:
        raise RuntimeError("TabularEncoder did not expose fitted category maps")
    holdout = bundle.frame.iloc[holdout_idx]
    fitted_unique: dict[str, int] = {}
    unseen_rows: dict[str, int] = {}
    unseen_rates: dict[str, float] = {}
    for column in bundle.categorical_columns:
        mapping = category_maps[column]
        values = holdout[column].astype("string").fillna("<NA>").astype(str)
        unseen = ~values.isin(mapping)
        fitted_unique[column] = len(mapping)
        unseen_rows[column] = int(unseen.sum())
        unseen_rates[column] = float(unseen.mean()) if len(unseen) else 0.0
    return {
        "encoder_fit_rows": len(train_idx),
        "encoder_transform_rows": len(holdout_idx),
        "category_maps_fitted_on": "stress_train_only",
        "numeric_scaling": "none",
        "fitted_category_counts": fitted_unique,
        "unseen_category_rows": unseen_rows,
        "unseen_category_rates": unseen_rates,
    }


def _counts_or_none(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float] | None:
    if not mask.any():
        return None
    return binary_counts(truth[mask], prediction[mask]).to_dict()


def _fallback_diagnostics(
    holdout_features: pd.DataFrame,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    if "peer_available" in holdout_features:
        peer = (
            pd.to_numeric(holdout_features["peer_available"], errors="coerce")
            .fillna(0)
            .gt(0)
            .to_numpy()
        )
    elif "peer_count" in holdout_features:
        peer = (
            pd.to_numeric(holdout_features["peer_count"], errors="coerce")
            .fillna(0)
            .gt(0)
            .to_numpy()
        )
    else:
        peer = np.zeros(len(holdout_features), dtype=bool)
    no_peer = ~peer
    reference_columns = [
        column
        for column in holdout_features.columns
        if column.startswith("reference_") and "resid" in column
    ]
    reference_finite_rate_no_peer: dict[str, float] = {}
    for column in reference_columns:
        values = pd.to_numeric(holdout_features[column], errors="coerce").to_numpy(dtype=float)
        reference_finite_rate_no_peer[column] = (
            float(np.isfinite(values[no_peer]).mean()) if no_peer.any() else float("nan")
        )
    peer_residual_missing_rate = float("nan")
    if "temp_peer_residual" in holdout_features and no_peer.any():
        peer_residual = pd.to_numeric(holdout_features["temp_peer_residual"], errors="coerce")
        peer_residual_missing_rate = float(
            peer_residual.iloc[np.flatnonzero(no_peer)].isna().mean()
        )
    return {
        "peer_available_rows": int(peer.sum()),
        "peer_available_rate": float(peer.mean()) if len(peer) else 0.0,
        "no_peer_rows": int(no_peer.sum()),
        "no_peer_rate": float(no_peer.mean()) if len(no_peer) else 0.0,
        "peer_metrics": _counts_or_none(truth, prediction, peer),
        "no_peer_metrics": _counts_or_none(truth, prediction, no_peer),
        "reference_columns": reference_columns,
        "reference_finite_rate_no_peer": reference_finite_rate_no_peer,
        "peer_residual_missing_rate_no_peer": peer_residual_missing_rate,
    }


def _apply_selected_postprocess(
    frame: pd.DataFrame,
    probability: np.ndarray,
    selection: Mapping[str, Any],
) -> np.ndarray:
    # Imported lazily because pipeline's persistence helpers depend on joblib;
    # feature/audit-only environments should still be able to import stress.py.
    from p1_qc.pipeline import apply_postprocess

    plateau = detect_plateaus(frame).to_numpy(dtype=bool)
    spike = detect_singleton_spikes(frame).to_numpy(dtype=bool)
    return apply_postprocess(frame, probability, plateau, spike, selection)


def evaluate_stress_fold(
    train: pd.DataFrame,
    bundle: FeatureBundle,
    config: P1QCConfig,
    fold: Fold,
    *,
    backend: Backend,
    postprocess_selection: Mapping[str, Any] | PostprocessConfig,
    selected_model_parameters: Mapping[str, Any] | None = None,
    selected_iteration: int | None = None,
    group_weights: Mapping[Any, float] | None = None,
    classifier_factory: ClassifierFactory = make_tabular_classifier,
) -> StressResult:
    """Fit once on ``fold.train_idx`` and score the untouched holdout.

    ``classifier_factory`` is an integration/testing hook with the public
    ``make_tabular_classifier`` signature.  It does not change the contract:
    the returned model is fitted once, without an eval set or holdout labels.
    """

    _validate_inputs(train, bundle, fold)
    parameters = _selected_parameters(
        config,
        backend,
        selected_model_parameters,
        selected_iteration,
    )
    postprocess = _postprocess_mapping(postprocess_selection)
    train_idx = np.asarray(fold.train_idx, dtype=np.int64)
    holdout_idx = np.asarray(fold.val_idx, dtype=np.int64)

    from p1_qc.pipeline import TabularEncoder

    encoder = TabularEncoder().fit(bundle, train_idx)
    train_features = encoder.transform(bundle, train_idx)
    holdout_features = encoder.transform(bundle, holdout_idx)
    train_target = train.iloc[train_idx]["label"].to_numpy(dtype=np.int8)
    model = classifier_factory(
        backend,
        seed=config.seed,
        n_jobs=_threads(config),
        parameters=parameters,
    )
    # No eval_set or early stopping: iterations and every other parameter were
    # selected before this holdout evaluation.
    model.fit(
        train_features,
        train_target,
        sample_weight=_class_weights(train_target),
    )
    probability = _positive_probability(model, holdout_features)

    holdout_frame = train.iloc[holdout_idx].copy()
    prediction = _apply_selected_postprocess(holdout_frame, probability, postprocess)
    holdout_target = holdout_frame["label"].to_numpy(dtype=np.int8)
    effective_weights = group_row_shares(holdout_frame) if group_weights is None else group_weights
    metrics = evaluate_predictions(
        holdout_target,
        prediction,
        holdout_frame,
        group_columns=config.data.group_columns,
        group_weights=effective_weights,
        anomaly_type=holdout_frame["anomaly_type"],
        cadence_minutes=config.data.cadence_minutes,
        event_min_iou=config.metrics.event_min_iou,
    )
    preprocessing = _preprocessing_diagnostics(bundle, encoder, train_idx, holdout_idx)
    fallback = _fallback_diagnostics(bundle.frame.iloc[holdout_idx], holdout_target, prediction)
    return StressResult(
        name=fold.name,
        backend=backend,
        train_idx=train_idx,
        holdout_idx=holdout_idx,
        probability=probability,
        prediction=prediction,
        model_parameters=parameters,
        postprocess_selection=postprocess,
        metrics=metrics,
        preprocessing=preprocessing,
        fallback=fallback,
    )


def run_year_transfer_stress(
    train: pd.DataFrame,
    bundle: FeatureBundle,
    config: P1QCConfig,
    *,
    backend: Backend,
    postprocess_selection: Mapping[str, Any] | PostprocessConfig,
    selected_model_parameters: Mapping[str, Any] | None = None,
    selected_iteration: int | None = None,
    group_weights: Mapping[Any, float] | None = None,
    validation_end: str = "2025-07-01T00:00:00+09:00",
    classifier_factory: ClassifierFactory = make_tabular_classifier,
) -> StressResult:
    """Evaluate S-ORS 2024 -> S-ORS 2025 H1 with a purge before H1."""

    fold = year_transfer_fold(
        train,
        train_year=2024,
        validation_year=2025,
        validation_end=validation_end,
        station="S-ORS",
    )
    time = pd.to_datetime(train["time"], errors="raise", utc=True)
    effective_purge = _year_transfer_purge(config, bundle)
    purge_cutoff = fold.val_start - effective_purge
    purged_train = fold.train_idx[time.iloc[fold.train_idx].lt(purge_cutoff).to_numpy()]
    if not len(purged_train):
        raise ValueError("year-transfer training is empty after the configured purge")
    fold = Fold(
        fold.name,
        purged_train,
        fold.val_idx,
        time.iloc[purged_train].max(),
        fold.val_start,
        fold.val_end,
    )
    result = evaluate_stress_fold(
        train,
        bundle,
        config,
        fold,
        backend=backend,
        postprocess_selection=postprocess_selection,
        selected_model_parameters=selected_model_parameters,
        selected_iteration=selected_iteration,
        group_weights=group_weights,
        classifier_factory=classifier_factory,
    )
    return replace(
        result,
        preprocessing={
            **result.preprocessing,
            "feature_mode": str(bundle.frame.attrs.get("feature_mode", config.features.mode)),
            "year_transfer_purge_hours": effective_purge.total_seconds() / 3600.0,
        },
    )


def run_gors_holdout_stress(
    train: pd.DataFrame,
    bundle: FeatureBundle,
    config: P1QCConfig,
    *,
    backend: Backend,
    postprocess_selection: Mapping[str, Any] | PostprocessConfig,
    selected_model_parameters: Mapping[str, Any] | None = None,
    selected_iteration: int | None = None,
    group_weights: Mapping[Any, float] | None = None,
    classifier_factory: ClassifierFactory = make_tabular_classifier,
) -> StressResult:
    """Train on non-G stations and score the fully held-out G-ORS station."""

    fold = group_holdout_fold(train, holdout_station="G-ORS")
    result = evaluate_stress_fold(
        train,
        bundle,
        config,
        fold,
        backend=backend,
        postprocess_selection=postprocess_selection,
        selected_model_parameters=selected_model_parameters,
        selected_iteration=selected_iteration,
        group_weights=group_weights,
        classifier_factory=classifier_factory,
    )
    holdout_station = train.iloc[result.holdout_idx]["station"]
    if not holdout_station.eq("G-ORS").all():  # pragma: no cover - protected by split helper
        raise RuntimeError("G-ORS stress holdout contains a non-G station")
    return result


def run_stress_suite(
    train: pd.DataFrame,
    bundle: FeatureBundle,
    config: P1QCConfig,
    *,
    backend: Backend,
    postprocess_selection: Mapping[str, Any] | PostprocessConfig,
    selected_model_parameters: Mapping[str, Any] | None = None,
    selected_iteration: int | None = None,
    group_weights: Mapping[Any, float] | None = None,
    classifier_factory: ClassifierFactory = make_tabular_classifier,
) -> dict[str, StressResult]:
    """Run both fixed stress tests with one immutable prior selection."""

    common = {
        "backend": backend,
        "postprocess_selection": postprocess_selection,
        "selected_model_parameters": selected_model_parameters,
        "selected_iteration": selected_iteration,
        "group_weights": group_weights,
        "classifier_factory": classifier_factory,
    }
    return {
        "year_transfer": run_year_transfer_stress(train, bundle, config, **common),
        "gors_holdout": run_gors_holdout_stress(train, bundle, config, **common),
    }


__all__ = [
    "StressResult",
    "evaluate_stress_fold",
    "run_gors_holdout_stress",
    "run_stress_suite",
    "run_year_transfer_stress",
]
