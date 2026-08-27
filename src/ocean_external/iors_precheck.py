"""Fixed external-only 2014--2022 to 2023 I-ORS profile precheck."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .iors_ctd import LooDataset


def _errors(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    if residual.size == 0 or not np.isfinite(residual).all():
        raise ValueError("metric inputs must be non-empty and finite")
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }


def evaluate_precheck(
    dataset: LooDataset,
    predictions: Mapping[float, np.ndarray],
) -> dict[str, Any]:
    """Compute aggregate-only errors and empirical interval coverage."""

    required = {0.1, 0.5, 0.9}
    if set(predictions) != required:
        raise ValueError(f"predictions must contain exactly {sorted(required)}")
    for quantile, value in predictions.items():
        if np.asarray(value).shape != dataset.y.shape:
            raise ValueError(f"prediction shape mismatch for q={quantile}")
    point = np.asarray(predictions[0.5], dtype=np.float64)
    raw_low = np.asarray(predictions[0.1], dtype=np.float64)
    raw_high = np.asarray(predictions[0.9], dtype=np.float64)
    low = np.minimum(raw_low, raw_high)
    high = np.maximum(raw_low, raw_high)
    overall_candidate = _errors(dataset.y, point)
    overall_baseline = _errors(dataset.y, dataset.baseline)
    per_layer: dict[str, Any] = {}
    for layer in sorted(int(item) for item in np.unique(dataset.layer)):
        mask = dataset.layer == layer
        candidate = _errors(dataset.y[mask], point[mask])
        baseline = _errors(dataset.y[mask], dataset.baseline[mask])
        per_layer[str(layer)] = {
            "rows": int(mask.sum()),
            "candidate": candidate,
            "baseline": baseline,
            "rmse_relative_improvement": float(
                (baseline["rmse"] - candidate["rmse"]) / baseline["rmse"]
            ),
            "mae_relative_improvement": float(
                (baseline["mae"] - candidate["mae"]) / baseline["mae"]
            ),
        }
    return {
        "rows": int(dataset.y.size),
        "candidate": overall_candidate,
        "baseline": overall_baseline,
        "rmse_relative_improvement": float(
            (overall_baseline["rmse"] - overall_candidate["rmse"]) / overall_baseline["rmse"]
        ),
        "mae_relative_improvement": float(
            (overall_baseline["mae"] - overall_candidate["mae"]) / overall_baseline["mae"]
        ),
        "q10_q90": {
            "coverage": float(np.mean((dataset.y >= low) & (dataset.y <= high))),
            "mean_width": float(np.mean(high - low)),
            "median_width": float(np.median(high - low)),
            "crossing_rate_before_reorder": float(np.mean(raw_low > raw_high)),
        },
        "per_layer": per_layer,
    }


def apply_stop_gate(
    metrics: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    source_integrity_verified: bool,
) -> dict[str, Any]:
    """Apply only preregistered thresholds; never optimize them on 2023."""

    per_layer = metrics["per_layer"]
    layer_values = list(per_layer.values())
    fraction_not_worse = float(
        np.mean([item["candidate"]["rmse"] <= item["baseline"]["rmse"] for item in layer_values])
    )
    layer_degradation = [
        (item["candidate"]["rmse"] - item["baseline"]["rmse"]) / item["baseline"]["rmse"]
        for item in layer_values
    ]
    maximum_degradation = float(max(layer_degradation))
    checks = {
        "source_integrity": source_integrity_verified
        if bool(gate["all_source_integrity_checks_required"])
        else True,
        "minimum_holdout_rows": int(metrics["rows"]) >= int(gate["minimum_holdout_rows"]),
        "minimum_evaluated_layers": len(layer_values) >= int(gate["minimum_evaluated_layers"]),
        "minimum_rmse_relative_improvement": float(metrics["rmse_relative_improvement"])
        >= float(gate["minimum_rmse_relative_improvement"]),
        "minimum_mae_relative_improvement": float(metrics["mae_relative_improvement"])
        >= float(gate["minimum_mae_relative_improvement"]),
        "minimum_fraction_layers_not_worse": fraction_not_worse
        >= float(gate["minimum_fraction_layers_not_worse"]),
        "maximum_single_layer_relative_rmse_degradation": maximum_degradation
        <= float(gate["maximum_single_layer_relative_rmse_degradation"]),
        "q10_q90_coverage_min": float(metrics["q10_q90"]["coverage"])
        >= float(gate["q10_q90_coverage_min"]),
        "q10_q90_coverage_max": float(metrics["q10_q90"]["coverage"])
        <= float(gate["q10_q90_coverage_max"]),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "decision": "GO_TO_ISOLATED_P1_OOF_ONLY" if passed else "NO_GO_EXTERNAL_PROFILE",
        "checks": checks,
        "diagnostics": {
            "fraction_layers_not_worse": fraction_not_worse,
            "maximum_single_layer_relative_rmse_degradation": maximum_degradation,
            "evaluated_layers": len(layer_values),
        },
        "scope": "external-only precheck; this is not P1 OOF, official validation, or submission evidence",
    }


def fit_quantile_models(
    fit: LooDataset,
    holdout: LooDataset,
    model_contract: Mapping[str, Any],
    *,
    progress: Any | None = None,
) -> tuple[dict[float, np.ndarray], dict[str, Any]]:
    """Fit the three fixed LightGBM models without 2023 tuning or early stopping."""

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM from requirements.txt is required") from exc
    if fit.feature_names != holdout.feature_names:
        raise ValueError("fit/holdout feature contracts differ")
    params = dict(model_contract["params"])
    quantiles = tuple(float(value) for value in model_contract["quantiles"])
    if quantiles != (0.1, 0.5, 0.9):
        raise ValueError("the preregistered quantiles must be [0.1, 0.5, 0.9]")
    predictions: dict[float, np.ndarray] = {}
    model_audit: dict[str, Any] = {
        "library": "lightgbm",
        "version": str(lgb.__version__),
        "fit_rows": int(fit.y.size),
        "holdout_rows": int(holdout.y.size),
        "feature_count": len(fit.feature_names),
        "feature_names": list(fit.feature_names),
        "params": params,
        "quantiles": list(quantiles),
        "early_stopping": False,
        "holdout_used_for_training": False,
        "models": {},
    }
    for position, quantile in enumerate(quantiles, start=1):
        if progress is not None:
            progress(position - 1, len(quantiles), quantile)
        model = lgb.LGBMRegressor(objective="quantile", alpha=quantile, **params)
        model.fit(fit.x, fit.y)
        prediction = np.asarray(model.predict(holdout.x), dtype=np.float64)
        if prediction.shape != holdout.y.shape or not np.isfinite(prediction).all():
            raise RuntimeError(f"invalid q={quantile} predictions")
        predictions[quantile] = prediction
        importance = np.asarray(
            model.booster_.feature_importance(importance_type="gain"), dtype=float
        )
        total = float(importance.sum())
        ranked = np.argsort(-importance)[:20]
        model_audit["models"][str(quantile)] = {
            "trees": int(model.booster_.num_trees()),
            "best_iteration": int(model.booster_.current_iteration()),
            "top_gain_features": [
                {
                    "feature": fit.feature_names[int(index)],
                    "gain_fraction": float(importance[index] / total) if total > 0 else 0.0,
                }
                for index in ranked
            ],
        }
    if progress is not None:
        progress(len(quantiles), len(quantiles), math.nan)
    return predictions, model_audit


def dataset_audit(dataset: LooDataset) -> dict[str, Any]:
    """Return only aggregate shape/missingness; never serialize observational rows."""

    finite_fraction = np.mean(np.isfinite(dataset.x), axis=0)
    return {
        "rows": int(dataset.y.size),
        "features": int(dataset.x.shape[1]),
        "years": sorted(int(item) for item in np.unique(dataset.year)),
        "layers": sorted(int(item) for item in np.unique(dataset.layer)),
        "group_counts": dict(dataset.group_counts),
        "target_finite": bool(np.isfinite(dataset.y).all()),
        "baseline_finite": bool(np.isfinite(dataset.baseline).all()),
        "feature_finite_fraction": {
            name: float(value)
            for name, value in zip(dataset.feature_names, finite_fraction, strict=True)
        },
    }
