"""Label-blind counterfactual stress tests for deployment-depth shift.

The tests in this module do not select a model, threshold, or post-processing
parameter.  They refit the already selected outer-fold model, first reproduce
its saved OOF prediction, and then alter only depth-derived validation inputs.
Labels are read only after both predictions have been produced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from p1_qc.config import P1QCConfig
from p1_qc.features import FeatureBundle
from p1_qc.metrics import binary_counts
from p1_qc.pipeline import (
    TabularEncoder,
    _fit_model,
    _iteration_parameter,
    _model_parameters,
    _threads,
    apply_postprocess,
)
from p1_qc.rules import detect_plateaus, detect_singleton_spikes
from p1_qc.splits import outer_folds

DEPTH_MISSING_NUMERIC_COLUMNS = (
    "depth_raw",
    "nominal_depth_m",
    "depth_diff_1",
    "depth_abs_diff_1",
)
DEPTH_MISSING_FLAG = "depth_missing"
DEPTH_REGIME_COLUMN = "depth_regime"
REFERENCE_KEY_COLUMNS = ("station", "year", "layer", "time")


@dataclass(frozen=True)
class ScenarioArrays:
    """Ephemeral arrays used to calculate aggregate-only diagnostics."""

    truth: np.ndarray
    original_prediction: np.ndarray
    counterfactual_prediction: np.ndarray
    affected: np.ndarray


def _column_positions(feature_columns: Sequence[str], required: Sequence[str]) -> dict[str, int]:
    positions = {str(column): index for index, column in enumerate(feature_columns)}
    missing = sorted(set(required).difference(positions))
    if missing:
        raise KeyError(f"feature matrix is missing depth columns: {missing}")
    return positions


def depth_fallback_codes(
    encoder: TabularEncoder,
    station: Sequence[Any],
    layer: Sequence[Any],
) -> np.ndarray:
    """Encode the production fallback ``station|unknown|l<layer>`` category.

    A fallback unseen in the fold-training prefix receives the encoder's
    documented unknown value, ``-1``.  This matches inference without fitting
    or extending a category map on validation data.
    """

    if encoder.category_maps is None:
        raise RuntimeError("encoder must be fitted before building fallback codes")
    mapping = encoder.category_maps.get(DEPTH_REGIME_COLUMN)
    if mapping is None:
        raise KeyError(f"encoder has no {DEPTH_REGIME_COLUMN!r} category map")
    station_values = np.asarray(station, dtype=object)
    layer_values = np.asarray(layer, dtype=object)
    if station_values.shape != layer_values.shape:
        raise ValueError("station and layer shapes differ")
    categories = [
        f"{station_value}|unknown|l{layer_value}"
        for station_value, layer_value in zip(station_values, layer_values, strict=True)
    ]
    return np.asarray([mapping.get(value, -1) for value in categories], dtype=np.float32)


def apply_depth_missing_counterfactual(
    encoded_features: np.ndarray,
    feature_columns: Sequence[str],
    affected: Sequence[bool],
    fallback_codes: Sequence[float],
) -> np.ndarray:
    """Return a copy with depth inputs replaced by all-missing deployment values."""

    matrix = np.asarray(encoded_features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("encoded_features must be two-dimensional")
    mask = np.asarray(affected, dtype=bool)
    codes = np.asarray(fallback_codes, dtype=np.float32)
    if mask.shape != (len(matrix),) or codes.shape != (len(matrix),):
        raise ValueError("counterfactual row masks/codes must match the feature matrix")
    positions = _column_positions(
        feature_columns,
        (*DEPTH_MISSING_NUMERIC_COLUMNS, DEPTH_MISSING_FLAG, DEPTH_REGIME_COLUMN),
    )
    result = matrix.copy()
    for column in DEPTH_MISSING_NUMERIC_COLUMNS:
        result[mask, positions[column]] = np.nan
    result[mask, positions[DEPTH_MISSING_FLAG]] = 1.0
    result[mask, positions[DEPTH_REGIME_COLUMN]] = codes[mask]
    return result


def apply_unknown_depth_regime_counterfactual(
    encoded_features: np.ndarray,
    feature_columns: Sequence[str],
    affected: Sequence[bool],
) -> np.ndarray:
    """Return a copy routing selected rows through the unseen-category code."""

    matrix = np.asarray(encoded_features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("encoded_features must be two-dimensional")
    mask = np.asarray(affected, dtype=bool)
    if mask.shape != (len(matrix),):
        raise ValueError("counterfactual row mask must match the feature matrix")
    position = _column_positions(feature_columns, (DEPTH_REGIME_COLUMN,))[DEPTH_REGIME_COLUMN]
    result = matrix.copy()
    result[mask, position] = -1.0
    return result


def _binary_summary(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    counts = binary_counts(truth, prediction)
    negatives = counts.fp + counts.tn
    return {
        "rows": int(len(truth)),
        "positive_rows": int(np.asarray(truth, dtype=np.int8).sum()),
        "predicted_positive_rows": int(np.asarray(prediction, dtype=np.int8).sum()),
        "tp": int(counts.tp),
        "fp": int(counts.fp),
        "fn": int(counts.fn),
        "tn": int(counts.tn),
        "precision": float(counts.precision),
        "recall": float(counts.recall),
        "f1": float(counts.f1),
        "fpr": float(counts.fp / negatives) if negatives else 0.0,
    }


def comparison_summary(
    truth: Sequence[int],
    original_prediction: Sequence[int],
    counterfactual_prediction: Sequence[int],
) -> dict[str, Any]:
    """Calculate exact aggregate deltas and directional prediction flips."""

    target = np.asarray(truth, dtype=np.int8)
    original = np.asarray(original_prediction, dtype=np.int8)
    counterfactual = np.asarray(counterfactual_prediction, dtype=np.int8)
    if target.shape != original.shape or target.shape != counterfactual.shape:
        raise ValueError("truth and prediction shapes differ")
    before = _binary_summary(target, original)
    after = _binary_summary(target, counterfactual)
    flips = original != counterfactual
    return {
        "original": before,
        "counterfactual": after,
        "delta_f1": float(after["f1"] - before["f1"]),
        "delta_fpr": float(after["fpr"] - before["fpr"]),
        "delta_predicted_positive_rows": int(
            after["predicted_positive_rows"] - before["predicted_positive_rows"]
        ),
        "flip_rows": int(flips.sum()),
        "flip_rate": float(flips.mean()) if len(flips) else 0.0,
        "zero_to_one_rows": int(((original == 0) & (counterfactual == 1)).sum()),
        "one_to_zero_rows": int(((original == 1) & (counterfactual == 0)).sum()),
    }


def _weighted_counts_summary(values: Mapping[str, float]) -> dict[str, float]:
    tp = float(values["tp"])
    fp = float(values["fp"])
    fn = float(values["fn"])
    tn = float(values["tn"])
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    negative_denominator = fp + tn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / precision_denominator if precision_denominator else 0.0,
        "recall": tp / recall_denominator if recall_denominator else 0.0,
        "f1": 2 * tp / f1_denominator if f1_denominator else 0.0,
        "fpr": fp / negative_denominator if negative_denominator else 0.0,
    }


def weighted_group_counterfactual_summary(
    reference_groups: Sequence[Mapping[str, Any]],
    target_group_weights: Mapping[Any, float],
    affected_key: tuple[Any, Any],
    affected_comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Reweight aggregate group counts after replacing one affected group.

    This uses no row-level output.  It mirrors ``weighted_group_counts`` by
    applying each target/test row share to within-group confusion rates.
    """

    weights = {
        tuple(key if isinstance(key, tuple) else (key,)): float(value)
        for key, value in target_group_weights.items()
    }
    groups = {(str(row["station"]), int(row["layer"])): row for row in reference_groups}
    key = (str(affected_key[0]), int(affected_key[1]))
    if key not in groups:
        raise KeyError(f"affected group is absent from reference metrics: {key}")
    reference = groups[key]
    original_affected = affected_comparison["original"]
    for field in ("rows", "tp", "fp", "fn", "tn"):
        if float(reference[field]) != float(original_affected[field]):
            raise RuntimeError(
                f"affected aggregate {field} does not reproduce reference group {key}"
            )
    total_weight = sum(weights.get(group_key, 0.0) for group_key in groups)
    if total_weight <= 0:
        raise ValueError("target group weights have no mass on outer-validation groups")

    summaries: dict[str, dict[str, float]] = {}
    for side in ("original", "counterfactual"):
        totals = {field: 0.0 for field in ("tp", "fp", "fn", "tn")}
        for group_key, group in groups.items():
            row = affected_comparison[side] if group_key == key else group
            rows = int(group["rows"])
            if rows <= 0:
                raise ValueError(f"reference group has no rows: {group_key}")
            per_row_weight = weights.get(group_key, 0.0) / total_weight / rows
            for field in totals:
                totals[field] += float(row[field]) * per_row_weight
        summaries[side] = _weighted_counts_summary(totals)
    return {
        **summaries,
        "delta_f1": summaries["counterfactual"]["f1"] - summaries["original"]["f1"],
        "delta_fpr": summaries["counterfactual"]["fpr"] - summaries["original"]["fpr"],
    }


def _probability_summary(
    original: np.ndarray,
    counterfactual: np.ndarray,
    affected: np.ndarray,
) -> dict[str, float | int]:
    difference = np.abs(np.asarray(counterfactual) - np.asarray(original))
    selected = difference[affected]
    return {
        "changed_rows_gt_1e_12": int((selected > 1.0e-12).sum()),
        "changed_rate_gt_1e_12": float((selected > 1.0e-12).mean()) if len(selected) else 0.0,
        "mean_absolute_change": float(selected.mean()) if len(selected) else 0.0,
        "max_absolute_change": float(selected.max()) if len(selected) else 0.0,
    }


def _fold_reference(
    reference_oof: pd.DataFrame,
    fold_name: str,
    validation_frame: pd.DataFrame,
) -> pd.DataFrame:
    part = reference_oof.loc[reference_oof["fold"].eq(fold_name)].reset_index(drop=True)
    expected = validation_frame.loc[:, list(REFERENCE_KEY_COLUMNS)].reset_index(drop=True)
    observed = part.loc[:, list(REFERENCE_KEY_COLUMNS)].reset_index(drop=True)
    if len(part) != len(validation_frame) or not observed.equals(expected):
        raise RuntimeError(f"reference OOF keys/order differ for {fold_name}")
    return part


def _selection_by_fold(reference_metrics: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    folds = reference_metrics.get("folds")
    if not isinstance(folds, list):
        raise TypeError("reference metrics must contain a fold list")
    result: dict[str, Mapping[str, Any]] = {}
    for value in folds:
        if not isinstance(value, Mapping):
            raise TypeError("reference fold metrics must be mappings")
        name = str(value["fold"])
        if name in result:
            raise ValueError(f"duplicate reference fold: {name}")
        result[name] = value
    return result


def run_depth_shift_stress(
    train: pd.DataFrame,
    bundle: FeatureBundle,
    config: P1QCConfig,
    reference_oof: pd.DataFrame,
    reference_metrics: Mapping[str, Any],
    reference_selection: Mapping[str, Any],
    *,
    target_group_weights: Mapping[Any, float] | None = None,
    probability_tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Reproduce frozen XGBoost OOF and run two label-blind depth shifts.

    The returned payload contains aggregate diagnostics only.  Row-level
    probabilities and labels remain ephemeral and are never returned.
    """

    if str(reference_selection.get("backend")) != "xgboost":
        raise ValueError("depth-shift stress is pinned to the frozen XGBoost candidate")
    if bool(reference_selection.get("augmentation", False)):
        raise ValueError("frozen XGBoost reference unexpectedly uses augmentation")
    if str(reference_selection.get("feature_mode")) != config.features.mode:
        raise ValueError("reference selection and feature mode differ")
    if len(train) != len(bundle.frame) or not train.index.equals(bundle.frame.index):
        raise ValueError("train and feature bundle rows/order differ")
    if probability_tolerance <= 0:
        raise ValueError("probability_tolerance must be positive")

    fold_selections = _selection_by_fold(reference_metrics)
    folds = outer_folds(
        train,
        config=config.splits,
        cadence_minutes=config.data.cadence_minutes,
        group_columns=config.data.group_columns,
    )
    parameters = _model_parameters(config, "xgboost")
    iteration_key = _iteration_parameter("xgboost")

    reproduction_rows: list[dict[str, Any]] = []
    scenario_folds: dict[str, list[dict[str, Any]]] = {
        "gors_depth_100pct_missing": [],
        "sors_layer5_unseen_depth_regime": [],
    }
    aggregate_arrays: dict[str, list[ScenarioArrays]] = {name: [] for name in scenario_folds}

    for fold_number, fold in enumerate(folds):
        selected = fold_selections.get(fold.name)
        if selected is None:
            raise KeyError(f"reference metrics are missing {fold.name}")
        iterations = int(selected["best_iterations"])
        postprocess = dict(selected["postprocess"])
        fold_parameters = dict(parameters)
        fold_parameters[iteration_key] = iterations

        encoder = TabularEncoder().fit(bundle, fold.train_idx)
        training_features = encoder.transform(bundle, fold.train_idx)
        validation_features = encoder.transform(bundle, fold.val_idx)
        training_target = train.iloc[fold.train_idx]["label"].to_numpy(dtype=np.int8)
        model = _fit_model(
            "xgboost",
            fold_parameters,
            config.seed + fold_number,
            _threads(config),
            training_features,
            training_target,
        )
        original_probability = model.predict_proba(validation_features)[:, 1]
        validation_frame = train.iloc[fold.val_idx].copy()
        plateau = detect_plateaus(validation_frame).to_numpy(dtype=bool)
        spike = detect_singleton_spikes(validation_frame).to_numpy(dtype=bool)
        original_prediction = apply_postprocess(
            validation_frame,
            original_probability,
            plateau,
            spike,
            postprocess,
        )

        reference = _fold_reference(reference_oof, fold.name, validation_frame)
        reference_probability = reference["probability"].to_numpy(dtype=float)
        maximum_error = float(np.max(np.abs(original_probability - reference_probability)))
        prediction_mismatches = int(
            (original_prediction != reference["prediction"].to_numpy(dtype=np.int8)).sum()
        )
        if maximum_error > probability_tolerance or prediction_mismatches:
            raise RuntimeError(
                f"{fold.name} failed frozen OOF reproduction: "
                f"max_probability_error={maximum_error}, "
                f"prediction_mismatches={prediction_mismatches}"
            )
        reproduction_rows.append(
            {
                "fold": fold.name,
                "rows": len(validation_frame),
                "iterations": iterations,
                "postprocess": postprocess,
                "maximum_absolute_probability_error": maximum_error,
                "prediction_mismatch_rows": prediction_mismatches,
            }
        )

        # Both masks are determined from metadata only, before labels are read.
        station = validation_frame["station"].astype("string")
        layer = validation_frame["layer"]
        masks = {
            "gors_depth_100pct_missing": station.eq("G-ORS").to_numpy(dtype=bool),
            "sors_layer5_unseen_depth_regime": (
                station.eq("S-ORS") & pd.to_numeric(layer, errors="coerce").eq(5)
            ).to_numpy(dtype=bool),
        }
        gors_codes = depth_fallback_codes(encoder, station, layer)
        counterfactual_features = {
            "gors_depth_100pct_missing": apply_depth_missing_counterfactual(
                validation_features,
                bundle.feature_columns,
                masks["gors_depth_100pct_missing"],
                gors_codes,
            ),
            "sors_layer5_unseen_depth_regime": apply_unknown_depth_regime_counterfactual(
                validation_features,
                bundle.feature_columns,
                masks["sors_layer5_unseen_depth_regime"],
            ),
        }

        # Holdout labels are accessed only now, after all counterfactual inputs
        # have been constructed without target information.
        truth = validation_frame["label"].to_numpy(dtype=np.int8)
        regime_position = _column_positions(bundle.feature_columns, (DEPTH_REGIME_COLUMN,))[
            DEPTH_REGIME_COLUMN
        ]
        for name, affected in masks.items():
            if not affected.any():
                continue
            probability = model.predict_proba(counterfactual_features[name])[:, 1]
            prediction = apply_postprocess(
                validation_frame,
                probability,
                plateau,
                spike,
                postprocess,
            )
            scenario_folds[name].append(
                {
                    "fold": fold.name,
                    "affected_group": (
                        {"station": "G-ORS"}
                        if name == "gors_depth_100pct_missing"
                        else {"station": "S-ORS", "layer": 5}
                    ),
                    "affected": comparison_summary(
                        truth[affected],
                        original_prediction[affected],
                        prediction[affected],
                    ),
                    "whole_fold": comparison_summary(
                        truth,
                        original_prediction,
                        prediction,
                    ),
                    "probability_change_on_affected": _probability_summary(
                        original_probability,
                        probability,
                        affected,
                    ),
                    "original_unseen_depth_regime_rows": int(
                        (validation_features[affected, regime_position] == -1).sum()
                    ),
                    "counterfactual_unseen_depth_regime_rows": int(
                        (counterfactual_features[name][affected, regime_position] == -1).sum()
                    ),
                }
            )
            aggregate_arrays[name].append(
                ScenarioArrays(truth, original_prediction, prediction, affected)
            )

    scenarios: dict[str, Any] = {}
    for name, parts in aggregate_arrays.items():
        if not parts:
            scenarios[name] = {"folds": [], "aggregate": None}
            continue
        truth = np.concatenate([part.truth for part in parts])
        original = np.concatenate([part.original_prediction for part in parts])
        counterfactual = np.concatenate([part.counterfactual_prediction for part in parts])
        affected = np.concatenate([part.affected for part in parts])
        affected_comparison = comparison_summary(
            truth[affected], original[affected], counterfactual[affected]
        )
        aggregate: dict[str, Any] = {
            "affected": affected_comparison,
            "all_outer_rows": comparison_summary(truth, original, counterfactual),
        }
        if target_group_weights is not None:
            affected_key = (
                ("G-ORS", 1)
                if name == "gors_depth_100pct_missing"
                else (
                    "S-ORS",
                    5,
                )
            )
            reference_aggregate = reference_metrics.get("aggregate")
            if not isinstance(reference_aggregate, Mapping) or not isinstance(
                reference_aggregate.get("groups"), list
            ):
                raise TypeError("reference metrics lack aggregate group counts")
            weighted = weighted_group_counterfactual_summary(
                reference_aggregate["groups"],
                target_group_weights,
                affected_key,
                affected_comparison,
            )
            reference_weighted = reference_aggregate.get("weighted")
            if not isinstance(reference_weighted, Mapping):
                raise TypeError("reference metrics lack aggregate weighted counts")
            if abs(weighted["original"]["f1"] - float(reference_weighted["f1"])) > 1.0e-12:
                raise RuntimeError("test-share weighted original F1 was not reproduced")
            aggregate["test_share_weighted_all_outer"] = weighted
        scenarios[name] = {
            "folds": scenario_folds[name],
            "aggregate": aggregate,
        }

    return {
        "methodology": {
            "selection_frozen": True,
            "model_backend": "xgboost",
            "outer_fold_model_refit": True,
            "threshold_retuning": False,
            "postprocess_retuning": False,
            "mask_uses_labels": False,
            "outer_labels_used_for": "final aggregate evaluation only",
            "saved_row_level_outputs": False,
            "test_share_group_weights": target_group_weights is not None,
            "gors_counterfactual": {
                "scope": "all G-ORS rows in each outer validation fold",
                "depth_numeric_features": "NaN",
                "depth_missing": 1,
                "depth_regime": "station/layer fallback encoded from fold-train map, else -1",
            },
            "sors_layer5_counterfactual": {
                "scope": "all S-ORS layer 5 rows in each outer validation fold",
                "depth_numeric_features": "unchanged",
                "depth_regime": "unseen category code -1",
            },
        },
        "reproduction": {
            "probability_tolerance": probability_tolerance,
            "folds": reproduction_rows,
            "all_prediction_mismatch_rows": int(
                sum(row["prediction_mismatch_rows"] for row in reproduction_rows)
            ),
            "maximum_absolute_probability_error": float(
                max(row["maximum_absolute_probability_error"] for row in reproduction_rows)
            ),
        },
        "scenarios": scenarios,
    }


__all__ = [
    "DEPTH_MISSING_FLAG",
    "DEPTH_MISSING_NUMERIC_COLUMNS",
    "DEPTH_REGIME_COLUMN",
    "apply_depth_missing_counterfactual",
    "apply_unknown_depth_regime_counterfactual",
    "comparison_summary",
    "depth_fallback_codes",
    "run_depth_shift_stress",
    "weighted_group_counterfactual_summary",
]
