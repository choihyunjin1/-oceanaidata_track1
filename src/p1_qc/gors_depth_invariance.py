"""Fold-local G-ORS depth invariance for the 2026 deployment shift.

The distributed test contract states that every G-ORS depth observation is
missing.  This module implements one deterministic, label-blind change: mask
the existing depth-derived G-ORS features *before* fitting the categorical
encoder.  Applying the same transform to fold-train and fold-validation rows
prevents the model from learning a depth-present G-ORS representation that is
unavailable at deployment.

No feature is added or removed.  Non-G-ORS rows are copied unchanged, and the
contract deliberately has no tunable parameter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FeatureBundle
from .pipeline import TabularEncoder

GORS_STATION = "G-ORS"
GORS_DEPTH_NUMERIC_COLUMNS = (
    "depth_raw",
    "nominal_depth_m",
    "depth_diff_1",
    "depth_abs_diff_1",
)
DEPTH_MISSING_COLUMN = "depth_missing"
DEPTH_REGIME_COLUMN = "depth_regime"


@dataclass(frozen=True)
class GORSInvariantFoldEncoding:
    """Encoded fold data with provenance for the fixed G-ORS transform."""

    bundle: FeatureBundle
    encoder: TabularEncoder
    train_features: np.ndarray
    validation_features: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    affected_train_rows: int
    affected_validation_rows: int


def _fold_train_only_invariant_encoder(
    original: FeatureBundle,
    transformed: FeatureBundle,
    train_indices: np.ndarray,
) -> TabularEncoder:
    """Fit candidate maps while preserving shared baseline category codes.

    Both the baseline and candidate vocabularies are learned from fold-train
    rows only.  Shared categories retain the baseline integer code, while a
    newly created G-ORS fallback receives a deterministic code above the
    baseline range.  Consequently every non-G-ORS encoded row is bitwise
    identical between arms; validation-only categories remain unknown (``-1``).
    """

    baseline = TabularEncoder().fit(original, train_indices)
    if baseline.category_maps is None:  # pragma: no cover - fit guarantees this
        raise RuntimeError("baseline encoder did not produce category maps")
    part = transformed.frame.iloc[train_indices]
    maps: dict[str, dict[str, int]] = {}
    for column in transformed.categorical_columns:
        values = sorted(part[column].astype("string").fillna("<NA>").unique().tolist())
        baseline_map = baseline.category_maps[column]
        next_code = max(baseline_map.values(), default=-1) + 1
        mapping: dict[str, int] = {}
        for raw_value in values:
            value = str(raw_value)
            if value in baseline_map:
                mapping[value] = baseline_map[value]
            else:
                mapping[value] = next_code
                next_code += 1
        maps[column] = mapping
    return TabularEncoder(
        feature_columns=transformed.feature_columns,
        categorical_columns=transformed.categorical_columns,
        category_maps=maps,
    )


def _validate_contract(frame: pd.DataFrame, bundle: FeatureBundle) -> None:
    missing_source = sorted({"station", "layer"}.difference(frame.columns))
    if missing_source:
        raise KeyError(f"missing source columns: {missing_source}")
    if len(frame) != len(bundle.frame) or not frame.index.equals(bundle.frame.index):
        raise ValueError("source frame and feature bundle must have identical rows and index")

    required_features = {
        *GORS_DEPTH_NUMERIC_COLUMNS,
        DEPTH_MISSING_COLUMN,
        DEPTH_REGIME_COLUMN,
    }
    missing_features = sorted(required_features.difference(bundle.feature_columns))
    if missing_features:
        raise KeyError(f"feature bundle is missing depth contract columns: {missing_features}")
    if DEPTH_REGIME_COLUMN not in bundle.categorical_columns:
        raise ValueError("depth_regime must remain categorical")
    invalid_numeric = sorted(
        set(GORS_DEPTH_NUMERIC_COLUMNS).intersection(bundle.categorical_columns)
    )
    if invalid_numeric:
        raise ValueError(f"depth numeric columns cannot be categorical: {invalid_numeric}")

    # Fail closed if the base feature builder later gains another depth-derived
    # field.  Adding it to this transform would be a new experimental decision,
    # not an implicit implementation detail.
    registered_depth_features = required_features
    observed_depth_features = {
        column
        for column in bundle.feature_columns
        if column == "nominal_depth_m" or column.startswith("depth_")
    }
    unexpected = sorted(observed_depth_features.difference(registered_depth_features))
    if unexpected:
        raise ValueError(f"unregistered depth-derived features require review: {unexpected}")


def apply_gors_depth_invariance(
    frame: pd.DataFrame,
    bundle: FeatureBundle,
) -> FeatureBundle:
    """Return a copy with the fixed deployment-depth mask applied to G-ORS.

    Only ``station`` and ``layer`` are read from ``frame``.  In particular,
    labels and anomaly types cannot affect the mask.  The exact registered
    change for every G-ORS row is:

    - four existing depth-derived numeric fields -> NaN;
    - ``depth_missing`` -> 1;
    - ``depth_regime`` -> ``G-ORS|unknown|l<layer>``.

    The input frame and feature bundle are never mutated.
    """

    _validate_contract(frame, bundle)
    station = frame["station"].astype("string")
    layer = frame["layer"].astype("string")
    affected = station.eq(GORS_STATION).fillna(False)
    if layer.loc[affected].isna().any():
        raise ValueError("G-ORS rows require a non-missing layer for the fallback category")

    transformed = bundle.frame.copy(deep=True)
    if bool(affected.any()):
        for column in GORS_DEPTH_NUMERIC_COLUMNS:
            transformed.loc[affected, column] = np.nan
        transformed.loc[affected, DEPTH_MISSING_COLUMN] = np.float32(1.0)
        fallback = station + "|unknown|l" + layer
        transformed.loc[affected, DEPTH_REGIME_COLUMN] = fallback.loc[affected]

    transformed.attrs = dict(bundle.frame.attrs)
    return FeatureBundle(
        transformed,
        bundle.feature_columns,
        bundle.categorical_columns,
    )


def _fold_indices(
    values: Sequence[int] | np.ndarray,
    *,
    name: str,
    row_count: int,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional index array")
    if raw.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must contain integer row positions")
    result = raw.astype(np.int64, copy=True)
    if result.min() < 0 or result.max() >= row_count:
        raise IndexError(f"{name} contains an out-of-range row position")
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} contains duplicate row positions")
    return result


def encode_gors_depth_invariant_fold(
    frame: pd.DataFrame,
    bundle: FeatureBundle,
    train_indices: Sequence[int] | np.ndarray,
    validation_indices: Sequence[int] | np.ndarray,
) -> GORSInvariantFoldEncoding:
    """Apply the invariant transform and encode one leakage-safe fold.

    Category maps use fold-train rows only.  Codes learned from the original
    fold-train data are preserved for shared categories, candidate-only
    fallback categories are appended above the original maximum, and
    validation-only categories receive the unknown code (``-1``).
    """

    train = _fold_indices(train_indices, name="train_indices", row_count=len(frame))
    validation = _fold_indices(
        validation_indices,
        name="validation_indices",
        row_count=len(frame),
    )
    if np.intersect1d(train, validation, assume_unique=True).size:
        raise ValueError("train_indices and validation_indices must be disjoint")

    transformed = apply_gors_depth_invariance(frame, bundle)
    encoder = _fold_train_only_invariant_encoder(bundle, transformed, train)
    station = frame["station"].astype("string").eq(GORS_STATION).fillna(False).to_numpy()
    return GORSInvariantFoldEncoding(
        bundle=transformed,
        encoder=encoder,
        train_features=encoder.transform(transformed, train),
        validation_features=encoder.transform(transformed, validation),
        train_indices=train,
        validation_indices=validation,
        affected_train_rows=int(station[train].sum()),
        affected_validation_rows=int(station[validation].sum()),
    )


__all__ = [
    "DEPTH_MISSING_COLUMN",
    "DEPTH_REGIME_COLUMN",
    "GORS_DEPTH_NUMERIC_COLUMNS",
    "GORS_STATION",
    "GORSInvariantFoldEncoding",
    "_fold_train_only_invariant_encoder",
    "apply_gors_depth_invariance",
    "encode_gors_depth_invariant_fold",
]
