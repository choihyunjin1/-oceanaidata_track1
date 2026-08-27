"""Fold-symmetric S-ORS layer-5 depth-regime routing.

The distributed 2026 test has observed numeric depth for S-ORS layer 5, but
its categorical ``depth_regime`` is outside the frozen training vocabulary.
This module implements exactly one label-blind change: route S-ORS layer-5
rows through one fixed category before fitting the fold-local encoder.

Numeric depth features and every non-target row remain unchanged.  The helper
has no configurable parameter so an outer result cannot be used to tune the
fallback spelling or scope.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FeatureBundle
from .gors_depth_invariance import (
    _fold_indices,
    _fold_train_only_invariant_encoder,
)
from .pipeline import TabularEncoder

SORS_STATION = "S-ORS"
SORS_LAYER = 5
DEPTH_REGIME_COLUMN = "depth_regime"
SORS_L5_FALLBACK = "S-ORS|deployment_unknown|l5"


@dataclass(frozen=True)
class SORSL5InvariantFoldEncoding:
    """Encoded fold arrays and fixed-transform provenance."""

    bundle: FeatureBundle
    encoder: TabularEncoder
    train_features: np.ndarray
    validation_features: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    affected_train_rows: int
    affected_validation_rows: int


def sors_l5_mask(frame: pd.DataFrame) -> np.ndarray:
    """Return the metadata-only target mask without reading labels or values."""

    missing = sorted({"station", "layer"}.difference(frame.columns))
    if missing:
        raise KeyError(f"missing source columns: {missing}")
    station = frame["station"].astype("string")
    layer = pd.to_numeric(frame["layer"], errors="coerce")
    return (station.eq(SORS_STATION) & layer.eq(SORS_LAYER)).fillna(False).to_numpy(dtype=bool)


def _validate_contract(frame: pd.DataFrame, bundle: FeatureBundle) -> None:
    if len(frame) != len(bundle.frame) or not frame.index.equals(bundle.frame.index):
        raise ValueError("source frame and feature bundle must have identical rows and index")
    if DEPTH_REGIME_COLUMN not in bundle.feature_columns:
        raise KeyError("feature bundle is missing depth_regime")
    if DEPTH_REGIME_COLUMN not in bundle.categorical_columns:
        raise ValueError("depth_regime must remain categorical")


def apply_sors_l5_regime_invariance(
    frame: pd.DataFrame,
    bundle: FeatureBundle,
) -> FeatureBundle:
    """Copy ``bundle`` and replace only target-row ``depth_regime`` values."""

    _validate_contract(frame, bundle)
    affected = sors_l5_mask(frame)
    transformed = bundle.frame.copy(deep=True)
    if bool(affected.any()):
        transformed.loc[affected, DEPTH_REGIME_COLUMN] = SORS_L5_FALLBACK

    # Fail closed if an implementation change ever mutates a second column or
    # any non-target row.  ``DataFrame.equals`` treats aligned NaNs as equal.
    other_columns = [column for column in bundle.frame.columns if column != DEPTH_REGIME_COLUMN]
    if not transformed.loc[:, other_columns].equals(bundle.frame.loc[:, other_columns]):
        raise RuntimeError("S-ORS L5 transform changed a non-depth_regime feature")
    if not transformed.loc[~affected].equals(bundle.frame.loc[~affected]):
        raise RuntimeError("S-ORS L5 transform changed a non-target row")

    transformed.attrs = dict(bundle.frame.attrs)
    return FeatureBundle(
        transformed,
        bundle.feature_columns,
        bundle.categorical_columns,
    )


def encode_sors_l5_regime_invariant_fold(
    frame: pd.DataFrame,
    bundle: FeatureBundle,
    train_indices: Sequence[int] | np.ndarray,
    validation_indices: Sequence[int] | np.ndarray,
) -> SORSL5InvariantFoldEncoding:
    """Apply the fixed transform and encode one leakage-safe fold.

    The existing G-ORS experiment's generic fold-local mapping helper is
    reused only as implementation structure: shared baseline categories keep
    their original integer codes, the candidate-only fallback is appended,
    and validation-only categories remain ``-1``.
    """

    train = _fold_indices(train_indices, name="train_indices", row_count=len(frame))
    validation = _fold_indices(
        validation_indices,
        name="validation_indices",
        row_count=len(frame),
    )
    if np.intersect1d(train, validation, assume_unique=True).size:
        raise ValueError("train_indices and validation_indices must be disjoint")

    transformed = apply_sors_l5_regime_invariance(frame, bundle)
    encoder = _fold_train_only_invariant_encoder(bundle, transformed, train)
    affected = sors_l5_mask(frame)
    return SORSL5InvariantFoldEncoding(
        bundle=transformed,
        encoder=encoder,
        train_features=encoder.transform(transformed, train),
        validation_features=encoder.transform(transformed, validation),
        train_indices=train,
        validation_indices=validation,
        affected_train_rows=int(affected[train].sum()),
        affected_validation_rows=int(affected[validation].sum()),
    )


__all__ = [
    "DEPTH_REGIME_COLUMN",
    "SORS_LAYER",
    "SORS_L5_FALLBACK",
    "SORS_STATION",
    "SORSL5InvariantFoldEncoding",
    "apply_sors_l5_regime_invariance",
    "encode_sors_l5_regime_invariant_fold",
    "sors_l5_mask",
]
