from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p1_qc.features import FeatureBundle
from p1_qc.pipeline import TabularEncoder
from p1_qc.sors_l5_regime_invariance import (
    DEPTH_REGIME_COLUMN,
    SORS_L5_FALLBACK,
    apply_sors_l5_regime_invariance,
    encode_sors_l5_regime_invariant_fold,
    sors_l5_mask,
)


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["S-ORS", "S-ORS", "S-ORS", "G-ORS", "S-ORS", "I-ORS"],
            "layer": [5, 4, 5, 1, 5, 2],
            "label": [0, 1, 1, 0, 0, 1],
            "anomaly_type": ["", "noise", "spike", "", "", "offset"],
        }
    )


def _bundle() -> FeatureBundle:
    frame = pd.DataFrame(
        {
            "station": pd.Series(
                ["S-ORS", "S-ORS", "S-ORS", "G-ORS", "S-ORS", "I-ORS"],
                dtype="string",
            ),
            "layer_category": pd.Series(["5", "4", "5", "1", "5", "2"], dtype="string"),
            "depth_regime": pd.Series(
                [
                    "S-ORS|d005.0",
                    "S-ORS|d003.0",
                    "S-ORS|d005.0",
                    "G-ORS|d003.0",
                    "S-ORS|d007.5",
                    "I-ORS|d010.0",
                ],
                dtype="string",
            ),
            "depth_raw": np.asarray([5.1, 3.2, 5.3, np.nan, 7.7, 10.2], dtype=np.float32),
            "nominal_depth_m": np.asarray([5.0, 2.5, 5.0, np.nan, 7.5, 10.0], dtype=np.float32),
            "temp_raw": np.asarray([10, 11, 12, 13, 14, 15], dtype=np.float32),
        }
    )
    return FeatureBundle(
        frame,
        tuple(frame.columns),
        ("station", "layer_category", "depth_regime"),
    )


def test_transform_changes_only_sors_layer5_depth_regime() -> None:
    source = _source()
    bundle = _bundle()
    result = apply_sors_l5_regime_invariance(source, bundle)
    affected = sors_l5_mask(source)

    assert affected.tolist() == [True, False, True, False, True, False]
    assert (result.frame.loc[affected, DEPTH_REGIME_COLUMN] == SORS_L5_FALLBACK).all()
    pd.testing.assert_frame_equal(result.frame.loc[~affected], bundle.frame.loc[~affected])
    pd.testing.assert_frame_equal(
        result.frame.drop(columns=DEPTH_REGIME_COLUMN),
        bundle.frame.drop(columns=DEPTH_REGIME_COLUMN),
    )
    pd.testing.assert_frame_equal(bundle.frame, _bundle().frame)


def test_transform_is_label_blind_and_fixed() -> None:
    source = _source()
    first = apply_sors_l5_regime_invariance(source, _bundle())
    relabelled = source.copy()
    relabelled["label"] = 1 - relabelled["label"]
    relabelled["anomaly_type"] = "drift"
    second = apply_sors_l5_regime_invariance(relabelled, _bundle())
    pd.testing.assert_frame_equal(first.frame, second.frame)


def test_fold_encoding_knows_fallback_and_preserves_non_target_inputs() -> None:
    source = _source()
    bundle = _bundle()
    train = np.asarray([0, 1, 3, 5])
    validation = np.asarray([2, 4])
    baseline = TabularEncoder().fit(bundle, train)
    baseline_train = baseline.transform(bundle, train)
    baseline_validation = baseline.transform(bundle, validation)
    candidate = encode_sors_l5_regime_invariant_fold(source, bundle, train, validation)
    position = bundle.feature_columns.index(DEPTH_REGIME_COLUMN)

    fallback_code = candidate.encoder.category_maps[DEPTH_REGIME_COLUMN][SORS_L5_FALLBACK]
    assert candidate.train_features[0, position] == fallback_code
    assert (candidate.validation_features[:, position] == fallback_code).all()
    assert candidate.affected_train_rows == 1
    assert candidate.affected_validation_rows == 2

    non_target_train = ~sors_l5_mask(source.iloc[train])
    np.testing.assert_array_equal(
        candidate.train_features[non_target_train],
        baseline_train[non_target_train],
    )
    # Both validation rows are targets; numeric depth columns still match.
    numeric_positions = [
        index
        for index, column in enumerate(bundle.feature_columns)
        if column != DEPTH_REGIME_COLUMN
    ]
    np.testing.assert_array_equal(
        candidate.validation_features[:, numeric_positions],
        baseline_validation[:, numeric_positions],
    )


def test_contract_rejects_invalid_rows_and_feature_schema() -> None:
    with pytest.raises(ValueError, match="identical rows"):
        apply_sors_l5_regime_invariance(_source().iloc[:-1], _bundle())
    bad = _bundle()
    bad = FeatureBundle(bad.frame, bad.feature_columns, ("station", "layer_category"))
    with pytest.raises(ValueError, match="categorical"):
        apply_sors_l5_regime_invariance(_source(), bad)
