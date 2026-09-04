from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p1_qc.features import FeatureBundle
from p1_qc.gors_depth_invariance import (
    DEPTH_MISSING_COLUMN,
    DEPTH_REGIME_COLUMN,
    GORS_DEPTH_NUMERIC_COLUMNS,
    apply_gors_depth_invariance,
    encode_gors_depth_invariant_fold,
)

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "configs" / "experiments" / "p1_gors_depth_invariance_draft.json"


def _fixture() -> tuple[pd.DataFrame, FeatureBundle]:
    source = pd.DataFrame(
        {
            "station": ["G-ORS", "S-ORS", "G-ORS", "I-ORS"],
            "layer": [1, 5, 2, 1],
            "label": [0, 1, 1, 0],
            "anomaly_type": ["", "flatline", "drift", ""],
        },
        index=[10, 11, 12, 13],
    )
    features = pd.DataFrame(
        {
            "station": pd.Series(source["station"], dtype="string"),
            "layer_category": pd.Series(["1", "5", "2", "1"], index=source.index, dtype="string"),
            "depth_regime": pd.Series(
                ["G-ORS|d005.0", "S-ORS|d010.0", "G-ORS|d006.0", "I-ORS|d005.0"],
                index=source.index,
                dtype="string",
            ),
            "temp_raw": np.asarray([10.0, 11.0, 12.0, 13.0], dtype=np.float32),
            "depth_raw": np.asarray([5.1, 10.2, 6.1, 5.2], dtype=np.float32),
            "nominal_depth_m": np.asarray([5.0, 10.0, 6.0, 5.0], dtype=np.float32),
            "depth_diff_1": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "depth_abs_diff_1": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "depth_missing": np.zeros(4, dtype=np.float32),
        },
        index=source.index,
    )
    categorical = ("station", "layer_category", "depth_regime")
    columns = tuple(features.columns)
    features.attrs.update(
        {
            "feature_columns": columns,
            "categorical_columns": categorical,
            "feature_mode": "offline",
        }
    )
    return source, FeatureBundle(features, columns, categorical)


def test_transform_masks_only_gors_and_does_not_mutate_inputs() -> None:
    source, bundle = _fixture()
    source_before = source.copy(deep=True)
    features_before = bundle.frame.copy(deep=True)

    result = apply_gors_depth_invariance(source, bundle)
    affected = source["station"].eq("G-ORS")
    unaffected = ~affected

    assert result is not bundle
    assert result.feature_columns == bundle.feature_columns
    assert result.categorical_columns == bundle.categorical_columns
    assert result.frame.attrs == bundle.frame.attrs
    assert result.frame.loc[affected, list(GORS_DEPTH_NUMERIC_COLUMNS)].isna().all().all()
    assert result.frame.loc[affected, DEPTH_MISSING_COLUMN].eq(1).all()
    assert result.frame.loc[affected, DEPTH_REGIME_COLUMN].tolist() == [
        "G-ORS|unknown|l1",
        "G-ORS|unknown|l2",
    ]
    pd.testing.assert_frame_equal(
        result.frame.loc[unaffected],
        features_before.loc[unaffected],
    )
    # In particular, S-ORS layer 5 is outside this experimental family.
    pd.testing.assert_series_equal(result.frame.loc[11], features_before.loc[11])
    pd.testing.assert_frame_equal(source, source_before)
    pd.testing.assert_frame_equal(bundle.frame, features_before)


def test_transform_is_label_blind() -> None:
    source, bundle = _fixture()
    relabelled = source.copy()
    relabelled["label"] = 1 - relabelled["label"]
    relabelled["anomaly_type"] = ["spike", "noise", "offset", "drift"]

    first = apply_gors_depth_invariance(source, bundle)
    second = apply_gors_depth_invariance(relabelled, bundle)
    pd.testing.assert_frame_equal(first.frame, second.frame)


def test_fold_encoder_fits_category_maps_on_transformed_train_only() -> None:
    source, bundle = _fixture()
    encoded = encode_gors_depth_invariant_fold(source, bundle, [0, 1], [2, 3])

    depth_map = encoded.encoder.category_maps[DEPTH_REGIME_COLUMN]
    assert "G-ORS|unknown|l1" in depth_map
    assert "G-ORS|unknown|l2" not in depth_map
    assert "I-ORS|d005.0" not in depth_map
    assert encoded.affected_train_rows == 1
    assert encoded.affected_validation_rows == 1

    depth_regime_position = encoded.bundle.feature_columns.index(DEPTH_REGIME_COLUMN)
    depth_missing_position = encoded.bundle.feature_columns.index(DEPTH_MISSING_COLUMN)
    depth_raw_position = encoded.bundle.feature_columns.index("depth_raw")
    assert encoded.train_features[0, depth_regime_position] == depth_map["G-ORS|unknown|l1"]
    assert encoded.validation_features[0, depth_regime_position] == -1
    assert encoded.validation_features[0, depth_missing_position] == 1
    assert np.isnan(encoded.validation_features[0, depth_raw_position])


def test_fold_encoder_keeps_every_non_g_encoded_value_bitwise_equal() -> None:
    source, bundle = _fixture()
    train = np.asarray([0, 1], dtype=np.int64)
    validation = np.asarray([2, 3], dtype=np.int64)
    encoded = encode_gors_depth_invariant_fold(source, bundle, train, validation)

    from p1_qc.pipeline import TabularEncoder

    baseline_encoder = TabularEncoder().fit(bundle, train)
    baseline_train = baseline_encoder.transform(bundle, train)
    baseline_validation = baseline_encoder.transform(bundle, validation)
    train_non_g = source.iloc[train]["station"].ne("G-ORS").to_numpy()
    validation_non_g = source.iloc[validation]["station"].ne("G-ORS").to_numpy()
    np.testing.assert_array_equal(
        encoded.train_features[train_non_g],
        baseline_train[train_non_g],
    )
    np.testing.assert_array_equal(
        encoded.validation_features[validation_non_g],
        baseline_validation[validation_non_g],
    )


def test_contract_fails_closed_on_unregistered_depth_feature() -> None:
    source, bundle = _fixture()
    changed = bundle.frame.copy()
    changed["depth_future_feature"] = 0.0
    changed_bundle = FeatureBundle(
        changed,
        (*bundle.feature_columns, "depth_future_feature"),
        bundle.categorical_columns,
    )
    with pytest.raises(ValueError, match="unregistered depth-derived"):
        apply_gors_depth_invariance(source, changed_bundle)


def test_fold_indices_must_be_disjoint_integer_positions() -> None:
    source, bundle = _fixture()
    with pytest.raises(ValueError, match="disjoint"):
        encode_gors_depth_invariant_fold(source, bundle, [0, 1], [1, 2])
    with pytest.raises(TypeError, match="integer"):
        encode_gors_depth_invariant_fold(source, bundle, [True, False], [2, 3])


def test_preregistration_freezes_the_single_change_and_no_actions() -> None:
    payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    change = payload["hypothesis"]["exactly_one_change"]
    assert change["name"] == "gors_fold_symmetric_depth_mask"
    assert change["station"] == "G-ORS"
    assert change["depth_numeric_columns"] == list(GORS_DEPTH_NUMERIC_COLUMNS)
    assert change["depth_missing"] == 1
    assert change["depth_regime_template"] == "{station}|unknown|l{layer}"
    assert change["apply_to"] == ["fold_train", "fold_validation", "full_train", "test"]
    assert payload["comparison"]["additional_hyperparameters"] == []
    assert payload["comparison"]["outer_execution_authorized"] is False
    assert set(payload["authorization"].values()) == {False}
