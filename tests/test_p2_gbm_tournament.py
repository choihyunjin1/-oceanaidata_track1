from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p2_restore.features import FeatureTable
from p2_restore.gbm_tournament import (
    GBM_ARM_SPECS,
    GBMArmSpec,
    _make_estimator,
    align_with_deep_stack,
    evaluate_deep_pair,
    fit_gbm_model,
    fit_pair_weight,
)


def _table(rows: int = 900) -> FeatureTable:
    layer = np.resize(np.array([2, 3, 4]), rows)
    x = np.linspace(-2.0, 2.0, rows)
    baseline = 20.0 + 0.4 * x + 0.03 * layer
    residual = 0.2 * np.sin(2.3 * x) + 0.02 * (layer - 3)
    frame = pd.DataFrame(
        {
            "station": "S-ORS",
            "layer": layer,
            "time": pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC"),
            "baseline": baseline,
            "x": x,
            "x_missing": np.where(np.arange(rows) % 17 == 0, np.nan, x**2),
            "residual": residual,
            "target": baseline + residual,
        }
    )
    return FeatureTable(frame, ("baseline", "x", "x_missing"))


def test_fixed_structure_screen_contains_exact_six_arms() -> None:
    assert [spec.name for spec in GBM_ARM_SPECS] == [
        "lgbm_gbdt",
        "lgbm_extra_trees",
        "lgbm_dart",
        "xgboost_hist",
        "catboost_pooled",
        "catboost_layerwise",
    ]
    assert {spec.iterations for spec in GBM_ARM_SPECS} == {400}


@pytest.mark.parametrize("spec", GBM_ARM_SPECS)
def test_each_backend_fits_and_predicts_finite(spec: GBMArmSpec) -> None:
    table = _table(450)
    smoke = GBMArmSpec(
        spec.name,
        spec.backend,
        iterations=8,
        layerwise=spec.layerwise,
        categorical_layer=spec.categorical_layer,
    )
    training = np.arange(len(table.frame)) % 5 != 0
    model = fit_gbm_model(table, smoke, training, seed=17)
    prediction = model.predict(table)
    assert prediction.shape == (len(table.frame),)
    assert np.isfinite(prediction).all()


def test_backend_parameters_are_cpu_and_fixed_budget() -> None:
    by_name = {spec.name: spec for spec in GBM_ARM_SPECS}
    extra = _make_estimator(by_name["lgbm_extra_trees"], seed=5).get_params()
    dart = _make_estimator(by_name["lgbm_dart"], seed=5).get_params()
    xgb = _make_estimator(by_name["xgboost_hist"], seed=5).get_params()
    cat = _make_estimator(by_name["catboost_pooled"], seed=5).get_params()
    assert extra["extra_trees"] is True
    assert extra["deterministic"] is True
    assert dart["boosting_type"] == "dart"
    assert xgb["tree_method"] == "hist"
    assert xgb["device"] is None
    assert cat["task_type"] == "CPU"
    assert cat["allow_writing_files"] is False


def test_target_columns_are_rejected_as_features() -> None:
    table = _table()
    leaked = FeatureTable(table.frame.assign(temp_2=99.0), (*table.feature_columns, "temp_2"))
    with pytest.raises(ValueError, match="leaked"):
        fit_gbm_model(leaked, GBM_ARM_SPECS[0])


def test_pair_weight_is_constrained() -> None:
    truth = np.array([0.0, 1.0, 2.0])
    reference = np.array([0.0, 0.0, 0.0])
    candidate = np.array([0.0, 1.0, 2.0])
    assert fit_pair_weight(truth, reference, candidate) == pytest.approx(1.0)
    assert fit_pair_weight(truth, candidate, reference) == pytest.approx(0.0)


def test_alignment_and_lobo_pair_blend_are_label_safe_by_block() -> None:
    rows = []
    for block_number, block in enumerate(("2024_sep_oct", "2025_jul_aug", "2025_nov_dec")):
        for layer in (2, 3, 4):
            for offset in range(8):
                truth = float(layer + block_number + offset / 10)
                rows.append(
                    {
                        "time": pd.Timestamp("2024-01-01", tz="UTC")
                        + pd.Timedelta(days=block_number * 20, minutes=offset * 10),
                        "layer": layer,
                        "truth": truth,
                        "block": block,
                        "prediction": truth + 0.2,
                        "lobo_prediction": truth + 0.25,
                    }
                )
    deep = pd.DataFrame(rows)
    candidate = deep[["time", "layer", "truth", "block"]].copy()
    candidate["prediction"] = candidate["truth"] + 0.1
    aligned = align_with_deep_stack(deep, candidate)
    result = evaluate_deep_pair(aligned)
    assert result["fitted_blend_rmse"] < result["deep_rmse"]
    assert result["lobo_blend_rmse"] < result["deep_lobo_rmse"]
    assert set(result["fitted_weights_by_layer"]) == {"2", "3", "4"}


def test_alignment_rejects_truth_mismatch() -> None:
    deep = pd.DataFrame(
        {
            "time": [pd.Timestamp("2024-01-01", tz="UTC")],
            "layer": [2],
            "truth": [20.0],
            "block": ["2024_sep_oct"],
            "prediction": [20.0],
            "lobo_prediction": [20.0],
        }
    )
    candidate = deep.drop(columns="lobo_prediction").copy()
    candidate["truth"] = 21.0
    with pytest.raises(ValueError, match="truth differs"):
        align_with_deep_stack(deep, candidate)
