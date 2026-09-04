from __future__ import annotations

import numpy as np
import pandas as pd

from p2_restore.features import FeatureTable
from p2_restore.tuning import (
    P2TunedBlendModel,
    fit_tuned_lean,
    freeze_best_iterations,
    time_mask,
)


def _table() -> FeatureTable:
    rows = []
    times = pd.date_range("2024-01-01", periods=120, freq="10min", tz="Asia/Seoul")
    for number, time in enumerate(times):
        for layer in (2, 3, 4):
            baseline = 20 - layer
            rows.append(
                {
                    "station": "S-ORS",
                    "layer": layer,
                    "time": time.isoformat(),
                    "baseline": baseline,
                    "target": baseline + 0.05 * layer + np.sin(number / 20) * 0.01,
                    "residual": 0.05 * layer + np.sin(number / 20) * 0.01,
                    "x": np.cos(number / 15),
                    "target_depth": float(layer),
                }
            )
    return FeatureTable(pd.DataFrame(rows), ("baseline", "target_depth", "x"))


def _parameters() -> dict[str, object]:
    return {
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 10,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "min_split_gain": 0.0,
        "max_bin": 127,
    }


def test_freeze_best_iterations_uses_median() -> None:
    assert freeze_best_iterations({"shared": [100, 300, 200, 400]}) == {"shared": 250}


def test_time_mask_uses_kst_half_open_bounds() -> None:
    table = _table()
    mask = time_mask(table, "2024-01-01", "2024-01-01 01:00")
    assert mask.sum() == 18


def test_shared_and_layerwise_fixed_iteration_models_are_finite() -> None:
    table = _table()
    rows = np.ones(len(table.frame), dtype=bool)
    for structure, iterations in (
        ("shared", {"shared": 5}),
        ("layerwise", {"2": 5, "3": 5, "4": 5}),
    ):
        model, best = fit_tuned_lean(
            table,
            rows,
            structure=structure,
            parameters=_parameters(),
            iterations=iterations,
            seed=7,
        )
        prediction = model.predict(table)
        assert np.isfinite(prediction).all()
        assert best == iterations


def test_tuned_blend_weight_is_frozen() -> None:
    class ConstantModel:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, table: FeatureTable) -> np.ndarray:
            return np.full(len(table.frame), self.value)

    table = FeatureTable(pd.DataFrame({"x": [1.0]}), ("x",))
    blend = P2TunedBlendModel(ConstantModel(1.0), ConstantModel(3.0))
    assert np.allclose(blend.predict(table, table), 2.0)
