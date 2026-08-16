"""Fixed low-complexity residual model and blocked validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from p2_restore.features import FeatureTable

VALIDATION_BLOCKS = {
    "2024_sep_oct": ("2024-09-01", "2024-11-01"),
    "2025_jul_aug": ("2025-07-01", "2025-09-01"),
    "2025_nov_dec": ("2025-11-01", "2026-01-01"),
}


@dataclass
class P2Model:
    estimator: object
    feature_columns: tuple[str, ...]

    def predict(self, table: FeatureTable) -> np.ndarray:
        if table.feature_columns != self.feature_columns:
            raise ValueError("P2 feature schema differs from fitted model")
        residual = self.estimator.predict(table.frame.loc[:, self.feature_columns])
        return np.clip(table.frame["baseline"].to_numpy(float) + residual, -5.0, 45.0)


def _estimator(seed: int = 20260816):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="regression_l2",
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=7,
        min_child_samples=200,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=8,
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )


def fit_model(
    table: FeatureTable, rows: np.ndarray | None = None, *, seed: int = 20260816
) -> P2Model:
    selected = (
        np.ones(len(table.frame), dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
    )
    model = _estimator(seed)
    model.fit(
        table.frame.loc[selected, table.feature_columns], table.frame.loc[selected, "residual"]
    )
    return P2Model(model, table.feature_columns)


def blocked_validation(table: FeatureTable) -> dict[str, object]:
    time = pd.to_datetime(table.frame["time"], utc=True)
    results: dict[str, object] = {}
    for number, (name, (start, stop)) in enumerate(VALIDATION_BLOCKS.items()):
        left = pd.Timestamp(start, tz="Asia/Seoul").tz_convert("UTC")
        right = pd.Timestamp(stop, tz="Asia/Seoul").tz_convert("UTC")
        validation = time.ge(left) & time.lt(right)
        fitted = fit_model(table, ~validation.to_numpy(), seed=20260816 + number)
        prediction = fitted.predict(
            FeatureTable(table.frame.loc[validation].reset_index(drop=True), table.feature_columns)
        )
        truth = table.frame.loc[validation, "target"].to_numpy(float)
        baseline = table.frame.loc[validation, "baseline"].to_numpy(float)
        layer = table.frame.loc[validation, "layer"].to_numpy(int)

        def rmse(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.sqrt(np.mean((a - b) ** 2)))

        results[name] = {
            "rows": int(validation.sum()),
            "baseline_rmse": rmse(baseline, truth),
            "model_rmse": rmse(prediction, truth),
            "by_layer": {
                str(target): {
                    "rows": int((layer == target).sum()),
                    "baseline_rmse": rmse(baseline[layer == target], truth[layer == target]),
                    "model_rmse": rmse(prediction[layer == target], truth[layer == target]),
                }
                for target in (2, 3, 4)
            },
        }
    return results
