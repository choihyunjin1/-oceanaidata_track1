"""Small deterministic model facade for P3 residual forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ResidualRegressor:
    backend: str
    seed: int = 20260816
    parameters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.model: Any | None = None
        self.columns: list[str] = []

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = frame.copy()
        matrix["station"] = matrix["station"].astype("category")
        matrix["lead_h"] = matrix["lead_h"].astype("category")
        return matrix

    def fit(
        self,
        frame: pd.DataFrame,
        target_delta: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> ResidualRegressor:
        self.columns = list(frame.columns)
        params = dict(self.parameters or {})
        matrix = self._matrix(frame)
        if self.backend == "ridge":
            categorical = ["station", "lead_h"]
            numeric = [c for c in self.columns if c not in categorical]
            transform = ColumnTransformer(
                [
                    (
                        "numeric",
                        Pipeline(
                            [
                                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                ("scale", StandardScaler()),
                            ]
                        ),
                        numeric,
                    ),
                    (
                        "categorical",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        categorical,
                    ),
                ]
            )
            self.model = Pipeline(
                [
                    ("transform", transform),
                    ("model", Ridge(alpha=float(params.get("alpha", 100.0)))),
                ]
            )
            self.model.fit(matrix, target_delta, model__sample_weight=sample_weight)
        elif self.backend == "lightgbm":
            from lightgbm import LGBMRegressor

            defaults = {
                "objective": "regression_l2",
                "n_estimators": 700,
                "learning_rate": 0.025,
                "num_leaves": 15,
                "max_depth": -1,
                "min_child_samples": 80,
                "subsample": 0.85,
                "colsample_bytree": 0.55,
                "reg_alpha": 1.0,
                "reg_lambda": 8.0,
                "random_state": self.seed,
                "n_jobs": 8,
                "deterministic": True,
                "force_row_wise": True,
                "verbosity": -1,
            }
            defaults.update(params)
            self.model = LGBMRegressor(**defaults)
            self.model.fit(
                matrix,
                target_delta,
                sample_weight=sample_weight,
                categorical_feature=["station", "lead_h"],
            )
        elif self.backend == "xgboost":
            from xgboost import XGBRegressor

            matrix = pd.get_dummies(matrix, columns=["station", "lead_h"], dtype=float)
            self.columns = list(matrix.columns)
            defaults = {
                "objective": "reg:squarederror",
                "n_estimators": 700,
                "learning_rate": 0.025,
                "max_depth": 4,
                "min_child_weight": 20.0,
                "subsample": 0.85,
                "colsample_bytree": 0.55,
                "reg_alpha": 1.0,
                "reg_lambda": 8.0,
                "random_state": self.seed,
                "n_jobs": 8,
                "tree_method": "hist",
            }
            defaults.update(params)
            self.model = XGBRegressor(**defaults)
            self.model.fit(matrix, target_delta, sample_weight=sample_weight, verbose=False)
        elif self.backend == "catboost":
            from catboost import CatBoostRegressor

            matrix = matrix.copy()
            matrix["station"] = matrix["station"].astype(str)
            matrix["lead_h"] = matrix["lead_h"].astype(str)
            defaults = {
                "loss_function": "RMSE",
                "iterations": 700,
                "learning_rate": 0.035,
                "depth": 6,
                "l2_leaf_reg": 8.0,
                "random_strength": 0.2,
                "random_seed": self.seed,
                "thread_count": 8,
                "verbose": False,
                "allow_writing_files": False,
            }
            defaults.update(params)
            self.model = CatBoostRegressor(**defaults)
            self.model.fit(
                matrix,
                target_delta,
                sample_weight=sample_weight,
                cat_features=[0, 1],
                verbose=False,
            )
        else:
            raise ValueError(f"unknown backend: {self.backend}")
        return self

    def predict_delta(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        matrix = self._matrix(frame)
        if self.backend == "xgboost":
            matrix = pd.get_dummies(matrix, columns=["station", "lead_h"], dtype=float)
            matrix = matrix.reindex(columns=self.columns, fill_value=0.0)
        elif self.backend == "catboost":
            matrix = matrix.copy()
            matrix["station"] = matrix["station"].astype(str)
            matrix["lead_h"] = matrix["lead_h"].astype(str)
        return np.asarray(self.model.predict(matrix), dtype=float)


def compact_feature_columns(columns: list[str] | tuple[str, ...]) -> list[str]:
    """Predeclared low-variance feature surface for the first structural tournament."""

    keep_tokens = (
        "_current",
        "_lag_1h",
        "_lag_3h",
        "_lag_6h",
        "_lag_12h",
        "_lag_24h",
        "_lag_48h",
        "_mean_3h",
        "_std_3h",
        "_delta_3h",
        "_slope_3h",
        "_valid_3h",
        "_mean_6h",
        "_std_6h",
        "_delta_6h",
        "_slope_6h",
        "_valid_6h",
        "_mean_12h",
        "_std_12h",
        "_delta_12h",
        "_slope_12h",
        "_valid_12h",
        "_mean_24h",
        "_std_24h",
        "_delta_24h",
        "_slope_24h",
        "_valid_24h",
        "_mean_48h",
        "_std_48h",
        "_delta_48h",
        "_slope_48h",
        "_valid_48h",
        "hs_change_",
        "wspd_change_",
        "caph_change_",
        "event_",
    )
    return [column for column in columns if any(token in column for token in keep_tokens)]


def threshold_case_weights(current_hs: np.ndarray) -> np.ndarray:
    """Fixed mild weighting for the public case-selection shift toward the 1.5m threshold."""

    current = np.asarray(current_hs, dtype=float)
    weight = np.exp(-0.45 * np.maximum(current - 1.5, 0.0))
    return weight / np.mean(weight)
