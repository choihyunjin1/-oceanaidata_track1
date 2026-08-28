"""Joint wave-state multitask transfer for a bounded P3 research experiment.

The model keeps the frozen 286 past-only feature surface and predicts six-lead
log changes of Hs, mean period, and maximum wave height with shared CatBoost
trees.  Only the Hs output is ever used for candidate evaluation.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .era5_context_transfer import (
    LEADS,
    LOCAL_CATBOOST_PARAMETERS,
    SOURCE_CATBOOST_PARAMETERS,
    common_feature_columns,
)

STATE_NAMES = ("hs", "tp", "hmax")


class JointWaveStateError(ValueError):
    """Raised when the preregistered multitask contract is violated."""


def _multitask_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(parameters)
    result["loss_function"] = "MultiRMSE"
    return result


class JointWaveStateTransferRegressor:
    """Frozen-context CatBoost facade with three jointly predicted state deltas."""

    def __init__(
        self,
        *,
        source_parameters: Mapping[str, Any] = SOURCE_CATBOOST_PARAMETERS,
        local_parameters: Mapping[str, Any] = LOCAL_CATBOOST_PARAMETERS,
    ) -> None:
        self._source_parameters = _multitask_parameters(source_parameters)
        self._local_parameters = _multitask_parameters(local_parameters)
        self._source_model: Any | None = None
        self._model: Any | None = None

    @staticmethod
    def _context(frame: pd.DataFrame) -> pd.DataFrame:
        columns = common_feature_columns()
        if tuple(frame.columns) != columns:
            raise JointWaveStateError("feature schema/order differs from the frozen 286 columns")
        result = frame.apply(pd.to_numeric, errors="raise").astype("float64")
        if np.isinf(result.to_numpy()).any():
            raise JointWaveStateError("context contains infinity")
        return result.reset_index(drop=True)

    @staticmethod
    def _targets(targets: Any, rows: int) -> np.ndarray:
        values = np.asarray(targets, dtype=np.float64)
        expected = (rows, len(LEADS), len(STATE_NAMES))
        if values.shape != expected or not np.isfinite(values).all():
            raise JointWaveStateError(f"targets must be finite with shape {expected}")
        return values

    @staticmethod
    def _long(context: pd.DataFrame) -> pd.DataFrame:
        result = context.loc[context.index.repeat(len(LEADS))].reset_index(drop=True)
        result["lead_h"] = np.tile(np.asarray(LEADS, dtype=np.float64), len(context))
        return result

    @property
    def source_model(self) -> Any:
        if self._source_model is None:
            raise JointWaveStateError("source model is not fitted")
        return self._source_model

    @property
    def model(self) -> Any:
        if self._model is None:
            raise JointWaveStateError("model is not fitted")
        return self._model

    def fit_pretrain(
        self, features: pd.DataFrame, targets: Any
    ) -> JointWaveStateTransferRegressor:
        from catboost import CatBoostRegressor

        context = self._context(features)
        values = self._targets(targets, len(context))
        model = CatBoostRegressor(**self._source_parameters)
        model.fit(self._long(context), values.reshape(-1, len(STATE_NAMES)), verbose=False)
        self._source_model = model
        self._model = model
        return self

    def clone_pretrained(self) -> JointWaveStateTransferRegressor:
        if self._source_model is None:
            raise JointWaveStateError("source model is not fitted")
        clone = type(self)(
            source_parameters=self._source_parameters,
            local_parameters=self._local_parameters,
        )
        clone._source_model = copy.deepcopy(self._source_model)
        clone._model = clone._source_model
        if clone._source_model is self._source_model:
            raise JointWaveStateError("source model clone shares identity")
        return clone

    def continue_local(
        self,
        features: pd.DataFrame,
        targets: Any,
        *,
        current_hs: Sequence[float] | np.ndarray,
    ) -> JointWaveStateTransferRegressor:
        from catboost import CatBoostRegressor

        if self._source_model is None:
            raise JointWaveStateError("source model is not fitted")
        context = self._context(features)
        values = self._targets(targets, len(context))
        current = np.asarray(current_hs, dtype=np.float64)
        if current.shape != (len(context),) or not np.isfinite(current).all():
            raise JointWaveStateError("current_hs must be finite and aligned")
        weights = np.exp(-0.45 * np.maximum(current - 1.5, 0.0))
        model = CatBoostRegressor(**self._local_parameters)
        model.fit(
            self._long(context),
            values.reshape(-1, len(STATE_NAMES)),
            sample_weight=np.repeat(weights, len(LEADS)),
            init_model=self._source_model,
            verbose=False,
        )
        self._model = model
        return self

    def predict_hs(
        self,
        features: pd.DataFrame,
        *,
        current_hs: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        context = self._context(features)
        current = np.asarray(current_hs, dtype=np.float64)
        if current.shape != (len(context),) or not np.isfinite(current).all() or (current < 0).any():
            raise JointWaveStateError("prediction current_hs must be finite and non-negative")
        raw = np.asarray(self.model.predict(self._long(context)), dtype=np.float64)
        expected = (len(context) * len(LEADS), len(STATE_NAMES))
        if raw.shape != expected or not np.isfinite(raw).all():
            raise JointWaveStateError(f"CatBoost prediction must have shape {expected}")
        hs_log_delta = raw.reshape(len(context), len(LEADS), len(STATE_NAMES))[:, :, 0]
        return np.clip(np.expm1(np.log1p(current)[:, None] + hs_log_delta), 0.0, 30.0)

    def save_model(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))


def apply_frozen_persistence_shrink(prediction: np.ndarray, current_hs: np.ndarray) -> np.ndarray:
    result = np.asarray(prediction, dtype=np.float64).copy()
    current = np.asarray(current_hs, dtype=np.float64)
    if result.shape != (len(current), len(LEADS)):
        raise JointWaveStateError("persistence-shrink input shape changed")
    for position, lead in enumerate(LEADS):
        if lead in (12, 18, 24):
            result[:, position] = 0.8 * result[:, position] + 0.2 * current
    return result


def apply_joint_increment(
    base_prediction: np.ndarray,
    joint_prediction: np.ndarray,
    *,
    weight: float = 0.20,
) -> np.ndarray:
    base = np.asarray(base_prediction, dtype=np.float64)
    joint = np.asarray(joint_prediction, dtype=np.float64)
    if base.shape != joint.shape or base.ndim != 2 or base.shape[1] != len(LEADS):
        raise JointWaveStateError("candidate increment arrays changed shape")
    if weight != 0.20:
        raise JointWaveStateError("candidate increment weight is frozen at 0.20")
    result = base.copy()
    for position, lead in enumerate(LEADS):
        if lead in (18, 24):
            result[:, position] = base[:, position] + weight * (
                joint[:, position] - base[:, position]
            )
    return np.clip(result, 0.0, 30.0)
