"""Past-only wave-energy-weighted circular memory for the P3 ERA5 experiment.

The feature surface is deliberately narrow.  It summarizes temporal persistence of
one representative wave direction and must not be interpreted as a reconstructed
directional spectrum or directional spread.
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
)

WINDOWS = (6, 12, 24, 48)
VALUE_FEATURES = (
    "dir_energy_concentration_6h",
    "dir_energy_concentration_12h",
    "dir_energy_concentration_24h",
    "dir_energy_concentration_48h",
    "dir_energy_rel_cos_6_24h",
    "dir_energy_rel_sin_6_24h",
    "dir_energy_rel_cos_12_48h",
    "dir_energy_rel_sin_12_48h",
    "dir_energy_turn_signed_24h",
    "dir_energy_turn_abs_24h",
)
MASK_FEATURES = tuple(f"{name}_mask" for name in VALUE_FEATURES)
DIRECTIONAL_FEATURES = (*VALUE_FEATURES, *MASK_FEATURES)


class DirectionalMemoryError(ValueError):
    """Raised when a directional feature or model contract is violated."""


def _circular_delta(right: float, left: float) -> float:
    return float(np.arctan2(np.sin(right - left), np.cos(right - left)))


def _hourly_energy_direction(
    hs: np.ndarray,
    direction_deg: np.ndarray,
    relative_hours: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate native samples into 49 causal, anchor-relative hourly bins.

    Bins are ``(-48,-47], ..., (-1,0]`` plus the exact ``-48`` endpoint.
    This preserves the hourly ERA5 grid and averages the denser local cadence.
    """

    heights = np.asarray(hs, dtype=np.float64)
    directions = np.asarray(direction_deg, dtype=np.float64)
    offsets = np.asarray(relative_hours, dtype=np.float64)
    if heights.ndim != 1 or heights.shape != directions.shape or heights.shape != offsets.shape:
        raise DirectionalMemoryError("hs, direction, and relative-hour arrays must align")
    if len(heights) == 0 or np.any(np.diff(offsets) < 0):
        raise DirectionalMemoryError("relative-hour context must be non-empty and ordered")
    if offsets[0] < -48.0 - 1e-9 or offsets[-1] > 1e-9:
        raise DirectionalMemoryError("directional context escaped the past 48 hours")

    labels = np.arange(-48, 1, dtype=np.int64)
    assigned = np.ceil(offsets - 1e-12).astype(np.int64)
    energy = np.full(len(labels), np.nan, dtype=np.float64)
    phase = np.full(len(labels), np.nan, dtype=np.float64)
    for position, label in enumerate(labels):
        selected = assigned == label
        valid = selected & np.isfinite(heights) & (heights >= 0.0) & np.isfinite(directions)
        if not valid.any():
            continue
        sample_energy = np.square(heights[valid])
        total = float(sample_energy.sum())
        if not np.isfinite(total) or total <= 1e-12:
            continue
        radians = np.deg2rad(np.mod(directions[valid], 360.0))
        moment = np.sum(sample_energy * np.exp(1j * radians))
        if abs(moment) <= 1e-12:
            continue
        energy[position] = float(np.mean(sample_energy))
        phase[position] = float(np.angle(moment))
    return labels, energy, phase


def summarize_directional_energy_memory(
    hs: Sequence[float] | np.ndarray,
    direction_deg: Sequence[float] | np.ndarray,
    relative_hours: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Return ten values and ten explicit masks from one past-only context."""

    labels, energy, phase = _hourly_energy_direction(
        np.asarray(hs, dtype=np.float64),
        np.asarray(direction_deg, dtype=np.float64),
        np.asarray(relative_hours, dtype=np.float64),
    )
    moments: dict[int, complex | None] = {}
    values: dict[str, float] = {name: np.nan for name in VALUE_FEATURES}
    valid_value: dict[str, bool] = {name: False for name in VALUE_FEATURES}

    for window in WINDOWS:
        selected = labels > -window
        finite = selected & np.isfinite(energy) & np.isfinite(phase)
        enough = int(finite.sum()) >= int(np.ceil(0.80 * window))
        total = float(np.nansum(energy[finite])) if enough else 0.0
        moment: complex | None = None
        if enough and total > 1e-12:
            unit = np.exp(1j * phase[finite])
            moment = complex(np.sum(energy[finite] * unit) / total)
            if not (np.isfinite(moment.real) and np.isfinite(moment.imag)):
                moment = None
        moments[window] = moment
        name = f"dir_energy_concentration_{window}h"
        if moment is not None:
            values[name] = float(abs(moment))
            valid_value[name] = True

    for short, long in ((6, 24), (12, 48)):
        left = moments[short]
        right = moments[long]
        cos_name = f"dir_energy_rel_cos_{short}_{long}h"
        sin_name = f"dir_energy_rel_sin_{short}_{long}h"
        if left is not None and right is not None and abs(left) > 1e-12 and abs(right) > 1e-12:
            delta = _circular_delta(float(np.angle(left)), float(np.angle(right)))
            values[cos_name] = float(np.cos(delta))
            values[sin_name] = float(np.sin(delta))
            valid_value[cos_name] = True
            valid_value[sin_name] = True

    selected = labels > -24
    finite_positions = np.flatnonzero(selected & np.isfinite(energy) & np.isfinite(phase))
    if len(finite_positions) >= int(np.ceil(0.80 * 24)):
        elapsed = np.diff(labels[finite_positions]).astype(np.float64)
        deltas = np.asarray(
            [
                _circular_delta(float(phase[right]), float(phase[left]))
                for left, right in zip(finite_positions[:-1], finite_positions[1:], strict=True)
            ],
            dtype=np.float64,
        )
        denominator = float(elapsed.sum())
        if denominator > 0.0 and np.isfinite(deltas).all():
            signed_name = "dir_energy_turn_signed_24h"
            absolute_name = "dir_energy_turn_abs_24h"
            values[signed_name] = float(deltas.sum() / denominator)
            values[absolute_name] = float(np.abs(deltas).sum() / denominator)
            valid_value[signed_name] = True
            valid_value[absolute_name] = True

    result = {name: float(values[name]) for name in VALUE_FEATURES}
    result.update(
        {f"{name}_mask": float(valid_value[name]) for name in VALUE_FEATURES}
    )
    if tuple(result) != DIRECTIONAL_FEATURES:
        raise AssertionError("directional feature ordering drifted")
    return result


def build_local_directional_features(
    raw_contexts: np.ndarray,
    anchor_ids: Sequence[int] | np.ndarray,
) -> pd.DataFrame:
    """Build the directional surface from the immutable local sequence cache."""

    raw = np.asarray(raw_contexts)
    ids = np.asarray(anchor_ids, dtype=np.int64)
    if raw.ndim != 3 or raw.shape[0] != len(ids) or raw.shape[1:] != (289, 10):
        raise DirectionalMemoryError("local raw context cache shape drifted")
    if len(np.unique(ids)) != len(ids):
        raise DirectionalMemoryError("local anchor ids are not unique")
    relative = np.linspace(-48.0, 0.0, raw.shape[1], dtype=np.float64)
    rows = [
        summarize_directional_energy_memory(raw[position, :, 0], raw[position, :, 3], relative)
        for position in range(len(raw))
    ]
    result = pd.DataFrame(rows, columns=DIRECTIONAL_FEATURES)
    result.insert(0, "anchor_id", ids)
    return result


def build_hourly_directional_features(
    contexts: Sequence[pd.DataFrame],
    anchor_ids: Sequence[int] | np.ndarray,
) -> pd.DataFrame:
    """Build the same surface for exact 49-row hourly ERA5 contexts."""

    ids = np.asarray(anchor_ids, dtype=np.int64)
    if len(contexts) != len(ids) or len(np.unique(ids)) != len(ids):
        raise DirectionalMemoryError("source contexts and unique anchor ids must align")
    relative = np.arange(-48, 1, dtype=np.float64)
    rows: list[dict[str, float]] = []
    for context in contexts:
        required = {"hs", "wvdir"}
        if len(context) != 49 or not required <= set(context.columns):
            raise DirectionalMemoryError("source context must contain 49 hourly hs/wvdir rows")
        rows.append(
            summarize_directional_energy_memory(
                context["hs"].to_numpy(dtype=np.float64),
                context["wvdir"].to_numpy(dtype=np.float64),
                relative,
            )
        )
    result = pd.DataFrame(rows, columns=DIRECTIONAL_FEATURES)
    result.insert(0, "anchor_id", ids)
    return result


class DirectionalContextTransferRegressor:
    """Frozen CatBoost transfer facade admitting one preregistered feature schema."""

    def __init__(
        self,
        feature_columns: Sequence[str],
        *,
        source_parameters: Mapping[str, Any] = SOURCE_CATBOOST_PARAMETERS,
        local_parameters: Mapping[str, Any] = LOCAL_CATBOOST_PARAMETERS,
    ) -> None:
        columns = tuple(str(column) for column in feature_columns)
        if not columns or len(set(columns)) != len(columns):
            raise DirectionalMemoryError("model feature schema is empty or duplicated")
        prohibited = {"station", "anchor_id", "anchor_time", "time", "calendar"}
        if prohibited.intersection(columns):
            raise DirectionalMemoryError("identity or absolute-time feature is prohibited")
        self._feature_columns = columns
        self._source_parameters = dict(source_parameters)
        self._local_parameters = dict(local_parameters)
        self._source_model: Any | None = None
        self._model: Any | None = None

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self._feature_columns

    @property
    def source_model(self) -> Any:
        if self._source_model is None:
            raise DirectionalMemoryError("source model is not fitted")
        return self._source_model

    @property
    def model(self) -> Any:
        if self._model is None:
            raise DirectionalMemoryError("transfer model is not fitted")
        return self._model

    def _context(self, frame: pd.DataFrame) -> pd.DataFrame:
        if tuple(frame.columns) != self._feature_columns:
            raise DirectionalMemoryError("model input feature names/order changed")
        result = frame.apply(pd.to_numeric, errors="raise").astype("float64")
        if np.isinf(result.to_numpy()).any():
            raise DirectionalMemoryError("model input contains infinity")
        return result.reset_index(drop=True)

    @staticmethod
    def _targets(targets: Any, rows: int) -> np.ndarray:
        values = np.asarray(targets, dtype=np.float64)
        if values.shape != (rows, len(LEADS)) or not np.isfinite(values).all():
            raise DirectionalMemoryError("target shape or finiteness changed")
        return values

    @staticmethod
    def _long(context: pd.DataFrame) -> pd.DataFrame:
        result = context.loc[context.index.repeat(len(LEADS))].reset_index(drop=True)
        result["lead_h"] = np.tile(np.asarray(LEADS, dtype=np.float64), len(context))
        return result

    def clone_pretrained(self) -> DirectionalContextTransferRegressor:
        if self._source_model is None:
            raise DirectionalMemoryError("source model is not fitted")
        clone = type(self)(
            self._feature_columns,
            source_parameters=self._source_parameters,
            local_parameters=self._local_parameters,
        )
        clone._source_model = copy.deepcopy(self._source_model)
        clone._model = clone._source_model
        if clone._source_model is self._source_model:
            raise DirectionalMemoryError("source model clone shares identity")
        return clone

    def fit_pretrain(self, features: pd.DataFrame, targets: Any) -> DirectionalContextTransferRegressor:
        from catboost import CatBoostRegressor

        context = self._context(features)
        values = self._targets(targets, len(context))
        model = CatBoostRegressor(**self._source_parameters)
        model.fit(self._long(context), values.reshape(-1), verbose=False)
        self._source_model = model
        self._model = model
        return self

    def continue_local(
        self,
        features: pd.DataFrame,
        targets: Any,
        *,
        current_hs: Sequence[float] | np.ndarray,
    ) -> DirectionalContextTransferRegressor:
        from catboost import CatBoostRegressor

        if self._source_model is None:
            raise DirectionalMemoryError("source model is not fitted")
        context = self._context(features)
        values = self._targets(targets, len(context))
        current = np.asarray(current_hs, dtype=np.float64)
        if current.shape != (len(context),) or not np.isfinite(current).all():
            raise DirectionalMemoryError("local current_hs changed shape or finiteness")
        weights = np.exp(-0.45 * np.maximum(current - 1.5, 0.0))
        model = CatBoostRegressor(**self._local_parameters)
        model.fit(
            self._long(context),
            values.reshape(-1),
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
            raise DirectionalMemoryError("prediction current_hs changed shape or finiteness")
        raw = np.asarray(self.model.predict(self._long(context)), dtype=np.float64)
        if raw.shape != (len(context) * len(LEADS),) or not np.isfinite(raw).all():
            raise DirectionalMemoryError("CatBoost prediction shape or finiteness changed")
        log_delta = raw.reshape(len(context), len(LEADS))
        return np.clip(np.expm1(np.log1p(current)[:, None] + log_delta), 0.0, 30.0)

    def save_model(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))


def apply_frozen_persistence_shrink(prediction: np.ndarray, current_hs: np.ndarray) -> np.ndarray:
    """Apply the frozen v1 20% persistence shrink on 12/18/24 hours."""

    result = np.asarray(prediction, dtype=np.float64).copy()
    current = np.asarray(current_hs, dtype=np.float64)
    if result.shape != (len(current), len(LEADS)):
        raise DirectionalMemoryError("persistence-shrink shape changed")
    for position, lead in enumerate(LEADS):
        if lead in (12, 18, 24):
            result[:, position] = 0.8 * result[:, position] + 0.2 * current
    return result


def apply_directional_increment(
    base_prediction: np.ndarray,
    enriched_prediction: np.ndarray,
    *,
    weight: float = 0.20,
) -> np.ndarray:
    """Apply the preregistered increment only at 18 and 24 hours."""

    base = np.asarray(base_prediction, dtype=np.float64)
    enriched = np.asarray(enriched_prediction, dtype=np.float64)
    if base.shape != enriched.shape or base.ndim != 2 or base.shape[1] != len(LEADS):
        raise DirectionalMemoryError("directional increment arrays changed shape")
    if weight != 0.20:
        raise DirectionalMemoryError("directional increment weight is frozen at 0.20")
    result = base.copy()
    for position, lead in enumerate(LEADS):
        if lead in (18, 24):
            result[:, position] = base[:, position] + weight * (
                enriched[:, position] - base[:, position]
            )
    return result
