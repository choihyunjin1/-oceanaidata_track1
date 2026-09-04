"""Compact NLinear/DLinear-inspired ridge residual helpers for P3.

The model is deliberately small: it predicts six future changes from the current
significant wave height using past-only multi-resolution lags and trend/seasonal
summaries, then optionally applies a protected long-lead blend to a frozen incumbent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

from .tsmixer_residual import DERIVED_CHANNELS, hourly_derived_numpy

CHANNEL_NAMES = (
    "hs",
    "tp",
    "hmax",
    "wspd",
    "gust",
    "caph",
    "airt",
    "relh",
    "wvdir_sin",
    "wvdir_cos",
    "wdir_sin",
    "wdir_cos",
)
LAG_HOURS = (1, 2, 3, 6, 9, 12, 18, 24, 36, 48)
SUMMARY_HOURS = (3, 6, 12, 24, 48)
LEADS = (3, 6, 9, 12, 18, 24)


def _fill_case(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    time = np.arange(len(output), dtype=np.float64)
    for channel in range(output.shape[1]):
        finite = np.isfinite(output[:, channel])
        if finite.any():
            output[:, channel] = np.interp(time, time[finite], output[finite, channel])
        else:
            output[:, channel] = 0.0
    return output


def compact_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    names.extend(f"{channel}__current" for channel in CHANNEL_NAMES)
    for lag in LAG_HOURS:
        names.extend(f"{channel}__delta_{lag}h" for channel in CHANNEL_NAMES)
    for horizon in SUMMARY_HOURS:
        for statistic in ("mean_delta", "std", "slope"):
            names.extend(f"{channel}__{statistic}_{horizon}h" for channel in CHANNEL_NAMES)
    names.extend(f"{channel}__missing_fraction" for channel in CHANNEL_NAMES)
    names.extend(f"{channel}__trailing_missing_hours" for channel in CHANNEL_NAMES)
    return tuple(names)


def build_compact_features(raw: np.ndarray) -> np.ndarray:
    """Build past-only multi-resolution NLinear-style features."""

    derived = hourly_derived_numpy(np.asarray(raw, dtype=np.float32)).astype(np.float64)
    missing = ~np.isfinite(derived)
    output = np.empty((len(derived), len(compact_feature_names())), dtype=np.float64)
    for case in range(len(derived)):
        filled = _fill_case(derived[case])
        current = filled[-1]
        features: list[np.ndarray] = [current]
        for lag in LAG_HOURS:
            features.append(filled[-1 - lag] - current)
        for horizon in SUMMARY_HOURS:
            window = filled[-(horizon + 1) :]
            features.extend(
                [
                    np.mean(window, axis=0) - current,
                    np.std(window, axis=0),
                    (window[-1] - window[0]) / float(horizon),
                ]
            )
        features.append(np.mean(missing[case], axis=0))
        trailing = np.zeros(DERIVED_CHANNELS, dtype=np.float64)
        for channel in range(DERIVED_CHANNELS):
            count = 0
            for value in missing[case, ::-1, channel]:
                if not value:
                    break
                count += 1
            trailing[channel] = count
        features.append(trailing)
        output[case] = np.concatenate(features)
    output[~np.isfinite(output)] = 0.0
    return output


@dataclass
class StandardizedStationRidge:
    alpha: float
    means: dict[int, np.ndarray]
    scales: dict[int, np.ndarray]
    models: dict[int, Ridge]

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        station: np.ndarray,
        target_delta: np.ndarray,
        rows: np.ndarray,
        *,
        alpha: float,
    ) -> StandardizedStationRidge:
        means: dict[int, np.ndarray] = {}
        scales: dict[int, np.ndarray] = {}
        models: dict[int, Ridge] = {}
        selected = np.asarray(rows, dtype=np.int64)
        for station_id in sorted(np.unique(station[selected]).astype(int)):
            local = selected[station[selected] == station_id]
            if len(local) < 20:
                raise ValueError(f"station {station_id} has insufficient ridge cases")
            x = np.asarray(features[local], dtype=np.float64)
            mean = np.mean(x, axis=0)
            scale = np.std(x, axis=0)
            scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
            design = (x - mean) / scale
            model = Ridge(alpha=float(alpha), fit_intercept=True, solver="cholesky")
            model.fit(design, np.asarray(target_delta[local], dtype=np.float64))
            means[station_id] = mean
            scales[station_id] = scale
            models[station_id] = model
        return cls(float(alpha), means, scales, models)

    def predict(self, features: np.ndarray, station: np.ndarray, rows: np.ndarray) -> np.ndarray:
        selected = np.asarray(rows, dtype=np.int64)
        output = np.empty((len(selected), len(LEADS)), dtype=np.float64)
        for station_id in sorted(np.unique(station[selected]).astype(int)):
            positions = np.flatnonzero(station[selected] == station_id)
            if station_id not in self.models:
                raise ValueError(f"unseen station {station_id}")
            design = (
                np.asarray(features[selected[positions]], dtype=np.float64) - self.means[station_id]
            ) / self.scales[station_id]
            output[positions] = self.models[station_id].predict(design)
        return output


def absolute_prediction(current_hs: np.ndarray, predicted_delta: np.ndarray) -> np.ndarray:
    current = np.asarray(current_hs, dtype=np.float64).reshape(-1, 1)
    delta = np.asarray(predicted_delta, dtype=np.float64)
    if delta.shape != (len(current), len(LEADS)):
        raise ValueError("six-lead delta prediction shape changed")
    return np.clip(current + delta, 0.0, 30.0)


def protected_long_blend(
    incumbent: np.ndarray, challenger: np.ndarray, *, long_weight: float
) -> np.ndarray:
    base = np.asarray(incumbent, dtype=np.float64)
    model = np.asarray(challenger, dtype=np.float64)
    if base.shape != model.shape or base.ndim != 2 or base.shape[1] != len(LEADS):
        raise ValueError("incumbent and challenger must be aligned six-lead matrices")
    output = base.copy()
    output[:, 3:] = (1.0 - float(long_weight)) * base[:, 3:] + float(long_weight) * model[:, 3:]
    return np.clip(output, 0.0, 30.0)


__all__ = [
    "CHANNEL_NAMES",
    "LEADS",
    "StandardizedStationRidge",
    "absolute_prediction",
    "build_compact_features",
    "compact_feature_names",
    "protected_long_blend",
]
