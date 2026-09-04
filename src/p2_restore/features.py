"""Leakage-safe public-layer features for P2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from p2_restore.data import KEYS, P2Data

PUBLIC_LAYERS = (1, 5, 6, 7, 8)
TARGET_LAYERS = (2, 3, 4)


@dataclass(frozen=True)
class FeatureTable:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]


def _wide(observations: pd.DataFrame, value: str) -> pd.DataFrame:
    return observations.pivot(index="time", columns="layer", values=value).sort_index()


def _nearest_public_baseline(
    public_temp: np.ndarray, public_depth: np.ndarray, target_depth: np.ndarray
) -> np.ndarray:
    result = np.full(len(target_depth), np.nan, dtype=float)
    for row in range(len(target_depth)):
        keep = np.isfinite(public_temp[row]) & np.isfinite(public_depth[row])
        values, depths = public_temp[row, keep], public_depth[row, keep]
        if len(values) < 2 or not np.isfinite(target_depth[row]):
            continue
        order = np.argsort(depths)
        depths, values = depths[order], values[order]
        lower = np.flatnonzero(depths <= target_depth[row])
        upper = np.flatnonzero(depths >= target_depth[row])
        if len(lower) and len(upper):
            left, right = lower[-1], upper[0]
        else:
            left, right = np.sort(np.argsort(np.abs(depths - target_depth[row]))[:2])
        span = depths[right] - depths[left]
        result[row] = (
            values[left]
            if span == 0
            else values[left]
            + (values[right] - values[left]) * (target_depth[row] - depths[left]) / span
        )
    return result


def _common_features(observations: pd.DataFrame) -> tuple[pd.Index, dict[str, np.ndarray]]:
    temp, psal = _wide(observations, "temp"), _wide(observations, "psal")
    depth, nominal = _wide(observations, "depth"), _wide(observations, "nominal_depth")
    times = temp.index
    common: dict[str, np.ndarray] = {}
    for layer in PUBLIC_LAYERS:
        for prefix, wide in (
            ("temp", temp),
            ("psal", psal),
            ("depth", depth),
            ("nominal", nominal),
        ):
            common[f"{prefix}_{layer}"] = (
                wide.get(layer, pd.Series(index=times, dtype=float)).reindex(times).to_numpy(float)
            )
    public = np.column_stack([common[f"temp_{layer}"] for layer in PUBLIC_LAYERS])
    count = np.isfinite(public).sum(axis=1)
    mean = np.divide(
        np.nansum(public, axis=1), count, out=np.full(len(public), np.nan), where=count > 0
    )
    variance = np.divide(
        np.nansum((public - mean[:, None]) ** 2, axis=1),
        count,
        out=np.full(len(public), np.nan),
        where=count > 0,
    )
    value_range = np.full(len(public), np.nan)
    populated = count > 0
    value_range[populated] = np.nanmax(public[populated], axis=1) - np.nanmin(
        public[populated], axis=1
    )
    common["public_temp_count"] = count
    common["public_temp_mean"] = mean
    common["public_temp_std"] = np.sqrt(variance)
    common["public_temp_range"] = value_range
    kst = pd.to_datetime(times, utc=True).tz_convert("Asia/Seoul")
    minute = kst.hour.to_numpy() * 60 + kst.minute.to_numpy()
    doy = kst.dayofyear.to_numpy() + minute / 1440
    parsed_time = pd.to_datetime(times, utc=True).as_unit("ns")
    epoch_seconds = parsed_time.asi8 / 1e9
    common.update(
        {
            "year": kst.year.to_numpy(),
            "elapsed_days": (epoch_seconds - epoch_seconds.min()) / 86_400,
            "doy_sin": np.sin(2 * np.pi * doy / 365.2425),
            "doy_cos": np.cos(2 * np.pi * doy / 365.2425),
            "hour_sin": np.sin(2 * np.pi * minute / 1440),
            "hour_cos": np.cos(2 * np.pi * minute / 1440),
            "m2_sin": np.sin(2 * np.pi * epoch_seconds / (12.42 * 3600)),
            "m2_cos": np.cos(2 * np.pi * epoch_seconds / (12.42 * 3600)),
        }
    )
    return times, common


def _finalize(frame: pd.DataFrame) -> FeatureTable:
    frame["temp_1_minus_5"] = frame["temp_1"] - frame["temp_5"]
    frame["psal_1_minus_5"] = frame["psal_1"] - frame["psal_5"]
    for layer in PUBLIC_LAYERS:
        frame[f"temp_{layer}_minus_baseline"] = frame[f"temp_{layer}"] - frame["baseline"]
    excluded = {*KEYS, "target", "residual"}
    features = tuple(sorted(column for column in frame.columns if column not in excluded))
    forbidden = {f"temp_{layer}" for layer in TARGET_LAYERS} | {
        f"psal_{layer}" for layer in TARGET_LAYERS
    }
    if forbidden.intersection(features):
        raise AssertionError("hidden target-layer values leaked into features")
    return FeatureTable(frame.reset_index(drop=True), features)


def build_training_features(observations: pd.DataFrame) -> FeatureTable:
    times, common = _common_features(observations)
    temp, nominal = (
        _wide(observations, "temp").reindex(times),
        _wide(observations, "nominal_depth").reindex(times),
    )
    public_temp = np.column_stack([common[f"temp_{layer}"] for layer in PUBLIC_LAYERS])
    public_depth = np.column_stack([common[f"nominal_{layer}"] for layer in PUBLIC_LAYERS])
    rows: list[pd.DataFrame] = []
    for layer in TARGET_LAYERS:
        target = temp[layer].to_numpy(float)
        target_depth = nominal[layer].to_numpy(float)
        baseline = _nearest_public_baseline(public_temp, public_depth, target_depth)
        keep = (
            np.isfinite(target)
            & np.isfinite(target_depth)
            & np.isfinite(baseline)
            & (common["public_temp_count"] >= 2)
        )
        part = pd.DataFrame({name: values[keep] for name, values in common.items()})
        part["station"] = "S-ORS"
        part["layer"] = layer
        part["time"] = times[keep].astype(str)
        part["target_depth"] = target_depth[keep]
        part["baseline"] = baseline[keep]
        part["target"] = target[keep]
        part["residual"] = target[keep] - baseline[keep]
        rows.append(part)
    return _finalize(pd.concat(rows, ignore_index=True))


def build_test_features(data: P2Data) -> FeatureTable:
    times, common = _common_features(data.observations)
    lookup = pd.DataFrame(common, index=times)
    index = data.test_index.copy()
    index["time_key"] = pd.to_datetime(index["time"], utc=True).astype(str)
    lookup.index = pd.to_datetime(lookup.index, utc=True).astype(str)
    frame = index.join(lookup, on="time_key", validate="many_to_one")
    frame = frame.merge(
        data.baseline, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_baseline")
    )
    frame = frame.rename(columns={"nominal_depth": "target_depth", "temp": "baseline"})
    frame = frame.drop(columns=["time_key"])
    if frame["baseline"].isna().any() or (frame["public_temp_count"] < 2).any():
        raise ValueError("test feature construction lost required baseline/public support")
    return _finalize(frame)
