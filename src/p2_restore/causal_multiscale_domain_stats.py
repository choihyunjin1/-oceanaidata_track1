"""Causal fixed-horizon public-layer domain-stat features for P2."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from p2_restore.features import PUBLIC_LAYERS


def augment_tokens(
    frame: pd.DataFrame,
    base_tokens: np.ndarray,
    half_life_steps: Sequence[int],
    min_periods: int,
    temp_scale_floor: float,
    psal_scale_floor: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Append shifted EWM z-scores and support flags without future access."""
    required = ["time"] + [
        f"{channel}_{layer}"
        for layer in PUBLIC_LAYERS
        for channel in ("temp", "psal")
    ]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"domain-stat columns missing: {missing}")
    if base_tokens.shape[:2] != (len(frame), len(PUBLIC_LAYERS)):
        raise ValueError("base token geometry drift")
    if any(int(value) <= 0 for value in half_life_steps):
        raise ValueError("half lives must be positive")

    source = frame[required].copy()
    source["time"] = pd.to_datetime(source["time"], utc=True)
    conflicts = source.groupby("time", sort=False)[required[1:]].nunique(dropna=False).max().max()
    if int(conflicts) > 1:
        raise ValueError("public values conflict across target-layer rows")
    unique = source.sort_values("time").drop_duplicates("time", keep="first").reset_index(drop=True)
    timeline = pd.DatetimeIndex(unique["time"])
    if timeline.has_duplicates or not timeline.is_monotonic_increasing:
        raise ValueError("domain-stat timeline is not strictly ordered")
    delta_minutes = np.diff(timeline.as_unit("ns").asi8) / 60_000_000_000
    exact_ten_minute_share = float(np.mean(delta_minutes == 10.0)) if len(delta_minutes) else 1.0

    horizons = tuple(int(value) for value in half_life_steps)
    extra = np.zeros((len(unique), len(PUBLIC_LAYERS), len(horizons) * 4), dtype=np.float32)
    support_receipt: dict[str, float] = {}
    for layer_index, layer in enumerate(PUBLIC_LAYERS):
        offset = 0
        for channel, floor in (("temp", temp_scale_floor), ("psal", psal_scale_floor)):
            current = pd.to_numeric(unique[f"{channel}_{layer}"], errors="coerce")
            past = current.shift(1)
            for horizon in horizons:
                mean = past.ewm(
                    halflife=horizon, adjust=False, min_periods=int(min_periods)
                ).mean()
                variance = past.ewm(
                    halflife=horizon, adjust=False, min_periods=int(min_periods)
                ).var(bias=True)
                scale = np.sqrt(np.maximum(variance.to_numpy(float), float(floor) ** 2))
                valid = (
                    np.isfinite(current.to_numpy(float))
                    & np.isfinite(mean.to_numpy(float))
                    & np.isfinite(scale)
                )
                z = np.zeros(len(unique), dtype=float)
                z[valid] = (
                    current.to_numpy(float)[valid] - mean.to_numpy(float)[valid]
                ) / scale[valid]
                extra[:, layer_index, offset] = np.clip(z, -12.0, 12.0).astype(np.float32)
                extra[:, layer_index, offset + 1] = valid.astype(np.float32)
                support_receipt[f"layer{layer}:{channel}:h{horizon}"] = float(valid.mean())
                offset += 2

    lookup = pd.Index(timeline.as_unit("ns").asi8)
    row_time_ns = pd.to_datetime(frame["time"], utc=True).dt.as_unit("ns").array.asi8
    positions = lookup.get_indexer(row_time_ns)
    if np.any(positions < 0):
        raise ValueError("domain-stat row alignment failed")
    augmented = np.concatenate((base_tokens.astype(np.float32), extra[positions]), axis=2)
    if not np.isfinite(augmented).all():
        raise ValueError("domain-stat tokens are non-finite")
    receipt = {
        "causal_shift_rows": 1,
        "grouping": "each_public_layer_x_channel_independently",
        "half_life_steps": list(horizons),
        "minimum_periods": int(min_periods),
        "unique_timestamps": int(len(unique)),
        "exact_10min_step_share": exact_ten_minute_share,
        "base_token_features": int(base_tokens.shape[2]),
        "added_domain_stat_features": int(extra.shape[2]),
        "total_token_features": int(augmented.shape[2]),
        "support_share": support_receipt,
    }
    return augmented, receipt
