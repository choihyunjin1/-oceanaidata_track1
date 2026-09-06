"""Unchanged pure event/day weighting from the historical runner."""
import math
import numpy as np
import pandas as pd

def _event_day_weight(metadata: pd.DataFrame, target: np.ndarray) -> np.ndarray:
    y = np.asarray(target, dtype=np.int8)
    work = metadata.loc[:, ["station", "layer", "time"]].reset_index(drop=True).copy()
    work["__position"] = np.arange(len(work), dtype=np.int64)
    work["__target"] = y
    work["__time"] = pd.to_datetime(work["time"], errors="raise", utc=True, format="mixed")
    work.sort_values(["station", "layer", "__time", "__position"], inplace=True)
    grouped = work.groupby(["station", "layer"], sort=False, observed=True)
    contiguous = grouped["__time"].diff().dt.total_seconds().eq(600)
    prior = grouped["__target"].shift(1).fillna(0).eq(1)
    starts = work["__target"].eq(1) & (~contiguous | ~prior)
    work["__event"] = starts.cumsum().where(work["__target"].eq(1), -1).astype(np.int64)
    positive = work["__target"].eq(1)
    event_length = work.loc[positive].groupby("__event", sort=False)["__event"].transform("size")
    pos_raw = 1.0 / np.sqrt(event_length.to_numpy(dtype=float))
    pos_raw /= pos_raw.mean()
    day = work["__time"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    normal = ~positive
    normal_length = (
        work.loc[normal]
        .assign(__day=day.loc[normal])
        .groupby(["station", "layer", "__day"], sort=False, observed=True)["__day"]
        .transform("size")
    )
    normal_raw = 1.0 / np.sqrt(normal_length.to_numpy(dtype=float))
    normal_raw /= normal_raw.mean()
    ordered_weight = np.empty(len(work), dtype=np.float64)
    ordered_weight[positive.to_numpy()] = pos_raw * math.sqrt(
        max(1, normal.sum()) / max(1, positive.sum())
    )
    ordered_weight[normal.to_numpy()] = normal_raw
    work["__weight"] = ordered_weight
    restored = work.sort_values("__position", kind="mergesort")
    result = restored["__weight"].to_numpy(dtype=np.float32)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise RuntimeError("invalid event/day training weight")
    return result
