"""Three-role chronological cross-fit helpers for the sealed P2 rank-one veto."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def contiguous_time_groups(
    times: pd.Series | pd.DatetimeIndex,
    *,
    groups: int = 3,
    minimum_profiles: int = 100,
) -> tuple[pd.DatetimeIndex, ...]:
    """Split sorted unique timestamps into deterministic contiguous balanced groups."""

    unique = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).unique().sort_values()
    if groups != 3:
        raise ValueError("the preregistered contract requires exactly three groups")
    parts = tuple(pd.DatetimeIndex(part) for part in np.array_split(unique, groups))
    if any(len(part) < minimum_profiles for part in parts):
        raise ValueError("three-way cross-fit group lacks minimum profile support")
    if len(pd.DatetimeIndex(np.concatenate([part.asi8 for part in parts])).unique()) != len(unique):
        raise ValueError("time groups overlap or omit timestamps")
    if not all(parts[index][-1] < parts[index + 1][0] for index in range(groups - 1)):
        raise ValueError("time groups are not strictly chronological")
    return parts


def time_group_sha256(times: pd.DatetimeIndex) -> str:
    """Hash nanosecond UTC timestamps for an auditable label-free partition seal."""

    values = pd.DatetimeIndex(times).as_unit("ns").asi8.astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


__all__ = ["contiguous_time_groups", "time_group_sha256"]
