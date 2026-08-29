"""Vectorized fixed-strength group-robust helpers for the P1 MS-TCN family."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def make_group_ids(keys: pd.DataFrame, *, minimum_rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Create station x layer x quarter groups and merge preregistered sparse cells."""

    required = {"station", "layer", "time"}
    if not required.issubset(keys.columns):
        raise ValueError(f"group keys missing: {sorted(required - set(keys.columns))}")
    if minimum_rows < 1:
        raise ValueError("minimum_rows must be positive")
    time = pd.to_datetime(keys["time"], utc=True, format="mixed")
    if time.isna().any():
        raise ValueError("group time contains NaT")
    quarter = time.dt.quarter.astype(str)
    raw = (
        keys["station"].astype(str)
        + "|layer_"
        + keys["layer"].astype(str)
        + "|q"
        + quarter
    ).tolist()
    counts = Counter(raw)
    merged = [value if counts[value] >= minimum_rows else "__SPARSE__" for value in raw]
    labels = tuple(sorted(set(merged)))
    code = {label: index for index, label in enumerate(labels)}
    ids = np.asarray([code[value] for value in merged], dtype=np.int16)
    observed = Counter(merged)
    return ids, {
        "definition": "station_x_layer_x_calendar_quarter_utc",
        "minimum_rows": int(minimum_rows),
        "raw_group_count": len(counts),
        "effective_group_count": len(labels),
        "sparse_raw_group_count": sum(count < minimum_rows for count in counts.values()),
        "effective_labels": list(labels),
        "effective_rows": {label: int(observed[label]) for label in labels},
    }


def materialize_group_batch(
    group_ids: np.ndarray, windows: Sequence[Any]
) -> tuple[np.ndarray, np.ndarray]:
    if not windows:
        raise ValueError("windows must not be empty")
    size = int(windows[0].window_size)
    values = np.full((len(windows), size), -1, dtype=np.int64)
    valid = np.zeros((len(windows), size), dtype=bool)
    for index, window in enumerate(windows):
        length = int(window.valid_length)
        rows = np.asarray(window.row_ids, dtype=np.int64)
        if length != len(rows):
            raise ValueError("window valid length differs from row ids")
        values[index, :length] = group_ids[rows]
        valid[index, :length] = True
    return values, valid


def changed_row_concentration(
    stations: Sequence[Any], candidate: Sequence[int], control: Sequence[int]
) -> dict[str, Any]:
    station = np.asarray(stations).astype(str)
    candidate_array = np.asarray(candidate, dtype=np.int8)
    control_array = np.asarray(control, dtype=np.int8)
    if station.shape != candidate_array.shape or station.shape != control_array.shape:
        raise ValueError("changed-row arrays are not aligned")
    changed = candidate_array != control_array
    total = int(changed.sum())
    counts = {
        name: int(np.sum(changed & (station == name))) for name in sorted(set(station.tolist()))
    }
    shares = {name: (count / total if total else 0.0) for name, count in counts.items()}
    return {
        "changed_rows": total,
        "by_station": counts,
        "shares": shares,
        "maximum_station_share": max(shares.values(), default=0.0),
    }


def robust_bce_from_rows(
    per_row_loss: Any,
    group_ids: Any,
    valid_mask: Any,
    *,
    group_count: int,
    strength: float,
) -> tuple[Any, dict[str, Any]]:
    """Return a fixed pooled/worst-group convex combination using torch scatter-add."""

    import torch

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    selected_loss = per_row_loss.masked_select(valid_mask)
    selected_group = group_ids.masked_select(valid_mask).long()
    if selected_loss.numel() == 0 or selected_group.numel() != selected_loss.numel():
        raise ValueError("valid robust BCE surface is empty or misaligned")
    if bool((selected_group < 0).any()) or bool((selected_group >= group_count).any()):
        raise ValueError("group id is outside the registered range")
    sums = torch.zeros(group_count, dtype=selected_loss.dtype, device=selected_loss.device)
    counts = torch.zeros(group_count, dtype=selected_loss.dtype, device=selected_loss.device)
    sums.scatter_add_(0, selected_group, selected_loss)
    counts.scatter_add_(0, selected_group, torch.ones_like(selected_loss))
    present = counts > 0
    means = sums[present] / counts[present]
    pooled = selected_loss.mean()
    worst = means.max()
    robust = (1.0 - strength) * pooled + strength * worst
    return robust, {
        "pooled_bce": pooled,
        "worst_group_bce": worst,
        "present_group_count": int(present.sum().detach().cpu()),
    }


__all__ = [
    "changed_row_concentration",
    "make_group_ids",
    "materialize_group_batch",
    "robust_bce_from_rows",
]
