"""Strictly backward-looking raw-row features for the superseding P1 Gen4r2."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

CAUSAL_FEATURE_COLUMNS = (
    "temp_raw",
    "psal_raw",
    "depth_raw",
    "psal_missing",
    "depth_missing",
    "has_gap_before",
    "temp_diff_1",
    "temp_abs_diff_1",
    "temp_backward_acceleration",
)

_REQUIRED_COLUMNS = ("station", "layer", "time", "temp", "psal", "depth")
_CADENCE_NS = int(pd.Timedelta(minutes=10).value)


def _strict_ids(values: Sequence[int] | np.ndarray, *, size: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError("prefix IDs must be a non-empty vector")
    if not np.issubdtype(result.dtype, np.integer):
        raise TypeError("prefix IDs must be integers")
    result = result.astype(np.int64, copy=False)
    if np.unique(result).size != len(result):
        raise ValueError("prefix IDs must be unique")
    if result.min() < 0 or result.max() >= size:
        raise IndexError("prefix IDs are outside the raw frame")
    return result


def build_causal_raw_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build current-row and exact-cadence backward-only features.

    The function never reads targets and never performs forward shifts, centered
    windows, terminal-run length calculations, interpolation, or centered
    rolling statistics. Output is restored to the caller's row order.
    """

    if missing := sorted(set(_REQUIRED_COLUMNS).difference(frame.columns)):
        raise KeyError(f"raw causal feature columns missing: {missing}")
    if len(frame) == 0:
        raise ValueError("raw causal feature frame cannot be empty")
    work = frame.loc[:, list(_REQUIRED_COLUMNS)].copy()
    work["_source_position"] = np.arange(len(work), dtype=np.int64)
    work["_time_ns"] = (
        pd.to_datetime(work["time"], errors="raise", utc=True, format="mixed")
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )
    for column in ("temp", "psal", "depth"):
        work[column] = pd.to_numeric(work[column], errors="coerce").astype(np.float64)
    work.sort_values(
        ["station", "layer", "_time_ns", "_source_position"],
        kind="mergesort",
        inplace=True,
    )
    grouped = work.groupby(["station", "layer"], sort=False, observed=True)
    previous_time = grouped["_time_ns"].shift(1)
    exact_previous = (work["_time_ns"] - previous_time) == _CADENCE_NS
    previous_temp = grouped["temp"].shift(1)
    temp_diff = (work["temp"] - previous_temp).where(exact_previous)
    previous_diff = temp_diff.groupby(
        [work["station"], work["layer"]], sort=False, observed=True
    ).shift(1)
    acceleration = (temp_diff - previous_diff).where(
        exact_previous & previous_diff.notna()
    )
    output = pd.DataFrame(
        {
            "_source_position": work["_source_position"].to_numpy(np.int64),
            "temp_raw": work["temp"].to_numpy(np.float64),
            "psal_raw": work["psal"].to_numpy(np.float64),
            "depth_raw": work["depth"].to_numpy(np.float64),
            "psal_missing": work["psal"].isna().to_numpy(np.float32),
            "depth_missing": work["depth"].isna().to_numpy(np.float32),
            "has_gap_before": (~exact_previous).to_numpy(np.float32),
            "temp_diff_1": temp_diff.to_numpy(np.float64),
            "temp_abs_diff_1": temp_diff.abs().to_numpy(np.float64),
            "temp_backward_acceleration": acceleration.to_numpy(np.float64),
        }
    )
    output.sort_values("_source_position", kind="mergesort", inplace=True)
    output.drop(columns="_source_position", inplace=True)
    output.reset_index(drop=True, inplace=True)
    output = output.loc[:, CAUSAL_FEATURE_COLUMNS].astype(np.float32)
    if len(output) != len(frame):
        raise AssertionError("causal feature row count differs")
    return output


def build_exact_prefix_causal_matrix(
    frame: pd.DataFrame,
    prefix_ids: Sequence[int] | np.ndarray,
    *,
    full_reference: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Rebuild features from prefix rows alone and prove equality to a full cache."""

    ids = _strict_ids(prefix_ids, size=len(frame))
    reference = np.asarray(full_reference, dtype=np.float32)
    if reference.shape != (len(frame), len(CAUSAL_FEATURE_COLUMNS)):
        raise ValueError("full causal reference shape differs")
    prefix = build_causal_raw_features(frame.iloc[ids].reset_index(drop=True)).to_numpy(
        np.float32
    )
    if not np.array_equal(prefix, reference[ids], equal_nan=True):
        raise PermissionError("full-cache prefix differs from raw prefix-only causal rebuild")
    result = np.zeros_like(reference)
    result[ids] = prefix
    digest = hashlib.sha256()
    digest.update(np.asarray(ids, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(prefix, dtype="<f4").tobytes(order="C"))
    return result, digest.hexdigest()


def assert_future_value_invariance(
    frame: pd.DataFrame, prefix_ids: Sequence[int] | np.ndarray
) -> str:
    """Perturb every future raw value and require prefix features to remain exact."""

    ids = _strict_ids(prefix_ids, size=len(frame))
    future = np.setdiff1d(np.arange(len(frame), dtype=np.int64), ids, assume_unique=False)
    if len(future) == 0:
        raise ValueError("future perturbation audit requires at least one held-out row")
    original = build_causal_raw_features(frame)
    changed = frame.copy()
    for offset, column in enumerate(("temp", "psal", "depth"), 1):
        changed.iloc[future, changed.columns.get_loc(column)] = (
            1_000_000.0 * offset + np.arange(len(future), dtype=np.float64)
        )
    perturbed = build_causal_raw_features(changed)
    left = original.iloc[ids].to_numpy(np.float32)
    right = perturbed.iloc[ids].to_numpy(np.float32)
    if not np.array_equal(left, right, equal_nan=True):
        raise PermissionError("future raw-value perturbation changed a prefix feature")
    return hashlib.sha256(np.asarray(left, dtype="<f4").tobytes(order="C")).hexdigest()


__all__ = [
    "CAUSAL_FEATURE_COLUMNS",
    "assert_future_value_invariance",
    "build_causal_raw_features",
    "build_exact_prefix_causal_matrix",
]
