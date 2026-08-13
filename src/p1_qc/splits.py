"""Leakage-resistant chronological validation splits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FoldWindowConfig, SplitConfig, _default_fold_windows


@dataclass(frozen=True)
class Fold:
    """One outer split; indices are positional and intended for ``.iloc``."""

    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_idx", np.asarray(self.train_idx, dtype=np.int64))
        object.__setattr__(self, "val_idx", np.asarray(self.val_idx, dtype=np.int64))


def _utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Seoul")
    return timestamp.tz_convert("UTC")


def _normalise_specs(
    specs: Sequence[FoldWindowConfig | Mapping[str, str]] | None,
    *,
    purge_days: int,
    cadence_minutes: int,
) -> tuple[FoldWindowConfig, ...]:
    source = _default_fold_windows() if specs is None else specs
    result: list[FoldWindowConfig] = []
    for item in source:
        if isinstance(item, FoldWindowConfig):
            result.append(item)
            continue
        val_start = _utc_timestamp(item["val_start"])
        if item.get("train_end"):
            train_end = str(item["train_end"])
        else:
            end = val_start - pd.Timedelta(days=purge_days, minutes=cadence_minutes)
            train_end = end.isoformat()
        result.append(
            FoldWindowConfig(
                name=str(item["name"]),
                train_end=train_end,
                val_start=str(item["val_start"]),
                val_end=str(item["val_end"]),
            )
        )
    return tuple(result)


def _positive_run_ids(
    frame: pd.DataFrame,
    parsed_time: pd.Series,
    *,
    cadence_minutes: int,
    group_columns: Sequence[str],
    label_column: str,
) -> np.ndarray:
    """Return gap-aware positive-run IDs in original positional order."""

    n = len(frame)
    positional = pd.DataFrame(
        {
            "__position": np.arange(n, dtype=np.int64),
            "__time": parsed_time.to_numpy(),
            "__label": pd.to_numeric(frame[label_column], errors="coerce")
            .fillna(0)
            .eq(1)
            .to_numpy(),
        }
    )
    for column in group_columns:
        positional[column] = frame[column].to_numpy()
    ordered = positional.sort_values([*group_columns, "__time", "__position"], kind="mergesort")
    grouped = ordered.groupby(list(group_columns), sort=False, observed=True)
    contiguous = grouped["__time"].diff().dt.total_seconds().eq(cadence_minutes * 60)
    prior_positive = grouped["__label"].shift(1).fillna(False).astype(bool)
    start = ordered["__label"] & (~contiguous | ~prior_positive)
    run_id = start.cumsum().where(ordered["__label"], -1).astype(np.int64)
    ordered["__run_id"] = run_id
    return ordered.sort_values("__position", kind="mergesort")["__run_id"].to_numpy()


def outer_folds(
    frame: pd.DataFrame,
    *,
    specs: Sequence[FoldWindowConfig | Mapping[str, str]] | None = None,
    config: SplitConfig | None = None,
    purge_days: int | None = None,
    cadence_minutes: int = 10,
    time_column: str = "time",
    group_columns: Sequence[str] = ("station", "layer"),
    label_column: str = "label",
    protect_positive_runs: bool | None = None,
    require_nonempty: bool = True,
) -> list[Fold]:
    """Build fixed 2025-Q2/Q3/Q4 rolling-origin outer folds.

    Training ends before a seven-day embargo.  If labels are available, any
    positive event touching a nominal validation interval is included in full
    so event boundaries are never split.  Training never expands and indices
    always refer to original row positions.
    """

    if time_column not in frame:
        raise KeyError(f"missing time column: {time_column}")
    for column in group_columns:
        if column not in frame:
            raise KeyError(f"missing group column: {column}")
    split_config = config or SplitConfig()
    purge = split_config.purge_days if purge_days is None else purge_days
    protect = (
        split_config.protect_positive_runs
        if protect_positive_runs is None
        else protect_positive_runs
    )
    if purge < 0:
        raise ValueError("purge_days cannot be negative")
    if specs is None and config is not None:
        specs = config.folds
    windows = _normalise_specs(
        specs,
        purge_days=purge,
        cadence_minutes=cadence_minutes,
    )
    time = pd.to_datetime(frame[time_column], errors="coerce", utc=True)
    if time.isna().any():
        raise ValueError(f"{int(time.isna().sum())} timestamps could not be parsed")
    positions = np.arange(len(frame), dtype=np.int64)
    run_ids: np.ndarray | None = None
    run_starts: dict[int, pd.Timestamp] = {}
    if protect and label_column in frame:
        run_ids = _positive_run_ids(
            frame,
            time,
            cadence_minutes=cadence_minutes,
            group_columns=group_columns,
            label_column=label_column,
        )
        positive_positions = np.flatnonzero(run_ids >= 0)
        if len(positive_positions):
            run_table = pd.DataFrame(
                {
                    "run_id": run_ids[positive_positions],
                    "time": time.iloc[positive_positions].to_numpy(),
                }
            )
            run_starts = {
                int(run_id): start
                for run_id, start in run_table.groupby("run_id", sort=False)["time"].min().items()
            }

    folds: list[Fold] = []
    for spec in windows:
        train_end = _utc_timestamp(spec.train_end)
        val_start = _utc_timestamp(spec.val_start)
        val_end = _utc_timestamp(spec.val_end)
        if not train_end < val_start < val_end:
            raise ValueError(f"invalid time order for fold {spec.name}")
        embargo = val_start - train_end
        minimum_embargo = pd.Timedelta(days=purge)
        if embargo < minimum_embargo:
            raise ValueError(
                f"fold {spec.name} embargo {embargo} is shorter than {minimum_embargo}"
            )
        train_mask = time.le(train_end).to_numpy(copy=True)
        nominal_val_mask = (time.ge(val_start) & time.lt(val_end)).to_numpy(copy=True)
        val_mask = nominal_val_mask.copy()
        if run_ids is not None:
            # Assign a whole positive event to the fold containing its start.
            # This prevents a boundary-crossing event from appearing in both
            # neighbouring validation folds.
            assigned_runs = np.asarray(
                [run_id for run_id, start in run_starts.items() if val_start <= start < val_end],
                dtype=np.int64,
            )
            val_mask[run_ids >= 0] = False
            if len(assigned_runs):
                val_mask |= np.isin(run_ids, assigned_runs)
        # The explicit temporal embargo takes precedence over run expansion.
        overlap = train_mask & val_mask
        if overlap.any():
            raise ValueError(
                f"fold {spec.name} has {int(overlap.sum())} overlapping train/validation rows"
            )
        train_idx = positions[train_mask]
        val_idx = positions[val_mask]
        if require_nonempty and (not len(train_idx) or not len(val_idx)):
            raise ValueError(
                f"fold {spec.name} is empty: train={len(train_idx)}, validation={len(val_idx)}"
            )
        folds.append(Fold(spec.name, train_idx, val_idx, train_end, val_start, val_end))
    return folds


def build_outer_folds(*args, **kwargs) -> list[Fold]:
    return outer_folds(*args, **kwargs)


def year_transfer_fold(
    frame: pd.DataFrame,
    *,
    train_year: int = 2024,
    validation_year: int = 2025,
    validation_end: str = "2025-07-01T00:00:00+09:00",
    station: str | None = "S-ORS",
) -> Fold:
    """Return the documented S-ORS seasonal-transfer stress split."""

    year = pd.to_numeric(frame["year"], errors="coerce")
    time = pd.to_datetime(frame["time"], errors="coerce", utc=True)
    station_mask = np.ones(len(frame), dtype=bool)
    if station is not None:
        station_mask = frame["station"].eq(station).to_numpy()
    train_mask = year.eq(train_year).to_numpy() & station_mask
    val_end = _utc_timestamp(validation_end)
    val_mask = year.eq(validation_year).to_numpy() & time.lt(val_end).to_numpy() & station_mask
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    if not len(train_idx) or not len(val_idx):
        raise ValueError("year-transfer fold has an empty side")
    train_end = time.iloc[train_idx].max()
    val_start = time.iloc[val_idx].min()
    return Fold(
        f"{station or 'all'}_{train_year}_to_{validation_year}_h1",
        train_idx,
        val_idx,
        train_end,
        val_start,
        val_end,
    )


def group_holdout_fold(
    frame: pd.DataFrame,
    *,
    holdout_station: str = "G-ORS",
) -> Fold:
    """Return a station holdout used only as a robustness stress test."""

    time = pd.to_datetime(frame["time"], errors="coerce", utc=True)
    val_mask = frame["station"].eq(holdout_station).to_numpy()
    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    if not len(train_idx) or not len(val_idx):
        raise ValueError("group holdout fold has an empty side")
    return Fold(
        f"holdout_{holdout_station}",
        train_idx,
        val_idx,
        time.iloc[train_idx].max(),
        time.iloc[val_idx].min(),
        time.iloc[val_idx].max() + pd.Timedelta(minutes=10),
    )


__all__ = [
    "Fold",
    "build_outer_folds",
    "group_holdout_fold",
    "outer_folds",
    "year_transfer_fold",
]
