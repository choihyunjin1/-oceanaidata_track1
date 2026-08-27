"""Station-agnostic ERA5 context-transfer primitives for P3.

This module deliberately has no downloader, runner, or file-loading entry point.  It
turns an already-canonical, hourly ERA5 time series into the same past-only feature
names used by the local P3 cache, builds complete six-lead source cases, and exposes
a fixed CatBoost pretrain/continuation facade.  Absolute time and source/station
identity are metadata only and cannot enter the model facade.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

LEADS = (3, 6, 9, 12, 18, 24)
LAG_HOURS = (3, 6, 12, 24, 48)
WINDOW_HOURS = (6, 12, 24, 48)
WINDOW_STATISTICS = ("mean", "std", "delta", "slope")
ANCHOR_SPACING_HOURS = 6
CONTEXT_HOURS = 48
SOURCE_CONTEXT_ROWS = CONTEXT_HOURS + 1
MINIMUM_CURRENT_HS = 1.5
VALIDATION_GAP_HOURS = 78
FOOTPRINT_CONTEXT_HOURS = 48
FOOTPRINT_TARGET_HOURS = 24
SOURCE_TRAIN_YEARS = tuple(range(2014, 2021))

CANONICAL_COLUMNS = (
    "hs",
    "tp",
    "hmax",
    "wvdir",
    "wspd",
    "wdir",
    "airt",
    "relh",
    "caph",
)
COMMON_SERIES = (
    "hs",
    "tp",
    "hmax",
    "wvdir_sin",
    "wvdir_cos",
    "wspd",
    "wdir_sin",
    "wdir_cos",
    "airt",
    "relh",
    "caph",
    "wave_energy",
    "wind_input_proxy",
)
TARGET_COLUMNS = tuple(f"target_log_delta_{lead}h" for lead in LEADS)
FUTURE_HS_COLUMNS = tuple(f"future_hs_{lead}h" for lead in LEADS)

SOURCE_CATBOOST_PARAMETERS: Mapping[str, Any] = MappingProxyType(
    {
        "loss_function": "RMSE",
        "iterations": 600,
        "depth": 8,
        "learning_rate": 0.04,
        "l2_leaf_reg": 8.0,
        "random_seed": 20260824,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": False,
    }
)
LOCAL_CATBOOST_PARAMETERS: Mapping[str, Any] = MappingProxyType(
    {
        "loss_function": "RMSE",
        "iterations": 250,
        "depth": 8,
        "learning_rate": 0.03,
        "l2_leaf_reg": 12.0,
        "random_seed": 20260824,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": False,
    }
)


class ERA5ContextTransferError(ValueError):
    """Fail-closed validation error for the context-transfer experiment."""


@dataclass(frozen=True)
class ERA5SourceCases:
    """Aligned source metadata, past-only features, and six log-relative targets."""

    anchors: pd.DataFrame
    features: pd.DataFrame
    log_delta_targets: np.ndarray
    current_hs: np.ndarray


def common_feature_columns() -> tuple[str, ...]:
    """Return the frozen 286-column source/local common feature surface."""

    columns: list[str] = []
    for name in COMMON_SERIES:
        columns.append(f"{name}_current")
        columns.extend(f"{name}_lag_{hour}h" for hour in LAG_HOURS)
        for hour in WINDOW_HOURS:
            columns.extend(
                f"{name}_{statistic}_{hour}h" for statistic in WINDOW_STATISTICS
            )
    if len(columns) != 286 or len(set(columns)) != len(columns):
        raise AssertionError("ERA5 common feature surface is not 286 unique columns")
    return tuple(columns)


# A descriptive alias is useful to callers that already have other P3 feature surfaces.
common_context_feature_columns = common_feature_columns


def _utc_timestamp(value: Any, *, label: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ERA5ContextTransferError(f"{label} is missing")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def canonicalize_era5_hourly(
    frame: pd.DataFrame,
    *,
    time_column: str = "time",
) -> pd.DataFrame:
    """Select canonical values and normalize timestamps without exposing metadata.

    Naive ERA5 timestamps are interpreted as UTC.  Aware timestamps are converted to
    UTC.  The implementation is cadence-agnostic: it validates ordering and duplicate
    instants but never relies on a fixed number of rows per hour.
    """

    if time_column in CANONICAL_COLUMNS:
        raise ERA5ContextTransferError("time column cannot also be a canonical value column")
    time_in_index = time_column not in frame and isinstance(frame.index, pd.DatetimeIndex)
    required = set(CANONICAL_COLUMNS) | (set() if time_in_index else {time_column})
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ERA5ContextTransferError(f"canonical ERA5 frame is missing {missing}")
    if frame.empty:
        raise ERA5ContextTransferError("canonical ERA5 frame is empty")

    raw_time = frame.index if time_in_index else frame[time_column]
    timestamp = pd.to_datetime(raw_time, errors="raise", utc=True)
    result = pd.DataFrame({"time": pd.array(timestamp)})
    for column in CANONICAL_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=np.float64,
            copy=True,
        )
        values[~np.isfinite(values)] = np.nan
        result[column] = values
    if result["time"].duplicated().any():
        raise ERA5ContextTransferError("canonical ERA5 frame has duplicate timestamps")
    return result.sort_values("time").reset_index(drop=True)


def _last_finite_at_or_before(
    times_ns: np.ndarray,
    values: np.ndarray,
    query: pd.Timestamp,
) -> float:
    stop = int(np.searchsorted(times_ns, query.value, side="right"))
    if stop == 0:
        return np.nan
    finite = np.flatnonzero(np.isfinite(values[:stop]))
    return float(values[finite[-1]]) if len(finite) else np.nan


def _window_summary(times_ns: np.ndarray, values: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(values)
    if not finite.any():
        return {statistic: np.nan for statistic in WINDOW_STATISTICS}
    y = values[finite]
    hours = (times_ns[finite] - times_ns[finite][-1]) / 3_600_000_000_000.0
    slope = np.nan
    if len(y) >= 2 and float(np.ptp(hours)) > 0.0:
        centered = hours - hours.mean()
        denominator = float(np.dot(centered, centered))
        if denominator > 0.0:
            slope = float(np.dot(centered, y - y.mean()) / denominator)
    return {
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "delta": float(y[-1] - y[0]),
        "slope": slope,
    }


def summarize_past_48h(
    frame: pd.DataFrame,
    anchor_time: Any | None = None,
    *,
    time_column: str = "time",
) -> dict[str, float]:
    """Summarize only ``[anchor-48h, anchor]`` using actual elapsed time.

    Current and lag values use a causal last-finite observation at or before the
    requested instant.  Window boundaries and regression slopes use timestamps, not
    row offsets, so regular hourly and irregularly sampled histories share semantics.
    Rows later than the anchor are ignored.
    """

    canonical = canonicalize_era5_hourly(frame, time_column=time_column)
    anchor = canonical["time"].iloc[-1] if anchor_time is None else anchor_time
    return _summarize_canonical_past_48h(canonical, anchor)


def _summarize_canonical_past_48h(
    canonical: pd.DataFrame,
    anchor_time: Any,
) -> dict[str, float]:
    """Summarize an already canonicalized/sorted frame without copying its full history."""

    anchor = _utc_timestamp(anchor_time, label="anchor_time")
    start = anchor - pd.Timedelta(hours=CONTEXT_HOURS)
    history = canonical.loc[canonical["time"].between(start, anchor, inclusive="both")]
    if history.empty:
        raise ERA5ContextTransferError("anchor has no observation in its past 48-hour context")

    # Pandas 3 may retain second/microsecond datetime resolution.  Convert
    # explicitly so comparisons with ``Timestamp.value`` are always nanoseconds.
    times_ns = (
        history["time"]
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )
    arrays: dict[str, np.ndarray] = {
        column: history[column].to_numpy(dtype=np.float64)
        for column in ("hs", "tp", "hmax", "wspd", "airt", "relh", "caph")
    }
    wave_radians = np.deg2rad(history["wvdir"].to_numpy(dtype=np.float64))
    wind_radians = np.deg2rad(history["wdir"].to_numpy(dtype=np.float64))
    arrays["wvdir_sin"] = np.sin(wave_radians)
    arrays["wvdir_cos"] = np.cos(wave_radians)
    arrays["wdir_sin"] = np.sin(wind_radians)
    arrays["wdir_cos"] = np.cos(wind_radians)

    hs = arrays["hs"]
    wspd = arrays["wspd"]
    wave_sin = arrays["wvdir_sin"]
    wave_cos = arrays["wvdir_cos"]
    wind_sin = arrays["wdir_sin"]
    wind_cos = arrays["wdir_cos"]
    with np.errstate(invalid="ignore"):
        arrays["wave_energy"] = hs**2
        alignment = wind_cos * wave_cos + wind_sin * wave_sin
        alignment[~(
            np.isfinite(wave_sin)
            & np.isfinite(wave_cos)
            & np.isfinite(wind_sin)
            & np.isfinite(wind_cos)
        )] = np.nan
        arrays["wind_input_proxy"] = wspd**2 * np.maximum(alignment, 0.0)

    if set(arrays) != set(COMMON_SERIES):
        raise AssertionError("ERA5 common-series construction drifted")
    row: dict[str, float] = {}
    for name in COMMON_SERIES:
        values = arrays[name]
        row[f"{name}_current"] = _last_finite_at_or_before(times_ns, values, anchor)
        for hour in LAG_HOURS:
            row[f"{name}_lag_{hour}h"] = _last_finite_at_or_before(
                times_ns,
                values,
                anchor - pd.Timedelta(hours=hour),
            )
        for hour in WINDOW_HOURS:
            lower_ns = (anchor - pd.Timedelta(hours=hour)).value
            selected = times_ns >= lower_ns
            summary = _window_summary(times_ns[selected], values[selected])
            for statistic in WINDOW_STATISTICS:
                row[f"{name}_{statistic}_{hour}h"] = summary[statistic]
    if tuple(row) != common_feature_columns():
        raise AssertionError("ERA5 common feature ordering drifted")
    return row


# Keep the shorter name parallel to the existing local/KMA feature modules.
summarize_common_history = summarize_past_48h


def _iter_source_groups(
    frame: pd.DataFrame,
    group_column: str | None,
) -> list[tuple[Any | None, pd.DataFrame]]:
    if group_column is None:
        return [(None, frame)]
    reserved = {"anchor_id", "anchor_time", "current_hs", *FUTURE_HS_COLUMNS, *TARGET_COLUMNS}
    if group_column in reserved:
        raise ERA5ContextTransferError("source group column conflicts with anchor metadata")
    if group_column not in frame:
        raise ERA5ContextTransferError(f"source group column {group_column!r} is missing")
    if frame[group_column].isna().any():
        raise ERA5ContextTransferError("source group column contains missing identities")
    return list(frame.groupby(group_column, sort=False, observed=True))


def _canonical_source_groups(
    frame: pd.DataFrame,
    *,
    time_column: str,
    group_column: str | None,
) -> dict[Any | None, pd.DataFrame]:
    return {
        key: canonicalize_era5_hourly(group, time_column=time_column)
        for key, group in _iter_source_groups(frame, group_column)
    }


def _build_source_anchors_from_canonical(
    grouped: Mapping[Any | None, pd.DataFrame],
    *,
    group_column: str | None,
) -> pd.DataFrame:
    """Build source anchors from station frames already canonicalized exactly once."""

    blocks: list[pd.DataFrame] = []
    for group_value, canonical in grouped.items():
        first_anchor = canonical["time"].iloc[0] + pd.Timedelta(hours=CONTEXT_HOURS)
        last_anchor = canonical["time"].iloc[-1] - pd.Timedelta(hours=max(LEADS))
        if first_anchor > last_anchor:
            continue
        candidates = pd.date_range(
            first_anchor,
            last_anchor,
            freq=pd.Timedelta(hours=ANCHOR_SPACING_HOURS),
        )
        hs_by_time = canonical.set_index("time")["hs"]
        current = hs_by_time.reindex(candidates).to_numpy(dtype=np.float64)
        future = np.column_stack(
            [
                hs_by_time.reindex(candidates + pd.Timedelta(hours=lead)).to_numpy(
                    dtype=np.float64
                )
                for lead in LEADS
            ]
        )
        valid = (
            np.isfinite(current)
            & (current >= MINIMUM_CURRENT_HS)
            & (current >= 0.0)
            & np.isfinite(future).all(axis=1)
            & (future >= 0.0).all(axis=1)
        )
        if not valid.any():
            continue
        selected_current = current[valid]
        selected_future = future[valid]
        block = pd.DataFrame(
            {
                "anchor_time": candidates[valid],
                "current_hs": selected_current,
            }
        )
        if group_column is not None:
            block.insert(0, group_column, group_value)
        for position, lead in enumerate(LEADS):
            block[f"future_hs_{lead}h"] = selected_future[:, position]
            block[f"target_log_delta_{lead}h"] = np.log1p(selected_future[:, position]) - np.log1p(
                selected_current
            )
        blocks.append(block)

    metadata_columns = [*([] if group_column is None else [group_column]), "anchor_time", "current_hs"]
    value_columns = [item for pair in zip(FUTURE_HS_COLUMNS, TARGET_COLUMNS, strict=True) for item in pair]
    if not blocks:
        return pd.DataFrame(columns=["anchor_id", *metadata_columns, *value_columns])
    anchors = pd.concat(blocks, ignore_index=True)
    anchors.insert(0, "anchor_id", np.arange(1, len(anchors) + 1, dtype=np.int64))
    return anchors.loc[:, ["anchor_id", *metadata_columns, *value_columns]]


def build_source_anchors(
    frame: pd.DataFrame,
    *,
    time_column: str = "time",
    group_column: str | None = None,
) -> pd.DataFrame:
    """Build fixed six-hour storm anchors with six complete exact-hour targets.

    ``group_column`` may separate independent ERA5 cells, but it is retained only as
    anchor metadata.  It is never accepted by :class:`FixedContextTransferRegressor`.
    """

    grouped = _canonical_source_groups(
        frame,
        time_column=time_column,
        group_column=group_column,
    )
    return _build_source_anchors_from_canonical(grouped, group_column=group_column)


def build_source_cases(
    frame: pd.DataFrame,
    *,
    time_column: str = "time",
    group_column: str | None = None,
) -> ERA5SourceCases:
    """Build complete anchors and their aligned past-only common features."""

    grouped = _canonical_source_groups(
        frame,
        time_column=time_column,
        group_column=group_column,
    )
    anchors = _build_source_anchors_from_canonical(grouped, group_column=group_column)
    rows: list[dict[str, float]] = []
    times_ns = {
        key: canonical["time"]
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
        for key, canonical in grouped.items()
    }
    for _, anchor in anchors.iterrows():
        key = None if group_column is None else anchor[group_column]
        canonical = grouped[key]
        station_times_ns = times_ns[key]
        anchor_time = _utc_timestamp(anchor["anchor_time"], label="anchor_time")
        start_ns = (anchor_time - pd.Timedelta(hours=CONTEXT_HOURS)).value
        stop_ns = anchor_time.value
        left = int(np.searchsorted(station_times_ns, start_ns, side="left"))
        right = int(np.searchsorted(station_times_ns, stop_ns, side="right"))
        history = canonical.iloc[left:right]
        expected_ns = np.arange(
            start_ns,
            stop_ns + 1,
            pd.Timedelta(hours=1).value,
            dtype=np.int64,
        )
        if len(history) != SOURCE_CONTEXT_ROWS or not np.array_equal(
            station_times_ns[left:right],
            expected_ns,
        ):
            raise ERA5ContextTransferError(
                "source anchor lacks its exact 49-row hourly context window"
            )
        rows.append(_summarize_canonical_past_48h(history, anchor_time))
    features = pd.DataFrame(rows, columns=common_feature_columns())
    targets = anchors.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float64)
    return ERA5SourceCases(
        anchors=anchors,
        features=features,
        log_delta_targets=targets,
        current_hs=anchors["current_hs"].to_numpy(dtype=np.float64),
    )


def _source_year_footprint_metadata(
    anchors: pd.DataFrame,
    *,
    years: Sequence[int],
    station_column: str,
    id_column: str,
    time_column: str,
) -> pd.DataFrame:
    """Return metadata whose full context/target footprint stays in its assigned year."""

    required = {id_column, station_column, time_column}
    missing = sorted(required - set(anchors.columns))
    if missing:
        raise ERA5ContextTransferError(f"source split metadata is missing {missing}")
    if anchors[id_column].isna().any() or anchors[id_column].duplicated().any():
        raise ERA5ContextTransferError("source split IDs must be non-missing and unique")
    if anchors[station_column].isna().any():
        raise ERA5ContextTransferError("source split station metadata is missing")
    assigned_years = tuple(int(year) for year in years)
    if (
        not assigned_years
        or len(set(assigned_years)) != len(assigned_years)
        or tuple(sorted(assigned_years)) != assigned_years
    ):
        raise ERA5ContextTransferError("source years must be a non-empty sorted unique sequence")
    metadata = pd.DataFrame(
        {
            "anchor_id": anchors[id_column].to_numpy(copy=True),
            "station": anchors[station_column].astype(str).to_numpy(copy=True),
            "anchor_time": pd.to_datetime(anchors[time_column], errors="raise", utc=True),
        }
    )
    footprint_start = metadata["anchor_time"] - pd.Timedelta(hours=FOOTPRINT_CONTEXT_HOURS)
    footprint_end = metadata["anchor_time"] + pd.Timedelta(hours=FOOTPRINT_TARGET_HOURS)
    blocks: list[pd.DataFrame] = []
    for year in assigned_years:
        year_start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        next_year = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
        eligible = metadata.loc[
            footprint_start.ge(year_start) & footprint_end.lt(next_year)
        ].copy()
        if eligible.empty:
            raise ERA5ContextTransferError(
                f"source year {year} has no case with a fully internal footprint"
            )
        eligible.insert(0, "year", year)
        blocks.append(eligible)
    result = pd.concat(blocks, ignore_index=True).sort_values(
        ["year", "anchor_time", "station", "anchor_id"],
        kind="mergesort",
    )
    if result["anchor_id"].duplicated().any():
        raise AssertionError("one source case was assigned to multiple calendar years")
    return result.reset_index(drop=True).loc[
        :, ["year", "anchor_id", "station", "anchor_time"]
    ]


def select_source_year_training(
    anchors: pd.DataFrame,
    *,
    train_years: Sequence[int] = SOURCE_TRAIN_YEARS,
    station_column: str = "station",
    id_column: str = "anchor_id",
    time_column: str = "anchor_time",
) -> pd.DataFrame:
    """Select all train cases whose complete footprint stays in 2014--2020 by default."""

    return _source_year_footprint_metadata(
        anchors,
        years=train_years,
        station_column=station_column,
        id_column=id_column,
        time_column=time_column,
    )


# Descriptive aliases for consumers that treat the returned metadata as ID assignments.
select_source_training_ids = select_source_year_training
select_source_train_ids = select_source_year_training


def select_source_year_validation(
    anchors: pd.DataFrame,
    *,
    held_years: Sequence[int],
    station_column: str = "station",
    id_column: str = "anchor_id",
    time_column: str = "anchor_time",
) -> pd.DataFrame:
    """Select deterministic, independent validation IDs inside held UTC years.

    An eligible anchor's complete closed ``[anchor-48h, anchor+24h]`` footprint
    must stay inside its held year.  Within each station across the ordered held
    years, the earliest eligible anchor is selected greedily and subsequent anchors
    must be at least 78 hours later.  The result contains metadata only: no current value, feature,
    future value, or target is copied from ``anchors``.
    """

    years = tuple(int(year) for year in held_years)
    eligible_metadata = _source_year_footprint_metadata(
        anchors,
        years=years,
        station_column=station_column,
        id_column=id_column,
        time_column=time_column,
    )
    chosen: list[dict[str, Any]] = []
    gap = pd.Timedelta(hours=VALIDATION_GAP_HOURS)
    # Do not reset the gap at a year boundary: consecutive held years can have
    # eligible edge footprints whose anchors are only 73 hours apart.
    for station, group in eligible_metadata.groupby("station", sort=True, observed=True):
        previous: pd.Timestamp | None = None
        ordered = group.sort_values(["anchor_time", "anchor_id"], kind="mergesort")
        for row in ordered.itertuples(index=False):
            timestamp = pd.Timestamp(row.anchor_time)
            if previous is not None and timestamp - previous < gap:
                continue
            chosen.append(
                {
                    "year": int(row.year),
                    "anchor_id": row.anchor_id,
                    "station": str(station),
                    "anchor_time": timestamp,
                }
            )
            previous = timestamp

    selected = pd.DataFrame(chosen).sort_values(
        ["year", "anchor_time", "station", "anchor_id"],
        kind="mergesort",
    )
    selected = selected.reset_index(drop=True)
    selected.insert(1, "episode_id", np.arange(1, len(selected) + 1, dtype=np.int64))
    missing_years = sorted(set(years) - set(selected["year"].astype(int)))
    if missing_years:
        raise ERA5ContextTransferError(
            f"held years lost all IDs after the global 78-hour gap: {missing_years}"
        )
    if selected["anchor_id"].duplicated().any():
        raise ERA5ContextTransferError("one source anchor was selected for multiple held years")
    for _, group in selected.groupby("station", sort=True, observed=True):
        differences = group.sort_values("anchor_time")["anchor_time"].diff().dropna()
        if not differences.ge(gap).all():
            raise AssertionError("source validation IDs violate the 78-hour station gap")
        if not differences.gt(
            pd.Timedelta(hours=FOOTPRINT_CONTEXT_HOURS + FOOTPRINT_TARGET_HOURS)
        ).all():
            raise AssertionError("source validation footprints overlap")
    return selected.loc[:, ["year", "episode_id", "anchor_id", "station", "anchor_time"]]


# Alternate wording used by split consumers; both names have identical fail-closed behavior.
select_held_year_validation_ids = select_source_year_validation


def common_cached_feature_columns(
    cached_columns: Sequence[str],
    *,
    source_columns: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return the canonical-order exact-name common subset of two feature surfaces."""

    cached = set(cached_columns)
    source = set(common_feature_columns() if source_columns is None else source_columns)
    return tuple(column for column in common_feature_columns() if column in cached and column in source)


def select_common_cached_features(
    frame: pd.DataFrame,
    *,
    source_columns: Sequence[str] | None = None,
    require_all: bool = True,
) -> pd.DataFrame:
    """Select only shared common context values from a local cached feature frame.

    Identity, anchor, and absolute-time columns are discarded because they are not in
    :func:`common_feature_columns`.  With ``source_columns`` supplied, ``require_all``
    requires the complete canonical subset requested by that source surface.
    """

    if frame.columns.duplicated().any():
        raise ERA5ContextTransferError("cached feature frame has duplicate column names")
    requested = tuple(
        column
        for column in common_feature_columns()
        if source_columns is None or column in set(source_columns)
    )
    selected = common_cached_feature_columns(frame.columns, source_columns=source_columns)
    if require_all and selected != requested:
        missing = sorted(set(requested) - set(selected))
        raise ERA5ContextTransferError(f"cached feature frame is missing common columns {missing}")
    if not selected:
        raise ERA5ContextTransferError("cached and source feature surfaces have no common columns")
    return frame.loc[:, selected].copy()


# An explicit local-domain alias makes the intended direction of transfer clear.
select_local_common_features = select_common_cached_features


def _new_catboost_regressor(parameters: Mapping[str, Any]) -> Any:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(**dict(parameters))


class FixedContextTransferRegressor:
    """Fixed six-lead CatBoost pretrain followed by local continuation.

    The public fit and predict methods accept only columns from the frozen 48-hour
    common surface.  Six numeric lead values are generated internally, so callers
    cannot slip station identity, anchor timestamp, or calendar fields into either
    training domain or inference.
    """

    def __init__(self) -> None:
        self._source_model: Any | None = None
        self._model: Any | None = None
        self._feature_columns: tuple[str, ...] = ()

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self._feature_columns

    @property
    def source_model(self) -> Any:
        if self._source_model is None:
            raise ERA5ContextTransferError("source model is not fitted")
        return self._source_model

    @property
    def model(self) -> Any:
        if self._model is None:
            raise ERA5ContextTransferError("context-transfer model is not fitted")
        return self._model

    def clone_pretrained(self) -> FixedContextTransferRegressor:
        """Return an independent facade containing only a deep-cloned source model.

        Fold-specific callers should clone the once-fitted source facade and continue
        that clone locally.  The original source object and other fold clones then
        never share CatBoost state, even if a backend changes its continuation
        implementation in the future.
        """

        if self._source_model is None or not self._feature_columns:
            raise ERA5ContextTransferError("source model is not fitted")
        try:
            source_model = copy.deepcopy(self._source_model)
        except Exception as error:  # pragma: no cover - backend-specific defensive path
            raise ERA5ContextTransferError("source CatBoost model could not be cloned") from error
        if source_model is self._source_model:
            raise ERA5ContextTransferError("source CatBoost clone unexpectedly shares identity")
        clone = type(self)()
        clone._feature_columns = tuple(self._feature_columns)
        clone._source_model = source_model
        clone._model = source_model
        return clone

    def _context_matrix(self, frame: pd.DataFrame, *, fitting_source: bool) -> pd.DataFrame:
        if frame.empty:
            raise ERA5ContextTransferError("context feature frame is empty")
        if frame.columns.duplicated().any():
            raise ERA5ContextTransferError("context feature frame has duplicate columns")
        allowed = set(common_feature_columns())
        unexpected = sorted(set(frame.columns) - allowed)
        if unexpected:
            raise ERA5ContextTransferError(
                f"non-context identity/time/unknown model columns are prohibited: {unexpected}"
            )
        columns = tuple(column for column in common_feature_columns() if column in frame)
        if not columns:
            raise ERA5ContextTransferError("model input has no common context feature")
        if fitting_source:
            self._feature_columns = columns
        elif set(columns) != set(self._feature_columns):
            missing = sorted(set(self._feature_columns) - set(columns))
            extra = sorted(set(columns) - set(self._feature_columns))
            raise ERA5ContextTransferError(
                f"context feature schema changed; missing={missing}, extra={extra}"
            )
        ordered = self._feature_columns if self._feature_columns else columns
        result = frame.loc[:, ordered].apply(pd.to_numeric, errors="raise").astype("float64")
        if np.isinf(result.to_numpy()).any():
            raise ERA5ContextTransferError("context feature frame contains infinity")
        return result.reset_index(drop=True)

    @staticmethod
    def _targets(targets: Any, rows: int) -> np.ndarray:
        if isinstance(targets, pd.DataFrame):
            if set(targets.columns) == set(TARGET_COLUMNS):
                values = targets.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float64)
            else:
                values = targets.to_numpy(dtype=np.float64)
        else:
            values = np.asarray(targets, dtype=np.float64)
        if values.shape != (rows, len(LEADS)):
            raise ERA5ContextTransferError(
                f"targets must have shape ({rows}, {len(LEADS)}), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ERA5ContextTransferError("log-relative targets must be finite")
        return values

    @staticmethod
    def _long_matrix(context: pd.DataFrame) -> pd.DataFrame:
        long = context.loc[context.index.repeat(len(LEADS))].reset_index(drop=True)
        long["lead_h"] = np.tile(np.asarray(LEADS, dtype=np.float64), len(context))
        return long

    def fit_pretrain(self, features: pd.DataFrame, targets: Any) -> FixedContextTransferRegressor:
        """Fit the frozen ERA5 source model without parameter search."""

        context = self._context_matrix(features, fitting_source=True)
        target = self._targets(targets, len(context))
        model = _new_catboost_regressor(SOURCE_CATBOOST_PARAMETERS)
        model.fit(self._long_matrix(context), target.reshape(-1), verbose=False)
        self._source_model = model
        self._model = model
        return self

    def continue_local(
        self,
        features: pd.DataFrame,
        targets: Any,
        *,
        current_hs: Sequence[float] | np.ndarray | None = None,
    ) -> FixedContextTransferRegressor:
        """Continue from the fixed source model on one causal local prefix."""

        if self._source_model is None:
            raise ERA5ContextTransferError("fit_pretrain must run before local continuation")
        context = self._context_matrix(features, fitting_source=False)
        target = self._targets(targets, len(context))
        if current_hs is None:
            if "hs_current" not in context:
                raise ERA5ContextTransferError(
                    "current_hs is required when hs_current is absent from the common subset"
                )
            current = context["hs_current"].to_numpy(dtype=np.float64)
        else:
            current = np.asarray(current_hs, dtype=np.float64)
        if current.shape != (len(context),) or not np.isfinite(current).all():
            raise ERA5ContextTransferError("current_hs must be one finite value per local context")
        case_weight = np.exp(-0.45 * np.maximum(current - MINIMUM_CURRENT_HS, 0.0))
        model = _new_catboost_regressor(LOCAL_CATBOOST_PARAMETERS)
        model.fit(
            self._long_matrix(context),
            target.reshape(-1),
            sample_weight=np.repeat(case_weight, len(LEADS)),
            init_model=self._source_model,
            verbose=False,
        )
        self._model = model
        return self

    def fit_transfer(
        self,
        source_features: pd.DataFrame,
        source_targets: Any,
        local_features: pd.DataFrame,
        local_targets: Any,
        *,
        local_current_hs: Sequence[float] | np.ndarray | None = None,
    ) -> FixedContextTransferRegressor:
        """Run the fixed pretrain and one fixed local continuation in order."""

        return self.fit_pretrain(source_features, source_targets).continue_local(
            local_features,
            local_targets,
            current_hs=local_current_hs,
        )

    def predict_log_delta(self, features: pd.DataFrame) -> np.ndarray:
        """Predict six log-relative changes from past-48h context features only."""

        context = self._context_matrix(features, fitting_source=False)
        prediction = np.asarray(
            self.model.predict(self._long_matrix(context)),
            dtype=np.float64,
        )
        if prediction.shape != (len(context) * len(LEADS),):
            raise ERA5ContextTransferError("CatBoost returned an unexpected prediction shape")
        if not np.isfinite(prediction).all():
            raise ERA5ContextTransferError("CatBoost returned non-finite predictions")
        return prediction.reshape(len(context), len(LEADS))

    def predict_hs(
        self,
        features: pd.DataFrame,
        *,
        current_hs: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert six log-relative outputs to wave height, clipped to 0..30 m."""

        context = self._context_matrix(features, fitting_source=False)
        if current_hs is None:
            if "hs_current" not in context:
                raise ERA5ContextTransferError(
                    "current_hs is required when hs_current is absent from the common subset"
                )
            current = context["hs_current"].to_numpy(dtype=np.float64)
        else:
            current = np.asarray(current_hs, dtype=np.float64)
        if current.shape != (len(context),) or not np.isfinite(current).all() or (current < 0).any():
            raise ERA5ContextTransferError("current_hs must be one finite non-negative value per case")
        log_delta = self.predict_log_delta(context)
        forecast = np.expm1(np.log1p(current)[:, None] + log_delta)
        return np.clip(forecast, 0.0, 30.0)


# A concise alias for callers that name the facade after the backend.
FixedCatBoostContextTransfer = FixedContextTransferRegressor
