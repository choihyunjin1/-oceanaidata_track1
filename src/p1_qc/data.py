"""Immutable dataset loading and gap-aware time-series utilities."""

from __future__ import annotations

import re
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

KEY_COLUMNS = ("station", "year", "layer", "time")
BASE_COLUMNS = KEY_COLUMNS + ("temp", "psal", "depth")
TRAIN_COLUMNS = BASE_COLUMNS + ("label", "anomaly_type")
ANOMALY_TYPES = ("spike", "noise", "flatline", "offset", "drift")


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _infer_kind(path: Path, columns: Sequence[str] | None = None) -> str:
    name = path.name.lower()
    if "train" in name:
        return "train"
    if "test" in name:
        return "test"
    if "submission" in name:
        return "submission"
    if columns is not None and "label" in columns:
        return "train" if "temp" in columns else "submission"
    return "test"


def load_dataset(
    path: str | Path,
    *,
    kind: Literal["train", "test", "submission", "auto"] | None = None,
    audit: bool = True,
    strict: bool = True,
    read_csv_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Read a source CSV without modifying it and optionally run an audit.

    A size/mtime signature is checked before and after the read.  The SHA-256
    and audit report are attached to ``DataFrame.attrs`` for experiment logs.
    The returned frame is independent from the source bytes; transformations
    elsewhere in this module always return another copy.
    """

    source = Path(path).expanduser().resolve(strict=True)
    before = source.stat()
    kwargs = {"low_memory": False}
    if read_csv_kwargs:
        kwargs.update(read_csv_kwargs)
    frame = pd.read_csv(source, **kwargs)
    after = source.stat()
    before_signature = (before.st_size, before.st_mtime_ns)
    after_signature = (after.st_size, after.st_mtime_ns)
    if before_signature != after_signature:
        raise RuntimeError(f"source file changed while it was being read: {source}")

    resolved_kind = _infer_kind(source, frame.columns) if kind in {None, "auto"} else kind
    frame.attrs.update(
        {
            "source_path": str(source),
            "source_size": before.st_size,
            "source_mtime_ns": before.st_mtime_ns,
            "source_sha256": sha256_file(source),
            "dataset_kind": resolved_kind,
        }
    )
    if audit:
        from .audit import audit_frame

        report = audit_frame(frame, kind=resolved_kind)
        frame.attrs["audit_report"] = report.to_dict()
        if strict:
            report.raise_for_errors()
    return frame


def load_train_test(
    data_dir: str | Path,
    *,
    audit: bool = True,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = Path(data_dir).expanduser().resolve(strict=True)
    return (
        load_dataset(directory / "train.csv", kind="train", audit=audit, strict=strict),
        load_dataset(directory / "test.csv", kind="test", audit=audit, strict=strict),
    )


def parse_anomaly_types(
    values: pd.Series,
    *,
    known_types: Sequence[str] = ANOMALY_TYPES,
    strict: bool = False,
) -> pd.DataFrame:
    """Parse ``+``-separated anomaly strings into deduplicated membership.

    Repeated tokens such as ``flatline+flatline`` intentionally produce one
    ``True`` membership.  Empty/NA strings produce an all-false row.
    """

    text = values.astype("string").fillna("").str.strip()
    if strict:
        tokens = text.str.split("+", regex=False).explode()
        observed = set(tokens[tokens.ne("")].astype(str).unique())
        unknown = sorted(observed.difference(known_types))
        if unknown:
            raise ValueError(f"unknown anomaly types: {unknown}")
    membership: dict[str, pd.Series] = {}
    for anomaly in known_types:
        token = re.escape(anomaly)
        membership[anomaly] = text.str.contains(rf"(?:^|\+){token}(?:\+|$)", regex=True, na=False)
    return pd.DataFrame(membership, index=values.index, dtype=bool)


def add_anomaly_membership(
    frame: pd.DataFrame,
    *,
    anomaly_column: str = "anomaly_type",
    prefix: str = "anomaly_",
    known_types: Sequence[str] = ANOMALY_TYPES,
    strict: bool = False,
) -> pd.DataFrame:
    if anomaly_column not in frame:
        raise KeyError(f"missing anomaly column: {anomaly_column}")
    result = frame.copy(deep=True)
    membership = parse_anomaly_types(result[anomaly_column], known_types=known_types, strict=strict)
    for column in membership:
        result[f"{prefix}{column}"] = membership[column]
    return result


def segment_timeseries(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("station", "layer"),
    time_column: str = "time",
    cadence_minutes: int = 10,
    parsed_time_column: str = "parsed_time",
    segment_column: str = "segment_id",
) -> pd.DataFrame:
    """Add exact-cadence segment metadata while preserving input row order.

    A new segment begins at each group start or whenever the observed delta is
    not exactly ``cadence_minutes``.  No interpolation is performed and a gap
    is never bridged.  Segment IDs are deterministic for the supplied frame.
    """

    required = set(group_columns) | {time_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"missing segmentation columns: {missing}")
    if cadence_minutes <= 0:
        raise ValueError("cadence_minutes must be positive")
    generated = {
        parsed_time_column,
        segment_column,
        "gap_minutes",
        "is_contiguous",
        "position_in_segment",
        "segment_size",
    }
    collisions = generated.intersection(frame.columns)
    if collisions:
        raise ValueError(f"generated columns already exist: {sorted(collisions)}")

    result = frame.copy(deep=True)
    original_index = result.index.copy()
    result["__p1_row_position"] = np.arange(len(result), dtype=np.int64)
    result[parsed_time_column] = pd.to_datetime(result[time_column], errors="coerce", utc=True)
    if result[parsed_time_column].isna().any():
        count = int(result[parsed_time_column].isna().sum())
        raise ValueError(f"{count} timestamps could not be parsed")

    sort_columns = [*group_columns, parsed_time_column, "__p1_row_position"]
    ordered = result.sort_values(sort_columns, kind="mergesort").copy()
    grouped = ordered.groupby(list(group_columns), sort=False, observed=True)
    delta = grouped[parsed_time_column].diff().dt.total_seconds().div(60.0)
    ordered["gap_minutes"] = delta
    ordered["is_contiguous"] = delta.eq(float(cadence_minutes))
    local_segment = (
        (~ordered["is_contiguous"])
        .groupby([ordered[column] for column in group_columns], sort=False)
        .cumsum()
    )
    ordered[segment_column] = ordered.groupby(
        [*group_columns, local_segment], sort=False, observed=True
    ).ngroup()
    segment_group = ordered.groupby(segment_column, sort=False, observed=True)
    ordered["position_in_segment"] = segment_group.cumcount().astype(np.int64)
    ordered["segment_size"] = segment_group[segment_column].transform("size").astype(np.int64)

    restored = ordered.sort_values("__p1_row_position", kind="mergesort")
    restored = restored.drop(columns="__p1_row_position")
    restored.index = original_index
    restored.attrs = dict(frame.attrs)
    return restored


def prepare_timeseries(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("station", "layer"),
    time_column: str = "time",
    cadence_minutes: int = 10,
) -> pd.DataFrame:
    """Return a gap-annotated copy sorted by group and timestamp."""

    segmented = segment_timeseries(
        frame,
        group_columns=group_columns,
        time_column=time_column,
        cadence_minutes=cadence_minutes,
    )
    return segmented.sort_values([*group_columns, "parsed_time"], kind="mergesort").copy()


def add_depth_regime(
    frame: pd.DataFrame,
    *,
    width_m: float = 2.5,
    station_column: str = "station",
    year_column: str = "year",
    layer_column: str = "layer",
    depth_column: str = "depth",
) -> pd.DataFrame:
    """Attach deployment-aware nominal-depth regimes.

    Layer ordinals are not stable sensor identities across years (notably the
    S-ORS 40/50 m deployment change).  Medians are therefore estimated within
    station-year-layer and binned by nominal depth.  An all-missing deployment
    receives an explicit station/layer fallback category.
    """

    required = {station_column, year_column, layer_column, depth_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"missing depth-regime columns: {missing}")
    if width_m <= 0:
        raise ValueError("width_m must be positive")
    result = frame.copy(deep=True)
    group_columns = [station_column, year_column, layer_column]
    nominal = result.groupby(group_columns, observed=True, sort=False)[depth_column].transform(
        "median"
    )
    rounded = (nominal / width_m).round() * width_m
    result["nominal_depth_m"] = rounded.astype(float)

    station = result[station_column].astype("string")
    layer = result[layer_column].astype("string")
    formatted = rounded.map(lambda value: f"d{value:06.1f}" if pd.notna(value) else "")
    regime = station + "|" + formatted.astype("string")
    missing_depth = rounded.isna()
    regime = regime.mask(missing_depth, station + "|unknown|l" + layer)
    result["depth_regime"] = regime.astype("string")
    result.attrs = dict(frame.attrs)
    return result


__all__ = [
    "ANOMALY_TYPES",
    "BASE_COLUMNS",
    "KEY_COLUMNS",
    "TRAIN_COLUMNS",
    "add_anomaly_membership",
    "add_depth_regime",
    "load_dataset",
    "load_train_test",
    "parse_anomaly_types",
    "prepare_timeseries",
    "segment_timeseries",
    "sha256_file",
]
