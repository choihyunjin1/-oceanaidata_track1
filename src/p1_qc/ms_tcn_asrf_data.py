"""Leakage-safe sequence data utilities for the P1 MS-TCN/ASRF challenger.

This module deliberately has no file-system entry points.  A runner must pass
already aligned arrays (or ``Series`` objects) and explicit fit/evaluation row
IDs.  The contracts here keep segmentation, scaling, categorical fitting,
target construction, and overlap-add reconstruction independently testable.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

ANOMALY_TYPES = ("spike", "noise", "flatline", "offset", "drift")
DEFAULT_CADENCE_MINUTES = 10
DEFAULT_WINDOW_SIZE = 2048
DEFAULT_WINDOW_STRIDE = 512


def _one_dimensional(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional aligned array")
    return result


def _aligned_length(*named_values: tuple[str, Sequence[Any] | np.ndarray]) -> int:
    lengths = {name: len(_one_dimensional(values, name=name)) for name, values in named_values}
    if not lengths:
        raise ValueError("at least one aligned array is required")
    if len(set(lengths.values())) != 1:
        raise ValueError(f"aligned array lengths differ: {lengths}")
    size = next(iter(lengths.values()))
    if size == 0:
        raise ValueError("aligned arrays cannot be empty")
    return int(size)


def _row_ids(
    values: Sequence[int] | np.ndarray,
    *,
    size: int,
    role: str,
    allow_empty: bool = False,
) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or (not allow_empty and len(result) == 0):
        suffix = "" if allow_empty else " non-empty"
        raise ValueError(f"{role} row IDs must be a{suffix} one-dimensional vector")
    if not np.issubdtype(result.dtype, np.integer):
        raise TypeError(f"{role} row IDs must be integers")
    result = result.astype(np.int64, copy=False)
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{role} row IDs must be unique")
    if len(result) and (int(result.min()) < 0 or int(result.max()) >= size):
        raise IndexError(f"{role} row IDs are outside the aligned arrays")
    return result


def _token(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "<MISSING>"
    return str(value)


def _ids_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class ExactSegment:
    """One exact-cadence station/year/layer run in chronological order."""

    segment_id: int
    key: tuple[str, str, str]
    row_ids: np.ndarray
    time_ns: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.row_ids))


@dataclass(frozen=True)
class SegmentLayout:
    """Exact ten-minute segments plus row-aligned gap metadata."""

    segments: tuple[ExactSegment, ...]
    segment_id_by_row: np.ndarray
    local_rank_by_row: np.ndarray
    gap_by_row: np.ndarray
    time_ns_by_row: np.ndarray
    cadence_ns: int

    @property
    def n_rows(self) -> int:
        return int(len(self.segment_id_by_row))

    @classmethod
    def from_aligned(
        cls,
        station: Sequence[Any] | np.ndarray,
        year: Sequence[Any] | np.ndarray,
        layer: Sequence[Any] | np.ndarray,
        time: Sequence[Any] | np.ndarray,
        *,
        cadence_minutes: int = DEFAULT_CADENCE_MINUTES,
    ) -> SegmentLayout:
        """Build station/year/layer runs separated by every non-exact cadence.

        Input row order need not be chronological.  ``row_ids`` stored in each
        segment point back to the original aligned order, while each segment is
        internally stable-sorted by timestamp.  Duplicate timestamps inside a
        station/year/layer key are rejected rather than silently coalesced.
        """

        n_rows = _aligned_length(
            ("station", station), ("year", year), ("layer", layer), ("time", time)
        )
        if cadence_minutes < 1:
            raise ValueError("cadence_minutes must be positive")
        stations = _one_dimensional(station, name="station")
        years = _one_dimensional(year, name="year")
        layers = _one_dimensional(layer, name="layer")
        parsed = pd.to_datetime(
            _one_dimensional(time, name="time"), errors="raise", utc=True, format="mixed"
        )
        time_ns = parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        cadence_ns = int(pd.Timedelta(minutes=cadence_minutes).value)
        keys = np.asarray(
            [
                (_token(stations[index]), _token(years[index]), _token(layers[index]))
                for index in range(n_rows)
            ],
            dtype=object,
        )

        segment_id_by_row = np.full(n_rows, -1, dtype=np.int64)
        local_rank_by_row = np.full(n_rows, -1, dtype=np.int64)
        gap_by_row = np.zeros(n_rows, dtype=np.float32)
        segments: list[ExactSegment] = []
        unique_keys = sorted({tuple(item) for item in keys.tolist()})
        for key in unique_keys:
            members = np.asarray(
                [index for index, current in enumerate(keys.tolist()) if tuple(current) == key],
                dtype=np.int64,
            )
            order = np.lexsort((members, time_ns[members]))
            members = members[order]
            key_times = time_ns[members]
            if len(np.unique(key_times)) != len(key_times):
                raise ValueError(f"duplicate timestamp inside station/year/layer key {key}")
            breaks = np.flatnonzero(np.diff(key_times) != cadence_ns) + 1
            starts = np.concatenate([np.asarray([0]), breaks])
            stops = np.concatenate([breaks, np.asarray([len(members)])])
            for run_number, (start, stop) in enumerate(zip(starts, stops, strict=True)):
                row_ids = members[int(start) : int(stop)].copy()
                run_times = key_times[int(start) : int(stop)].copy()
                segment_id = len(segments)
                segment_id_by_row[row_ids] = segment_id
                local_rank_by_row[row_ids] = np.arange(len(row_ids), dtype=np.int64)
                if run_number > 0:
                    gap_by_row[int(row_ids[0])] = 1.0
                segments.append(ExactSegment(segment_id, key, row_ids, run_times))

        if np.any(segment_id_by_row < 0) or np.any(local_rank_by_row < 0):
            raise AssertionError("some aligned rows were not assigned to an exact segment")
        return cls(
            tuple(segments),
            segment_id_by_row,
            local_rank_by_row,
            gap_by_row,
            time_ns,
            cadence_ns,
        )


@dataclass(frozen=True)
class WindowIndex:
    """A right-padded slice of one exact segment."""

    segment_id: int
    start: int
    window_size: int
    row_ids: np.ndarray

    @property
    def valid_length(self) -> int:
        return int(len(self.row_ids))


def build_window_index(
    layout: SegmentLayout,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_WINDOW_STRIDE,
) -> tuple[WindowIndex, ...]:
    """Enumerate fixed-size, right-padded windows over every exact segment."""

    if window_size < 1 or stride < 1 or stride > window_size:
        raise ValueError("window_size and stride must satisfy 1 <= stride <= window_size")
    windows: list[WindowIndex] = []
    for segment in layout.segments:
        for start in range(0, segment.size, stride):
            rows = segment.row_ids[start : min(start + window_size, segment.size)].copy()
            windows.append(WindowIndex(segment.segment_id, int(start), int(window_size), rows))
    return tuple(windows)


def _window_digest(window: WindowIndex, seed: int) -> bytes:
    payload = (
        f"{int(seed)}|{window.segment_id}|{window.start}|{window.window_size}|{window.valid_length}"
    ).encode("ascii")
    return hashlib.blake2b(payload, digest_size=16).digest()


def select_training_windows(
    windows: Sequence[WindowIndex],
    labels: Sequence[int] | np.ndarray,
    *,
    negative_ratio: float = 1.0,
    min_negative_windows: int = 1,
    seed: int = 20260827,
) -> tuple[WindowIndex, ...]:
    """Keep every event-containing window and a deterministic normal subset."""

    binary = _one_dimensional(labels, name="labels")
    if not np.isin(binary, [0, 1]).all():
        raise ValueError("labels must contain only binary 0/1 values")
    if not np.isfinite(negative_ratio) or negative_ratio < 0.0:
        raise ValueError("negative_ratio must be a finite non-negative value")
    if min_negative_windows < 0:
        raise ValueError("min_negative_windows cannot be negative")
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    for index, window in enumerate(windows):
        if len(window.row_ids) == 0:
            raise ValueError("a window cannot have zero valid rows")
        if int(window.row_ids.min()) < 0 or int(window.row_ids.max()) >= len(binary):
            raise IndexError("window row IDs are outside the label vector")
        destination = positive_indices if np.any(binary[window.row_ids] == 1) else negative_indices
        destination.append(index)
    requested = max(min_negative_windows, int(math.ceil(len(positive_indices) * negative_ratio)))
    requested = min(requested, len(negative_indices))
    ranked_negative = sorted(
        negative_indices,
        key=lambda index: (_window_digest(windows[index], seed), index),
    )
    keep = set(positive_indices)
    keep.update(ranked_negative[:requested])
    return tuple(window for index, window in enumerate(windows) if index in keep)


def materialize_windows(
    row_features: np.ndarray,
    windows: Sequence[WindowIndex],
    *,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather aligned row features into right-padded windows and valid masks."""

    values = np.asarray(row_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("row_features must be a non-empty two-dimensional matrix")
    if not windows:
        raise ValueError("windows cannot be empty")
    window_size = int(windows[0].window_size)
    if any(window.window_size != window_size for window in windows):
        raise ValueError("all windows must use the same window size")
    result = np.full(
        (len(windows), window_size, values.shape[1]), float(pad_value), dtype=np.float32
    )
    valid = np.zeros((len(windows), window_size), dtype=np.float32)
    for index, window in enumerate(windows):
        length = window.valid_length
        if length > window_size:
            raise AssertionError("window valid length exceeds its padded size")
        if int(window.row_ids.min()) < 0 or int(window.row_ids.max()) >= len(values):
            raise IndexError("window row IDs are outside row_features")
        result[index, :length] = values[window.row_ids]
        valid[index, :length] = 1.0
    return result, valid


@dataclass(frozen=True)
class EncodedRows:
    """Dense numeric channels and compact categorical embedding codes."""

    dense: np.ndarray
    station_code: np.ndarray
    layer_code: np.ndarray
    depth_regime_code: np.ndarray
    dense_feature_names: tuple[str, ...]


def _fit_vocab(values: np.ndarray, fit_ids: np.ndarray) -> tuple[str, ...]:
    return tuple(sorted({_token(values[index]) for index in fit_ids}))


def _encode_vocab(values: np.ndarray, vocab: tuple[str, ...]) -> np.ndarray:
    mapping = {value: index + 1 for index, value in enumerate(vocab)}
    return np.asarray([mapping.get(_token(value), 0) for value in values], dtype=np.int64)


def _depth_tokens(depth: np.ndarray, thresholds: tuple[float, float] | None) -> np.ndarray:
    numeric = np.asarray(depth, dtype=np.float64)
    if numeric.ndim != 1:
        raise ValueError("depth must be a one-dimensional aligned array")
    result = np.full(len(numeric), "missing", dtype=object)
    finite = np.isfinite(numeric)
    if thresholds is None:
        result[finite] = "finite"
        return result
    lower, upper = thresholds
    result[finite & (numeric <= lower)] = "shallow"
    result[finite & (numeric > lower) & (numeric <= upper)] = "mid"
    result[finite & (numeric > upper)] = "deep"
    return result


@dataclass(frozen=True)
class RobustRowEncoder:
    """Median/IQR numeric scaler and train-ID-only categorical vocabularies."""

    center: np.ndarray
    scale: np.ndarray
    station_vocab: tuple[str, ...]
    layer_vocab: tuple[str, ...]
    depth_regime_vocab: tuple[str, ...]
    depth_thresholds: tuple[float, float] | None
    uses_supplied_depth_regime: bool
    numeric_names: tuple[str, ...]
    fit_ids_sha256: str

    @classmethod
    def fit(
        cls,
        numeric_values: np.ndarray,
        station: Sequence[Any] | np.ndarray,
        layer: Sequence[Any] | np.ndarray,
        train_row_ids: Sequence[int] | np.ndarray,
        *,
        depth: Sequence[float] | np.ndarray | None = None,
        depth_regime: Sequence[Any] | np.ndarray | None = None,
        forbidden_row_ids: Sequence[int] | np.ndarray | None = None,
        numeric_names: Sequence[str] | None = None,
    ) -> RobustRowEncoder:
        """Fit every learned preprocessing statistic on ``train_row_ids`` only."""

        matrix = np.asarray(numeric_values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("numeric_values must be a non-empty two-dimensional matrix")
        n_rows = int(matrix.shape[0])
        stations = _one_dimensional(station, name="station")
        layers = _one_dimensional(layer, name="layer")
        if len(stations) != n_rows or len(layers) != n_rows:
            raise ValueError("station/layer arrays must align with numeric_values")
        fit_ids = _row_ids(train_row_ids, size=n_rows, role="training")
        if forbidden_row_ids is not None:
            forbidden = _row_ids(forbidden_row_ids, size=n_rows, role="forbidden", allow_empty=True)
            if np.intersect1d(fit_ids, forbidden).size:
                raise PermissionError("training row IDs overlap forbidden row IDs")
        selected = matrix[fit_ids]
        center = np.zeros(matrix.shape[1], dtype=np.float64)
        scale = np.ones(matrix.shape[1], dtype=np.float64)
        for column in range(matrix.shape[1]):
            finite = selected[:, column][np.isfinite(selected[:, column])]
            if not len(finite):
                continue
            center[column] = float(np.median(finite))
            q25, q75 = np.quantile(finite, [0.25, 0.75])
            iqr = float(q75 - q25)
            scale[column] = iqr if np.isfinite(iqr) and iqr > 1e-6 else 1.0

        if depth_regime is not None and depth is not None:
            raise ValueError("supply either depth or depth_regime, not both")
        uses_supplied = depth_regime is not None
        thresholds: tuple[float, float] | None = None
        if uses_supplied:
            regimes = _one_dimensional(depth_regime, name="depth_regime")
            if len(regimes) != n_rows:
                raise ValueError("depth_regime must align with numeric_values")
        else:
            if depth is None:
                raise ValueError("depth or depth_regime is required")
            depths = np.asarray(_one_dimensional(depth, name="depth"), dtype=np.float64)
            if len(depths) != n_rows:
                raise ValueError("depth must align with numeric_values")
            fit_depth = depths[fit_ids]
            fit_depth = fit_depth[np.isfinite(fit_depth)]
            if len(fit_depth):
                q33, q67 = np.quantile(fit_depth, [1.0 / 3.0, 2.0 / 3.0])
                thresholds = (float(q33), float(q67))
            regimes = _depth_tokens(depths, thresholds)

        if numeric_names is None:
            names = tuple(f"numeric_{column}" for column in range(matrix.shape[1]))
        else:
            names = tuple(str(value) for value in numeric_names)
            if len(names) != matrix.shape[1] or len(set(names)) != len(names):
                raise ValueError("numeric_names must be unique and match the feature count")
        return cls(
            center.astype(np.float32),
            scale.astype(np.float32),
            _fit_vocab(stations, fit_ids),
            _fit_vocab(layers, fit_ids),
            _fit_vocab(np.asarray(regimes), fit_ids),
            thresholds,
            uses_supplied,
            names,
            _ids_sha256(fit_ids),
        )

    @property
    def numeric_feature_count(self) -> int:
        return int(len(self.center))

    def transform(
        self,
        numeric_values: np.ndarray,
        station: Sequence[Any] | np.ndarray,
        layer: Sequence[Any] | np.ndarray,
        *,
        depth: Sequence[float] | np.ndarray | None = None,
        depth_regime: Sequence[Any] | np.ndarray | None = None,
        row_valid: Sequence[bool] | np.ndarray | None = None,
        gap: Sequence[bool] | np.ndarray | None = None,
        one_hot_categories: bool = False,
    ) -> EncodedRows:
        """Transform aligned rows without changing any fitted state."""

        matrix = np.asarray(numeric_values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.numeric_feature_count:
            raise ValueError("numeric_values feature count differs from the fitted encoder")
        n_rows = int(matrix.shape[0])
        stations = _one_dimensional(station, name="station")
        layers = _one_dimensional(layer, name="layer")
        if len(stations) != n_rows or len(layers) != n_rows:
            raise ValueError("station/layer arrays must align with numeric_values")
        if self.uses_supplied_depth_regime:
            if depth_regime is None or depth is not None:
                raise ValueError("this encoder requires supplied depth_regime and no depth")
            regimes = _one_dimensional(depth_regime, name="depth_regime")
        else:
            if depth is None or depth_regime is not None:
                raise ValueError("this encoder requires numeric depth and no depth_regime")
            regimes = _depth_tokens(np.asarray(depth, dtype=np.float64), self.depth_thresholds)
        if len(regimes) != n_rows:
            raise ValueError("depth/depth_regime must align with numeric_values")

        finite = np.isfinite(matrix)
        scaled = np.where(finite, (matrix - self.center) / self.scale, 0.0)
        scaled = np.clip(scaled, -20.0, 20.0).astype(np.float32, copy=False)
        missing = (~finite).astype(np.float32)
        if row_valid is None:
            valid_channel = np.ones(n_rows, dtype=np.float32)
        else:
            valid_channel = np.asarray(
                _one_dimensional(row_valid, name="row_valid"), dtype=np.float32
            )
            if len(valid_channel) != n_rows or not np.isin(valid_channel, [0.0, 1.0]).all():
                raise ValueError("row_valid must be an aligned boolean/binary array")
        if gap is None:
            gap_channel = np.zeros(n_rows, dtype=np.float32)
        else:
            gap_channel = np.asarray(_one_dimensional(gap, name="gap"), dtype=np.float32)
            if len(gap_channel) != n_rows or not np.isin(gap_channel, [0.0, 1.0]).all():
                raise ValueError("gap must be an aligned boolean/binary array")
        station_code = _encode_vocab(stations, self.station_vocab)
        layer_code = _encode_vocab(layers, self.layer_vocab)
        regime_code = _encode_vocab(np.asarray(regimes), self.depth_regime_vocab)
        dense_parts = [scaled, missing, valid_channel[:, None], gap_channel[:, None]]
        feature_names = (
            tuple(f"{name}_scaled" for name in self.numeric_names)
            + tuple(f"{name}_missing" for name in self.numeric_names)
            + ("row_valid", "gap")
        )
        if one_hot_categories:
            code_specs = (
                ("station", station_code, len(self.station_vocab)),
                ("layer", layer_code, len(self.layer_vocab)),
                ("depth_regime", regime_code, len(self.depth_regime_vocab)),
            )
            for prefix, codes, width in code_specs:
                one_hot = np.zeros((n_rows, width), dtype=np.float32)
                known = codes > 0
                known_rows = np.flatnonzero(known)
                one_hot[known_rows, codes[known] - 1] = 1.0
                dense_parts.append(one_hot)
                feature_names += tuple(f"{prefix}_{index}" for index in range(width))
        dense = np.concatenate(dense_parts, axis=1).astype(np.float32, copy=False)
        if not np.isfinite(dense).all():
            raise AssertionError("encoded dense features contain non-finite values")
        return EncodedRows(dense, station_code, layer_code, regime_code, feature_names)


@dataclass(frozen=True)
class ASRFTargets:
    """Row, boundary, and five-type supervision aligned to original rows."""

    row_label: np.ndarray
    start_boundary: np.ndarray
    end_boundary: np.ndarray
    anomaly_type: np.ndarray
    anomaly_type_names: tuple[str, ...] = ANOMALY_TYPES


def _parsed_types(value: Any) -> tuple[str, ...]:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ()
    text = str(value).strip().lower()
    if not text or text in {"none", "normal", "nan"}:
        return ()
    tokens = tuple(token.strip() for token in text.split("+") if token.strip())
    unknown = sorted(set(tokens).difference(ANOMALY_TYPES))
    if unknown:
        raise ValueError(f"unknown anomaly type token(s): {unknown}")
    return tokens


def build_asrf_targets(
    labels: Sequence[int] | np.ndarray,
    anomaly_type: Sequence[Any] | np.ndarray,
    layout: SegmentLayout,
    *,
    sigma_rows: float = 3.0,
) -> ASRFTargets:
    """Build binary, Gaussian boundary, and plus-delimited type targets."""

    binary = _one_dimensional(labels, name="labels")
    types = _one_dimensional(anomaly_type, name="anomaly_type")
    if len(binary) != layout.n_rows or len(types) != layout.n_rows:
        raise ValueError("labels/anomaly_type must align with the segment layout")
    if not np.isin(binary, [0, 1]).all():
        raise ValueError("labels must contain only binary 0/1 values")
    if not np.isfinite(sigma_rows) or sigma_rows <= 0.0:
        raise ValueError("sigma_rows must be finite and positive")
    row_label = binary.astype(np.float32, copy=False)
    start = np.zeros(layout.n_rows, dtype=np.float32)
    end = np.zeros(layout.n_rows, dtype=np.float32)
    multi_hot = np.zeros((layout.n_rows, len(ANOMALY_TYPES)), dtype=np.float32)
    type_to_column = {value: index for index, value in enumerate(ANOMALY_TYPES)}
    for row_id, value in enumerate(types):
        parsed = _parsed_types(value)
        if not binary[row_id] and parsed:
            raise ValueError("normal rows cannot carry an anomaly type target")
        for token in parsed:
            multi_hot[row_id, type_to_column[token]] = 1.0

    for segment in layout.segments:
        segment_labels = binary[segment.row_ids].astype(np.int8, copy=False)
        positions = np.arange(segment.size, dtype=np.float64)
        cursor = 0
        while cursor < segment.size:
            if segment_labels[cursor] == 0:
                cursor += 1
                continue
            stop = cursor + 1
            while stop < segment.size and segment_labels[stop] == 1:
                stop += 1
            start_kernel = np.exp(-0.5 * ((positions - cursor) / sigma_rows) ** 2)
            end_kernel = np.exp(-0.5 * ((positions - (stop - 1)) / sigma_rows) ** 2)
            rows = segment.row_ids
            start[rows] = np.maximum(start[rows], start_kernel.astype(np.float32))
            end[rows] = np.maximum(end[rows], end_kernel.astype(np.float32))
            cursor = stop
    return ASRFTargets(row_label, start, end, multi_hot)


def _center_weights(window_size: int) -> np.ndarray:
    positions = np.arange(window_size, dtype=np.float64)
    center = (window_size - 1) / 2.0
    denominator = center + 1.0
    weights = 1.0 - np.abs(positions - center) / denominator
    return np.maximum(weights, np.finfo(np.float64).eps)


def stitch_center_weighted(
    window_predictions: np.ndarray,
    windows: Sequence[WindowIndex],
    *,
    n_rows: int,
    require_row_ids: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Center-weighted overlap-add into original aligned row order."""

    predictions = np.asarray(window_predictions)
    if predictions.ndim < 2 or predictions.shape[0] != len(windows):
        raise ValueError("window_predictions must align with the window index")
    if n_rows < 1 or not windows:
        raise ValueError("n_rows and windows must be non-empty")
    window_size = int(windows[0].window_size)
    if predictions.shape[1] != window_size:
        raise ValueError("window prediction length differs from window_size")
    if any(window.window_size != window_size for window in windows):
        raise ValueError("all windows must use the same window size")
    tail_shape = predictions.shape[2:]
    accumulator = np.zeros((n_rows, *tail_shape), dtype=np.float64)
    denominator = np.zeros(n_rows, dtype=np.float64)
    weights = _center_weights(window_size)
    for index, window in enumerate(windows):
        length = window.valid_length
        rows = window.row_ids
        if int(rows.min()) < 0 or int(rows.max()) >= n_rows:
            raise IndexError("window row IDs are outside the requested output")
        current_weights = weights[:length]
        reshape = (length,) + (1,) * len(tail_shape)
        accumulator[rows] += predictions[index, :length] * current_weights.reshape(reshape)
        denominator[rows] += current_weights
    if require_row_ids is None:
        required = np.unique(np.concatenate([window.row_ids for window in windows]))
    else:
        required = _row_ids(require_row_ids, size=n_rows, role="required", allow_empty=True)
    if len(required) and np.any(denominator[required] <= 0.0):
        raise AssertionError("one or more required rows received no window prediction")
    output = np.full((n_rows, *tail_shape), np.nan, dtype=np.float64)
    covered = denominator > 0.0
    reshape = (int(covered.sum()),) + (1,) * len(tail_shape)
    output[covered] = accumulator[covered] / denominator[covered].reshape(reshape)
    return output.astype(
        predictions.dtype if np.issubdtype(predictions.dtype, np.floating) else np.float32
    )


def assert_disjoint_and_time_purged(
    train_row_ids: Sequence[int] | np.ndarray,
    validation_row_ids: Sequence[int] | np.ndarray,
    station: Sequence[Any] | np.ndarray,
    year: Sequence[Any] | np.ndarray,
    layer: Sequence[Any] | np.ndarray,
    time: Sequence[Any] | np.ndarray,
    *,
    purge: str | pd.Timedelta | np.timedelta64,
) -> None:
    """Assert ID disjointness and minimum same-key temporal separation."""

    n_rows = _aligned_length(("station", station), ("year", year), ("layer", layer), ("time", time))
    train_ids = _row_ids(train_row_ids, size=n_rows, role="training")
    validation_ids = _row_ids(validation_row_ids, size=n_rows, role="validation")
    overlap = np.intersect1d(train_ids, validation_ids)
    if overlap.size:
        raise AssertionError(f"training/validation row IDs overlap ({len(overlap)} row(s))")
    purge_delta = pd.Timedelta(purge)
    if purge_delta <= pd.Timedelta(0):
        raise ValueError("purge must be a positive duration")
    purge_ns = int(purge_delta.value)
    stations = _one_dimensional(station, name="station")
    years = _one_dimensional(year, name="year")
    layers = _one_dimensional(layer, name="layer")
    parsed = pd.to_datetime(
        _one_dimensional(time, name="time"), errors="raise", utc=True, format="mixed"
    )
    time_ns = parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    keys = np.asarray(
        [
            (_token(stations[index]), _token(years[index]), _token(layers[index]))
            for index in range(n_rows)
        ],
        dtype=object,
    )
    train_by_key: dict[tuple[str, str, str], list[int]] = {}
    validation_by_key: dict[tuple[str, str, str], list[int]] = {}
    for row_id in train_ids:
        train_by_key.setdefault(tuple(keys[int(row_id)]), []).append(int(time_ns[int(row_id)]))
    for row_id in validation_ids:
        validation_by_key.setdefault(tuple(keys[int(row_id)]), []).append(int(time_ns[int(row_id)]))
    for key in sorted(set(train_by_key).intersection(validation_by_key)):
        train_times = np.sort(np.asarray(train_by_key[key], dtype=np.int64))
        validation_times = np.sort(np.asarray(validation_by_key[key], dtype=np.int64))
        insertions = np.searchsorted(train_times, validation_times)
        nearest = np.full(len(validation_times), np.iinfo(np.int64).max, dtype=np.int64)
        right = insertions < len(train_times)
        nearest[right] = np.minimum(
            nearest[right], np.abs(train_times[insertions[right]] - validation_times[right])
        )
        left = insertions > 0
        nearest[left] = np.minimum(
            nearest[left], np.abs(train_times[insertions[left] - 1] - validation_times[left])
        )
        if np.any(nearest < purge_ns):
            minimum = pd.Timedelta(int(nearest.min()), unit="ns")
            raise AssertionError(
                f"time purge failed for key {key}: nearest separation {minimum} < {purge_delta}"
            )


__all__ = [
    "ANOMALY_TYPES",
    "ASRFTargets",
    "EncodedRows",
    "ExactSegment",
    "RobustRowEncoder",
    "SegmentLayout",
    "WindowIndex",
    "assert_disjoint_and_time_purged",
    "build_asrf_targets",
    "build_window_index",
    "materialize_windows",
    "select_training_windows",
    "stitch_center_weighted",
]
