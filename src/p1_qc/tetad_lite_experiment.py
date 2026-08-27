"""Leakage-safe utilities for the preregistered P1 TE-TAD-lite experiment.

This module deliberately contains no CLI and never discovers competition data.
Callers must pass the five frozen, non-official local artefact families explicitly.
All interval coordinates are normalized half-open ``[start, end)`` coordinates.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from tinygrad import Tensor, nn
from tinygrad.helpers import Context
from tinygrad.nn.state import get_parameters

from p1_qc.tetad_lite_tinygrad import (
    TETADLiteTinygrad,
    aggregate_binary_metrics,
    interval_set_loss,
)

KEY_COLUMNS = ("station", "year", "layer", "time")
TARGET_TYPES = frozenset({"offset", "drift", "noise"})
CADENCE = timedelta(minutes=10)
CANONICAL_SUFFIXES = {
    "feature_cache": "artifacts/cache/train_offline_e9fe1eb46cb7431f.parquet",
    "key_sidecar": (
        "artifacts/p1_round_b_nonspike_long_event_residual_v1r3_preexecution/"
        "feature_cache_row_keys.parquet"
    ),
    "label_cache": "artifacts/cache/io_benchmark/parquet/train.zstd.parquet",
    "truth_oof": "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet",
}
CANONICAL_HASHES = {
    "feature_cache": "f37c56ff016e90fb9a8d86299b4d9528c8f2e03181d326169b561fe3b27bc912",
    "key_sidecar": "7a79d9beaeaf5344c10d876882c39bd35a9196b20cab70caa407c33e2892b100",
    "label_cache": "1115e5d939220f2e0ca03f8cda6ffa4f680ad27b5e3c3308a2b0752a54fbe9d0",
    "truth_oof": "d1b9439db6d0d906fa080bd01f1eb8fc21d051c3d056a274e2b02e43c1e55f4a",
}
ROUND_B_PARTS = {
    "2025_q2": (
        "artifacts/p1_meaningful_learning_curve_generation_v1/prediction_parts/2025_q2_p100.parquet",
        "67ee943c6b577c8ae2f1f89cc36ee72a0aa204e5a9a6e45e1be9998d32846626",
        133_170,
    ),
    "2025_q3": (
        "artifacts/p1_meaningful_learning_curve_generation_v1/prediction_parts/2025_q3_p100.parquet",
        "8c8f3b1f9f944cacca370201fbf9acd32560ad2a9307a127253ec1ab7c76c369",
        176_738,
    ),
    "2025_q4": (
        "artifacts/p1_meaningful_learning_curve_generation_v1/prediction_parts/2025_q4_p100.parquet",
        "f484c69ea7bc05bd59080b9e480faba814a50144a54ee291b5153320794a5bd3",
        111_124,
    ),
}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenParquet:
    path: Path
    sha256: str
    rows: int
    canonical_suffix: str
    fold: str | None = None

    def verify(self) -> None:
        normalized = self.path.resolve().as_posix().casefold()
        suffix = self.canonical_suffix.replace("\\", "/").casefold()
        if not normalized.endswith(suffix):
            raise ValueError(f"non-canonical frozen path: {self.path}")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        observed = sha256_file(self.path)
        if observed != self.sha256.lower():
            raise RuntimeError(f"SHA-256 mismatch for frozen input: {self.path}")


@dataclass(frozen=True)
class FrozenTrainingInputs:
    feature_cache: FrozenParquet
    key_sidecar: FrozenParquet
    label_cache: FrozenParquet


def canonical_frozen_inputs(
    repository_root: Path,
) -> tuple[FrozenTrainingInputs, FrozenParquet, tuple[FrozenParquet, ...]]:
    """Return the sealed local-only inputs without searching the filesystem."""

    root = repository_root.resolve()
    training = FrozenTrainingInputs(
        feature_cache=FrozenParquet(
            root / CANONICAL_SUFFIXES["feature_cache"],
            CANONICAL_HASHES["feature_cache"],
            776_706,
            CANONICAL_SUFFIXES["feature_cache"],
        ),
        key_sidecar=FrozenParquet(
            root / CANONICAL_SUFFIXES["key_sidecar"],
            CANONICAL_HASHES["key_sidecar"],
            776_706,
            CANONICAL_SUFFIXES["key_sidecar"],
        ),
        label_cache=FrozenParquet(
            root / CANONICAL_SUFFIXES["label_cache"],
            CANONICAL_HASHES["label_cache"],
            776_706,
            CANONICAL_SUFFIXES["label_cache"],
        ),
    )
    truth = FrozenParquet(
        root / CANONICAL_SUFFIXES["truth_oof"],
        CANONICAL_HASHES["truth_oof"],
        421_032,
        CANONICAL_SUFFIXES["truth_oof"],
    )
    parts = tuple(
        FrozenParquet(root / suffix, checksum, rows, suffix, fold=fold)
        for fold, (suffix, checksum, rows) in ROUND_B_PARTS.items()
    )
    return training, truth, parts


@dataclass(frozen=True)
class TrainingBundle:
    features: pl.DataFrame
    keys: pl.DataFrame
    labels: pl.DataFrame

    @property
    def rows(self) -> int:
        return self.keys.height


@dataclass(frozen=True)
class ValidationBundle:
    """Label-free features and keys in frozen validation-membership order."""

    features: pl.DataFrame
    keys: pl.DataFrame

    @property
    def rows(self) -> int:
        return self.keys.height


def _read_frozen(spec: FrozenParquet, columns: Sequence[str]) -> pl.DataFrame:
    spec.verify()
    schema = pl.read_parquet_schema(spec.path)
    missing = set(columns).difference(schema)
    if missing:
        raise RuntimeError(f"missing frozen columns in {spec.path.name}: {sorted(missing)}")
    frame = pl.read_parquet(spec.path, columns=list(columns), rechunk=True)
    if frame.height != spec.rows:
        raise RuntimeError(f"frozen row count changed for {spec.path}")
    return frame


def _key_arrays(frame: pl.DataFrame) -> tuple[np.ndarray, ...]:
    return tuple(frame.get_column(column).to_numpy() for column in KEY_COLUMNS)


def _assert_key_order(left: pl.DataFrame, right: pl.DataFrame, context: str) -> None:
    if left.height != right.height:
        raise RuntimeError(f"{context}: row count differs")
    for column, lhs, rhs in zip(KEY_COLUMNS, _key_arrays(left), _key_arrays(right), strict=True):
        if not np.array_equal(lhs.astype(str), rhs.astype(str)):
            raise RuntimeError(f"{context}: exact {column} order differs")


def _load_bound_features_keys(
    inputs: FrozenTrainingInputs, numeric_feature_allowlist: Sequence[str]
) -> tuple[pl.DataFrame, pl.DataFrame, tuple[str, ...]]:
    if inputs.feature_cache.canonical_suffix != CANONICAL_SUFFIXES["feature_cache"]:
        raise ValueError("feature-cache canonical suffix is not the frozen contract")
    if inputs.key_sidecar.canonical_suffix != CANONICAL_SUFFIXES["key_sidecar"]:
        raise ValueError("key-sidecar canonical suffix is not the frozen contract")
    if inputs.label_cache.canonical_suffix != CANONICAL_SUFFIXES["label_cache"]:
        raise ValueError("label-cache canonical suffix is not the frozen contract")
    allowlist = tuple(numeric_feature_allowlist)
    if not allowlist or len(set(allowlist)) != len(allowlist):
        raise ValueError("numeric feature allowlist must be non-empty and unique")
    features = _read_frozen(
        inputs.feature_cache, ["station", "layer_category", *allowlist]
    )
    keys = _read_frozen(inputs.key_sidecar, ["ordinal", *KEY_COLUMNS])
    if features.height != keys.height:
        raise RuntimeError("frozen feature and key surfaces have different row counts")
    np.testing.assert_array_equal(
        keys.get_column("ordinal").to_numpy(), np.arange(keys.height, dtype=np.int64)
    )
    if not np.array_equal(
        features.get_column("station").cast(pl.String).to_numpy(),
        keys.get_column("station").cast(pl.String).to_numpy(),
    ):
        raise RuntimeError("feature-cache station order differs from frozen key sidecar")
    if not np.array_equal(
        features.get_column("layer_category").cast(pl.String).to_numpy(),
        keys.get_column("layer").cast(pl.String).to_numpy(),
    ):
        raise RuntimeError("feature-cache layer order differs from frozen key sidecar")
    return features, keys, allowlist


def load_training_prefix_bundle(
    inputs: FrozenTrainingInputs,
    numeric_feature_allowlist: Sequence[str],
    *,
    cutoff: datetime | str | np.datetime64,
) -> TrainingBundle:
    """Load an exact ``time < cutoff`` prefix without exposing later labels."""

    features, keys, _allowlist = _load_bound_features_keys(
        inputs, numeric_feature_allowlist
    )

    cutoff_time = _to_datetime(cutoff)
    key_times = np.asarray(
        [_to_datetime(value) for value in keys.get_column("time")], dtype=object
    )
    prefix_indices = np.flatnonzero(
        np.asarray([value < cutoff_time for value in key_times], dtype=bool)
    )
    if prefix_indices.size == 0:
        raise RuntimeError("declared training prefix is empty")
    # Verify the full label file identity/schema, then let Polars push the time
    # predicate below the sensitive label/anomaly projection.
    inputs.label_cache.verify()
    label_schema = pl.read_parquet_schema(inputs.label_cache.path)
    required = {*KEY_COLUMNS, "label", "anomaly_type"}
    if missing := required.difference(label_schema):
        raise RuntimeError(f"missing frozen label columns: {sorted(missing)}")
    cutoff_text = cutoff_time.isoformat()
    labels = (
        pl.scan_parquet(inputs.label_cache.path)
        .filter(pl.col("time") < pl.lit(cutoff_text))
        .select([*KEY_COLUMNS, "label", "anomaly_type"])
        .collect()
    )
    prefix_keys = keys.gather(prefix_indices)
    prefix_features = features.gather(prefix_indices)
    _assert_key_order(prefix_keys, labels, "prefix key/label binding")
    if not set(labels.get_column("label").unique().to_list()).issubset({0, 1}):
        raise RuntimeError("training label cache is not binary")
    return TrainingBundle(features=prefix_features, keys=prefix_keys, labels=labels)


def load_frozen_training_bundle(
    inputs: FrozenTrainingInputs,
    numeric_feature_allowlist: Sequence[str],
    *,
    label_cutoff: datetime | str | np.datetime64,
) -> TrainingBundle:
    """Compatibility name for the mandatory label-firewalled prefix loader."""

    return load_training_prefix_bundle(
        inputs, numeric_feature_allowlist, cutoff=label_cutoff
    )


def load_validation_membership(truth_oof: FrozenParquet) -> pl.DataFrame:
    """Load validation keys/folds only; labels are intentionally not projected."""

    if truth_oof.canonical_suffix != CANONICAL_SUFFIXES["truth_oof"]:
        raise ValueError("truth OOF canonical suffix is not the frozen contract")
    membership = _read_frozen(truth_oof, [*KEY_COLUMNS, "fold"])
    if membership.select(pl.struct(KEY_COLUMNS).is_duplicated().any()).item():
        raise RuntimeError("validation membership keys are duplicated")
    return membership


def load_validation_feature_bundle(
    inputs: FrozenTrainingInputs,
    truth_oof: FrozenParquet,
    numeric_feature_allowlist: Sequence[str],
    *,
    fold: str,
) -> ValidationBundle:
    """Align label-free feature rows to exactly one frozen validation fold."""

    features, all_keys, _allowlist = _load_bound_features_keys(
        inputs, numeric_feature_allowlist
    )
    membership = load_validation_membership(truth_oof).filter(pl.col("fold") == fold)
    if membership.is_empty():
        raise RuntimeError(f"validation membership fold is empty: {fold}")
    lookup = all_keys.select([*KEY_COLUMNS, "ordinal"])
    aligned = (
        membership.with_row_index("__membership_order")
        .join(lookup, on=list(KEY_COLUMNS), how="left", validate="1:1")
        .sort("__membership_order")
    )
    if aligned.get_column("ordinal").null_count():
        raise RuntimeError("validation membership does not bind to frozen feature keys")
    _assert_key_order(membership, aligned, "validation feature membership binding")
    ordinals = aligned.get_column("ordinal").to_numpy().astype(np.int64)
    selected_features = features.gather(ordinals)
    selected_keys = aligned.select(["ordinal", *KEY_COLUMNS, "fold"])
    if not np.array_equal(
        selected_features.get_column("station").cast(pl.String).to_numpy(),
        selected_keys.get_column("station").cast(pl.String).to_numpy(),
    ):
        raise RuntimeError("validation feature station binding failed")
    return ValidationBundle(selected_features, selected_keys)


def load_frozen_anchor_surface(
    truth_oof: FrozenParquet,
    round_b_parts: Sequence[FrozenParquet],
    *,
    prediction_column: str = "event_day_balanced_binary_lgbm__prediction",
) -> pl.DataFrame:
    """Load anchor predictions aligned to key/fold membership, without OOF labels."""

    membership = load_validation_membership(truth_oof)
    parts: list[pl.DataFrame] = []
    for spec in round_b_parts:
        if spec.fold is None:
            raise ValueError("each Round-B part must declare its frozen fold")
        if spec.fold not in ROUND_B_PARTS or spec.canonical_suffix != ROUND_B_PARTS[spec.fold][0]:
            raise ValueError(f"non-canonical Round-B part: {spec.fold}")
        part = _read_frozen(
            spec,
            [*KEY_COLUMNS, "row_position", "fold", "fraction", prediction_column],
        )
        if set(part.get_column("fold").unique().to_list()) != {spec.fold}:
            raise RuntimeError(f"Round-B fold identity changed: {spec.fold}")
        if not np.allclose(part.get_column("fraction").to_numpy(), 1.0):
            raise RuntimeError(f"Round-B part is not p100: {spec.fold}")
        parts.append(part)
    if not parts:
        raise ValueError("at least one Round-B part is required")
    anchor = pl.concat(parts, how="vertical")
    if anchor.select(pl.struct(KEY_COLUMNS).is_duplicated().any()).item():
        raise RuntimeError("Round-B keys are duplicated")
    if anchor.get_column("row_position").n_unique() != anchor.height:
        raise RuntimeError("Round-B row positions are duplicated")
    _assert_key_order(membership, anchor, "membership/Round-B anchor binding")
    if not np.array_equal(
        membership.get_column("fold").cast(pl.String).to_numpy(),
        anchor.get_column("fold").cast(pl.String).to_numpy(),
    ):
        raise RuntimeError("truth/Round-B fold order differs")
    anchor_values = anchor.get_column(prediction_column).to_numpy()
    if not np.isin(anchor_values, (0, 1)).all():
        raise RuntimeError("Round-B anchor is not binary")
    return membership.with_columns(
        pl.Series("anchor_prediction", anchor_values.astype(np.int8))
    )


def load_validation_fold_truth(truth_oof: FrozenParquet, fold: str) -> pl.DataFrame:
    """Open labels for exactly one named fold after its blind prediction is sealed."""

    if not fold:
        raise ValueError("fold must be non-empty")
    truth_oof.verify()
    schema = pl.read_parquet_schema(truth_oof.path)
    required = {*KEY_COLUMNS, "fold", "label", "anomaly_type"}
    if missing := required.difference(schema):
        raise RuntimeError(f"missing frozen truth columns: {sorted(missing)}")
    frame = (
        pl.scan_parquet(truth_oof.path)
        .filter(pl.col("fold") == fold)
        .select([*KEY_COLUMNS, "fold", "label", "anomaly_type"])
        .collect()
    )
    if frame.is_empty():
        raise RuntimeError(f"frozen truth fold is empty: {fold}")
    return frame


def _to_datetime(value: datetime | str | np.datetime64) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def assert_split_dependency_safe(
    train_end: datetime | str | np.datetime64,
    validation_start: datetime | str | np.datetime64,
    *,
    purge: timedelta,
    window_length: int = 1024,
    cadence: timedelta = CADENCE,
    maximum_feature_lookahead: timedelta = timedelta(hours=84),
) -> None:
    """Fail unless the declared purge covers window span plus feature lookahead."""

    if window_length <= 0 or cadence <= timedelta(0):
        raise ValueError("window length and cadence must be positive")
    train_time, validation_time = _to_datetime(train_end), _to_datetime(validation_start)
    required = (window_length - 1) * cadence + maximum_feature_lookahead
    observed = validation_time - train_time
    if purge < required:
        raise RuntimeError(f"purge {purge} is below dependency span {required}")
    if observed < purge:
        raise RuntimeError(f"observed split gap {observed} is below declared purge {purge}")


@dataclass(frozen=True)
class CadenceSegment:
    segment_id: int
    station: str
    year: int
    layer: int
    row_indices: np.ndarray
    times: tuple[datetime, ...]


def exact_cadence_segments(keys: pl.DataFrame) -> list[CadenceSegment]:
    """Create station/year/layer segments, breaking at every non-10-minute gap."""

    missing = set(KEY_COLUMNS).difference(keys.columns)
    if missing:
        raise ValueError(f"missing key columns: {sorted(missing)}")
    groups: dict[tuple[str, int, int], list[tuple[datetime, int]]] = defaultdict(list)
    for index, station, year, layer, time in zip(
        np.arange(keys.height), *(_key_arrays(keys)), strict=True
    ):
        groups[(str(station), int(year), int(layer))].append((_to_datetime(time), int(index)))
    segments: list[CadenceSegment] = []
    segment_id = 0
    for (station, year, layer), entries in sorted(groups.items()):
        entries.sort(key=lambda item: item[0])
        if any(
            right[0] <= left[0]
            for left, right in zip(entries, entries[1:], strict=False)
        ):
            raise RuntimeError("duplicate or decreasing timestamps within station/layer")
        start = 0
        for cursor in range(1, len(entries) + 1):
            boundary = cursor == len(entries) or entries[cursor][0] - entries[cursor - 1][0] != CADENCE
            if boundary:
                block = entries[start:cursor]
                segments.append(
                    CadenceSegment(
                        segment_id=segment_id,
                        station=station,
                        year=year,
                        layer=layer,
                        row_indices=np.asarray([entry[1] for entry in block], dtype=np.int64),
                        times=tuple(entry[0] for entry in block),
                    )
                )
                segment_id += 1
                start = cursor
    return segments


def exact_segment_cluster_ids(
    segments: Sequence[CadenceSegment], *, total_rows: int
) -> np.ndarray:
    """Map every row to its station/year/layer exact-cadence segment key."""

    if total_rows <= 0:
        raise ValueError("total rows must be positive")
    output = np.empty(total_rows, dtype=object)
    assigned = np.zeros(total_rows, dtype=bool)
    for segment in segments:
        if len(segment.row_indices) == 0:
            raise RuntimeError("empty cadence segment")
        if (
            (segment.row_indices < 0).any()
            or (segment.row_indices >= total_rows).any()
            or assigned[segment.row_indices].any()
        ):
            raise RuntimeError("cadence segment rows overlap or are out of range")
        cluster = (
            f"{segment.station}|{segment.year}|{segment.layer}|"
            f"{segment.times[0].isoformat()}|{segment.times[-1].isoformat()}"
        )
        output[segment.row_indices] = cluster
        assigned[segment.row_indices] = True
    if not assigned.all():
        raise RuntimeError("cadence segments do not cover every row exactly once")
    return output


@dataclass(frozen=True)
class TargetEvent:
    segment_id: int
    anomaly_type: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def eligible_target_events(
    segments: Sequence[CadenceSegment],
    labels: pl.DataFrame,
    *,
    minimum_original_rows: int = 19,
    right_censor_cutoff: datetime | str | np.datetime64 | None = None,
) -> list[TargetEvent]:
    """Find long events in the binary union of offset/drift/noise tokens."""

    if minimum_original_rows <= 0:
        raise ValueError("minimum event length must be positive")
    label_values = labels.get_column("label").to_numpy()
    type_values = labels.get_column("anomaly_type").fill_null("").cast(pl.String).to_numpy()
    events: list[TargetEvent] = []
    cutoff = _to_datetime(right_censor_cutoff) if right_censor_cutoff is not None else None
    for segment in segments:
        local_labels = label_values[segment.row_indices]
        local_types = type_values[segment.row_indices]
        eligible = np.asarray(
            [
                int(label) == 1
                and bool(TARGET_TYPES.intersection(str(kind).split("+")))
                for label, kind in zip(local_labels, local_types, strict=True)
            ],
            dtype=bool,
        )
        cursor = 0
        while cursor < len(eligible):
            if not eligible[cursor]:
                cursor += 1
                continue
            end = cursor + 1
            while end < len(eligible) and eligible[end]:
                end += 1
            right_censored = bool(
                cutoff is not None
                and end == len(eligible)
                and segment.times[-1] + CADENCE >= cutoff
            )
            if end - cursor >= minimum_original_rows and not right_censored:
                events.append(
                    TargetEvent(segment.segment_id, "binary_union", cursor, end)
                )
            cursor = end
    return events


@dataclass(frozen=True)
class WindowSpec:
    segment_id: int
    start: int
    valid_length: int
    row_indices: np.ndarray
    targets: np.ndarray

    @property
    def identity(self) -> str:
        return f"{self.segment_id}:{self.start}:{self.valid_length}"


def build_windows(
    segments: Sequence[CadenceSegment],
    events: Sequence[TargetEvent],
    *,
    window_length: int = 1024,
    stride: int = 512,
    max_queries: int = 5,
) -> list[WindowSpec]:
    """Build split-local right-padded windows and clipped half-open targets."""

    if min(window_length, stride, max_queries) <= 0:
        raise ValueError("window length, stride, and query budget must be positive")
    if stride > window_length:
        raise ValueError("stride cannot exceed window length because inference must cover every row")
    events_by_segment: dict[int, list[TargetEvent]] = defaultdict(list)
    for event in events:
        events_by_segment[event.segment_id].append(event)
    windows: list[WindowSpec] = []
    for segment in segments:
        length = len(segment.row_indices)
        if length == 0:
            continue
        if length < window_length:
            starts = [0]
        else:
            starts = list(range(0, length - window_length + 1, stride))
            final_start = length - window_length
            if starts[-1] != final_start:
                starts.append(final_start)
        for start in starts:
            valid_length = min(window_length, length - start)
            end = start + valid_length
            clipped: list[tuple[float, float]] = []
            for event in events_by_segment.get(segment.segment_id, []):
                left, right = max(start, event.start), min(end, event.end)
                if left < right:
                    clipped.append(
                        ((left - start) / window_length, (right - start) / window_length)
                    )
            if len(clipped) > max_queries:
                raise RuntimeError(
                    f"target count {len(clipped)} exceeds query budget {max_queries} "
                    f"in window {segment.segment_id}:{start}"
                )
            windows.append(
                WindowSpec(
                    segment_id=segment.segment_id,
                    start=start,
                    valid_length=valid_length,
                    row_indices=segment.row_indices[start:end].copy(),
                    targets=np.asarray(clipped, dtype=np.float32).reshape(-1, 2),
                )
            )
    return windows


def deterministic_training_sample(
    windows: Sequence[WindowSpec], *, seed: int, empty_ratio: int = 2
) -> list[WindowSpec]:
    """Keep every target window and hash-rank at most 2x empty windows."""

    if empty_ratio < 0:
        raise ValueError("empty ratio must be non-negative")
    positive = [window for window in windows if len(window.targets)]
    empty = [window for window in windows if not len(window.targets)]
    ranked = sorted(
        empty,
        key=lambda window: hashlib.sha256(
            f"{seed}:{window.identity}".encode()
        ).digest(),
    )
    selected = positive + ranked[: min(len(ranked), empty_ratio * len(positive))]
    return sorted(selected, key=lambda window: (window.segment_id, window.start))


@dataclass(frozen=True)
class RobustPreprocessor:
    feature_names: tuple[str, ...]
    median: np.ndarray
    scale: np.ndarray
    stations: tuple[str, ...]
    layer_median: float
    layer_scale: float

    @classmethod
    def fit(
        cls,
        bundle: TrainingBundle,
        train_indices: np.ndarray | Sequence[int],
        feature_names: Sequence[str],
        *,
        train_end: datetime | str | np.datetime64,
    ) -> RobustPreprocessor:
        indices = np.asarray(train_indices, dtype=np.int64)
        if indices.size == 0 or (indices < 0).any() or (indices >= bundle.rows).any():
            raise ValueError("train indices must be a non-empty in-range prefix selection")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("train indices contain duplicates")
        cutoff = _to_datetime(train_end)
        times = np.asarray(
            [_to_datetime(value) for value in bundle.keys.get_column("time")], dtype=object
        )
        expected = np.flatnonzero(
            np.asarray([value <= cutoff for value in times], dtype=bool)
        )
        if not np.array_equal(np.sort(indices), expected):
            raise RuntimeError(
                "preprocessor fit rows are not exactly the declared chronological prefix"
            )
        names = tuple(feature_names)
        matrix = bundle.features.select(names).to_numpy()[indices].astype(np.float64)
        matrix[~np.isfinite(matrix)] = np.nan
        median = np.nanmedian(matrix, axis=0)
        q25, q75 = np.nanpercentile(matrix, [25.0, 75.0], axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        scale = np.where(np.isfinite(q75 - q25) & ((q75 - q25) > 1e-6), q75 - q25, 1.0)
        station_values = bundle.keys.get_column("station").cast(pl.String).to_numpy()[indices]
        layers = bundle.keys.get_column("layer").to_numpy()[indices].astype(np.float64)
        layer_median = float(np.median(layers))
        layer_scale = float(np.subtract(*np.percentile(layers, [75.0, 25.0])))
        if not np.isfinite(layer_scale) or layer_scale <= 1e-6:
            layer_scale = 1.0
        return cls(names, median, scale, tuple(sorted(set(station_values))), layer_median, layer_scale)

    @property
    def output_features(self) -> int:
        return 2 * len(self.feature_names) + len(self.stations) + 2

    def transform_window(
        self,
        bundle: TrainingBundle | ValidationBundle,
        window: WindowSpec,
        *,
        window_length: int = 1024,
    ) -> np.ndarray:
        raw_all = bundle.features.select(self.feature_names).to_numpy().astype(np.float64)
        station_all = bundle.keys.get_column("station").cast(pl.String).to_numpy()
        layer_all = bundle.keys.get_column("layer").to_numpy().astype(np.float64)
        return self._transform_selected(
            raw_all[window.row_indices],
            station_all[window.row_indices],
            layer_all[window.row_indices],
            window,
            window_length=window_length,
        )

    def _transform_selected(
        self,
        raw: np.ndarray,
        station_values: np.ndarray,
        layer_values: np.ndarray,
        window: WindowSpec,
        *,
        window_length: int,
    ) -> np.ndarray:
        if window.valid_length > window_length:
            raise ValueError("valid window exceeds materialized window length")
        missing = ~np.isfinite(raw)
        normalized = (np.where(missing, self.median, raw) - self.median) / self.scale
        station_matrix = np.zeros((window.valid_length, len(self.stations)), dtype=np.float32)
        station_lookup = {station: index for index, station in enumerate(self.stations)}
        for row, station in enumerate(station_values):
            if station in station_lookup:
                station_matrix[row, station_lookup[station]] = 1.0
        layer = ((layer_values - self.layer_median) / self.layer_scale).reshape(-1, 1)
        valid = np.ones((window.valid_length, 1), dtype=np.float32)
        materialized = np.concatenate(
            [normalized, missing.astype(np.float32), station_matrix, layer, valid], axis=1
        ).astype(np.float32)
        output = np.zeros((window_length, self.output_features), dtype=np.float32)
        output[: window.valid_length] = materialized
        return output


def materialize_windows(
    bundle: TrainingBundle | ValidationBundle,
    windows: Sequence[WindowSpec],
    preprocessor: RobustPreprocessor,
    *,
    window_length: int = 1024,
) -> np.ndarray:
    if not windows:
        return np.empty((0, window_length, preprocessor.output_features), dtype=np.float32)
    raw_all = bundle.features.select(preprocessor.feature_names).to_numpy().astype(np.float64)
    station_all = bundle.keys.get_column("station").cast(pl.String).to_numpy()
    layer_all = bundle.keys.get_column("layer").to_numpy().astype(np.float64)
    materialized = np.stack(
        [
            preprocessor._transform_selected(
                raw_all[window.row_indices],
                station_all[window.row_indices],
                layer_all[window.row_indices],
                window,
                window_length=window_length,
            )
            for window in windows
        ]
    )
    if not np.isfinite(materialized).all():
        raise RuntimeError("materialized model features contain non-finite values")
    return materialized


@dataclass(frozen=True)
class TrainingResult:
    model: TETADLiteTinygrad
    epoch_losses: tuple[float, ...]


def train_tetad(
    model: TETADLiteTinygrad,
    features: np.ndarray,
    targets: Sequence[np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    positive_class_weight: float,
    seed: int,
    classification_weight: float = 1.0,
    endpoint_weight: float = 2.0,
    iou_weight: float = 2.0,
) -> TrainingResult:
    """Train for exactly the caller-frozen epoch count with deterministic batches."""

    if features.ndim != 3 or len(features) != len(targets) or len(features) == 0:
        raise ValueError("features/targets must be aligned non-empty windows")
    if min(epochs, batch_size) <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid fixed training configuration")
    Tensor.manual_seed(seed)
    parameters = get_parameters(model)
    if not parameters:
        raise RuntimeError("model has no trainable parameters")
    device = parameters[0].device
    optimizer = nn.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay, device=device
    )
    rng = np.random.default_rng(seed)
    history: list[float] = []
    with Context(TRAINING=1):
        for _epoch in range(epochs):
            order = rng.permutation(len(features))
            weighted_loss, seen = 0.0, 0
            for offset in range(0, len(order), batch_size):
                indices = order[offset : offset + batch_size]
                batch = Tensor(features[indices], device=device)
                logits, intervals = model(batch)
                loss = interval_set_loss(
                    logits,
                    intervals,
                    [targets[int(index)] for index in indices],
                    classification_weight=classification_weight,
                    endpoint_weight=endpoint_weight,
                    iou_weight=iou_weight,
                    positive_class_weight=positive_class_weight,
                )
                optimizer.zero_grad()
                loss.total.backward()
                optimizer.step()
                value = float(loss.total.numpy())
                if not np.isfinite(value):
                    raise RuntimeError("non-finite training loss")
                weighted_loss += value * len(indices)
                seen += len(indices)
            history.append(weighted_loss / seen)
    return TrainingResult(model=model, epoch_losses=tuple(history))


def predict_window_proposals(
    model: TETADLiteTinygrad,
    features: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 3 or batch_size <= 0:
        raise ValueError("invalid inference features or batch size")
    parameters = get_parameters(model)
    device = parameters[0].device
    all_intervals: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    with Context(TRAINING=0):
        for offset in range(0, len(features), batch_size):
            logits, intervals = model(Tensor(features[offset : offset + batch_size], device=device))
            all_intervals.append(intervals.numpy().astype(np.float32))
            all_scores.append(logits.sigmoid().numpy().astype(np.float32))
    return np.concatenate(all_intervals), np.concatenate(all_scores)


def stitch_proposals(
    windows: Sequence[WindowSpec],
    intervals: np.ndarray,
    scores: np.ndarray,
    *,
    total_rows: int,
    threshold: float,
    minimum_decoded_rows: int = 19,
    coordinate_length: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode proposals then aggregate row confidence by maximum over overlaps."""

    if intervals.shape[:2] != scores.shape or intervals.shape[0] != len(windows) or intervals.shape[2] != 2:
        raise ValueError("proposal arrays do not align with windows")
    if (
        total_rows <= 0
        or minimum_decoded_rows <= 0
        or coordinate_length <= 0
        or not 0 <= threshold <= 1
    ):
        raise ValueError("invalid stitching configuration")
    confidence = np.zeros(total_rows, dtype=np.float32)
    for window, window_intervals, window_scores in zip(windows, intervals, scores, strict=True):
        for endpoints, score in zip(window_intervals, window_scores, strict=True):
            start_value, end_value = sorted(np.clip(endpoints.astype(float), 0.0, 1.0))
            decoded_start = int(np.floor(start_value * coordinate_length))
            decoded_end = int(np.ceil(end_value * coordinate_length))
            # Evaluate length in the fixed coordinate system before clipping to
            # the real (unpadded) part of a short window.
            if decoded_end - decoded_start < minimum_decoded_rows:
                continue
            decoded_start = min(max(decoded_start, 0), window.valid_length)
            decoded_end = min(max(decoded_end, 0), window.valid_length)
            rows = window.row_indices[decoded_start:decoded_end]
            if len(rows):
                confidence[rows] = np.maximum(confidence[rows], float(score))
    return confidence, (confidence >= threshold).astype(np.int8)


def anchor_preserving_union(
    anchor: np.ndarray | Sequence[int], proposal: np.ndarray | Sequence[int]
) -> np.ndarray:
    anchor_array = np.asarray(anchor, dtype=np.int8).reshape(-1)
    proposal_array = np.asarray(proposal, dtype=np.int8).reshape(-1)
    if anchor_array.shape != proposal_array.shape or not np.isin(anchor_array, (0, 1)).all() or not np.isin(proposal_array, (0, 1)).all():
        raise ValueError("anchor and proposal must be aligned binary vectors")
    candidate = np.maximum(anchor_array, proposal_array).astype(np.int8)
    if np.any((anchor_array == 1) & (candidate == 0)):
        raise AssertionError("anchor union attempted a forbidden 1->0 change")
    return candidate


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    return 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def compare_anchor_candidate(
    truth: np.ndarray | Sequence[int],
    anchor: np.ndarray | Sequence[int],
    candidate: np.ndarray | Sequence[int],
    *,
    folds: Sequence[str] | np.ndarray | None = None,
) -> dict[str, Any]:
    truth_array = np.asarray(truth, dtype=np.int8).reshape(-1)
    anchor_array = np.asarray(anchor, dtype=np.int8).reshape(-1)
    candidate_array = np.asarray(candidate, dtype=np.int8).reshape(-1)
    if not (truth_array.shape == anchor_array.shape == candidate_array.shape):
        raise ValueError("truth/anchor/candidate shapes differ")
    if np.any((anchor_array == 1) & (candidate_array == 0)):
        raise RuntimeError("candidate violates anchor-preserving union")

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        base = aggregate_binary_metrics(truth_array[mask], anchor_array[mask])
        new = aggregate_binary_metrics(truth_array[mask], candidate_array[mask])
        added = (anchor_array[mask] == 0) & (candidate_array[mask] == 1)
        added_tp = int(np.sum(added & (truth_array[mask] == 1)))
        added_fp = int(np.sum(added & (truth_array[mask] == 0)))
        return {
            "anchor": base,
            "candidate": new,
            "f1_delta": float(new["f1"] - base["f1"]),
            "additional_rows": int(np.sum(added)),
            "additional_tp": added_tp,
            "additional_fp": added_fp,
            "added_precision": added_tp / (added_tp + added_fp) if added_tp + added_fp else 0.0,
        }

    output: dict[str, Any] = {"overall": summarize(np.ones(len(truth_array), dtype=bool))}
    if folds is not None:
        fold_array = np.asarray(folds).astype(str)
        if fold_array.shape != truth_array.shape:
            raise ValueError("fold vector shape differs")
        output["folds"] = {
            fold: summarize(fold_array == fold) for fold in dict.fromkeys(fold_array.tolist())
        }
    return output


def paired_cluster_bootstrap_ci90(
    truth: np.ndarray | Sequence[int],
    anchor: np.ndarray | Sequence[int],
    candidate: np.ndarray | Sequence[int],
    cluster_ids: Sequence[Any],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Paired cluster bootstrap using the caller's exact segment identifiers."""

    truth_array = np.asarray(truth, dtype=np.int8).reshape(-1)
    anchor_array = np.asarray(anchor, dtype=np.int8).reshape(-1)
    candidate_array = np.asarray(candidate, dtype=np.int8).reshape(-1)
    identifiers = list(cluster_ids)
    if not (
        anchor_array.shape == truth_array.shape
        and candidate_array.shape == truth_array.shape
        and len(identifiers) == len(truth_array)
    ):
        raise ValueError("bootstrap arrays are not aligned")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    grouped: dict[Any, list[int]] = defaultdict(list)
    for row, cluster_id in enumerate(identifiers):
        try:
            grouped[cluster_id].append(row)
        except TypeError as exc:
            raise ValueError("cluster identifiers must be hashable") from exc
    clusters = sorted(grouped, key=repr)
    if not clusters:
        raise ValueError("no bootstrap clusters")
    counts = np.zeros((len(clusters), 6), dtype=np.int64)
    for index, cluster in enumerate(clusters):
        mask = np.zeros(len(truth_array), dtype=bool)
        mask[np.asarray(grouped[cluster], dtype=np.int64)] = True
        for offset, prediction in ((0, anchor_array), (3, candidate_array)):
            positive = truth_array[mask] == 1
            predicted = prediction[mask] == 1
            counts[index, offset : offset + 3] = (
                np.sum(positive & predicted),
                np.sum(~positive & predicted),
                np.sum(positive & ~predicted),
            )
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        summed = counts[sampled].sum(axis=0)
        deltas[replicate] = _f1_from_counts(*summed[3:6]) - _f1_from_counts(*summed[0:3])
    low, high = np.quantile(deltas, [0.05, 0.95])
    observed_anchor = aggregate_binary_metrics(truth_array, anchor_array)["f1"]
    observed_candidate = aggregate_binary_metrics(truth_array, candidate_array)["f1"]
    return {
        "replicates": int(replicates),
        "clusters": int(len(clusters)),
        "seed": int(seed),
        "observed_delta": float(observed_candidate - observed_anchor),
        "bootstrap_mean": float(np.mean(deltas)),
        "ci90_low": float(low),
        "ci90_high": float(high),
    }


def station_layer_cluster_bootstrap_ci90(
    truth: np.ndarray | Sequence[int],
    anchor: np.ndarray | Sequence[int],
    candidate: np.ndarray | Sequence[int],
    stations: Sequence[str] | np.ndarray,
    layers: Sequence[int] | np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Compatibility wrapper; segment IDs are preferred for experiment evidence."""

    station_values, layer_values = list(stations), list(layers)
    if len(station_values) != len(layer_values):
        raise ValueError("station/layer vectors differ")
    return paired_cluster_bootstrap_ci90(
        truth,
        anchor,
        candidate,
        list(zip(station_values, layer_values, strict=True)),
        replicates=replicates,
        seed=seed,
    )


def _interval_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = max(0.0, min(float(left[1]), float(right[1])) - max(float(left[0]), float(right[0])))
    union = max(float(left[1] - left[0]) + float(right[1] - right[0]) - intersection, 1e-12)
    return intersection / union


def evaluate_overfit_sanity(
    intervals: np.ndarray,
    scores: np.ndarray,
    targets: Sequence[np.ndarray],
    *,
    score_threshold: float = 0.5,
    match_iou: float = 0.5,
) -> dict[str, float | int | bool]:
    """Aggregate implementation-sanity metrics; this is not validation evidence."""

    if intervals.shape[:2] != scores.shape or intervals.shape[0] != len(targets):
        raise ValueError("overfit proposals/targets are not aligned")
    finite = bool(np.isfinite(intervals).all() and np.isfinite(scores).all())
    matched_ious: list[float] = []
    target_count = 0
    negative_window_fp = 0
    for proposed, confidence, target in zip(intervals, scores, targets, strict=True):
        selected = proposed[confidence >= score_threshold]
        target_array = np.asarray(target, dtype=np.float32).reshape(-1, 2)
        if len(target_array) == 0:
            negative_window_fp += int(len(selected) > 0)
            continue
        target_count += len(target_array)
        available = set(range(len(selected)))
        for truth_interval in target_array:
            if not available:
                continue
            best = max(available, key=lambda index: _interval_iou(selected[index], truth_interval))
            iou = _interval_iou(selected[best], truth_interval)
            if iou >= match_iou:
                matched_ious.append(iou)
                available.remove(best)
    return {
        "finite": finite,
        "targets": int(target_count),
        "matched_targets": int(len(matched_ious)),
        "target_recall": len(matched_ious) / target_count if target_count else 0.0,
        "median_matched_iou": float(np.median(matched_ious)) if matched_ious else 0.0,
        "negative_window_fp_windows": int(negative_window_fp),
    }


def overfit_sanity_passes(
    metrics: Mapping[str, float | int | bool],
    *,
    minimum_recall: float = 0.95,
    minimum_median_iou: float = 0.80,
) -> bool:
    return bool(
        metrics.get("finite")
        and float(metrics.get("target_recall", 0.0)) >= minimum_recall
        and float(metrics.get("median_matched_iou", 0.0)) >= minimum_median_iou
        and int(metrics.get("negative_window_fp_windows", 1)) == 0
    )
