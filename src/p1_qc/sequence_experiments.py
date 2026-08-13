"""Nested, fold-local orchestration for the P1 sequence and SSL models.

The outer validation labels are used only for the final diagnostic report.
Architecture, model configuration, seed ensemble, epoch checkpoint, and
decision threshold are selected inside each outer training prefix.  The
module deliberately has no CLI side effects; callers choose an ignored
artifact directory when they want checkpoints and result contracts.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .config import FoldWindowConfig, P1QCConfig
from .data import ANOMALY_TYPES, parse_anomaly_types, segment_timeseries
from .deep_training import (
    SequenceTrainConfig,
    SequenceTrainingResult,
    predefined_search_space,
    predict_sequence,
    robust_fit,
    robust_transform,
    save_sequence_checkpoint,
    train_sequence_model,
)
from .features import FeatureBundle
from .metrics import evaluate_predictions, micro_f1, weighted_group_f1
from .models_ssl import (
    SSLModelConfig,
    SSLTrainConfig,
    assert_fold_local_rows,
    extract_ssl_embeddings,
    save_ssl_checkpoint,
    train_masked_reconstruction,
)
from .splits import Fold, outer_folds

Architecture = Literal["tcn", "patch_transformer"]
RESULT_CONTRACT_VERSION = 1
CHECKPOINT_CONTRACT_VERSION = 1
_FIXED_OUTER_NAMES = ("2025_q2", "2025_q3", "2025_q4")
_FIXED_OUTER_WINDOWS = (
    (
        "2025_q2",
        "2025-03-24T23:50:00+09:00",
        "2025-04-01T00:00:00+09:00",
        "2025-07-01T00:00:00+09:00",
    ),
    (
        "2025_q3",
        "2025-06-23T23:50:00+09:00",
        "2025-07-01T00:00:00+09:00",
        "2025-10-01T00:00:00+09:00",
    ),
    (
        "2025_q4",
        "2025-09-23T23:50:00+09:00",
        "2025-10-01T00:00:00+09:00",
        # The stated validation range ends on 12-10, so this is exclusive.
        "2025-12-11T00:00:00+09:00",
    ),
)


@dataclass(frozen=True)
class SequenceExperimentConfig:
    """Auditable search policy, with an explicitly bounded smoke mode."""

    architectures: tuple[Architecture, ...] = ("tcn", "patch_transformer")
    causal: bool = False
    inner_validation_days: int = 60
    purge_days: int = 7
    window_steps: int = 2016
    stride_steps: int = 1008
    batch_size: int = 16
    screen_max_epochs: int = 50
    final_max_epochs: int = 50
    patience: int = 8
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    negative_ratio: float = 1.0
    auxiliary_weight: float = 0.2
    use_bfloat16: bool = True
    top_configurations: int = 1
    final_seeds: tuple[int, ...] = (20260813, 20260814, 20260815)
    threshold_grid: tuple[float, ...] = (
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
    )
    use_ssl: bool = False
    ssl_channels: tuple[int, ...] = (32, 64, 64)
    ssl_window_steps: int = 288
    ssl_stride_steps: int = 144
    ssl_max_epochs: int = 40
    ssl_patience: int = 6
    ssl_mask_fraction: float = 0.25
    smoke: bool = False
    smoke_fold_limit: int = 1
    smoke_config_limit: int = 1
    smoke_seed_limit: int = 1
    smoke_max_epochs: int = 1
    smoke_window_steps: int = 32

    def __post_init__(self) -> None:
        if not self.architectures or any(
            item not in {"tcn", "patch_transformer"} for item in self.architectures
        ):
            raise ValueError("architectures must contain tcn and/or patch_transformer")
        if len(set(self.architectures)) != len(self.architectures):
            raise ValueError("architectures must be unique")
        positive_counts = (
            self.inner_validation_days,
            self.window_steps,
            self.stride_steps,
            self.batch_size,
            self.screen_max_epochs,
            self.final_max_epochs,
            self.patience,
            self.top_configurations,
            self.smoke_fold_limit,
            self.smoke_config_limit,
            self.smoke_seed_limit,
            self.smoke_max_epochs,
            self.smoke_window_steps,
        )
        if any(value < 1 for value in positive_counts):
            raise ValueError("experiment count/window settings must be positive")
        if self.purge_days < 0:
            raise ValueError("purge_days cannot be negative")
        if not self.final_seeds:
            raise ValueError("at least one final seed is required")
        if not self.threshold_grid or any(
            not 0.0 < threshold < 1.0 for threshold in self.threshold_grid
        ):
            raise ValueError("threshold_grid values must lie in (0, 1)")
        if len(set(self.threshold_grid)) != len(self.threshold_grid):
            raise ValueError("threshold_grid values must be unique")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer values")
        if not 0 < self.ssl_mask_fraction < 1:
            raise ValueError("ssl_mask_fraction must lie in (0, 1)")
        if not self.smoke and not self.causal and self.purge_days < 7:
            raise ValueError("offline features require at least a seven-day inner purge")


@dataclass(frozen=True)
class TrialRecord:
    architecture: Architecture
    configuration_index: int
    parameters: Mapping[str, Any]
    seed: int
    threshold: float
    inner_f1: float
    best_epoch: int
    best_validation_loss: float


@dataclass(frozen=True)
class CheckpointRecord:
    contract_version: int
    architecture: str
    fold: str
    configuration_index: int | None
    seed: int
    path: str
    sha256: str
    bytes: int
    inner_train_rows_sha256: str
    inner_validation_rows_sha256: str
    model_parameters: Mapping[str, Any]
    training_parameters: Mapping[str, Any]


@dataclass
class FoldSequenceResult:
    fold: str
    train_rows: int
    outer_validation_rows: int
    inner_train_rows: int
    inner_validation_rows: int
    inner_train_rows_sha256: str
    inner_validation_rows_sha256: str
    outer_validation_rows_sha256: str
    inner_train_end: str
    inner_validation_start: str
    inner_validation_end: str
    ssl: Mapping[str, Any] | None
    architectures: dict[str, dict[str, Any]]


@dataclass
class SequenceExperimentResult:
    """In-memory result plus a JSON-safe, versioned persistence contract."""

    contract_version: int
    experiment: Mapping[str, Any]
    folds: list[FoldSequenceResult]
    comparison: list[dict[str, Any]]
    selection: Mapping[str, Any]
    checkpoints: list[CheckpointRecord]
    oof_predictions: dict[str, pd.DataFrame] = field(repr=False)
    artifacts: dict[str, Mapping[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "contract_version": self.contract_version,
                "experiment": self.experiment,
                "folds": [asdict(item) for item in self.folds],
                "comparison": self.comparison,
                "selection": self.selection,
                "checkpoints": [asdict(item) for item in self.checkpoints],
                "artifacts": self.artifacts,
            }
        )


@dataclass(frozen=True)
class _Partition:
    indices: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    auxiliary: np.ndarray
    segments: np.ndarray
    metadata: pd.DataFrame


def search_space(
    architecture: Architecture,
    *,
    causal: bool,
) -> tuple[dict[str, Any], ...]:
    """Return the immutable 12-setting search space promised by the plan."""

    values = tuple(dict(item) for item in predefined_search_space(architecture, causal=causal))
    if len(values) != 12:
        raise RuntimeError(f"{architecture} search space must contain exactly 12 settings")
    canonical = {json.dumps(_json_safe(item), sort_keys=True) for item in values}
    if len(canonical) != 12:
        raise RuntimeError(f"{architecture} search space contains duplicate settings")
    return values


def _index_digest(values: Sequence[int] | np.ndarray) -> str:
    array = np.asarray(values, dtype="<i8")
    return sha256(array.tobytes(order="C")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_fixed_outer_folds(folds: Sequence[Fold]) -> None:
    if tuple(item.name for item in folds) != _FIXED_OUTER_NAMES:
        raise ValueError(
            f"normal sequence experiments require fixed outer folds {_FIXED_OUTER_NAMES}"
        )
    for fold, expected in zip(folds, _FIXED_OUTER_WINDOWS, strict=True):
        observed = (fold.train_end, fold.val_start, fold.val_end)
        required = tuple(pd.Timestamp(value).tz_convert("UTC") for value in expected[1:])
        if any(left != right for left, right in zip(observed, required, strict=True)):
            raise ValueError(
                f"outer fold {fold.name} dates differ from the fixed rolling-origin contract"
            )


def build_inner_fold(
    frame: pd.DataFrame,
    outer: Fold,
    *,
    validation_days: int = 60,
    purge_days: int = 7,
    cadence_minutes: int = 10,
    group_columns: Sequence[str] = ("station", "layer"),
) -> Fold:
    """Create one past-only blocked split wholly inside ``outer.train_idx``."""

    if validation_days < 1 or purge_days < 0:
        raise ValueError("validation_days must be positive and purge_days non-negative")
    if not len(outer.train_idx):
        raise ValueError("outer training side is empty")
    time = pd.to_datetime(frame["time"], errors="coerce", utc=True)
    if time.isna().any():
        raise ValueError("timestamps could not be parsed")
    last_train_time = time.iloc[outer.train_idx].max()
    validation_start = (
        last_train_time - pd.Timedelta(days=validation_days) + pd.Timedelta(minutes=cadence_minutes)
    )
    train_end = validation_start - pd.Timedelta(days=purge_days, minutes=cadence_minutes)
    validation_end = last_train_time + pd.Timedelta(minutes=cadence_minutes)
    spec = FoldWindowConfig(
        name=f"{outer.name}_inner",
        train_end=train_end.isoformat(),
        val_start=validation_start.isoformat(),
        val_end=validation_end.isoformat(),
    )
    candidate = outer_folds(
        frame,
        specs=(spec,),
        purge_days=purge_days,
        cadence_minutes=cadence_minutes,
        group_columns=group_columns,
        protect_positive_runs=True,
    )[0]
    allowed = np.zeros(len(frame), dtype=bool)
    allowed[outer.train_idx] = True
    train_idx = candidate.train_idx[allowed[candidate.train_idx]]
    validation_idx = candidate.val_idx[allowed[candidate.val_idx]]
    if not len(train_idx) or not len(validation_idx):
        raise ValueError(f"inner split for {outer.name} has an empty side")
    if np.intersect1d(train_idx, validation_idx).size:
        raise RuntimeError("inner training and validation overlap")
    if (
        not np.isin(train_idx, outer.train_idx).all()
        or not np.isin(validation_idx, outer.train_idx).all()
    ):
        raise RuntimeError("inner split escaped the outer training prefix")
    if np.intersect1d(np.r_[train_idx, validation_idx], outer.val_idx).size:
        raise RuntimeError("outer validation rows leaked into the inner split")
    return Fold(
        name=spec.name,
        train_idx=train_idx,
        val_idx=validation_idx,
        train_end=train_end,
        val_start=validation_start,
        val_end=validation_end,
    )


def _partition(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    auxiliary: np.ndarray,
    indices: np.ndarray,
    *,
    cadence_minutes: int,
    group_columns: Sequence[str],
) -> _Partition:
    selected = frame.iloc[indices].copy()
    selected["__global_position"] = np.asarray(indices, dtype=np.int64)
    selected = selected.sort_values([*group_columns, "time", "__global_position"], kind="mergesort")
    ordered = selected["__global_position"].to_numpy(dtype=np.int64)
    segmented = segment_timeseries(
        selected.drop(columns="__global_position"),
        group_columns=group_columns,
        cadence_minutes=cadence_minutes,
    )
    return _Partition(
        indices=ordered,
        features=np.asarray(matrix[ordered], dtype=np.float32),
        labels=pd.to_numeric(frame.iloc[ordered]["label"], errors="raise").to_numpy(dtype=np.int8),
        auxiliary=np.asarray(auxiliary[ordered], dtype=np.float32),
        segments=segmented["segment_id"].to_numpy(),
        metadata=frame.iloc[ordered].loc[:, [*group_columns, "time"]].reset_index(drop=True),
    )


def _normal_ssl_partition(
    partition: _Partition,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = partition.labels == 0
    normal_positions = np.flatnonzero(normal)
    features = partition.features[normal]
    row_ids = partition.indices[normal]
    original_segments = partition.segments[normal]
    if len(features) < 2:
        raise ValueError("SSL needs at least two normal rows")
    # Removing injected anomalies must not make the encoder bridge across them.
    contiguous = np.r_[
        False,
        (original_segments[1:] == original_segments[:-1])
        & (normal_positions[1:] == normal_positions[:-1] + 1),
    ]
    segments = np.cumsum(~contiguous).astype(np.int64)
    counts = pd.Series(segments).value_counts().to_dict()
    keep = np.asarray([counts[int(item)] >= 2 for item in segments])
    if keep.sum() < 2:
        raise ValueError("SSL has no normal contiguous segment with at least two rows")
    return features[keep], segments[keep], row_ids[keep]


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    grid: Sequence[float],
    *,
    metadata: pd.DataFrame | None = None,
    group_weights: Mapping[Any, float] | None = None,
    group_columns: Sequence[str] = ("station", "layer"),
) -> tuple[float, float]:
    """Select on inner validation only; ties favor 0.5 then specificity."""

    truth = np.asarray(labels, dtype=np.int8)
    probability = np.asarray(probabilities, dtype=float)
    if truth.shape != probability.shape or truth.ndim != 1:
        raise ValueError("labels and probabilities must be equal one-dimensional arrays")
    if not np.isfinite(probability).all():
        raise ValueError("probabilities must be finite")
    if group_weights is not None and metadata is None:
        raise ValueError("metadata is required with group_weights")
    rows = []
    for threshold in grid:
        prediction = probability >= threshold
        score = (
            micro_f1(truth, prediction)
            if group_weights is None
            else weighted_group_f1(
                truth,
                prediction,
                metadata,
                group_weights,
                group_columns=group_columns,
            )
        )
        rows.append((score, float(threshold)))
    score, threshold = max(rows, key=lambda item: (item[0], -abs(item[1] - 0.5), item[1]))
    return threshold, score


def _train_config(
    experiment: SequenceExperimentConfig,
    architecture: Architecture,
    *,
    seed: int,
    screen: bool,
) -> SequenceTrainConfig:
    smoke = experiment.smoke
    epochs = experiment.screen_max_epochs if screen else experiment.final_max_epochs
    window = experiment.window_steps
    stride = experiment.stride_steps
    if smoke:
        epochs = min(epochs, experiment.smoke_max_epochs)
        window = min(window, experiment.smoke_window_steps)
        stride = min(stride, max(1, window // 2))
    return SequenceTrainConfig(
        architecture=architecture,
        window_steps=window,
        stride_steps=stride,
        batch_size=experiment.batch_size,
        max_epochs=epochs,
        patience=min(experiment.patience, epochs),
        learning_rate=experiment.learning_rate,
        weight_decay=experiment.weight_decay,
        negative_ratio=experiment.negative_ratio,
        auxiliary_weight=experiment.auxiliary_weight,
        use_bfloat16=experiment.use_bfloat16,
        seed=seed,
    )


def _assert_fitted_on_train_only(
    result: SequenceTrainingResult,
    training_features: np.ndarray,
) -> None:
    expected_center, expected_scale = robust_fit(training_features)
    if not np.allclose(result.center, expected_center, rtol=1.0e-5, atol=1.0e-5):
        raise RuntimeError("sequence center was not fit on the supplied fold training rows")
    if not np.allclose(result.scale, expected_scale, rtol=1.0e-5, atol=1.0e-5):
        raise RuntimeError("sequence scale was not fit on the supplied fold training rows")


def _append_ssl_embeddings(
    partition: _Partition,
    *,
    ssl_result: Any,
    center: np.ndarray,
    scale: np.ndarray,
    window_steps: int,
    stride_steps: int,
    batch_size: int,
    device: str | None,
    extractor: Callable[..., np.ndarray],
) -> _Partition:
    normalized = robust_transform(partition.features, center, scale)
    segment_counts = pd.Series(partition.segments).value_counts().to_dict()
    singleton = np.asarray([segment_counts[item] == 1 for item in partition.segments], dtype=bool)
    embeddings: np.ndarray | None = None
    if (~singleton).any():
        multi_embeddings = extractor(
            ssl_result,
            normalized[~singleton],
            partition.segments[~singleton],
            window_steps=window_steps,
            stride_steps=stride_steps,
            batch_size=batch_size,
            device=device,
        )
        embeddings = np.zeros((len(partition.indices), multi_embeddings.shape[1]), dtype=np.float32)
        embeddings[~singleton] = multi_embeddings
    if singleton.any():
        # The SSL dataset requires two rows. Duplicate each isolated observation
        # inside its own artificial segment, never across the real data gap.
        duplicated = np.repeat(normalized[singleton], 2, axis=0)
        duplicate_segments = np.repeat(np.arange(singleton.sum()), 2)
        singleton_embeddings = extractor(
            ssl_result,
            duplicated,
            duplicate_segments,
            window_steps=max(2, min(window_steps, 2)),
            stride_steps=1,
            batch_size=batch_size,
            device=device,
        )[::2]
        if embeddings is None:
            embeddings = np.zeros(
                (len(partition.indices), singleton_embeddings.shape[1]), dtype=np.float32
            )
        if embeddings.shape[1] != singleton_embeddings.shape[1]:
            raise RuntimeError("SSL embedding dimensions changed across segment fallbacks")
        embeddings[singleton] = singleton_embeddings
    if embeddings is None or embeddings.shape[0] != len(partition.indices):
        raise RuntimeError("SSL embeddings are not row aligned")
    return replace(
        partition,
        features=np.column_stack([partition.features, embeddings]).astype(np.float32),
    )


def _checkpoint_record(
    *,
    checkpoint: Path,
    architecture: str,
    fold: str,
    configuration_index: int | None,
    seed: int,
    inner_train_hash: str,
    inner_validation_hash: str,
    model_parameters: Mapping[str, Any],
    training_parameters: Mapping[str, Any],
) -> CheckpointRecord:
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise RuntimeError(f"checkpoint was not created: {checkpoint}")
    record = CheckpointRecord(
        contract_version=CHECKPOINT_CONTRACT_VERSION,
        architecture=architecture,
        fold=fold,
        configuration_index=configuration_index,
        seed=seed,
        path=str(checkpoint),
        sha256=_file_digest(checkpoint),
        bytes=checkpoint.stat().st_size,
        inner_train_rows_sha256=inner_train_hash,
        inner_validation_rows_sha256=inner_validation_hash,
        model_parameters=_json_safe(model_parameters),
        training_parameters=_json_safe(training_parameters),
    )
    _write_json(checkpoint.with_suffix(checkpoint.suffix + ".json"), asdict(record))
    return record


def run_sequence_experiments(
    frame: pd.DataFrame,
    bundle: FeatureBundle,
    *,
    pipeline_config: P1QCConfig | None = None,
    experiment_config: SequenceExperimentConfig | None = None,
    folds: Sequence[Fold] | None = None,
    group_weights: Mapping[Any, float] | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    trainer: Callable[..., SequenceTrainingResult] = train_sequence_model,
    predictor: Callable[..., tuple[np.ndarray, np.ndarray | None]] = predict_sequence,
    checkpoint_saver: Callable[
        [SequenceTrainingResult, str | Path], Path
    ] = save_sequence_checkpoint,
    ssl_trainer: Callable[..., Any] = train_masked_reconstruction,
    ssl_extractor: Callable[..., np.ndarray] = extract_ssl_embeddings,
    ssl_checkpoint_saver: Callable[[Any, str | Path], Path] = save_ssl_checkpoint,
) -> SequenceExperimentResult:
    """Run nested search and one-shot outer evaluation for both architectures.

    Normal mode requires the fixed three rolling-origin folds.  Smoke mode is
    intentionally bounded to a configurable prefix and is the only mode that
    accepts fewer outer folds for unit/integration tests.
    """

    config = pipeline_config or P1QCConfig()
    experiment = experiment_config or SequenceExperimentConfig(causal=config.mode == "causal")
    if len(frame) != len(bundle.frame):
        raise ValueError("frame and feature bundle row counts differ")
    if not frame.index.equals(bundle.frame.index):
        raise ValueError("frame and feature bundle indices differ")
    required = {"station", "layer", "time", "label", "anomaly_type"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"missing sequence experiment columns: {missing}")
    observed_mode = bundle.frame.attrs.get("feature_mode")
    expected_mode = "causal" if experiment.causal else "offline"
    if observed_mode is not None and observed_mode != expected_mode:
        raise ValueError(
            f"feature bundle mode {observed_mode!r} does not match {expected_mode!r} experiment"
        )
    numeric = bundle.numeric_matrix(dtype=np.float32)
    auxiliary = (
        parse_anomaly_types(frame["anomaly_type"], strict=True)
        .loc[:, list(ANOMALY_TYPES)]
        .to_numpy(dtype=np.float32)
    )
    outer = list(
        folds
        if folds is not None
        else outer_folds(
            frame,
            config=config.splits,
            cadence_minutes=config.data.cadence_minutes,
            group_columns=config.data.group_columns,
        )
    )
    if experiment.smoke:
        outer = outer[: experiment.smoke_fold_limit]
    else:
        _validate_fixed_outer_folds(outer)
    if not outer:
        raise ValueError("no outer folds were supplied")
    all_outer = np.concatenate([item.val_idx for item in outer])
    if len(np.unique(all_outer)) != len(all_outer):
        raise ValueError("outer validation folds overlap")

    artifact_root = None if output_dir is None else Path(output_dir).expanduser().resolve()
    if artifact_root is not None:
        artifact_root.mkdir(parents=True, exist_ok=True)
    fold_results: list[FoldSequenceResult] = []
    checkpoints: list[CheckpointRecord] = []
    oof_parts: dict[str, list[pd.DataFrame]] = {
        architecture: [] for architecture in experiment.architectures
    }

    for outer_fold in outer:
        inner = build_inner_fold(
            frame,
            outer_fold,
            validation_days=experiment.inner_validation_days,
            purge_days=experiment.purge_days,
            cadence_minutes=config.data.cadence_minutes,
            group_columns=config.data.group_columns,
        )
        train_part = _partition(
            frame,
            numeric,
            auxiliary,
            inner.train_idx,
            cadence_minutes=config.data.cadence_minutes,
            group_columns=config.data.group_columns,
        )
        inner_validation_part = _partition(
            frame,
            numeric,
            auxiliary,
            inner.val_idx,
            cadence_minutes=config.data.cadence_minutes,
            group_columns=config.data.group_columns,
        )
        outer_validation_part = _partition(
            frame,
            numeric,
            auxiliary,
            outer_fold.val_idx,
            cadence_minutes=config.data.cadence_minutes,
            group_columns=config.data.group_columns,
        )
        assert_fold_local_rows(train_part.indices, inner_validation_part.indices)
        assert_fold_local_rows(train_part.indices, outer_validation_part.indices)
        assert_fold_local_rows(inner_validation_part.indices, outer_validation_part.indices)
        train_hash = _index_digest(train_part.indices)
        inner_validation_hash = _index_digest(inner_validation_part.indices)
        outer_validation_hash = _index_digest(outer_validation_part.indices)

        ssl_record: dict[str, Any] | None = None
        if experiment.use_ssl:
            ssl_center, ssl_scale = robust_fit(train_part.features)
            train_ssl_features, train_ssl_segments, train_ssl_ids = _normal_ssl_partition(
                train_part
            )
            validation_ssl_features, validation_ssl_segments, validation_ssl_ids = (
                _normal_ssl_partition(inner_validation_part)
            )
            train_ssl_features = robust_transform(train_ssl_features, ssl_center, ssl_scale)
            validation_ssl_features = robust_transform(
                validation_ssl_features, ssl_center, ssl_scale
            )
            ssl_epochs = experiment.ssl_max_epochs
            ssl_window = experiment.ssl_window_steps
            ssl_stride = experiment.ssl_stride_steps
            if experiment.smoke:
                ssl_epochs = min(ssl_epochs, experiment.smoke_max_epochs)
                ssl_window = min(ssl_window, experiment.smoke_window_steps)
                ssl_stride = min(ssl_stride, max(1, ssl_window // 2))
            ssl_train_config = SSLTrainConfig(
                window_steps=ssl_window,
                stride_steps=ssl_stride,
                mask_fraction=experiment.ssl_mask_fraction,
                mask_block_steps=min(12, ssl_window),
                batch_size=experiment.batch_size,
                max_epochs=ssl_epochs,
                patience=min(experiment.ssl_patience, ssl_epochs),
                learning_rate=experiment.learning_rate,
                weight_decay=experiment.weight_decay,
                use_bfloat16=experiment.use_bfloat16,
                seed=experiment.final_seeds[0],
            )
            ssl_model_config = SSLModelConfig(
                input_dim=train_ssl_features.shape[1],
                channels=experiment.ssl_channels,
                causal=experiment.causal,
            )
            ssl_result = ssl_trainer(
                train_ssl_features,
                train_ssl_segments,
                train_ssl_ids,
                validation_features=validation_ssl_features,
                validation_segment_ids=validation_ssl_segments,
                validation_row_ids=validation_ssl_ids,
                model_config=ssl_model_config,
                train_config=ssl_train_config,
                device=device,
            )
            assert_fold_local_rows(ssl_result.train_row_ids, inner_validation_part.indices)
            assert_fold_local_rows(ssl_result.train_row_ids, outer_validation_part.indices)

            embedding_arguments = {
                "ssl_result": ssl_result,
                "center": ssl_center,
                "scale": ssl_scale,
                "window_steps": ssl_window,
                "stride_steps": ssl_stride,
                "batch_size": experiment.batch_size,
                "device": device,
                "extractor": ssl_extractor,
            }
            train_part = _append_ssl_embeddings(train_part, **embedding_arguments)
            inner_validation_part = _append_ssl_embeddings(
                inner_validation_part, **embedding_arguments
            )
            outer_validation_part = _append_ssl_embeddings(
                outer_validation_part, **embedding_arguments
            )
            ssl_record = {
                "enabled": True,
                "training_normal_rows": len(train_ssl_ids),
                "validation_normal_rows": len(validation_ssl_ids),
                "training_rows_sha256": _index_digest(train_ssl_ids),
                "validation_rows_sha256": _index_digest(validation_ssl_ids),
                "normalizer_center_sha256": sha256(ssl_center.tobytes()).hexdigest(),
                "normalizer_scale_sha256": sha256(ssl_scale.tobytes()).hexdigest(),
                "embedding_dim": int(train_part.features.shape[1] - numeric.shape[1]),
                "best_epoch": int(ssl_result.best_epoch),
                "best_validation_loss": float(ssl_result.best_validation_loss),
                "checkpoint": None,
            }
            if artifact_root is not None:
                checkpoint = ssl_checkpoint_saver(
                    ssl_result,
                    artifact_root / "checkpoints" / "ssl" / f"{outer_fold.name}.pt",
                )
                ssl_checkpoint = _checkpoint_record(
                    checkpoint=Path(checkpoint),
                    architecture="ssl_masked_tcn",
                    fold=outer_fold.name,
                    configuration_index=None,
                    seed=experiment.final_seeds[0],
                    inner_train_hash=train_hash,
                    inner_validation_hash=inner_validation_hash,
                    model_parameters=asdict(ssl_model_config),
                    training_parameters=asdict(ssl_train_config),
                )
                checkpoints.append(ssl_checkpoint)
                ssl_record["checkpoint"] = asdict(ssl_checkpoint)

        architecture_results: dict[str, dict[str, Any]] = {}
        for architecture in experiment.architectures:
            settings = list(search_space(architecture, causal=experiment.causal))
            if experiment.smoke:
                settings = settings[: experiment.smoke_config_limit]
            trials: list[TrialRecord] = []
            for configuration_index, parameters in enumerate(settings):
                screen_config = _train_config(
                    experiment,
                    architecture,
                    seed=experiment.final_seeds[0],
                    screen=True,
                )
                trained = trainer(
                    train_part.features,
                    train_part.labels,
                    train_part.auxiliary,
                    train_part.segments,
                    inner_validation_part.features,
                    inner_validation_part.labels,
                    inner_validation_part.auxiliary,
                    inner_validation_part.segments,
                    config=screen_config,
                    model_parameters=parameters,
                    device=device,
                )
                _assert_fitted_on_train_only(trained, train_part.features)
                probability, _ = predictor(
                    trained,
                    inner_validation_part.features,
                    inner_validation_part.segments,
                    window_steps=screen_config.window_steps,
                    stride_steps=screen_config.stride_steps,
                    batch_size=screen_config.batch_size,
                    device=device,
                )
                threshold, score = select_threshold(
                    inner_validation_part.labels,
                    probability,
                    experiment.threshold_grid,
                    metadata=inner_validation_part.metadata,
                    group_weights=group_weights,
                    group_columns=config.metrics.group_columns,
                )
                trials.append(
                    TrialRecord(
                        architecture=architecture,
                        configuration_index=configuration_index,
                        parameters=parameters,
                        seed=screen_config.seed,
                        threshold=threshold,
                        inner_f1=score,
                        best_epoch=int(trained.best_epoch),
                        best_validation_loss=float(trained.best_validation_loss),
                    )
                )
            ranked = sorted(
                trials,
                key=lambda item: (
                    -item.inner_f1,
                    item.best_validation_loss,
                    item.configuration_index,
                ),
            )
            selected_trials = ranked[: min(experiment.top_configurations, len(ranked))]
            seeds = experiment.final_seeds
            if experiment.smoke:
                seeds = seeds[: experiment.smoke_seed_limit]
            inner_probability_parts: list[np.ndarray] = []
            outer_probability_parts: list[np.ndarray] = []
            final_members: list[dict[str, Any]] = []
            for selected in selected_trials:
                for seed in seeds:
                    final_config = _train_config(experiment, architecture, seed=seed, screen=False)
                    trained = trainer(
                        train_part.features,
                        train_part.labels,
                        train_part.auxiliary,
                        train_part.segments,
                        inner_validation_part.features,
                        inner_validation_part.labels,
                        inner_validation_part.auxiliary,
                        inner_validation_part.segments,
                        config=final_config,
                        model_parameters=dict(selected.parameters),
                        device=device,
                    )
                    _assert_fitted_on_train_only(trained, train_part.features)
                    inner_probability, _ = predictor(
                        trained,
                        inner_validation_part.features,
                        inner_validation_part.segments,
                        window_steps=final_config.window_steps,
                        stride_steps=final_config.stride_steps,
                        batch_size=final_config.batch_size,
                        device=device,
                    )
                    outer_probability, _ = predictor(
                        trained,
                        outer_validation_part.features,
                        outer_validation_part.segments,
                        window_steps=final_config.window_steps,
                        stride_steps=final_config.stride_steps,
                        batch_size=final_config.batch_size,
                        device=device,
                    )
                    inner_probability_parts.append(np.asarray(inner_probability, dtype=np.float32))
                    outer_probability_parts.append(np.asarray(outer_probability, dtype=np.float32))
                    member: dict[str, Any] = {
                        "configuration_index": selected.configuration_index,
                        "seed": seed,
                        "best_epoch": int(trained.best_epoch),
                        "best_validation_loss": float(trained.best_validation_loss),
                        "checkpoint": None,
                    }
                    if artifact_root is not None:
                        checkpoint = checkpoint_saver(
                            trained,
                            artifact_root
                            / "checkpoints"
                            / architecture
                            / outer_fold.name
                            / f"config_{selected.configuration_index:02d}_seed_{seed}.pt",
                        )
                        checkpoint_record = _checkpoint_record(
                            checkpoint=Path(checkpoint),
                            architecture=architecture,
                            fold=outer_fold.name,
                            configuration_index=selected.configuration_index,
                            seed=seed,
                            inner_train_hash=train_hash,
                            inner_validation_hash=inner_validation_hash,
                            model_parameters=selected.parameters,
                            training_parameters=asdict(final_config),
                        )
                        checkpoints.append(checkpoint_record)
                        member["checkpoint"] = asdict(checkpoint_record)
                    final_members.append(member)
            inner_ensemble = np.mean(np.stack(inner_probability_parts), axis=0)
            outer_ensemble = np.mean(np.stack(outer_probability_parts), axis=0)
            threshold, inner_f1 = select_threshold(
                inner_validation_part.labels,
                inner_ensemble,
                experiment.threshold_grid,
                metadata=inner_validation_part.metadata,
                group_weights=group_weights,
                group_columns=config.metrics.group_columns,
            )
            prediction = (outer_ensemble >= threshold).astype(np.int8)
            report = evaluate_predictions(
                outer_validation_part.labels,
                prediction,
                outer_validation_part.metadata,
                group_columns=config.metrics.group_columns,
                group_weights=group_weights,
                anomaly_type=frame.iloc[outer_validation_part.indices]["anomaly_type"].reset_index(
                    drop=True
                ),
                cadence_minutes=config.metrics.event_cadence_minutes,
                event_min_iou=config.metrics.event_min_iou,
            )
            architecture_results[architecture] = {
                "screen_trials": [asdict(item) for item in trials],
                "selected_configuration_indices": [
                    item.configuration_index for item in selected_trials
                ],
                "selection_basis": "inner_validation_f1",
                "selection_metric": (
                    "test-share-weighted_group_f1" if group_weights is not None else "micro_f1"
                ),
                "ensemble_members": final_members,
                "inner_ensemble_threshold": threshold,
                "inner_ensemble_f1": inner_f1,
                "outer_evaluation": report.to_dict(),
            }
            oof_parts[architecture].append(
                pd.DataFrame(
                    {
                        "row_index": outer_validation_part.indices,
                        "fold": outer_fold.name,
                        "label": outer_validation_part.labels,
                        "probability": outer_ensemble,
                        "prediction": prediction,
                    }
                )
            )
        fold_results.append(
            FoldSequenceResult(
                fold=outer_fold.name,
                train_rows=len(outer_fold.train_idx),
                outer_validation_rows=len(outer_fold.val_idx),
                inner_train_rows=len(inner.train_idx),
                inner_validation_rows=len(inner.val_idx),
                inner_train_rows_sha256=train_hash,
                inner_validation_rows_sha256=inner_validation_hash,
                outer_validation_rows_sha256=outer_validation_hash,
                inner_train_end=inner.train_end.isoformat(),
                inner_validation_start=inner.val_start.isoformat(),
                inner_validation_end=inner.val_end.isoformat(),
                ssl=ssl_record,
                architectures=architecture_results,
            )
        )

    oof = {
        architecture: pd.concat(parts, ignore_index=True)
        .sort_values("row_index", kind="mergesort")
        .reset_index(drop=True)
        for architecture, parts in oof_parts.items()
    }
    comparison: list[dict[str, Any]] = []
    for architecture in experiment.architectures:
        inner_scores = np.asarray(
            [fold.architectures[architecture]["inner_ensemble_f1"] for fold in fold_results],
            dtype=float,
        )
        table = oof[architecture]
        outer_micro = micro_f1(table["label"], table["prediction"])
        outer_weighted = outer_micro
        if group_weights is not None:
            outer_metadata = frame.iloc[table["row_index"].to_numpy(dtype=np.int64)].reset_index(
                drop=True
            )
            outer_weighted = weighted_group_f1(
                table["label"],
                table["prediction"],
                outer_metadata,
                group_weights,
                group_columns=config.metrics.group_columns,
            )
        comparison.append(
            {
                "architecture": architecture,
                "selection_mean_inner_f1": float(inner_scores.mean()),
                "selection_std_inner_f1": float(inner_scores.std(ddof=0)),
                "diagnostic_outer_micro_f1": outer_micro,
                "diagnostic_outer_weighted_f1": outer_weighted,
                "outer_rows": len(table),
            }
        )
    winner = sorted(
        comparison,
        key=lambda row: (
            -row["selection_mean_inner_f1"],
            row["selection_std_inner_f1"],
            row["architecture"],
        ),
    )[0]
    winning_architecture = winner["architecture"]
    configuration_votes: dict[int, int] = {}
    configuration_scores: dict[int, list[float]] = {}
    deployment_thresholds: list[float] = []
    for fold in fold_results:
        architecture_result = fold.architectures[winning_architecture]
        deployment_thresholds.append(architecture_result["inner_ensemble_threshold"])
        trial_by_index = {
            int(item["configuration_index"]): float(item["inner_f1"])
            for item in architecture_result["screen_trials"]
        }
        for index in architecture_result["selected_configuration_indices"]:
            configuration_votes[index] = configuration_votes.get(index, 0) + 1
            configuration_scores.setdefault(index, []).append(trial_by_index[index])
    deployment_configuration = sorted(
        configuration_votes,
        key=lambda index: (
            -configuration_votes[index],
            -float(np.mean(configuration_scores[index])),
            index,
        ),
    )[0]
    deployment_space = search_space(winning_architecture, causal=experiment.causal)
    deployment_seeds = experiment.final_seeds
    if experiment.smoke:
        deployment_seeds = deployment_seeds[: experiment.smoke_seed_limit]
    result = SequenceExperimentResult(
        contract_version=RESULT_CONTRACT_VERSION,
        experiment={
            **asdict(experiment),
            "selection_uses_outer_labels": False,
            "outer_labels_used_once_for_diagnostics": True,
            "inner_selection_metric": (
                "test-share-weighted_group_f1" if group_weights is not None else "micro_f1"
            ),
            "fold_count": len(outer),
            "feature_mode": expected_mode,
            "numeric_features": list(bundle.numeric_columns),
        },
        folds=fold_results,
        comparison=comparison,
        selection={
            "architecture": winning_architecture,
            "configuration_index": deployment_configuration,
            "model_parameters": deployment_space[deployment_configuration],
            "seeds": list(deployment_seeds),
            "threshold_median": float(np.median(deployment_thresholds)),
            "criterion": "maximum mean inner-validation ensemble F1 across outer folds",
            "configuration_criterion": (
                "inner-selected fold vote, then mean inner-validation F1, then index"
            ),
            "outer_metrics_used_for_selection": False,
        },
        checkpoints=checkpoints,
        oof_predictions=oof,
    )
    validate_result_contract(result)
    if artifact_root is not None:
        save_result_contract(result, artifact_root)
    return result


def validate_result_contract(result: SequenceExperimentResult) -> None:
    if result.contract_version != RESULT_CONTRACT_VERSION:
        raise ValueError("unsupported sequence result contract version")
    architectures = tuple(result.experiment["architectures"])
    if result.selection.get("architecture") not in architectures:
        raise ValueError("selected architecture is absent from the experiment")
    if bool(result.experiment.get("selection_uses_outer_labels")):
        raise ValueError("outer labels cannot be used for model selection")
    for architecture in architectures:
        table = result.oof_predictions.get(architecture)
        if table is None or table.empty:
            raise ValueError(f"missing OOF predictions for {architecture}")
        if table["row_index"].duplicated().any():
            raise ValueError(f"duplicate OOF rows for {architecture}")
        if not np.isfinite(table["probability"]).all():
            raise ValueError(f"non-finite OOF probabilities for {architecture}")
        if not table["prediction"].isin([0, 1]).all():
            raise ValueError(f"invalid OOF labels for {architecture}")


def save_result_contract(
    result: SequenceExperimentResult,
    output_dir: str | Path,
) -> Path:
    """Persist sanitized JSON plus compact row-position OOF arrays."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Mapping[str, Any]] = {}
    for architecture, table in result.oof_predictions.items():
        destination = root / f"oof_{architecture}.npz"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                row_index=table["row_index"].to_numpy(dtype=np.int64),
                label=table["label"].to_numpy(dtype=np.int8),
                probability=table["probability"].to_numpy(dtype=np.float32),
                prediction=table["prediction"].to_numpy(dtype=np.int8),
            )
        temporary.replace(destination)
        artifacts[f"oof_{architecture}"] = {
            "path": str(destination),
            "sha256": _file_digest(destination),
            "bytes": destination.stat().st_size,
            "rows": len(table),
        }
    result.artifacts = artifacts
    destination = root / "sequence_experiment.json"
    _write_json(destination, result.to_dict())
    return destination


__all__ = [
    "CHECKPOINT_CONTRACT_VERSION",
    "RESULT_CONTRACT_VERSION",
    "CheckpointRecord",
    "FoldSequenceResult",
    "SequenceExperimentConfig",
    "SequenceExperimentResult",
    "TrialRecord",
    "build_inner_fold",
    "run_sequence_experiments",
    "save_result_contract",
    "search_space",
    "select_threshold",
    "validate_result_contract",
]
