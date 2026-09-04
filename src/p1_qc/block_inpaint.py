"""Leakage-safe dual-flank block inpainting features for the P1 offline QC task.

Only target temperature outside a masked block is exposed to the model.  Every
contemporaneous peer statistic is leave-one-target-layer-out, and every block
stays inside one physical 10-minute station/layer segment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCORE_COLUMNS: tuple[str, ...] = (
    "bmi_signed_residual_at_max",
    "bmi_abs_residual_max",
    "bmi_abs_slope_max",
    "bmi_fb_disagreement_at_max",
)
CONTINUOUS_COVARIATES: tuple[str, ...] = (
    "psal",
    "peer_temp_mean",
    "peer_psal_mean",
    "depth",
)
CYCLIC_COVARIATES: tuple[str, ...] = (
    "annual_sin",
    "annual_cos",
    "clock_sin",
    "clock_cos",
    "m2_sin",
    "m2_cos",
)


@dataclass(frozen=True)
class BlockInpaintConfig:
    cadence_minutes: int = 10
    mask_hours: tuple[int, ...] = (8, 24, 48, 96)
    stride_hours: int = 6
    left_flank_hours: int = 24
    right_flank_hours: int = 24
    hidden_size: int = 32
    decoder_size: int = 64
    dropout: float = 0.1
    batch_size: int = 64
    max_epochs: int = 12
    patience: int = 3
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    consistency_weight: float = 0.05
    train_windows_per_mask_length: int = 2500
    validation_fraction: float = 0.1
    use_bfloat16: bool = True
    seed: int = 20260813

    def __post_init__(self) -> None:
        if self.cadence_minutes <= 0 or 60 % self.cadence_minutes:
            raise ValueError("cadence_minutes must be a positive divisor of 60")
        if tuple(self.mask_hours) != (8, 24, 48, 96):
            raise ValueError("v1 mask_hours are frozen at 8/24/48/96 hours")
        if self.stride_hours != 6:
            raise ValueError("v1 stride_hours is frozen at 6")
        if self.left_flank_hours != 24 or self.right_flank_hours != 24:
            raise ValueError("v1 flanks are frozen at 24 hours on each side")
        if self.maximum_context_rows * self.cadence_minutes > 6 * 24 * 60:
            raise ValueError("block contract exceeds the frozen six-day dependency")
        if self.hidden_size != 32 or self.decoder_size != 64:
            raise ValueError("v1 model width is frozen at hidden=32, decoder=64")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")
        if min(self.batch_size, self.max_epochs, self.patience) < 1:
            raise ValueError("training counts must be positive")

    @property
    def rows_per_hour(self) -> int:
        return 60 // self.cadence_minutes

    @property
    def mask_rows(self) -> tuple[int, ...]:
        return tuple(value * self.rows_per_hour for value in self.mask_hours)

    @property
    def stride_rows(self) -> int:
        return self.stride_hours * self.rows_per_hour

    @property
    def left_flank_rows(self) -> int:
        return self.left_flank_hours * self.rows_per_hour

    @property
    def right_flank_rows(self) -> int:
        return self.right_flank_hours * self.rows_per_hour

    @property
    def maximum_context_rows(self) -> int:
        return self.left_flank_rows + max(self.mask_rows) + self.right_flank_rows


@dataclass(frozen=True)
class SafeDesign:
    target_temp: np.ndarray
    continuous: np.ndarray
    cyclic: np.ndarray
    continuous_columns: tuple[str, ...]
    cyclic_columns: tuple[str, ...]


@dataclass(frozen=True)
class RobustCovariateScaler:
    center: np.ndarray
    scale: np.ndarray

    def transform(self, design: SafeDesign) -> np.ndarray:
        continuous = (design.continuous - self.center) / self.scale
        missing = ~np.isfinite(continuous)
        continuous = np.where(missing, 0.0, np.clip(continuous, -20.0, 20.0))
        return np.column_stack([continuous, missing.astype(np.float32), design.cyclic]).astype(
            np.float32, copy=False
        )


@dataclass(frozen=True)
class PreparedSeries:
    global_indices: np.ndarray
    local_order: np.ndarray
    inverse_order: np.ndarray
    segment_ids: np.ndarray
    target_temp: np.ndarray
    covariates: np.ndarray
    labels: np.ndarray | None
    stations: np.ndarray
    layers: np.ndarray
    times_ns: np.ndarray

    @property
    def rows(self) -> int:
        return len(self.global_indices)


@dataclass(frozen=True)
class BlockSpec:
    start: int
    stop: int
    segment_id: int

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass
class InpaintTrainingResult:
    model: Any
    config: BlockInpaintConfig
    best_epoch: int
    best_validation_loss: float
    train_windows: int
    validation_windows: int
    windows_by_length: dict[int, int]
    device: str


@dataclass(frozen=True)
class AdditiveGate:
    coefficients: np.ndarray
    scales: np.ndarray
    l2_penalty: float
    objective: float
    iterations: int
    success: bool

    def delta(self, scores: pd.DataFrame) -> np.ndarray:
        matrix = scores.loc[:, SCORE_COLUMNS].to_numpy(dtype=np.float64)
        available = np.isfinite(matrix).all(axis=1)
        transformed = np.zeros_like(matrix)
        transformed[available] = matrix[available] / self.scales
        result = transformed @ self.coefficients
        result[~available] = 0.0
        return result


def _leave_one_out_mean(values: pd.Series, groups: list[pd.Series]) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    finite = numeric.notna()
    filled = numeric.fillna(0.0)
    grouped_sum = filled.groupby(groups, sort=False, observed=True).transform("sum")
    grouped_count = (
        finite.astype(np.int64).groupby(groups, sort=False, observed=True).transform("sum")
    )
    other_count = grouped_count - finite.astype(np.int64)
    other_sum = grouped_sum - filled
    result = (other_sum / other_count.where(other_count > 0)).to_numpy(
        dtype=np.float64,
        copy=True,
    )
    result[other_count.to_numpy() <= 0] = np.nan
    return result


def build_safe_design(frame: pd.DataFrame) -> SafeDesign:
    """Build target-safe raw inputs aligned one-to-one with ``frame`` rows."""

    required = {"station", "layer", "time", "temp", "psal", "depth"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"frame is missing required columns: {missing}")
    parsed = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    groups = [frame["station"].astype("string"), parsed]
    peer_temp = _leave_one_out_mean(frame["temp"], groups)
    peer_psal = _leave_one_out_mean(frame["psal"], groups)
    local = parsed.dt.tz_convert("Asia/Seoul")
    minute_of_day = local.dt.hour.to_numpy(dtype=np.float64) * 60.0 + local.dt.minute.to_numpy(
        dtype=np.float64
    )
    day_phase = 2.0 * np.pi * minute_of_day / (24.0 * 60.0)
    year_fraction = (
        local.dt.dayofyear.to_numpy(dtype=np.float64) - 1.0 + minute_of_day / (24.0 * 60.0)
    ) / 365.2425
    annual_phase = 2.0 * np.pi * year_fraction
    # pandas 3 preserves the source datetime resolution (often microseconds),
    # so ``astype("int64")`` is not guaranteed to be nanoseconds.  Normalize
    # explicitly before computing cadence-dependent quantities.
    parsed_ns = parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    minutes_from_epoch = parsed_ns.astype(np.float64) / 60.0e9
    m2_phase = 2.0 * np.pi * minutes_from_epoch / (12.42 * 60.0)
    continuous = np.column_stack(
        [
            pd.to_numeric(frame["psal"], errors="coerce").to_numpy(dtype=np.float64),
            peer_temp,
            peer_psal,
            pd.to_numeric(frame["depth"], errors="coerce").to_numpy(dtype=np.float64),
        ]
    )
    cyclic = np.column_stack(
        [
            np.sin(annual_phase),
            np.cos(annual_phase),
            np.sin(day_phase),
            np.cos(day_phase),
            np.sin(m2_phase),
            np.cos(m2_phase),
        ]
    ).astype(np.float32)
    return SafeDesign(
        target_temp=pd.to_numeric(frame["temp"], errors="coerce").to_numpy(dtype=np.float64),
        continuous=continuous,
        cyclic=cyclic,
        continuous_columns=CONTINUOUS_COVARIATES,
        cyclic_columns=CYCLIC_COVARIATES,
    )


def assert_target_safe_contract(frame: pd.DataFrame, design: SafeDesign) -> None:
    if len(frame) != len(design.target_temp) or len(frame) != len(design.continuous):
        raise ValueError("safe design is not row aligned")
    if design.continuous_columns != CONTINUOUS_COVARIATES:
        raise ValueError("continuous safe-design columns changed")
    if design.cyclic_columns != CYCLIC_COVARIATES:
        raise ValueError("cyclic safe-design columns changed")
    forbidden = ("target_temp", "temp_lag", "temp_diff", "temp_roll", "temp_resid")
    exposed = [*design.continuous_columns, *design.cyclic_columns]
    if any(any(token in column for token in forbidden) for column in exposed):
        raise ValueError("target-temperature-derived input escaped the mask contract")
    if not np.array_equal(
        design.target_temp,
        pd.to_numeric(frame["temp"], errors="coerce").to_numpy(dtype=np.float64),
        equal_nan=True,
    ):
        raise ValueError("target temperature side channel is not exact")


def fit_covariate_scaler(design: SafeDesign, indices: Sequence[int]) -> RobustCovariateScaler:
    rows = np.asarray(indices, dtype=np.int64)
    values = design.continuous[rows]
    center = np.nanmedian(values, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    mad = np.nanmedian(np.abs(values - center), axis=0)
    fallback = np.nanstd(values, axis=0)
    scale = np.where(np.isfinite(mad) & (mad > 1.0e-6), 1.4826 * mad, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-6), scale, 1.0)
    return RobustCovariateScaler(center=center.astype(np.float64), scale=scale.astype(np.float64))


def prepare_series(
    frame: pd.DataFrame,
    design: SafeDesign,
    scaler: RobustCovariateScaler,
    indices: Sequence[int],
    *,
    cadence_minutes: int = 10,
) -> PreparedSeries:
    rows = np.asarray(indices, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("indices must be a non-empty one-dimensional vector")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("indices contain duplicates")
    metadata = frame.iloc[rows].loc[:, ["station", "layer", "time"]].copy()
    parsed = pd.to_datetime(metadata["time"], errors="raise", utc=True, format="mixed")
    metadata["__time"] = parsed
    metadata["__local"] = np.arange(len(rows), dtype=np.int64)
    ordered = metadata.sort_values(["station", "layer", "__time", "__local"], kind="mergesort")
    order = ordered["__local"].to_numpy(dtype=np.int64)
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order), dtype=np.int64)
    sorted_rows = rows[order]
    station = ordered["station"].astype(str).to_numpy()
    layer = pd.to_numeric(ordered["layer"], errors="raise").to_numpy(dtype=np.int64)
    time_ns = ordered["__time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    contiguous = np.r_[
        False,
        (station[1:] == station[:-1])
        & (layer[1:] == layer[:-1])
        & ((time_ns[1:] - time_ns[:-1]) == cadence_minutes * 60 * 1_000_000_000),
    ]
    segment_ids = np.cumsum(~contiguous).astype(np.int64) - 1
    labels = (
        pd.to_numeric(frame.iloc[sorted_rows]["label"], errors="raise").to_numpy(dtype=np.int8)
        if "label" in frame.columns
        else None
    )
    subset_design = SafeDesign(
        target_temp=design.target_temp[sorted_rows],
        continuous=design.continuous[sorted_rows],
        cyclic=design.cyclic[sorted_rows],
        continuous_columns=design.continuous_columns,
        cyclic_columns=design.cyclic_columns,
    )
    return PreparedSeries(
        global_indices=sorted_rows,
        local_order=order,
        inverse_order=inverse,
        segment_ids=segment_ids,
        target_temp=design.target_temp[sorted_rows].astype(np.float32),
        covariates=scaler.transform(subset_design),
        labels=labels,
        stations=station,
        layers=layer,
        times_ns=time_ns,
    )


def _segment_bounds(segment_ids: np.ndarray) -> list[tuple[int, int, int]]:
    starts = np.r_[0, np.flatnonzero(segment_ids[1:] != segment_ids[:-1]) + 1]
    stops = np.r_[starts[1:], len(segment_ids)]
    return [
        (int(start), int(stop), int(segment_ids[start]))
        for start, stop in zip(starts, stops, strict=True)
    ]


def enumerate_blocks(
    prepared: PreparedSeries,
    config: BlockInpaintConfig,
    *,
    normal_only: bool,
    cap_per_length: int | None = None,
    seed: int | None = None,
) -> list[BlockSpec]:
    if normal_only and prepared.labels is None:
        raise ValueError("normal-only enumeration requires labels")
    by_length: dict[int, list[BlockSpec]] = {length: [] for length in config.mask_rows}
    for lower, upper, segment_id in _segment_bounds(prepared.segment_ids):
        if normal_only:
            abnormal = (prepared.labels[lower:upper] != 0).astype(np.int64)
            prefix = np.r_[0, np.cumsum(abnormal)]
        for length in config.mask_rows:
            first = lower + config.left_flank_rows
            last = upper - config.right_flank_rows - length
            if last < first:
                continue
            starts = list(range(first, last + 1, config.stride_rows))
            if not starts or starts[-1] != last:
                starts.append(last)
            for start in starts:
                stop = start + length
                if normal_only:
                    window_lower = start - config.left_flank_rows - lower
                    window_upper = stop + config.right_flank_rows - lower
                    if prefix[window_upper] != prefix[window_lower]:
                        continue
                by_length[length].append(BlockSpec(start, stop, segment_id))
    rng = np.random.default_rng(config.seed if seed is None else seed)
    result: list[BlockSpec] = []
    for length in config.mask_rows:
        candidates = by_length[length]
        if cap_per_length is not None and len(candidates) > cap_per_length:
            chosen = np.sort(rng.choice(len(candidates), size=cap_per_length, replace=False))
            candidates = [candidates[int(position)] for position in chosen]
        result.extend(candidates)
    result.sort(key=lambda item: (item.segment_id, item.start, item.stop))
    return result


def coverage_audit(
    prepared: PreparedSeries, config: BlockInpaintConfig
) -> tuple[np.ndarray, dict[str, Any]]:
    specs = enumerate_blocks(prepared, config, normal_only=False)
    covered = np.zeros(prepared.rows, dtype=bool)
    by_length: dict[str, int] = {}
    for spec in specs:
        covered[spec.start : spec.stop] = True
        key = str(spec.length)
        by_length[key] = by_length.get(key, 0) + 1
    return covered[prepared.inverse_order], {
        "rows": prepared.rows,
        "covered_rows": int(covered.sum()),
        "covered_fraction": float(covered.mean()) if prepared.rows else 0.0,
        "blocks": len(specs),
        "blocks_by_mask_rows": by_length,
        "segments": int(prepared.segment_ids.max() + 1),
        "maximum_context_rows": config.maximum_context_rows,
        "maximum_context_hours": config.maximum_context_rows * config.cadence_minutes / 60.0,
    }


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1.0e-4:
        scale = float(np.std(values))
    return center, max(scale if np.isfinite(scale) else 1.0, 1.0e-3)


def materialize_example(
    prepared: PreparedSeries, spec: BlockSpec, config: BlockInpaintConfig
) -> dict[str, np.ndarray | float | int]:
    left_start = spec.start - config.left_flank_rows
    right_stop = spec.stop + config.right_flank_rows
    if left_start < 0 or right_stop > prepared.rows:
        raise ValueError("block flank exceeds prepared rows")
    segment = prepared.segment_ids[left_start:right_stop]
    if not np.all(segment == spec.segment_id):
        raise ValueError("block crosses a physical segment")
    left_temp = prepared.target_temp[left_start : spec.start].astype(np.float64)
    right_temp = prepared.target_temp[spec.stop : right_stop].astype(np.float64)
    if not np.isfinite(left_temp).all() or not np.isfinite(right_temp).all():
        raise ValueError("target flank contains non-finite temperature")
    left_center, left_scale = _robust_location_scale(left_temp)
    right_center, right_scale = _robust_location_scale(right_temp)
    left_context = np.column_stack(
        [
            ((left_temp - left_center) / left_scale).astype(np.float32),
            np.ones(len(left_temp), dtype=np.float32),
            prepared.covariates[left_start : spec.start],
        ]
    ).astype(np.float32)
    right_context = (
        np.column_stack(
            [
                ((right_temp - right_center) / right_scale).astype(np.float32),
                np.ones(len(right_temp), dtype=np.float32),
                prepared.covariates[spec.stop : right_stop],
            ]
        )
        .astype(np.float32)[::-1]
        .copy()
    )
    # No target temperature or target-derived value from [start, stop) enters
    # either context or target_covariates.  The target is returned separately
    # and is consumed only by the loss/scoring code.
    return {
        "left": left_context,
        "right": right_context,
        "target_covariates": prepared.covariates[spec.start : spec.stop].copy(),
        "target": prepared.target_temp[spec.start : spec.stop].copy(),
        "left_center": left_center,
        "left_scale": left_scale,
        "right_center": right_center,
        "right_scale": right_scale,
        "length": spec.length,
        "start": spec.start,
        "stop": spec.stop,
    }


def assert_mask_invariance(
    prepared: PreparedSeries, spec: BlockSpec, config: BlockInpaintConfig
) -> None:
    original = materialize_example(prepared, spec, config)
    changed_temp = prepared.target_temp.copy()
    changed_temp[spec.start : spec.stop] += np.linspace(100.0, 200.0, spec.length).astype(
        np.float32
    )
    changed = PreparedSeries(**{**asdict(prepared), "target_temp": changed_temp})
    mutated = materialize_example(changed, spec, config)
    for key in ("left", "right", "target_covariates"):
        if not np.array_equal(np.asarray(original[key]), np.asarray(mutated[key]), equal_nan=True):
            raise ValueError(f"masked target leaked into model input: {key}")


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for block inpainting") from exc
    return torch, nn


def build_model(context_dim: int, covariate_dim: int, config: BlockInpaintConfig) -> Any:
    torch, nn = _require_torch()

    class DualFlankInpaint(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context_dim = context_dim
            self.covariate_dim = covariate_dim
            self.encoder = nn.GRU(
                input_size=context_dim,
                hidden_size=config.hidden_size,
                num_layers=1,
                batch_first=True,
            )
            decoder_input = config.hidden_size + covariate_dim + 6

            def decoder() -> Any:
                return nn.Sequential(
                    nn.Linear(decoder_input, config.decoder_size),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.decoder_size, 1),
                )

            self.forward_decoder = decoder()
            self.backward_decoder = decoder()

        def forward(
            self,
            left: Any,
            right: Any,
            target_covariates: Any,
            lengths: Any,
        ) -> tuple[Any, Any]:
            _, left_hidden = self.encoder(left)
            _, right_hidden = self.encoder(right)
            batch, width, _ = target_covariates.shape
            position = torch.arange(width, device=target_covariates.device).view(1, -1)
            denominator = torch.clamp(lengths.view(-1, 1) - 1, min=1)
            relative = position / denominator
            valid = position < lengths.view(-1, 1)
            forward_horizon = (position + 1) / float(max(config.mask_rows))
            backward_horizon = torch.clamp(lengths.view(-1, 1) - position, min=0) / float(
                max(config.mask_rows)
            )
            length_feature = torch.log1p(lengths.float()).view(-1, 1) / np.log1p(
                max(config.mask_rows)
            )
            length_feature = length_feature.expand(-1, width)
            shared = torch.stack(
                [
                    relative,
                    1.0 - relative,
                    forward_horizon.expand(batch, -1),
                    backward_horizon,
                    torch.sin(torch.pi * relative),
                    length_feature,
                ],
                dim=-1,
            )
            left_state = left_hidden[-1].unsqueeze(1).expand(-1, width, -1)
            right_state = right_hidden[-1].unsqueeze(1).expand(-1, width, -1)
            forward = self.forward_decoder(
                torch.cat([left_state, target_covariates, shared], dim=-1)
            ).squeeze(-1)
            reverse_shared = shared.clone()
            reverse_shared[..., 0], reverse_shared[..., 1] = (
                shared[..., 1],
                shared[..., 0],
            )
            backward = self.backward_decoder(
                torch.cat([right_state, target_covariates, reverse_shared], dim=-1)
            ).squeeze(-1)
            return forward.masked_fill(~valid, 0.0), backward.masked_fill(~valid, 0.0)

    return DualFlankInpaint()


class _BlockDataset:
    def __init__(
        self, prepared: PreparedSeries, specs: Sequence[BlockSpec], config: BlockInpaintConfig
    ) -> None:
        self.prepared = prepared
        self.specs = tuple(specs)
        self.config = config

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return materialize_example(self.prepared, self.specs[index], self.config)


def _collate(examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    torch, _ = _require_torch()
    lengths = np.asarray([int(example["length"]) for example in examples], dtype=np.int64)
    maximum = int(lengths.max())
    covariate_dim = np.asarray(examples[0]["target_covariates"]).shape[1]
    target_covariates = np.zeros((len(examples), maximum, covariate_dim), dtype=np.float32)
    target = np.zeros((len(examples), maximum), dtype=np.float32)
    valid = np.zeros((len(examples), maximum), dtype=bool)
    for row, example in enumerate(examples):
        length = int(example["length"])
        target_covariates[row, :length] = np.asarray(example["target_covariates"])
        target[row, :length] = np.asarray(example["target"])
        valid[row, :length] = True
    return {
        "left": torch.from_numpy(np.stack([np.asarray(item["left"]) for item in examples])),
        "right": torch.from_numpy(np.stack([np.asarray(item["right"]) for item in examples])),
        "target_covariates": torch.from_numpy(target_covariates),
        "target": torch.from_numpy(target),
        "valid": torch.from_numpy(valid),
        "lengths": torch.from_numpy(lengths),
        "left_center": torch.tensor(
            [item["left_center"] for item in examples], dtype=torch.float32
        ),
        "left_scale": torch.tensor([item["left_scale"] for item in examples], dtype=torch.float32),
        "right_center": torch.tensor(
            [item["right_center"] for item in examples], dtype=torch.float32
        ),
        "right_scale": torch.tensor(
            [item["right_scale"] for item in examples], dtype=torch.float32
        ),
        "starts": np.asarray([item["start"] for item in examples], dtype=np.int64),
        "stops": np.asarray([item["stop"] for item in examples], dtype=np.int64),
    }


def _loss(
    batch: dict[str, Any], forward_z: Any, backward_z: Any, config: BlockInpaintConfig
) -> Any:
    torch, _ = _require_torch()
    target = batch["target"]
    valid = batch["valid"]
    left_center = batch["left_center"].unsqueeze(1)
    left_scale = batch["left_scale"].unsqueeze(1)
    right_center = batch["right_center"].unsqueeze(1)
    right_scale = batch["right_scale"].unsqueeze(1)
    forward_target = (target - left_center) / left_scale
    backward_target = (target - right_center) / right_scale
    forward_abs = forward_z * left_scale + left_center
    backward_abs = backward_z * right_scale + right_center
    width = target.shape[1]
    position = torch.arange(width, device=target.device).view(1, -1)
    relative = position / torch.clamp(batch["lengths"].view(-1, 1) - 1, min=1)
    blended = (1.0 - relative) * forward_abs + relative * backward_abs
    common_scale = torch.maximum(left_scale, right_scale)
    huber = torch.nn.functional.smooth_l1_loss
    forward_loss = huber(forward_z[valid], forward_target[valid])
    backward_loss = huber(backward_z[valid], backward_target[valid])
    blend_loss = huber(((blended - target) / common_scale)[valid], torch.zeros_like(target[valid]))
    consistency = huber(
        ((forward_abs - backward_abs) / common_scale)[valid], torch.zeros_like(target[valid])
    )
    return (
        0.4 * forward_loss
        + 0.4 * backward_loss
        + 0.2 * blend_loss
        + config.consistency_weight * consistency
    )


def train_inpaint_model(
    prepared: PreparedSeries,
    config: BlockInpaintConfig,
    *,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> InpaintTrainingResult:
    torch, _ = _require_torch()
    from torch.utils.data import DataLoader

    specs = enumerate_blocks(
        prepared,
        config,
        normal_only=True,
        cap_per_length=config.train_windows_per_mask_length,
        seed=config.seed,
    )
    minimum = max(8, len(config.mask_rows) * 2)
    if len(specs) < minimum:
        raise ValueError(f"only {len(specs)} normal block windows are available")
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(len(specs))
    validation_count = max(1, int(round(len(specs) * config.validation_fraction)))
    validation_specs = [specs[int(index)] for index in order[:validation_count]]
    train_specs = [specs[int(index)] for index in order[validation_count:]]
    context_dim = 2 + prepared.covariates.shape[1]
    covariate_dim = prepared.covariates.shape[1]
    # Seed before construction so parameter initialization is reproducible as
    # well as the loader order and optimizer trajectory.
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = build_model(context_dim, covariate_dim, config)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.use_deterministic_algorithms(True, warn_only=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loaders = {
        "train": DataLoader(
            _BlockDataset(prepared, train_specs, config),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=_collate,
            generator=torch.Generator().manual_seed(config.seed),
        ),
        "validation": DataLoader(
            _BlockDataset(prepared, validation_specs, config),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate,
        ),
    }
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    stale = 0
    for epoch in range(config.max_epochs):
        epoch_losses: dict[str, float] = {}
        for phase in ("train", "validation"):
            training = phase == "train"
            model.train(training)
            total = 0.0
            rows = 0
            context = torch.enable_grad if training else torch.inference_mode
            with context():
                for batch in loaders[phase]:
                    for key in (
                        "left",
                        "right",
                        "target_covariates",
                        "target",
                        "valid",
                        "lengths",
                        "left_center",
                        "left_scale",
                        "right_center",
                        "right_scale",
                    ):
                        batch[key] = batch[key].to(device)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type="cuda" if device.startswith("cuda") else "cpu",
                        dtype=torch.bfloat16,
                        enabled=config.use_bfloat16,
                    ):
                        forward_z, backward_z = model(
                            batch["left"],
                            batch["right"],
                            batch["target_covariates"],
                            batch["lengths"],
                        )
                        loss = _loss(batch, forward_z, backward_z, config)
                    if training:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                    total += float(loss.detach().cpu()) * len(batch["left"])
                    rows += len(batch["left"])
            epoch_losses[phase] = total / max(rows, 1)
        if epoch_losses["validation"] < best_loss - 1.0e-6:
            best_loss = epoch_losses["validation"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("inpainting training produced no finite checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    by_length: dict[int, int] = {}
    for spec in specs:
        by_length[spec.length] = by_length.get(spec.length, 0) + 1
    result = InpaintTrainingResult(
        model=model,
        config=config,
        best_epoch=best_epoch,
        best_validation_loss=float(best_loss),
        train_windows=len(train_specs),
        validation_windows=len(validation_specs),
        windows_by_length=by_length,
        device=device,
    )
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "config": asdict(config),
                "context_dim": context_dim,
                "covariate_dim": covariate_dim,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "train_windows": len(train_specs),
                "validation_windows": len(validation_specs),
                "windows_by_length": by_length,
            },
            path,
        )
    return result


def score_inpaint_model(
    result: InpaintTrainingResult,
    prepared: PreparedSeries,
    *,
    batch_size: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    torch, _ = _require_torch()
    from torch.utils.data import DataLoader

    specs = enumerate_blocks(prepared, result.config, normal_only=False)
    signed = np.full(prepared.rows, np.nan, dtype=np.float64)
    absolute = np.full(prepared.rows, np.nan, dtype=np.float64)
    slope = np.full(prepared.rows, np.nan, dtype=np.float64)
    disagreement = np.full(prepared.rows, np.nan, dtype=np.float64)
    coverage = np.zeros(prepared.rows, dtype=np.int32)
    if specs:
        loader = DataLoader(
            _BlockDataset(prepared, specs, result.config),
            batch_size=batch_size or result.config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate,
        )
        result.model.to(result.device).eval()
        with torch.inference_mode():
            for batch in loader:
                starts = batch["starts"]
                stops = batch["stops"]
                for key in (
                    "left",
                    "right",
                    "target_covariates",
                    "lengths",
                    "left_center",
                    "left_scale",
                    "right_center",
                    "right_scale",
                ):
                    batch[key] = batch[key].to(result.device)
                with torch.autocast(
                    device_type="cuda" if result.device.startswith("cuda") else "cpu",
                    dtype=torch.bfloat16,
                    enabled=result.config.use_bfloat16,
                ):
                    forward_z, backward_z = result.model(
                        batch["left"],
                        batch["right"],
                        batch["target_covariates"],
                        batch["lengths"],
                    )
                forward_z = forward_z.float().cpu().numpy()
                backward_z = backward_z.float().cpu().numpy()
                left_center = batch["left_center"].float().cpu().numpy()
                left_scale = batch["left_scale"].float().cpu().numpy()
                right_center = batch["right_center"].float().cpu().numpy()
                right_scale = batch["right_scale"].float().cpu().numpy()
                lengths = batch["lengths"].cpu().numpy()
                for row, (start, stop, length) in enumerate(
                    zip(starts, stops, lengths, strict=True)
                ):
                    length = int(length)
                    relative = np.linspace(0.0, 1.0, length, dtype=np.float64)
                    forward_abs = forward_z[row, :length] * left_scale[row] + left_center[row]
                    backward_abs = backward_z[row, :length] * right_scale[row] + right_center[row]
                    prediction = (1.0 - relative) * forward_abs + relative * backward_abs
                    common_scale = max(float(left_scale[row]), float(right_scale[row]), 1.0e-3)
                    residual = (prepared.target_temp[start:stop] - prediction) / common_scale
                    abs_residual = np.abs(residual)
                    centered_position = relative - relative.mean()
                    denominator = float(np.square(centered_position).sum())
                    block_slope = (
                        abs(float(np.dot(centered_position, residual) / denominator))
                        if denominator > 0
                        else 0.0
                    )
                    block_disagreement = np.abs(forward_abs - backward_abs) / common_scale
                    current = absolute[start:stop]
                    replace = ~np.isfinite(current) | (abs_residual > current)
                    target_positions = np.arange(start, stop, dtype=np.int64)[replace]
                    signed[target_positions] = residual[replace]
                    absolute[target_positions] = abs_residual[replace]
                    slope[target_positions] = block_slope
                    disagreement[target_positions] = block_disagreement[replace]
                    coverage[start:stop] += 1
    scores_sorted = pd.DataFrame(
        {
            SCORE_COLUMNS[0]: signed,
            SCORE_COLUMNS[1]: absolute,
            SCORE_COLUMNS[2]: slope,
            SCORE_COLUMNS[3]: disagreement,
            "bmi_coverage_count": coverage,
        }
    )
    scores = scores_sorted.iloc[prepared.inverse_order].reset_index(drop=True)
    available = scores["bmi_coverage_count"].to_numpy() > 0
    return scores, {
        "rows": prepared.rows,
        "blocks": len(specs),
        "covered_rows": int(available.sum()),
        "covered_fraction": float(available.mean()),
        "finite_score_rows": int(np.isfinite(scores.loc[:, SCORE_COLUMNS]).all(axis=1).sum()),
    }


def fit_additive_gate(
    base_probability: Sequence[float],
    scores: pd.DataFrame,
    labels: Sequence[int],
    *,
    l2_penalty: float = 1.0,
    maximum_iterations: int = 200,
    bounds: Sequence[tuple[float, float]] = (
        (-5.0, 5.0),
        (0.0, 5.0),
        (0.0, 5.0),
        (-5.0, 0.0),
    ),
) -> AdditiveGate:
    from scipy.optimize import minimize
    from scipy.special import expit

    probability = np.clip(np.asarray(base_probability, dtype=np.float64), 1.0e-6, 1.0 - 1.0e-6)
    truth = np.asarray(labels, dtype=np.int8)
    matrix = scores.loc[:, SCORE_COLUMNS].to_numpy(dtype=np.float64)
    available = np.isfinite(matrix).all(axis=1)
    if probability.shape != truth.shape or len(matrix) != len(truth):
        raise ValueError("gate inputs are not row aligned")
    if available.sum() < 10 or truth[available].sum() == 0:
        raise ValueError("gate calibration lacks covered positive support")
    matrix = matrix[available]
    truth = truth[available].astype(np.float64)
    base_logit = np.log(probability[available] / (1.0 - probability[available]))
    scales = np.nanpercentile(np.abs(matrix), 75, axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1.0e-6), scales, 1.0)
    x = matrix / scales
    positive = max(1.0, float(truth.sum()))
    negative = max(1.0, float(len(truth) - truth.sum()))
    weights = np.where(truth == 1, np.sqrt(negative / positive), 1.0)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logit = base_logit + x @ beta
        # stable softplus(logit) - y*logit
        loss = np.logaddexp(0.0, logit) - truth * logit
        sigmoid = expit(logit)
        value = float(np.average(loss, weights=weights) + 0.5 * l2_penalty * np.dot(beta, beta))
        gradient = (x.T @ (weights * (sigmoid - truth))) / weights.sum() + l2_penalty * beta
        return value, gradient

    optimization = minimize(
        lambda beta: objective(beta)[0],
        np.zeros(len(SCORE_COLUMNS), dtype=np.float64),
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
        bounds=list(bounds),
        options={"maxiter": int(maximum_iterations), "ftol": 1.0e-10},
    )
    return AdditiveGate(
        coefficients=np.asarray(optimization.x, dtype=np.float64),
        scales=np.asarray(scales, dtype=np.float64),
        l2_penalty=float(l2_penalty),
        objective=float(optimization.fun),
        iterations=int(optimization.nit),
        success=bool(optimization.success),
    )


def apply_additive_gate(
    base_probability: Sequence[float], scores: pd.DataFrame, gate: AdditiveGate
) -> np.ndarray:
    from scipy.special import expit

    probability = np.clip(np.asarray(base_probability, dtype=np.float64), 1.0e-6, 1.0 - 1.0e-6)
    delta = gate.delta(scores)
    logit = np.log(probability / (1.0 - probability)) + delta
    return expit(logit)


def contract_hash(config: BlockInpaintConfig | None = None) -> str:
    frozen = config or BlockInpaintConfig()
    payload = {
        "config": asdict(frozen),
        "scores": SCORE_COLUMNS,
        "continuous": CONTINUOUS_COVARIATES,
        "cyclic": CYCLIC_COVARIATES,
        "target_contract": "raw target only outside mask; target block returned only as loss/scoring target",
    }
    import json

    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


__all__ = [
    "AdditiveGate",
    "BlockInpaintConfig",
    "BlockSpec",
    "CONTINUOUS_COVARIATES",
    "CYCLIC_COVARIATES",
    "InpaintTrainingResult",
    "PreparedSeries",
    "RobustCovariateScaler",
    "SCORE_COLUMNS",
    "SafeDesign",
    "apply_additive_gate",
    "assert_mask_invariance",
    "assert_target_safe_contract",
    "build_model",
    "build_safe_design",
    "contract_hash",
    "coverage_audit",
    "enumerate_blocks",
    "fit_additive_gate",
    "fit_covariate_scaler",
    "materialize_example",
    "prepare_series",
    "score_inpaint_model",
    "train_inpaint_model",
]
