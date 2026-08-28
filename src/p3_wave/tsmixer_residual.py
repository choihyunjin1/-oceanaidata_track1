"""Observed-weather TSMixer residual model for the isolated P3 v1 experiment.

The model consumes only one case's past 48-hour context.  Native 10-minute
contexts are sampled on the inclusive hourly grid (49 points); no absolute
timestamps or future covariates enter the network.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import LEADS

CONTEXT_ROWS_10MIN = 289
HOURLY_ROWS = 49
RAW_CHANNELS = 10
DERIVED_CHANNELS = 12
ENCODED_CHANNELS = DERIVED_CHANNELS * 2 + 1
EARLY_LEADS = (3, 6, 9)
LONG_LEADS = (12, 18, 24)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_raw_shape(raw: np.ndarray | torch.Tensor) -> None:
    if raw.ndim != 3 or tuple(raw.shape[1:]) != (CONTEXT_ROWS_10MIN, RAW_CHANNELS):
        raise ValueError(
            "raw contexts must have shape [batch, 289, 10], got "
            f"{tuple(raw.shape)}"
        )


def hourly_derived_numpy(raw: np.ndarray) -> np.ndarray:
    """Return the frozen 12 observed channels on the inclusive hourly grid."""

    values = np.asarray(raw, dtype=np.float32)
    _validate_raw_shape(values)
    hourly = values[:, ::6, :]
    if hourly.shape[1] != HOURLY_ROWS:
        raise AssertionError("hourly sampling must yield 49 inclusive points")
    wave_direction = np.deg2rad(hourly[..., 3])
    wind_direction = np.deg2rad(hourly[..., 6])
    continuous = hourly[..., [0, 1, 2, 4, 5, 9, 7, 8]]
    circular = np.stack(
        [
            np.sin(wave_direction),
            np.cos(wave_direction),
            np.sin(wind_direction),
            np.cos(wind_direction),
        ],
        axis=-1,
    )
    return np.concatenate([continuous, circular], axis=-1).astype(np.float32)


def fit_hourly_statistics(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    derived = hourly_derived_numpy(raw)
    center = np.nanmean(derived, axis=(0, 1)).astype(np.float32)
    scale = np.nanstd(derived, axis=(0, 1)).astype(np.float32)
    center[~np.isfinite(center)] = 0.0
    scale[~np.isfinite(scale) | (scale < 1e-4)] = 1.0
    return center, scale


class HourlyObservedEncoder(nn.Module):
    """Train-fold normalization plus missing masks and relative time."""

    def __init__(self, center: np.ndarray, scale: np.ndarray) -> None:
        super().__init__()
        center_array = np.asarray(center, dtype=np.float32)
        scale_array = np.asarray(scale, dtype=np.float32)
        if center_array.shape != (DERIVED_CHANNELS,) or scale_array.shape != (
            DERIVED_CHANNELS,
        ):
            raise ValueError("center and scale must each have 12 values")
        self.register_buffer("center", torch.from_numpy(center_array.copy()))
        self.register_buffer("scale", torch.from_numpy(scale_array.copy()))
        self.register_buffer(
            "relative_time",
            torch.linspace(-1.0, 0.0, HOURLY_ROWS, dtype=torch.float32).view(
                1, HOURLY_ROWS, 1
            ),
        )

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        _validate_raw_shape(raw)
        hourly = raw[:, ::6, :]
        continuous = hourly[..., [0, 1, 2, 4, 5, 9, 7, 8]]
        wave_direction = torch.deg2rad(hourly[..., 3])
        wind_direction = torch.deg2rad(hourly[..., 6])
        derived = torch.cat(
            [
                continuous,
                torch.sin(wave_direction).unsqueeze(-1),
                torch.cos(wave_direction).unsqueeze(-1),
                torch.sin(wind_direction).unsqueeze(-1),
                torch.cos(wind_direction).unsqueeze(-1),
            ],
            dim=-1,
        )
        finite = torch.isfinite(derived)
        normalized = (derived - self.center) / self.scale
        normalized = torch.where(finite, normalized, torch.zeros_like(normalized))
        relative = self.relative_time.expand(len(raw), -1, -1)
        return torch.cat([normalized, finite.to(normalized.dtype), relative], dim=-1)


class MixerBlock(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        time_hidden: int,
        feature_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.time_norm = nn.LayerNorm(width)
        self.time_mlp = nn.Sequential(
            nn.Linear(HOURLY_ROWS, time_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(time_hidden, HOURLY_ROWS),
            nn.Dropout(dropout),
        )
        self.feature_norm = nn.LayerNorm(width)
        self.feature_mlp = nn.Sequential(
            nn.Linear(width, feature_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_hidden, width),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        time_input = self.time_norm(value).transpose(1, 2)
        value = value + self.time_mlp(time_input).transpose(1, 2)
        return value + self.feature_mlp(self.feature_norm(value))


@dataclass(frozen=True)
class TSMixerConfig:
    width: int = 64
    blocks: int = 4
    time_hidden: int = 128
    feature_hidden: int = 128
    dropout: float = 0.1
    station_embedding: int = 8

    def validate(self) -> None:
        if self != TSMixerConfig():
            raise ValueError(f"TSMixer architecture differs from frozen v1: {self!r}")


class ObservedResidualTSMixer(nn.Module):
    """Compact all-MLP backbone predicting six persistence residuals."""

    def __init__(
        self,
        center: np.ndarray,
        scale: np.ndarray,
        config: TSMixerConfig | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = TSMixerConfig()
        config.validate()
        self.context = HourlyObservedEncoder(center, scale)
        self.input_projection = nn.Linear(ENCODED_CHANNELS, config.width)
        self.station = nn.Embedding(3, config.station_embedding)
        self.station_projection = nn.Linear(config.station_embedding, config.width)
        self.blocks = nn.Sequential(
            *(
                MixerBlock(
                    width=config.width,
                    time_hidden=config.time_hidden,
                    feature_hidden=config.feature_hidden,
                    dropout=config.dropout,
                )
                for _ in range(config.blocks)
            )
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.temporal_head = nn.Linear(HOURLY_ROWS, len(LEADS))
        self.final_head = nn.Sequential(
            nn.Linear(config.width * len(LEADS) + config.station_embedding, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, len(LEADS)),
        )

    def forward(self, raw: torch.Tensor, station: torch.Tensor) -> torch.Tensor:
        station_embedding = self.station(station)
        value = self.input_projection(self.context(raw))
        value = value + self.station_projection(station_embedding).unsqueeze(1)
        value = self.output_norm(self.blocks(value))
        lead_features = self.temporal_head(value.transpose(1, 2)).flatten(1)
        return self.final_head(torch.cat([lead_features, station_embedding], dim=1))


def incumbent_preserving_blend(
    incumbent: np.ndarray,
    tsmixer: np.ndarray,
    *,
    long_model_weight: float = 0.2,
) -> np.ndarray:
    """Keep 3/6/9h bit-exact and blend only 12/18/24h."""

    base = np.asarray(incumbent, dtype=np.float64)
    model = np.asarray(tsmixer, dtype=np.float64)
    if base.shape != model.shape or base.ndim != 2 or base.shape[1] != len(LEADS):
        raise ValueError("prediction matrices must both have shape [cases, 6]")
    if not 0.0 <= long_model_weight <= 1.0:
        raise ValueError("long_model_weight must lie in [0, 1]")
    output = base.copy()
    long_columns = [LEADS.index(lead) for lead in LONG_LEADS]
    output[:, long_columns] = (
        (1.0 - long_model_weight) * base[:, long_columns]
        + long_model_weight * model[:, long_columns]
    )
    return np.clip(output, 0.0, 30.0)


def decision_gates(
    *,
    pooled_delta_m: float,
    fold_deltas_m: dict[str, float],
    station_deltas_m: dict[str, float],
    lead_deltas_m: dict[str, float],
    bootstrap_ci90_upper_m: float,
    probability_improved: float,
    novelty_rms_m: float,
    seed_rmse_spread_m: float,
    runtime_seconds: float,
    maximum_seed_seconds: float,
) -> dict[str, Any]:
    folds_improved = sum(value < 0.0 for value in fold_deltas_m.values())
    max_station_degradation = max(station_deltas_m.values())
    critical_degradation = max(lead_deltas_m[str(lead)] for lead in LONG_LEADS[1:])
    performance_go = (
        pooled_delta_m <= -0.010
        and folds_improved >= 2
        and bootstrap_ci90_upper_m < 0.0
        and all(lead_deltas_m[str(lead)] <= 0.0 for lead in LONG_LEADS[1:])
        and max_station_degradation <= 0.010
    )
    official_info_go = (
        pooled_delta_m < 0.0
        and folds_improved >= 2
        and probability_improved >= 0.80
        and novelty_rms_m >= 0.030
        and max_station_degradation <= 0.010
        and critical_degradation <= 0.010
        and seed_rmse_spread_m <= 0.020
        and maximum_seed_seconds <= 1200.0
        and runtime_seconds <= 10800.0
    )
    stop_reasons: list[str] = []
    if pooled_delta_m >= 0.0:
        stop_reasons.append("pooled_not_better_than_incumbent")
    if folds_improved < 2:
        stop_reasons.append("fewer_than_two_folds_improved")
    if critical_degradation > 0.010:
        stop_reasons.append("critical_18_or_24h_degradation_above_0p01m")
    if seed_rmse_spread_m > 0.020:
        stop_reasons.append("seed_rmse_spread_above_0p02m")
    if maximum_seed_seconds > 1200.0:
        stop_reasons.append("single_seed_runtime_above_20min")
    if runtime_seconds > 10800.0:
        stop_reasons.append("total_runtime_above_3h")
    return {
        "performance_go": bool(performance_go),
        "official_info_go": bool(official_info_go),
        "stop": bool(stop_reasons),
        "stop_reasons": stop_reasons,
        "folds_improved": int(folds_improved),
        "maximum_station_degradation_m": float(max_station_degradation),
        "maximum_critical_lead_degradation_m": float(critical_degradation),
    }


__all__ = [
    "CONTEXT_ROWS_10MIN",
    "DERIVED_CHANNELS",
    "EARLY_LEADS",
    "ENCODED_CHANNELS",
    "HOURLY_ROWS",
    "LONG_LEADS",
    "ObservedResidualTSMixer",
    "TSMixerConfig",
    "decision_gates",
    "fit_hourly_statistics",
    "hourly_derived_numpy",
    "incumbent_preserving_blend",
    "sha256_file",
]
