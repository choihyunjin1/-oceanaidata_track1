"""Past-only asymmetric exogenous encoder for the P3 direct six-lead experiment.

This is a compact TimeXer-style adaptation, not a verbatim copy of the research
implementation.  One case's observed 48-hour history is the complete receptive
field.  The endogenous wave-height patches and the past-only exogenous variates
are embedded separately and joined by variate cross-attention.  No future
covariate, absolute timestamp, or cross-case sequence can enter the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import LEADS
from .tsmixer_residual import (
    DERIVED_CHANNELS,
    HOURLY_ROWS,
    HourlyObservedEncoder,
    fit_hourly_statistics,
    hourly_derived_numpy,
    sha256_file,
)

ENDOGENOUS_INDEX = 0
EXOGENOUS_INDICES = tuple(range(1, DERIVED_CHANNELS))
EXOGENOUS_CHANNELS = (
    "tp",
    "hmax",
    "wspd",
    "gust",
    "relh",
    "caph",
    "airt",
    "wvdir_sin",
    "wvdir_cos",
    "wdir_sin",
    "wdir_cos",
)


@dataclass(frozen=True)
class DirectTimeXerConfig:
    patch_length: int = 7
    patch_stride: int = 7
    patch_count: int = 7
    d_model: int = 64
    attention_heads: int = 4
    encoder_layers: int = 2
    feedforward_width: int = 128
    dropout: float = 0.1
    station_embedding: int = 8

    def validate(self) -> None:
        if self != DirectTimeXerConfig():
            raise ValueError(f"TimeXer architecture differs from frozen v1: {self!r}")
        if self.patch_length * self.patch_count != HOURLY_ROWS:
            raise ValueError("patches must partition all 49 inclusive hourly points")
        if self.patch_stride != self.patch_length:
            raise ValueError("frozen v1 uses non-overlapping endogenous patches")
        if self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")


class AsymmetricPastTokenizer(nn.Module):
    """Create endogenous patch tokens and exogenous variate tokens."""

    def __init__(
        self,
        center: np.ndarray,
        scale: np.ndarray,
        config: DirectTimeXerConfig,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.context = HourlyObservedEncoder(center, scale)
        self.endogenous_projection = nn.Linear(config.patch_length * 3, config.d_model)
        self.exogenous_projection = nn.Linear(HOURLY_ROWS * 3, config.d_model)
        self.patch_position = nn.Parameter(
            torch.zeros(1, config.patch_count, config.d_model)
        )
        self.exogenous_identity = nn.Parameter(
            torch.zeros(1, len(EXOGENOUS_INDICES), config.d_model)
        )
        nn.init.trunc_normal_(self.patch_position, std=0.02)
        nn.init.trunc_normal_(self.exogenous_identity, std=0.02)

    def forward(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.context(raw)
        normalized = encoded[..., :DERIVED_CHANNELS]
        observed = encoded[..., DERIVED_CHANNELS : 2 * DERIVED_CHANNELS]
        relative = encoded[..., -1:]

        endogenous = torch.cat(
            [
                normalized[..., ENDOGENOUS_INDEX : ENDOGENOUS_INDEX + 1],
                observed[..., ENDOGENOUS_INDEX : ENDOGENOUS_INDEX + 1],
                relative,
            ],
            dim=-1,
        )
        endogenous = endogenous.reshape(
            len(raw), self.config.patch_count, self.config.patch_length * 3
        )
        endogenous_tokens = self.endogenous_projection(endogenous) + self.patch_position

        exogenous_values = normalized[..., EXOGENOUS_INDICES].transpose(1, 2)
        exogenous_masks = observed[..., EXOGENOUS_INDICES].transpose(1, 2)
        exogenous_time = relative.transpose(1, 2).expand(
            -1, len(EXOGENOUS_INDICES), -1
        )
        exogenous = torch.stack(
            [exogenous_values, exogenous_masks, exogenous_time], dim=-1
        ).flatten(2)
        exogenous_tokens = (
            self.exogenous_projection(exogenous) + self.exogenous_identity
        )
        return endogenous_tokens, exogenous_tokens


class PastExogenousDirectTimeXer(nn.Module):
    """Jointly predict six persistence residuals from one case's past context."""

    def __init__(
        self,
        center: np.ndarray,
        scale: np.ndarray,
        config: DirectTimeXerConfig | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = DirectTimeXerConfig()
        config.validate()
        self.config = config
        self.tokenizer = AsymmetricPastTokenizer(center, scale, config)
        self.global_endogenous = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.trunc_normal_(self.global_endogenous, std=0.02)
        self.station = nn.Embedding(3, config.station_embedding)
        self.station_projection = nn.Linear(config.station_embedding, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.endogenous_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            enable_nested_tensor=False,
        )
        self.exogenous_cross_attention = nn.MultiheadAttention(
            config.d_model,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Sequential(
            nn.LayerNorm(config.d_model * 2 + config.station_embedding),
            nn.Linear(
                config.d_model * 2 + config.station_embedding,
                config.feedforward_width,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_width, len(LEADS)),
        )

    def encode_tokens(
        self, raw: torch.Tensor, station: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        endogenous, exogenous = self.tokenizer(raw)
        station_embedding = self.station(station)
        station_context = self.station_projection(station_embedding).unsqueeze(1)
        global_token = self.global_endogenous.expand(len(raw), -1, -1) + station_context
        endogenous = self.endogenous_encoder(
            torch.cat([global_token, endogenous + station_context], dim=1)
        )
        return endogenous, exogenous, station_embedding

    def forward(self, raw: torch.Tensor, station: torch.Tensor) -> torch.Tensor:
        endogenous, exogenous, station_embedding = self.encode_tokens(raw, station)
        global_token = endogenous[:, :1]
        cross, _ = self.exogenous_cross_attention(
            global_token, exogenous, exogenous, need_weights=False
        )
        fused_global = self.cross_norm(global_token + cross).squeeze(1)
        patch_summary = endogenous[:, 1:].mean(dim=1)
        return self.output(
            torch.cat([fused_global, patch_summary, station_embedding], dim=1)
        )


def persistence_additive_prediction(
    current_hs: np.ndarray, residual: np.ndarray
) -> np.ndarray:
    current = np.asarray(current_hs, dtype=np.float64)
    delta = np.asarray(residual, dtype=np.float64)
    if current.ndim != 1 or delta.shape != (len(current), len(LEADS)):
        raise ValueError("current must be [cases] and residual must be [cases, 6]")
    return np.clip(current[:, None] + delta, 0.0, 30.0)


def promotion_gates(
    *,
    pooled_delta_m: float,
    fold_deltas_m: dict[str, float],
    station_deltas_m: dict[str, float],
    lead_deltas_m: dict[str, float],
    bootstrap_ci90_upper_m: float,
) -> dict[str, Any]:
    improved_folds = sum(value < 0.0 for value in fold_deltas_m.values())
    worst_station = max(station_deltas_m.values())
    long_leads_non_degrading = all(
        lead_deltas_m[str(lead)] <= 0.0 for lead in (12, 18, 24)
    )
    passed = (
        pooled_delta_m <= -0.005
        and improved_folds >= 2
        and bootstrap_ci90_upper_m < 0.0
        and worst_station <= 0.01
        and long_leads_non_degrading
    )
    return {
        "local_promotion_go": bool(passed),
        "pooled_delta_gate": bool(pooled_delta_m <= -0.005),
        "fold_consistency_gate": bool(improved_folds >= 2),
        "bootstrap_gate": bool(bootstrap_ci90_upper_m < 0.0),
        "station_safety_gate": bool(worst_station <= 0.01),
        "long_lead_safety_gate": bool(long_leads_non_degrading),
        "improved_folds": int(improved_folds),
        "worst_station_regression_m": float(worst_station),
    }


__all__ = [
    "AsymmetricPastTokenizer",
    "DirectTimeXerConfig",
    "EXOGENOUS_CHANNELS",
    "EXOGENOUS_INDICES",
    "PastExogenousDirectTimeXer",
    "fit_hourly_statistics",
    "hourly_derived_numpy",
    "persistence_additive_prediction",
    "promotion_gates",
    "sha256_file",
]
