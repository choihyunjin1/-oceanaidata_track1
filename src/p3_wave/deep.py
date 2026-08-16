"""Compact GRU and TCN trajectory models for P3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def derived_channels(raw: np.ndarray) -> np.ndarray:
    """Convert two circular directions into sin/cos without imputing missing values."""

    continuous = raw[..., [0, 1, 2, 4, 5, 7, 8, 9]].astype(np.float32, copy=False)
    wave_direction = np.deg2rad(raw[..., 3])
    wind_direction = np.deg2rad(raw[..., 6])
    circular = np.stack(
        [
            np.sin(wave_direction),
            np.cos(wave_direction),
            np.sin(wind_direction),
            np.cos(wind_direction),
        ],
        axis=-1,
    ).astype(np.float32)
    return np.concatenate([continuous, circular], axis=-1)


def fit_channel_statistics(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    derived = derived_channels(raw)
    center = np.nanmean(derived, axis=(0, 1)).astype(np.float32)
    scale = np.nanstd(derived, axis=(0, 1)).astype(np.float32)
    center[~np.isfinite(center)] = 0.0
    scale[~np.isfinite(scale) | (scale < 1e-4)] = 1.0
    return center, scale


class ContextEncoder(nn.Module):
    def __init__(self, center: np.ndarray, scale: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        relative_time = torch.linspace(-1.0, 0.0, 289, dtype=torch.float32).view(1, 289, 1)
        self.register_buffer("relative_time", relative_time)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        continuous = raw[..., [0, 1, 2, 4, 5, 7, 8, 9]]
        wave = torch.deg2rad(raw[..., 3])
        wind = torch.deg2rad(raw[..., 6])
        derived = torch.cat(
            [
                continuous,
                torch.sin(wave).unsqueeze(-1),
                torch.cos(wave).unsqueeze(-1),
                torch.sin(wind).unsqueeze(-1),
                torch.cos(wind).unsqueeze(-1),
            ],
            dim=-1,
        )
        mask = torch.isfinite(derived)
        normalized = (derived - self.center) / self.scale
        normalized = torch.where(mask, normalized, torch.zeros_like(normalized))
        time = self.relative_time.expand(len(raw), -1, -1)
        return torch.cat([normalized, mask.to(normalized.dtype), time], dim=-1)


class GRUTrajectory(nn.Module):
    def __init__(self, center: np.ndarray, scale: np.ndarray, hidden: int = 128) -> None:
        super().__init__()
        self.context = ContextEncoder(center, scale)
        self.input_projection = nn.Sequential(nn.Linear(25, 96), nn.GELU(), nn.LayerNorm(96))
        self.gru = nn.GRU(
            96,
            hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.15,
            bidirectional=True,
        )
        self.attention = nn.Linear(hidden * 2, 1)
        self.station = nn.Embedding(3, 8)
        self.head = nn.Sequential(
            nn.Linear(hidden * 4 + 8, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Linear(96, 6),
        )

    def forward(self, raw: torch.Tensor, station: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.gru(self.input_projection(self.context(raw)))
        attention = torch.softmax(self.attention(encoded).squeeze(-1), dim=1)
        pooled = torch.sum(encoded * attention.unsqueeze(-1), dim=1)
        last = encoded[:, -1]
        return self.head(torch.cat([last, pooled, self.station(station)], dim=1))


class TCNBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.GroupNorm(8, width),
            nn.Dropout(0.1),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.GroupNorm(8, width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class TCNTrajectory(nn.Module):
    def __init__(self, center: np.ndarray, scale: np.ndarray, width: int = 96) -> None:
        super().__init__()
        self.context = ContextEncoder(center, scale)
        self.input_projection = nn.Conv1d(25, width, 1)
        self.blocks = nn.Sequential(*(TCNBlock(width, 2**level) for level in range(7)))
        self.station = nn.Embedding(3, 8)
        self.head = nn.Sequential(
            nn.Linear(width * 2 + 8, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 6),
        )

    def forward(self, raw: torch.Tensor, station: torch.Tensor) -> torch.Tensor:
        encoded = self.blocks(self.input_projection(self.context(raw).transpose(1, 2)))
        pooled = torch.mean(encoded, dim=2)
        last = encoded[:, :, -1]
        return self.head(torch.cat([last, pooled, self.station(station)], dim=1))


@dataclass(frozen=True)
class DeepTrainConfig:
    architecture: str = "gru"
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 2e-4
    max_epochs: int = 60
    patience: int = 8
    seed: int = 20260816


def build_model(
    architecture: str, center: np.ndarray, scale: np.ndarray
) -> GRUTrajectory | TCNTrajectory:
    if architecture == "gru":
        return GRUTrajectory(center, scale)
    if architecture == "tcn":
        return TCNTrajectory(center, scale)
    raise ValueError(f"unknown architecture: {architecture}")
