"""NCAD-inspired causal TCN utilities for sparse long-event P1 recovery.

This module intentionally implements a small, auditable model rather than claiming an
exact reproduction of NCAD.  The transferable idea is contextual outlier exposure:
normal historical windows are altered with long offset/drift/noise/flatline segments so
the detector can learn useful boundaries before scarce real anomalies are used.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

CADENCE = pd.Timedelta(minutes=10)


@dataclass(frozen=True)
class ScaleState:
    center: np.ndarray
    scale: np.ndarray


def fit_robust_scale(values: np.ndarray) -> ScaleState:
    """Fit a finite median/IQR scale on fit-only rows."""

    matrix = np.asarray(values, dtype=np.float64)
    center = np.nanmedian(matrix, axis=0)
    q25 = np.nanquantile(matrix, 0.25, axis=0)
    q75 = np.nanquantile(matrix, 0.75, axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.nanstd(matrix, axis=0)
    bad = ~np.isfinite(scale) | (scale < 1e-6)
    scale[bad] = fallback[bad]
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    center[~np.isfinite(center)] = 0.0
    return ScaleState(center.astype(np.float32), scale.astype(np.float32))


def transform_robust(values: np.ndarray, state: ScaleState, clip: float = 12.0) -> np.ndarray:
    matrix = (np.asarray(values, dtype=np.float32) - state.center) / state.scale
    matrix[~np.isfinite(matrix)] = 0.0
    return np.clip(matrix, -clip, clip).astype(np.float32, copy=False)


class CausalResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(width, width, kernel_size=3, dilation=dilation)
        self.norm = nn.GroupNorm(1, width)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = nn.functional.pad(x, (self.pad, 0))
        y = self.conv(y)
        y = self.dropout(self.act(self.norm(y)))
        return x + y


class SyntheticContextTCN(nn.Module):
    """Small causal per-row anomaly scorer."""

    def __init__(
        self,
        channels: int,
        width: int = 64,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input = nn.Conv1d(channels, width, kernel_size=1)
        self.blocks = nn.Sequential(
            *(CausalResidualBlock(width, int(d), dropout) for d in dilations)
        )
        self.output = nn.Sequential(
            nn.Conv1d(width, width // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is batch x time x channels; output is batch x time.
        y = x.transpose(1, 2)
        y = self.blocks(self.input(y))
        return self.output(y).squeeze(1)


def inject_synthetic_event(
    window: np.ndarray,
    rng: np.random.Generator,
    *,
    event_min_rows: int,
    event_max_rows: int,
    primary_channels: Sequence[int],
    difference_channels: Sequence[int],
    donor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Inject one long contextual anomaly into a normalized feature window."""

    x = np.asarray(window, dtype=np.float32).copy()
    rows = len(x)
    if rows < event_min_rows + 2:
        raise ValueError("window is shorter than the minimum synthetic event")
    maximum = min(int(event_max_rows), rows // 2)
    length = int(rng.integers(event_min_rows, maximum + 1))
    start = int(rng.integers(max(1, rows // 4), rows - length + 1))
    stop = start + length
    mask = np.zeros(rows, dtype=np.float32)
    mask[start:stop] = 1.0
    kind = str(rng.choice(np.asarray(["offset", "drift", "noise", "flatline", "coe"])))
    primary = np.asarray(tuple(primary_channels), dtype=np.int64)
    diffs = np.asarray(tuple(difference_channels), dtype=np.int64)
    sign = float(rng.choice(np.asarray([-1.0, 1.0])))
    amplitude = float(rng.uniform(1.25, 3.25))

    if kind == "offset":
        x[start:stop, primary] += sign * amplitude
        if len(diffs):
            x[start, diffs] += sign * amplitude
    elif kind == "drift":
        ramp = np.linspace(0.0, sign * amplitude, length, dtype=np.float32)[:, None]
        x[start:stop, primary] += ramp
        if len(diffs):
            x[start:stop, diffs] += sign * amplitude / max(length, 1)
    elif kind == "noise":
        noise = rng.normal(0.0, amplitude, size=(length, len(primary))).astype(np.float32)
        x[start:stop, primary] += noise
        if len(diffs):
            x[start:stop, diffs] += rng.normal(0.0, amplitude, size=(length, len(diffs))).astype(
                np.float32
            )
    elif kind == "flatline":
        source = x[max(0, start - 1), primary].copy()
        x[start:stop, primary] = source
        if len(diffs):
            x[start:stop, diffs] = 0.0
    else:
        if donor is None or donor.shape != x.shape:
            donor = np.roll(x, rows // 2, axis=0)
        donor_start = int(rng.integers(0, rows - length + 1))
        x[start:stop, primary] = donor[donor_start : donor_start + length, primary]
        if len(diffs):
            x[start:stop, diffs] = donor[donor_start : donor_start + length, diffs]

    return np.clip(x, -12.0, 12.0), mask, kind


def continuous_segments(keys: pd.DataFrame, rows: Iterable[int]) -> list[np.ndarray]:
    """Return station/year/layer and cadence-contiguous row-id segments."""

    chosen = np.asarray(tuple(rows), dtype=np.int64)
    if not len(chosen):
        return []
    frame = keys.iloc[chosen].loc[:, ["station", "year", "layer", "time"]].copy()
    frame["row_id"] = chosen
    frame["time"] = pd.to_datetime(frame["time"], utc=True, format="mixed")
    output: list[np.ndarray] = []
    for _, part in frame.groupby(["station", "year", "layer"], sort=False, observed=True):
        part = part.sort_values("time")
        row_ids = part["row_id"].to_numpy(dtype=np.int64)
        times = part["time"].to_numpy()
        breaks = np.r_[True, np.diff(times) != CADENCE]
        ids = np.cumsum(breaks)
        output.extend(row_ids[ids == value] for value in np.unique(ids))
    return output


def window_rows(
    segments: Sequence[np.ndarray],
    window: int,
    stride: int,
    *,
    cover_tail: bool = True,
    pad_short: bool = False,
) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for segment in segments:
        if len(segment) < window:
            if pad_short and len(segment):
                padding = np.repeat(segment[0], window - len(segment))
                output.append(np.concatenate([padding, segment]))
            continue
        starts = list(range(0, len(segment) - window + 1, stride))
        if cover_tail and starts[-1] != len(segment) - window:
            starts.append(len(segment) - window)
        output.extend(segment[start : start + window] for start in starts)
    return output


def decode_long_components(
    scores: np.ndarray,
    keys: pd.DataFrame,
    rows: Sequence[int],
    *,
    threshold: float,
    minimum_rows: int,
    bridge_rows: int = 2,
) -> np.ndarray:
    """Threshold scores, bridge tiny gaps, and keep only long components."""

    output = np.zeros(len(keys), dtype=np.int8)
    for segment in continuous_segments(keys, rows):
        active = np.asarray(scores[segment] >= threshold, dtype=bool)
        if bridge_rows:
            cursor = 0
            while cursor < len(active):
                if active[cursor]:
                    cursor += 1
                    continue
                stop = cursor
                while stop < len(active) and not active[stop]:
                    stop += 1
                if (
                    cursor > 0
                    and stop < len(active)
                    and stop - cursor <= bridge_rows
                    and active[cursor - 1]
                    and active[stop]
                ):
                    active[cursor:stop] = True
                cursor = stop
        cursor = 0
        while cursor < len(active):
            if not active[cursor]:
                cursor += 1
                continue
            stop = cursor + 1
            while stop < len(active) and active[stop]:
                stop += 1
            if stop - cursor >= minimum_rows:
                output[segment[cursor:stop]] = 1
            cursor = stop
    return output


def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    f1 = 2.0 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "f1": f1}


def union_diagnostics(
    truth: np.ndarray, anchor: np.ndarray, additions: np.ndarray, rows: Sequence[int]
) -> dict[str, float | int]:
    selected = np.asarray(rows, dtype=np.int64)
    base = binary_metrics(truth[selected], anchor[selected])
    candidate = np.maximum(anchor, additions).astype(np.int8)
    updated = binary_metrics(truth[selected], candidate[selected])
    changed = (anchor[selected] == 0) & (candidate[selected] == 1)
    added_tp = int(np.sum(changed & (truth[selected] == 1)))
    added_fp = int(np.sum(changed & (truth[selected] == 0)))
    return {
        "anchor_f1": float(base["f1"]),
        "candidate_f1": float(updated["f1"]),
        "delta_f1": float(updated["f1"] - base["f1"]),
        "added_rows": int(changed.sum()),
        "added_tp": added_tp,
        "added_fp": added_fp,
        "added_precision": (added_tp / (added_tp + added_fp) if added_tp + added_fp else 1.0),
        "anchor_f1_over_2": float(base["f1"]) / 2.0,
        "anchor_positive_removed_rows": 0,
    }


__all__ = [
    "ScaleState",
    "SyntheticContextTCN",
    "binary_metrics",
    "continuous_segments",
    "decode_long_components",
    "fit_robust_scale",
    "inject_synthetic_event",
    "transform_robust",
    "union_diagnostics",
    "window_rows",
]
