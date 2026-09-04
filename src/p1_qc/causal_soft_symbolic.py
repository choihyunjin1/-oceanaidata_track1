"""Causal soft-symbolic transition features for the sealed P1 v18 preflight.

The representation is deliberately continuous in every observed value.  It
uses fixed Gaussian memberships rather than hard SAX bins, so the later linear
head remains a smooth profile instead of a discontinuous row router.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SoftSymbolicSpec:
    centers: np.ndarray
    bandwidth: float
    segment_count: int
    points_per_segment: int
    clip: float

    @property
    def feature_count(self) -> int:
        symbols = len(self.centers)
        return self.segment_count * symbols + (self.segment_count - 1) * symbols**2 + self.segment_count


def build_spec() -> SoftSymbolicSpec:
    return SoftSymbolicSpec(
        centers=np.asarray(
            [-1.2815515655446004, -0.5244005127080409, 0.0, 0.5244005127080409, 1.2815515655446004],
            dtype=np.float64,
        ),
        bandwidth=0.35,
        segment_count=12,
        points_per_segment=12,
        clip=12.0,
    )


def causal_robust_windows(
    times_minutes: np.ndarray,
    values: np.ndarray,
    query_minutes: np.ndarray,
    *,
    sample_minutes: int = 10,
    window_minutes: int = 1440,
    normalization_minutes: int = 4320,
    maximum_age_minutes: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Return past-only robust-z windows and their observation masks.

    Resampling is backward-as-of only.  A grid point never uses a source row
    later than that grid point, and a source row older than 20 minutes is
    treated as missing.  Each retained value is normalized using only source
    values from its own trailing 72-hour prefix.
    """

    times = np.asarray(times_minutes, dtype=np.int64)
    signal = np.asarray(values, dtype=np.float64)
    queries = np.asarray(query_minutes, dtype=np.int64)
    if times.ndim != 1 or signal.ndim != 1 or len(times) != len(signal):
        raise ValueError("times and values must be aligned vectors")
    if len(times) and (np.diff(times) <= 0).any():
        raise ValueError("source times must be strictly increasing")
    if window_minutes != 1440 or sample_minutes != 10:
        raise ValueError("v18 is sealed at a 24h/10min grid")
    grid_size = window_minutes // sample_minutes + 1
    windows = np.zeros((len(queries), grid_size), dtype=np.float32)
    observed = np.zeros((len(queries), grid_size), dtype=bool)
    for row, query in enumerate(queries):
        grid = np.arange(query - window_minutes, query + 1, sample_minutes, dtype=np.int64)
        source = np.searchsorted(times, grid, side="right") - 1
        valid_source = source >= 0
        age = np.full(grid_size, maximum_age_minutes + 1, dtype=np.int64)
        age[valid_source] = grid[valid_source] - times[source[valid_source]]
        available = valid_source & (age >= 0) & (age <= maximum_age_minutes)
        for column in np.flatnonzero(available):
            source_index = int(source[column])
            point = int(grid[column])
            start = int(np.searchsorted(times, point - normalization_minutes, side="left"))
            stop = int(np.searchsorted(times, point, side="right"))
            prefix = signal[start:stop]
            prefix = prefix[np.isfinite(prefix)]
            value = signal[source_index]
            if not np.isfinite(value) or len(prefix) < 12:
                continue
            median = float(np.median(prefix))
            mad = float(np.median(np.abs(prefix - median)))
            scale = max(1.4826 * mad, 1.0e-4)
            windows[row, column] = np.float32(np.clip((value - median) / scale, -12.0, 12.0))
            observed[row, column] = True
    return windows, observed


def soft_symbolic_transform(
    windows: np.ndarray,
    observed: np.ndarray,
    spec: SoftSymbolicSpec | None = None,
) -> np.ndarray:
    """Map 24-hour windows to fixed soft-symbol and transition features."""

    contract = spec or build_spec()
    values = np.asarray(windows, dtype=np.float64)
    mask = np.asarray(observed, dtype=bool)
    if values.shape != mask.shape or values.ndim != 2 or values.shape[1] != 145:
        raise ValueError("windows and observed must have shape (n,145)")
    # Drop the oldest endpoint: 144 points form exactly 12 non-overlapping 2h blocks.
    body = np.clip(values[:, 1:], -contract.clip, contract.clip).reshape(
        len(values), contract.segment_count, contract.points_per_segment
    )
    body_mask = mask[:, 1:].reshape(len(values), contract.segment_count, contract.points_per_segment)
    counts = body_mask.sum(axis=2)
    sums = np.where(body_mask, body, 0.0).sum(axis=2)
    paa = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    coverage = counts / float(contract.points_per_segment)
    distance = (paa[..., None] - contract.centers[None, None, :]) / contract.bandwidth
    membership = np.exp(-0.5 * distance**2)
    membership_sum = membership.sum(axis=2, keepdims=True)
    membership = np.divide(
        membership,
        membership_sum,
        out=np.zeros_like(membership),
        where=membership_sum > 0,
    )
    membership *= (counts > 0)[..., None]
    transition = membership[:, :-1, :, None] * membership[:, 1:, None, :]
    features = np.concatenate(
        [membership.reshape(len(values), -1), transition.reshape(len(values), -1), coverage],
        axis=1,
    ).astype(np.float32)
    if features.shape[1] != contract.feature_count:
        raise RuntimeError("soft-symbolic feature contract changed")
    if not np.isfinite(features).all():
        raise RuntimeError("soft-symbolic features are non-finite")
    return features


__all__ = ["SoftSymbolicSpec", "build_spec", "causal_robust_windows", "soft_symbolic_transform"]
