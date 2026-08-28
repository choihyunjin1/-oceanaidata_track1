"""Fixed asynchronous leave-one-layer-out Matérn state and subset scan utilities."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

CADENCE_NS = int(pd.Timedelta(minutes=10).value)


@dataclass(frozen=True)
class BlockProposal:
    """Highest preregistered interval score inside one fixed seven-day block."""

    block_id: int
    score: float
    anomaly_type: str
    duration_rows: int
    row_positions: np.ndarray
    block_positions: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_metrics(truth: Sequence[int], prediction: Sequence[int]) -> dict[str, float | int]:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def robust_center_scale(values: Sequence[float], minimum_scale: float = 1e-6) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return 0.0, 1.0
    center = float(np.median(array))
    scale = float(1.4826 * np.median(np.abs(array - center)))
    if not np.isfinite(scale) or scale < minimum_scale:
        scale = float(np.std(array))
    if not np.isfinite(scale) or scale < minimum_scale:
        scale = 1.0
    return center, scale


def _matern32_factor(lengthscale_seconds: float, variance: float) -> tuple[np.ndarray, np.ndarray]:
    rate = np.sqrt(3.0) / float(lengthscale_seconds)
    drift = np.asarray([[0.0, 1.0], [-rate * rate, -2.0 * rate]], dtype=np.float64)
    stationary = variance * np.asarray([[1.0, 0.0], [0.0, rate * rate]], dtype=np.float64)
    return drift, stationary


def matern32_transition(
    delta_seconds: float,
    lengthscale_seconds: float,
    variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an exact discretization of a stationary Matérn-3/2 SDE factor."""

    drift, stationary = _matern32_factor(lengthscale_seconds, variance)
    transition = expm(drift * max(float(delta_seconds), 0.0))
    innovation = stationary - transition @ stationary @ transition.T
    innovation = 0.5 * (innovation + innovation.T)
    eigenvalues, eigenvectors = np.linalg.eigh(innovation)
    innovation = (eigenvectors * np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T
    return transition, innovation


def loo_matern_smoother(
    times_ns: Sequence[int],
    peer_observation: Sequence[float],
    peer_count: Sequence[int],
    *,
    minimum_peers: int,
    lengthscale_hours: Sequence[float],
    factor_variance: Sequence[float],
    observation_variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth a latent station state using peers only; the target is never an update."""

    times = np.asarray(times_ns, dtype=np.int64)
    observations = np.asarray(peer_observation, dtype=np.float64)
    counts = np.asarray(peer_count, dtype=np.int16)
    if len(times) != len(observations) or len(times) != len(counts):
        raise ValueError("state inputs have different lengths")
    if len(lengthscale_hours) != 2 or len(factor_variance) != 2:
        raise ValueError("the sealed state has exactly two Matérn factors")
    if len(times) and np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if not len(times):
        return np.empty(0), np.empty(0), np.empty(0, dtype=bool)

    stationary = np.zeros((4, 4), dtype=np.float64)
    for factor, (lengthscale, variance) in enumerate(
        zip(lengthscale_hours, factor_variance, strict=True)
    ):
        _, block = _matern32_factor(float(lengthscale) * 3600.0, float(variance))
        start = 2 * factor
        stationary[start : start + 2, start : start + 2] = block
    measurement = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    observed = np.isfinite(observations) & (counts >= int(minimum_peers))
    filtered_mean = np.zeros((len(times), 4), dtype=np.float64)
    filtered_covariance = np.zeros((len(times), 4, 4), dtype=np.float64)
    predicted_mean = np.zeros_like(filtered_mean)
    predicted_covariance = np.zeros_like(filtered_covariance)
    transitions = np.zeros((max(len(times) - 1, 0), 4, 4), dtype=np.float64)

    mean = np.zeros(4, dtype=np.float64)
    covariance = stationary.copy()
    transition_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index in range(len(times)):
        if index:
            delta_ns = int(times[index] - times[index - 1])
            cached = transition_cache.get(delta_ns)
            if cached is None:
                delta = float(delta_ns) / 1e9
                transition = np.zeros((4, 4), dtype=np.float64)
                innovation = np.zeros((4, 4), dtype=np.float64)
                for factor, (lengthscale, variance) in enumerate(
                    zip(lengthscale_hours, factor_variance, strict=True)
                ):
                    factor_transition, factor_innovation = matern32_transition(
                        delta, float(lengthscale) * 3600.0, float(variance)
                    )
                    start = 2 * factor
                    transition[start : start + 2, start : start + 2] = factor_transition
                    innovation[start : start + 2, start : start + 2] = factor_innovation
                transition_cache[delta_ns] = (transition, innovation)
            else:
                transition, innovation = cached
            transitions[index - 1] = transition
            mean = transition @ mean
            covariance = transition @ covariance @ transition.T + innovation
        predicted_mean[index] = mean
        predicted_covariance[index] = covariance
        if observed[index]:
            residual = observations[index] - float(measurement @ mean)
            residual_variance = float(measurement @ covariance @ measurement + observation_variance)
            gain = covariance @ measurement / max(residual_variance, 1e-12)
            mean = mean + gain * residual
            covariance = covariance - np.outer(gain, measurement @ covariance)
            covariance = 0.5 * (covariance + covariance.T)
        filtered_mean[index] = mean
        filtered_covariance[index] = covariance

    smoothed_mean = filtered_mean.copy()
    smoothed_covariance = filtered_covariance.copy()
    for index in range(len(times) - 2, -1, -1):
        transition = transitions[index]
        predicted = predicted_covariance[index + 1]
        gain = np.linalg.solve(predicted.T, (filtered_covariance[index] @ transition.T).T).T
        smoothed_mean[index] += gain @ (smoothed_mean[index + 1] - predicted_mean[index + 1])
        smoothed_covariance[index] += gain @ (
            smoothed_covariance[index + 1] - predicted
        ) @ gain.T
        smoothed_covariance[index] = 0.5 * (
            smoothed_covariance[index] + smoothed_covariance[index].T
        )
    latent_mean = smoothed_mean @ measurement
    latent_variance = np.einsum("i,nij,j->n", measurement, smoothed_covariance, measurement)
    return latent_mean, np.clip(latent_variance, 0.0, None), observed


def best_subset_interval(
    residual: Sequence[float], duration_rows: Mapping[str, Sequence[int]]
) -> tuple[float, str, int, int, int] | None:
    """Find the fixed-duration maximum GLRT score in one finite contiguous segment."""

    values = np.asarray(residual, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        return None
    prefix = np.r_[0.0, np.cumsum(values)]
    prefix_square = np.r_[0.0, np.cumsum(values * values)]
    index_prefix = np.r_[0.0, np.cumsum(np.arange(len(values), dtype=np.float64) * values)]
    best: tuple[float, str, int, int, int] | None = None
    for anomaly_type in ("noise", "offset", "drift"):
        for duration in duration_rows[anomaly_type]:
            width = int(duration)
            if width > len(values):
                continue
            starts = np.arange(len(values) - width + 1, dtype=np.int64)
            stops = starts + width
            sums = prefix[stops] - prefix[starts]
            sums_square = prefix_square[stops] - prefix_square[starts]
            if anomaly_type == "offset":
                scores = sums * sums / width
            elif anomaly_type == "drift":
                weighted = index_prefix[stops] - index_prefix[starts]
                centered = weighted - (starts + (width - 1.0) / 2.0) * sums
                denominator = width * (width * width - 1.0) / 12.0
                scores = centered * centered / max(denominator, 1e-12)
            else:
                variance = np.maximum(sums_square / width - (sums / width) ** 2, 1e-12)
                scores = np.where(variance > 1.0, width * (variance - 1.0 - np.log(variance)), 0.0)
            local = int(np.argmax(scores))
            candidate = (float(scores[local]), anomaly_type, width, int(starts[local]), int(stops[local]))
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best


def block_proposals(
    frame: pd.DataFrame,
    duration_rows: Mapping[str, Sequence[int]],
    *,
    block_days: int,
) -> list[BlockProposal]:
    """Emit at most one fixed-duration proposal per cell and fixed calendar block."""

    if frame.empty:
        return []
    ordered = frame.sort_values("time").reset_index(drop=True)
    times = pd.DatetimeIndex(ordered["time"])
    times_ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    blocks = times_ns // int(pd.Timedelta(days=block_days).value)
    output: list[BlockProposal] = []
    for block_id in np.unique(blocks):
        block_positions = np.flatnonzero(blocks == block_id)
        finite = np.isfinite(ordered.loc[block_positions, "residual"].to_numpy(dtype=np.float64))
        usable = block_positions[finite]
        if not len(usable):
            continue
        split_points = np.flatnonzero(np.diff(times_ns[usable]) != CADENCE_NS) + 1
        best: tuple[float, str, int, np.ndarray] | None = None
        for segment in np.split(usable, split_points):
            result = best_subset_interval(
                ordered.loc[segment, "residual"].to_numpy(dtype=np.float64), duration_rows
            )
            if result is None:
                continue
            score, anomaly_type, duration, start, stop = result
            candidate = (score, anomaly_type, duration, segment[start:stop])
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is not None:
            output.append(
                BlockProposal(
                    block_id=int(block_id),
                    score=float(best[0]),
                    anomaly_type=str(best[1]),
                    duration_rows=int(best[2]),
                    row_positions=np.asarray(best[3], dtype=np.int64),
                    block_positions=block_positions.astype(np.int64),
                )
            )
    return output


def conformal_threshold(scores: Sequence[float], alpha: float) -> float:
    values = np.sort(np.asarray(scores, dtype=np.float64))
    values = values[np.isfinite(values)]
    if not len(values):
        return float("inf")
    order = int(np.ceil((len(values) + 1) * (1.0 - float(alpha))))
    return float(values[min(max(order, 1), len(values)) - 1])


def block_bootstrap_delta(
    frame: pd.DataFrame,
    *,
    block_days: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    work = frame.copy()
    time = pd.to_datetime(work["time"], utc=True)
    work["block"] = (
        work["fold"].astype(str)
        + "|"
        + work["station"].astype(str)
        + "|"
        + work["layer"].astype(str)
        + "|"
        + (time.astype("int64") // int(pd.Timedelta(days=block_days).value)).astype(str)
    )
    counts: list[tuple[int, int, int, int, int, int]] = []
    for _, group in work.groupby("block", sort=True):
        old = binary_metrics(group["label"], group["anchor_prediction"])
        new = binary_metrics(group["label"], group["union_prediction"])
        counts.append(
            (
                int(old["tp"]), int(old["fp"]), int(old["fn"]),
                int(new["tp"]), int(new["fp"]), int(new["fn"]),
            )
        )
    matrix = np.asarray(counts, dtype=np.int64)
    if not len(matrix):
        return {"probability_delta_positive": 0.0, "ci90_lower": -1.0, "ci90_upper": -1.0}
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for iteration in range(replicates):
        total = matrix[rng.integers(0, len(matrix), len(matrix))].sum(axis=0)
        old_f1 = 2 * total[0] / max(2 * total[0] + total[1] + total[2], 1)
        new_f1 = 2 * total[3] / max(2 * total[3] + total[4] + total[5], 1)
        deltas[iteration] = new_f1 - old_f1
    return {
        "probability_delta_positive": float(np.mean(deltas > 0.0)),
        "ci90_lower": float(np.quantile(deltas, 0.05)),
        "ci90_upper": float(np.quantile(deltas, 0.95)),
    }
