from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class RocketSpec:
    weights: np.ndarray
    dilations: np.ndarray
    same_padding: np.ndarray
    quantiles: np.ndarray
    channels: np.ndarray


def build_spec(feature_count: int = 512, seed: int = 20260831) -> RocketSpec:
    if feature_count != 512:
        raise ValueError("v17 is sealed at exactly 512 features")
    kernels = []
    for positive in combinations(range(9), 3):
        kernel = np.full(9, -1.0, dtype=np.float32)
        kernel[list(positive)] = 2.0
        kernels.append(kernel)
    bank = np.stack(kernels)
    rng = np.random.default_rng(seed)
    order = rng.permutation(feature_count)
    weights = bank[order % len(bank)]
    dilations = np.asarray([1, 2, 4, 8], dtype=np.int64)[(order // len(bank)) % 4]
    same = (order % 2 == 0)
    channels = np.where(order % 4 == 0, 1, 0).astype(np.int64)
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    quantiles = np.clip(((order + 1) * phi) % 1.0, 0.01, 0.99).astype(np.float32)
    return RocketSpec(weights, dilations, same, quantiles, channels)


def _convolutions(windows: np.ndarray, spec: RocketSpec, device: str) -> list[np.ndarray]:
    if windows.ndim != 3 or windows.shape[1:] != (2, 97):
        raise ValueError("windows must have shape (n,2,97)")
    x = torch.as_tensor(windows, dtype=torch.float32, device=device)
    outputs: list[np.ndarray] = [np.empty((0, 0), dtype=np.float32)] * 512
    for dilation in (1, 2, 4, 8):
        for same in (False, True):
            indices = np.flatnonzero((spec.dilations == dilation) & (spec.same_padding == same))
            for channel in (0, 1):
                chosen = indices[spec.channels[indices] == channel]
                if not len(chosen):
                    continue
                weight = torch.as_tensor(spec.weights[chosen, None, :], device=device)
                padding = 4 * dilation if same else 0
                values = F.conv1d(x[:, channel : channel + 1], weight, dilation=dilation, padding=padding)
                array = values.detach().cpu().numpy()
                for local, global_index in enumerate(chosen):
                    outputs[int(global_index)] = array[:, local, :]
    return outputs


def calibrate_biases(prefix_windows: np.ndarray, spec: RocketSpec, device: str = "cpu") -> np.ndarray:
    convolutions = _convolutions(prefix_windows, spec, device)
    return np.asarray(
        [np.quantile(values, spec.quantiles[index]) for index, values in enumerate(convolutions)],
        dtype=np.float32,
    )


def transform(windows: np.ndarray, spec: RocketSpec, biases: np.ndarray, device: str = "cpu") -> np.ndarray:
    if biases.shape != (512,):
        raise ValueError("biases must contain exactly 512 values")
    convolutions = _convolutions(windows, spec, device)
    return np.column_stack(
        [(values > biases[index]).mean(axis=1) for index, values in enumerate(convolutions)]
    ).astype(np.float32)


def causal_windows(
    times_minutes: np.ndarray,
    values: np.ndarray,
    query_minutes: np.ndarray,
    max_gap_minutes: int = 120,
) -> np.ndarray:
    """Build trailing windows; interpolation never sees a timestamp after its query."""
    times = np.asarray(times_minutes, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)
    result = np.zeros((len(query_minutes), 2, 97), dtype=np.float32)
    for row, query in enumerate(np.asarray(query_minutes, dtype=np.int64)):
        usable = times <= query
        prefix_t, prefix_v = times[usable], values[usable]
        grid = np.arange(query - 1440, query + 1, 15, dtype=np.int64)
        right = np.searchsorted(prefix_t, grid, side="left")
        for column, (point, r_index) in enumerate(zip(grid, right, strict=True)):
            if r_index < len(prefix_t) and prefix_t[r_index] == point:
                result[row, 0, column] = prefix_v[r_index]
                result[row, 1, column] = 1.0
            elif 0 < r_index < len(prefix_t):
                left_t, right_t = prefix_t[r_index - 1], prefix_t[r_index]
                if right_t - left_t <= max_gap_minutes:
                    fraction = (point - left_t) / (right_t - left_t)
                    result[row, 0, column] = prefix_v[r_index - 1] + fraction * (
                        prefix_v[r_index] - prefix_v[r_index - 1]
                    )
                    result[row, 1, column] = 1.0
    return result


def causal_trailing_robust_z(
    times_minutes: np.ndarray,
    values: np.ndarray,
    horizon_minutes: int = 4320,
) -> np.ndarray:
    """Exact causal trailing median/MAD normalization for a single ordered series."""
    times = np.asarray(times_minutes, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)
    output = np.zeros(len(values), dtype=np.float32)
    left = 0
    for index, current_time in enumerate(times):
        while times[left] < current_time - horizon_minutes:
            left += 1
        prefix = values[left : index + 1]
        finite = prefix[np.isfinite(prefix)]
        if not len(finite) or not np.isfinite(values[index]):
            continue
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        output[index] = np.clip((values[index] - median) / (1.4826 * mad + 1e-4), -12.0, 12.0)
    return output
