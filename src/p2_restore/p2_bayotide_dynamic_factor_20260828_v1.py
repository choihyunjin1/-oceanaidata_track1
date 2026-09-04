"""Fixed BayOTIDE-style functional low-rank state-space pilot for P2.

The model is deliberately small and deterministic: three Matérn-3/2 trend
factors and one oscillator at each of 12.42 and 24 hours.  Temperature and
salinity channels share the latent state.  Fold target values are removed
before both loading estimation and Kalman measurement updates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import expm

from p2_tide_rts import cadence_segments, kalman_rts_smoother

TARGET_LAYERS = (2, 3, 4)
VARIABLES = ("temp", "psal")
CADENCE_HOURS = 1.0 / 6.0


@dataclass(frozen=True)
class RegisteredPanel:
    times: pd.DatetimeIndex
    layers: tuple[int, ...]
    values: np.ndarray
    depths: np.ndarray


@dataclass(frozen=True)
class DynamicFactorModel:
    channel_layers: tuple[int, ...]
    channel_variables: tuple[str, ...]
    support_depth_m: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    loading: np.ndarray
    observation_noise: np.ndarray
    transition: np.ndarray
    process_covariance: np.ndarray
    initial_covariance: np.ndarray
    posterior_guard_c: np.ndarray
    periodic_state_indices: tuple[tuple[int, int], ...]
    periods_hours: tuple[float, ...]


@dataclass(frozen=True)
class DynamicPrediction:
    candidate: np.ndarray
    dynamic: np.ndarray
    posterior_sd_c: np.ndarray
    active: np.ndarray


def build_registered_panel(observations: pd.DataFrame, year: int) -> RegisteredPanel:
    """Build a year-regime panel without reading deployment inputs."""

    required = {"year", "time", "layer", "temp", "psal", "depth", "nominal_depth"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"P2 observations miss columns: {sorted(missing)}")
    work = observations.loc[observations["year"].eq(int(year)), list(required)].copy()
    work["time"] = pd.to_datetime(work["time"], utc=True)
    if work.empty or work.duplicated(["time", "layer"]).any():
        raise ValueError("year-regime observations are empty or duplicate")
    layers = tuple(sorted(int(value) for value in work["layer"].unique()))
    times = pd.DatetimeIndex(work["time"].drop_duplicates()).sort_values()
    matrices: list[np.ndarray] = []
    for variable in VARIABLES:
        wide = work.pivot(index="time", columns="layer", values=variable).reindex(
            index=times, columns=layers
        )
        matrices.append(wide.to_numpy(np.float64))
    depth = work.pivot(index="time", columns="layer", values="depth").reindex(
        index=times, columns=layers
    )
    nominal = work.pivot(index="time", columns="layer", values="nominal_depth").reindex(
        index=times, columns=layers
    )
    depth_values = depth.to_numpy(np.float64)
    depth_values = np.where(np.isfinite(depth_values), depth_values, nominal.to_numpy(float))
    return RegisteredPanel(times, layers, np.column_stack(matrices), depth_values)


def fold_masks(
    times: pd.DatetimeIndex,
    start: str,
    stop: str,
    *,
    purge_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = pd.Timestamp(start, tz="Asia/Seoul").tz_convert("UTC")
    right = pd.Timestamp(stop, tz="Asia/Seoul").tz_convert("UTC")
    validation = np.asarray((times >= left) & (times < right), dtype=bool)
    purged = np.asarray(
        (times >= left - pd.Timedelta(days=int(purge_days)))
        & (times < right + pd.Timedelta(days=int(purge_days))),
        dtype=bool,
    )
    training = ~purged
    if not validation.any() or np.any(training & purged):
        raise ValueError("invalid P2 blocked split")
    return training, validation, purged


def period_phase(times: pd.DatetimeIndex, periods_hours: tuple[float, ...]) -> np.ndarray:
    seconds = times.as_unit("ns").asi8.astype(np.float64) / 1e9
    pieces: list[np.ndarray] = []
    for period in periods_hours:
        angle = 2.0 * np.pi * seconds / (float(period) * 3600.0)
        pieces.extend((np.sin(angle), np.cos(angle)))
    return np.column_stack(pieces)


def matern32_discretization(lengthscale_hours: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one exact-discrete stationary Matérn-3/2 state pair."""

    lengthscale = float(lengthscale_hours)
    if lengthscale <= 0.0:
        raise ValueError("Matérn lengthscale must be positive")
    rate = np.sqrt(3.0) / lengthscale
    generator = np.asarray([[0.0, 1.0], [-rate**2, -2.0 * rate]], dtype=np.float64)
    transition = expm(generator * CADENCE_HOURS)
    stationary = np.diag([1.0, rate**2])
    process = stationary - transition @ stationary @ transition.T
    process = (process + process.T) * 0.5
    eigenvalue, eigenvector = np.linalg.eigh(process)
    process = (eigenvector * np.maximum(eigenvalue, 1e-12)) @ eigenvector.T
    return transition, process, stationary


def fixed_state_matrices(
    lengthscales_hours: tuple[float, ...],
    periods_hours: tuple[float, ...],
    *,
    periodic_damping: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    state_pairs = len(lengthscales_hours) + len(periods_hours)
    size = state_pairs * 2
    transition = np.zeros((size, size), dtype=np.float64)
    process = np.zeros_like(transition)
    initial = np.zeros_like(transition)
    for factor, lengthscale in enumerate(lengthscales_hours):
        indices = slice(2 * factor, 2 * factor + 2)
        current_transition, current_process, stationary = matern32_discretization(lengthscale)
        transition[indices, indices] = current_transition
        process[indices, indices] = current_process
        initial[indices, indices] = stationary
    periodic_indices: list[tuple[int, int]] = []
    for offset, period in enumerate(periods_hours):
        left = 2 * (len(lengthscales_hours) + offset)
        pair = (left, left + 1)
        angle = 2.0 * np.pi * CADENCE_HOURS / float(period)
        rotation = float(periodic_damping) * np.asarray(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]],
            dtype=np.float64,
        )
        transition[left : left + 2, left : left + 2] = rotation
        process[left : left + 2, left : left + 2] = (
            1.0 - float(periodic_damping) ** 2
        ) * np.eye(2)
        initial[left : left + 2, left : left + 2] = np.eye(2)
        periodic_indices.append(pair)
    return transition, process, initial, tuple(periodic_indices)


def _registered_channels(
    panel: RegisteredPanel, training: np.ndarray
) -> tuple[np.ndarray, tuple[int, ...], tuple[str, ...], np.ndarray]:
    train = np.asarray(training, dtype=bool)
    if train.shape != (len(panel.times),) or train.sum() < 100:
        raise ValueError("insufficient training rows")
    support = np.nanmedian(panel.depths[train], axis=0)
    if not np.isfinite(support).all():
        raise ValueError("actual-depth channel alignment failure")
    order = np.argsort(support, kind="stable")
    support = support[order]
    if np.any(np.diff(support) <= 0.05):
        raise ValueError("actual-depth supports are not uniquely ordered")
    layers = tuple(int(panel.layers[index]) for index in order)
    layer_count = len(panel.layers)
    temp = panel.values[:, :layer_count][:, order]
    psal = panel.values[:, layer_count:][:, order]
    values = np.column_stack((temp, psal))
    channel_layers = layers + layers
    channel_variables = ("temp",) * len(layers) + ("psal",) * len(layers)
    channel_depth = np.r_[support, support]
    return values, channel_layers, channel_variables, channel_depth


def _robust_center_scale(values: np.ndarray, training: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = values[np.asarray(training, dtype=bool)]
    center = np.nanmedian(train, axis=0)
    scale = np.nanmedian(np.abs(train - center), axis=0) * 1.4826
    fallback = np.nanstd(train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fallback)
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 1e-6):
        raise ValueError("channel normalization failed")
    return center, scale


def fit_fixed_dynamic_factor(
    panel: RegisteredPanel,
    training: np.ndarray,
    *,
    trend_lengthscales_hours: tuple[float, ...],
    periods_hours: tuple[float, ...],
    completion_iterations: int,
    minimum_channel_coverage: float,
    observation_noise_floor: float,
    periodic_damping: float,
    posterior_multiplier: float,
    posterior_absolute_cap_c: float,
) -> tuple[DynamicFactorModel, np.ndarray]:
    """Fit fixed loadings using only outer-training rows; no model search."""

    if len(trend_lengthscales_hours) != 3 or len(periods_hours) != 2:
        raise ValueError("fixed factor contract requires three trend and two periodic factors")
    values, channel_layers, channel_variables, channel_depth = _registered_channels(panel, training)
    selected = np.asarray(training, dtype=bool)
    coverage = np.isfinite(values[selected]).mean(axis=0)
    if np.any(coverage < float(minimum_channel_coverage)):
        raise ValueError("actual-depth channel coverage failure")
    center, scale = _robust_center_scale(values, selected)
    standardized = (values - center) / scale
    phases = period_phase(panel.times, periods_hours)
    periodic_loading = np.zeros((values.shape[1], phases.shape[1]), dtype=np.float64)
    for channel in range(values.shape[1]):
        keep = selected & np.isfinite(standardized[:, channel])
        if keep.sum() < 100:
            raise ValueError("periodic loading support is insufficient")
        periodic_loading[channel] = np.linalg.lstsq(
            phases[keep], standardized[keep, channel], rcond=None
        )[0]
    residual = standardized - phases @ periodic_loading.T
    train_residual = residual[selected]
    observed = np.isfinite(train_residual)
    filled = np.where(observed, train_residual, 0.0)
    rank = len(trend_lengthscales_hours)
    for _ in range(int(completion_iterations)):
        _, _, right = np.linalg.svd(filled, full_matrices=False)
        components = right[:rank]
        reconstructed = (filled @ components.T) @ components
        filled = np.where(observed, train_residual, reconstructed)
    _, _, right = np.linalg.svd(filled, full_matrices=False)
    components = right[:rank]
    score = filled @ components.T
    score_scale = np.std(score, axis=0)
    if not np.isfinite(score_scale).all() or np.any(score_scale <= 1e-8):
        raise ValueError("trend factor scale failed")
    trend_loading = components.T * score_scale
    normalized_score = score / score_scale

    transition, process, initial, periodic_indices = fixed_state_matrices(
        trend_lengthscales_hours, periods_hours, periodic_damping=periodic_damping
    )
    loading = np.zeros((values.shape[1], transition.shape[0]), dtype=np.float64)
    for factor in range(rank):
        loading[:, 2 * factor] = trend_loading[:, factor]
    for period_index, (left, right_index) in enumerate(periodic_indices):
        loading[:, left] = periodic_loading[:, 2 * period_index]
        loading[:, right_index] = periodic_loading[:, 2 * period_index + 1]
    proxy_state = np.zeros((int(selected.sum()), transition.shape[0]), dtype=np.float64)
    for factor in range(rank):
        proxy_state[:, 2 * factor] = normalized_score[:, factor]
    proxy_state[:, 2 * rank :] = phases[selected]
    fitted = proxy_state @ loading.T
    errors = train_residual + phases[selected] @ periodic_loading.T - fitted
    noise = np.nanvar(np.where(observed, errors, np.nan), axis=0)
    noise = np.maximum(np.where(np.isfinite(noise), noise, 1.0), float(observation_noise_floor))
    residual_c = np.sqrt(noise) * scale
    guard = np.minimum(
        float(posterior_absolute_cap_c), float(posterior_multiplier) * residual_c
    )
    model = DynamicFactorModel(
        channel_layers=channel_layers,
        channel_variables=channel_variables,
        support_depth_m=channel_depth,
        center=center,
        scale=scale,
        loading=loading,
        observation_noise=noise,
        transition=transition,
        process_covariance=process,
        initial_covariance=initial,
        posterior_guard_c=guard,
        periodic_state_indices=periodic_indices,
        periods_hours=periods_hours,
    )
    return model, values


def mask_target_updates(
    values: np.ndarray,
    model: DynamicFactorModel,
    purged: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    masked = np.asarray(values, dtype=np.float64).copy()
    observed = np.isfinite(masked)
    target = np.asarray(
        [layer in TARGET_LAYERS for layer in model.channel_layers], dtype=bool
    )
    observed[np.asarray(purged, dtype=bool)[:, None] & target[None, :]] = False
    masked[~observed] = np.nan
    return masked, observed


def smooth_dynamic_factor(
    panel: RegisteredPanel,
    values: np.ndarray,
    model: DynamicFactorModel,
    purged: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masked, observed = mask_target_updates(values, model, purged)
    normalized = (masked - model.center) / model.scale
    state_mean = np.full((len(panel.times), model.transition.shape[0]), np.nan)
    state_covariance = np.full(
        (len(panel.times), model.transition.shape[0], model.transition.shape[0]), np.nan
    )
    phases = period_phase(panel.times, model.periods_hours)
    for start, stop in cadence_segments(panel.times):
        initial_mean = np.zeros(model.transition.shape[0], dtype=np.float64)
        for period, (left, right) in enumerate(model.periodic_state_indices):
            initial_mean[left] = phases[start, 2 * period]
            initial_mean[right] = phases[start, 2 * period + 1]
        result = kalman_rts_smoother(
            normalized[start:stop],
            observed[start:stop],
            model.loading,
            model.observation_noise,
            model.transition,
            model.process_covariance,
            initial_mean=initial_mean,
            initial_covariance=model.initial_covariance,
        )
        state_mean[start:stop] = result.mean
        state_covariance[start:stop] = result.covariance
    if not np.isfinite(state_mean).all() or not np.isfinite(state_covariance).all():
        raise RuntimeError("non-finite filter or smoother")
    predicted = (state_mean @ model.loading.T) * model.scale + model.center
    variance = np.einsum(
        "ci,nij,cj->nc", model.loading, state_covariance, model.loading
    ) + model.observation_noise[None, :]
    posterior_sd = np.sqrt(np.maximum(variance, 0.0)) * model.scale
    return predicted, posterior_sd, observed


def guarded_temperature_candidate(
    *,
    incumbent: np.ndarray,
    dynamic_temperature: np.ndarray,
    posterior_sd_c: np.ndarray,
    public_observed_counts: np.ndarray,
    posterior_guard_c: np.ndarray,
    minimum_public_channels: int,
) -> DynamicPrediction:
    base = np.asarray(incumbent, dtype=np.float64)
    dynamic = np.asarray(dynamic_temperature, dtype=np.float64)
    posterior = np.asarray(posterior_sd_c, dtype=np.float64)
    guard = np.asarray(posterior_guard_c, dtype=np.float64)
    if base.shape != dynamic.shape or base.shape != posterior.shape or base.shape[1] != len(guard):
        raise ValueError("guarded candidate arrays are not aligned")
    profile = (
        np.isfinite(dynamic).all(axis=1)
        & np.isfinite(posterior).all(axis=1)
        & (posterior <= guard[None, :]).all(axis=1)
        & (np.asarray(public_observed_counts) >= int(minimum_public_channels))
    )
    active = np.broadcast_to(profile[:, None], base.shape).copy()
    candidate = base.copy()
    candidate[active] = dynamic[active]
    if not np.array_equal(candidate[~active], base[~active], equal_nan=True):
        raise AssertionError("posterior fallback changed incumbent")
    return DynamicPrediction(candidate, dynamic, posterior, active)


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    actual = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(actual - estimate))))


def paired_kst_day_bootstrap(
    frame: pd.DataFrame, *, replicates: int, seed: int
) -> dict[str, float | int]:
    work = frame[["time", "truth", "reference", "candidate"]].copy()
    work["day"] = (
        pd.to_datetime(work["time"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
    )
    work["reference_se"] = np.square(work["reference"] - work["truth"])
    work["candidate_se"] = np.square(work["candidate"] - work["truth"])
    daily = work.groupby("day", sort=True).agg(
        rows=("truth", "size"), reference_se=("reference_se", "sum"), candidate_se=("candidate_se", "sum")
    )
    values = daily.to_numpy(np.float64)
    generator = np.random.default_rng(int(seed))
    delta = np.empty(int(replicates), dtype=np.float64)
    for draw in range(int(replicates)):
        sampled = values[generator.integers(0, len(values), size=len(values))]
        rows = sampled[:, 0].sum()
        delta[draw] = np.sqrt(sampled[:, 2].sum() / rows) - np.sqrt(
            sampled[:, 1].sum() / rows
        )
    return {
        "replicates": int(replicates),
        "kst_days": int(len(values)),
        "ci90_low_c": float(np.quantile(delta, 0.05)),
        "ci90_high_c": float(np.quantile(delta, 0.95)),
        "probability_improved": float(np.mean(delta < 0.0)),
    }


def evaluate_gate(
    *,
    aggregate_delta: float,
    ci90_high: float,
    fold_deltas: dict[str, float],
    layer_deltas: dict[str, float],
    thresholds: dict[str, object],
    posterior_guard_applied: bool,
) -> dict[str, object]:
    checks = {
        "pooled_delta": aggregate_delta <= float(thresholds["pooled_delta_rmse_c_max"]),
        "paired_ci90_upper": ci90_high < float(thresholds["paired_day_bootstrap_ci90_upper_lt"]),
        "improved_folds": sum(value < 0.0 for value in fold_deltas.values())
        >= int(thresholds["improved_folds_min"]),
        "fold_count": len(fold_deltas) == int(thresholds["fold_count"]),
        "worst_layer": max(layer_deltas.values())
        <= float(thresholds["worst_layer_regression_c_max"]),
        "posterior_sd_guard": bool(posterior_guard_applied)
        is bool(thresholds["posterior_sd_guard_required"]),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}
