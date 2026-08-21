"""Public-only observability precheck and a compact tide-aware RTS smoother for P2.

This module is deliberately independent from the frozen P2 inference code.  It
never reads target-layer temperature or salinity as an input inside a pseudo
blackout.  Target values are accepted only as outer-train supervision or as
outer-validation labels after prediction.

The model is a small linear-Gaussian residual model, not BayOTIDE and not a
TEOS-10 approximation.  Public temperature, practical salinity, and measured
depth are decomposed into slow PCA factors plus one fixed 12.42-hour resonator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

PUBLIC_LAYERS = (1, 5, 6, 7, 8)
TARGET_LAYERS = (2, 3, 4)
PUBLIC_VARIABLES = ("temp", "psal", "depth")
TARGET_VARIABLES = ("temp", "psal")
CADENCE_MINUTES = 10
M2_HOURS = 12.42


@dataclass(frozen=True)
class TideRTSConfig:
    factors: int = 4
    ridge_alpha: float = 10.0
    minimum_feature_coverage: float = 0.50
    slow_phi_min: float = 0.90
    slow_phi_max: float = 0.9999
    resonator_damping: float = 0.9995
    resonator_process_variance: float = 1e-4
    gramian_horizon_steps: int = 1008
    gramian_rank_tolerance: float = 1e-8
    gramian_condition_max: float = 1e8
    support_quantile: float = 0.99
    minimum_support_share: float = 0.80
    minimum_two_temp_coverage: float = 0.95
    posterior_sd_scale_max: float = 2.5

    def __post_init__(self) -> None:
        if self.factors < 1:
            raise ValueError("at least one slow factor is required")
        if self.ridge_alpha < 0:
            raise ValueError("ridge alpha must be non-negative")
        if not 0 < self.minimum_feature_coverage <= 1:
            raise ValueError("minimum feature coverage must be in (0, 1]")
        if not 0 < self.resonator_damping <= 1:
            raise ValueError("resonator damping must be in (0, 1]")
        if self.gramian_horizon_steps < 1:
            raise ValueError("observability horizon must be positive")


@dataclass(frozen=True)
class P2TidePanel:
    times: pd.DatetimeIndex
    public_values: np.ndarray
    public_feature_names: tuple[str, ...]
    public_temp: np.ndarray
    public_psal: np.ndarray
    public_depth: np.ndarray
    target_temp: np.ndarray
    target_psal: np.ndarray
    target_depth: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.times)
        if self.public_values.shape != (rows, len(self.public_feature_names)):
            raise ValueError("public feature panel shape is inconsistent")
        for values, columns in (
            (self.public_temp, len(PUBLIC_LAYERS)),
            (self.public_psal, len(PUBLIC_LAYERS)),
            (self.public_depth, len(PUBLIC_LAYERS)),
            (self.target_temp, len(TARGET_LAYERS)),
            (self.target_psal, len(TARGET_LAYERS)),
            (self.target_depth, len(TARGET_LAYERS)),
        ):
            if values.shape != (rows, columns):
                raise ValueError("P2 tide panel matrix shape is inconsistent")
        forbidden = {
            f"{variable}_{layer}" for variable in TARGET_VARIABLES for layer in TARGET_LAYERS
        }
        if forbidden.intersection(self.public_feature_names):
            raise ValueError("target temperature/salinity leaked into public features")


@dataclass(frozen=True)
class OuterSplit:
    training: np.ndarray
    validation: np.ndarray
    purged: np.ndarray


@dataclass(frozen=True)
class PublicFactorEncoder:
    feature_names: tuple[str, ...]
    selected: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    harmonic_coefficients: np.ndarray
    pca: PCA
    transition: np.ndarray
    process_covariance: np.ndarray
    public_loading: np.ndarray
    public_noise: np.ndarray

    @property
    def state_dimension(self) -> int:
        return int(self.transition.shape[0])

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        feature_names: tuple[str, ...],
        times: pd.DatetimeIndex,
        training: np.ndarray,
        *,
        config: TideRTSConfig,
    ) -> PublicFactorEncoder:
        matrix = np.asarray(values, dtype=np.float64)
        selected_rows = np.asarray(training, dtype=bool)
        if matrix.shape != (len(times), len(feature_names)):
            raise ValueError("public encoder input is not aligned")
        if selected_rows.shape != (len(times),) or selected_rows.sum() < 100:
            raise ValueError("public encoder training support is insufficient")

        train = matrix[selected_rows]
        coverage = np.isfinite(train).mean(axis=0)
        center = np.nanmedian(train, axis=0)
        center = np.where(np.isfinite(center), center, 0.0)
        scale = np.nanmedian(np.abs(train - center), axis=0) * 1.4826
        fallback = np.nanstd(train, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback)
        selected = (
            (coverage >= config.minimum_feature_coverage) & np.isfinite(scale) & (scale > 1e-8)
        )
        if selected.sum() < config.factors + 2:
            raise ValueError("too few covered public T/S/depth features")

        standardized = _standardize(matrix, selected, center, scale)
        phase = m2_phase(times)
        harmonic = np.linalg.lstsq(phase[selected_rows], standardized[selected_rows], rcond=None)[0]
        deharmonized = standardized - phase @ harmonic
        factors = min(config.factors, int(selected.sum()), int(selected_rows.sum() - 1))
        pca = PCA(n_components=factors, svd_solver="full")
        pca.fit(deharmonized[selected_rows])
        score = pca.transform(deharmonized)

        valid_pairs = selected_rows[1:] & selected_rows[:-1] & _exact_cadence(times)
        phi = np.empty(factors, dtype=np.float64)
        innovation = np.empty(factors, dtype=np.float64)
        for column in range(factors):
            prior = score[:-1, column][valid_pairs]
            current = score[1:, column][valid_pairs]
            denominator = float(np.dot(prior, prior))
            estimate = float(np.dot(prior, current) / denominator) if denominator > 1e-10 else 0.99
            phi[column] = np.clip(estimate, config.slow_phi_min, config.slow_phi_max)
            residual = current - phi[column] * prior
            innovation[column] = max(float(np.var(residual)), 1e-6)

        angle = 2 * np.pi * CADENCE_MINUTES / (M2_HOURS * 60.0)
        rotation = config.resonator_damping * np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]],
            dtype=np.float64,
        )
        transition = np.zeros((factors + 2, factors + 2), dtype=np.float64)
        transition[:factors, :factors] = np.diag(phi)
        transition[factors:, factors:] = rotation
        process = np.diag(np.r_[innovation, [config.resonator_process_variance] * 2])
        public_loading = np.column_stack((pca.components_.T, harmonic.T))
        reconstructed = np.column_stack((score, phase)) @ public_loading.T
        public_residual = standardized[selected_rows] - reconstructed[selected_rows]
        public_noise = np.nanvar(public_residual, axis=0)
        public_noise = np.where(
            np.isfinite(public_noise) & (public_noise > 1e-6), public_noise, 1e-3
        )
        return cls(
            feature_names,
            selected,
            center,
            scale,
            harmonic,
            pca,
            transition,
            process,
            public_loading,
            public_noise,
        )

    def transform(
        self, values: np.ndarray, times: pd.DatetimeIndex
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape != (len(times), len(self.feature_names)):
            raise ValueError("public transform input is not aligned")
        standardized = _standardize(matrix, self.selected, self.center, self.scale)
        phase = m2_phase(times)
        deharmonized = standardized - phase @ self.harmonic_coefficients
        score = self.pca.transform(deharmonized)
        state = np.column_stack((score, phase))
        selected_values = matrix[:, self.selected]
        observed_share = np.isfinite(selected_values).mean(axis=1)
        return state, observed_share

    def standardized_public(self, values: np.ndarray) -> np.ndarray:
        return _standardize(
            np.asarray(values, dtype=np.float64), self.selected, self.center, self.scale
        )


@dataclass(frozen=True)
class ResidualRegressor:
    coefficients: np.ndarray
    intercept: np.ndarray
    residual_scale: np.ndarray
    label_center: np.ndarray
    label_scale: np.ndarray

    @classmethod
    def fit(
        cls,
        state: np.ndarray,
        labels: np.ndarray,
        training: np.ndarray,
        *,
        alpha: float,
    ) -> ResidualRegressor:
        design = np.asarray(state, dtype=np.float64)
        target = np.asarray(labels, dtype=np.float64)
        if target.ndim == 1:
            target = target[:, None]
        if design.shape[0] != target.shape[0]:
            raise ValueError("residual labels are not state aligned")
        selected = np.asarray(training, dtype=bool)
        coefficients = np.zeros((design.shape[1], target.shape[1]), dtype=np.float64)
        intercept = np.zeros(target.shape[1], dtype=np.float64)
        residual_scale = np.ones(target.shape[1], dtype=np.float64)
        label_center = np.zeros(target.shape[1], dtype=np.float64)
        label_scale = np.ones(target.shape[1], dtype=np.float64)
        for column in range(target.shape[1]):
            keep = selected & np.isfinite(target[:, column]) & np.isfinite(design).all(axis=1)
            if keep.sum() < max(100, design.shape[1] * 10):
                raise ValueError(f"residual output {column} has insufficient outer-train rows")
            current = target[keep, column]
            label_center[column] = float(np.median(current))
            robust = float(np.median(np.abs(current - label_center[column])) * 1.4826)
            label_scale[column] = robust if robust > 1e-6 else max(float(np.std(current)), 1e-3)
            normalized = (current - label_center[column]) / label_scale[column]
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(design[keep], normalized)
            coefficients[:, column] = np.asarray(model.coef_, dtype=np.float64)
            intercept[column] = float(model.intercept_)
            fitted = model.predict(design[keep])
            residual_scale[column] = max(float(np.std(normalized - fitted)), 1e-3)
        return cls(coefficients, intercept, residual_scale, label_center, label_scale)

    def predict(self, state: np.ndarray) -> np.ndarray:
        normalized = np.asarray(state, dtype=np.float64) @ self.coefficients + self.intercept
        return normalized * self.label_scale + self.label_center

    def normalized_loading(self) -> np.ndarray:
        return self.coefficients.T

    def normalized_observations(self, labels: np.ndarray) -> np.ndarray:
        target = np.asarray(labels, dtype=np.float64)
        return (target - self.label_center) / self.label_scale - self.intercept


@dataclass(frozen=True)
class RTSResult:
    mean: np.ndarray
    covariance: np.ndarray


def build_tide_panel(observations: pd.DataFrame) -> P2TidePanel:
    """Create a time-aligned panel whose inputs contain public layers only."""

    required = {"time", "layer", "temp", "psal", "depth"}
    if missing := required.difference(observations.columns):
        raise ValueError(f"P2 observations are missing columns: {sorted(missing)}")
    keyed = observations.loc[:, sorted(required)].copy()
    keyed["_time"] = pd.to_datetime(keyed["time"], utc=True)
    if keyed.duplicated(["_time", "layer"]).any():
        raise ValueError("P2 observation keys are not unique")
    times = pd.DatetimeIndex(keyed["_time"].drop_duplicates()).sort_values()

    wide: dict[str, pd.DataFrame] = {}
    for variable in PUBLIC_VARIABLES:
        wide[variable] = keyed.pivot(index="_time", columns="layer", values=variable).reindex(times)

    public_arrays: dict[str, np.ndarray] = {
        variable: np.column_stack(
            [wide[variable][layer].to_numpy(float) for layer in PUBLIC_LAYERS]
        )
        for variable in PUBLIC_VARIABLES
    }
    features: list[np.ndarray] = []
    names: list[str] = []
    for variable in PUBLIC_VARIABLES:
        for offset, layer in enumerate(PUBLIC_LAYERS):
            features.append(public_arrays[variable][:, offset])
            names.append(f"{variable}_{layer}")
            features.append(np.isfinite(public_arrays[variable][:, offset]).astype(float))
            names.append(f"{variable}_{layer}_mask")

    target_temp = np.column_stack([wide["temp"][layer].to_numpy(float) for layer in TARGET_LAYERS])
    target_psal = np.column_stack([wide["psal"][layer].to_numpy(float) for layer in TARGET_LAYERS])
    target_depth = np.column_stack(
        [wide["depth"][layer].to_numpy(float) for layer in TARGET_LAYERS]
    )
    return P2TidePanel(
        times=times,
        public_values=np.column_stack(features),
        public_feature_names=tuple(names),
        public_temp=public_arrays["temp"],
        public_psal=public_arrays["psal"],
        public_depth=public_arrays["depth"],
        target_temp=target_temp,
        target_psal=target_psal,
        target_depth=target_depth,
    )


def outer_split(
    times: pd.DatetimeIndex,
    start: str,
    stop: str,
    *,
    purge_days: int,
) -> OuterSplit:
    left = pd.Timestamp(start, tz="Asia/Seoul").tz_convert("UTC")
    right = pd.Timestamp(stop, tz="Asia/Seoul").tz_convert("UTC")
    validation = np.asarray((times >= left) & (times < right), dtype=bool)
    purged = np.asarray(
        (times >= left - pd.Timedelta(days=purge_days))
        & (times < right + pd.Timedelta(days=purge_days)),
        dtype=bool,
    )
    training = ~purged
    if np.any(training & validation):
        raise AssertionError("outer validation entered P2 precheck training")
    return OuterSplit(training, validation, purged)


def m2_phase(times: pd.DatetimeIndex) -> np.ndarray:
    seconds = times.as_unit("ns").asi8 / 1e9
    angle = 2 * np.pi * seconds / (M2_HOURS * 3600.0)
    return np.column_stack((np.sin(angle), np.cos(angle)))


def actual_depth_interpolation(
    public_value: np.ndarray,
    public_depth: np.ndarray,
    target_depth: np.ndarray,
) -> np.ndarray:
    """Interpolate a variable using measured public depths and fixed query depths."""

    values = np.asarray(public_value, dtype=np.float64)
    depths = np.asarray(public_depth, dtype=np.float64)
    queries = np.asarray(target_depth, dtype=np.float64)
    if values.shape != depths.shape or values.shape[1] != len(PUBLIC_LAYERS):
        raise ValueError("public value/depth matrices are inconsistent")
    if queries.shape != (len(values), len(TARGET_LAYERS)):
        raise ValueError("target depth query matrix is inconsistent")
    result = np.full(queries.shape, np.nan, dtype=np.float64)
    for row in range(len(values)):
        valid = np.isfinite(values[row]) & np.isfinite(depths[row])
        if valid.sum() < 2:
            continue
        x = depths[row, valid]
        y = values[row, valid]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        unique = np.r_[True, np.diff(x) > 1e-6]
        x = x[unique]
        y = y[unique]
        if len(x) < 2:
            continue
        for column, query in enumerate(queries[row]):
            if np.isfinite(query) and x[0] <= query <= x[-1]:
                result[row, column] = float(np.interp(query, x, y))
    return result


def fold_target_depths(panel: P2TidePanel, training: np.ndarray) -> np.ndarray:
    """Use only outer-train measured target depths to define query coordinates."""

    selected = np.asarray(training, dtype=bool)
    medians = np.nanmedian(panel.target_depth[selected], axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("outer-train target depth medians are unavailable")
    return np.broadcast_to(medians, (len(panel.times), len(TARGET_LAYERS))).copy()


def residual_skill_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Skill against an exact zero residual (the frozen/base prediction)."""

    actual = np.asarray(truth, dtype=np.float64)
    current = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(actual) & np.isfinite(current)
    if valid.sum() < 2:
        return float("nan")
    denominator = float(np.sum(np.square(actual[valid])))
    if denominator <= 1e-12:
        return float("nan")
    return 1.0 - float(np.sum(np.square(actual[valid] - current[valid]))) / denominator


def observability_diagnostics(
    transition: np.ndarray,
    loading: np.ndarray,
    noise: np.ndarray,
    *,
    horizon_steps: int,
    rank_tolerance: float,
) -> dict[str, Any]:
    state = transition.shape[0]
    if transition.shape != (state, state) or loading.shape[1] != state:
        raise ValueError("observability matrices are inconsistent")
    variance = np.asarray(noise, dtype=np.float64)
    if variance.shape != (loading.shape[0],):
        raise ValueError("public observation noise is inconsistent")
    weighted = loading.T @ (loading / variance[:, None])
    gramian = np.zeros((state, state), dtype=np.float64)
    power = np.eye(state, dtype=np.float64)
    for _ in range(horizon_steps):
        gramian += power.T @ weighted @ power
        power = transition @ power
    gramian = (gramian + gramian.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(gramian)
    largest = max(float(eigenvalues[-1]), 0.0)
    threshold = max(largest * rank_tolerance, 1e-12)
    positive = eigenvalues[eigenvalues > threshold]
    rank = int(len(positive))
    condition = float(largest / positive[0]) if len(positive) else float("inf")
    return {
        "state_dimension": int(state),
        "rank": rank,
        "condition": condition,
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "rank_threshold": threshold,
    }


def support_diagnostics(
    state: np.ndarray,
    training: np.ndarray,
    validation: np.ndarray,
    *,
    quantile: float,
) -> dict[str, float | int]:
    score = np.asarray(state, dtype=np.float64)
    train = score[np.asarray(training, dtype=bool)]
    valid = score[np.asarray(validation, dtype=bool)]
    if len(train) < score.shape[1] * 5 or len(valid) == 0:
        raise ValueError("support diagnostic rows are insufficient")
    center = np.mean(train, axis=0)
    covariance = np.cov(train, rowvar=False)
    covariance = np.atleast_2d(covariance) + np.eye(score.shape[1]) * 1e-6
    inverse = np.linalg.pinv(covariance, rcond=1e-10)

    def distance(values: np.ndarray) -> np.ndarray:
        delta = values - center
        return np.einsum("ni,ij,nj->n", delta, inverse, delta)

    train_distance = distance(train)
    validation_distance = distance(valid)
    threshold = float(np.quantile(train_distance, quantile))
    return {
        "training_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "distance_threshold": threshold,
        "validation_supported_share": float(np.mean(validation_distance <= threshold)),
        "validation_distance_median": float(np.median(validation_distance)),
        "validation_distance_p95": float(np.quantile(validation_distance, 0.95)),
    }


def m2_relationship_diagnostics(
    times: pd.DatetimeIndex,
    public: np.ndarray,
    target: np.ndarray,
    training: np.ndarray,
    *,
    window_days: tuple[int, ...] = (30, 61),
) -> dict[str, Any]:
    """Estimate replicate-window M2 coherence and circular phase stability.

    A single-window harmonic coefficient cannot identify coherence.  We treat
    non-overlapping 30/61-day windows as replicates and aggregate their complex
    coefficients.  This is a diagnostic only; it does not tune the model.
    """

    source = np.asarray(public, dtype=np.float64)
    hidden = np.asarray(target, dtype=np.float64)
    selected = np.asarray(training, dtype=bool)
    if source.shape[0] != len(times) or hidden.shape[0] != len(times):
        raise ValueError("M2 diagnostic arrays are not time aligned")
    result: dict[str, Any] = {}
    for days in window_days:
        pair_rows: list[dict[str, Any]] = []
        for target_column in range(hidden.shape[1]):
            best: dict[str, Any] | None = None
            for public_column in range(source.shape[1]):
                public_coefficients: list[complex] = []
                target_coefficients: list[complex] = []
                for indices in _calendar_windows(times, selected, days):
                    x = source[indices, public_column]
                    y = hidden[indices, target_column]
                    valid = np.isfinite(x) & np.isfinite(y)
                    expected = days * 24 * 60 // CADENCE_MINUTES
                    if valid.sum() < max(100, int(expected * 0.50)):
                        continue
                    seconds = times[indices][valid].as_unit("ns").asi8 / 1e9
                    public_coefficients.append(_harmonic_coefficient(seconds, x[valid]))
                    target_coefficients.append(_harmonic_coefficient(seconds, y[valid]))
                if len(public_coefficients) < 2:
                    continue
                xcoef = np.asarray(public_coefficients, dtype=np.complex128)
                ycoef = np.asarray(target_coefficients, dtype=np.complex128)
                cross = ycoef * np.conj(xcoef)
                denominator = float(np.sum(np.abs(xcoef) ** 2) * np.sum(np.abs(ycoef) ** 2))
                coherence = (
                    float(np.abs(np.sum(cross)) ** 2 / denominator) if denominator > 1e-20 else 0.0
                )
                weight = np.abs(cross)
                stability = (
                    float(np.abs(np.sum(weight * np.exp(1j * np.angle(cross)))) / np.sum(weight))
                    if np.sum(weight) > 1e-20
                    else 0.0
                )
                current = {
                    "target_column": int(target_column),
                    "public_column": int(public_column),
                    "windows": int(len(xcoef)),
                    "coherence": coherence,
                    "phase_stability": stability,
                    "phase_radians": float(np.angle(np.sum(cross))),
                }
                if best is None or coherence > float(best["coherence"]):
                    best = current
            pair_rows.append(
                best
                if best is not None
                else {
                    "target_column": int(target_column),
                    "public_column": -1,
                    "windows": 0,
                    "coherence": float("nan"),
                    "phase_stability": float("nan"),
                    "phase_radians": float("nan"),
                }
            )
        finite_coherence = np.array([row["coherence"] for row in pair_rows], dtype=np.float64)
        finite_stability = np.array([row["phase_stability"] for row in pair_rows], dtype=np.float64)
        result[str(days)] = {
            "targets": pair_rows,
            "median_best_coherence": _finite_median(finite_coherence),
            "median_phase_stability": _finite_median(finite_stability),
        }
    return result


def kalman_rts_smoother(
    observations: np.ndarray,
    observed: np.ndarray,
    loading: np.ndarray,
    observation_noise: np.ndarray,
    transition: np.ndarray,
    process_covariance: np.ndarray,
    *,
    initial_mean: np.ndarray | None = None,
    initial_covariance: np.ndarray | None = None,
) -> RTSResult:
    """Deterministic linear-Gaussian Kalman filter followed by RTS smoothing."""

    values = np.asarray(observations, dtype=np.float64)
    mask = np.asarray(observed, dtype=bool)
    h = np.asarray(loading, dtype=np.float64)
    r = np.asarray(observation_noise, dtype=np.float64)
    f = np.asarray(transition, dtype=np.float64)
    q = np.asarray(process_covariance, dtype=np.float64)
    rows, channels = values.shape
    states = f.shape[0]
    if mask.shape != values.shape or h.shape != (channels, states):
        raise ValueError("Kalman observation shapes are inconsistent")
    if r.shape != (channels,) or f.shape != (states, states) or q.shape != (states, states):
        raise ValueError("Kalman covariance shapes are inconsistent")

    filtered_mean = np.zeros((rows, states), dtype=np.float64)
    filtered_covariance = np.zeros((rows, states, states), dtype=np.float64)
    predicted_mean = np.zeros_like(filtered_mean)
    predicted_covariance = np.zeros_like(filtered_covariance)
    mean = np.zeros(states, dtype=np.float64) if initial_mean is None else initial_mean.copy()
    covariance = (
        np.eye(states, dtype=np.float64) * 10.0
        if initial_covariance is None
        else initial_covariance.copy()
    )
    identity = np.eye(states, dtype=np.float64)
    for row in range(rows):
        if row:
            mean = f @ mean
            covariance = f @ covariance @ f.T + q
        predicted_mean[row] = mean
        predicted_covariance[row] = covariance
        keep = mask[row] & np.isfinite(values[row])
        if keep.any():
            current_h = h[keep]
            innovation = values[row, keep] - current_h @ mean
            covariance_y = current_h @ covariance @ current_h.T + np.diag(r[keep])
            gain = np.linalg.solve(covariance_y, current_h @ covariance).T
            mean = mean + gain @ innovation
            joseph = identity - gain @ current_h
            covariance = joseph @ covariance @ joseph.T + gain @ np.diag(r[keep]) @ gain.T
            covariance = (covariance + covariance.T) * 0.5
        filtered_mean[row] = mean
        filtered_covariance[row] = covariance

    smooth_mean = filtered_mean.copy()
    smooth_covariance = filtered_covariance.copy()
    for row in range(rows - 2, -1, -1):
        gain = np.linalg.solve(predicted_covariance[row + 1], f @ filtered_covariance[row]).T
        smooth_mean[row] += gain @ (smooth_mean[row + 1] - predicted_mean[row + 1])
        smooth_covariance[row] += (
            gain @ (smooth_covariance[row + 1] - predicted_covariance[row + 1]) @ gain.T
        )
        smooth_covariance[row] = (smooth_covariance[row] + smooth_covariance[row].T) * 0.5
    return RTSResult(smooth_mean, smooth_covariance)


def posterior_target_sd(
    covariance: np.ndarray,
    loading: np.ndarray,
    residual_scale: np.ndarray,
    label_scale: np.ndarray,
) -> np.ndarray:
    cov = np.asarray(covariance, dtype=np.float64)
    h = np.asarray(loading, dtype=np.float64)
    state_variance = np.einsum("oi,nij,oj->no", h, cov, h)
    normalized = np.maximum(state_variance + np.square(residual_scale)[None, :], 0.0)
    return np.sqrt(normalized) * label_scale[None, :]


def exact_fallback(
    frozen: np.ndarray,
    correction: np.ndarray,
    supported: np.ndarray,
) -> np.ndarray:
    base = np.asarray(frozen, dtype=np.float64)
    adjustment = np.asarray(correction, dtype=np.float64)
    keep = np.asarray(supported, dtype=bool)
    if base.shape != adjustment.shape or base.shape != keep.shape:
        raise ValueError("fallback arrays are not aligned")
    result = base.copy()
    result[keep] = base[keep] + adjustment[keep]
    if not np.array_equal(result[~keep], base[~keep], equal_nan=True):
        raise AssertionError("unsupported P2 rows changed from frozen")
    return result


def cadence_segments(times: pd.DatetimeIndex) -> tuple[tuple[int, int], ...]:
    """Return half-open exact-10-minute segments; state never crosses a gap."""

    if len(times) == 0:
        return ()
    exact = _exact_cadence(times)
    starts = np.r_[0, np.flatnonzero(~exact) + 1]
    stops = np.r_[starts[1:], len(times)]
    return tuple(zip(starts.tolist(), stops.tolist(), strict=True))


def _standardize(
    values: np.ndarray,
    selected: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    current = values[:, selected]
    standardized = np.divide(
        current - center[selected],
        scale[selected],
        out=np.zeros_like(current, dtype=np.float64),
        where=np.isfinite(current),
    )
    return np.clip(standardized, -12.0, 12.0)


def _exact_cadence(times: pd.DatetimeIndex) -> np.ndarray:
    if len(times) < 2:
        return np.zeros(0, dtype=bool)
    return np.asarray((times[1:] - times[:-1]) == pd.Timedelta(minutes=CADENCE_MINUTES), dtype=bool)


def _calendar_windows(
    times: pd.DatetimeIndex,
    selected: np.ndarray,
    days: int,
) -> list[np.ndarray]:
    keep = np.flatnonzero(selected)
    if len(keep) == 0:
        return []
    result: list[np.ndarray] = []
    breaks = np.r_[0, np.flatnonzero(np.diff(keep) > 1) + 1, len(keep)]
    width = pd.Timedelta(days=days)
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        run = keep[left:right]
        if len(run) < 2:
            continue
        start = times[run[0]]
        stop = times[run[-1]] + pd.Timedelta(minutes=CADENCE_MINUTES)
        while start + width <= stop + pd.Timedelta(minutes=1):
            rows = np.flatnonzero(selected & np.asarray((times >= start) & (times < start + width)))
            if len(rows):
                result.append(rows)
            start += width
    return result


def _harmonic_coefficient(seconds: np.ndarray, values: np.ndarray) -> complex:
    time = np.asarray(seconds, dtype=np.float64)
    current = np.asarray(values, dtype=np.float64)
    relative = (time - time.mean()) / 86400.0
    design = np.column_stack((np.ones(len(time)), relative))
    detrended = current - design @ np.linalg.lstsq(design, current, rcond=None)[0]
    angle = 2 * np.pi * time / (M2_HOURS * 3600.0)
    return complex(2.0 * np.mean(detrended * np.exp(-1j * angle)))


def _finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else float("nan")
