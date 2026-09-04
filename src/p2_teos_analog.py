"""Leakage-safe physical-state analog transfer for the P2 restoration task.

The density anomaly used here is a deliberately simple, fixed linear equation
of state.  It is only a similarity proxy: it is not UNESCO-1983 density,
TEOS-10, sigma0, or a valid replacement for the ``gsw`` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

PUBLIC_LAYERS = (1, 5, 6, 7, 8)
TARGET_LAYERS = (2, 3, 4)
PUBLIC_PAIRS = ((1, 5), (5, 6), (6, 7), (7, 8))
CADENCE_MINUTES = 10


@dataclass(frozen=True)
class LinearEOSConfig:
    """Coefficients for a fixed linear seawater-density proxy."""

    rho0: float = 1025.0
    reference_temp: float = 15.0
    reference_psal: float = 35.0
    thermal_expansion: float = 2.0e-4
    haline_contraction: float = 7.6e-4


DEFAULT_EOS = LinearEOSConfig()


@dataclass(frozen=True)
class AnalogConfig:
    """Single preregistered analog/local-linear setting."""

    neighbors: int = 128
    pca_components: int = 12
    ridge: float = 2.0
    blend: float = 0.35
    max_normalized_neighbor_distance: float = 3.0
    min_effective_neighbors: float = 32.0
    max_query_missing_fraction: float = 0.45
    minimum_feature_coverage: float = 0.20
    residual_clip_low: float = 0.005
    residual_clip_high: float = 0.995
    batch_size: int = 512
    n_jobs: int = 8
    seed: int = 20260821

    def __post_init__(self) -> None:
        if self.neighbors < 8 or self.pca_components < 2:
            raise ValueError("analog neighborhood/PCA setting is too small")
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        if not 0.0 <= self.residual_clip_low < self.residual_clip_high <= 1.0:
            raise ValueError("residual clipping quantiles are invalid")


@dataclass(frozen=True)
class PublicState:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]


@dataclass(frozen=True)
class CatalogSplit:
    training: np.ndarray
    validation: np.ndarray
    purged: np.ndarray


@dataclass(frozen=True)
class AnalogPrediction:
    residual: np.ndarray
    supported: np.ndarray
    normalized_neighbor_distance: np.ndarray
    effective_neighbors: np.ndarray
    query_missing_fraction: np.ndarray

    def diagnostics(self) -> dict[str, float | int]:
        supported = self.supported
        return {
            "queries": int(len(supported)),
            "supported_queries": int(supported.sum()),
            "supported_share": float(supported.mean()) if len(supported) else 0.0,
            "median_normalized_neighbor_distance": _finite_quantile(
                self.normalized_neighbor_distance, 0.5
            ),
            "p95_normalized_neighbor_distance": _finite_quantile(
                self.normalized_neighbor_distance, 0.95
            ),
            "median_effective_neighbors": _finite_quantile(self.effective_neighbors, 0.5),
            "median_query_missing_fraction": _finite_quantile(
                self.query_missing_fraction, 0.5
            ),
        }


def _finite_quantile(values: np.ndarray, probability: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, probability)) if len(finite) else float("nan")


def linear_density_anomaly(
    temp: np.ndarray | pd.Series,
    psal: np.ndarray | pd.Series,
    *,
    config: LinearEOSConfig = DEFAULT_EOS,
) -> np.ndarray:
    """Return a fixed linear-EOS density anomaly relative to ``rho0``.

    The units are kg m-3-like because the constant coefficients are applied to
    practical salinity and in-situ temperature without pressure correction.
    The result is suitable only as a deterministic analog-search feature.
    """

    temperature = np.asarray(temp, dtype=np.float64)
    salinity = np.asarray(psal, dtype=np.float64)
    temperature, salinity = np.broadcast_arrays(temperature, salinity)
    density = config.rho0 * (
        1.0
        - config.thermal_expansion * (temperature - config.reference_temp)
        + config.haline_contraction * (salinity - config.reference_psal)
    )
    anomaly = density - config.rho0
    invalid = ~np.isfinite(temperature) | ~np.isfinite(salinity)
    return np.where(invalid, np.nan, anomaly)


def _segment_ids(times: pd.DatetimeIndex) -> np.ndarray:
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("state timestamps must be sorted and unique")
    difference = times.to_series().diff().dt.total_seconds().div(60).to_numpy()
    return (
        np.cumsum(np.r_[True, ~np.isclose(difference[1:], CADENCE_MINUTES)]).astype(np.int32)
        - 1
    )


def gap_aware_change(
    values: np.ndarray | pd.Series,
    times: pd.DatetimeIndex,
    horizon_minutes: int,
) -> np.ndarray:
    """Compute a backward change without crossing a non-10-minute gap."""

    source = np.asarray(values, dtype=np.float64)
    if source.shape != (len(times),):
        raise ValueError("gap-aware change input is not time aligned")
    if horizon_minutes <= 0 or horizon_minutes % CADENCE_MINUTES:
        raise ValueError("trajectory horizon must be a positive 10-minute multiple")
    steps = horizon_minutes // CADENCE_MINUTES
    result = np.full(len(source), np.nan, dtype=np.float64)
    if steps >= len(source):
        return result
    segment = _segment_ids(times)
    exact = (
        (times[steps:] - times[:-steps])
        == pd.Timedelta(minutes=horizon_minutes)
    )
    valid = (segment[steps:] == segment[:-steps]) & np.asarray(exact)
    difference = source[steps:] - source[:-steps]
    valid &= np.isfinite(difference)
    result[np.flatnonzero(valid) + steps] = difference[valid]
    return result


def mask_target_interval(
    observations: pd.DataFrame,
    start: str,
    stop: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """In-memory simultaneous temperature/salinity blackout for layers 2-4."""

    required = {"time", "layer", "temp", "psal"}
    if missing := required.difference(observations.columns):
        raise ValueError(f"mask input is missing columns: {sorted(missing)}")
    left = pd.Timestamp(start, tz="Asia/Seoul").tz_convert("UTC")
    right = pd.Timestamp(stop, tz="Asia/Seoul").tz_convert("UTC")
    time = pd.to_datetime(observations["time"], utc=True)
    selected = time.ge(left) & time.lt(right) & observations["layer"].isin(TARGET_LAYERS)
    masked = observations.copy()
    masked.loc[selected, ["temp", "psal"]] = np.nan
    if not masked.loc[selected, ["temp", "psal"]].isna().all().all():
        raise AssertionError("target blackout did not mask temp and psal together")
    return masked, {
        "start_kst": start,
        "stop_kst": stop,
        "duration_days": int((right - left) / pd.Timedelta(days=1)),
        "grid_rows": int(selected.sum()),
        "layers": list(TARGET_LAYERS),
        "variables": ["temp", "psal"],
    }


def _wide(
    public: pd.DataFrame,
    value: str,
    times: pd.DatetimeIndex,
) -> pd.DataFrame:
    return public.pivot(index="_time", columns="layer", values=value).reindex(
        index=times, columns=PUBLIC_LAYERS
    )


def _row_mean(values: np.ndarray) -> np.ndarray:
    count = np.isfinite(values).sum(axis=1)
    return np.divide(
        np.nansum(values, axis=1),
        count,
        out=np.full(len(values), np.nan),
        where=count > 0,
    )


def _row_range(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    populated = np.isfinite(values).any(axis=1)
    if populated.any():
        result[populated] = np.nanmax(values[populated], axis=1) - np.nanmin(
            values[populated], axis=1
        )
    return result


def build_public_state(
    observations: pd.DataFrame,
    *,
    eos: LinearEOSConfig = DEFAULT_EOS,
    trajectory_horizons: tuple[int, ...] = (360, 1440, 10080),
) -> PublicState:
    """Build a public-only static and short-trajectory physical state table."""

    required = {"time", "layer", "temp", "psal", "nominal_depth"}
    if missing := required.difference(observations.columns):
        raise ValueError(f"state input is missing columns: {sorted(missing)}")
    public = observations.loc[
        observations["layer"].isin(PUBLIC_LAYERS),
        ["time", "layer", "temp", "psal", "nominal_depth"],
    ].copy()
    public["_time"] = pd.to_datetime(public["time"], utc=True)
    if public.duplicated(["_time", "layer"]).any():
        raise ValueError("public layer keys are not unique")
    times = pd.DatetimeIndex(public["_time"].drop_duplicates()).sort_values()
    _segment_ids(times)
    temperature = _wide(public, "temp", times)
    salinity = _wide(public, "psal", times)
    nominal = _wide(public, "nominal_depth", times)
    density = pd.DataFrame(
        linear_density_anomaly(temperature.to_numpy(float), salinity.to_numpy(float), config=eos),
        index=times,
        columns=PUBLIC_LAYERS,
    )

    frame = pd.DataFrame(index=times)
    for layer in PUBLIC_LAYERS:
        frame[f"temp_{layer}"] = temperature[layer].to_numpy(float)
        frame[f"psal_{layer}"] = salinity[layer].to_numpy(float)
        frame[f"density_proxy_{layer}"] = density[layer].to_numpy(float)

    for shallow, deep in PUBLIC_PAIRS:
        span = nominal[deep].to_numpy(float) - nominal[shallow].to_numpy(float)
        usable = np.isfinite(span) & (np.abs(span) > 0.1)
        for name, values in (
            ("temp", temperature[deep].to_numpy(float) - temperature[shallow].to_numpy(float)),
            ("psal", salinity[deep].to_numpy(float) - salinity[shallow].to_numpy(float)),
            ("density", density[deep].to_numpy(float) - density[shallow].to_numpy(float)),
        ):
            frame[f"vertical_{name}_gradient_{shallow}_{deep}"] = np.divide(
                values,
                span,
                out=np.full(len(span), np.nan),
                where=usable & np.isfinite(values),
            )

    summary_sources: dict[str, np.ndarray] = {}
    for name, values in (
        ("temp", temperature.to_numpy(float)),
        ("psal", salinity.to_numpy(float)),
        ("density", density.to_numpy(float)),
    ):
        summary_sources[f"{name}_mean"] = _row_mean(values)
        summary_sources[f"{name}_range"] = _row_range(values)
    for name, values in summary_sources.items():
        frame[name] = values

    trajectory_sources = tuple(frame.columns)
    trajectory: dict[str, np.ndarray] = {}
    for horizon in trajectory_horizons:
        for column in trajectory_sources:
            trajectory[f"change_{horizon}m__{column}"] = gap_aware_change(
                frame[column].to_numpy(float), times, horizon
            )

    kst = times.tz_convert("Asia/Seoul")
    minute = kst.hour.to_numpy() * 60 + kst.minute.to_numpy()
    day = kst.dayofyear.to_numpy() + minute / 1440.0
    epoch_seconds = times.as_unit("ns").asi8 / 1e9
    cyclic = {
        "doy_sin": np.sin(2 * np.pi * day / 365.2425),
        "doy_cos": np.cos(2 * np.pi * day / 365.2425),
        "m2_sin": np.sin(2 * np.pi * epoch_seconds / (12.42 * 3600.0)),
        "m2_cos": np.cos(2 * np.pi * epoch_seconds / (12.42 * 3600.0)),
    }
    frame = pd.concat(
        [frame, pd.DataFrame(trajectory, index=times), pd.DataFrame(cyclic, index=times)], axis=1
    )
    features = tuple(frame.columns)
    forbidden = {
        f"temp_{layer}" for layer in TARGET_LAYERS
    } | {f"psal_{layer}" for layer in TARGET_LAYERS}
    if forbidden.intersection(features):
        raise AssertionError("target-layer values leaked into public state")
    return PublicState(frame.astype(np.float64), features)


def catalog_split(
    times: pd.DatetimeIndex,
    target_mask: np.ndarray,
    start: str,
    stop: str,
    *,
    purge_days: int,
) -> CatalogSplit:
    """Create an outer-train-only complete-profile catalog with a time purge."""

    labels = np.asarray(target_mask, dtype=bool)
    if labels.shape != (len(times), len(TARGET_LAYERS)):
        raise ValueError("target mask must be time x three target layers")
    left = pd.Timestamp(start, tz="Asia/Seoul").tz_convert("UTC")
    right = pd.Timestamp(stop, tz="Asia/Seoul").tz_convert("UTC")
    validation = np.asarray((times >= left) & (times < right), dtype=bool)
    purged = np.asarray(
        (times >= left - pd.Timedelta(days=purge_days))
        & (times < right + pd.Timedelta(days=purge_days)),
        dtype=bool,
    )
    training = labels.all(axis=1) & ~purged
    if np.any(training & validation):
        raise AssertionError("validation timestamps entered the analog catalog")
    return CatalogSplit(training, validation, purged)


@dataclass
class RobustPCAState:
    feature_names: tuple[str, ...]
    selected: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    pca: PCA
    component_scale: np.ndarray

    @classmethod
    def fit(
        cls,
        state: pd.DataFrame,
        feature_names: tuple[str, ...],
        *,
        components: int,
        minimum_coverage: float,
        seed: int,
    ) -> tuple[RobustPCAState, np.ndarray]:
        values = state.loc[:, feature_names].to_numpy(float)
        coverage = np.isfinite(values).mean(axis=0)
        center = np.nanmedian(values, axis=0)
        center = np.where(np.isfinite(center), center, 0.0)
        scale = np.nanmedian(np.abs(values - center), axis=0) * 1.4826
        fallback = np.nanstd(values, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback)
        selected = (coverage >= minimum_coverage) & np.isfinite(scale) & (scale > 1e-8)
        if selected.sum() < components:
            raise ValueError("too few covered physical-state features for fixed PCA")
        normalized, _ = cls._normalize(values, selected, center, scale)
        count = min(components, normalized.shape[1], len(normalized) - 1)
        pca = PCA(
            n_components=count,
            svd_solver="randomized",
            iterated_power=4,
            random_state=seed,
        )
        projected = pca.fit_transform(normalized)
        component_scale = np.std(projected, axis=0)
        component_scale = np.where(component_scale > 1e-8, component_scale, 1.0)
        projected /= component_scale
        fitted = cls(feature_names, selected, center, scale, pca, component_scale)
        return fitted, projected

    @staticmethod
    def _normalize(
        values: np.ndarray,
        selected: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        current = values[:, selected]
        missing = ~np.isfinite(current)
        standardized = np.divide(
            current - center[selected],
            scale[selected],
            out=np.zeros_like(current),
            where=~missing,
        )
        standardized = np.clip(standardized, -12.0, 12.0)
        return np.column_stack((standardized, missing.astype(float))), missing.mean(axis=1)

    def transform(self, state: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if tuple(state.columns) != self.feature_names:
            raise ValueError("public-state feature schema differs from fitted catalog")
        values = state.to_numpy(float)
        normalized, missing = self._normalize(values, self.selected, self.center, self.scale)
        return self.pca.transform(normalized) / self.component_scale, missing


@dataclass
class AnalogResidualModel:
    config: AnalogConfig
    transform: RobustPCAState
    neighbor_index: NearestNeighbors
    catalog_state: np.ndarray
    catalog_residual: np.ndarray
    residual_lower: np.ndarray
    residual_upper: np.ndarray

    @classmethod
    def fit(
        cls,
        state: PublicState,
        residual: np.ndarray,
        *,
        config: AnalogConfig,
    ) -> AnalogResidualModel:
        labels = np.asarray(residual, dtype=np.float64)
        if labels.shape != (len(state.frame), len(TARGET_LAYERS)):
            raise ValueError("catalog residual must be rows x three target layers")
        complete = np.isfinite(labels).all(axis=1)
        if complete.sum() < config.neighbors:
            raise ValueError("catalog has insufficient complete target profiles")
        training = state.frame.loc[complete, state.feature_columns]
        transform, projected = RobustPCAState.fit(
            training,
            state.feature_columns,
            components=config.pca_components,
            minimum_coverage=config.minimum_feature_coverage,
            seed=config.seed,
        )
        target = labels[complete]
        lower = np.quantile(target, config.residual_clip_low, axis=0)
        upper = np.quantile(target, config.residual_clip_high, axis=0)
        neighbors = NearestNeighbors(
            n_neighbors=config.neighbors,
            algorithm="brute",
            metric="euclidean",
            n_jobs=config.n_jobs,
        ).fit(projected)
        return cls(config, transform, neighbors, projected, target, lower, upper)

    def predict(self, state: PublicState) -> AnalogPrediction:
        query_frame = state.frame.loc[:, state.feature_columns]
        projected, missing = self.transform.transform(query_frame)
        distance, indices = self.neighbor_index.kneighbors(projected, return_distance=True)
        normalized_distance = distance[:, -1] / np.sqrt(projected.shape[1])
        bandwidth = np.maximum(distance[:, -1], 1e-8)
        weights = np.exp(-0.5 * (distance / bandwidth[:, None]) ** 2)
        effective = np.square(weights.sum(axis=1)) / np.square(weights).sum(axis=1)
        supported = (
            (normalized_distance <= self.config.max_normalized_neighbor_distance)
            & (effective >= self.config.min_effective_neighbors)
            & (missing <= self.config.max_query_missing_fraction)
        )
        prediction = np.full((len(projected), len(TARGET_LAYERS)), np.nan, dtype=np.float64)
        supported_rows = np.flatnonzero(supported)
        for start in range(0, len(supported_rows), self.config.batch_size):
            rows = supported_rows[start : start + self.config.batch_size]
            neighbor_state = self.catalog_state[indices[rows]]
            delta = neighbor_state - projected[rows, None, :]
            design = np.concatenate(
                (np.ones((len(rows), self.config.neighbors, 1)), delta), axis=2
            )
            current_weights = weights[rows]
            gram = np.einsum("bki,bk,bkj->bij", design, current_weights, design)
            rhs = np.einsum(
                "bki,bk,bko->bio",
                design,
                current_weights,
                self.catalog_residual[indices[rows]],
            )
            penalty = np.eye(design.shape[2], dtype=np.float64) * self.config.ridge
            penalty[0, 0] = 0.0
            coefficients = np.linalg.solve(gram + penalty[None, :, :], rhs)
            prediction[rows] = coefficients[:, 0, :]
        prediction = np.clip(prediction, self.residual_lower, self.residual_upper)
        return AnalogPrediction(prediction, supported, normalized_distance, effective, missing)


def blend_with_frozen(
    frozen: np.ndarray,
    analog_absolute: np.ndarray,
    supported: np.ndarray,
    *,
    blend: float,
) -> np.ndarray:
    """Blend supported analog estimates and preserve fallback rows bit-for-bit."""

    base = np.asarray(frozen, dtype=np.float64)
    analog = np.asarray(analog_absolute, dtype=np.float64)
    keep = np.asarray(supported, dtype=bool)
    if base.shape != analog.shape or keep.shape != base.shape:
        raise ValueError("frozen/analog/support arrays are not aligned")
    output = base.copy()
    usable = keep & np.isfinite(analog)
    output[usable] = base[usable] + blend * (analog[usable] - base[usable])
    if not np.array_equal(output[~usable], base[~usable]):
        raise AssertionError("unsupported frozen fallback changed")
    return output


__all__ = [
    "AnalogConfig",
    "AnalogPrediction",
    "AnalogResidualModel",
    "CatalogSplit",
    "LinearEOSConfig",
    "PublicState",
    "blend_with_frozen",
    "build_public_state",
    "catalog_split",
    "gap_aware_change",
    "linear_density_anomaly",
    "mask_target_interval",
]
