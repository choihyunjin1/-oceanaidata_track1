"""Train-only causal spectral random-feature kernel for P3 research.

This module intentionally contains no test-data loader and no submission writer.
It maps the observed 48-hour context to fixed multiresolution summaries, then fits
one closed-form multi-output random Fourier feature ridge model.  All centering and
scaling state is learned from the supplied training IDs only.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .sequences import CONTEXT_ROWS, RAW_COLUMNS

OFFICIAL_LEADS = (3, 6, 9, 12, 18, 24)
WAVE_ROWS = 145
WAVE_WINDOWS = (19, 37, 73, 145)
ATMOS_WINDOWS = (37, 73, 145, 289)
SPECTRAL_HARMONICS = 8


def _ids_sha256(ids: np.ndarray) -> str:
    values = np.asarray(ids, dtype="<i8")
    if values.ndim != 1 or len(np.unique(values)) != len(values):
        raise ValueError("IDs must be a unique one-dimensional array")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class CausalSpectralKernelConfig:
    random_feature_count: int = 128
    ridge_penalty: float = 1.0
    standardized_clip: float = 8.0
    gamma_sample_count: int = 256

    def __post_init__(self) -> None:
        if self.random_feature_count != 128:
            raise ValueError("Gen3 random-feature width is frozen at 128")
        if self.ridge_penalty != 1.0:
            raise ValueError("Gen3 ridge penalty is frozen at 1.0")
        if self.standardized_clip != 8.0:
            raise ValueError("Gen3 standardized clipping is frozen at 8.0")
        if self.gamma_sample_count != 256:
            raise ValueError("Gen3 train-only gamma sample count is frozen at 256")


@dataclass(frozen=True)
class TrainOnlyRobustScaler:
    center: np.ndarray
    scale: np.ndarray
    fit_ids_sha256: str

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        train_ids: np.ndarray,
        *,
        forbidden_ids: np.ndarray | None = None,
    ) -> TrainOnlyRobustScaler:
        values = np.asarray(features, dtype=np.float64)
        ids = np.asarray(train_ids, dtype=np.int64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("spectral features must be a finite matrix")
        if ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids):
            raise ValueError("training IDs must be nonempty and unique")
        if ids.min() < 0 or ids.max() >= len(values):
            raise IndexError("training ID is outside the feature matrix")
        if forbidden_ids is not None and np.intersect1d(ids, forbidden_ids).size:
            raise PermissionError("forbidden validation IDs overlap scaler fit IDs")
        selected = values[ids]
        center = np.median(selected, axis=0)
        q25, q75 = np.quantile(selected, (0.25, 0.75), axis=0)
        scale = q75 - q25
        scale = np.where(scale > 1e-8, scale, 1.0)
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ValueError("train-only scaler contains non-finite state")
        return cls(center=center, scale=scale, fit_ids_sha256=_ids_sha256(ids))

    def transform(self, features: np.ndarray, *, clip: float = 8.0) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.center):
            raise ValueError("feature/scaler shape mismatch")
        transformed = np.clip((values - self.center) / self.scale, -clip, clip)
        if not np.isfinite(transformed).all():
            raise ValueError("scaled features are non-finite")
        return transformed


@dataclass(frozen=True)
class FittedCausalSpectralKernel:
    config: CausalSpectralKernelConfig
    scaler: TrainOnlyRobustScaler
    frequency: np.ndarray
    coefficient: np.ndarray
    median_squared_distance: float
    seed: int
    feature_names_sha256: str
    train_ids_sha256: str


def _safe_summary(series: np.ndarray, windows: tuple[int, ...]) -> list[np.ndarray]:
    source = np.asarray(series, dtype=np.float64)
    if source.ndim != 3:
        raise ValueError("summary series must have shape [case,time,channel]")
    result: list[np.ndarray] = []
    for width in windows:
        current = source[:, -width:, :]
        finite = np.isfinite(current)
        count = finite.sum(axis=1).astype(np.float64)
        safe_count = np.maximum(count, 1.0)
        clean = np.where(finite, current, 0.0)
        total = clean.sum(axis=1)
        mean = total / safe_count
        centered = np.where(finite, current - mean[:, None, :], 0.0)
        std = np.sqrt(np.square(centered).sum(axis=1) / safe_count)
        minimum = np.where(finite, current, np.inf).min(axis=1)
        maximum = np.where(finite, current, -np.inf).max(axis=1)
        minimum = np.where(count > 0, minimum, 0.0)
        maximum = np.where(count > 0, maximum, 0.0)

        time = np.arange(width, dtype=np.float64)[None, :, None]
        sx = np.where(finite, time, 0.0).sum(axis=1)
        sxx = np.where(finite, np.square(time), 0.0).sum(axis=1)
        sxy = np.where(finite, time * clean, 0.0).sum(axis=1)
        denominator = count * sxx - np.square(sx)
        slope = np.divide(
            count * sxy - sx * total,
            denominator,
            out=np.zeros_like(total),
            where=np.abs(denominator) > 1e-12,
        ) * max(width - 1, 1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(current, axis=1)
        median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(finite, current, median[:, None, :])
        delta = filled[:, -1, :] - filled[:, 0, :]
        fraction = count / float(width)
        result.extend((mean, std, minimum, maximum, slope, delta, fraction))
    return result


def _spectral_amplitudes(series: np.ndarray) -> np.ndarray:
    source = np.asarray(series, dtype=np.float64)
    finite = np.isfinite(source)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(source, axis=1)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(finite, source, median[:, None, :])
    filled -= filled.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(filled, axis=1)
    amplitude = np.abs(spectrum[:, 1 : SPECTRAL_HARMONICS + 1, :]) / source.shape[1]
    return amplitude.transpose(0, 2, 1).reshape(len(source), -1)


def _direction_components(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(np.asarray(values, dtype=np.float64))
    finite = np.isfinite(radians)
    sine = np.where(finite, np.sin(radians), np.nan)
    cosine = np.where(finite, np.cos(radians), np.nan)
    return sine, cosine


def build_causal_spectral_features(
    raw: np.ndarray, station: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build fixed past-only multiresolution summaries and spectra."""

    source = np.asarray(raw)
    station_values = np.asarray(station, dtype=np.int64)
    if source.ndim != 3 or source.shape[1:] != (CONTEXT_ROWS, len(RAW_COLUMNS)):
        raise ValueError("raw context shape differs")
    if station_values.shape != (len(source),) or not np.isin(station_values, (0, 1, 2)).all():
        raise ValueError("station code contract differs")
    if np.isfinite(source[:, 1::2, :4]).any():
        raise ValueError("wave values on structural ten-minute rows are forbidden")
    if not np.isfinite(source[:, -1, 0]).all():
        raise ValueError("current hs must be finite")

    wave_native = source[:, ::2, :4].astype(np.float64, copy=False)
    wave_sin, wave_cos = _direction_components(wave_native[:, :, 3])
    wave = np.concatenate(
        [wave_native[:, :, :3], wave_sin[:, :, None], wave_cos[:, :, None]], axis=2
    )
    atmos_native = source[:, :, 4:].astype(np.float64, copy=False)
    wind_sin, wind_cos = _direction_components(atmos_native[:, :, 2])
    atmos = np.concatenate(
        [
            atmos_native[:, :, :2],
            atmos_native[:, :, 3:],
            wind_sin[:, :, None],
            wind_cos[:, :, None],
        ],
        axis=2,
    )

    blocks = [*_safe_summary(wave, WAVE_WINDOWS), *_safe_summary(atmos, ATMOS_WINDOWS)]
    blocks.extend((_spectral_amplitudes(wave), _spectral_amplitudes(atmos)))
    station_one_hot = np.eye(3, dtype=np.float64)[station_values]
    blocks.append(station_one_hot)
    features = np.concatenate([block.reshape(len(source), -1) for block in blocks], axis=1)
    if features.shape != (len(source), 435):
        raise AssertionError(f"causal spectral feature width changed: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("causal spectral features contain non-finite values")

    stat_names = ("mean", "std", "min", "max", "slope_span", "delta", "finite_fraction")
    wave_names = ("hs", "tp", "hmax", "wvdir_sin", "wvdir_cos")
    atmos_names = ("wspd", "gust", "airt", "relh", "caph", "wdir_sin", "wdir_cos")
    names: list[str] = []
    for width in WAVE_WINDOWS:
        for stat in stat_names:
            names.extend(f"wave_w{width}_{stat}_{channel}" for channel in wave_names)
    for width in ATMOS_WINDOWS:
        for stat in stat_names:
            names.extend(f"atmos_w{width}_{stat}_{channel}" for channel in atmos_names)
    names.extend(
        f"wave_fft_h{harmonic}_{channel}"
        for channel in wave_names
        for harmonic in range(1, SPECTRAL_HARMONICS + 1)
    )
    names.extend(
        f"atmos_fft_h{harmonic}_{channel}"
        for channel in atmos_names
        for harmonic in range(1, SPECTRAL_HARMONICS + 1)
    )
    names.extend(("station_G", "station_I", "station_S"))
    if len(names) != features.shape[1] or len(set(names)) != len(names):
        raise AssertionError("feature-name contract differs")
    return features.astype(np.float32), tuple(names)


def feature_names_sha256(names: tuple[str, ...]) -> str:
    payload = json.dumps(list(names), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _design_matrix(scaled: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    projected = np.asarray(scaled, dtype=np.float64) @ np.asarray(frequency, dtype=np.float64)
    random_features = np.concatenate([np.cos(projected), np.sin(projected)], axis=1)
    random_features *= np.sqrt(1.0 / frequency.shape[1])
    return np.concatenate([np.ones((len(projected), 1)), random_features], axis=1)


def _train_only_median_squared_distance(scaled_train: np.ndarray, *, sample_count: int) -> float:
    values = np.asarray(scaled_train, dtype=np.float64)
    count = min(len(values), sample_count)
    if count < 2:
        raise ValueError("at least two training rows are required for kernel gamma")
    indices = np.linspace(0, len(values) - 1, num=count, dtype=np.int64)
    sample = values[indices]
    norms = np.square(sample).sum(axis=1)
    distance = norms[:, None] + norms[None, :] - 2.0 * (sample @ sample.T)
    upper = distance[np.triu_indices(count, k=1)]
    positive = upper[upper > 1e-12]
    if len(positive) == 0:
        raise ValueError("train-only kernel distance collapsed to zero")
    median = float(np.median(positive))
    if not np.isfinite(median) or median <= 0.0:
        raise ValueError("train-only kernel median distance is invalid")
    return median


def fit_causal_spectral_kernel(
    features: np.ndarray,
    train_target_delta: np.ndarray,
    train_case_weight: np.ndarray,
    train_ids: np.ndarray,
    *,
    seed: int,
    config: CausalSpectralKernelConfig | None = None,
    scaler: TrainOnlyRobustScaler | None = None,
    forbidden_ids: np.ndarray | None = None,
    names: tuple[str, ...] | None = None,
) -> FittedCausalSpectralKernel:
    cfg = config or CausalSpectralKernelConfig()
    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(train_target_delta, dtype=np.float64)
    weights = np.asarray(train_case_weight, dtype=np.float64)
    ids = np.asarray(train_ids, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 435 or not np.isfinite(values).all():
        raise ValueError("feature matrix contract differs")
    if target.shape != (len(ids), len(OFFICIAL_LEADS)) or not np.isfinite(target).all():
        raise ValueError("train-only six-lead residual target contract differs")
    if weights.shape != (len(ids),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("train-only case-weight contract differs")
    if forbidden_ids is not None and np.intersect1d(ids, forbidden_ids).size:
        raise PermissionError("forbidden validation IDs overlap model fit IDs")
    expected_id_sha = _ids_sha256(ids)
    fitted_scaler = scaler or TrainOnlyRobustScaler.fit(values, ids, forbidden_ids=forbidden_ids)
    if fitted_scaler.fit_ids_sha256 != expected_id_sha:
        raise PermissionError("reused scaler was not fit on the exact training IDs")

    scaled = fitted_scaler.transform(values[ids], clip=cfg.standardized_clip)
    median_squared_distance = _train_only_median_squared_distance(
        scaled, sample_count=cfg.gamma_sample_count
    )
    rng = np.random.default_rng(int(seed))
    frequency_count = cfg.random_feature_count // 2
    frequency = rng.normal(
        0.0,
        1.0 / np.sqrt(median_squared_distance),
        size=(values.shape[1], frequency_count),
    )
    design = _design_matrix(scaled, frequency)
    normalized_weight = weights / weights.mean()
    weighted_design = design * np.sqrt(normalized_weight)[:, None]
    weighted_target = target * np.sqrt(normalized_weight)[:, None]
    penalty = np.eye(design.shape[1], dtype=np.float64) * cfg.ridge_penalty
    penalty[0, 0] = 1e-10
    coefficient = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    if coefficient.shape != (cfg.random_feature_count + 1, len(OFFICIAL_LEADS)):
        raise AssertionError("kernel coefficient shape differs")
    if not np.isfinite(coefficient).all():
        raise ValueError("kernel coefficient is non-finite")
    resolved_names = names or tuple(f"feature_{index}" for index in range(values.shape[1]))
    return FittedCausalSpectralKernel(
        config=cfg,
        scaler=fitted_scaler,
        frequency=frequency,
        coefficient=coefficient,
        median_squared_distance=median_squared_distance,
        seed=int(seed),
        feature_names_sha256=feature_names_sha256(resolved_names),
        train_ids_sha256=expected_id_sha,
    )


def predict_causal_spectral_kernel(
    fitted: FittedCausalSpectralKernel,
    features: np.ndarray,
    prediction_ids: np.ndarray,
    *,
    names: tuple[str, ...],
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    ids = np.asarray(prediction_ids, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != len(fitted.scaler.center):
        raise ValueError("prediction feature matrix contract differs")
    if ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("prediction IDs must be nonempty and unique")
    if ids.min() < 0 or ids.max() >= len(values):
        raise IndexError("prediction ID is outside the feature matrix")
    if feature_names_sha256(names) != fitted.feature_names_sha256:
        raise PermissionError("prediction feature-name identity differs from fitted model")
    scaled = fitted.scaler.transform(values[ids], clip=fitted.config.standardized_clip)
    design = _design_matrix(scaled, fitted.frequency)
    prediction = design @ fitted.coefficient
    if prediction.shape != (len(ids), len(OFFICIAL_LEADS)) or not np.isfinite(prediction).all():
        raise ValueError("kernel prediction contract differs")
    return prediction


def fit_and_predict_causal_spectral_kernel(
    features: np.ndarray,
    train_target_delta: np.ndarray,
    train_case_weight: np.ndarray,
    train_ids: np.ndarray,
    prediction_ids: np.ndarray,
    *,
    seed: int,
    config: CausalSpectralKernelConfig | None = None,
    scaler: TrainOnlyRobustScaler | None = None,
    forbidden_ids: np.ndarray | None = None,
    names: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, FittedCausalSpectralKernel]:
    blocked_parts = [np.asarray(prediction_ids, dtype=np.int64)]
    if forbidden_ids is not None:
        blocked_parts.append(np.asarray(forbidden_ids, dtype=np.int64))
    blocked = np.unique(np.concatenate(blocked_parts))
    fitted = fit_causal_spectral_kernel(
        features,
        train_target_delta,
        train_case_weight,
        train_ids,
        seed=seed,
        config=config,
        scaler=scaler,
        forbidden_ids=blocked,
        names=names,
    )
    resolved_names = names or tuple(f"feature_{index}" for index in range(features.shape[1]))
    return (
        predict_causal_spectral_kernel(fitted, features, prediction_ids, names=resolved_names),
        fitted,
    )


def save_fitted_causal_spectral_kernel(fitted: FittedCausalSpectralKernel, path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "p3_causal_spectral_kernel.v1",
        "config": asdict(fitted.config),
        "seed": fitted.seed,
        "feature_names_sha256": fitted.feature_names_sha256,
        "train_ids_sha256": fitted.train_ids_sha256,
        "scaler_fit_ids_sha256": fitted.scaler.fit_ids_sha256,
        "median_squared_distance": fitted.median_squared_distance,
    }
    with destination.open("xb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            center=fitted.scaler.center,
            scale=fitted.scaler.scale,
            frequency=fitted.frequency,
            coefficient=fitted.coefficient,
        )


def load_fitted_causal_spectral_kernel(path: Path) -> FittedCausalSpectralKernel:
    with np.load(Path(path), allow_pickle=False) as data:
        if set(data.files) != {"metadata", "center", "scale", "frequency", "coefficient"}:
            raise ValueError("saved kernel bundle fields differ")
        metadata: dict[str, Any] = json.loads(str(data["metadata"].item()))
        if metadata.get("schema_version") != "p3_causal_spectral_kernel.v1":
            raise ValueError("saved kernel schema differs")
        scaler = TrainOnlyRobustScaler(
            center=np.asarray(data["center"], dtype=np.float64),
            scale=np.asarray(data["scale"], dtype=np.float64),
            fit_ids_sha256=str(metadata["scaler_fit_ids_sha256"]),
        )
        fitted = FittedCausalSpectralKernel(
            config=CausalSpectralKernelConfig(**metadata["config"]),
            scaler=scaler,
            frequency=np.asarray(data["frequency"], dtype=np.float64),
            coefficient=np.asarray(data["coefficient"], dtype=np.float64),
            median_squared_distance=float(metadata["median_squared_distance"]),
            seed=int(metadata["seed"]),
            feature_names_sha256=str(metadata["feature_names_sha256"]),
            train_ids_sha256=str(metadata["train_ids_sha256"]),
        )
    if fitted.frequency.shape != (
        len(fitted.scaler.center),
        fitted.config.random_feature_count // 2,
    ):
        raise ValueError("saved frequency shape differs")
    if fitted.coefficient.shape != (fitted.config.random_feature_count + 1, 6):
        raise ValueError("saved coefficient shape differs")
    arrays = (
        fitted.scaler.center,
        fitted.scaler.scale,
        fitted.frequency,
        fitted.coefficient,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("saved kernel bundle contains non-finite state")
    if not np.isfinite(fitted.median_squared_distance) or fitted.median_squared_distance <= 0:
        raise ValueError("saved train-only kernel distance is invalid")
    return fitted


__all__ = [
    "ATMOS_WINDOWS",
    "CausalSpectralKernelConfig",
    "FittedCausalSpectralKernel",
    "SPECTRAL_HARMONICS",
    "TrainOnlyRobustScaler",
    "WAVE_WINDOWS",
    "build_causal_spectral_features",
    "feature_names_sha256",
    "fit_and_predict_causal_spectral_kernel",
    "fit_causal_spectral_kernel",
    "load_fitted_causal_spectral_kernel",
    "predict_causal_spectral_kernel",
    "save_fitted_causal_spectral_kernel",
]
