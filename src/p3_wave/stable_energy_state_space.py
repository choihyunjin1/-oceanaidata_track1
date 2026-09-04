"""Stable partially pooled wave-energy state-space model for P3 research.

The model learns only one-step transitions contained inside safe training
contexts.  It has no target-table, test-data, candidate, or upload API.
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

STATE_COLUMNS = (
    "log1p_hs_energy",
    "log1p_tp",
    "log1p_hmax_energy",
    "wave_direction_sin",
    "wave_direction_cos",
    "log1p_wind_speed",
    "log1p_gust",
    "wind_direction_sin",
    "wind_direction_cos",
    "air_temperature",
    "relative_humidity",
    "pressure",
)
OFFICIAL_LEADS = (3, 6, 9, 12, 18, 24)
OFFICIAL_STEPS_20M = (9, 18, 27, 36, 54, 72)
STEP_NS = 20 * 60 * 1_000_000_000


def _ids_sha256(ids: np.ndarray) -> str:
    values = np.asarray(ids, dtype="<i8")
    if values.ndim != 1 or len(values) == 0 or len(np.unique(values)) != len(values):
        raise ValueError("IDs must be a nonempty unique one-dimensional array")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class StableEnergyStateSpaceConfig:
    global_ridge: float = 1e-3
    station_residual_ridge: float = 1e-2
    maximum_spectral_radius: float = 0.995
    rollout_steps: int = 72
    standardized_state_clip: float = 10.0

    def __post_init__(self) -> None:
        expected = (1e-3, 1e-2, 0.995, 72, 10.0)
        observed = (
            self.global_ridge,
            self.station_residual_ridge,
            self.maximum_spectral_radius,
            self.rollout_steps,
            self.standardized_state_clip,
        )
        if observed != expected:
            raise ValueError("Gen4 state-space constants are frozen")


@dataclass(frozen=True)
class TrainOnlyStateScaler:
    center: np.ndarray
    scale: np.ndarray
    fit_ids_sha256: str

    @classmethod
    def fit(
        cls,
        state_sequences: np.ndarray,
        train_ids: np.ndarray,
        *,
        forbidden_ids: np.ndarray | None = None,
    ) -> TrainOnlyStateScaler:
        states = np.asarray(state_sequences, dtype=np.float64)
        ids = np.asarray(train_ids, dtype=np.int64)
        if states.ndim != 3 or states.shape[1:] != (145, len(STATE_COLUMNS)):
            raise ValueError("state sequence shape differs")
        _ids_sha256(ids)
        if ids.min() < 0 or ids.max() >= len(states):
            raise IndexError("training ID outside state sequence")
        if forbidden_ids is not None and np.intersect1d(ids, forbidden_ids).size:
            raise PermissionError("forbidden IDs overlap state-scaler fit IDs")
        selected = states[ids].reshape(-1, len(STATE_COLUMNS))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            center = np.nanmedian(selected, axis=0)
            q25, q75 = np.nanquantile(selected, (0.25, 0.75), axis=0)
        center = np.where(np.isfinite(center), center, 0.0)
        scale = q75 - q25
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        return cls(center=center, scale=scale, fit_ids_sha256=_ids_sha256(ids))

    def transform(self, values: np.ndarray) -> np.ndarray:
        source = np.asarray(values, dtype=np.float64)
        if source.shape[-1] != len(self.center):
            raise ValueError("state/scaler width differs")
        result = (source - self.center) / self.scale
        return np.where(np.isfinite(result), result, 0.0)

    def inverse_energy(self, standardized_energy: np.ndarray) -> np.ndarray:
        log_energy = np.asarray(standardized_energy, dtype=np.float64) * self.scale[0]
        log_energy += self.center[0]
        log_energy = np.clip(log_energy, 0.0, np.log1p(30.0**2))
        return np.sqrt(np.expm1(log_energy))


@dataclass(frozen=True)
class FittedStableEnergyStateSpace:
    config: StableEnergyStateSpaceConfig
    scaler: TrainOnlyStateScaler
    transition: np.ndarray
    intercept: np.ndarray
    spectral_radius_before: np.ndarray
    spectral_radius_after: np.ndarray
    train_ids: np.ndarray
    train_ids_sha256: str
    transition_key_sha256: str
    transition_count: int


def build_wave_energy_state_sequences(raw: np.ndarray) -> np.ndarray:
    """Convert native-cadence past context to 145 causal physical-state rows."""

    source = np.asarray(raw)
    if source.ndim != 3 or source.shape[1:] != (CONTEXT_ROWS, len(RAW_COLUMNS)):
        raise ValueError("raw context shape differs")
    if np.isfinite(source[:, 1::2, :4]).any():
        raise ValueError("wave values on structural ten-minute rows are forbidden")
    native = source[:, ::2, :].astype(np.float64, copy=False)
    if native.shape[1] != 145:
        raise AssertionError("native 20-minute row count differs")
    wave_direction = np.deg2rad(native[:, :, 3])
    wind_direction = np.deg2rad(native[:, :, 6])
    result = np.stack(
        [
            np.log1p(np.square(np.clip(native[:, :, 0], 0.0, None))),
            np.log1p(np.clip(native[:, :, 1], 0.0, None)),
            np.log1p(np.square(np.clip(native[:, :, 2], 0.0, None))),
            np.sin(wave_direction),
            np.cos(wave_direction),
            np.log1p(np.clip(native[:, :, 4], 0.0, None)),
            np.log1p(np.clip(native[:, :, 5], 0.0, None)),
            np.sin(wind_direction),
            np.cos(wind_direction),
            native[:, :, 7],
            native[:, :, 8],
            native[:, :, 9],
        ],
        axis=-1,
    )
    if not np.isfinite(result[:, -1, 0]).all():
        raise ValueError("current hs energy must be finite")
    return result.astype(np.float32)


def _transition_key_sha(station: np.ndarray, destination_time_ns: np.ndarray) -> str:
    packed = np.empty(len(station), dtype=np.dtype([("station", "<i8"), ("time_ns", "<i8")]))
    packed["station"] = station
    packed["time_ns"] = destination_time_ns
    return hashlib.sha256(packed.tobytes(order="C")).hexdigest()


def build_unique_train_transitions(
    state_sequences: np.ndarray,
    station: np.ndarray,
    anchor_time_ns: np.ndarray,
    train_ids: np.ndarray,
    *,
    scaler: TrainOnlyStateScaler,
    forbidden_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    states = np.asarray(state_sequences, dtype=np.float64)
    stations = np.asarray(station, dtype=np.int64)
    times = np.asarray(anchor_time_ns, dtype=np.int64)
    ids = np.asarray(train_ids, dtype=np.int64)
    expected_sha = _ids_sha256(ids)
    if scaler.fit_ids_sha256 != expected_sha:
        raise PermissionError("state scaler was not fit on exact training IDs")
    if forbidden_ids is not None and np.intersect1d(ids, forbidden_ids).size:
        raise PermissionError("forbidden IDs overlap transition fit IDs")
    if stations.shape != (len(states),) or times.shape != (len(states),):
        raise ValueError("station/time identity shape differs")
    if not np.isin(stations, (0, 1, 2)).all():
        raise ValueError("station code contract differs")

    selected = scaler.transform(states[ids])
    source = selected[:, :-1, :].reshape(-1, len(STATE_COLUMNS))
    destination = selected[:, 1:, :].reshape(-1, len(STATE_COLUMNS))
    finite_energy = np.isfinite(states[ids, :-1, 0]) & np.isfinite(states[ids, 1:, 0])
    valid = finite_energy.reshape(-1)
    station_rows = np.repeat(stations[ids], 144)[valid]
    offsets = np.arange(-143, 1, dtype=np.int64) * STEP_NS
    destination_time = (times[ids, None] + offsets[None, :]).reshape(-1)[valid]
    source = source[valid]
    destination = destination[valid]

    order = np.lexsort((destination_time, station_rows))
    station_rows = station_rows[order]
    destination_time = destination_time[order]
    source = source[order]
    destination = destination[order]
    duplicate = (station_rows[1:] == station_rows[:-1]) & (
        destination_time[1:] == destination_time[:-1]
    )
    if duplicate.any():
        difference = np.max(
            np.abs(
                np.concatenate(
                    [
                        source[1:][duplicate] - source[:-1][duplicate],
                        destination[1:][duplicate] - destination[:-1][duplicate],
                    ],
                    axis=1,
                )
            )
        )
        if difference > 1e-5:
            raise ValueError("duplicate station/time transitions disagree")
    keep = np.ones(len(source), dtype=bool)
    keep[1:] = ~duplicate
    source = source[keep]
    destination = destination[keep]
    station_rows = station_rows[keep]
    destination_time = destination_time[keep]
    if len(source) == 0 or not np.isfinite(source).all() or not np.isfinite(destination).all():
        raise ValueError("train transition matrix is empty or non-finite")
    receipt = {
        "raw_transition_rows": int(valid.sum()),
        "unique_station_time_transitions": int(len(source)),
        "duplicate_rows_removed": int(valid.sum() - len(source)),
        "transition_key_sha256": _transition_key_sha(station_rows, destination_time),
        "train_ids_sha256": expected_sha,
        "validation_ids_used": 0,
    }
    return source, destination, station_rows, receipt


def _ridge(source: np.ndarray, target: np.ndarray, penalty: float) -> tuple[np.ndarray, np.ndarray]:
    design = np.concatenate([source, np.ones((len(source), 1))], axis=1)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * penalty
    regularizer[-1, -1] = 1e-10
    coefficient = np.linalg.solve(
        design.T @ design + regularizer,
        design.T @ target,
    )
    return coefficient[:-1], coefficient[-1]


def _stabilize(matrix: np.ndarray, maximum_radius: float) -> tuple[np.ndarray, float, float]:
    values, vectors = np.linalg.eig(np.asarray(matrix, dtype=np.float64))
    before = float(np.max(np.abs(values)))
    magnitudes = np.abs(values)
    adjusted = np.where(
        magnitudes > maximum_radius,
        values * (maximum_radius / np.maximum(magnitudes, 1e-12)),
        values,
    )
    stable = vectors @ np.diag(adjusted) @ np.linalg.inv(vectors)
    stable = np.real_if_close(stable, tol=1000)
    if np.iscomplexobj(stable):
        raise ValueError("stability projection retained complex state")
    stable = np.asarray(stable, dtype=np.float64)
    after = float(np.max(np.abs(np.linalg.eigvals(stable))))
    if after > maximum_radius + 1e-10 or not np.isfinite(stable).all():
        raise ValueError("state transition stability projection failed")
    return stable, before, after


def fit_stable_energy_state_space(
    state_sequences: np.ndarray,
    station: np.ndarray,
    anchor_time_ns: np.ndarray,
    train_ids: np.ndarray,
    *,
    forbidden_ids: np.ndarray,
    scaler: TrainOnlyStateScaler | None = None,
    config: StableEnergyStateSpaceConfig | None = None,
) -> tuple[FittedStableEnergyStateSpace, dict[str, Any]]:
    cfg = config or StableEnergyStateSpaceConfig()
    ids = np.asarray(train_ids, dtype=np.int64)
    blocked = np.asarray(forbidden_ids, dtype=np.int64)
    _ids_sha256(blocked)
    if np.intersect1d(ids, blocked).size:
        raise PermissionError("forbidden IDs overlap state-space fit IDs")
    fitted_scaler = scaler or TrainOnlyStateScaler.fit(state_sequences, ids, forbidden_ids=blocked)
    source, target, station_rows, receipt = build_unique_train_transitions(
        state_sequences,
        station,
        anchor_time_ns,
        ids,
        scaler=fitted_scaler,
        forbidden_ids=blocked,
    )
    global_transition, global_intercept = _ridge(source, target, cfg.global_ridge)
    global_prediction = source @ global_transition + global_intercept
    transition = np.empty((3, len(STATE_COLUMNS), len(STATE_COLUMNS)), dtype=np.float64)
    intercept = np.empty((3, len(STATE_COLUMNS)), dtype=np.float64)
    before = np.empty(3, dtype=np.float64)
    after = np.empty(3, dtype=np.float64)
    for code in range(3):
        mask = station_rows == code
        if int(mask.sum()) < 100:
            raise ValueError("station transition pool is too small")
        residual_transition, residual_intercept = _ridge(
            source[mask],
            target[mask] - global_prediction[mask],
            cfg.station_residual_ridge,
        )
        current = global_transition + residual_transition
        transition[code], before[code], after[code] = _stabilize(
            current, cfg.maximum_spectral_radius
        )
        intercept[code] = global_intercept + residual_intercept
    fitted = FittedStableEnergyStateSpace(
        config=cfg,
        scaler=fitted_scaler,
        transition=transition,
        intercept=intercept,
        spectral_radius_before=before,
        spectral_radius_after=after,
        train_ids=ids.copy(),
        train_ids_sha256=_ids_sha256(ids),
        transition_key_sha256=receipt["transition_key_sha256"],
        transition_count=int(receipt["unique_station_time_transitions"]),
    )
    return fitted, receipt


def predict_stable_energy_state_space(
    fitted: FittedStableEnergyStateSpace,
    state_sequences: np.ndarray,
    station: np.ndarray,
    prediction_ids: np.ndarray,
) -> np.ndarray:
    states = np.asarray(state_sequences, dtype=np.float64)
    stations = np.asarray(station, dtype=np.int64)
    ids = np.asarray(prediction_ids, dtype=np.int64)
    _ids_sha256(ids)
    if ids.min() < 0 or ids.max() >= len(states):
        raise IndexError("prediction ID outside state sequence")
    if np.intersect1d(fitted.train_ids, ids).size:
        raise PermissionError("prediction IDs overlap fitted training IDs")
    current = fitted.scaler.transform(states[ids, -1, :])
    trajectories = np.empty((len(ids), fitted.config.rollout_steps), dtype=np.float64)
    for step in range(fitted.config.rollout_steps):
        next_state = np.empty_like(current)
        for code in range(3):
            mask = stations[ids] == code
            next_state[mask] = current[mask] @ fitted.transition[code] + fitted.intercept[code]
        current = np.clip(
            next_state,
            -fitted.config.standardized_state_clip,
            fitted.config.standardized_state_clip,
        )
        trajectories[:, step] = fitted.scaler.inverse_energy(current[:, 0])
    result = trajectories[:, np.asarray(OFFICIAL_STEPS_20M) - 1]
    if result.shape != (len(ids), 6) or not np.isfinite(result).all():
        raise ValueError("state-space prediction contract differs")
    return result


def save_fitted_stable_energy_state_space(fitted: FittedStableEnergyStateSpace, path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "p3_stable_energy_state_space.v1",
        "config": asdict(fitted.config),
        "fit_ids_sha256": fitted.scaler.fit_ids_sha256,
        "train_ids_sha256": fitted.train_ids_sha256,
        "transition_key_sha256": fitted.transition_key_sha256,
        "transition_count": fitted.transition_count,
    }
    with destination.open("xb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            center=fitted.scaler.center,
            scale=fitted.scaler.scale,
            transition=fitted.transition,
            intercept=fitted.intercept,
            spectral_radius_before=fitted.spectral_radius_before,
            spectral_radius_after=fitted.spectral_radius_after,
            train_ids=fitted.train_ids,
        )


def load_fitted_stable_energy_state_space(path: Path) -> FittedStableEnergyStateSpace:
    with np.load(Path(path), allow_pickle=False) as data:
        expected = {
            "metadata",
            "center",
            "scale",
            "transition",
            "intercept",
            "spectral_radius_before",
            "spectral_radius_after",
            "train_ids",
        }
        if set(data.files) != expected:
            raise ValueError("saved state-space fields differ")
        metadata: dict[str, Any] = json.loads(str(data["metadata"].item()))
        if metadata.get("schema_version") != "p3_stable_energy_state_space.v1":
            raise ValueError("saved state-space schema differs")
        scaler = TrainOnlyStateScaler(
            center=np.asarray(data["center"], dtype=np.float64),
            scale=np.asarray(data["scale"], dtype=np.float64),
            fit_ids_sha256=str(metadata["fit_ids_sha256"]),
        )
        fitted = FittedStableEnergyStateSpace(
            config=StableEnergyStateSpaceConfig(**metadata["config"]),
            scaler=scaler,
            transition=np.asarray(data["transition"], dtype=np.float64),
            intercept=np.asarray(data["intercept"], dtype=np.float64),
            spectral_radius_before=np.asarray(data["spectral_radius_before"], dtype=np.float64),
            spectral_radius_after=np.asarray(data["spectral_radius_after"], dtype=np.float64),
            train_ids=np.asarray(data["train_ids"], dtype=np.int64),
            train_ids_sha256=str(metadata["train_ids_sha256"]),
            transition_key_sha256=str(metadata["transition_key_sha256"]),
            transition_count=int(metadata["transition_count"]),
        )
    arrays = (
        fitted.scaler.center,
        fitted.scaler.scale,
        fitted.transition,
        fitted.intercept,
        fitted.spectral_radius_before,
        fitted.spectral_radius_after,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("saved state-space contains non-finite state")
    if fitted.transition.shape != (3, 12, 12) or fitted.intercept.shape != (3, 12):
        raise ValueError("saved state-space shape differs")
    if (fitted.scaler.scale <= 0).any():
        raise ValueError("saved state scaler is non-positive")
    if _ids_sha256(fitted.train_ids) != fitted.train_ids_sha256:
        raise ValueError("saved training IDs differ from their seal")
    if (fitted.spectral_radius_after > fitted.config.maximum_spectral_radius + 1e-10).any():
        raise ValueError("saved state-space is unstable")
    return fitted


__all__ = [
    "FittedStableEnergyStateSpace",
    "OFFICIAL_LEADS",
    "OFFICIAL_STEPS_20M",
    "STATE_COLUMNS",
    "StableEnergyStateSpaceConfig",
    "TrainOnlyStateScaler",
    "build_unique_train_transitions",
    "build_wave_energy_state_sequences",
    "fit_stable_energy_state_space",
    "load_fitted_stable_energy_state_space",
    "predict_stable_energy_state_space",
    "save_fitted_stable_energy_state_space",
]
