"""Fixed robust trajectory geometry for the P1 slow-anomaly unary expert.

This module is an append-only implementation contract for
``p1_multiscale_cross_layer_offset_drift_unary_v6``.  It deliberately contains
no file-system entry point and no test-data path.  The future authorized runner
may fit the label-free seasonal/graph state on an explicit train prefix, build
the fixed multiscale feature bank, and fit exactly one combined offset/drift
unary head.  The static owner runner imports only the constants and pure audit
helpers; it never calls a fit routine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT_ID = "p1_multiscale_cross_layer_offset_drift_unary_v6"
HYPOTHESIS_ID = "robust_multiscale_cross_layer_offset_drift_unary"

KEY_COLUMNS = ("station", "year", "layer", "time")
INPUT_ONLY_COLUMNS = ("station", "year", "layer", "time", "temp", "psal", "depth")
SLOW_TYPES = ("offset", "drift")

# Ten-minute rows.  The longest scale deliberately extends beyond the injected
# 86.5-hour maximum so both flanks of a long event can be represented.
MULTISCALE_ROWS = (48, 96, 192, 384, 576)
MIN_SLOW_RUN_ROWS = 48
MAX_SLOW_RUN_ROWS = 519
UNARY_THRESHOLD = 0.5
UNARY_C = 0.25
UNARY_MAX_ITER = 64
UNARY_TOL = 1.0e-6
DETERMINISTIC_SEED = 20260823
SPIKE_SINGLETON_MAX_ROWS = 1
SPIKE_PROTECTION_RADIUS_ROWS = 6
GAP_BREAK_MINUTES = 30

ANNUAL_HARMONICS = (1, 2, 3, 4)
DIURNAL_HARMONICS = (1, 2)
SEASONAL_IRLS_ITERATIONS = 8
SEASONAL_HUBER_DELTA = 1.5
SEASONAL_RIDGE = 1.0e-6
MIN_GROUP_FIT_ROWS = 32

BASE_GEOMETRY_FEATURES = (
    "abs_seasonal_residual_z",
    "abs_graph_residual_z",
    "graph_available",
    "peer_count",
)
SCALE_GEOMETRY_KINDS = (
    "level_abs_z",
    "haar_abs_z",
    "slope_abs_z",
    "curvature_abs_z",
    "coherence_deficit_z",
)
GEOMETRY_FEATURES = BASE_GEOMETRY_FEATURES + tuple(
    f"{kind}_{rows}" for rows in MULTISCALE_ROWS for kind in SCALE_GEOMETRY_KINDS
)

GATE_THRESHOLDS: dict[str, float | int] = {
    "minimum_micro_f1_delta": 0.002,
    "minimum_offset_recall_delta": 0.03,
    "minimum_drift_recall_delta": 0.03,
    "minimum_mean_offset_drift_recall_delta": 0.04,
    "minimum_spike_f1_delta": 0.0,
    "minimum_worst_station_layer_f1_delta": -0.002,
    "maximum_normal_fp_relative_increase": 0.05,
    "minimum_nondegrading_inner_blocks": 3,
    "required_inner_blocks": 3,
}


def _deep_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def int64_ids_sha256(ids: np.ndarray) -> str:
    """Hash a unique explicit row-ID vector in portable little-endian form."""

    values = np.asarray(ids, dtype=np.int64)
    if values.ndim != 1 or len(np.unique(values)) != len(values):
        raise ValueError("row IDs must be a unique one-dimensional vector")
    return hashlib.sha256(values.astype("<i8", copy=False).tobytes()).hexdigest()


def _group_key(station: object, layer: object) -> str:
    return f"{station}|{int(layer)}"


def _edge_key(station: object, low_layer: int, high_layer: int) -> str:
    return f"{station}|{int(low_layer)}|{int(high_layer)}"


def _robust_scale(values: np.ndarray, *, floor: float = 1.0e-6) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return floor
    center = float(np.median(finite))
    scale = 1.4826 * float(np.median(np.abs(finite - center)))
    return max(scale, floor)


def seasonal_design(time_values: pd.Series | np.ndarray) -> np.ndarray:
    """Return the fixed annual/diurnal Fourier design without target fields."""

    parsed = pd.to_datetime(time_values, errors="raise", utc=True, format="mixed")
    nanos = parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    seconds = nanos.astype(np.float64) / 1.0e9
    day = seconds / 86_400.0
    columns: list[np.ndarray] = [np.ones(len(day), dtype=np.float64)]
    for harmonic in ANNUAL_HARMONICS:
        angle = 2.0 * np.pi * harmonic * day / 365.2425
        columns.extend((np.sin(angle), np.cos(angle)))
    for harmonic in DIURNAL_HARMONICS:
        angle = 2.0 * np.pi * harmonic * day
        columns.extend((np.sin(angle), np.cos(angle)))
    return np.column_stack(columns)


def exact_gap_safe_segment_ids(frame: pd.DataFrame) -> np.ndarray:
    """Assign contiguous station-layer segments with a fixed 30-minute break."""

    missing = sorted(set(("station", "layer", "time")).difference(frame.columns))
    if missing:
        raise KeyError(f"missing segment columns: {missing}")
    if frame.empty:
        return np.empty(0, dtype=np.int64)
    parsed = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    station = frame["station"].astype(str).to_numpy()
    layer = frame["layer"].astype(np.int64).to_numpy()
    nanos = parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    boundary = np.ones(len(frame), dtype=bool)
    if len(frame) > 1:
        delta_minutes = (nanos[1:] - nanos[:-1]) / (60.0 * 1.0e9)
        boundary[1:] = (
            (station[1:] != station[:-1])
            | (layer[1:] != layer[:-1])
            | (delta_minutes <= 0.0)
            | (delta_minutes > GAP_BREAK_MINUTES)
        )
    return np.cumsum(boundary, dtype=np.int64) - 1


def _fixed_huber_irls(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    if design.ndim != 2 or target.ndim != 1 or len(design) != len(target):
        raise ValueError("IRLS design/target shape differs")
    finite = np.isfinite(target) & np.isfinite(design).all(axis=1)
    x = np.asarray(design[finite], dtype=np.float64)
    y = np.asarray(target[finite], dtype=np.float64)
    if len(y) < MIN_GROUP_FIT_ROWS:
        raise ValueError("seasonal group has too few finite prefix rows")
    ridge = np.eye(x.shape[1], dtype=np.float64) * SEASONAL_RIDGE
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    scale = _robust_scale(y - x @ beta)
    for _ in range(SEASONAL_IRLS_ITERATIONS):
        residual = y - x @ beta
        scale = _robust_scale(residual)
        ratio = np.abs(residual) / (SEASONAL_HUBER_DELTA * scale)
        weights = np.ones_like(ratio)
        large = ratio > 1.0
        weights[large] = 1.0 / ratio[large]
        root_weight = np.sqrt(weights)
        weighted_x = x * root_weight[:, None]
        weighted_y = y * root_weight
        beta = np.linalg.solve(weighted_x.T @ weighted_x + ridge, weighted_x.T @ weighted_y)
    return beta, scale


@dataclass(frozen=True)
class RobustSeasonalGraphState:
    """Serializable label-free state fitted on one explicit prefix only."""

    train_ids_sha256: str
    seasonal_coefficients: dict[str, tuple[float, ...]]
    seasonal_scales: dict[str, float]
    edge_residual_deltas: dict[str, float]
    edge_residual_scales: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "p1_robust_seasonal_graph_state.v1",
            "train_ids_sha256": self.train_ids_sha256,
            "seasonal_coefficients": {
                key: list(value) for key, value in sorted(self.seasonal_coefficients.items())
            },
            "seasonal_scales": dict(sorted(self.seasonal_scales.items())),
            "edge_residual_deltas": dict(sorted(self.edge_residual_deltas.items())),
            "edge_residual_scales": dict(sorted(self.edge_residual_scales.items())),
        }

    @property
    def state_sha256(self) -> str:
        return _deep_sha(self.as_dict())


@dataclass(frozen=True)
class FixedSlowUnaryState:
    """Portable frozen state for the one preregistered logistic unary head."""

    train_ids_sha256: str
    feature_names: tuple[str, ...]
    robust_center: tuple[float, ...]
    robust_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    negative_rows: int
    positive_rows: int
    optimizer_iterations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "p1_fixed_slow_unary_state.v1",
            "train_ids_sha256": self.train_ids_sha256,
            "feature_names": list(self.feature_names),
            "robust_center": list(self.robust_center),
            "robust_scale": list(self.robust_scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "negative_rows": self.negative_rows,
            "positive_rows": self.positive_rows,
            "optimizer_iterations": self.optimizer_iterations,
            "fixed_hyperparameters": {
                "C": UNARY_C,
                "class_weight": "balanced",
                "max_iter": UNARY_MAX_ITER,
                "penalty": "l2",
                "random_state": DETERMINISTIC_SEED,
                "robust_quantile_range": [25.0, 75.0],
                "solver": "lbfgs",
                "tol": UNARY_TOL,
            },
        }

    @property
    def state_sha256(self) -> str:
        return _deep_sha(self.as_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FixedSlowUnaryState:
        expected = {
            "schema_version",
            "train_ids_sha256",
            "feature_names",
            "robust_center",
            "robust_scale",
            "coefficients",
            "intercept",
            "negative_rows",
            "positive_rows",
            "optimizer_iterations",
            "fixed_hyperparameters",
        }
        if set(value) != expected or value["schema_version"] != "p1_fixed_slow_unary_state.v1":
            raise ValueError("fixed slow-unary state schema differs")
        expected_hyperparameters = {
            "C": UNARY_C,
            "class_weight": "balanced",
            "max_iter": UNARY_MAX_ITER,
            "penalty": "l2",
            "random_state": DETERMINISTIC_SEED,
            "robust_quantile_range": [25.0, 75.0],
            "solver": "lbfgs",
            "tol": UNARY_TOL,
        }
        if value["fixed_hyperparameters"] != expected_hyperparameters:
            raise ValueError("fixed slow-unary hyperparameters differ")
        state = cls(
            train_ids_sha256=str(value["train_ids_sha256"]),
            feature_names=tuple(str(item) for item in value["feature_names"]),
            robust_center=tuple(float(item) for item in value["robust_center"]),
            robust_scale=tuple(float(item) for item in value["robust_scale"]),
            coefficients=tuple(float(item) for item in value["coefficients"]),
            intercept=float(value["intercept"]),
            negative_rows=int(value["negative_rows"]),
            positive_rows=int(value["positive_rows"]),
            optimizer_iterations=int(value["optimizer_iterations"]),
        )
        width = len(GEOMETRY_FEATURES)
        if not (
            state.feature_names == GEOMETRY_FEATURES
            and len(state.robust_center) == width
            and len(state.robust_scale) == width
            and len(state.coefficients) == width
            and np.isfinite(
                np.asarray(
                    [
                        *state.robust_center,
                        *state.robust_scale,
                        *state.coefficients,
                        state.intercept,
                    ]
                )
            ).all()
            and min(state.robust_scale) > 0.0
            and state.negative_rows > 0
            and state.positive_rows > 0
            and 0 <= state.optimizer_iterations <= UNARY_MAX_ITER
        ):
            raise ValueError("fixed slow-unary state values differ")
        return state


def fit_robust_seasonal_graph_state(
    input_only_frame: pd.DataFrame,
    train_ids: np.ndarray,
) -> RobustSeasonalGraphState:
    """Fit label-free seasonal and adjacent-layer graph state on explicit IDs."""

    missing = sorted(set(INPUT_ONLY_COLUMNS).difference(input_only_frame.columns))
    if missing:
        raise KeyError(f"missing input-only columns: {missing}")
    ids = np.asarray(train_ids, dtype=np.int64)
    if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("train IDs must be non-empty, unique, and one-dimensional")
    if int(ids.min()) < 0 or int(ids.max()) >= len(input_only_frame):
        raise IndexError("train IDs escaped the input-only frame")
    prefix = input_only_frame.iloc[ids].copy()
    design = seasonal_design(prefix["time"])
    coefficients: dict[str, tuple[float, ...]] = {}
    scales: dict[str, float] = {}
    residual = np.full(len(prefix), np.nan, dtype=np.float64)
    grouped = prefix.groupby(["station", "layer"], sort=True, observed=True).indices
    for (station, layer), positions_raw in grouped.items():
        positions = np.asarray(positions_raw, dtype=np.int64)
        beta, scale = _fixed_huber_irls(
            design[positions],
            prefix.iloc[positions]["temp"].to_numpy(dtype=np.float64),
        )
        key = _group_key(station, layer)
        coefficients[key] = tuple(float(value) for value in beta)
        scales[key] = float(scale)
        residual[positions] = (
            prefix.iloc[positions]["temp"].to_numpy(dtype=np.float64) - design[positions] @ beta
        )
    prefix["_seasonal_residual"] = residual
    edge_deltas: dict[str, float] = {}
    edge_scales: dict[str, float] = {}
    for station, station_rows in prefix.groupby("station", sort=True, observed=True):
        pivot = station_rows.pivot_table(
            index="time",
            columns="layer",
            values="_seasonal_residual",
            aggfunc="first",
            observed=True,
        )
        layers = sorted(int(value) for value in pivot.columns)
        for low_layer, high_layer in zip(layers[:-1], layers[1:], strict=True):
            difference = pivot[low_layer].to_numpy(dtype=np.float64) - pivot[high_layer].to_numpy(
                dtype=np.float64
            )
            finite = difference[np.isfinite(difference)]
            if len(finite) < MIN_GROUP_FIT_ROWS:
                continue
            key = _edge_key(station, low_layer, high_layer)
            edge_deltas[key] = float(np.median(finite))
            edge_scales[key] = _robust_scale(finite)
    return RobustSeasonalGraphState(
        train_ids_sha256=int64_ids_sha256(ids),
        seasonal_coefficients=coefficients,
        seasonal_scales=scales,
        edge_residual_deltas=edge_deltas,
        edge_residual_scales=edge_scales,
    )


def apply_robust_seasonal_graph_state(
    input_only_frame: pd.DataFrame,
    state: RobustSeasonalGraphState,
) -> pd.DataFrame:
    """Apply a frozen prefix state to label-free rows without refitting it."""

    missing = sorted(set(INPUT_ONLY_COLUMNS).difference(input_only_frame.columns))
    if missing:
        raise KeyError(f"missing input-only columns: {missing}")
    design = seasonal_design(input_only_frame["time"])
    seasonal_residual = np.full(len(input_only_frame), np.nan, dtype=np.float64)
    seasonal_scale = np.full(len(input_only_frame), np.nan, dtype=np.float64)
    keys = np.asarray(
        [
            _group_key(station, layer)
            for station, layer in zip(
                input_only_frame["station"], input_only_frame["layer"], strict=True
            )
        ],
        dtype=object,
    )
    temperature = input_only_frame["temp"].to_numpy(dtype=np.float64)
    for key in sorted(set(keys)):
        positions = np.flatnonzero(keys == key)
        beta_raw = state.seasonal_coefficients.get(str(key))
        scale = state.seasonal_scales.get(str(key))
        if beta_raw is None or scale is None:
            continue
        beta = np.asarray(beta_raw, dtype=np.float64)
        seasonal_residual[positions] = temperature[positions] - design[positions] @ beta
        seasonal_scale[positions] = scale

    peer_predictions: list[list[float]] = [[] for _ in range(len(input_only_frame))]
    peer_scales: list[list[float]] = [[] for _ in range(len(input_only_frame))]
    indexed = input_only_frame.loc[:, ["station", "layer", "time"]].copy()
    indexed["_row"] = np.arange(len(indexed), dtype=np.int64)
    indexed["_residual"] = seasonal_residual
    for station, rows in indexed.groupby("station", sort=True, observed=True):
        for _, same_time in rows.groupby("time", sort=False, observed=True):
            by_layer = {
                int(layer_value): (int(row_position), float(residual_value))
                for _, layer_value, _, row_position, residual_value in same_time.itertuples(
                    index=False,
                    name=None,
                )
                if np.isfinite(float(residual_value))
            }
            for edge, delta in state.edge_residual_deltas.items():
                edge_station, low_raw, high_raw = edge.rsplit("|", 2)
                if edge_station != str(station):
                    continue
                low_layer, high_layer = int(low_raw), int(high_raw)
                if low_layer in by_layer and high_layer in by_layer:
                    edge_scale = max(float(state.edge_residual_scales[edge]), 1.0e-6)
                    low_row, low_value = by_layer[low_layer]
                    high_row, high_value = by_layer[high_layer]
                    peer_predictions[low_row].append(high_value + delta)
                    peer_predictions[high_row].append(low_value - delta)
                    peer_scales[low_row].append(edge_scale)
                    peer_scales[high_row].append(edge_scale)

    peer_count = np.asarray([len(values) for values in peer_predictions], dtype=np.float64)
    peer_consensus = np.asarray(
        [float(np.median(values)) if values else np.nan for values in peer_predictions],
        dtype=np.float64,
    )
    peer_consensus_scale = np.asarray(
        [float(np.median(values)) if values else np.nan for values in peer_scales],
        dtype=np.float64,
    )
    has_peer = np.isfinite(peer_consensus)
    # A single-layer station (G-ORS) has no graph edge.  Its preregistered
    # fallback is the frozen seasonal residual itself, with graph_available=0
    # so the unary head can distinguish fallback rows from true peer geometry.
    graph_residual = np.where(has_peer, seasonal_residual - peer_consensus, seasonal_residual)
    safe_scale = np.where(np.isfinite(seasonal_scale), np.maximum(seasonal_scale, 1.0e-6), np.nan)
    seasonal_z = seasonal_residual / safe_scale
    graph_scale = np.where(
        has_peer,
        np.maximum(peer_consensus_scale, 1.0e-6),
        safe_scale,
    )
    graph_z = graph_residual / graph_scale
    return pd.DataFrame(
        {
            "seasonal_residual_z": seasonal_z,
            "graph_residual_z": graph_z,
            "peer_consensus_z": peer_consensus / safe_scale,
            "graph_available": (has_peer & np.isfinite(graph_z)).astype(np.float64),
            "peer_count": peer_count,
        },
        index=input_only_frame.index,
    )


def _contiguous_bounds(segment_ids: np.ndarray) -> list[tuple[int, int]]:
    segments = np.asarray(segment_ids, dtype=np.int64)
    if segments.ndim != 1:
        raise ValueError("segment IDs must be one-dimensional")
    if not len(segments):
        return []
    changes = np.flatnonzero(segments[1:] != segments[:-1]) + 1
    edges = np.concatenate(([0], changes, [len(segments)]))
    return [(int(start), int(end)) for start, end in zip(edges[:-1], edges[1:], strict=True)]


def _rolling_median(values: np.ndarray, window: int, *, center: bool) -> np.ndarray:
    minimum = max(3, window // 2)
    return (
        pd.Series(values, dtype=np.float64)
        .rolling(window=window, min_periods=minimum, center=center)
        .median()
        .to_numpy(dtype=np.float64)
    )


def build_multiscale_geometry(
    baseline_projection: pd.DataFrame,
    segment_ids: np.ndarray,
) -> pd.DataFrame:
    """Build the fixed 29-feature Haar/slope/curvature/coherence bank."""

    required = {
        "seasonal_residual_z",
        "graph_residual_z",
        "graph_available",
        "peer_count",
    }
    missing = sorted(required.difference(baseline_projection.columns))
    if missing:
        raise KeyError(f"missing baseline projection columns: {missing}")
    segments = np.asarray(segment_ids, dtype=np.int64)
    if len(segments) != len(baseline_projection):
        raise ValueError("segment IDs and baseline projection length differ")
    seasonal = baseline_projection["seasonal_residual_z"].to_numpy(dtype=np.float64)
    graph = baseline_projection["graph_residual_z"].to_numpy(dtype=np.float64)
    graph_available = baseline_projection["graph_available"].to_numpy(dtype=np.float64)
    peer_count = baseline_projection["peer_count"].to_numpy(dtype=np.float64)
    output: dict[str, np.ndarray] = {
        "abs_seasonal_residual_z": np.nan_to_num(np.abs(seasonal), nan=0.0, posinf=0.0, neginf=0.0),
        "abs_graph_residual_z": np.nan_to_num(np.abs(graph), nan=0.0, posinf=0.0, neginf=0.0),
        "graph_available": np.nan_to_num(graph_available, nan=0.0, posinf=0.0, neginf=0.0),
        "peer_count": np.nan_to_num(peer_count, nan=0.0, posinf=0.0, neginf=0.0),
    }
    bounds = _contiguous_bounds(segments)
    for rows in MULTISCALE_ROWS:
        values = {kind: np.zeros(len(seasonal), dtype=np.float64) for kind in SCALE_GEOMETRY_KINDS}
        for start, end in bounds:
            local = seasonal[start:end]
            local_graph = graph[start:end]
            center_window = max(3, rows // 4)
            center = _rolling_median(local, center_window, center=True)
            left = _rolling_median(local, rows, center=False)
            left = np.roll(left, 1)
            left[0] = np.nan
            right = _rolling_median(local[::-1], rows, center=False)[::-1]
            right = np.roll(right, -1)
            right[-1] = np.nan
            level = _rolling_median(np.abs(local), rows, center=True)
            coherence = _rolling_median(np.abs(local_graph), rows, center=True)
            values["level_abs_z"][start:end] = level
            values["haar_abs_z"][start:end] = np.abs(center - 0.5 * (left + right))
            values["slope_abs_z"][start:end] = np.abs(right - left) / float(2 * rows)
            values["curvature_abs_z"][start:end] = np.abs(left - 2.0 * center + right)
            values["coherence_deficit_z"][start:end] = coherence
        for kind, array in values.items():
            output[f"{kind}_{rows}"] = np.nan_to_num(
                array,
                copy=False,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
    result = pd.DataFrame(output, index=baseline_projection.index)
    result = result.loc[:, list(GEOMETRY_FEATURES)].astype(np.float32)
    if not np.isfinite(result.to_numpy(dtype=np.float32)).all():
        raise AssertionError("multiscale geometry is not finite")
    return result


def fit_fixed_slow_unary_head(
    train_geometry: pd.DataFrame,
    decoded_train_target: np.ndarray,
    explicit_train_ids: np.ndarray,
) -> FixedSlowUnaryState:
    """Fit the only registered supervised head on already-selected train rows.

    The caller must pass a target vector decoded only for these explicit IDs;
    accepting no full target frame makes holdout-target access unnecessary.
    """

    if tuple(train_geometry.columns) != GEOMETRY_FEATURES:
        raise ValueError("slow-unary feature order differs")
    values = train_geometry.to_numpy(dtype=np.float64)
    target = np.asarray(decoded_train_target)
    ids = np.asarray(explicit_train_ids, dtype=np.int64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("slow-unary geometry must be a finite matrix")
    if target.ndim != 1 or ids.ndim != 1 or len(values) != len(target) or len(values) != len(ids):
        raise ValueError("slow-unary train arrays differ")
    if not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("slow-unary train IDs must be non-empty and unique")
    if not np.isin(target, [0, 1]).all() or len(np.unique(target)) != 2:
        raise ValueError("slow-unary target must contain both binary classes")

    # Local imports keep the static contract audit independent of estimator
    # construction.  These are the exact, non-searching preregistered objects.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        unit_variance=False,
        copy=True,
    )
    scaled = scaler.fit_transform(values)
    model = LogisticRegression(
        penalty="l2",
        C=UNARY_C,
        solver="lbfgs",
        tol=UNARY_TOL,
        max_iter=UNARY_MAX_ITER,
        class_weight="balanced",
        fit_intercept=True,
        random_state=DETERMINISTIC_SEED,
    )
    model.fit(scaled, target.astype(np.int8, copy=False))
    iterations = int(model.n_iter_[0])
    if iterations > UNARY_MAX_ITER:
        raise AssertionError("slow-unary optimizer escaped its iteration ceiling")
    state = FixedSlowUnaryState(
        train_ids_sha256=int64_ids_sha256(ids),
        feature_names=GEOMETRY_FEATURES,
        robust_center=tuple(float(item) for item in scaler.center_),
        robust_scale=tuple(float(item) for item in scaler.scale_),
        coefficients=tuple(float(item) for item in model.coef_[0]),
        intercept=float(model.intercept_[0]),
        negative_rows=int(np.count_nonzero(target == 0)),
        positive_rows=int(np.count_nonzero(target == 1)),
        optimizer_iterations=iterations,
    )
    return FixedSlowUnaryState.from_dict(state.as_dict())


def predict_fixed_slow_unary_probability(
    geometry: pd.DataFrame,
    state: FixedSlowUnaryState,
) -> np.ndarray:
    """Reload-compatible deterministic probability inference for a frozen head."""

    validated = FixedSlowUnaryState.from_dict(state.as_dict())
    if tuple(geometry.columns) != validated.feature_names:
        raise ValueError("slow-unary inference feature order differs")
    values = geometry.to_numpy(dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("slow-unary inference geometry must be finite")
    center = np.asarray(validated.robust_center, dtype=np.float64)
    scale = np.asarray(validated.robust_scale, dtype=np.float64)
    coefficient = np.asarray(validated.coefficients, dtype=np.float64)
    logits = ((values - center) / scale) @ coefficient + validated.intercept
    probability = np.empty(len(logits), dtype=np.float64)
    nonnegative = logits >= 0.0
    probability[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exp_logit = np.exp(logits[~nonnegative])
    probability[~nonnegative] = exp_logit / (1.0 + exp_logit)
    if not np.isfinite(probability).all():
        raise AssertionError("slow-unary inference produced non-finite probability")
    return probability


def _positive_runs(mask: np.ndarray, segment_ids: np.ndarray) -> list[tuple[int, int]]:
    binary = np.asarray(mask, dtype=bool)
    segments = np.asarray(segment_ids, dtype=np.int64)
    if binary.ndim != 1 or len(binary) != len(segments):
        raise ValueError("mask/segment shape differs")
    runs: list[tuple[int, int]] = []
    for start, end in _contiguous_bounds(segments):
        local = binary[start:end]
        padded = np.concatenate(([False], local, [False])).astype(np.int8)
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        runs.extend((start + int(a), start + int(b)) for a, b in zip(starts, ends, strict=True))
    return runs


def protected_incumbent_union(
    incumbent_probability: np.ndarray,
    incumbent_prediction: np.ndarray,
    slow_probability: np.ndarray,
    segment_ids: np.ndarray,
    *,
    gate_passed: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add only sustained slow proposals or return exact incumbent bytes."""

    incumbent_prob = np.asarray(incumbent_probability)
    incumbent_pred = np.asarray(incumbent_prediction)
    slow_prob = np.asarray(slow_probability)
    segments = np.asarray(segment_ids, dtype=np.int64)
    if any(array.ndim != 1 for array in (incumbent_prob, incumbent_pred, slow_prob, segments)):
        raise ValueError("protected union arrays must be one-dimensional")
    if len({len(incumbent_prob), len(incumbent_pred), len(slow_prob), len(segments)}) != 1:
        raise ValueError("protected union array lengths differ")
    if not np.isfinite(incumbent_prob).all():
        raise ValueError("incumbent probabilities must be finite")
    if not np.isin(incumbent_pred, [0, 1]).all():
        raise ValueError("incumbent prediction must be binary")
    if not gate_passed:
        return incumbent_prob.copy(), incumbent_pred.copy(), np.zeros(len(segments), dtype=bool)
    if not np.isfinite(slow_prob).all():
        return incumbent_prob.copy(), incumbent_pred.copy(), np.zeros(len(segments), dtype=bool)

    singleton_block = np.zeros(len(segments), dtype=bool)
    for start, end in _positive_runs(incumbent_pred == 1, segments):
        if end - start <= SPIKE_SINGLETON_MAX_ROWS:
            segment_start = start
            while segment_start > 0 and segments[segment_start - 1] == segments[start]:
                segment_start -= 1
            segment_end = end
            while segment_end < len(segments) and segments[segment_end] == segments[start]:
                segment_end += 1
            lo = max(segment_start, start - SPIKE_PROTECTION_RADIUS_ROWS)
            hi = min(segment_end, end + SPIKE_PROTECTION_RADIUS_ROWS)
            singleton_block[lo:hi] = True

    raw_proposal = (slow_prob >= UNARY_THRESHOLD) & (incumbent_pred == 0) & ~singleton_block
    additions = np.zeros(len(segments), dtype=bool)
    for start, end in _positive_runs(raw_proposal, segments):
        length = end - start
        if MIN_SLOW_RUN_ROWS <= length <= MAX_SLOW_RUN_ROWS:
            additions[start:end] = True

    candidate_prob = incumbent_prob.copy()
    candidate_pred = incumbent_pred.copy()
    candidate_pred[additions] = 1
    floor = np.nextafter(np.asarray(UNARY_THRESHOLD, dtype=candidate_prob.dtype), np.inf)
    candidate_prob[additions] = np.maximum(slow_prob[additions], floor)
    if not np.array_equal(candidate_prob[incumbent_pred == 1], incumbent_prob[incumbent_pred == 1]):
        raise AssertionError("incumbent positive probabilities changed")
    if not np.array_equal(candidate_pred[incumbent_pred == 1], incumbent_pred[incumbent_pred == 1]):
        raise AssertionError("incumbent positive predictions changed")
    return candidate_prob, candidate_pred, additions


def strict_inner_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the single preregistered train-only fail-closed gate."""

    required = {
        "micro_f1_delta",
        "offset_recall_delta",
        "drift_recall_delta",
        "spike_f1_delta",
        "worst_station_layer_f1_delta",
        "normal_fp_relative_increase",
        "nondegrading_inner_blocks",
        "inner_block_count",
        "both_slow_types_observed",
        "spike_observed",
        "all_required_station_layers_observed",
        "blind_predictions_sealed_before_gate_labels",
    }
    if set(metrics) != required:
        raise ValueError("inner-gate metric keys differ")
    numeric_keys = (
        "micro_f1_delta",
        "offset_recall_delta",
        "drift_recall_delta",
        "spike_f1_delta",
        "worst_station_layer_f1_delta",
        "normal_fp_relative_increase",
        "nondegrading_inner_blocks",
        "inner_block_count",
    )
    try:
        numeric_values = np.asarray([float(metrics[key]) for key in numeric_keys], dtype=np.float64)
    except (TypeError, ValueError):
        numeric_values = np.asarray([np.nan], dtype=np.float64)
    if not np.isfinite(numeric_values).all():
        return {
            "passed": False,
            "checks": {"all_gate_metrics_finite": False},
            "fallback": "EXACT_INCUMBENT_BYTES",
        }
    offset = float(metrics["offset_recall_delta"])
    drift = float(metrics["drift_recall_delta"])
    checks = {
        "micro_f1_gain": float(metrics["micro_f1_delta"])
        >= float(GATE_THRESHOLDS["minimum_micro_f1_delta"]),
        "offset_recall_gain": offset >= float(GATE_THRESHOLDS["minimum_offset_recall_delta"]),
        "drift_recall_gain": drift >= float(GATE_THRESHOLDS["minimum_drift_recall_delta"]),
        "mean_offset_drift_recall_gain": 0.5 * (offset + drift)
        >= float(GATE_THRESHOLDS["minimum_mean_offset_drift_recall_delta"]),
        "spike_f1_nonregression": float(metrics["spike_f1_delta"])
        >= float(GATE_THRESHOLDS["minimum_spike_f1_delta"]),
        "worst_station_layer_nonregression": float(metrics["worst_station_layer_f1_delta"])
        >= float(GATE_THRESHOLDS["minimum_worst_station_layer_f1_delta"]),
        "normal_fp_guard": float(metrics["normal_fp_relative_increase"])
        <= float(GATE_THRESHOLDS["maximum_normal_fp_relative_increase"]),
        "all_inner_blocks_nondegrading": int(metrics["nondegrading_inner_blocks"])
        >= int(GATE_THRESHOLDS["minimum_nondegrading_inner_blocks"]),
        "inner_block_count_exact": int(metrics["inner_block_count"])
        == int(GATE_THRESHOLDS["required_inner_blocks"]),
        "both_slow_types_observed": metrics["both_slow_types_observed"] is True,
        "spike_observed": metrics["spike_observed"] is True,
        "all_required_station_layers_observed": metrics["all_required_station_layers_observed"]
        is True,
        "blind_predictions_sealed_before_gate_labels": metrics[
            "blind_predictions_sealed_before_gate_labels"
        ]
        is True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "fallback": "APPLY_FIXED_SLOW_UNARY" if all(checks.values()) else "EXACT_INCUMBENT_BYTES",
    }


def static_contract_audit() -> dict[str, Any]:
    """Exercise pure invariants only; perform no fit, prediction, or scoring."""

    incumbent_probability = np.asarray([0.1] * 160, dtype=np.float64)
    incumbent_prediction = np.zeros(160, dtype=np.int8)
    incumbent_prediction[20] = 1
    incumbent_probability[20] = 0.9
    slow_probability = np.asarray([0.1] * 160, dtype=np.float64)
    slow_probability[60:120] = 0.8
    segments = np.zeros(160, dtype=np.int64)
    fallback_prob, fallback_pred, fallback_add = protected_incumbent_union(
        incumbent_probability,
        incumbent_prediction,
        slow_probability,
        segments,
        gate_passed=False,
    )
    active_prob, active_pred, active_add = protected_incumbent_union(
        incumbent_probability,
        incumbent_prediction,
        slow_probability,
        segments,
        gate_passed=True,
    )
    if fallback_prob.tobytes() != incumbent_probability.tobytes():
        raise AssertionError("probability fallback is not byte-exact")
    if fallback_pred.tobytes() != incumbent_prediction.tobytes():
        raise AssertionError("prediction fallback is not byte-exact")
    if fallback_add.any() or int(active_add.sum()) != 60:
        raise AssertionError("fixed duration proposal contract differs")
    if active_prob[20] != incumbent_probability[20] or active_pred[20] != 1:
        raise AssertionError("incumbent singleton protection differs")
    return {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "feature_count": len(GEOMETRY_FEATURES),
        "multiscale_rows": list(MULTISCALE_ROWS),
        "fallback_probability_byte_exact": True,
        "fallback_prediction_byte_exact": True,
        "incumbent_singleton_preserved": True,
        "fixed_slow_addition_rows": int(active_add.sum()),
        "model_fits": 0,
        "predictions_generated": 0,
        "scores_computed": 0,
        "test_value_reads": 0,
    }


__all__ = [
    "ANNUAL_HARMONICS",
    "BASE_GEOMETRY_FEATURES",
    "DIURNAL_HARMONICS",
    "DETERMINISTIC_SEED",
    "EXPERIMENT_ID",
    "GATE_THRESHOLDS",
    "GEOMETRY_FEATURES",
    "HYPOTHESIS_ID",
    "INPUT_ONLY_COLUMNS",
    "MAX_SLOW_RUN_ROWS",
    "MIN_SLOW_RUN_ROWS",
    "MULTISCALE_ROWS",
    "RobustSeasonalGraphState",
    "FixedSlowUnaryState",
    "SCALE_GEOMETRY_KINDS",
    "SEASONAL_IRLS_ITERATIONS",
    "SLOW_TYPES",
    "UNARY_THRESHOLD",
    "UNARY_C",
    "UNARY_MAX_ITER",
    "UNARY_TOL",
    "apply_robust_seasonal_graph_state",
    "build_multiscale_geometry",
    "exact_gap_safe_segment_ids",
    "fit_fixed_slow_unary_head",
    "fit_robust_seasonal_graph_state",
    "int64_ids_sha256",
    "protected_incumbent_union",
    "predict_fixed_slow_unary_probability",
    "seasonal_design",
    "static_contract_audit",
    "strict_inner_gate",
]
