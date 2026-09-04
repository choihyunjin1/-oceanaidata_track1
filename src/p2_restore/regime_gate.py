"""Missing-aware public-state soft gating for frozen P2 contributors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from p2_restore.features import PUBLIC_LAYERS

M2_PERIOD_HOURS = 12.42
M2_DETREND_STEPS = 144
M2_DETREND_MIN = 72
M2_WINDOW_STEPS = 1008
M2_WINDOW_MIN = 504

STATE_FEATURES = (
    "abs_t1_t5",
    "public_temp_range",
    "public_psal_range",
    "contrast_delta_past_24h",
    "temp_range_delta_past_24h",
    "contrast_center_change_24h",
    "temp_range_center_change_24h",
    "m2_amplitude_mean",
    "m2_amplitude_spread",
    "m2_phase_coherence",
)


@dataclass(frozen=True)
class RobustStateTransform:
    feature_names: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_names: Sequence[str]) -> RobustStateTransform:
        names = tuple(feature_names)
        values = frame.loc[:, names].to_numpy(float)
        center = np.zeros(len(names), dtype=np.float64)
        scale = np.ones(len(names), dtype=np.float64)
        for column in range(len(names)):
            finite = values[np.isfinite(values[:, column]), column]
            if not len(finite):
                continue
            center[column] = np.median(finite)
            spread = np.subtract(*np.quantile(finite, [0.75, 0.25]))
            if np.isfinite(spread) and spread > 1e-8:
                scale[column] = spread
        return cls(names, center, scale)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, self.feature_names].to_numpy(float)
        missing = ~np.isfinite(values)
        standardized = np.divide(
            values - self.center,
            self.scale,
            out=np.zeros_like(values),
            where=~missing,
        )
        standardized = np.clip(standardized, -10.0, 10.0)
        return np.column_stack(
            [np.ones(len(frame), dtype=np.float64), standardized, missing.astype(np.float64)]
        )


@dataclass(frozen=True)
class LayerSoftGate:
    layer: int
    prior: np.ndarray
    coefficients: np.ndarray
    transform: RobustStateTransform
    regularization: float
    optimizer_iterations: int
    objective_mse: float


@dataclass(frozen=True)
class SoftRegimeGate:
    feature_names: tuple[str, ...]
    prediction_columns: tuple[str, ...]
    regularization: float
    layers: dict[int, LayerSoftGate]


@dataclass(frozen=True)
class NestedGateResult:
    prediction: np.ndarray
    baseline_prediction: np.ndarray
    selected_regularization: dict[str, float]
    inner_scores: dict[str, list[dict[str, float]]]
    final_regularization: float
    final_gate: SoftRegimeGate


def _segment_ids(times: pd.DatetimeIndex, cadence_minutes: int = 10) -> np.ndarray:
    difference = pd.Series(times).diff().dt.total_seconds().div(60)
    return difference.ne(cadence_minutes).cumsum().to_numpy()


def build_public_state_features(observations: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Build exactly ten public-layer features aligned to ``keys``.

    Only layers 1, 5, 6, 7 and 8 are read. Centered public context is valid for
    the offline restoration task and every rolling calculation resets at gaps.
    """

    required = {"time", "layer", "temp", "psal"}
    if not required.issubset(observations.columns) or not {"time", "layer"}.issubset(keys):
        raise ValueError("public-state input schema is incomplete")
    public = observations.loc[
        observations["layer"].isin(PUBLIC_LAYERS), ["time", "layer", "temp", "psal"]
    ].copy()
    public["time_key"] = pd.to_datetime(public["time"], utc=True)
    times = pd.Index(sorted(public["time_key"].unique()))
    temperature = public.pivot(index="time_key", columns="layer", values="temp").reindex(times)
    salinity = public.pivot(index="time_key", columns="layer", values="psal").reindex(times)
    segment = _segment_ids(pd.DatetimeIndex(times))
    derived = pd.DataFrame(index=times)
    contrast = (temperature.get(1) - temperature.get(5)).abs()
    temp_range = temperature.max(axis=1) - temperature.min(axis=1)
    derived["abs_t1_t5"] = contrast
    derived["public_temp_range"] = temp_range
    derived["public_psal_range"] = salinity.max(axis=1) - salinity.min(axis=1)
    contrast_group = contrast.groupby(segment, sort=False)
    range_group = temp_range.groupby(segment, sort=False)
    derived["contrast_delta_past_24h"] = contrast - contrast_group.shift(144)
    derived["temp_range_delta_past_24h"] = temp_range - range_group.shift(144)
    derived["contrast_center_change_24h"] = contrast_group.shift(-72) - contrast_group.shift(72)
    derived["temp_range_center_change_24h"] = range_group.shift(-72) - range_group.shift(72)

    epoch_seconds = pd.to_datetime(times, utc=True).as_unit("ns").asi8 / 1e9
    angle = 2 * np.pi * epoch_seconds / (M2_PERIOD_HOURS * 3600)
    cosine = pd.Series(np.cos(angle), index=times)
    sine = pd.Series(np.sin(angle), index=times)
    amplitudes: list[np.ndarray] = []
    phase_cosines: list[np.ndarray] = []
    phase_sines: list[np.ndarray] = []
    for layer in PUBLIC_LAYERS:
        series = temperature.get(layer, pd.Series(index=times, dtype=float)).astype(float)
        coefficient_cos = pd.Series(np.nan, index=times, dtype=float)
        coefficient_sin = pd.Series(np.nan, index=times, dtype=float)
        for segment_id in pd.unique(segment):
            keep = segment == segment_id
            local = series.loc[keep]
            background = local.rolling(
                M2_DETREND_STEPS,
                center=True,
                min_periods=M2_DETREND_MIN,
            ).mean()
            residual = local - background
            coefficient_cos.loc[keep] = (
                2
                * (residual * cosine.loc[keep])
                .rolling(M2_WINDOW_STEPS, center=True, min_periods=M2_WINDOW_MIN)
                .mean()
            )
            coefficient_sin.loc[keep] = (
                2
                * (residual * sine.loc[keep])
                .rolling(M2_WINDOW_STEPS, center=True, min_periods=M2_WINDOW_MIN)
                .mean()
            )
        amplitude = np.sqrt(coefficient_cos**2 + coefficient_sin**2).to_numpy(float)
        amplitudes.append(amplitude)
        phase_cosines.append(
            np.divide(
                coefficient_cos.to_numpy(float),
                amplitude,
                out=np.full(len(times), np.nan),
                where=amplitude > 1e-8,
            )
        )
        phase_sines.append(
            np.divide(
                coefficient_sin.to_numpy(float),
                amplitude,
                out=np.full(len(times), np.nan),
                where=amplitude > 1e-8,
            )
        )

    amplitude_matrix = np.column_stack(amplitudes)
    valid_amplitude = np.isfinite(amplitude_matrix)
    amplitude_count = valid_amplitude.sum(axis=1)
    derived["m2_amplitude_mean"] = np.divide(
        np.nansum(amplitude_matrix, axis=1),
        amplitude_count,
        out=np.full(len(times), np.nan),
        where=amplitude_count > 0,
    )
    amplitude_range = np.full(len(times), np.nan)
    populated = amplitude_count > 0
    amplitude_range[populated] = np.nanmax(amplitude_matrix[populated], axis=1) - np.nanmin(
        amplitude_matrix[populated], axis=1
    )
    derived["m2_amplitude_spread"] = amplitude_range
    cos_matrix = np.column_stack(phase_cosines)
    sin_matrix = np.column_stack(phase_sines)
    phase_valid = np.isfinite(cos_matrix) & np.isfinite(sin_matrix)
    phase_count = phase_valid.sum(axis=1)
    mean_cos = np.divide(
        np.nansum(cos_matrix, axis=1),
        phase_count,
        out=np.full(len(times), np.nan),
        where=phase_count > 0,
    )
    mean_sin = np.divide(
        np.nansum(sin_matrix, axis=1),
        phase_count,
        out=np.full(len(times), np.nan),
        where=phase_count > 0,
    )
    derived["m2_phase_coherence"] = np.sqrt(mean_cos**2 + mean_sin**2)

    keyed = keys.loc[:, ["time", "layer"]].copy()
    keyed["_time_key"] = pd.to_datetime(keyed["time"], utc=True)
    result = keyed.join(derived.loc[:, STATE_FEATURES], on="_time_key", validate="many_to_one")
    result = result.drop(columns="_time_key")
    if len(result) != len(keys) or result[["time", "layer"]].duplicated().any():
        raise ValueError("public-state features lost or duplicated key rows")
    if any(f"temp_{layer}" in result.columns for layer in (2, 3, 4)):
        raise AssertionError("target-layer temperature leaked into public-state features")
    return result


def fit_simplex_weights(predictions: np.ndarray, truth: np.ndarray) -> np.ndarray:
    inputs = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if inputs.ndim != 2 or len(inputs) != len(target) or not np.isfinite(inputs).all():
        raise ValueError("simplex training arrays are invalid")

    def objective(weights: np.ndarray) -> float:
        return float(np.mean((inputs @ weights - target) ** 2))

    result = minimize(
        objective,
        np.full(inputs.shape[1], 1.0 / inputs.shape[1]),
        bounds=[(0.0, 1.0)] * inputs.shape[1],
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success or not np.isclose(result.x.sum(), 1.0, atol=1e-7):
        raise RuntimeError(f"simplex optimization failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def _fit_layer_gate(
    frame: pd.DataFrame,
    *,
    layer: int,
    feature_names: tuple[str, ...],
    prediction_columns: tuple[str, ...],
    regularization: float,
) -> LayerSoftGate:
    selected = frame["layer"].to_numpy(int) == layer
    current = frame.loc[selected]
    transform = RobustStateTransform.fit(current, feature_names)
    design = transform.transform(current)
    predictions = current.loc[:, prediction_columns].to_numpy(float)
    truth = current["truth"].to_numpy(float)
    prior = fit_simplex_weights(predictions, truth)
    log_prior = np.log(np.maximum(prior, 1e-8))
    rows, design_columns = design.shape
    models = predictions.shape[1]

    def objective_and_gradient(flat: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = flat.reshape(design_columns, models)
        logits = log_prior + design @ coefficients
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        estimate = np.sum(weights * predictions, axis=1)
        error = estimate - truth
        penalty = regularization * float(np.mean(coefficients**2))
        objective = float(np.mean(error**2) + penalty)
        logit_gradient = (2.0 / rows) * error[:, None] * weights * (predictions - estimate[:, None])
        gradient = design.T @ logit_gradient
        if regularization:
            gradient += 2.0 * regularization * coefficients / coefficients.size
        return objective, gradient.ravel()

    initial = np.zeros(design_columns * models, dtype=np.float64)
    optimized = minimize(
        objective_and_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not optimized.success and optimized.status not in {1, 2}:
        raise RuntimeError(f"soft-gate optimization failed: {optimized.message}")
    coefficients = optimized.x.reshape(design_columns, models)
    return LayerSoftGate(
        layer=layer,
        prior=prior,
        coefficients=coefficients,
        transform=transform,
        regularization=float(regularization),
        optimizer_iterations=int(optimized.nit),
        objective_mse=float(objective_and_gradient(optimized.x)[0]),
    )


def fit_soft_gate(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str] = STATE_FEATURES,
    prediction_columns: Sequence[str],
    regularization: float,
) -> SoftRegimeGate:
    names = tuple(feature_names)
    predictions = tuple(prediction_columns)
    required = {"layer", "truth", *names, *predictions}
    if not required.issubset(frame.columns):
        raise ValueError(f"soft-gate frame is missing {sorted(required - set(frame.columns))}")
    layers = {
        layer: _fit_layer_gate(
            frame,
            layer=layer,
            feature_names=names,
            prediction_columns=predictions,
            regularization=regularization,
        )
        for layer in (2, 3, 4)
    }
    return SoftRegimeGate(names, predictions, float(regularization), layers)


def predict_soft_gate(gate: SoftRegimeGate, frame: pd.DataFrame) -> np.ndarray:
    weights = soft_gate_weights(gate, frame)
    inputs = frame.loc[:, gate.prediction_columns].to_numpy(float)
    prediction = np.sum(weights * inputs, axis=1)
    if not np.isfinite(prediction).all():
        raise ValueError("soft gate produced non-finite predictions")
    return prediction


def soft_gate_weights(gate: SoftRegimeGate, frame: pd.DataFrame) -> np.ndarray:
    """Return row-aligned nonnegative contributor weights that sum to one."""

    result = np.full((len(frame), len(gate.prediction_columns)), np.nan, dtype=np.float64)
    for layer, fitted in gate.layers.items():
        selected = frame["layer"].to_numpy(int) == layer
        current = frame.loc[selected]
        design = fitted.transform.transform(current)
        logits = np.log(np.maximum(fitted.prior, 1e-8)) + design @ fitted.coefficients
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        result[selected] = weights
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("soft gate produced invalid contributor weights")
    return result


def predict_simplex_baseline(
    train: pd.DataFrame, test: pd.DataFrame, prediction_columns: Sequence[str]
) -> np.ndarray:
    columns = tuple(prediction_columns)
    prediction = np.full(len(test), np.nan, dtype=np.float64)
    for layer in (2, 3, 4):
        train_rows = train["layer"].to_numpy(int) == layer
        test_rows = test["layer"].to_numpy(int) == layer
        weights = fit_simplex_weights(
            train.loc[train_rows, columns].to_numpy(float),
            train.loc[train_rows, "truth"].to_numpy(float),
        )
        prediction[test_rows] = test.loc[test_rows, columns].to_numpy(float) @ weights
    return prediction


def nested_lobo_soft_gate(
    frame: pd.DataFrame,
    *,
    regularization_grid: Sequence[float],
    feature_names: Sequence[str] = STATE_FEATURES,
    prediction_columns: Sequence[str],
) -> NestedGateResult:
    blocks = tuple(sorted(frame["block"].unique()))
    if len(blocks) != 3:
        raise ValueError("soft-gate nested LOBO requires exactly three outer blocks")
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    baseline = np.full(len(frame), np.nan, dtype=np.float64)
    selected: dict[str, float] = {}
    inner_scores: dict[str, list[dict[str, float]]] = {}
    truth = frame["truth"].to_numpy(float)
    for outer in blocks:
        outer_rows = frame["block"].to_numpy() == outer
        train_blocks = [block for block in blocks if block != outer]
        scores: list[dict[str, float]] = []
        for regularization in regularization_grid:
            squared_error = 0.0
            count = 0
            for inner_held in train_blocks:
                inner_train = frame["block"].to_numpy() == next(
                    block for block in train_blocks if block != inner_held
                )
                inner_test = frame["block"].to_numpy() == inner_held
                gate = fit_soft_gate(
                    frame.loc[inner_train],
                    feature_names=feature_names,
                    prediction_columns=prediction_columns,
                    regularization=float(regularization),
                )
                current = predict_soft_gate(gate, frame.loc[inner_test])
                error = current - truth[inner_test]
                squared_error += float(error @ error)
                count += len(error)
            scores.append(
                {
                    "regularization": float(regularization),
                    "rmse": float((squared_error / count) ** 0.5),
                }
            )
        chosen = min(scores, key=lambda row: (row["rmse"], -row["regularization"]))
        selected[outer] = float(chosen["regularization"])
        inner_scores[outer] = scores
        outer_train = ~outer_rows
        gate = fit_soft_gate(
            frame.loc[outer_train],
            feature_names=feature_names,
            prediction_columns=prediction_columns,
            regularization=selected[outer],
        )
        prediction[outer_rows] = predict_soft_gate(gate, frame.loc[outer_rows])
        baseline[outer_rows] = predict_simplex_baseline(
            frame.loc[outer_train], frame.loc[outer_rows], prediction_columns
        )
    if not np.isfinite(prediction).all() or not np.isfinite(baseline).all():
        raise ValueError("nested soft-gate output is incomplete")
    final_regularization = float(np.median(list(selected.values())))
    final_gate = fit_soft_gate(
        frame,
        feature_names=feature_names,
        prediction_columns=prediction_columns,
        regularization=final_regularization,
    )
    return NestedGateResult(
        prediction,
        baseline,
        selected,
        inner_scores,
        final_regularization,
        final_gate,
    )
