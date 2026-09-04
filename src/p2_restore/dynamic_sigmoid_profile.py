"""Leakage-safe dynamic sigmoid-profile utilities for the P2 precheck.

The module deliberately contains no submission or hidden-interval inference path.
It implements one research hypothesis: a four-parameter vertical sigmoid whose
thermocline center and log-width are predicted from public-layer state.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit
from sklearn.linear_model import Ridge

PUBLIC_LAYERS = (1, 5, 6, 7, 8)
TARGET_LAYERS = (2, 3, 4)


@dataclass(frozen=True)
class TimeBlock:
    """Half-open KST validation interval."""

    name: str
    start: pd.Timestamp
    stop: pd.Timestamp

    @classmethod
    def from_strings(cls, name: str, values: Sequence[str]) -> TimeBlock:
        if len(values) != 2:
            raise ValueError(f"block {name!r} must contain start and stop")
        start, stop = (pd.Timestamp(value) for value in values)
        if start.tz is None or stop.tz is None:
            raise ValueError(f"block {name!r} timestamps must be timezone-aware")
        start = start.tz_convert("Asia/Seoul")
        stop = stop.tz_convert("Asia/Seoul")
        if stop <= start:
            raise ValueError(f"block {name!r} is empty")
        return cls(name=name, start=start, stop=stop)

    @property
    def days(self) -> int:
        seconds = (self.stop - self.start).total_seconds()
        if seconds % 86_400:
            raise ValueError(f"block {self.name!r} is not an integer number of days")
        return int(seconds // 86_400)

    def mask(self, times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        parsed = pd.to_datetime(times, utc=True)
        start = self.start.tz_convert("UTC")
        stop = self.stop.tz_convert("UTC")
        return np.asarray((parsed >= start) & (parsed < stop), dtype=bool)

    def expanded_mask(self, times: pd.Series | pd.DatetimeIndex, *, purge_days: int) -> np.ndarray:
        parsed = pd.to_datetime(times, utc=True)
        delta = pd.Timedelta(days=purge_days)
        start = self.start.tz_convert("UTC") - delta
        stop = self.stop.tz_convert("UTC") + delta
        return np.asarray((parsed >= start) & (parsed < stop), dtype=bool)


@dataclass(frozen=True)
class SigmoidSpec:
    center_bounds_m: tuple[float, float]
    width_bounds_m: tuple[float, float]
    center_start_fractions: tuple[float, float, float]
    width_starts_m: tuple[float, float, float]
    max_nfev: int
    ftol: float
    xtol: float
    gtol: float
    boundary_fraction: float
    target_depths_m: tuple[float, float, float]


@dataclass(frozen=True)
class ProfileFit:
    success: bool
    center_m: float
    log_width: float
    width_m: float
    offset_c: float
    amplitude_c: float
    r2: float
    scaled_jacobian_condition: float
    boundary_saturated: bool
    multistart_target_spread_c: float
    point_count: int
    depth_span_m: float


@dataclass(frozen=True)
class PublicProfileFit:
    supported: bool
    offset_c: float
    amplitude_c: float
    profiled_jacobian_condition: float
    observable: bool
    target_prediction: np.ndarray
    point_count: int
    depth_span_m: float


@dataclass(frozen=True)
class LatentRidge:
    feature_columns: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    estimator: Ridge
    center_bounds_m: tuple[float, float]
    log_width_bounds: tuple[float, float]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        values = features.loc[:, self.feature_columns].to_numpy(dtype=np.float64)
        values = np.where(np.isfinite(values), values, self.medians)
        values = (values - self.means) / self.scales
        prediction = np.asarray(self.estimator.predict(values), dtype=np.float64)
        prediction[:, 0] = np.clip(prediction[:, 0], *self.center_bounds_m)
        prediction[:, 1] = np.clip(prediction[:, 1], *self.log_width_bounds)
        return prediction


def effective_depth(depth: np.ndarray, nominal_depth: np.ndarray) -> np.ndarray:
    """Use finite positive actual depth, otherwise the documented nominal depth."""

    actual = np.asarray(depth, dtype=np.float64)
    nominal = np.asarray(nominal_depth, dtype=np.float64)
    if actual.shape != nominal.shape:
        raise ValueError("depth and nominal depth shapes differ")
    use_actual = np.isfinite(actual) & (actual > 0.0)
    result = np.where(use_actual, actual, nominal)
    result[~np.isfinite(result) | (result <= 0.0)] = np.nan
    return result


def joint_mask_target_intervals(
    observations: pd.DataFrame,
    blocks: Iterable[TimeBlock],
    *,
    target_layers: Sequence[int] = TARGET_LAYERS,
) -> pd.DataFrame:
    """Mask target temperature and salinity before fold-local feature construction."""

    required = {"time", "layer", "temp", "psal"}
    if missing := required.difference(observations.columns):
        raise ValueError(f"observations missing columns: {sorted(missing)}")
    result = observations.copy()
    times = pd.to_datetime(result["time"], utc=True)
    selected = np.zeros(len(result), dtype=bool)
    for block in blocks:
        selected |= block.mask(times)
    selected &= result["layer"].isin(target_layers).to_numpy()
    result.loc[selected, ["temp", "psal"]] = np.nan
    if not result.loc[selected, ["temp", "psal"]].isna().all().all():
        raise AssertionError("joint target mask failed")
    return result


def _profiled_linear_fit(
    depth: np.ndarray, temperature: np.ndarray, center_m: float, log_width: float
) -> tuple[np.ndarray, float, float, np.ndarray]:
    width = float(np.exp(log_width))
    basis = expit((center_m - depth) / width)
    design = np.column_stack((np.ones(len(depth), dtype=np.float64), basis))
    coefficient, *_ = np.linalg.lstsq(design, temperature, rcond=None)
    fitted = design @ coefficient
    residual = fitted - temperature
    return residual, float(coefficient[0]), float(coefficient[1]), fitted


def _scaled_condition(jacobian: np.ndarray, *, tolerance: float = 1e-10) -> float:
    values = np.asarray(jacobian, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        return float("inf")
    norm = np.linalg.norm(values, axis=0)
    if np.any(norm <= tolerance):
        return float("inf")
    singular = np.linalg.svd(values / norm, compute_uv=False)
    if len(singular) < values.shape[1] or singular[-1] <= tolerance:
        return float("inf")
    return float(singular[0] / singular[-1])


def _full_profile_jacobian(
    depth: np.ndarray,
    *,
    center_m: float,
    log_width: float,
    amplitude_c: float,
) -> np.ndarray:
    width = float(np.exp(log_width))
    scaled = (center_m - depth) / width
    basis = expit(scaled)
    derivative = basis * (1.0 - basis)
    return np.column_stack(
        (
            np.ones(len(depth), dtype=np.float64),
            basis,
            amplitude_c * derivative / width,
            -amplitude_c * derivative * scaled,
        )
    )


def _clean_profile(depth: np.ndarray, temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(depth, dtype=np.float64)
    y = np.asarray(temperature, dtype=np.float64)
    keep = np.isfinite(z) & np.isfinite(y) & (z > 0.0)
    z, y = z[keep], y[keep]
    if len(z) == 0:
        return z, y
    order = np.argsort(z)
    z, y = z[order], y[order]
    unique = np.concatenate(([True], np.diff(z) > 1e-6))
    return z[unique], y[unique]


def fit_sigmoid_profile(
    depth: np.ndarray,
    temperature: np.ndarray,
    spec: SigmoidSpec,
    *,
    minimum_points: int,
) -> ProfileFit:
    """Fit a four-parameter sigmoid with 2-D bounded variable projection."""

    z, y = _clean_profile(depth, temperature)
    failed = ProfileFit(
        False,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.inf,
        True,
        np.inf,
        int(len(z)),
        float(np.ptp(z)) if len(z) else 0.0,
    )
    if len(z) < minimum_points:
        return failed
    center_low = max(float(spec.center_bounds_m[0]), float(z.min()))
    center_high = min(float(spec.center_bounds_m[1]), float(z.max()))
    if center_high - center_low <= 1e-6:
        return failed
    log_width_low, log_width_high = np.log(np.asarray(spec.width_bounds_m, dtype=float))
    target_depths = np.asarray(spec.target_depths_m, dtype=np.float64)
    solutions: list[tuple[float, np.ndarray, float, float, np.ndarray]] = []

    def residual(parameter: np.ndarray) -> np.ndarray:
        current, *_ = _profiled_linear_fit(z, y, float(parameter[0]), float(parameter[1]))
        return current

    for fraction, width_start in zip(spec.center_start_fractions, spec.width_starts_m, strict=True):
        start = np.array(
            [
                center_low + float(fraction) * (center_high - center_low),
                np.log(np.clip(width_start, *spec.width_bounds_m)),
            ],
            dtype=np.float64,
        )
        try:
            result = least_squares(
                residual,
                start,
                bounds=(
                    np.array([center_low, log_width_low]),
                    np.array([center_high, log_width_high]),
                ),
                max_nfev=spec.max_nfev,
                ftol=spec.ftol,
                xtol=spec.xtol,
                gtol=spec.gtol,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.isfinite(result.x).all():
            continue
        current_residual, offset, amplitude, fitted = _profiled_linear_fit(
            z, y, float(result.x[0]), float(result.x[1])
        )
        sse = float(np.dot(current_residual, current_residual))
        target_basis = expit((float(result.x[0]) - target_depths) / np.exp(result.x[1]))
        target_prediction = offset + amplitude * target_basis
        solutions.append((sse, result.x.copy(), offset, amplitude, target_prediction))
    if not solutions:
        return failed

    solutions.sort(key=lambda item: item[0])
    sse, parameter, offset, amplitude, _ = solutions[0]
    center_m, log_width = float(parameter[0]), float(parameter[1])
    width_m = float(np.exp(log_width))
    total = float(np.sum(np.square(y - np.mean(y))))
    r2 = float(1.0 - sse / total) if total > 1e-12 else np.nan
    jacobian = _full_profile_jacobian(
        z,
        center_m=center_m,
        log_width=log_width,
        amplitude_c=amplitude,
    )
    condition = _scaled_condition(jacobian)
    stacked_target = np.stack([item[4] for item in solutions])
    spread = float(np.max(np.ptp(stacked_target, axis=0))) if len(solutions) > 1 else 0.0
    center_margin = spec.boundary_fraction * (center_high - center_low)
    log_margin = spec.boundary_fraction * (log_width_high - log_width_low)
    boundary = bool(
        center_m <= center_low + center_margin
        or center_m >= center_high - center_margin
        or log_width <= log_width_low + log_margin
        or log_width >= log_width_high - log_margin
    )
    return ProfileFit(
        True,
        center_m,
        log_width,
        width_m,
        offset,
        amplitude,
        r2,
        condition,
        boundary,
        spread,
        int(len(z)),
        float(np.ptp(z)),
    )


def _public_profiled_residual(
    depth: np.ndarray,
    temperature: np.ndarray,
    center_m: float,
    log_width: float,
) -> np.ndarray:
    residual, *_ = _profiled_linear_fit(depth, temperature, center_m, log_width)
    return residual


def fit_public_profile(
    depth: np.ndarray,
    temperature: np.ndarray,
    target_depth: np.ndarray,
    *,
    center_m: float,
    log_width: float,
    minimum_points: int,
    minimum_depth_span_m: float,
    center_step_m: float,
    log_width_step: float,
    condition_max: float,
) -> PublicProfileFit:
    """Profile out offset/amplitude and diagnose public-only center/width observability."""

    z, y = _clean_profile(depth, temperature)
    targets = np.asarray(target_depth, dtype=np.float64)
    empty = PublicProfileFit(
        False,
        np.nan,
        np.nan,
        np.inf,
        False,
        np.full(len(targets), np.nan, dtype=np.float64),
        int(len(z)),
        float(np.ptp(z)) if len(z) else 0.0,
    )
    if (
        len(z) < minimum_points
        or float(np.ptp(z)) < minimum_depth_span_m
        or not np.isfinite([center_m, log_width]).all()
    ):
        return empty
    _, offset, amplitude, _ = _profiled_linear_fit(z, y, center_m, log_width)
    target_basis = expit((center_m - targets) / np.exp(log_width))
    prediction = offset + amplitude * target_basis
    center_plus = _public_profiled_residual(z, y, center_m + center_step_m, log_width)
    center_minus = _public_profiled_residual(z, y, center_m - center_step_m, log_width)
    width_plus = _public_profiled_residual(z, y, center_m, log_width + log_width_step)
    width_minus = _public_profiled_residual(z, y, center_m, log_width - log_width_step)
    jacobian = np.column_stack(
        (
            (center_plus - center_minus) / (2.0 * center_step_m),
            (width_plus - width_minus) / (2.0 * log_width_step),
        )
    )
    condition = _scaled_condition(jacobian)
    return PublicProfileFit(
        True,
        offset,
        amplitude,
        condition,
        bool(np.isfinite(condition) and condition <= condition_max),
        prediction,
        int(len(z)),
        float(np.ptp(z)),
    )


def build_public_features(
    observations: pd.DataFrame,
    *,
    public_layers: Sequence[int] = PUBLIC_LAYERS,
    gradient_pairs: Sequence[Sequence[int]] = ((1, 5), (5, 6), (6, 7), (7, 8)),
    change_hours: Sequence[int] = (6, 24, 72, 168),
) -> pd.DataFrame:
    """Build the fixed public-only feature vector on an exact timestamp index."""

    required = {"time", "layer", "temp", "psal", "depth", "nominal_depth"}
    if missing := required.difference(observations.columns):
        raise ValueError(f"observations missing columns: {sorted(missing)}")
    public = observations.loc[observations["layer"].isin(public_layers)].copy()
    public["time"] = pd.to_datetime(public["time"], utc=True)
    public["effective_depth"] = effective_depth(
        public["depth"].to_numpy(float), public["nominal_depth"].to_numpy(float)
    )
    if public.duplicated(["time", "layer"]).any():
        raise ValueError("public time/layer keys are not unique")
    index = pd.DatetimeIndex(sorted(public["time"].unique()))

    def wide(value: str) -> pd.DataFrame:
        return public.pivot(index="time", columns="layer", values=value).reindex(index)

    temp, psal, depth = wide("temp"), wide("psal"), wide("effective_depth")
    frame = pd.DataFrame(index=index)
    for layer in public_layers:
        for prefix, table in (("temp", temp), ("psal", psal), ("depth", depth)):
            frame[f"{prefix}_{layer}"] = table.get(
                layer, pd.Series(index=index, dtype=float)
            ).to_numpy(float)
        frame[f"temp_missing_{layer}"] = (~np.isfinite(frame[f"temp_{layer}"])).astype(float)
        frame[f"psal_missing_{layer}"] = (~np.isfinite(frame[f"psal_{layer}"])).astype(float)

    for left, right in gradient_pairs:
        span = frame[f"depth_{right}"] - frame[f"depth_{left}"]
        for variable in ("temp", "psal"):
            numerator = frame[f"{variable}_{right}"] - frame[f"{variable}_{left}"]
            frame[f"{variable}_gradient_{left}_{right}"] = np.divide(
                numerator,
                span,
                out=np.full(len(frame), np.nan, dtype=np.float64),
                where=np.isfinite(numerator) & np.isfinite(span) & (np.abs(span) > 1e-6),
            )

    temp_values = frame[[f"temp_{layer}" for layer in public_layers]].to_numpy(float)
    psal_values = frame[[f"psal_{layer}" for layer in public_layers]].to_numpy(float)
    depth_values = frame[[f"depth_{layer}" for layer in public_layers]].to_numpy(float)
    temp_count = np.isfinite(temp_values).sum(axis=1)
    psal_count = np.isfinite(psal_values).sum(axis=1)
    frame["public_temp_count"] = temp_count
    frame["public_psal_count"] = psal_count
    frame["public_temp_range"] = _finite_row_range(temp_values)
    frame["public_psal_range"] = _finite_row_range(psal_values)
    frame["public_depth_span"] = _finite_row_range(
        np.where(np.isfinite(temp_values), depth_values, np.nan)
    )
    frame["temp_1_minus_5"] = frame["temp_1"] - frame["temp_5"]

    for signal in ("temp_1_minus_5", "public_temp_range"):
        current = frame[signal]
        for hours in change_hours:
            lag_index = frame.index - pd.Timedelta(hours=int(hours))
            lag = current.reindex(lag_index).to_numpy(float)
            frame[f"{signal}_change_{hours}h"] = current.to_numpy(float) - lag

    kst = frame.index.tz_convert("Asia/Seoul")
    seconds = frame.index.as_unit("ns").asi8 / 1e9
    day = kst.dayofyear.to_numpy() + (kst.hour.to_numpy() * 60 + kst.minute.to_numpy()) / 1440
    frame["annual_sin"] = np.sin(2.0 * np.pi * day / 365.2425)
    frame["annual_cos"] = np.cos(2.0 * np.pi * day / 365.2425)
    frame["m2_sin"] = np.sin(2.0 * np.pi * seconds / (12.42 * 3600.0))
    frame["m2_cos"] = np.cos(2.0 * np.pi * seconds / (12.42 * 3600.0))
    if any(f"temp_{layer}" in frame for layer in TARGET_LAYERS):
        raise AssertionError("target-layer temperature entered public features")
    return frame.sort_index()


def _finite_row_range(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    result = np.full(len(array), np.nan, dtype=np.float64)
    for row in range(len(array)):
        finite = array[row, np.isfinite(array[row])]
        if len(finite):
            result[row] = float(np.ptp(finite))
    return result


def feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return the fixed model features, excluding raw depth and support diagnostics."""

    excluded = {"public_temp_count", "public_psal_count", "public_depth_span"}
    columns = tuple(
        column
        for column in frame.columns
        if column not in excluded and not column.startswith("depth_")
    )
    forbidden = {f"temp_{layer}" for layer in TARGET_LAYERS} | {
        f"psal_{layer}" for layer in TARGET_LAYERS
    }
    if forbidden.intersection(columns):
        raise AssertionError("target-layer value exposed as a feature")
    return columns


def fit_latent_ridge(
    features: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    columns: Sequence[str],
    alpha: float,
    minimum_feature_coverage: float,
    minimum_rows: int,
    center_bounds_m: tuple[float, float],
    width_bounds_m: tuple[float, float],
    training_mask: np.ndarray | None = None,
) -> LatentRidge:
    """Fit h/log(width) from public features with fold-local preprocessing."""

    if catalog.empty:
        raise ValueError("parameter catalog is empty")
    keyed = catalog.copy()
    keyed["time"] = pd.to_datetime(keyed["time"], utc=True)
    if training_mask is not None:
        if len(training_mask) != len(keyed):
            raise ValueError("ridge training mask length differs")
        keyed = keyed.loc[np.asarray(training_mask, dtype=bool)]
    joined = keyed.join(features.loc[:, list(columns)], on="time", how="inner")
    joined = joined.loc[
        np.isfinite(joined["center_m"])
        & np.isfinite(joined["log_width"])
        & np.isfinite(joined["sample_weight"])
        & (joined["sample_weight"] > 0.0)
    ]
    if len(joined) < minimum_rows:
        raise ValueError(f"only {len(joined)} stable parameter rows for ridge")
    raw = joined.loc[:, list(columns)].to_numpy(dtype=np.float64)
    coverage = np.isfinite(raw).mean(axis=0)
    selected = coverage >= minimum_feature_coverage
    if not selected.any():
        raise ValueError("no public features meet minimum coverage")
    selected_columns = tuple(np.asarray(columns, dtype=object)[selected].tolist())
    raw = raw[:, selected]
    medians = np.nanmedian(raw, axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("ridge medians are non-finite")
    raw = np.where(np.isfinite(raw), raw, medians)
    means = raw.mean(axis=0)
    scales = raw.std(axis=0)
    scales = np.where(scales > 1e-8, scales, 1.0)
    standardized = (raw - means) / scales
    target = joined[["center_m", "log_width"]].to_numpy(dtype=np.float64)
    estimator = Ridge(alpha=float(alpha), fit_intercept=True)
    estimator.fit(
        standardized,
        target,
        sample_weight=joined["sample_weight"].to_numpy(dtype=np.float64),
    )
    return LatentRidge(
        selected_columns,
        medians,
        means,
        scales,
        estimator,
        center_bounds_m,
        tuple(np.log(np.asarray(width_bounds_m, dtype=float))),
    )


def stable_parameter_mask(
    catalog: pd.DataFrame,
    *,
    minimum_abs_amplitude_c: float,
    minimum_r2: float,
    maximum_condition: float,
    maximum_spread_c: float,
) -> np.ndarray:
    return (
        catalog["success"].astype(bool).to_numpy()
        & (np.abs(catalog["amplitude_c"].to_numpy(float)) >= minimum_abs_amplitude_c)
        & (catalog["r2"].to_numpy(float) >= minimum_r2)
        & (catalog["scaled_jacobian_condition"].to_numpy(float) <= maximum_condition)
        & (catalog["multistart_target_spread_c"].to_numpy(float) <= maximum_spread_c)
        & ~catalog["boundary_saturated"].astype(bool).to_numpy()
    )


def closed_form_convex_alpha(
    truth: np.ndarray,
    incumbent: np.ndarray,
    challenger: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> float:
    """One common convex weight; a zero/orthogonal direction returns exact no-op."""

    y = np.asarray(truth, dtype=np.float64)
    base = np.asarray(incumbent, dtype=np.float64)
    other = np.asarray(challenger, dtype=np.float64)
    if y.shape != base.shape or y.shape != other.shape:
        raise ValueError("alpha inputs have different shapes")
    if not (np.isfinite(y).all() and np.isfinite(base).all() and np.isfinite(other).all()):
        raise ValueError("alpha inputs must be finite")
    direction = other - base
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-15:
        return 0.0
    alpha = float(np.dot(direction, y - base) / denominator)
    return float(np.clip(alpha, *bounds))


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if y.shape != pred.shape or len(y) == 0:
        raise ValueError("RMSE inputs are empty or misaligned")
    return float(np.sqrt(np.mean(np.square(y - pred))))


def fit_parameter_catalog(
    observations: pd.DataFrame,
    *,
    spec: SigmoidSpec,
    allowed_time: Callable[[pd.DatetimeIndex], np.ndarray],
    stride_minutes: int,
    minimum_points: int,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Fit aggregate-safe per-profile sigmoid labels on allowed fold-local timestamps."""

    frame = observations.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["effective_depth"] = effective_depth(
        frame["depth"].to_numpy(float), frame["nominal_depth"].to_numpy(float)
    )
    temp = frame.pivot(index="time", columns="layer", values="temp").sort_index()
    depth = frame.pivot(index="time", columns="layer", values="effective_depth").reindex(temp.index)
    index = pd.DatetimeIndex(temp.index)
    kst = index.tz_convert("Asia/Seoul")
    minute_of_day = kst.hour.to_numpy() * 60 + kst.minute.to_numpy()
    selected = (minute_of_day % int(stride_minutes) == 0) & allowed_time(index)
    rows = np.flatnonzero(selected)
    records: list[dict[str, object]] = []
    for number, row in enumerate(rows, start=1):
        result = fit_sigmoid_profile(
            depth.iloc[row].to_numpy(float),
            temp.iloc[row].to_numpy(float),
            spec,
            minimum_points=minimum_points,
        )
        records.append(
            {
                "time": index[row],
                "success": result.success,
                "center_m": result.center_m,
                "log_width": result.log_width,
                "width_m": result.width_m,
                "offset_c": result.offset_c,
                "amplitude_c": result.amplitude_c,
                "r2": result.r2,
                "scaled_jacobian_condition": result.scaled_jacobian_condition,
                "boundary_saturated": result.boundary_saturated,
                "multistart_target_spread_c": result.multistart_target_spread_c,
                "point_count": result.point_count,
                "depth_span_m": result.depth_span_m,
            }
        )
        if progress is not None and (number == len(rows) or number % 100 == 0):
            progress(number, len(rows))
    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise ValueError("no training profiles selected for sigmoid fit")
    result["sample_weight"] = np.clip(result["r2"].fillna(0.0).to_numpy(float), 0.05, 1.0)
    return result


def public_profile_arrays(
    feature_row: pd.Series,
    *,
    public_layers: Sequence[int] = PUBLIC_LAYERS,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray([feature_row[f"depth_{layer}"] for layer in public_layers], dtype=float)
    temp = np.asarray([feature_row[f"temp_{layer}"] for layer in public_layers], dtype=float)
    return depth, temp


def target_depth_frame(observations: pd.DataFrame) -> pd.DataFrame:
    """Return key/depth only; target temperature and salinity are never included."""

    selected = observations.loc[observations["layer"].isin(TARGET_LAYERS)].copy()
    selected["time"] = pd.to_datetime(selected["time"], utc=True)
    selected["target_depth"] = effective_depth(
        selected["depth"].to_numpy(float), selected["nominal_depth"].to_numpy(float)
    )
    result = selected.loc[:, ["time", "layer", "target_depth"]]
    if result.duplicated(["time", "layer"]).any():
        raise ValueError("target depth keys are not unique")
    return result
