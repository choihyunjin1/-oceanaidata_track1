"""Public-only one-mode thermocline-heave correction for a frozen P2 incumbent.

The module estimates a single vertical-displacement amplitude from public
layers 1 and 5--8.  It never consumes target-layer temperature or salinity.
Every unsupported profile is a bit-exact no-op relative to the incumbent.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

PUBLIC_LAYERS = (1, 5, 6, 7, 8)
TARGET_LAYERS = (2, 3, 4)


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    actual = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if actual.shape != estimate.shape:
        raise ValueError("RMSE inputs are not aligned")
    if not np.isfinite(actual).all() or not np.isfinite(estimate).all():
        raise ValueError("RMSE inputs must be finite")
    return float(np.sqrt(np.mean((estimate - actual) ** 2)))


def circular_day_distance(day: np.ndarray, center: float) -> np.ndarray:
    distance = np.abs(np.asarray(day, dtype=np.float64) - float(center))
    return np.minimum(distance, 365.2425 - distance)


def season_bins(index: pd.DatetimeIndex, bin_days: int) -> np.ndarray:
    local = index.tz_convert("Asia/Seoul")
    return ((local.dayofyear.to_numpy() - 1) // int(bin_days)).astype(int)


def mask_validation_targets(
    observations: pd.DataFrame,
    folds: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, int]:
    masked = observations.copy()
    total = np.zeros(len(masked), dtype=bool)
    for specification in folds.values():
        start = pd.Timestamp(str(specification["start"])).tz_convert("UTC")
        stop = pd.Timestamp(str(specification["stop"])).tz_convert("UTC")
        total |= (
            masked["time"].ge(start) & masked["time"].lt(stop) & masked["layer"].isin(TARGET_LAYERS)
        ).to_numpy(bool)
    masked.loc[total, ["temp", "psal"]] = np.nan
    if not masked.loc[total, ["temp", "psal"]].isna().all().all():
        raise RuntimeError("target temperature/salinity joint mask failed")
    return masked, int(total.sum())


def build_public_panel(observations: pd.DataFrame) -> pd.DataFrame:
    public = observations.loc[
        observations["layer"].isin(PUBLIC_LAYERS),
        ["time", "layer", "temp", "depth"],
    ].copy()
    if public.duplicated(["time", "layer"]).any():
        raise ValueError("public observation keys duplicate")
    pieces: list[pd.DataFrame] = []
    for value in ("temp", "depth"):
        wide = public.pivot(index="time", columns="layer", values=value)
        wide = wide.reindex(columns=PUBLIC_LAYERS)
        wide.columns = [f"{value}_{int(layer)}" for layer in wide.columns]
        pieces.append(wide)
    panel = pd.concat(pieces, axis=1).sort_index()
    if not isinstance(panel.index, pd.DatetimeIndex) or panel.index.tz is None:
        raise ValueError("public panel requires timezone-aware timestamps")
    return panel


@dataclass(frozen=True)
class BackgroundProfile:
    season_bin: int
    depth: np.ndarray
    temperature: np.ndarray
    source_layers: tuple[int, ...]
    source_rows: dict[int, int]

    def interpolator(self) -> PchipInterpolator:
        return PchipInterpolator(self.depth, self.temperature, extrapolate=False)


def fit_seasonal_backgrounds(
    panel: pd.DataFrame,
    *,
    train_stop: pd.Timestamp,
    query_bins: set[int],
    purge_days: int,
    season_bin_days: int,
    season_window_days: float,
    minimum_rows_per_layer: int,
) -> tuple[dict[int, BackgroundProfile], dict[str, object]]:
    cutoff = pd.Timestamp(train_stop).tz_convert("UTC") - pd.Timedelta(days=int(purge_days))
    training = panel.loc[panel.index < cutoff]
    if training.empty:
        raise RuntimeError("public background prefix is empty")
    day = training.index.tz_convert("Asia/Seoul").dayofyear.to_numpy(np.float64)
    fitted: dict[int, BackgroundProfile] = {}
    rejected: dict[str, str] = {}
    for season_bin in sorted(query_bins):
        center = float(int(season_bin) * int(season_bin_days) + (int(season_bin_days) + 1) / 2)
        nearby = circular_day_distance(day, center) <= float(season_window_days)
        depths: list[float] = []
        temperatures: list[float] = []
        layers: list[int] = []
        counts: dict[int, int] = {}
        for layer in PUBLIC_LAYERS:
            temp = training[f"temp_{layer}"].to_numpy(np.float64)
            depth = training[f"depth_{layer}"].to_numpy(np.float64)
            valid = nearby & np.isfinite(temp) & np.isfinite(depth)
            counts[layer] = int(valid.sum())
            if counts[layer] < int(minimum_rows_per_layer):
                continue
            layers.append(layer)
            depths.append(float(np.mean(depth[valid])))
            temperatures.append(float(np.mean(temp[valid])))
        if len(layers) < 4:
            rejected[str(season_bin)] = "fewer_than_four_public_background_layers"
            continue
        order = np.argsort(np.asarray(depths), kind="stable")
        ordered_depth = np.asarray(depths, dtype=np.float64)[order]
        ordered_temp = np.asarray(temperatures, dtype=np.float64)[order]
        ordered_layers = tuple(
            int(np.asarray(layers)[order][position]) for position in range(len(order))
        )
        if np.any(np.diff(ordered_depth) <= 0.0):
            rejected[str(season_bin)] = "non_unique_background_depth"
            continue
        fitted[int(season_bin)] = BackgroundProfile(
            season_bin=int(season_bin),
            depth=ordered_depth,
            temperature=ordered_temp,
            source_layers=ordered_layers,
            source_rows=counts,
        )
    receipt = {
        "train_stop_exclusive_utc": cutoff.isoformat(),
        "training_public_timestamps": int(len(training)),
        "requested_bins": sorted(int(value) for value in query_bins),
        "fitted_bins": sorted(fitted),
        "rejected_bins": rejected,
        "target_layer_values_used": False,
    }
    return fitted, receipt


@dataclass(frozen=True)
class HeaveEstimate:
    supported: bool
    reason: str
    intercept_c: float
    eta_m: float
    public_mode_rms_c_per_m: float
    design_condition_number: float
    target_mode: np.ndarray


def estimate_public_heave(
    *,
    public_temperature: np.ndarray,
    public_depth: np.ndarray,
    background: BackgroundProfile,
    target_depth: np.ndarray,
    minimum_public_layers: int,
    minimum_public_span_m: float,
    minimum_gradient_rms_c_per_m: float,
    maximum_design_condition_number: float,
) -> HeaveEstimate:
    temperature = np.asarray(public_temperature, dtype=np.float64)
    depth = np.asarray(public_depth, dtype=np.float64)
    targets = np.asarray(target_depth, dtype=np.float64)
    valid = np.isfinite(temperature) & np.isfinite(depth)
    empty = np.full(targets.shape, np.nan, dtype=np.float64)
    if int(valid.sum()) < int(minimum_public_layers):
        return HeaveEstimate(
            False, "insufficient_public_layers", np.nan, np.nan, np.nan, np.inf, empty
        )
    if float(np.ptp(depth[valid])) < float(minimum_public_span_m):
        return HeaveEstimate(
            False, "insufficient_public_depth_span", np.nan, np.nan, np.nan, np.inf, empty
        )
    interpolator = background.interpolator()
    background_temp = np.asarray(interpolator(depth[valid]), dtype=np.float64)
    public_mode = -np.asarray(interpolator.derivative()(depth[valid]), dtype=np.float64)
    target_mode = -np.asarray(interpolator.derivative()(targets), dtype=np.float64)
    if not np.isfinite(background_temp).all() or not np.isfinite(public_mode).all():
        return HeaveEstimate(
            False, "public_background_extrapolation", np.nan, np.nan, np.nan, np.inf, empty
        )
    if not np.isfinite(target_mode).all():
        return HeaveEstimate(
            False, "target_background_extrapolation", np.nan, np.nan, np.nan, np.inf, empty
        )
    gradient_rms = float(np.sqrt(np.mean(public_mode**2)))
    if gradient_rms < float(minimum_gradient_rms_c_per_m):
        return HeaveEstimate(
            False, "weak_public_gradient", np.nan, np.nan, gradient_rms, np.inf, target_mode
        )
    design = np.column_stack((np.ones(int(valid.sum()), dtype=np.float64), public_mode))
    condition = float(np.linalg.cond(design))
    if not np.isfinite(condition) or condition > float(maximum_design_condition_number):
        return HeaveEstimate(
            False,
            "ill_conditioned_intercept_heave",
            np.nan,
            np.nan,
            gradient_rms,
            condition,
            target_mode,
        )
    residual = temperature[valid] - background_temp
    coefficient, *_ = np.linalg.lstsq(design, residual, rcond=None)
    intercept, eta = (float(coefficient[0]), float(coefficient[1]))
    if not np.isfinite(intercept) or not np.isfinite(eta):
        return HeaveEstimate(
            False, "non_finite_heave_fit", np.nan, np.nan, gradient_rms, condition, target_mode
        )
    return HeaveEstimate(True, "supported", intercept, eta, gradient_rms, condition, target_mode)


def estimate_training_eta_cap(
    panel: pd.DataFrame,
    backgrounds: dict[int, BackgroundProfile],
    *,
    train_stop: pd.Timestamp,
    purge_days: int,
    season_bin_days: int,
    target_depth: np.ndarray,
    stride: int,
    hard_cap_m: float,
    quantile: float,
    minimum_eta_rows: int,
    support: dict[str, float | int],
) -> tuple[float, dict[str, object]]:
    cutoff = pd.Timestamp(train_stop).tz_convert("UTC") - pd.Timedelta(days=int(purge_days))
    training = panel.loc[panel.index < cutoff].iloc[:: int(stride)]
    bins = season_bins(training.index, int(season_bin_days))
    eta: list[float] = []
    reasons: Counter[str] = Counter()
    for position, (_, row) in enumerate(training.iterrows()):
        background = backgrounds.get(int(bins[position]))
        if background is None:
            reasons["missing_background_bin"] += 1
            continue
        estimate = estimate_public_heave(
            public_temperature=np.asarray([row[f"temp_{layer}"] for layer in PUBLIC_LAYERS]),
            public_depth=np.asarray([row[f"depth_{layer}"] for layer in PUBLIC_LAYERS]),
            background=background,
            target_depth=target_depth,
            minimum_public_layers=int(support["minimum_public_layers"]),
            minimum_public_span_m=float(support["minimum_public_span_m"]),
            minimum_gradient_rms_c_per_m=float(support["minimum_gradient_rms_c_per_m"]),
            maximum_design_condition_number=float(support["maximum_design_condition_number"]),
        )
        reasons[estimate.reason] += 1
        if estimate.supported:
            eta.append(abs(estimate.eta_m))
    if len(eta) < int(minimum_eta_rows):
        raise RuntimeError(f"too few public-only eta rows: {len(eta)}")
    empirical = float(np.quantile(np.asarray(eta, dtype=np.float64), float(quantile)))
    cap = min(float(hard_cap_m), empirical)
    if not np.isfinite(cap) or cap <= 0.0:
        raise RuntimeError("invalid public-only eta cap")
    return cap, {
        "sample_stride": int(stride),
        "supported_eta_rows": int(len(eta)),
        "eta_absolute_quantile": float(quantile),
        "eta_empirical_quantile_m": empirical,
        "eta_hard_cap_m": float(hard_cap_m),
        "eta_effective_cap_m": cap,
        "support_reasons": dict(sorted(reasons.items())),
        "target_layer_values_used": False,
    }


def _preserves_order(
    reference: np.ndarray, candidate: np.ndarray, tolerance: float = 1e-12
) -> bool:
    old = np.diff(np.asarray(reference, dtype=np.float64))
    new = np.diff(np.asarray(candidate, dtype=np.float64))
    flat = np.abs(old) <= tolerance
    return bool(np.all(np.where(flat, np.abs(new) <= tolerance, old * new >= 0.0)))


def apply_heave_to_incumbent(
    incumbent: pd.DataFrame,
    panel: pd.DataFrame,
    backgrounds: dict[int, BackgroundProfile],
    *,
    season_bin_days: int,
    target_depth_by_layer: dict[int, float],
    eta_cap_m: float,
    maximum_correction_c: float,
    support: dict[str, float | int],
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"time", "layer", "reference", "block"}
    if not required.issubset(incumbent.columns):
        raise ValueError(f"incumbent columns missing: {sorted(required - set(incumbent.columns))}")
    work = incumbent.sort_values(["time", "layer"]).reset_index(drop=True).copy()
    reference = work["reference"].to_numpy(np.float64)
    candidate = reference.copy()
    correction = np.zeros(len(work), dtype=np.float64)
    enabled = np.zeros(len(work), dtype=bool)
    eta_values = np.full(len(work), np.nan, dtype=np.float64)
    reasons: Counter[str] = Counter()
    bins = season_bins(pd.DatetimeIndex(work["time"]), int(season_bin_days))
    work["_season_bin"] = bins
    for _, positions in work.groupby("time", sort=False).indices.items():
        positions = np.asarray(positions, dtype=int)
        group = work.iloc[positions]
        layers = group["layer"].to_numpy(int)
        if len(group) != len(TARGET_LAYERS) or set(layers) != set(TARGET_LAYERS):
            reasons["incomplete_target_profile"] += 1
            continue
        ordered = positions[np.argsort(layers, kind="stable")]
        timestamp = pd.Timestamp(group["time"].iloc[0])
        if timestamp not in panel.index:
            reasons["missing_public_timestamp"] += 1
            continue
        public = panel.loc[timestamp]
        endpoint = np.asarray(
            [public["temp_1"], public["depth_1"], public["temp_5"], public["depth_5"]],
            dtype=np.float64,
        )
        if not np.isfinite(endpoint).all():
            reasons["missing_l1_or_l5_endpoint"] += 1
            continue
        target_depth = np.asarray(
            [target_depth_by_layer[int(layer)] for layer in np.asarray(TARGET_LAYERS)],
            dtype=np.float64,
        )
        lower_depth, upper_depth = sorted((float(endpoint[1]), float(endpoint[3])))
        if np.any(target_depth < lower_depth) or np.any(target_depth > upper_depth):
            reasons["target_not_bracketed_by_l1_l5"] += 1
            continue
        background = backgrounds.get(int(group["_season_bin"].iloc[0]))
        if background is None:
            reasons["missing_background_bin"] += 1
            continue
        estimate = estimate_public_heave(
            public_temperature=np.asarray([public[f"temp_{layer}"] for layer in PUBLIC_LAYERS]),
            public_depth=np.asarray([public[f"depth_{layer}"] for layer in PUBLIC_LAYERS]),
            background=background,
            target_depth=target_depth,
            minimum_public_layers=int(support["minimum_public_layers"]),
            minimum_public_span_m=float(support["minimum_public_span_m"]),
            minimum_gradient_rms_c_per_m=float(support["minimum_gradient_rms_c_per_m"]),
            maximum_design_condition_number=float(support["maximum_design_condition_number"]),
        )
        if not estimate.supported:
            reasons[estimate.reason] += 1
            continue
        if abs(estimate.eta_m) > float(eta_cap_m):
            reasons["eta_q99_or_hard_cap"] += 1
            continue
        proposed = estimate.eta_m * estimate.target_mode
        if not np.isfinite(proposed).all() or float(np.max(np.abs(proposed))) > float(
            maximum_correction_c
        ):
            reasons["correction_bound"] += 1
            continue
        incumbent_profile = reference[ordered]
        candidate_profile = incumbent_profile + proposed
        low_temp, high_temp = sorted((float(endpoint[0]), float(endpoint[2])))
        if np.any(candidate_profile < low_temp) or np.any(candidate_profile > high_temp):
            reasons["public_endpoint_envelope"] += 1
            continue
        if not _preserves_order(incumbent_profile, candidate_profile):
            reasons["incumbent_order_change"] += 1
            continue
        candidate[ordered] = candidate_profile
        correction[ordered] = proposed
        eta_values[ordered] = estimate.eta_m
        enabled[ordered] = np.abs(proposed) > 0.0
        reasons["enabled_profile"] += 1
    disabled = ~enabled
    if not np.array_equal(candidate[disabled], reference[disabled]):
        raise RuntimeError("unsupported rows differ from incumbent")
    work = work.drop(columns=["_season_bin"])
    work["candidate"] = candidate
    work["correction"] = correction
    work["enabled"] = enabled
    work["eta_m"] = eta_values
    diagnostics = {
        "rows": int(len(work)),
        "enabled_rows": int(enabled.sum()),
        "enabled_fraction": float(enabled.mean()),
        "enabled_profiles": int(reasons["enabled_profile"]),
        "reason_counts_by_profile": dict(sorted(reasons.items())),
        "correction_rms_c": float(np.sqrt(np.mean(correction**2))),
        "correction_p99_absolute_c": float(np.quantile(np.abs(correction), 0.99)),
        "correction_maximum_absolute_c": float(np.max(np.abs(correction))),
        "fallback_maximum_absolute_c": float(np.max(np.abs(correction[disabled])))
        if disabled.any()
        else 0.0,
    }
    return work, diagnostics


def paired_kst_day_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    work = frame.loc[:, ["time", "truth", "reference", "candidate"]].copy()
    work["time"] = pd.to_datetime(work["time"], utc=True)
    work["day"] = work["time"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    work["reference_se"] = (work["reference"] - work["truth"]) ** 2
    work["candidate_se"] = (work["candidate"] - work["truth"]) ** 2
    daily = work.groupby("day", sort=True).agg(
        rows=("truth", "size"),
        reference_se=("reference_se", "sum"),
        candidate_se=("candidate_se", "sum"),
    )
    values = daily.to_numpy(np.float64)
    if len(values) < 10:
        raise ValueError("too few KST days")
    rng = np.random.default_rng(int(seed))
    delta = np.empty(int(replicates), dtype=np.float64)
    for draw in range(int(replicates)):
        sampled = values[rng.integers(0, len(values), size=len(values))]
        rows = float(sampled[:, 0].sum())
        delta[draw] = np.sqrt(sampled[:, 2].sum() / rows) - np.sqrt(sampled[:, 1].sum() / rows)
    return {
        "unit": "KST calendar day",
        "days": int(len(values)),
        "replicates": int(replicates),
        "seed": int(seed),
        "mean_delta_rmse_c": float(delta.mean()),
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
    active_fraction: float,
    correction_rms: float,
    correction_p99: float,
    correction_maximum: float,
    thresholds: dict[str, float | int],
) -> dict[str, object]:
    checks = {
        "aggregate_delta_rmse": aggregate_delta <= float(thresholds["aggregate_delta_rmse_max_c"]),
        "paired_ci90_upper": ci90_high
        < float(thresholds["paired_kst_day_bootstrap_ci90_upper_max_c"]),
        "improved_folds": sum(value < 0.0 for value in fold_deltas.values())
        >= int(thresholds["minimum_improved_folds"]),
        "worst_fold_regression": max(fold_deltas.values())
        <= float(thresholds["maximum_worst_fold_regression_c"]),
        "maximum_layer_regression": max(layer_deltas.values())
        <= float(thresholds["maximum_layer_regression_c"]),
        "minimum_active_fraction": active_fraction >= float(thresholds["minimum_active_fraction"]),
        "correction_rms": correction_rms <= float(thresholds["maximum_correction_rms_c"]),
        "correction_p99": correction_p99 <= float(thresholds["maximum_correction_p99_c"]),
        "correction_maximum": correction_maximum
        <= float(thresholds["maximum_correction_absolute_c"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "observed": {
            "aggregate_delta_rmse_c": float(aggregate_delta),
            "bootstrap_ci90_high_c": float(ci90_high),
            "improved_folds": int(sum(value < 0.0 for value in fold_deltas.values())),
            "worst_fold_regression_c": float(max(fold_deltas.values())),
            "maximum_layer_regression_c": float(max(layer_deltas.values())),
            "active_fraction": float(active_fraction),
            "correction_rms_c": float(correction_rms),
            "correction_p99_c": float(correction_p99),
            "correction_maximum_c": float(correction_maximum),
        },
        "thresholds": thresholds,
    }


__all__ = [
    "BackgroundProfile",
    "HeaveEstimate",
    "PUBLIC_LAYERS",
    "TARGET_LAYERS",
    "apply_heave_to_incumbent",
    "build_public_panel",
    "estimate_public_heave",
    "estimate_training_eta_cap",
    "evaluate_gate",
    "fit_seasonal_backgrounds",
    "mask_validation_targets",
    "paired_kst_day_bootstrap",
    "rmse",
    "season_bins",
]
