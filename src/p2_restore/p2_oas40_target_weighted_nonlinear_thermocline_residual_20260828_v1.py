"""Target-weighted nonlinear thermocline residual utilities for P2.

This module is deliberately historical-validation only.  It creates no
deployment frame and has no official test, sample, submission, or upload path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from p2_restore.dynamic_sigmoid_profile import LatentRidge


@dataclass(frozen=True)
class ShiftWeightResult:
    source_weights: np.ndarray
    query_density_ratio: np.ndarray
    cross_day_auc: float
    effective_sample_fraction: float
    overlap_passed: bool
    columns: tuple[str, ...]
    source_rows: int
    query_rows: int

    def receipt(self) -> dict[str, object]:
        return {
            "source_rows": self.source_rows,
            "query_rows": self.query_rows,
            "columns": list(self.columns),
            "cross_day_auc": self.cross_day_auc,
            "effective_sample_fraction": self.effective_sample_fraction,
            "overlap_passed": self.overlap_passed,
            "source_weight_min": float(np.min(self.source_weights)),
            "source_weight_median": float(np.median(self.source_weights)),
            "source_weight_max": float(np.max(self.source_weights)),
            "query_density_ratio_min": float(np.min(self.query_density_ratio)),
            "query_density_ratio_median": float(np.median(self.query_density_ratio)),
            "query_density_ratio_max": float(np.max(self.query_density_ratio)),
        }


def _prepare_shift_matrix(
    source: pd.DataFrame,
    query: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = tuple(columns)
    source_values = source.loc[:, selected].to_numpy(np.float64)
    query_values = query.loc[:, selected].to_numpy(np.float64)
    center = np.nanmedian(source_values, axis=0)
    center[~np.isfinite(center)] = 0.0
    source_values = np.where(np.isfinite(source_values), source_values, center)
    query_values = np.where(np.isfinite(query_values), query_values, center)
    scale = np.nanmedian(np.abs(source_values - center), axis=0) * 1.4826
    fallback = np.nanstd(source_values, axis=0)
    bad = ~np.isfinite(scale) | (scale < 1e-6)
    scale[bad] = fallback[bad]
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return (
        np.clip((source_values - center) / scale, -8.0, 8.0),
        np.clip((query_values - center) / scale, -8.0, 8.0),
        center,
        scale,
    )


def _cross_day_auc(
    source_matrix: np.ndarray,
    query_matrix: np.ndarray,
    source_time: pd.DatetimeIndex,
    query_time: pd.DatetimeIndex,
    *,
    logistic_c: float,
    max_iter: int,
    seed: int,
) -> float:
    matrix = np.vstack((source_matrix, query_matrix))
    label = np.concatenate(
        (np.zeros(len(source_matrix), dtype=np.int8), np.ones(len(query_matrix), dtype=np.int8))
    )
    source_day = source_time.tz_convert("Asia/Seoul").normalize().asi8 // (86_400 * 10**9)
    query_day = query_time.tz_convert("Asia/Seoul").normalize().asi8 // (86_400 * 10**9)
    parity = np.concatenate((source_day % 2, query_day % 2)).astype(int)
    probability = np.full(len(matrix), np.nan, dtype=np.float64)
    for holdout in (0, 1):
        train = parity != holdout
        test = parity == holdout
        if not test.any() or len(np.unique(label[train])) != 2 or len(np.unique(label[test])) != 2:
            continue
        model = LogisticRegression(
            C=float(logistic_c),
            class_weight="balanced",
            max_iter=int(max_iter),
            random_state=int(seed) + holdout,
            solver="lbfgs",
        )
        model.fit(matrix[train], label[train])
        probability[test] = model.predict_proba(matrix[test])[:, 1]
    valid = np.isfinite(probability)
    if valid.sum() == 0 or len(np.unique(label[valid])) != 2:
        return 1.0
    return float(roc_auc_score(label[valid], probability[valid]))


def fit_covariate_shift_weights(
    source: pd.DataFrame,
    query: pd.DataFrame,
    *,
    source_time: pd.DatetimeIndex,
    query_time: pd.DatetimeIndex,
    columns: Sequence[str],
    logistic_c: float,
    max_iter: int,
    seed: int,
    weight_clip: tuple[float, float],
    minimum_effective_sample_fraction: float,
    maximum_auc: float,
) -> ShiftWeightResult:
    """Fit a fixed public-X density-ratio model with an overlap fail-safe."""

    if len(source) != len(source_time) or len(query) != len(query_time):
        raise ValueError("shift feature/time lengths differ")
    if len(source) < 200 or len(query) < 200:
        raise ValueError("too few rows for shift weighting")
    selected = tuple(columns)
    source_matrix, query_matrix, _, _ = _prepare_shift_matrix(source, query, selected)
    auc = _cross_day_auc(
        source_matrix,
        query_matrix,
        source_time,
        query_time,
        logistic_c=logistic_c,
        max_iter=max_iter,
        seed=seed,
    )
    matrix = np.vstack((source_matrix, query_matrix))
    label = np.concatenate(
        (np.zeros(len(source_matrix), dtype=np.int8), np.ones(len(query_matrix), dtype=np.int8))
    )
    model = LogisticRegression(
        C=float(logistic_c),
        class_weight="balanced",
        max_iter=int(max_iter),
        random_state=int(seed),
        solver="lbfgs",
    )
    model.fit(matrix, label)
    probability = np.clip(model.predict_proba(matrix)[:, 1], 1e-6, 1.0 - 1e-6)
    density_ratio = probability / (1.0 - probability)
    lower, upper = map(float, weight_clip)
    source_weight = np.clip(density_ratio[: len(source)], lower, upper)
    query_ratio = density_ratio[len(source) :]
    ess = float(np.square(source_weight.sum()) / np.square(source_weight).sum())
    ess_fraction = ess / len(source_weight)
    overlap = bool(auc <= float(maximum_auc) and ess_fraction >= float(minimum_effective_sample_fraction))
    return ShiftWeightResult(
        source_weights=source_weight,
        query_density_ratio=query_ratio,
        cross_day_auc=auc,
        effective_sample_fraction=ess_fraction,
        overlap_passed=overlap,
        columns=selected,
        source_rows=len(source),
        query_rows=len(query),
    )


def latent_standardized_distance(model: LatentRidge, features: pd.DataFrame) -> np.ndarray:
    values = features.loc[:, model.feature_columns].to_numpy(np.float64)
    values = np.where(np.isfinite(values), values, model.medians)
    standardized = (values - model.means) / model.scales
    return np.sqrt(np.mean(np.square(np.clip(standardized, -20.0, 20.0)), axis=1))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    current = np.asarray(values, dtype=np.float64)
    importance = np.asarray(weights, dtype=np.float64)
    if current.shape != importance.shape or len(current) == 0:
        raise ValueError("weighted quantile inputs differ")
    if not np.isfinite(current).all() or not np.isfinite(importance).all() or np.any(importance <= 0):
        raise ValueError("weighted quantile inputs are invalid")
    order = np.argsort(current, kind="stable")
    cumulative = np.cumsum(importance[order])
    threshold = float(quantile) * cumulative[-1]
    position = min(int(np.searchsorted(cumulative, threshold, side="left")), len(order) - 1)
    return float(current[order[position]])


def bounded_fixed_correction(
    raw_correction: np.ndarray,
    enabled: np.ndarray,
    *,
    rms_cap: float,
    p99_cap: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    raw = np.asarray(raw_correction, dtype=np.float64)
    gate = np.asarray(enabled, dtype=bool)
    if raw.shape != gate.shape or not np.isfinite(raw).all():
        raise ValueError("correction inputs differ")
    correction = np.zeros_like(raw)
    correction[gate] = np.clip(raw[gate], -float(p99_cap), float(p99_cap))
    current_rms = float(np.sqrt(np.mean(np.square(correction))))
    scale = min(1.0, float(rms_cap) / current_rms) if current_rms > 0.0 else 1.0
    correction *= scale
    p99 = float(np.quantile(np.abs(correction), 0.99)) if len(correction) else 0.0
    fallback = float(np.max(np.abs(correction[~gate]))) if (~gate).any() else 0.0
    if fallback != 0.0 or p99 > float(p99_cap) + 1e-12:
        raise RuntimeError("correction safety contract failed")
    return correction, {
        "rows": int(len(correction)),
        "enabled_rows": int(gate.sum()),
        "enabled_fraction": float(gate.mean()) if len(gate) else 0.0,
        "scale_to_rms_cap": scale,
        "rms_c": float(np.sqrt(np.mean(np.square(correction)))),
        "p99_absolute_c": p99,
        "maximum_absolute_c": float(np.max(np.abs(correction))) if len(correction) else 0.0,
        "fallback_maximum_absolute_c": fallback,
    }


def vector_cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("cosine vectors differ")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-15 else 0.0


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("RMSE inputs are invalid")
    return float(np.sqrt(np.mean(np.square(y - p))))


def paired_kst_day_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    work = frame.loc[:, ["time", "truth", "reference", "candidate"]].copy()
    work["time"] = pd.to_datetime(work["time"], utc=True)
    work["day"] = work["time"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    work["se_reference"] = np.square(work["reference"] - work["truth"])
    work["se_candidate"] = np.square(work["candidate"] - work["truth"])
    daily = work.groupby("day", sort=True).agg(
        rows=("truth", "size"),
        se_reference=("se_reference", "sum"),
        se_candidate=("se_candidate", "sum"),
    )
    if len(daily) < 10:
        raise ValueError("too few KST days")
    values = daily.to_numpy(np.float64)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(replicates), dtype=np.float64)
    for draw in range(int(replicates)):
        sampled = values[rng.integers(0, len(values), size=len(values))]
        rows = sampled[:, 0].sum()
        draws[draw] = np.sqrt(sampled[:, 2].sum() / rows) - np.sqrt(sampled[:, 1].sum() / rows)
    return {
        "unit": "KST calendar day",
        "days": int(len(daily)),
        "replicates": int(replicates),
        "seed": int(seed),
        "mean_delta_rmse_c": float(draws.mean()),
        "ci90_low_c": float(np.quantile(draws, 0.05)),
        "ci90_high_c": float(np.quantile(draws, 0.95)),
        "probability_improved": float(np.mean(draws < 0.0)),
    }


def evaluate_gate(
    *,
    aggregate_delta: float,
    ci90_high: float,
    fold_deltas: dict[str, float],
    layer_deltas: dict[str, float],
    active_share: float,
    correction_rms: float,
    correction_p99: float,
    cosine: float,
    all_shift_folds_passed: bool,
    thresholds: dict[str, float | int],
) -> dict[str, object]:
    checks = {
        "all_shift_folds_passed": bool(all_shift_folds_passed),
        "adapted_pooled_delta_rmse": aggregate_delta <= float(thresholds["adapted_pooled_delta_rmse_max_c"]),
        "paired_ci90_upper": ci90_high <= float(thresholds["paired_kst_day_bootstrap_ci90_upper_max_c"]),
        "improved_folds": sum(value < 0.0 for value in fold_deltas.values()) >= int(thresholds["minimum_improved_folds"]),
        "unweighted_pooled_regression": aggregate_delta <= float(thresholds["maximum_unweighted_pooled_regression_c"]),
        "worst_layer_regression": max(layer_deltas.values()) <= float(thresholds["maximum_worst_layer_regression_c"]),
        "active_share_minimum": active_share >= float(thresholds["minimum_active_share"]),
        "active_share_maximum": active_share <= float(thresholds["maximum_active_share"]),
        "correction_rms_minimum": correction_rms >= float(thresholds["minimum_correction_rms_c"]),
        "correction_rms_maximum": correction_rms <= float(thresholds["maximum_correction_rms_c"]) + 1e-12,
        "correction_p99": correction_p99 <= float(thresholds["maximum_correction_p99_c"]) + 1e-12,
        "alpha_direction_cosine": abs(cosine) <= float(thresholds["maximum_absolute_cosine_with_alpha20_to_alpha40"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "observed": {
            "aggregate_delta_rmse_c": aggregate_delta,
            "bootstrap_ci90_high_c": ci90_high,
            "improved_folds": int(sum(value < 0.0 for value in fold_deltas.values())),
            "worst_layer_regression_c": float(max(layer_deltas.values())),
            "active_share": active_share,
            "correction_rms_c": correction_rms,
            "correction_p99_c": correction_p99,
            "absolute_cosine_with_alpha20_to_alpha40": abs(cosine),
        },
        "thresholds": thresholds,
    }


__all__ = [
    "ShiftWeightResult",
    "bounded_fixed_correction",
    "evaluate_gate",
    "fit_covariate_shift_weights",
    "latent_standardized_distance",
    "paired_kst_day_bootstrap",
    "rmse",
    "vector_cosine",
    "weighted_quantile",
]
