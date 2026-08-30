"""Fixed Bayesian random-feature residual head for the P3 exposed research surface.

The candidate is a sparse-GP analogue: one analytic multi-output Bayesian linear
fit in frozen random Fourier features per historical fold.  It never deletes a
row.  At inference it returns the paired incumbent exactly whenever the
posterior interval for the proposed correction includes zero or the case is
outside the train-fold feature footprint.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from p3_wave.validation import rmse

LEADS = (3, 6, 9, 12, 18, 24)
STATIONS = ("G-ORS", "I-ORS", "S-ORS")


@dataclass(frozen=True)
class FeatureTransform:
    median: np.ndarray
    scale: np.ndarray
    all_missing_columns: int
    radius_limit: float
    missing_fraction_limit: float
    clip_abs: float


@dataclass(frozen=True)
class BayesianRFFModel:
    transform: FeatureTransform
    random_weights: np.ndarray
    random_phase: np.ndarray
    coefficients: np.ndarray
    coefficient_covariance_base: np.ndarray
    residual_variance: np.ndarray
    feature_columns: tuple[str, ...]
    seed: int
    fit_receipt: dict[str, Any]


@dataclass(frozen=True)
class FoldPrediction:
    frame: pd.DataFrame
    receipt: dict[str, Any]


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _aligned_numeric_features(
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    required_features = {"anchor_id", "station", *feature_columns}
    required_anchors = {"anchor_id", "station"}
    if missing := required_features.difference(features.columns):
        raise ValueError(f"feature columns missing: {sorted(missing)}")
    if missing := required_anchors.difference(anchors.columns):
        raise ValueError(f"anchor columns missing: {sorted(missing)}")
    if features["anchor_id"].duplicated().any() or anchors["anchor_id"].duplicated().any():
        raise ValueError("anchor_id must be unique")
    left = anchors[["anchor_id", "station"]].copy()
    right = features[["anchor_id", "station", *feature_columns]].rename(
        columns={"station": "__feature_station"}
    )
    aligned = left.merge(right, on="anchor_id", how="left", validate="one_to_one")
    if len(aligned) != len(anchors) or aligned["__feature_station"].isna().any():
        raise ValueError("past-only feature alignment failed")
    if not aligned["station"].astype(str).equals(aligned["__feature_station"].astype(str)):
        raise ValueError("feature station differs from anchor station")
    numeric = aligned[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    stations = aligned["station"].astype(str).to_numpy()
    if not set(stations).issubset(STATIONS):
        raise ValueError("unknown station in candidate design")
    return numeric.to_numpy(dtype=np.float64), stations


def _fit_feature_transform(
    raw: np.ndarray,
    *,
    clip_abs: float,
    ood_quantile: float,
    minimum_radius: float,
) -> FeatureTransform:
    finite = np.isfinite(raw)
    count = finite.sum(axis=0)
    median = np.zeros(raw.shape[1], dtype=np.float64)
    scale = np.ones(raw.shape[1], dtype=np.float64)
    for column in range(raw.shape[1]):
        values = raw[finite[:, column], column]
        if len(values):
            median[column] = float(np.median(values))
            q25, q75 = np.quantile(values, [0.25, 0.75])
            width = float(q75 - q25)
            scale[column] = width if np.isfinite(width) and width > 1.0e-8 else 1.0
    imputed = np.where(finite, raw, median[None, :])
    standardized = np.clip(
        (imputed - median[None, :]) / scale[None, :],
        -float(clip_abs),
        float(clip_abs),
    )
    radius = np.sqrt(np.mean(np.square(standardized), axis=1))
    missing_fraction = 1.0 - count.sum() / max(1, raw.shape[0] * raw.shape[1])
    row_missing = 1.0 - finite.mean(axis=1)
    return FeatureTransform(
        median=median,
        scale=scale,
        all_missing_columns=int(np.sum(count == 0)),
        radius_limit=max(float(minimum_radius), float(np.quantile(radius, ood_quantile))),
        missing_fraction_limit=max(float(missing_fraction), float(np.quantile(row_missing, ood_quantile))),
        clip_abs=float(clip_abs),
    )


def _transform_features(
    raw: np.ndarray,
    stations: np.ndarray,
    transform: FeatureTransform,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    finite = np.isfinite(raw)
    imputed = np.where(finite, raw, transform.median[None, :])
    standardized = np.clip(
        (imputed - transform.median[None, :]) / transform.scale[None, :],
        -transform.clip_abs,
        transform.clip_abs,
    )
    station_one_hot = np.column_stack([stations == station for station in STATIONS]).astype(
        np.float64
    )
    design = np.column_stack([standardized, station_one_hot])
    radius = np.sqrt(np.mean(np.square(standardized), axis=1))
    missing_fraction = 1.0 - finite.mean(axis=1)
    ood = (radius > transform.radius_limit) | (
        missing_fraction > transform.missing_fraction_limit + 1.0e-12
    )
    receipt = {
        "rows": int(len(raw)),
        "numeric_feature_count": int(raw.shape[1]),
        "design_feature_count": int(design.shape[1]),
        "all_missing_training_columns": int(transform.all_missing_columns),
        "radius_limit": float(transform.radius_limit),
        "maximum_radius": float(np.max(radius)) if len(radius) else None,
        "missing_fraction_limit": float(transform.missing_fraction_limit),
        "maximum_missing_fraction": float(np.max(missing_fraction)) if len(raw) else None,
        "ood_rows": int(ood.sum()),
    }
    if not np.isfinite(design).all():
        raise ValueError("candidate design contains non-finite values")
    return design, ood, receipt


def _rff(design: np.ndarray, weights: np.ndarray, phase: np.ndarray) -> np.ndarray:
    projected = design @ weights + phase[None, :]
    return np.sqrt(2.0 / weights.shape[1]) * np.cos(projected)


def fit_bayesian_rff_multi_lead(
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    seed: int,
    random_feature_count: int,
    ridge_precision: float,
    clip_abs: float,
    ood_quantile: float,
    minimum_radius: float,
) -> BayesianRFFModel:
    """Perform exactly one analytic multi-output fit."""

    started = time.perf_counter()
    raw, stations = _aligned_numeric_features(features, anchors, feature_columns)
    transform = _fit_feature_transform(
        raw,
        clip_abs=clip_abs,
        ood_quantile=ood_quantile,
        minimum_radius=minimum_radius,
    )
    design, ood, design_receipt = _transform_features(raw, stations, transform)
    targets = np.column_stack(
        [
            anchors[f"target_{lead}"].to_numpy(dtype=np.float64)
            - anchors["current_hs"].to_numpy(dtype=np.float64)
            for lead in LEADS
        ]
    )
    if not np.isfinite(targets).all():
        raise ValueError("candidate fit target contains non-finite values")
    rng = np.random.default_rng(int(seed))
    weights = rng.normal(
        0.0,
        1.0 / np.sqrt(design.shape[1]),
        size=(design.shape[1], int(random_feature_count)),
    )
    phase = rng.uniform(0.0, 2.0 * np.pi, size=int(random_feature_count))
    phi = np.column_stack([_rff(design, weights, phase), np.ones(len(design))])
    penalty = np.eye(phi.shape[1], dtype=np.float64)
    penalty[-1, -1] = 1.0e-6
    precision = phi.T @ phi + float(ridge_precision) * penalty
    coefficients = np.linalg.solve(precision, phi.T @ targets)
    covariance = np.linalg.inv(precision)
    residual = targets - phi @ coefficients
    residual_variance = np.maximum(np.mean(np.square(residual), axis=0), 1.0e-8)
    condition_number = float(np.linalg.cond(precision))
    receipt = {
        "fit_count": 1,
        "hyperparameter_search_count": 0,
        "train_cases": int(len(anchors)),
        "output_leads": list(LEADS),
        "random_feature_count": int(random_feature_count),
        "ridge_precision": float(ridge_precision),
        "seed": int(seed),
        "precision_condition_number": condition_number,
        "finite_solution": bool(
            np.isfinite(coefficients).all()
            and np.isfinite(covariance).all()
            and np.isfinite(residual_variance).all()
        ),
        "feature_transform_sha256": array_sha256(transform.median, transform.scale),
        "random_basis_sha256": array_sha256(weights, phase),
        "coefficient_sha256": array_sha256(coefficients, covariance, residual_variance),
        "residual_rmse_by_lead_m": {
            str(lead): float(np.sqrt(residual_variance[index]))
            for index, lead in enumerate(LEADS)
        },
        "design": design_receipt,
        "training_rows_outside_frozen_ood_quantile": int(ood.sum()),
        "elapsed_seconds": float(time.perf_counter() - started),
        "rows_deleted": 0,
    }
    if not receipt["finite_solution"]:
        raise ValueError("Bayesian random-feature fit is non-finite")
    return BayesianRFFModel(
        transform=transform,
        random_weights=weights,
        random_phase=phase,
        coefficients=coefficients,
        coefficient_covariance_base=covariance,
        residual_variance=residual_variance,
        feature_columns=tuple(feature_columns),
        seed=int(seed),
        fit_receipt=receipt,
    )


def predict_with_abstention(
    model: BayesianRFFModel,
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    incumbent: pd.DataFrame,
    *,
    interval_z: float,
    correction_cap_m: float,
) -> FoldPrediction:
    raw, stations = _aligned_numeric_features(features, anchors, model.feature_columns)
    design, ood, design_receipt = _transform_features(raw, stations, model.transform)
    phi = np.column_stack(
        [_rff(design, model.random_weights, model.random_phase), np.ones(len(design))]
    )
    mean_change = phi @ model.coefficients
    leverage = np.einsum(
        "ij,jk,ik->i", phi, model.coefficient_covariance_base, phi, optimize=True
    )
    mean_sd = np.sqrt(
        np.maximum(leverage[:, None], 0.0) * model.residual_variance[None, :]
    )
    expected_keys = pd.MultiIndex.from_product(
        [anchors["anchor_id"].to_numpy(dtype=np.int64), LEADS],
        names=["anchor_id", "lead_h"],
    )
    incumbent_series = (
        incumbent.set_index(["anchor_id", "lead_h"])["incumbent_prediction"]
        .reindex(expected_keys)
    )
    if incumbent_series.isna().any():
        raise ValueError("paired incumbent does not cover validation keys")
    incumbent_matrix = incumbent_series.to_numpy(dtype=np.float64).reshape(len(anchors), len(LEADS))
    proposed = anchors["current_hs"].to_numpy(dtype=np.float64)[:, None] + mean_change
    correction_mean = proposed - incumbent_matrix
    interval_excludes_zero = np.abs(correction_mean) > float(interval_z) * mean_sd
    active = interval_excludes_zero & ~ood[:, None]
    applied = np.where(
        active,
        np.clip(correction_mean, -float(correction_cap_m), float(correction_cap_m)),
        0.0,
    )
    candidate = np.clip(incumbent_matrix + applied, 0.0, 30.0)
    target = np.column_stack(
        [anchors[f"target_{lead}"].to_numpy(dtype=np.float64) for lead in LEADS]
    )
    rows = len(anchors) * len(LEADS)
    frame = pd.DataFrame(
        {
            "fold": np.repeat(anchors["fold"].astype(str).to_numpy(), len(LEADS)),
            "anchor_id": np.repeat(
                anchors["anchor_id"].to_numpy(dtype=np.int64), len(LEADS)
            ),
            "anchor_time": np.repeat(
                pd.to_datetime(anchors["anchor_time"], utc=True).to_numpy(), len(LEADS)
            ),
            "station": np.repeat(anchors["station"].astype(str).to_numpy(), len(LEADS)),
            "lead_h": np.tile(np.asarray(LEADS, dtype=int), len(anchors)),
            "target_hs": target.reshape(rows),
            "candidate_prediction": candidate.reshape(rows),
            "correction_mean_m": correction_mean.reshape(rows),
            "posterior_mean_sd_m": mean_sd.reshape(rows),
            "correction_applied_m": applied.reshape(rows),
            "active_correction": active.reshape(rows),
            "ood_case": np.repeat(ood, len(LEADS)),
        }
    )
    finite = np.isfinite(
        frame[
            [
                "target_hs",
                "candidate_prediction",
                "correction_mean_m",
                "posterior_mean_sd_m",
                "correction_applied_m",
            ]
        ].to_numpy(dtype=np.float64)
    ).all()
    if not finite or frame.duplicated(["fold", "anchor_id", "lead_h"]).any():
        raise ValueError("candidate prediction contract failed")
    receipt = {
        "validation_cases": int(len(anchors)),
        "validation_rows": int(rows),
        "active_correction_rows": int(active.sum()),
        "active_correction_cases": int(np.any(active, axis=1).sum()),
        "abstained_interval_rows": int((~interval_excludes_zero).sum()),
        "abstained_ood_rows": int(np.repeat(ood, len(LEADS)).sum()),
        "maximum_absolute_applied_correction_m": float(np.max(np.abs(applied))),
        "mean_absolute_applied_correction_m": float(np.mean(np.abs(applied))),
        "exact_incumbent_rows": int(np.sum(applied == 0.0)),
        "finite_predictions": bool(finite),
        "design": design_receipt,
        "rows_deleted": 0,
    }
    return FoldPrediction(frame=frame, receipt=receipt)


def fit_predict_fold(
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    train_anchors: pd.DataFrame,
    validation_anchors: pd.DataFrame,
    incumbent: pd.DataFrame,
    *,
    recipe: dict[str, Any],
    seed: int,
) -> FoldPrediction:
    model = fit_bayesian_rff_multi_lead(
        train_features,
        train_anchors,
        feature_columns=tuple(recipe["feature_columns"]),
        seed=int(seed),
        random_feature_count=int(recipe["random_feature_count"]),
        ridge_precision=float(recipe["ridge_precision"]),
        clip_abs=float(recipe["robust_feature_clip_abs"]),
        ood_quantile=float(recipe["ood_train_quantile"]),
        minimum_radius=float(recipe["ood_minimum_radius"]),
    )
    prediction = predict_with_abstention(
        model,
        validation_features,
        validation_anchors,
        incumbent,
        interval_z=float(recipe["posterior_mean_z"]),
        correction_cap_m=float(recipe["maximum_absolute_correction_m"]),
    )
    receipt = {
        "fold": str(validation_anchors["fold"].iloc[0]),
        "fit": model.fit_receipt,
        "prediction": prediction.receipt,
    }
    return FoldPrediction(frame=prediction.frame, receipt=receipt)


def comparison_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "target_hs",
        "candidate_prediction",
        "incumbent_prediction",
        "persistence",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"comparison columns missing: {sorted(missing)}")
    if frame.duplicated(["anchor_id", "lead_h"]).any():
        raise ValueError("comparison keys duplicated")

    def summarize(part: pd.DataFrame) -> dict[str, Any]:
        truth = part["target_hs"].to_numpy(dtype=float)
        candidate = part["candidate_prediction"].to_numpy(dtype=float)
        incumbent = part["incumbent_prediction"].to_numpy(dtype=float)
        persistence = part["persistence"].to_numpy(dtype=float)
        candidate_rmse = rmse(truth, candidate)
        incumbent_rmse = rmse(truth, incumbent)
        return {
            "cases": int(part["anchor_id"].nunique()),
            "rows": int(len(part)),
            "candidate_rmse_m": candidate_rmse,
            "paired_incumbent_rmse_m": incumbent_rmse,
            "persistence_rmse_m": rmse(truth, persistence),
            "benefit_incumbent_minus_candidate_rmse_m": incumbent_rmse - candidate_rmse,
            "delta_candidate_minus_incumbent_rmse_m": candidate_rmse - incumbent_rmse,
        }

    return {
        "overall": summarize(frame),
        "by_window": {
            str(key): summarize(group)
            for key, group in frame.groupby("fold", sort=True, observed=True)
        },
        "by_station": {
            str(key): summarize(group)
            for key, group in frame.groupby("station", sort=True, observed=True)
        },
        "by_lead": {
            str(int(key)): summarize(group)
            for key, group in frame.groupby("lead_h", sort=True, observed=True)
        },
    }


def contiguous_anchor_day_block_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    block_length_days: int,
) -> dict[str, Any]:
    required = {
        "fold",
        "anchor_id",
        "anchor_time",
        "lead_h",
        "target_hs",
        "candidate_prediction",
        "incumbent_prediction",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"bootstrap columns missing: {sorted(missing)}")
    ordered = frame.sort_values(["anchor_id", "lead_h"]).reset_index(drop=True)
    lead_inventory = ordered.groupby("anchor_id", sort=False)["lead_h"].agg(tuple)
    if not lead_inventory.map(lambda row: tuple(map(int, row)) == LEADS).all():
        raise ValueError("each bootstrap case must preserve all six ordered leads")
    cases = len(lead_inventory)
    truth = ordered["target_hs"].to_numpy(dtype=float).reshape(cases, len(LEADS))
    candidate = ordered["candidate_prediction"].to_numpy(dtype=float).reshape(cases, len(LEADS))
    incumbent = ordered["incumbent_prediction"].to_numpy(dtype=float).reshape(cases, len(LEADS))
    case_meta = ordered.drop_duplicates("anchor_id", keep="first").reset_index(drop=True)
    case_meta["anchor_day"] = pd.to_datetime(
        case_meta["anchor_time"], utc=True
    ).dt.floor("D")
    strata: dict[str, list[np.ndarray]] = {}
    unique_days: dict[str, int] = {}
    for fold, part in case_meta.groupby("fold", sort=True, observed=True):
        days = sorted(part["anchor_day"].unique())
        groups = [
            part.index[part["anchor_day"].eq(day)].to_numpy(dtype=int) for day in days
        ]
        if not groups:
            raise ValueError("bootstrap window has no anchor days")
        strata[str(fold)] = groups
        unique_days[str(fold)] = len(groups)
    rng = np.random.default_rng(int(seed))
    benefit = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        selected_cases: list[np.ndarray] = []
        for groups in strata.values():
            selected_days: list[int] = []
            while len(selected_days) < len(groups):
                start = int(rng.integers(0, len(groups)))
                selected_days.extend(
                    (start + offset) % len(groups)
                    for offset in range(int(block_length_days))
                )
            for day_index in selected_days[: len(groups)]:
                selected_cases.append(groups[day_index])
        selected = np.concatenate(selected_cases)
        benefit[replicate] = rmse(truth[selected], incumbent[selected]) - rmse(
            truth[selected], candidate[selected]
        )
    point = rmse(truth, incumbent) - rmse(truth, candidate)
    low, high = np.quantile(benefit, [0.05, 0.95])
    return {
        "unit": "forward_window_stratified_contiguous_anchor_day_block_with_six_leads_intact",
        "episode_id_available": False,
        "cases": int(cases),
        "rows": int(len(frame)),
        "unique_anchor_days_by_window": unique_days,
        "block_length_anchor_days": int(block_length_days),
        "replicates": int(replicates),
        "seed": int(seed),
        "benefit_incumbent_minus_candidate_point_m": float(point),
        "benefit_ci90_m": [float(low), float(high)],
        "benefit_median_m": float(np.median(benefit)),
    }


def classify_evidence(
    *,
    benefit_point: float,
    benefit_ci90: Sequence[float],
    fatal_integrity_checks: dict[str, bool],
) -> str:
    if not fatal_integrity_checks or not all(fatal_integrity_checks.values()):
        return "QA_BLOCKED"
    low, high = map(float, benefit_ci90)
    if benefit_point > 0.0 and low > 0.0:
        return "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY"
    if benefit_point > 0.0:
        return "EXPLORATORY_CHALLENGER_RESEARCH_ONLY"
    if benefit_point < 0.0 and high < 0.0:
        return "PRIMARY_HARM_RESEARCH_ONLY"
    return "INCONCLUSIVE_RESEARCH_ONLY"
