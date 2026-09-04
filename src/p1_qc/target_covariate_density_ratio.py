"""Label-blind target-covariate density weighting for the P1 incumbent.

The domain model operates only on station-layer KST-day summaries.  It never
receives the P1 target, anomaly type, incumbent probability, calendar year, or
leaderboard information.  The resulting source-day ratio is mapped back to
training rows and multiplied by the frozen square-root class weight.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from p1_qc.metrics import binary_counts


DOMAIN_CATEGORICAL_COLUMNS = ("station", "layer_category")
DOMAIN_NUMERIC_COLUMNS = (
    "coverage_fraction",
    "temp_missing_rate",
    "psal_missing_rate",
    "depth_missing_rate",
    "temp_median",
    "temp_iqr",
    "temp_std_population",
    "temp_median_abs_diff",
    "psal_median",
    "psal_iqr",
    "psal_std_population",
    "psal_median_abs_diff",
    "depth_median",
    "depth_iqr",
    "depth_std_population",
    "depth_median_abs_diff",
    "day_of_year_sin",
    "day_of_year_cos",
)
DOMAIN_FEATURE_COLUMNS = DOMAIN_CATEGORICAL_COLUMNS + DOMAIN_NUMERIC_COLUMNS
DOMAIN_FORBIDDEN_COLUMNS = (
    "label",
    "anomaly_type",
    "year",
    "incumbent_probability",
    "incumbent_prediction",
    "official_score",
)


def sha256_array(values: np.ndarray, *, dtype: str = "<f8") -> str:
    array = np.asarray(values, dtype=dtype)
    return sha256(array.tobytes(order="C")).hexdigest()


def effective_sample_fraction(weight: Sequence[float]) -> float:
    values = np.asarray(weight, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("weight must be a non-empty one-dimensional vector")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("weight must be finite and strictly positive")
    return float(values.sum() ** 2 / (len(values) * np.square(values).sum()))


def _nan_iqr(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.quantile(0.75) - numeric.quantile(0.25))


def _nan_population_std(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.std(ddof=0))


def build_daily_domain_covariates(
    frame: pd.DataFrame,
    *,
    cadence_minutes: int = 10,
    expected_rows_per_day: int = 144,
) -> pd.DataFrame:
    """Return one label-free row per station-layer KST day.

    First differences are retained only across exact-cadence observations in
    the same station-layer series.  A physical gap therefore never contributes
    to a daily variation statistic.
    """

    required = {"station", "layer", "time", "temp", "psal", "depth"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"missing domain-covariate columns: {missing}")
    if cadence_minutes <= 0 or expected_rows_per_day <= 0:
        raise ValueError("cadence_minutes and expected_rows_per_day must be positive")

    work = frame.loc[:, ["station", "layer", "time", "temp", "psal", "depth"]].copy()
    work["__position"] = np.arange(len(work), dtype=np.int64)
    work["__time"] = pd.to_datetime(work["time"], errors="raise", utc=True, format="mixed")
    work.sort_values(["station", "layer", "__time", "__position"], kind="mergesort", inplace=True)
    grouped_series = work.groupby(["station", "layer"], sort=False, observed=True)
    contiguous = grouped_series["__time"].diff().dt.total_seconds().eq(cadence_minutes * 60)
    for column in ("temp", "psal", "depth"):
        numeric = pd.to_numeric(work[column], errors="coerce")
        prior = numeric.groupby(
            [work["station"], work["layer"]], sort=False, observed=True
        ).shift(1)
        work[f"__{column}_abs_diff"] = (numeric - prior).abs().where(contiguous)
        work[column] = numeric

    local = work["__time"].dt.tz_convert("Asia/Seoul")
    work["kst_day"] = local.dt.strftime("%Y-%m-%d")
    iso = local.dt.isocalendar()
    work["iso_year_week"] = (
        iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    )
    day_of_year = local.dt.dayofyear.groupby(
        [work["station"], work["layer"], work["kst_day"]],
        sort=False,
        observed=True,
    ).first()

    keys = ["station", "layer", "kst_day"]
    grouped = work.groupby(keys, sort=True, observed=True, dropna=False)
    result = grouped.size().rename("row_count").reset_index()
    result["layer_category"] = result["layer"].astype("string")
    result["coverage_fraction"] = result["row_count"].astype(float) / expected_rows_per_day
    result["iso_year_week"] = grouped["iso_year_week"].first().to_numpy()
    result["day_of_year"] = day_of_year.to_numpy(dtype=float)
    phase = 2.0 * np.pi * (result["day_of_year"] - 1.0) / 365.2425
    result["day_of_year_sin"] = np.sin(phase)
    result["day_of_year_cos"] = np.cos(phase)

    for column in ("temp", "psal", "depth"):
        result[f"{column}_missing_rate"] = grouped[column].apply(
            lambda values: float(values.isna().mean())
        ).to_numpy()
        result[f"{column}_median"] = grouped[column].median().to_numpy(dtype=float)
        result[f"{column}_iqr"] = grouped[column].apply(_nan_iqr).to_numpy(dtype=float)
        result[f"{column}_std_population"] = grouped[column].apply(
            _nan_population_std
        ).to_numpy(dtype=float)
        result[f"{column}_median_abs_diff"] = grouped[f"__{column}_abs_diff"].median().to_numpy(
            dtype=float
        )

    result["station"] = result["station"].astype("string")
    result["layer_category"] = result["layer_category"].astype("string")
    if result.duplicated(keys).any():
        raise RuntimeError("daily domain keys are not unique")
    if set(DOMAIN_FEATURE_COLUMNS).intersection(DOMAIN_FORBIDDEN_COLUMNS):
        raise RuntimeError("domain feature contract contains a forbidden feature")
    output_columns = [
        *keys,
        "iso_year_week",
        "row_count",
        "layer_category",
        *DOMAIN_NUMERIC_COLUMNS,
    ]
    if len(output_columns) != len(set(output_columns)):
        raise RuntimeError("daily domain output schema contains duplicate columns")
    return result.loc[:, output_columns].reset_index(drop=True)


def _domain_pipeline(*, seed: int, regularization_c: float) -> Pipeline:
    numeric = Pipeline(
        [
            ("median", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("mode", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    transform = ColumnTransformer(
        [
            ("numeric", numeric, list(DOMAIN_NUMERIC_COLUMNS)),
            ("categorical", categorical, list(DOMAIN_CATEGORICAL_COLUMNS)),
        ]
    )
    classifier = LogisticRegression(
        C=regularization_c,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )
    return Pipeline([("features", transform), ("classifier", classifier)])


@dataclass(frozen=True)
class DensityRatioResult:
    source_daily_ratio: np.ndarray
    source_oof_target_probability: np.ndarray
    audit: Mapping[str, Any]


def estimate_source_daily_density_ratio(
    source_daily: pd.DataFrame,
    target_daily: pd.DataFrame,
    *,
    seed: int,
    n_splits: int = 5,
    regularization_c: float = 0.1,
    ratio_clip: tuple[float, float] = (0.1, 8.0),
) -> DensityRatioResult:
    """Estimate OOF ``P(target|x) / P(source|x)`` for every source day."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    low, high = (float(ratio_clip[0]), float(ratio_clip[1]))
    if not 0 < low <= high:
        raise ValueError("invalid ratio_clip")
    for name, daily in (("source", source_daily), ("target", target_daily)):
        required = {*DOMAIN_FEATURE_COLUMNS, "iso_year_week", "station", "layer", "kst_day"}
        missing = sorted(required.difference(daily.columns))
        if missing:
            raise KeyError(f"{name} daily frame lacks {missing}")
        if not len(daily):
            raise ValueError(f"{name} daily frame is empty")

    source_support = set(
        source_daily.loc[:, ["station", "layer_category"]].itertuples(index=False, name=None)
    )
    target_support = set(
        target_daily.loc[:, ["station", "layer_category"]].itertuples(index=False, name=None)
    )
    missing_target_support = sorted(target_support.difference(source_support), key=str)

    source = source_daily.copy()
    target = target_daily.copy()
    source["__domain"] = np.int8(0)
    target["__domain"] = np.int8(1)
    source["__origin_position"] = np.arange(len(source), dtype=np.int64)
    target["__origin_position"] = np.arange(len(target), dtype=np.int64)
    combined = pd.concat([source, target], ignore_index=True)
    y = combined["__domain"].to_numpy(dtype=np.int8)
    groups = combined["iso_year_week"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probability = np.full(len(combined), np.nan, dtype=np.float64)
    fold_audit: list[dict[str, Any]] = []
    for ordinal, (fit_idx, holdout_idx) in enumerate(
        splitter.split(combined.loc[:, DOMAIN_FEATURE_COLUMNS], y, groups), start=1
    ):
        if set(groups[fit_idx]).intersection(groups[holdout_idx]):
            raise RuntimeError("domain OOF group overlap")
        if len(np.unique(y[fit_idx])) != 2 or len(np.unique(y[holdout_idx])) != 2:
            raise RuntimeError("domain OOF fold lacks one domain class")
        model = _domain_pipeline(seed=seed, regularization_c=regularization_c)
        model.fit(combined.iloc[fit_idx].loc[:, DOMAIN_FEATURE_COLUMNS], y[fit_idx])
        fold_probability = model.predict_proba(
            combined.iloc[holdout_idx].loc[:, DOMAIN_FEATURE_COLUMNS]
        )[:, 1]
        probability[holdout_idx] = fold_probability
        fold_audit.append(
            {
                "fold": ordinal,
                "fit_days": len(fit_idx),
                "holdout_days": len(holdout_idx),
                "fit_source_days": int((y[fit_idx] == 0).sum()),
                "fit_target_days": int((y[fit_idx] == 1).sum()),
                "holdout_source_days": int((y[holdout_idx] == 0).sum()),
                "holdout_target_days": int((y[holdout_idx] == 1).sum()),
                "group_overlap": 0,
            }
        )
    if not np.isfinite(probability).all() or ((probability <= 0) | (probability >= 1)).any():
        raise RuntimeError("domain OOF probability is incomplete or out of range")

    source_probability = probability[: len(source)]
    raw_ratio = source_probability / (1.0 - source_probability)
    ratio = np.clip(raw_ratio, low, high)
    per_station_layer_ess: dict[str, float] = {}
    for key, positions in source.groupby(
        ["station", "layer_category"], sort=True, observed=True
    ).indices.items():
        label = "|".join(str(value) for value in (key if isinstance(key, tuple) else (key,)))
        per_station_layer_ess[label] = effective_sample_fraction(ratio[np.asarray(positions)])

    audit = {
        "seed": int(seed),
        "source_days": len(source),
        "target_days": len(target),
        "domain_feature_count": len(DOMAIN_FEATURE_COLUMNS),
        "domain_features": list(DOMAIN_FEATURE_COLUMNS),
        "forbidden_feature_intersection": sorted(
            set(DOMAIN_FEATURE_COLUMNS).intersection(DOMAIN_FORBIDDEN_COLUMNS)
        ),
        "n_splits": int(n_splits),
        "regularization_C": float(regularization_c),
        "ratio_clip": [low, high],
        "oof_auc": float(roc_auc_score(y, probability)),
        "all_oof_probabilities_finite": bool(np.isfinite(probability).all()),
        "all_groups_disjoint": all(row["group_overlap"] == 0 for row in fold_audit),
        "folds": fold_audit,
        "source_station_layer_support": sorted("|".join(map(str, key)) for key in source_support),
        "target_station_layer_support": sorted("|".join(map(str, key)) for key in target_support),
        "missing_target_station_layer_support": ["|".join(map(str, key)) for key in missing_target_support],
        "daily_ratio_ess_fraction": effective_sample_fraction(ratio),
        "per_station_layer_daily_ratio_ess_fraction": per_station_layer_ess,
        "ratio_min": float(ratio.min()),
        "ratio_median": float(np.median(ratio)),
        "ratio_max": float(ratio.max()),
        "low_clip_fraction": float(np.mean(raw_ratio <= low)),
        "high_clip_fraction": float(np.mean(raw_ratio >= high)),
        "source_probability_sha256": sha256_array(source_probability),
        "source_ratio_sha256": sha256_array(ratio),
    }
    return DensityRatioResult(ratio, source_probability, audit)


def map_daily_ratio_to_rows(
    frame: pd.DataFrame,
    source_daily: pd.DataFrame,
    daily_ratio: Sequence[float],
) -> np.ndarray:
    ratio = np.asarray(daily_ratio, dtype=np.float64)
    if ratio.shape != (len(source_daily),):
        raise ValueError("daily_ratio shape differs from source_daily")
    local = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed").dt.tz_convert(
        "Asia/Seoul"
    )
    row_keys = pd.MultiIndex.from_arrays(
        [
            frame["station"].astype("string"),
            frame["layer"].astype("string"),
            local.dt.strftime("%Y-%m-%d"),
        ],
        names=["station", "layer_category", "kst_day"],
    )
    daily_keys = pd.MultiIndex.from_arrays(
        [
            source_daily["station"].astype("string"),
            source_daily["layer_category"].astype("string"),
            source_daily["kst_day"].astype("string"),
        ],
        names=row_keys.names,
    )
    if daily_keys.has_duplicates:
        raise RuntimeError("source daily ratio keys are duplicated")
    positions = daily_keys.get_indexer(row_keys)
    if (positions < 0).any():
        raise RuntimeError(f"{int((positions < 0).sum())} source rows lack a daily ratio")
    result = ratio[positions]
    if not np.isfinite(result).all() or (result <= 0).any():
        raise RuntimeError("mapped row density ratio is invalid")
    return result


def square_root_class_weight(target: Sequence[int]) -> np.ndarray:
    y = np.asarray(target, dtype=np.int8)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("target must be a one-dimensional binary vector")
    positive = max(1, int(y.sum()))
    negative = max(1, len(y) - positive)
    return np.where(y == 1, math.sqrt(negative / positive), 1.0).astype(np.float64)


def combined_training_weight(
    target: Sequence[int],
    density_ratio: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    y = np.asarray(target, dtype=np.int8)
    ratio = np.asarray(density_ratio, dtype=np.float64)
    if y.shape != ratio.shape:
        raise ValueError("target and density_ratio shapes differ")
    if not np.isfinite(ratio).all() or (ratio <= 0).any():
        raise ValueError("density_ratio must be finite and positive")
    base = square_root_class_weight(y)
    combined = base * ratio
    normalizer = base.sum() / combined.sum()
    combined *= normalizer
    if not np.isfinite(combined).all() or (combined <= 0).any():
        raise RuntimeError("combined training weight is invalid")
    audit = {
        "rows": len(y),
        "positive_rows": int(y.sum()),
        "base_weight_sum": float(base.sum()),
        "combined_weight_sum": float(combined.sum()),
        "sum_difference": float(combined.sum() - base.sum()),
        "normalization_factor": float(normalizer),
        "density_ratio_ess_fraction": effective_sample_fraction(ratio),
        "combined_weight_ess_fraction": effective_sample_fraction(combined),
        "combined_weight_sha256": sha256_array(combined),
    }
    return combined.astype(np.float32), audit


def _weighted_row_weights(
    metadata: pd.DataFrame,
    group_weights: Mapping[Any, float],
) -> np.ndarray:
    keys = [
        tuple(values)
        for values in metadata.loc[:, ["station", "layer"]].itertuples(index=False, name=None)
    ]
    counts: dict[tuple[Any, ...], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    normalised = {
        tuple(key if isinstance(key, tuple) else (key,)): float(value)
        for key, value in group_weights.items()
    }
    mass = sum(normalised.get(key, 0.0) for key in counts)
    if mass <= 0:
        raise ValueError("group_weights have no mass on metadata")
    return np.asarray(
        [normalised.get(key, 0.0) / mass / counts[key] for key in keys], dtype=np.float64
    )


def paired_weighted_block_bootstrap(
    truth: Sequence[int],
    candidate: Sequence[int],
    baseline: Sequence[int],
    metadata: pd.DataFrame,
    group_weights: Mapping[Any, float],
    *,
    replicates: int,
    seed: int,
    cadence_minutes: int = 10,
) -> dict[str, Any]:
    """Paired KST event/day bootstrap for the test-share weighted F1."""

    y = np.asarray(truth, dtype=np.int8)
    cand = np.asarray(candidate, dtype=np.int8)
    base = np.asarray(baseline, dtype=np.int8)
    if y.shape != cand.shape or y.shape != base.shape or y.ndim != 1:
        raise ValueError("truth and predictions must be aligned vectors")
    if len(metadata) != len(y) or replicates < 1:
        raise ValueError("invalid metadata length or replicate count")
    work = metadata.loc[:, ["station", "layer", "time"]].reset_index(drop=True).copy()
    work["__position"] = np.arange(len(work), dtype=np.int64)
    work["__truth"] = y
    work["__time"] = pd.to_datetime(work["time"], errors="raise", utc=True, format="mixed")
    ordered = work.sort_values(["station", "layer", "__time", "__position"], kind="mergesort")
    grouped = ordered.groupby(["station", "layer"], sort=False, observed=True)
    contiguous = grouped["__time"].diff().dt.total_seconds().eq(cadence_minutes * 60)
    prior_positive = grouped["__truth"].shift(1).fillna(0).eq(1)
    starts = ordered["__truth"].eq(1) & (~contiguous | ~prior_positive)
    ordered["__event"] = starts.cumsum().where(ordered["__truth"].eq(1), -1).astype(np.int64)
    event = ordered.sort_values("__position", kind="mergesort")["__event"].to_numpy()
    local_day = work["__time"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    normal_index = pd.MultiIndex.from_arrays(
        [work["station"].astype(str), work["layer"].astype(str), local_day]
    )
    normal_code = pd.factorize(normal_index, sort=True)[0]
    block_key = np.empty(len(y), dtype=object)
    positive = event >= 0
    block_key[positive] = [f"event:{value}" for value in event[positive]]
    block_key[~positive] = [f"normal:{value}" for value in normal_code[~positive]]
    codes, uniques = pd.factorize(block_key, sort=True)
    row_weight = _weighted_row_weights(metadata, group_weights)

    def by_block(prediction: np.ndarray) -> np.ndarray:
        values = np.zeros((len(uniques), 3), dtype=np.float64)
        for block in range(len(uniques)):
            mask = codes == block
            counts = binary_counts(y[mask], prediction[mask], sample_weight=row_weight[mask])
            values[block] = (counts.tp, counts.fp, counts.fn)
        return values

    candidate_counts = by_block(cand)
    baseline_counts = by_block(base)
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(uniques), size=len(uniques))
        c = candidate_counts[sampled].sum(axis=0)
        b = baseline_counts[sampled].sum(axis=0)
        candidate_denominator = 2 * c[0] + c[1] + c[2]
        baseline_denominator = 2 * b[0] + b[1] + b[2]
        candidate_f1 = 2 * c[0] / candidate_denominator if candidate_denominator else 0.0
        baseline_f1 = 2 * b[0] / baseline_denominator if baseline_denominator else 0.0
        differences[replicate] = candidate_f1 - baseline_f1
    q = np.quantile(differences, [0.05, 0.5, 0.95])
    return {
        "replicates": int(replicates),
        "blocks": len(uniques),
        "difference_mean": float(differences.mean()),
        "difference_median": float(q[1]),
        "difference_ci90": [float(q[0]), float(q[2])],
        "probability_improved": float(np.mean(differences > 0)),
        "metric": "test_share_weighted_row_F1",
        "normal_day_timezone": "Asia/Seoul",
    }


__all__ = [
    "DOMAIN_CATEGORICAL_COLUMNS",
    "DOMAIN_FEATURE_COLUMNS",
    "DOMAIN_FORBIDDEN_COLUMNS",
    "DOMAIN_NUMERIC_COLUMNS",
    "DensityRatioResult",
    "build_daily_domain_covariates",
    "combined_training_weight",
    "effective_sample_fraction",
    "estimate_source_daily_density_ratio",
    "map_daily_ratio_to_rows",
    "paired_weighted_block_bootstrap",
    "sha256_array",
    "square_root_class_weight",
]
