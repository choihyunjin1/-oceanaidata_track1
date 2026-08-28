"""Depth-registered conditional multivariate functional PCA for P2.

The model represents temperature and salinity profiles with one fixed cubic
B-spline basis over physical nominal depth.  It estimates the seasonal mean in
coefficient space, learns a train-only joint low-rank residual distribution,
and conditions latent scores only on public-layer observations.  Target-layer
temperature and salinity are therefore masked together at prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from sklearn.covariance import OAS

TARGET_LAYERS = (2, 3, 4)
LAYER_ID_PUBLIC = (1, 5, 6, 7)
SPLINE_DEGREE = 3
SPLINE_DF = 5
SPLINE_KNOTS = np.array([4.0] * 4 + [20.0] + [50.0] * 4, dtype=np.float64)


def cubic_bspline_df5(depth: np.ndarray | list[float]) -> np.ndarray:
    """Evaluate the fixed five-function cubic B-spline physical-depth basis."""

    values = np.asarray(depth, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("depth must be a finite one-dimensional vector")
    if np.any((values < 4.0) | (values > 50.0)):
        raise ValueError("depth escaped the preregistered 4..50 m spline domain")
    coefficients = np.eye(SPLINE_DF, dtype=np.float64)
    basis = np.column_stack(
        [
            BSpline(SPLINE_KNOTS, coefficients[index], SPLINE_DEGREE)(values)
            for index in range(SPLINE_DF)
        ]
    )
    if basis.shape != (len(values), SPLINE_DF) or not np.isfinite(basis).all():
        raise RuntimeError("invalid cubic B-spline design")
    return basis


def seasonal_harmonics(times: pd.DatetimeIndex) -> np.ndarray:
    """Return intercept plus four annual sine/cosine pairs in KST."""

    utc = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    local = utc.tz_convert("Asia/Seoul")
    minute = local.hour.to_numpy() * 60 + local.minute.to_numpy()
    day = local.dayofyear.to_numpy(dtype=np.float64) + minute / 1440.0
    columns = [np.ones(len(local), dtype=np.float64)]
    for harmonic in (1, 2, 3, 4):
        angle = 2.0 * np.pi * harmonic * day / 365.2425
        columns.extend((np.sin(angle), np.cos(angle)))
    return np.column_stack(columns)


def _profile_coefficient(
    depth: np.ndarray,
    values: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, float] | None:
    keep = np.isfinite(depth) & np.isfinite(values)
    if int(keep.sum()) < SPLINE_DF:
        return None
    basis = cubic_bspline_df5(depth[keep])
    lhs = basis.T @ basis + ridge * np.eye(SPLINE_DF)
    coefficient = np.linalg.solve(lhs, basis.T @ values[keep])
    residual = values[keep] - basis @ coefficient
    return coefficient, float(np.mean(residual**2))


@dataclass(frozen=True)
class PreparedProfiles:
    """Per-time spline coefficients derived without cross-time statistics."""

    times: pd.DatetimeIndex
    coefficients: np.ndarray
    temp_fit_mse: np.ndarray
    psal_fit_mse: np.ndarray


def prepare_complete_target_profiles(
    observations: pd.DataFrame,
    *,
    coefficient_ridge: float = 1e-6,
) -> PreparedProfiles:
    """Fit T/S coefficients only where all target T/S labels are observed.

    Requiring all target-layer temperature and salinity values excludes the
    official hidden interval and makes simultaneous masking auditable.
    """

    required = {"time", "layer", "temp", "psal", "nominal_depth"}
    if missing := required.difference(observations.columns):
        raise ValueError(f"observations missing profile columns: {sorted(missing)}")
    frame = observations.loc[:, list(required)].copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["layer"] = pd.to_numeric(frame["layer"], errors="raise").astype(int)
    frame["nominal_depth"] = pd.to_numeric(frame["nominal_depth"], errors="coerce")
    frame = frame.sort_values(["time", "layer"])

    times: list[pd.Timestamp] = []
    coefficients: list[np.ndarray] = []
    temp_mse: list[float] = []
    psal_mse: list[float] = []
    for timestamp, group in frame.groupby("time", sort=True, observed=True):
        target = group.loc[group["layer"].isin(TARGET_LAYERS)]
        if set(target["layer"].astype(int)) != set(TARGET_LAYERS):
            continue
        if not target[["temp", "psal"]].notna().all().all():
            continue
        depth = target_depth = group["nominal_depth"].to_numpy(np.float64)
        if not np.isfinite(target_depth).all():
            continue
        temp = _profile_coefficient(
            depth,
            group["temp"].to_numpy(np.float64),
            ridge=coefficient_ridge,
        )
        psal = _profile_coefficient(
            depth,
            group["psal"].to_numpy(np.float64),
            ridge=coefficient_ridge,
        )
        if temp is None or psal is None:
            continue
        times.append(pd.Timestamp(timestamp))
        coefficients.append(np.concatenate((temp[0], psal[0])))
        temp_mse.append(temp[1])
        psal_mse.append(psal[1])
    if len(coefficients) < 100:
        raise RuntimeError("too few complete-target profiles for CMFPCA")
    return PreparedProfiles(
        times=pd.DatetimeIndex(times),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        temp_fit_mse=np.asarray(temp_mse, dtype=np.float64),
        psal_fit_mse=np.asarray(psal_mse, dtype=np.float64),
    )


def select_rank(explained_variance: np.ndarray, *, threshold: float = 0.95, cap: int = 4) -> int:
    """Select the smallest train-only cumulative-variance rank, capped at four."""

    variance = np.asarray(explained_variance, dtype=np.float64)
    if variance.ndim != 1 or len(variance) == 0 or np.any(variance < 0):
        raise ValueError("explained variance must be a nonnegative vector")
    total = float(variance.sum())
    if total <= 0:
        return 1
    needed = int(np.searchsorted(np.cumsum(variance) / total, threshold, side="left") + 1)
    return max(1, min(needed, cap, len(variance)))


@dataclass(frozen=True)
class ConditionalMFPCA:
    mean_beta: np.ndarray
    coefficient_scale: np.ndarray
    components: np.ndarray
    latent_variance: np.ndarray
    temp_noise: float
    psal_noise: float
    rank: int
    explained_fraction: float
    train_profiles: int

    @classmethod
    def fit(
        cls,
        profiles: PreparedProfiles,
        train_mask: np.ndarray,
        *,
        variance_threshold: float = 0.95,
        rank_cap: int = 4,
        noise_floor: float = 1e-4,
    ) -> ConditionalMFPCA:
        mask = np.asarray(train_mask, dtype=bool)
        if mask.shape != (len(profiles.times),) or int(mask.sum()) < 100:
            raise ValueError("invalid or insufficient CMFPCA training mask")
        coefficients = profiles.coefficients[mask]
        harmonics = seasonal_harmonics(profiles.times[mask])
        mean_beta = np.linalg.lstsq(harmonics, coefficients, rcond=None)[0]
        residual = coefficients - harmonics @ mean_beta
        scale = residual.std(axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        standardized = residual / scale
        _, singular, right = np.linalg.svd(standardized, full_matrices=False)
        variance = singular**2 / max(len(standardized) - 1, 1)
        rank = select_rank(variance, threshold=variance_threshold, cap=rank_cap)
        explained = float(variance[:rank].sum() / variance.sum())
        temp_noise = max(float(np.median(profiles.temp_fit_mse[mask])), noise_floor)
        psal_noise = max(float(np.median(profiles.psal_fit_mse[mask])), noise_floor)
        return cls(
            mean_beta=mean_beta,
            coefficient_scale=scale,
            components=right[:rank],
            latent_variance=variance[:rank],
            temp_noise=temp_noise,
            psal_noise=psal_noise,
            rank=rank,
            explained_fraction=explained,
            train_profiles=int(mask.sum()),
        )

    def predict(
        self,
        observations: pd.DataFrame,
        query: pd.DataFrame,
    ) -> np.ndarray:
        """Predict target temperature using public T/S observations only."""

        required_query = {"time", "layer", "nominal_depth"}
        if missing := required_query.difference(query.columns):
            raise ValueError(f"query missing columns: {sorted(missing)}")
        public = observations.loc[
            ~observations["layer"].isin(TARGET_LAYERS),
            ["time", "layer", "nominal_depth", "temp", "psal"],
        ].copy()
        public["time"] = pd.to_datetime(public["time"], utc=True)
        public_by_time = {timestamp: group for timestamp, group in public.groupby("time")}

        ordered = query.reset_index(drop=True).copy()
        ordered["time"] = pd.to_datetime(ordered["time"], utc=True)
        if ordered.duplicated(["time", "layer"]).any():
            raise ValueError("query keys are duplicated")
        output = np.empty(len(ordered), dtype=np.float64)
        loading = self.coefficient_scale[:, None] * self.components.T
        latent_cov = np.diag(self.latent_variance)

        for timestamp, rows in ordered.groupby("time", sort=False):
            row_ids = rows.index.to_numpy(int)
            mean_coefficient = seasonal_harmonics(pd.DatetimeIndex([timestamp]))[0] @ self.mean_beta
            group = public_by_time.get(timestamp)
            design_rows: list[np.ndarray] = []
            observed_values: list[float] = []
            noise: list[float] = []
            if group is not None:
                depths = group["nominal_depth"].to_numpy(np.float64)
                valid_depth = np.isfinite(depths) & (depths >= 4.0) & (depths <= 50.0)
                basis = np.zeros((len(group), SPLINE_DF), dtype=np.float64)
                if valid_depth.any():
                    basis[valid_depth] = cubic_bspline_df5(depths[valid_depth])
                for position, (_, item) in enumerate(group.iterrows()):
                    if not valid_depth[position]:
                        continue
                    if np.isfinite(item["temp"]):
                        vector = np.zeros(2 * SPLINE_DF, dtype=np.float64)
                        vector[:SPLINE_DF] = basis[position]
                        design_rows.append(vector)
                        observed_values.append(float(item["temp"]))
                        noise.append(self.temp_noise)
                    if np.isfinite(item["psal"]):
                        vector = np.zeros(2 * SPLINE_DF, dtype=np.float64)
                        vector[SPLINE_DF:] = basis[position]
                        design_rows.append(vector)
                        observed_values.append(float(item["psal"]))
                        noise.append(self.psal_noise)
            if design_rows:
                design = np.vstack(design_rows)
                mapping = design @ loading
                residual = np.asarray(observed_values) - design @ mean_coefficient
                covariance = mapping @ latent_cov @ mapping.T + np.diag(noise)
                score = latent_cov @ mapping.T @ np.linalg.pinv(covariance, rcond=1e-10) @ residual
                conditional_coefficient = mean_coefficient + loading @ score
            else:
                conditional_coefficient = mean_coefficient
            target_basis = cubic_bspline_df5(rows["nominal_depth"].to_numpy(np.float64))
            output[row_ids] = target_basis @ conditional_coefficient[:SPLINE_DF]
        if not np.isfinite(output).all():
            raise RuntimeError("CMFPCA produced non-finite predictions")
        return output

    def receipt(self) -> dict[str, float | int]:
        return {
            "train_profiles": self.train_profiles,
            "rank": self.rank,
            "explained_fraction_at_rank": self.explained_fraction,
            "temp_noise": self.temp_noise,
            "psal_noise": self.psal_noise,
        }


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    actual = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if actual.shape != estimate.shape or not np.isfinite(actual).all() or not np.isfinite(estimate).all():
        raise ValueError("RMSE vectors must be aligned and finite")
    return float(np.sqrt(np.mean((estimate - actual) ** 2)))


def paired_kst_day_bootstrap(
    frame: pd.DataFrame,
    *,
    reference: str,
    candidate: str,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Paired RMSE bootstrap over KST calendar days."""

    work = frame.loc[:, ["time", "truth", reference, candidate]].copy()
    work["time"] = pd.to_datetime(work["time"], utc=True)
    work["day"] = work["time"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    work["se_reference"] = (work[reference] - work["truth"]) ** 2
    work["se_candidate"] = (work[candidate] - work["truth"]) ** 2
    daily = work.groupby("day", sort=True).agg(
        rows=("truth", "size"),
        se_reference=("se_reference", "sum"),
        se_candidate=("se_candidate", "sum"),
    )
    if len(daily) < 10:
        raise ValueError("too few KST days for paired bootstrap")
    values = daily.to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        sampled = values[rng.integers(0, len(values), size=len(values))]
        rows = sampled[:, 0].sum()
        draws[draw] = np.sqrt(sampled[:, 2].sum() / rows) - np.sqrt(
            sampled[:, 1].sum() / rows
        )
    return {
        "unit": "KST calendar day",
        "days": int(len(daily)),
        "replicates": int(replicates),
        "seed": int(seed),
        "mean_delta_rmse": float(draws.mean()),
        "ci90_low": float(np.quantile(draws, 0.05)),
        "ci90_high": float(np.quantile(draws, 0.95)),
        "probability_improved": float(np.mean(draws < 0.0)),
    }


def evaluate_promotion_gate(
    *,
    aggregate_delta: float,
    bootstrap_ci90_high: float,
    fold_deltas: dict[str, float],
    layer_deltas: dict[str, float],
    thresholds: dict[str, float | int],
) -> dict[str, object]:
    checks = {
        "aggregate_delta_rmse": aggregate_delta
        <= float(thresholds["aggregate_delta_rmse_max_c"]),
        "paired_ci90_upper": bootstrap_ci90_high
        < float(thresholds["paired_kst_day_bootstrap_ci90_upper_max_c"]),
        "improved_folds": sum(value < 0.0 for value in fold_deltas.values())
        >= int(thresholds["minimum_improved_folds"]),
        "worst_fold_regression": max(fold_deltas.values())
        <= float(thresholds["maximum_worst_fold_regression_c"]),
        "maximum_layer_regression": max(layer_deltas.values())
        <= float(thresholds["maximum_layer_regression_c"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "observed": {
            "aggregate_delta_rmse_c": aggregate_delta,
            "bootstrap_ci90_high_c": bootstrap_ci90_high,
            "improved_folds": int(sum(value < 0.0 for value in fold_deltas.values())),
            "worst_fold_regression_c": float(max(fold_deltas.values())),
            "maximum_layer_regression_c": float(max(layer_deltas.values())),
        },
        "thresholds": thresholds,
    }


def build_layer_identity_panel(
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Recreate the historical layer-ID OAS panel without physical alignment."""

    frame = observations.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    temp = frame.pivot(index="time", columns="layer", values="temp").sort_index()
    psal = frame.pivot(index="time", columns="layer", values="psal").sort_index()
    panel = pd.DataFrame(index=temp.index)
    x_columns: list[str] = []
    y_columns: list[str] = []
    for layer in LAYER_ID_PUBLIC:
        for name, source in (("temp", temp), ("psal", psal)):
            column = f"{name}_{layer}"
            panel[column] = source[layer]
            x_columns.append(column)
    harmonic = seasonal_harmonics(panel.index)
    for index, name in enumerate(
        ["intercept"]
        + [f"doy_{kind}_{order}" for order in (1, 2, 3, 4) for kind in ("sin", "cos")]
    ):
        if name == "intercept":
            continue
        panel[name] = harmonic[:, index]
        x_columns.append(name)
    for layer in TARGET_LAYERS:
        for name, source in (("temp", temp), ("psal", psal)):
            column = f"{name}_{layer}"
            panel[column] = source[layer]
            y_columns.append(column)
    return panel, x_columns, y_columns


def predict_layer_identity_oas(
    panel: pd.DataFrame,
    query: pd.DataFrame,
    *,
    exclude_start: pd.Timestamp,
    exclude_stop: pd.Timestamp,
    season_bin_days: int = 14,
    season_window_days: float = 60.0,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """Recreate the current complete-case layer-ID seasonal OAS prediction."""

    x_columns = [column for column in panel if column.startswith(("temp_", "psal_", "doy_"))]
    x_columns = [
        column
        for column in x_columns
        if not (
            column.startswith(("temp_", "psal_"))
            and int(column.rsplit("_", 1)[1]) in TARGET_LAYERS
        )
    ]
    y_columns = [
        f"{name}_{layer}" for layer in TARGET_LAYERS for name in ("temp", "psal")
    ]
    query_times = pd.DatetimeIndex(pd.to_datetime(query["time"], utc=True))
    unique_times = pd.DatetimeIndex(query_times.drop_duplicates().sort_values())
    evaluate = panel.loc[unique_times, x_columns]
    values_x = evaluate.to_numpy(np.float64)
    patterns = np.isfinite(values_x)
    prediction = np.full((len(evaluate), len(y_columns)), np.nan, dtype=np.float64)
    local = unique_times.tz_convert("Asia/Seoul")
    bins = ((local.dayofyear.to_numpy() - 1) // season_bin_days).astype(int)
    train_index = panel.index[(panel.index < exclude_start) | (panel.index >= exclude_stop)]
    train_doy = train_index.tz_convert("Asia/Seoul").dayofyear.to_numpy(np.float64)
    receipts: list[dict[str, float | int]] = []
    nx = len(x_columns)
    for season_bin in np.unique(bins):
        center = float(season_bin * season_bin_days + 7.5)
        distance = np.abs(train_doy - center)
        distance = np.minimum(distance, 365.2425 - distance)
        train = panel.loc[
            train_index[distance <= season_window_days], x_columns + y_columns
        ].dropna()
        if len(train) < 100:
            raise RuntimeError(f"insufficient layer-ID OAS rows for bin {season_bin}")
        matrix = train.to_numpy(np.float64)
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale == 0] = 1.0
        estimator = OAS(store_precision=False, assume_centered=False).fit(
            (matrix - mean) / scale
        )
        covariance = estimator.covariance_
        sigma_xx = covariance[:nx, :nx]
        sigma_yx = covariance[nx:, :nx]
        bin_rows = np.flatnonzero(bins == season_bin)
        for pattern in np.unique(patterns[bin_rows], axis=0):
            row_ids = bin_rows[np.all(patterns[bin_rows] == pattern, axis=1)]
            observed = np.flatnonzero(pattern)
            if len(observed):
                conditional = sigma_yx[:, observed] @ np.linalg.pinv(
                    sigma_xx[np.ix_(observed, observed)], rcond=1e-10
                )
                standardized_x = (
                    values_x[np.ix_(row_ids, observed)] - mean[observed]
                ) / scale[observed]
                conditional_y = standardized_x @ conditional.T
            else:
                conditional_y = np.zeros((len(row_ids), len(y_columns)))
            prediction[row_ids] = mean[nx:] + conditional_y * scale[nx:]
        receipts.append(
            {
                "season_bin": int(season_bin),
                "train_timestamps": int(len(train)),
                "oas_shrinkage": float(estimator.shrinkage_),
            }
        )
    lookup = {
        (timestamp, layer): prediction[position, y_columns.index(f"temp_{layer}")]
        for position, timestamp in enumerate(unique_times)
        for layer in TARGET_LAYERS
    }
    result = np.asarray(
        [lookup[(timestamp, int(layer))] for timestamp, layer in zip(query_times, query["layer"], strict=True)],
        dtype=np.float64,
    )
    if not np.isfinite(result).all():
        raise RuntimeError("layer-ID OAS produced non-finite predictions")
    return result, receipts
