"""Low-complexity, leakage-safe method screening for P2.

This module deliberately keeps the research surface small.  Every method uses
only public layers at prediction time and is evaluated on the same contiguous
blocks as the frozen P2 v0 model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from p2_restore.features import PUBLIC_LAYERS, TARGET_LAYERS, FeatureTable
from p2_restore.model import VALIDATION_BLOCKS, P2Model, fit_model

DYNAMIC_STEPS = (6, 36, 75, 144)  # 1 h, 6 h, approximately M2, and 24 h.
M2_PERIOD_HOURS = 12.42
M2_DETREND_STEPS = 144
M2_DETREND_MIN_OBSERVATIONS = 72
M2_HARMONIC_STEPS = 1008
M2_HARMONIC_MIN_OBSERVATIONS = 504
STABILITY_BLOCKS = {
    "2024_jan_feb": ("2024-01-01", "2024-03-01"),
    "2024_mar_apr": ("2024-03-01", "2024-05-01"),
    "2024_may_jun": ("2024-05-01", "2024-07-01"),
    "2024_jul_aug": ("2024-07-01", "2024-09-01"),
    "2024_sep_oct": ("2024-09-01", "2024-11-01"),
    "2024_nov_dec": ("2024-11-01", "2025-01-01"),
    "2025_jan_feb": ("2025-01-01", "2025-03-01"),
    "2025_mar_apr": ("2025-03-01", "2025-05-01"),
    "2025_may_jun": ("2025-05-01", "2025-07-01"),
    "2025_jul_aug": ("2025-07-01", "2025-09-01"),
    "2025_nov_dec": ("2025-11-01", "2026-01-01"),
}


@dataclass(frozen=True)
class MethodPredictions:
    """Predictions for one validation block, aligned to ``rows``."""

    rows: np.ndarray
    truth: np.ndarray
    layer: np.ndarray
    stratification: np.ndarray
    public_count: np.ndarray
    predictions: dict[str, np.ndarray]


@dataclass
class P2ResearchBlendModel:
    """Fixed 50:50 ensemble of the v0 and lean-M2 residual models."""

    base_model: P2Model
    lean_model: P2Model
    weight: float = 0.5

    def predict(self, base: FeatureTable, lean: FeatureTable) -> np.ndarray:
        if self.weight != 0.5:
            raise ValueError("the research blend weight is frozen at 0.5")
        return 0.5 * self.base_model.predict(base) + 0.5 * self.lean_model.predict(lean)


def fit_research_blend(base: FeatureTable, lean: FeatureTable) -> P2ResearchBlendModel:
    """Fit the two deterministic full-data arms used by the fixed blend."""

    if not base.frame[["station", "layer", "time"]].equals(
        lean.frame[["station", "layer", "time"]]
    ):
        raise ValueError("lean dynamics are not aligned to base features")
    return P2ResearchBlendModel(
        base_model=fit_model(base, seed=20260816),
        lean_model=fit_model(lean, seed=20260816),
    )


def append_public_dynamics(
    table: FeatureTable, observations: pd.DataFrame, *, cadence_minutes: int = 10
) -> FeatureTable:
    """Append fixed public-layer lag/lead features without crossing time gaps.

    The target layers are never read.  Leads are allowed because P2 is an
    offline restoration task and the public layers are distributed for the
    whole hidden interval.
    """

    public = observations.loc[
        observations["layer"].isin(PUBLIC_LAYERS), ["time", "layer", "temp", "psal"]
    ].copy()
    public["time_key"] = pd.to_datetime(public["time"], utc=True)
    times = pd.Index(sorted(public["time_key"].unique()))
    delta = pd.Series(times).diff().dt.total_seconds().div(60)
    segment = delta.ne(cadence_minutes).cumsum().to_numpy()
    derived_columns: dict[str, pd.Series] = {}

    for value in ("temp", "psal"):
        wide = public.pivot(index="time_key", columns="layer", values=value).reindex(times)
        for layer in PUBLIC_LAYERS:
            series = wide.get(layer, pd.Series(index=times, dtype=float)).astype(float)
            grouped = series.groupby(segment, sort=False)
            for steps in DYNAMIC_STEPS:
                label = "m2" if steps == 75 else f"{steps * cadence_minutes // 60}h"
                past = grouped.shift(steps)
                future = grouped.shift(-steps)
                derived_columns[f"{value}_{layer}_delta_past_{label}"] = series - past
                derived_columns[f"{value}_{layer}_delta_future_{label}"] = future - series
                derived_columns[f"{value}_{layer}_center_change_{label}"] = future - past

    derived = pd.DataFrame(derived_columns, index=times)

    keyed = table.frame.copy()
    keyed["_time_key"] = pd.to_datetime(keyed["time"], utc=True)
    keyed = keyed.join(derived, on="_time_key", validate="many_to_one").drop(columns="_time_key")
    added = tuple(column for column in derived.columns if column not in table.feature_columns)
    forbidden = {f"temp_{layer}" for layer in TARGET_LAYERS} | {
        f"psal_{layer}" for layer in TARGET_LAYERS
    }
    if forbidden.intersection(added):
        raise AssertionError("dynamic feature construction exposed a hidden target layer")
    return FeatureTable(keyed, tuple(sorted((*table.feature_columns, *added))))


def select_lean_m2_dynamics(base: FeatureTable, dynamic: FeatureTable) -> FeatureTable:
    """Keep exactly 20 public-temperature changes at 6 h and the M2 period."""

    additions = tuple(
        column
        for column in dynamic.feature_columns
        if column.startswith("temp_")
        and ("_delta_past_" in column or "_delta_future_" in column)
        and (column.endswith("_6h") or column.endswith("_m2"))
    )
    if len(additions) != 20:
        raise ValueError(f"expected exactly 20 lean dynamic features, found {len(additions)}")
    columns = tuple(sorted((*base.feature_columns, *additions)))
    return FeatureTable(dynamic.frame.copy(), columns)


def append_public_m2_harmonics(
    table: FeatureTable, observations: pd.DataFrame, *, cadence_minutes: int = 10
) -> FeatureTable:
    """Append one fixed local M2 amplitude/phase representation per public layer.

    The transform is label-blind and only reads public-layer temperature.  A
    centered 24-hour mean removes the slowly varying background, then a fixed
    centered seven-day projection estimates the local M2 quadratures.  Rolling
    calculations reset at every non-10-minute time gap.  Centering is permitted
    because P2 is an offline restoration task with the public layers available
    over the complete hidden interval.
    """

    public = observations.loc[
        observations["layer"].isin(PUBLIC_LAYERS), ["time", "layer", "temp"]
    ].copy()
    public["time_key"] = pd.to_datetime(public["time"], utc=True)
    times = pd.Index(sorted(public["time_key"].unique()))
    delta = pd.Series(times).diff().dt.total_seconds().div(60)
    segments = delta.ne(cadence_minutes).cumsum().to_numpy()
    wide = public.pivot(index="time_key", columns="layer", values="temp").reindex(times)
    epoch_seconds = pd.to_datetime(times, utc=True).as_unit("ns").asi8 / 1e9
    omega_time = 2 * np.pi * epoch_seconds / (M2_PERIOD_HOURS * 3600)
    cosine = pd.Series(np.cos(omega_time), index=times)
    sine = pd.Series(np.sin(omega_time), index=times)
    derived_columns: dict[str, pd.Series] = {}

    for layer in PUBLIC_LAYERS:
        series = wide.get(layer, pd.Series(index=times, dtype=float)).astype(float)
        coefficient_cos = pd.Series(np.nan, index=times, dtype=float)
        coefficient_sin = pd.Series(np.nan, index=times, dtype=float)
        for segment_id in pd.unique(segments):
            keep = segments == segment_id
            local = series.loc[keep]
            background = local.rolling(
                M2_DETREND_STEPS,
                center=True,
                min_periods=M2_DETREND_MIN_OBSERVATIONS,
            ).mean()
            residual = local - background
            coefficient_cos.loc[keep] = (
                2
                * (residual * cosine.loc[keep])
                .rolling(
                    M2_HARMONIC_STEPS,
                    center=True,
                    min_periods=M2_HARMONIC_MIN_OBSERVATIONS,
                )
                .mean()
            )
            coefficient_sin.loc[keep] = (
                2
                * (residual * sine.loc[keep])
                .rolling(
                    M2_HARMONIC_STEPS,
                    center=True,
                    min_periods=M2_HARMONIC_MIN_OBSERVATIONS,
                )
                .mean()
            )

        amplitude = np.sqrt(coefficient_cos**2 + coefficient_sin**2)
        stable_amplitude = amplitude.where(amplitude > 1e-8)
        prefix = f"temp_{layer}_m2_local"
        derived_columns[f"{prefix}_amplitude"] = amplitude
        derived_columns[f"{prefix}_phase_cos"] = coefficient_cos / stable_amplitude
        derived_columns[f"{prefix}_phase_sin"] = coefficient_sin / stable_amplitude
        derived_columns[f"{prefix}_reconstruction"] = (
            coefficient_cos * cosine + coefficient_sin * sine
        )

    derived = pd.DataFrame(derived_columns, index=times)
    keyed = table.frame.copy()
    keyed["_time_key"] = pd.to_datetime(keyed["time"], utc=True)
    keyed = keyed.join(derived, on="_time_key", validate="many_to_one").drop(columns="_time_key")
    added = tuple(sorted(derived.columns))
    if len(added) != 20:
        raise AssertionError(f"expected exactly 20 M2 harmonic features, found {len(added)}")
    forbidden = tuple(
        column
        for column in added
        if any(column.startswith(f"temp_{layer}_") for layer in TARGET_LAYERS)
    )
    if forbidden:
        raise AssertionError("M2 harmonic features exposed a hidden target layer")
    return FeatureTable(keyed, tuple(sorted((*table.feature_columns, *added))))


def pchip_profile_prediction(frame: pd.DataFrame) -> np.ndarray:
    """Shape-preserving vertical interpolation using all available public levels."""

    result = frame["baseline"].to_numpy(float).copy()
    temperatures = frame[[f"temp_{layer}" for layer in PUBLIC_LAYERS]].to_numpy(float)
    depths = frame[[f"nominal_{layer}" for layer in PUBLIC_LAYERS]].to_numpy(float)
    targets = frame["target_depth"].to_numpy(float)
    for row in range(len(frame)):
        keep = np.isfinite(temperatures[row]) & np.isfinite(depths[row])
        if keep.sum() < 3 or not np.isfinite(targets[row]):
            continue
        x, y = depths[row, keep], temperatures[row, keep]
        order = np.argsort(x)
        x, y = x[order], y[order]
        unique = np.concatenate(([True], np.diff(x) > 1e-9))
        x, y = x[unique], y[unique]
        if len(x) < 3 or targets[row] < x[0] or targets[row] > x[-1]:
            continue
        value = float(PchipInterpolator(x, y, extrapolate=False)(targets[row]))
        if np.isfinite(value):
            result[row] = value
    return np.clip(result, -5.0, 45.0)


def _fit_layerwise_ridge(
    table: FeatureTable, train_rows: np.ndarray, validation_rows: np.ndarray
) -> np.ndarray:
    prediction = table.frame.loc[validation_rows, "baseline"].to_numpy(float).copy()
    train_layer = table.frame.loc[train_rows, "layer"].to_numpy(int)
    validation_layer = table.frame.loc[validation_rows, "layer"].to_numpy(int)
    x_train = table.frame.loc[train_rows, table.feature_columns]
    x_validation = table.frame.loc[validation_rows, table.feature_columns]
    y_train = table.frame.loc[train_rows, "residual"].to_numpy(float)
    for layer in TARGET_LAYERS:
        fit_mask = train_layer == layer
        predict_mask = validation_layer == layer
        estimator = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            Ridge(alpha=10.0),
        )
        estimator.fit(x_train.loc[fit_mask], y_train[fit_mask])
        prediction[predict_mask] += estimator.predict(x_validation.loc[predict_mask])
    return np.clip(prediction, -5.0, 45.0)


class EOFProfileModel:
    """Vertical EOF reconstruction from incomplete public profiles.

    EOF coefficients are estimated by least squares from the public levels,
    following the standard incomplete-profile projection idea.  A fixed rank of
    three is used to avoid tuning on the validation blocks.
    """

    def __init__(self, *, rank: int = 3) -> None:
        self.rank = rank
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None

    @staticmethod
    def _profile_matrix(observations: pd.DataFrame) -> tuple[pd.Index, np.ndarray]:
        wide = observations.pivot(index="time", columns="layer", values="temp").sort_index()
        # 2024 layer 7 occupies the ~49 m regime that becomes layer 8 in 2025.
        year = pd.to_datetime(wide.index, utc=True).year
        deep_39 = wide.get(7, pd.Series(index=wide.index, dtype=float)).where(year >= 2025)
        deep_49 = wide.get(8, pd.Series(index=wide.index, dtype=float)).where(
            year >= 2025,
            wide.get(7, pd.Series(index=wide.index, dtype=float)),
        )
        columns = [
            wide.get(layer, pd.Series(index=wide.index, dtype=float))
            for layer in (1, 2, 3, 4, 5, 6)
        ] + [deep_39, deep_49]
        return wide.index, np.column_stack([column.to_numpy(float) for column in columns])

    def fit(self, observations: pd.DataFrame, excluded_times: set[str]) -> EOFProfileModel:
        times, matrix = self._profile_matrix(observations)
        canonical = pd.to_datetime(times, utc=True).astype(str)
        keep = np.isfinite(matrix).all(axis=1) & ~canonical.isin(excluded_times)
        if keep.sum() < 100:
            raise ValueError("too few complete profiles to fit the EOF model")
        pca = PCA(n_components=self.rank, svd_solver="full", random_state=20260816)
        pca.fit(matrix[keep])
        self.mean_ = pca.mean_.copy()
        self.components_ = pca.components_.copy()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("EOFProfileModel must be fitted before prediction")
        result = frame["baseline"].to_numpy(float).copy()
        public_positions = np.array([0, 4, 5, 6, 7])
        public_values = frame[[f"temp_{layer}" for layer in PUBLIC_LAYERS]].to_numpy(float)
        target_positions = frame["layer"].to_numpy(int) - 1
        for row in range(len(frame)):
            keep = np.isfinite(public_values[row])
            positions = public_positions[keep]
            if len(positions) < self.rank:
                continue
            design = self.components_[:, positions].T
            centered = public_values[row, keep] - self.mean_[positions]
            coefficient, *_ = np.linalg.lstsq(design, centered, rcond=None)
            target = target_positions[row]
            value = self.mean_[target] + self.components_[:, target] @ coefficient
            if np.isfinite(value):
                result[row] = value
        return np.clip(result, -5.0, 45.0)


def physics_blend(frame: pd.DataFrame, model_prediction: np.ndarray) -> np.ndarray:
    """Shrink to linear interpolation as the public 1--5 layer contrast collapses."""

    contrast = np.abs(frame["temp_1_minus_5"].to_numpy(float))
    weight = np.clip((contrast - 0.2) / 1.8, 0.0, 1.0)
    weight = np.where(np.isfinite(weight), weight, 0.0)
    baseline = frame["baseline"].to_numpy(float)
    return baseline + weight * (np.asarray(model_prediction, dtype=float) - baseline)


def run_method_screen(
    table: FeatureTable, dynamic_table: FeatureTable, observations: pd.DataFrame
) -> tuple[dict[str, object], list[MethodPredictions]]:
    """Run a fixed six-method screen on the three predeclared blocks."""

    time = pd.to_datetime(table.frame["time"], utc=True)
    if not table.frame[["station", "layer", "time"]].equals(
        dynamic_table.frame[["station", "layer", "time"]]
    ):
        raise ValueError("dynamic features are not aligned to the base feature table")
    reports: dict[str, object] = {}
    outputs: list[MethodPredictions] = []
    for number, (name, (start, stop)) in enumerate(VALIDATION_BLOCKS.items()):
        left = pd.Timestamp(start, tz="Asia/Seoul").tz_convert("UTC")
        right = pd.Timestamp(stop, tz="Asia/Seoul").tz_convert("UTC")
        validation = (time.ge(left) & time.lt(right)).to_numpy()
        train = ~validation
        validation_table = FeatureTable(
            table.frame.loc[validation].reset_index(drop=True), table.feature_columns
        )
        dynamic_validation = FeatureTable(
            dynamic_table.frame.loc[validation].reset_index(drop=True),
            dynamic_table.feature_columns,
        )
        base_model = fit_model(table, train, seed=20260816 + number)
        dynamic_model = fit_model(dynamic_table, train, seed=20260816 + number)
        dynamic_prediction = dynamic_model.predict(dynamic_validation)
        validation_times = set(
            pd.to_datetime(table.frame.loc[validation, "time"], utc=True).astype(str)
        )
        eof = EOFProfileModel(rank=3).fit(observations, validation_times)
        frame = validation_table.frame
        predictions = {
            "linear_baseline": frame["baseline"].to_numpy(float),
            "pchip_profile": pchip_profile_prediction(frame),
            "ridge_residual": _fit_layerwise_ridge(table, train, validation),
            "eof_rank3": eof.predict(frame),
            "lgbm_v0": base_model.predict(validation_table),
            "lgbm_dynamic": dynamic_prediction,
            "dynamic_physics_blend": physics_blend(frame, dynamic_prediction),
        }
        truth = frame["target"].to_numpy(float)
        layer = frame["layer"].to_numpy(int)
        block_report: dict[str, object] = {"rows": int(validation.sum()), "methods": {}}
        for method, prediction in predictions.items():
            error = prediction - truth
            block_report["methods"][method] = {
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
                "by_layer": {
                    str(target): float(np.sqrt(np.mean(error[layer == target] ** 2)))
                    for target in TARGET_LAYERS
                },
            }
        reports[name] = block_report
        outputs.append(
            MethodPredictions(
                rows=np.flatnonzero(validation),
                truth=truth,
                layer=layer,
                stratification=np.abs(frame["temp_1_minus_5"].to_numpy(float)),
                public_count=frame["public_temp_count"].to_numpy(float),
                predictions=predictions,
            )
        )
    return reports, outputs


def diagnostic_summary(outputs: list[MethodPredictions]) -> dict[str, object]:
    """Aggregate error drivers without exposing raw observations."""

    bins = np.array([-np.inf, 0.2, 0.5, 1.0, 2.0, 5.0, np.inf])
    labels = ("<=0.2", "0.2-0.5", "0.5-1", "1-2", "2-5", ">5")
    combined: dict[str, object] = {"stratification_bins": {}, "public_count": {}}
    truth = np.concatenate([output.truth for output in outputs])
    stratification = np.concatenate([output.stratification for output in outputs])
    public_count = np.concatenate([output.public_count for output in outputs])
    methods = outputs[0].predictions
    predictions = {
        method: np.concatenate([output.predictions[method] for output in outputs])
        for method in methods
    }

    for index, label in enumerate(labels):
        keep = (stratification > bins[index]) & (stratification <= bins[index + 1])
        combined["stratification_bins"][label] = {
            "rows": int(keep.sum()),
            "rmse": {
                method: float(np.sqrt(np.mean((prediction[keep] - truth[keep]) ** 2)))
                for method, prediction in predictions.items()
                if keep.any()
            },
        }
    for count in sorted(np.unique(public_count[np.isfinite(public_count)]).astype(int)):
        keep = public_count == count
        combined["public_count"][str(count)] = {
            "rows": int(keep.sum()),
            "rmse": {
                method: float(np.sqrt(np.mean((prediction[keep] - truth[keep]) ** 2)))
                for method, prediction in predictions.items()
            },
        }
    return combined


def run_seasonal_stability_screen(
    base: FeatureTable, lean: FeatureTable
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compare v0 with one fixed lean candidate on 11 non-overlapping blocks."""

    if not base.frame[["station", "layer", "time"]].equals(
        lean.frame[["station", "layer", "time"]]
    ):
        raise ValueError("lean dynamics are not aligned to base features")
    time = pd.to_datetime(base.frame["time"], utc=True)
    reports: dict[str, object] = {}
    oof_parts: list[pd.DataFrame] = []
    for number, (name, (start, stop)) in enumerate(STABILITY_BLOCKS.items()):
        left = pd.Timestamp(start, tz="Asia/Seoul").tz_convert("UTC")
        right = pd.Timestamp(stop, tz="Asia/Seoul").tz_convert("UTC")
        validation = (time.ge(left) & time.lt(right)).to_numpy()
        if not validation.any():
            continue
        train = ~validation
        base_validation = FeatureTable(
            base.frame.loc[validation].reset_index(drop=True), base.feature_columns
        )
        lean_validation = FeatureTable(
            lean.frame.loc[validation].reset_index(drop=True), lean.feature_columns
        )
        seed = 20260816 + number
        base_prediction = fit_model(base, train, seed=seed).predict(base_validation)
        lean_prediction = fit_model(lean, train, seed=seed).predict(lean_validation)
        blend_prediction = 0.5 * base_prediction + 0.5 * lean_prediction
        truth = base_validation.frame["target"].to_numpy(float)
        layer = base_validation.frame["layer"].to_numpy(int)

        def metric(
            prediction: np.ndarray, truth: np.ndarray = truth, layer: np.ndarray = layer
        ) -> dict[str, object]:
            error = prediction - truth
            return {
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
                "by_layer": {
                    str(target): float(np.sqrt(np.mean(error[layer == target] ** 2)))
                    for target in TARGET_LAYERS
                },
            }

        reports[name] = {
            "rows": int(validation.sum()),
            "v0": metric(base_prediction),
            "lean_m2": metric(lean_prediction),
            "blend50": metric(blend_prediction),
        }
        oof_parts.append(
            pd.DataFrame(
                {
                    "time": pd.to_datetime(base_validation.frame["time"], utc=True).dt.tz_convert(
                        "Asia/Seoul"
                    ),
                    "layer": layer,
                    "truth": truth,
                    "v0": base_prediction,
                    "lean_m2": lean_prediction,
                    "blend50": blend_prediction,
                }
            )
        )
    oof = pd.concat(oof_parts, ignore_index=True)
    oof["day"] = oof["time"].dt.floor("D").astype(str)
    oof["month"] = oof["time"].dt.strftime("%Y-%m")
    return reports, oof


def run_m2_phase_stability_screen(
    base: FeatureTable, lean: FeatureTable, phase: FeatureTable
) -> tuple[dict[str, object], pd.DataFrame]:
    """One-shot comparison of current Blend50 with the fixed M2-phase arm."""

    keys = ["station", "layer", "time"]
    if not base.frame[keys].equals(lean.frame[keys]) or not base.frame[keys].equals(
        phase.frame[keys]
    ):
        raise ValueError("M2 phase features are not aligned to the frozen feature arms")
    harmonic_additions = tuple(
        column for column in phase.feature_columns if column not in lean.feature_columns
    )
    if len(harmonic_additions) != 20 or not all(
        "_m2_local_" in column for column in harmonic_additions
    ):
        raise ValueError("the M2 phase experiment requires exactly 20 local harmonic features")

    time = pd.to_datetime(base.frame["time"], utc=True)
    reports: dict[str, object] = {}
    oof_parts: list[pd.DataFrame] = []
    for number, (name, (start, stop)) in enumerate(STABILITY_BLOCKS.items()):
        left = pd.Timestamp(start, tz="Asia/Seoul").tz_convert("UTC")
        right = pd.Timestamp(stop, tz="Asia/Seoul").tz_convert("UTC")
        validation = (time.ge(left) & time.lt(right)).to_numpy()
        if not validation.any():
            continue
        train = ~validation

        def subset(table: FeatureTable, selected: np.ndarray = validation) -> FeatureTable:
            return FeatureTable(
                table.frame.loc[selected].reset_index(drop=True), table.feature_columns
            )

        seed = 20260816 + number
        base_prediction = fit_model(base, train, seed=seed).predict(subset(base))
        lean_prediction = fit_model(lean, train, seed=seed).predict(subset(lean))
        phase_prediction = fit_model(phase, train, seed=seed).predict(subset(phase))
        current_prediction = 0.5 * base_prediction + 0.5 * lean_prediction
        candidate_prediction = 0.5 * base_prediction + 0.5 * phase_prediction
        truth = base.frame.loc[validation, "target"].to_numpy(float)
        layer = base.frame.loc[validation, "layer"].to_numpy(int)

        def metric(
            prediction: np.ndarray,
            expected: np.ndarray = truth,
            target_layer: np.ndarray = layer,
        ) -> dict[str, object]:
            error = prediction - expected
            return {
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
                "by_layer": {
                    str(target): float(np.sqrt(np.mean(error[target_layer == target] ** 2)))
                    for target in TARGET_LAYERS
                },
            }

        reports[name] = {
            "rows": int(validation.sum()),
            "current_blend50": metric(current_prediction),
            "phase_blend50": metric(candidate_prediction),
        }
        oof_parts.append(
            pd.DataFrame(
                {
                    "time": pd.to_datetime(base.frame.loc[validation, "time"], utc=True)
                    .dt.tz_convert("Asia/Seoul")
                    .to_numpy(),
                    "layer": layer,
                    "truth": truth,
                    "current_blend50": current_prediction,
                    "phase_blend50": candidate_prediction,
                }
            )
        )
    oof = pd.concat(oof_parts, ignore_index=True)
    oof["day"] = oof["time"].dt.floor("D").astype(str)
    oof["month"] = oof["time"].dt.strftime("%Y-%m")
    return reports, oof


def comparison_diagnostics(
    oof: pd.DataFrame, *, reference: str, candidate: str
) -> dict[str, object]:
    """Aggregate a fixed RMSE comparison by month and target layer."""

    result: dict[str, object] = {"by_month": {}, "by_layer": {}}

    def values(frame: pd.DataFrame) -> dict[str, float | int]:
        truth = frame["truth"].to_numpy(float)
        reference_prediction = frame[reference].to_numpy(float)
        candidate_prediction = frame[candidate].to_numpy(float)
        reference_rmse = float(np.sqrt(np.mean((reference_prediction - truth) ** 2)))
        candidate_rmse = float(np.sqrt(np.mean((candidate_prediction - truth) ** 2)))
        return {
            "rows": len(frame),
            "reference_rmse": reference_rmse,
            "candidate_rmse": candidate_rmse,
            "delta_rmse": candidate_rmse - reference_rmse,
        }

    result["aggregate"] = values(oof)
    result["by_month"] = {
        str(month): values(frame) for month, frame in oof.groupby("month", sort=True)
    }
    result["by_layer"] = {
        str(layer): values(frame) for layer, frame in oof.groupby("layer", sort=True)
    }
    return result


def paired_rmse_bootstrap(
    oof: pd.DataFrame,
    *,
    reference: str,
    candidate: str,
    replicates: int = 2000,
    seed: int = 20260816,
) -> dict[str, float | int | str]:
    """Paired KST-day bootstrap for candidate-minus-reference RMSE."""

    required = {"day", "truth", reference, candidate}
    missing = required.difference(oof.columns)
    if missing:
        raise ValueError(f"OOF comparison is missing columns: {sorted(missing)}")
    days = oof["day"].drop_duplicates().to_numpy()
    grouped = {day: group.index.to_numpy() for day, group in oof.groupby("day", sort=False)}
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    truth = oof["truth"].to_numpy(float)
    reference_prediction = oof[reference].to_numpy(float)
    candidate_prediction = oof[candidate].to_numpy(float)
    for replicate in range(replicates):
        selected = rng.choice(days, size=len(days), replace=True)
        indices = np.concatenate([grouped[day] for day in selected])
        reference_rmse = np.sqrt(np.mean((reference_prediction[indices] - truth[indices]) ** 2))
        candidate_rmse = np.sqrt(np.mean((candidate_prediction[indices] - truth[indices]) ** 2))
        deltas[replicate] = candidate_rmse - reference_rmse
    observed = float(
        np.sqrt(np.mean((candidate_prediction - truth) ** 2))
        - np.sqrt(np.mean((reference_prediction - truth) ** 2))
    )
    return {
        "reference": reference,
        "candidate": candidate,
        "delta_rmse": observed,
        "ci90_low": float(np.quantile(deltas, 0.05)),
        "ci90_high": float(np.quantile(deltas, 0.95)),
        "probability_improved": float(np.mean(deltas < 0)),
        "replicates": replicates,
        "kst_days": int(len(days)),
    }


def stability_diagnostics(oof: pd.DataFrame) -> dict[str, object]:
    """Month and layer cuts for the two fixed stability-screen arms."""

    result: dict[str, object] = {"by_month": {}, "by_layer": {}}

    def values(frame: pd.DataFrame) -> dict[str, float | int]:
        truth = frame["truth"].to_numpy(float)
        v0 = frame["v0"].to_numpy(float)
        lean = frame["lean_m2"].to_numpy(float)
        blend = frame["blend50"].to_numpy(float)
        v0_rmse = float(np.sqrt(np.mean((v0 - truth) ** 2)))
        lean_rmse = float(np.sqrt(np.mean((lean - truth) ** 2)))
        blend_rmse = float(np.sqrt(np.mean((blend - truth) ** 2)))
        return {
            "rows": len(frame),
            "v0_rmse": v0_rmse,
            "lean_m2_rmse": lean_rmse,
            "lean_delta_rmse": lean_rmse - v0_rmse,
            "blend50_rmse": blend_rmse,
            "blend50_delta_rmse": blend_rmse - v0_rmse,
        }

    result["by_month"] = {
        str(month): values(frame) for month, frame in oof.groupby("month", sort=True)
    }
    result["by_layer"] = {
        str(layer): values(frame) for layer, frame in oof.groupby("layer", sort=True)
    }
    return result


def paired_day_bootstrap(
    oof: pd.DataFrame,
    *,
    candidate: str = "lean_m2",
    replicates: int = 2000,
    seed: int = 20260816,
) -> dict[str, float]:
    """Paired KST-day bootstrap for a candidate-minus-v0 RMSE difference."""

    if candidate not in {"lean_m2", "blend50"}:
        raise ValueError("candidate must be lean_m2 or blend50")

    result = paired_rmse_bootstrap(
        oof,
        reference="v0",
        candidate=candidate,
        replicates=replicates,
        seed=seed,
    )
    return {
        "candidate": candidate,
        "delta_rmse": float(result["delta_rmse"]),
        "ci90_low": float(result["ci90_low"]),
        "ci90_high": float(result["ci90_high"]),
        "probability_improved": float(result["probability_improved"]),
        "replicates": int(result["replicates"]),
        "kst_days": int(result["kst_days"]),
    }
