"""Chronological soft routing over cross-fitted P3 component forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .validation import rmse

LEADS = (3, 6, 9, 12, 18, 24)
COMPONENTS = ("single", "multi", "persistence")
OBSERVED_FEATURES = (
    "hs_current",
    "hs_delta_1h",
    "hs_delta_3h",
    "hs_delta_6h",
    "hs_delta_12h",
    "hs_std_3h",
    "hs_std_6h",
    "hs_std_12h",
    "hs_mean_24h",
    "tp_current",
    "hmax_current",
    "wspd_current",
    "wspd_delta_3h",
    "wspd_mean_12h",
    "gust_current",
    "caph_current",
    "caph_delta_6h",
)
LOSS_FLOOR = 0.05


@dataclass(frozen=True)
class RouterConfig:
    alpha: float
    temperature_multiplier: float
    strength: float
    name: str

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.temperature_multiplier <= 0.0:
            raise ValueError("temperature_multiplier must be positive")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")


ROUTER_GRID = (
    RouterConfig(10.0, 1.0, 0.0, "no_op"),
    RouterConfig(10.0, 0.5, 0.25, "sharp_weak"),
    RouterConfig(10.0, 1.0, 0.25, "balanced_weak"),
    RouterConfig(10.0, 1.0, 0.50, "balanced_medium"),
    RouterConfig(10.0, 2.0, 0.50, "smooth_medium"),
)


def build_inference_router_features(
    observed: pd.DataFrame,
    station: np.ndarray,
    current_hs: np.ndarray,
    components: np.ndarray,
) -> pd.DataFrame:
    """Build the exact label-free router feature surface for train or hidden cases."""

    station_values = np.asarray(station).astype(str)
    current = np.asarray(current_hs, dtype=float)
    values = np.asarray(components, dtype=float)
    cases = len(observed)
    if len(station_values) != cases or current.shape != (cases,):
        raise ValueError("station/current rows are not aligned")
    if values.shape != (cases, len(LEADS), len(COMPONENTS)):
        raise ValueError("components must have shape (cases, 6, 3)")
    missing = set(OBSERVED_FEATURES).difference(observed.columns)
    if missing:
        raise ValueError(f"observed feature cache is incomplete: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for case in range(cases):
        row: dict[str, object] = {"station": station_values[case]}
        for lead_index, lead in enumerate(LEADS):
            row[f"single_delta_{lead}h"] = float(values[case, lead_index, 0] - current[case])
            row[f"multi_delta_{lead}h"] = float(values[case, lead_index, 1] - current[case])
            row[f"component_absdiff_{lead}h"] = float(
                abs(values[case, lead_index, 0] - values[case, lead_index, 1])
            )
        for index, name in enumerate(("single", "multi")):
            delta = values[case, :, index] - current[case]
            row[f"{name}_peak_gain"] = float(np.max(delta))
            row[f"{name}_final_gain"] = float(delta[-1])
            row[f"{name}_drawdown"] = float(np.max(delta) - delta[-1])
        for feature in OBSERVED_FEATURES:
            row[feature] = observed.iloc[case][feature]
        rows.append(row)
    return pd.DataFrame(rows, index=observed.index)


def build_case_router_data(
    oof: pd.DataFrame,
    observed_features: pd.DataFrame,
    anchors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Return label-free router inputs, metadata, component forecasts, and case losses."""

    required_oof = {
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "current_hs",
        "target_hs",
        "single_prediction",
        "multi_prediction",
        "persistence",
    }
    missing = required_oof.difference(oof.columns)
    if missing:
        raise ValueError(f"OOF is missing columns: {sorted(missing)}")
    keys = ["fold", "anchor_id", "station", "lead_h"]
    if oof.duplicated(keys).any():
        raise ValueError("OOF component keys must be unique")
    counts = oof.groupby(["fold", "anchor_id"], sort=False)["lead_h"].agg(["size", "nunique"])
    if not counts.eq(len(LEADS)).all().all():
        raise ValueError("each router case must contain six distinct leads")
    if set(oof["lead_h"].astype(int)) != set(LEADS):
        raise ValueError("unexpected lead values")

    ordered = oof.sort_values(["fold", "anchor_id", "lead_h"]).reset_index(drop=True)
    case_rows: list[dict[str, object]] = []
    components: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    current_values: list[float] = []
    for (fold, anchor_id), group in ordered.groupby(["fold", "anchor_id"], sort=False):
        group = group.sort_values("lead_h")
        if tuple(group["lead_h"].astype(int)) != LEADS:
            raise ValueError("case lead order does not match the official contract")
        component = np.column_stack(
            [
                group["single_prediction"].to_numpy(float),
                group["multi_prediction"].to_numpy(float),
                group["persistence"].to_numpy(float),
            ]
        )
        truth = group["target_hs"].to_numpy(float)
        current = float(group["current_hs"].iloc[0])
        row: dict[str, object] = {
            "fold": str(fold),
            "anchor_id": int(anchor_id),
            "station": str(group["station"].iloc[0]),
        }
        case_rows.append(row)
        components.append(component)
        losses.append(np.mean(np.square(component - truth[:, None]), axis=0))
        current_values.append(current)

    metadata = pd.DataFrame(case_rows)
    if metadata["anchor_id"].duplicated().any():
        raise ValueError("anchor_id must be globally unique in router OOF")
    observed = observed_features[["anchor_id", *OBSERVED_FEATURES]].copy()
    if observed["anchor_id"].duplicated().any():
        raise ValueError("observed feature anchor_id must be unique")
    observed = metadata[["anchor_id"]].merge(
        observed, on="anchor_id", how="left", validate="one_to_one"
    )
    component_array = np.stack(components)
    inputs = build_inference_router_features(
        observed.loc[:, OBSERVED_FEATURES],
        metadata["station"].to_numpy(str),
        np.asarray(current_values, dtype=float),
        component_array,
    )
    times = anchors[["anchor_id", "anchor_time"]].copy()
    if times["anchor_id"].duplicated().any():
        raise ValueError("anchor metadata must be unique")
    metadata = metadata.merge(times, on="anchor_id", how="left", validate="one_to_one")
    if metadata["anchor_time"].isna().any():
        raise ValueError("router anchor times are incomplete")
    forbidden = {"target_hs", "truth", "label", "anomaly_type"}
    if forbidden.intersection(inputs.columns):
        raise ValueError("future targets leaked into router inputs")
    return inputs, metadata, component_array, np.stack(losses)


class ComponentLossRouter:
    """Small ridge model predicting log case-MSE for three frozen components."""

    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self.model: Pipeline | None = None
        self.columns: list[str] = []
        self.temperature_: float | None = None

    def fit(self, frame: pd.DataFrame, case_losses: np.ndarray) -> ComponentLossRouter:
        forbidden = {"target_hs", "truth", "label", "anomaly_type"}
        leaked = forbidden.intersection(frame.columns)
        if leaked:
            raise ValueError(f"future target columns are forbidden: {sorted(leaked)}")
        losses = np.asarray(case_losses, dtype=float)
        if losses.ndim != 2 or losses.shape != (len(frame), len(COMPONENTS)):
            raise ValueError("case_losses must have shape (cases, 3)")
        if not np.isfinite(losses).all() or np.any(losses < 0.0):
            raise ValueError("case losses must be finite and non-negative")
        if len(frame) < 12:
            raise ValueError("at least 12 past cases are required to fit the router")
        self.columns = list(frame.columns)
        if "station" not in self.columns:
            raise ValueError("station is required")
        numeric = [column for column in self.columns if column != "station"]
        transform = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
                (
                    "station",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ["station"],
                ),
            ]
        )
        self.model = Pipeline([("transform", transform), ("ridge", Ridge(alpha=self.config.alpha))])
        log_loss = np.log(losses + LOSS_FLOOR)
        self.model.fit(frame[self.columns], log_loss)
        span = np.ptp(log_loss, axis=1)
        robust_scale = float(np.median(span[np.isfinite(span)]))
        self.temperature_ = max(0.10, robust_scale * self.config.temperature_multiplier)
        return self

    def predict_weights(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.temperature_ is None:
            raise RuntimeError("router is not fitted")
        forbidden = {"target_hs", "truth", "label", "anomaly_type"}
        leaked = forbidden.intersection(frame.columns)
        if leaked:
            raise ValueError(f"future target columns are forbidden: {sorted(leaked)}")
        predicted = np.asarray(self.model.predict(frame[self.columns]), dtype=float)
        logits = -predicted / self.temperature_
        logits -= np.max(logits, axis=1, keepdims=True)
        adaptive = np.exp(logits)
        adaptive /= np.sum(adaptive, axis=1, keepdims=True)
        incumbent = np.broadcast_to(np.array([0.5, 0.5, 0.0]), adaptive.shape)
        weights = (1.0 - self.config.strength) * incumbent + self.config.strength * adaptive
        if not np.allclose(np.sum(weights, axis=1), 1.0, rtol=0.0, atol=1e-12):
            raise RuntimeError("router weights do not sum to one")
        return weights


def route_case_predictions(components: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(components, dtype=float)
    mixing = np.asarray(weights, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (len(LEADS), len(COMPONENTS)):
        raise ValueError("components must have shape (cases, 6, 3)")
    if mixing.shape != (values.shape[0], len(COMPONENTS)):
        raise ValueError("weights must have shape (cases, 3)")
    if np.any(mixing < 0.0) or not np.allclose(mixing.sum(axis=1), 1.0):
        raise ValueError("weights must be non-negative and sum to one")
    return np.einsum("clm,cm->cl", values, mixing)


def select_router_config(
    fit_features: pd.DataFrame,
    fit_losses: np.ndarray,
    calibration_features: pd.DataFrame,
    calibration_components: np.ndarray,
    calibration_truth: np.ndarray,
    *,
    grid: tuple[RouterConfig, ...] = ROUTER_GRID,
) -> tuple[RouterConfig, list[dict[str, float | str]]]:
    """Choose only on a chronological past calibration block."""

    if calibration_truth.shape != (len(calibration_features), len(LEADS)):
        raise ValueError("calibration_truth must have shape (cases, 6)")
    diagnostics: list[dict[str, float | str]] = []
    for config in grid:
        if config.strength == 0.0:
            weights = np.broadcast_to(
                np.array([0.5, 0.5, 0.0]), (len(calibration_features), len(COMPONENTS))
            )
        else:
            router = ComponentLossRouter(config).fit(fit_features, fit_losses)
            weights = router.predict_weights(calibration_features)
        prediction = route_case_predictions(calibration_components, weights)
        score = rmse(calibration_truth.reshape(-1), prediction.reshape(-1))
        diagnostics.append({"name": config.name, "rmse": score})
    selected = min(
        zip(grid, diagnostics, strict=True),
        key=lambda item: (float(item[1]["rmse"]), item[0].strength, item[0].name),
    )[0]
    return selected, diagnostics


@dataclass(frozen=True)
class PrequentialRouterResult:
    prediction: np.ndarray
    weights: np.ndarray
    selections: tuple[dict[str, Any], ...]


def expand_case_router_rows(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    components: np.ndarray,
    truth: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Expand safe case inputs to lead rows for lead-conditioned routing."""

    cases = len(metadata)
    if len(features) != cases or components.shape != (cases, len(LEADS), len(COMPONENTS)):
        raise ValueError("case router inputs are not aligned")
    if truth.shape != (cases, len(LEADS)):
        raise ValueError("truth must have shape (cases, 6)")
    repeated, row_metadata, row_components = expand_case_router_features(
        features, metadata, components
    )
    row_truth = truth.reshape(-1)
    row_losses = np.square(row_components - row_truth[:, None])
    return repeated, row_metadata, row_components, row_losses


def expand_case_router_features(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    components: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Expand label-free case router inputs without requiring any future truth."""

    cases = len(metadata)
    if len(features) != cases or components.shape != (cases, len(LEADS), len(COMPONENTS)):
        raise ValueError("case router inputs are not aligned")
    repeated = features.iloc[np.repeat(np.arange(cases), len(LEADS))].reset_index(drop=True)
    repeated.insert(1, "lead_h", np.tile(np.asarray(LEADS).astype(str), cases))
    row_metadata = metadata.iloc[np.repeat(np.arange(cases), len(LEADS))].reset_index(drop=True)
    row_metadata["lead_h"] = np.tile(np.asarray(LEADS, dtype=int), cases)
    return repeated, row_metadata, components.reshape(-1, len(COMPONENTS))


def route_row_predictions(components: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(components, dtype=float)
    mixing = np.asarray(weights, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(COMPONENTS):
        raise ValueError("row components must have shape (rows, 3)")
    if mixing.shape != values.shape:
        raise ValueError("row weights must match component shape")
    if np.any(mixing < 0.0) or not np.allclose(mixing.sum(axis=1), 1.0):
        raise ValueError("weights must be non-negative and sum to one")
    return np.sum(values * mixing, axis=1)


def _select_row_router_config(
    fit_features: pd.DataFrame,
    fit_losses: np.ndarray,
    calibration_features: pd.DataFrame,
    calibration_components: np.ndarray,
    calibration_truth: np.ndarray,
    grid: tuple[RouterConfig, ...],
    calibration_active: np.ndarray,
) -> tuple[RouterConfig, list[dict[str, float | str]]]:
    diagnostics: list[dict[str, float | str]] = []
    for config in grid:
        if config.strength == 0.0:
            weights = np.broadcast_to(
                np.array([0.5, 0.5, 0.0]), (len(calibration_features), len(COMPONENTS))
            )
        else:
            router = ComponentLossRouter(config).fit(fit_features, fit_losses)
            weights = router.predict_weights(calibration_features)
        weights = weights.copy()
        weights[~calibration_active] = np.array([0.5, 0.5, 0.0])
        score = rmse(
            calibration_truth,
            route_row_predictions(calibration_components, weights),
        )
        diagnostics.append({"name": config.name, "rmse": score})
    selected = min(
        zip(grid, diagnostics, strict=True),
        key=lambda item: (float(item[1]["rmse"]), item[0].strength, item[0].name),
    )[0]
    return selected, diagnostics


def run_prequential_lead_router(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    components: np.ndarray,
    row_losses: np.ndarray,
    truth: np.ndarray,
    *,
    fold_order: tuple[str, ...],
    grid: tuple[RouterConfig, ...] = ROUTER_GRID,
    active_leads: tuple[int, ...] = LEADS,
) -> PrequentialRouterResult:
    """Lead-conditioned router with exactly the same past-only selection chronology."""

    rows = len(metadata)
    if len(features) != rows or components.shape != (rows, len(COMPONENTS)):
        raise ValueError("lead router inputs are not aligned")
    if row_losses.shape != components.shape or truth.shape != (rows,):
        raise ValueError("lead router targets are not aligned")
    if not grid or grid[0].strength != 0.0:
        raise ValueError("router grid must start with an exact no-op")
    if not active_leads or not set(active_leads).issubset(LEADS):
        raise ValueError("active_leads must be a non-empty subset of official leads")
    output_weights = np.full_like(components, np.nan, dtype=float)
    selections: list[dict[str, Any]] = []
    completed_folds: list[str] = []
    for fold in fold_order:
        current = metadata["fold"].astype(str).eq(fold).to_numpy()
        if not current.any():
            raise ValueError(f"fold has no rows: {fold}")
        past = metadata["fold"].astype(str).isin(completed_folds).to_numpy()
        if not past.any():
            selected = grid[0]
            diagnostics: list[dict[str, float | str]] = []
            output_weights[current] = np.array([0.5, 0.5, 0.0])
            fit_cases = calibration_cases = 0
        else:
            past_metadata = metadata.loc[past]
            past_folds = [name for name in fold_order if name in set(past_metadata["fold"])]
            if len(past_folds) >= 2:
                calibration_fold = past_folds[-1]
                fit = past & ~metadata["fold"].astype(str).eq(calibration_fold).to_numpy()
                calibration = past & metadata["fold"].astype(str).eq(calibration_fold).to_numpy()
            else:
                case_times = (
                    past_metadata[["anchor_id", "anchor_time"]]
                    .drop_duplicates("anchor_id")
                    .sort_values("anchor_time")
                )
                split = max(12, int(np.floor(0.60 * len(case_times))))
                split = min(split, len(case_times) - 6)
                if split < 12:
                    raise ValueError("insufficient chronological past cases for lead router")
                fit_ids = set(case_times.iloc[:split]["anchor_id"])
                calibration_ids = set(case_times.iloc[split:]["anchor_id"])
                fit = past & metadata["anchor_id"].isin(fit_ids).to_numpy()
                calibration = past & metadata["anchor_id"].isin(calibration_ids).to_numpy()
            selected, diagnostics = _select_row_router_config(
                features.loc[fit],
                row_losses[fit],
                features.loc[calibration],
                components[calibration],
                truth[calibration],
                grid,
                metadata.loc[calibration, "lead_h"].isin(active_leads).to_numpy(),
            )
            if selected.strength == 0.0:
                output_weights[current] = np.array([0.5, 0.5, 0.0])
            else:
                router = ComponentLossRouter(selected).fit(features.loc[past], row_losses[past])
                output_weights[current] = router.predict_weights(features.loc[current])
                inactive = current & ~metadata["lead_h"].isin(active_leads).to_numpy()
                output_weights[inactive] = np.array([0.5, 0.5, 0.0])
            fit_cases = int(metadata.loc[fit, "anchor_id"].nunique())
            calibration_cases = int(metadata.loc[calibration, "anchor_id"].nunique())
        selections.append(
            {
                "fold": fold,
                "past_cases": int(metadata.loc[past, "anchor_id"].nunique()),
                "selection_fit_cases": fit_cases,
                "selection_calibration_cases": calibration_cases,
                "selected": selected.name,
                "config": {
                    "alpha": selected.alpha,
                    "temperature_multiplier": selected.temperature_multiplier,
                    "strength": selected.strength,
                },
                "candidates": diagnostics,
                "current_fold_truth_used_for_selection": False,
                "active_leads": list(active_leads),
            }
        )
        completed_folds.append(fold)
    prediction = route_row_predictions(components, output_weights)
    return PrequentialRouterResult(prediction, output_weights, tuple(selections))


def run_prequential_router(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    components: np.ndarray,
    case_losses: np.ndarray,
    truth: np.ndarray,
    *,
    fold_order: tuple[str, ...],
    grid: tuple[RouterConfig, ...] = ROUTER_GRID,
) -> PrequentialRouterResult:
    """Fit each fold's router on earlier cross-fitted cases only."""

    if not grid or grid[0].strength != 0.0:
        raise ValueError("router grid must start with an exact no-op")
    cases = len(metadata)
    if len(features) != cases or components.shape != (cases, len(LEADS), len(COMPONENTS)):
        raise ValueError("router inputs are not aligned")
    if case_losses.shape != (cases, len(COMPONENTS)) or truth.shape != (cases, len(LEADS)):
        raise ValueError("router targets are not aligned")
    unknown = set(metadata["fold"].astype(str)).difference(fold_order)
    if unknown:
        raise ValueError(f"unexpected folds: {sorted(unknown)}")

    output_weights = np.full((cases, len(COMPONENTS)), np.nan, dtype=float)
    selections: list[dict[str, Any]] = []
    completed_folds: list[str] = []
    for fold in fold_order:
        current = metadata["fold"].astype(str).eq(fold).to_numpy()
        if not current.any():
            raise ValueError(f"fold has no cases: {fold}")
        past = metadata["fold"].astype(str).isin(completed_folds).to_numpy()
        if not past.any():
            selected = grid[0]
            diagnostics: list[dict[str, float | str]] = []
            output_weights[current] = np.array([0.5, 0.5, 0.0])
            fit_count = calibration_count = 0
        else:
            past_indices = np.flatnonzero(past)
            past_metadata = metadata.iloc[past_indices]
            past_folds = [name for name in fold_order if name in set(past_metadata["fold"])]
            if len(past_folds) >= 2:
                calibration_fold = past_folds[-1]
                calibration_local = (
                    past_metadata["fold"].astype(str).eq(calibration_fold).to_numpy()
                )
                fit_indices = past_indices[~calibration_local]
                calibration_indices = past_indices[calibration_local]
            else:
                ordered = past_metadata.sort_values("anchor_time").index.to_numpy(dtype=int)
                split = max(12, int(np.floor(0.60 * len(ordered))))
                split = min(split, len(ordered) - 6)
                if split < 12:
                    raise ValueError("insufficient chronological past cases for router selection")
                fit_indices = ordered[:split]
                calibration_indices = ordered[split:]
            selected, diagnostics = select_router_config(
                features.iloc[fit_indices],
                case_losses[fit_indices],
                features.iloc[calibration_indices],
                components[calibration_indices],
                truth[calibration_indices],
                grid=grid,
            )
            if selected.strength == 0.0:
                output_weights[current] = np.array([0.5, 0.5, 0.0])
            else:
                router = ComponentLossRouter(selected).fit(
                    features.iloc[past_indices], case_losses[past_indices]
                )
                output_weights[current] = router.predict_weights(features.loc[current])
            fit_count = int(len(fit_indices))
            calibration_count = int(len(calibration_indices))
        selections.append(
            {
                "fold": fold,
                "past_cases": int(past.sum()),
                "selection_fit_cases": fit_count,
                "selection_calibration_cases": calibration_count,
                "selected": selected.name,
                "config": {
                    "alpha": selected.alpha,
                    "temperature_multiplier": selected.temperature_multiplier,
                    "strength": selected.strength,
                },
                "candidates": diagnostics,
                "current_fold_truth_used_for_selection": False,
            }
        )
        completed_folds.append(fold)
    if not np.isfinite(output_weights).all():
        raise RuntimeError("some router weights were not assigned")
    prediction = route_case_predictions(components, output_weights)
    return PrequentialRouterResult(prediction, output_weights, tuple(selections))
