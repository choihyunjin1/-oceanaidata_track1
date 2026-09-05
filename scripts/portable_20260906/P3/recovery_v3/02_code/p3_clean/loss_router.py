"""Minimal clean recipe functions extracted without algorithm changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
