"""One-shot mixed/stratified expert for the frozen P2 lean-M2 arm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from p2_restore.features import TARGET_LAYERS, FeatureTable
from p2_restore.model import P2Model, fit_model
from p2_restore.research import STABILITY_BLOCKS


@dataclass(frozen=True)
class StatePartition:
    q40: float
    q60: float
    mixed_rows: np.ndarray
    stratified_rows: np.ndarray


@dataclass
class StateConditionalLeanModel:
    mixed_model: P2Model
    stratified_model: P2Model
    q40: float
    q60: float
    mixed_training_rows: int
    stratified_training_rows: int

    def weights(self, table: FeatureTable) -> np.ndarray:
        return state_weights(table, self.q40, self.q60)

    def predict(self, table: FeatureTable) -> np.ndarray:
        weight = self.weights(table)
        mixed = self.mixed_model.predict(table)
        stratified = self.stratified_model.predict(table)
        return np.clip((1.0 - weight) * mixed + weight * stratified, -5.0, 45.0)


@dataclass
class P2StateBlendModel:
    base_model: P2Model
    state_lean_model: StateConditionalLeanModel
    weight: float = 0.5

    def predict(self, base: FeatureTable, lean: FeatureTable) -> np.ndarray:
        if self.weight != 0.5:
            raise ValueError("the state blend weight is frozen at 0.5")
        return 0.5 * self.base_model.predict(base) + 0.5 * self.state_lean_model.predict(lean)


def _contrast(table: FeatureTable) -> np.ndarray:
    if "temp_1_minus_5" not in table.frame:
        raise ValueError("state conditioning requires public temp_1_minus_5")
    return np.abs(table.frame["temp_1_minus_5"].to_numpy(float))


def compute_state_partition(
    table: FeatureTable,
    rows: np.ndarray | None = None,
    *,
    quantile_low: float = 0.4,
    quantile_high: float = 0.6,
) -> StatePartition:
    """Fit the two thresholds using only the supplied training rows."""

    if (quantile_low, quantile_high) != (0.4, 0.6):
        raise ValueError("the one-shot state quantiles are frozen at 0.4 and 0.6")
    selected = (
        np.ones(len(table.frame), dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
    )
    if selected.shape != (len(table.frame),):
        raise ValueError("state partition row mask is not aligned")
    contrast = _contrast(table)
    finite_train = selected & np.isfinite(contrast)
    if finite_train.sum() < 100:
        raise ValueError("too few finite training rows to fit state thresholds")
    q40, q60 = np.quantile(contrast[finite_train], [quantile_low, quantile_high])
    if not np.isfinite(q40) or not np.isfinite(q60) or q60 <= q40:
        raise ValueError("degenerate public-temperature state thresholds")
    missing = ~np.isfinite(contrast)
    mixed = selected & (missing | (contrast <= q60))
    stratified = selected & (missing | (contrast >= q40))
    if mixed.sum() < 100 or stratified.sum() < 100:
        raise ValueError("a state expert has too few training rows")
    return StatePartition(float(q40), float(q60), mixed, stratified)


def state_weights(table: FeatureTable, q40: float, q60: float) -> np.ndarray:
    if not np.isfinite(q40) or not np.isfinite(q60) or q60 <= q40:
        raise ValueError("state thresholds must be finite and strictly increasing")
    contrast = _contrast(table)
    weight = np.clip((contrast - q40) / (q60 - q40), 0.0, 1.0)
    return np.where(np.isfinite(weight), weight, 0.5)


def fit_state_conditional_lean(
    lean: FeatureTable, rows: np.ndarray | None = None, *, seed: int = 20260816
) -> StateConditionalLeanModel:
    partition = compute_state_partition(lean, rows)
    return StateConditionalLeanModel(
        mixed_model=fit_model(lean, partition.mixed_rows, seed=seed + 101),
        stratified_model=fit_model(lean, partition.stratified_rows, seed=seed + 202),
        q40=partition.q40,
        q60=partition.q60,
        mixed_training_rows=int(partition.mixed_rows.sum()),
        stratified_training_rows=int(partition.stratified_rows.sum()),
    )


def _metric(prediction: np.ndarray, truth: np.ndarray, layer: np.ndarray) -> dict[str, object]:
    error = prediction - truth
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "by_layer": {
            str(target): float(np.sqrt(np.mean(error[layer == target] ** 2)))
            for target in TARGET_LAYERS
        },
    }


def run_state_conditional_stability_screen(
    base: FeatureTable, lean: FeatureTable
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate the exact one-shot state expert on the frozen seasonal blocks."""

    keys = ["station", "layer", "time"]
    if not base.frame[keys].equals(lean.frame[keys]):
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

        def subset(table: FeatureTable, selected: np.ndarray = validation) -> FeatureTable:
            return FeatureTable(
                table.frame.loc[selected].reset_index(drop=True), table.feature_columns
            )

        seed = 20260816 + number
        base_prediction = fit_model(base, train, seed=seed).predict(subset(base))
        lean_prediction = fit_model(lean, train, seed=seed).predict(subset(lean))
        state_model = fit_state_conditional_lean(lean, train, seed=seed)
        lean_validation = subset(lean)
        state_prediction = state_model.predict(lean_validation)
        state_weight = state_model.weights(lean_validation)
        current = 0.5 * base_prediction + 0.5 * lean_prediction
        candidate = 0.5 * base_prediction + 0.5 * state_prediction
        truth = base.frame.loc[validation, "target"].to_numpy(float)
        layer = base.frame.loc[validation, "layer"].to_numpy(int)
        reports[name] = {
            "rows": int(validation.sum()),
            "q40": state_model.q40,
            "q60": state_model.q60,
            "mixed_training_rows": state_model.mixed_training_rows,
            "stratified_training_rows": state_model.stratified_training_rows,
            "current_blend50": _metric(current, truth, layer),
            "state_blend50": _metric(candidate, truth, layer),
            "state_lean": _metric(state_prediction, truth, layer),
            "mean_state_weight": float(np.mean(state_weight)),
        }
        oof_parts.append(
            pd.DataFrame(
                {
                    "time": pd.to_datetime(base.frame.loc[validation, "time"], utc=True)
                    .dt.tz_convert("Asia/Seoul")
                    .to_numpy(),
                    "layer": layer,
                    "truth": truth,
                    "current_blend50": current,
                    "state_blend50": candidate,
                    "state_lean": state_prediction,
                    "state_weight": state_weight,
                    "block": name,
                }
            )
        )
    if not oof_parts:
        raise ValueError("the state screen found no validation rows")
    oof = pd.concat(oof_parts, ignore_index=True)
    oof["day"] = oof["time"].dt.floor("D").astype(str)
    oof["month"] = oof["time"].dt.strftime("%Y-%m")
    return reports, oof


def state_bin_diagnostics(oof: pd.DataFrame) -> dict[str, object]:
    """Describe candidate errors in three fixed, label-blind gate ranges."""

    bins = {
        "mixed_weight_0_025": oof["state_weight"].le(0.25),
        "transition_weight_025_075": oof["state_weight"].gt(0.25) & oof["state_weight"].lt(0.75),
        "stratified_weight_075_1": oof["state_weight"].ge(0.75),
    }
    result: dict[str, object] = {}
    for name, keep in bins.items():
        frame = oof.loc[keep]
        truth = frame["truth"].to_numpy(float)
        current = frame["current_blend50"].to_numpy(float)
        candidate = frame["state_blend50"].to_numpy(float)
        current_rmse = float(np.sqrt(np.mean((current - truth) ** 2)))
        candidate_rmse = float(np.sqrt(np.mean((candidate - truth) ** 2)))
        result[name] = {
            "rows": len(frame),
            "current_rmse": current_rmse,
            "candidate_rmse": candidate_rmse,
            "delta_rmse": candidate_rmse - current_rmse,
        }
    return result


def fit_full_state_blend(base: FeatureTable, lean: FeatureTable) -> P2StateBlendModel:
    keys = ["station", "layer", "time"]
    if not base.frame[keys].equals(lean.frame[keys]):
        raise ValueError("lean dynamics are not aligned to base features")
    return P2StateBlendModel(
        base_model=fit_model(base, seed=20260816),
        state_lean_model=fit_state_conditional_lean(lean, seed=20260816),
    )
