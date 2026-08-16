"""Maximum-round convergence screen for the score-oriented P2 router.

The experiment deliberately changes only the boosting horizon.  Every model is
fit once to 5,000 trees and is then evaluated at frozen checkpoints by passing
``num_iteration`` to LightGBM.  This makes the 400-round incumbent an exact
checkpoint of the same fitted booster sequence.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from p2_restore.features import TARGET_LAYERS, FeatureTable
from p2_restore.model import P2Model
from p2_restore.research import STABILITY_BLOCKS
from p2_restore.score_optimization import TARGET_RELEVANT_BLOCKS, route_predictions
from p2_restore.state_conditional import (
    StateConditionalLeanModel,
    compute_state_partition,
    state_weights,
)

MAX_ROUNDS = 5_000
ROUND_CHECKPOINTS = (
    50,
    100,
    150,
    200,
    300,
    400,
    600,
    800,
    1_200,
    1_600,
    2_400,
    3_200,
    4_000,
    5_000,
)
SCORE_LAYER_ARMS = {2: "phase", 3: "phase", 4: "state"}


def _make_estimator(*, seed: int, rounds: int = MAX_ROUNDS):
    """Return the incumbent LightGBM configuration with only rounds changed."""

    if rounds < 1 or rounds > MAX_ROUNDS:
        raise ValueError(f"rounds must be in 1..{MAX_ROUNDS}")
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="regression_l2",
        n_estimators=rounds,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=7,
        min_child_samples=200,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=8,
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )


def fit_max_model(
    table: FeatureTable,
    rows: np.ndarray | None = None,
    *,
    seed: int = 20260816,
    rounds: int = MAX_ROUNDS,
) -> P2Model:
    selected = (
        np.ones(len(table.frame), dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
    )
    if selected.shape != (len(table.frame),) or not selected.any():
        raise ValueError("maximum-round training mask is invalid")
    estimator = _make_estimator(seed=seed, rounds=rounds)
    estimator.fit(
        table.frame.loc[selected, table.feature_columns], table.frame.loc[selected, "residual"]
    )
    return P2Model(estimator, table.feature_columns)


def predict_model_at(model: P2Model, table: FeatureTable, round_number: int) -> np.ndarray:
    if table.feature_columns != model.feature_columns:
        raise ValueError("P2 feature schema differs from maximum-round model")
    if round_number < 1 or round_number > int(model.estimator.n_estimators):
        raise ValueError("requested checkpoint is outside the fitted boosting horizon")
    residual = model.estimator.predict(
        table.frame.loc[:, table.feature_columns], num_iteration=round_number
    )
    return np.clip(table.frame["baseline"].to_numpy(float) + residual, -5.0, 45.0)


@dataclass
class MaxRoundRouterModel:
    base_model: P2Model
    phase_model: P2Model
    state_model: StateConditionalLeanModel
    layer_arms: dict[int, str]
    max_rounds: int = MAX_ROUNDS

    def predict_components_at(
        self,
        base: FeatureTable,
        lean: FeatureTable,
        phase: FeatureTable,
        round_number: int,
    ) -> dict[str, np.ndarray]:
        keys = ["station", "layer", "time"]
        if not base.frame[keys].equals(lean.frame[keys]) or not base.frame[keys].equals(
            phase.frame[keys]
        ):
            raise ValueError("maximum-round feature arms are not aligned")
        base_prediction = predict_model_at(self.base_model, base, round_number)
        phase_prediction = 0.5 * base_prediction + 0.5 * predict_model_at(
            self.phase_model, phase, round_number
        )
        mixed = predict_model_at(self.state_model.mixed_model, lean, round_number)
        stratified = predict_model_at(self.state_model.stratified_model, lean, round_number)
        weight = state_weights(lean, self.state_model.q40, self.state_model.q60)
        state_lean = np.clip((1.0 - weight) * mixed + weight * stratified, -5.0, 45.0)
        state_prediction = 0.5 * base_prediction + 0.5 * state_lean
        routing = pd.DataFrame(
            {
                "layer": base.frame["layer"].to_numpy(int),
                "phase": phase_prediction,
                "state": state_prediction,
            }
        )
        router = route_predictions(routing, self.layer_arms)
        return {
            "phase": np.clip(phase_prediction, -5.0, 45.0),
            "state": np.clip(state_prediction, -5.0, 45.0),
            "router": np.clip(router, -5.0, 45.0),
        }


def fit_max_round_router(
    base: FeatureTable,
    lean: FeatureTable,
    phase: FeatureTable,
    *,
    rows: np.ndarray | None = None,
    seed: int = 20260816,
    rounds: int = MAX_ROUNDS,
    layer_arms: dict[int, str] | None = None,
) -> MaxRoundRouterModel:
    keys = ["station", "layer", "time"]
    if not base.frame[keys].equals(lean.frame[keys]) or not base.frame[keys].equals(
        phase.frame[keys]
    ):
        raise ValueError("maximum-round training arms are not aligned")
    partition = compute_state_partition(lean, rows)
    state = StateConditionalLeanModel(
        mixed_model=fit_max_model(lean, partition.mixed_rows, seed=seed + 101, rounds=rounds),
        stratified_model=fit_max_model(
            lean, partition.stratified_rows, seed=seed + 202, rounds=rounds
        ),
        q40=partition.q40,
        q60=partition.q60,
        mixed_training_rows=int(partition.mixed_rows.sum()),
        stratified_training_rows=int(partition.stratified_rows.sum()),
    )
    return MaxRoundRouterModel(
        base_model=fit_max_model(base, rows, seed=seed, rounds=rounds),
        phase_model=fit_max_model(phase, rows, seed=seed, rounds=rounds),
        state_model=state,
        layer_arms=dict(SCORE_LAYER_ARMS if layer_arms is None else layer_arms),
        max_rounds=rounds,
    )


def _subset(table: FeatureTable, selected: np.ndarray) -> FeatureTable:
    return FeatureTable(table.frame.loc[selected].reset_index(drop=True), table.feature_columns)


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(truth)) ** 2)))


def select_best_round(curve: Sequence[dict[str, object]]) -> int:
    if not curve:
        raise ValueError("round curve is empty")
    checked: list[tuple[float, int]] = []
    for row in curve:
        round_number = int(row["round"])
        rmse = float(row["router_rmse"])
        if round_number < 1 or not np.isfinite(rmse):
            raise ValueError("round curve contains an invalid checkpoint")
        checked.append((rmse, round_number))
    return min(checked)[1]


def run_target_round_screen(
    base: FeatureTable,
    lean: FeatureTable,
    phase: FeatureTable,
    *,
    checkpoints: Sequence[int] = ROUND_CHECKPOINTS,
    max_rounds: int = MAX_ROUNDS,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit max-round models on exactly three target-relevant seasonal blocks."""

    checkpoints = tuple(sorted(set(int(value) for value in checkpoints)))
    if not checkpoints or checkpoints[-1] != max_rounds or 400 not in checkpoints:
        raise ValueError("checkpoints must include the 400-round incumbent and max_rounds")
    if checkpoints[0] < 1 or checkpoints[-1] > max_rounds:
        raise ValueError("checkpoint is outside the fitted range")
    keys = ["station", "layer", "time"]
    if not base.frame[keys].equals(lean.frame[keys]) or not base.frame[keys].equals(
        phase.frame[keys]
    ):
        raise ValueError("maximum-round feature arms are not aligned")
    time = pd.to_datetime(base.frame["time"], utc=True)
    oof_parts: list[pd.DataFrame] = []
    block_reports: dict[str, object] = {}
    relevant = [
        (number, name, STABILITY_BLOCKS[name])
        for number, name in enumerate(STABILITY_BLOCKS)
        if name in TARGET_RELEVANT_BLOCKS
    ]
    for position, (seed_offset, name, (start, stop)) in enumerate(relevant, start=1):
        left = pd.Timestamp(start, tz="Asia/Seoul").tz_convert("UTC")
        right = pd.Timestamp(stop, tz="Asia/Seoul").tz_convert("UTC")
        validation = (time.ge(left) & time.lt(right)).to_numpy()
        if not validation.any():
            raise ValueError(f"target-relevant block has no rows: {name}")
        if progress:
            progress(position, len(relevant), name)
        model = fit_max_round_router(
            base,
            lean,
            phase,
            rows=~validation,
            seed=20260816 + seed_offset,
            rounds=max_rounds,
        )
        base_validation = _subset(base, validation)
        lean_validation = _subset(lean, validation)
        phase_validation = _subset(phase, validation)
        truth = base.frame.loc[validation, "target"].to_numpy(float)
        layer = base.frame.loc[validation, "layer"].to_numpy(int)
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(base.frame.loc[validation, "time"], utc=True)
                .dt.tz_convert("Asia/Seoul")
                .to_numpy(),
                "layer": layer,
                "truth": truth,
                "block": name,
            }
        )
        checkpoint_metrics: dict[str, object] = {}
        for round_number in checkpoints:
            prediction = model.predict_components_at(
                base_validation, lean_validation, phase_validation, round_number
            )
            for arm, values in prediction.items():
                frame[f"{arm}_{round_number}"] = values
            checkpoint_metrics[str(round_number)] = {
                "router_rmse": _rmse(truth, prediction["router"]),
                "phase_rmse": _rmse(truth, prediction["phase"]),
                "state_rmse": _rmse(truth, prediction["state"]),
                "by_layer_router_rmse": {
                    str(target): _rmse(
                        truth[layer == target], prediction["router"][layer == target]
                    )
                    for target in TARGET_LAYERS
                },
            }
        block_reports[name] = {"rows": int(validation.sum()), "checkpoints": checkpoint_metrics}
        oof_parts.append(frame)

    oof = pd.concat(oof_parts, ignore_index=True)
    curve: list[dict[str, object]] = []
    truth = oof["truth"].to_numpy(float)
    layer = oof["layer"].to_numpy(int)
    for round_number in checkpoints:
        row: dict[str, object] = {
            "round": round_number,
            "router_rmse": _rmse(truth, oof[f"router_{round_number}"]),
            "phase_rmse": _rmse(truth, oof[f"phase_{round_number}"]),
            "state_rmse": _rmse(truth, oof[f"state_{round_number}"]),
            "by_layer_router_rmse": {
                str(target): _rmse(
                    truth[layer == target], oof.loc[layer == target, f"router_{round_number}"]
                )
                for target in TARGET_LAYERS
            },
        }
        curve.append(row)
    selected = select_best_round(curve)
    curve_by_round = {int(row["round"]): row for row in curve}
    return (
        {
            "max_rounds": max_rounds,
            "checkpoints": list(checkpoints),
            "rows": len(oof),
            "blocks": block_reports,
            "curve": curve,
            "selected_round": selected,
            "selected_router_rmse": curve_by_round[selected]["router_rmse"],
            "round_400_router_rmse": curve_by_round[400]["router_rmse"],
            "round_5000_router_rmse": curve_by_round[max_rounds]["router_rmse"],
            "delta_selected_minus_400": float(
                curve_by_round[selected]["router_rmse"] - curve_by_round[400]["router_rmse"]
            ),
            "delta_5000_minus_400": float(
                curve_by_round[max_rounds]["router_rmse"] - curve_by_round[400]["router_rmse"]
            ),
            "converged_before_maximum": selected < max_rounds,
        },
        oof,
    )
