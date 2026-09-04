"""Official-RMSE-oriented P2 candidate selection and final layer routing."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from p2_restore.features import TARGET_LAYERS, FeatureTable
from p2_restore.model import P2Model, fit_model
from p2_restore.state_conditional import StateConditionalLeanModel, fit_state_conditional_lean

TARGET_RELEVANT_BLOCKS = ("2024_sep_oct", "2025_jul_aug", "2025_nov_dec")
ARMS = ("phase", "state")


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(truth)) ** 2)))


def align_score_oof(phase: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """Align and verify the two saved OOF prediction families."""

    required_phase = {"time", "layer", "truth", "current_blend50", "phase_blend50"}
    required_state = {
        "time",
        "layer",
        "truth",
        "current_blend50",
        "state_blend50",
        "block",
    }
    if missing := required_phase.difference(phase.columns):
        raise ValueError(f"phase OOF is missing columns: {sorted(missing)}")
    if missing := required_state.difference(state.columns):
        raise ValueError(f"state OOF is missing columns: {sorted(missing)}")
    phase_sorted = phase.sort_values(["time", "layer"]).reset_index(drop=True)
    state_sorted = state.sort_values(["time", "layer"]).reset_index(drop=True)
    keys = ["time", "layer"]
    if not phase_sorted[keys].equals(state_sorted[keys]):
        raise ValueError("phase/state OOF keys do not align")
    for column in ("truth", "current_blend50"):
        if not np.array_equal(
            phase_sorted[column].to_numpy(float), state_sorted[column].to_numpy(float)
        ):
            raise ValueError(f"phase/state OOF {column} differs")
    if phase_sorted[keys].duplicated().any():
        raise ValueError("OOF keys are not unique")
    result = state_sorted[["time", "layer", "truth", "current_blend50", "block"]].copy()
    result["phase"] = phase_sorted["phase_blend50"].to_numpy(float)
    result["state"] = state_sorted["state_blend50"].to_numpy(float)
    if not np.isfinite(result[["truth", "current_blend50", "phase", "state"]]).all().all():
        raise ValueError("OOF score columns contain non-finite values")
    return result


def route_predictions(frame: pd.DataFrame, layer_arms: dict[int, str]) -> np.ndarray:
    if set(layer_arms) != set(TARGET_LAYERS) or any(arm not in ARMS for arm in layer_arms.values()):
        raise ValueError("layer router must map layers 2, 3, and 4 to phase/state")
    prediction = np.empty(len(frame), dtype=float)
    assigned = np.zeros(len(frame), dtype=bool)
    for layer in TARGET_LAYERS:
        keep = frame["layer"].to_numpy(int) == layer
        prediction[keep] = frame.loc[keep, layer_arms[layer]].to_numpy(float)
        assigned |= keep
    if not assigned.all():
        raise ValueError("router received a non-target layer")
    return prediction


def _router_name(layer_arms: dict[int, str]) -> str:
    return "_".join(f"l{layer}_{layer_arms[layer]}" for layer in TARGET_LAYERS)


def select_layer_router(
    frame: pd.DataFrame, *, blocks: tuple[str, ...] = TARGET_RELEVANT_BLOCKS
) -> dict[str, object]:
    """Enumerate the exact eight phase/state layer routers on target proxies."""

    relevant = frame["block"].isin(blocks).to_numpy()
    if not relevant.any() or set(frame.loc[relevant, "block"]) != set(blocks):
        raise ValueError("target-relevant proxy blocks are incomplete")
    candidates: list[dict[str, object]] = []
    truth = frame["truth"].to_numpy(float)
    for choices in product(ARMS, repeat=len(TARGET_LAYERS)):
        layer_arms = dict(zip(TARGET_LAYERS, choices, strict=True))
        prediction = route_predictions(frame, layer_arms)
        candidates.append(
            {
                "name": _router_name(layer_arms),
                "layer_arms": {str(key): value for key, value in layer_arms.items()},
                "target_relevant_rmse": _rmse(truth[relevant], prediction[relevant]),
                "all_blocks_rmse": _rmse(truth, prediction),
            }
        )
    candidates.sort(key=lambda row: (row["target_relevant_rmse"], row["name"]))
    return {"selected": candidates[0], "candidates": candidates}


def leave_one_relevant_block_out(frame: pd.DataFrame) -> dict[str, object]:
    """Fit the discrete router on two relevant blocks and score the third."""

    prediction = np.full(len(frame), np.nan, dtype=float)
    selections: dict[str, object] = {}
    for held_out in TARGET_RELEVANT_BLOCKS:
        training_blocks = tuple(block for block in TARGET_RELEVANT_BLOCKS if block != held_out)
        selected = select_layer_router(frame, blocks=training_blocks)["selected"]
        layer_arms = {int(key): value for key, value in selected["layer_arms"].items()}
        held = frame["block"].eq(held_out).to_numpy()
        prediction[held] = route_predictions(frame.loc[held].reset_index(drop=True), layer_arms)
        selections[held_out] = selected
    relevant = frame["block"].isin(TARGET_RELEVANT_BLOCKS).to_numpy()
    if not np.isfinite(prediction[relevant]).all():
        raise AssertionError("leave-one-block-out router left relevant rows unpredicted")
    truth = frame["truth"].to_numpy(float)
    return {
        "rmse": _rmse(truth[relevant], prediction[relevant]),
        "phase_rmse": _rmse(truth[relevant], frame.loc[relevant, "phase"].to_numpy(float)),
        "state_rmse": _rmse(truth[relevant], frame.loc[relevant, "state"].to_numpy(float)),
        "selections": selections,
    }


def score_diagnostics(frame: pd.DataFrame, layer_arms: dict[int, str]) -> dict[str, object]:
    prediction = route_predictions(frame, layer_arms)
    truth = frame["truth"].to_numpy(float)
    relevant = frame["block"].isin(TARGET_RELEVANT_BLOCKS).to_numpy()

    def metric(keep: np.ndarray) -> dict[str, float | int]:
        return {
            "rows": int(keep.sum()),
            "current_rmse": _rmse(truth[keep], frame.loc[keep, "current_blend50"]),
            "phase_rmse": _rmse(truth[keep], frame.loc[keep, "phase"]),
            "state_rmse": _rmse(truth[keep], frame.loc[keep, "state"]),
            "router_rmse": _rmse(truth[keep], prediction[keep]),
        }

    return {
        "all_blocks": metric(np.ones(len(frame), dtype=bool)),
        "target_relevant": metric(relevant),
        "by_block": {
            block: metric(frame["block"].eq(block).to_numpy())
            for block in frame["block"].drop_duplicates()
        },
        "by_layer_target_relevant": {
            str(layer): metric(relevant & frame["layer"].eq(layer).to_numpy())
            for layer in TARGET_LAYERS
        },
    }


@dataclass
class P2ScoreRouterModel:
    base_model: P2Model
    phase_model: P2Model
    state_lean_model: StateConditionalLeanModel
    layer_arms: dict[int, str]

    def predict_components(
        self, base: FeatureTable, lean: FeatureTable, phase: FeatureTable
    ) -> dict[str, np.ndarray]:
        keys = ["station", "layer", "time"]
        if not base.frame[keys].equals(lean.frame[keys]) or not base.frame[keys].equals(
            phase.frame[keys]
        ):
            raise ValueError("score model feature arms are not aligned")
        base_prediction = self.base_model.predict(base)
        phase_prediction = 0.5 * base_prediction + 0.5 * self.phase_model.predict(phase)
        state_prediction = 0.5 * base_prediction + 0.5 * self.state_lean_model.predict(lean)
        frame = pd.DataFrame(
            {
                "layer": base.frame["layer"].to_numpy(int),
                "phase": phase_prediction,
                "state": state_prediction,
            }
        )
        router_prediction = route_predictions(frame, self.layer_arms)
        return {
            "phase": np.clip(phase_prediction, -5.0, 45.0),
            "state": np.clip(state_prediction, -5.0, 45.0),
            "router": np.clip(router_prediction, -5.0, 45.0),
        }


def fit_score_router(
    base: FeatureTable,
    lean: FeatureTable,
    phase: FeatureTable,
    layer_arms: dict[int, str],
    *,
    seed: int = 20260816,
) -> P2ScoreRouterModel:
    keys = ["station", "layer", "time"]
    if not base.frame[keys].equals(lean.frame[keys]) or not base.frame[keys].equals(
        phase.frame[keys]
    ):
        raise ValueError("score model training arms are not aligned")
    return P2ScoreRouterModel(
        base_model=fit_model(base, seed=seed),
        phase_model=fit_model(phase, seed=seed),
        state_lean_model=fit_state_conditional_lean(lean, seed=seed),
        layer_arms=dict(layer_arms),
    )
