"""Executable nested-chronological P2 architecture reference for Stage A v2.

The module is intentionally imported only after the v2 runner has verified an
independent QA receipt, verified a separate execution authorization, and
consumed the one-shot attempt lock.  It trains every component from scratch at
each prefix; no frozen stack, gate, or official-incumbent weight is reused.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from p2_restore.architecture_matched_stage_a_contract_v2 import (
    CONFIG_RELATIVE,
    CONFIG_SHA256,
    CONTRIBUTORS,
    MODE,
    PIPELINE_SEEDS,
    PREFIX_FRACTIONS,
    canonical_mapping_sha256,
    contained_path,
    exclusive_bytes,
    exclusive_json,
    implementation_pins,
    load_canonical_config,
    sha256_file,
    stage_paths,
    static_preflight,
    verify_consumed_attempt_lock,
    verify_execution_authorization,
    verify_pre_execution_qa,
)
from p2_restore.data import load_p2_data
from p2_restore.deep_data import P2Panel, PanelNormalizer, build_panel, make_chunk_bounds
from p2_restore.deep_models import ConditionalDiffusion, build_model, count_parameters
from p2_restore.deep_training import (
    TrainingConfig,
    _device,
    _materialize_chunks,
    _predict_chunks,
    set_deterministic_seed,
)
from p2_restore.features import FeatureTable, build_training_features
from p2_restore.final_inference import csv_float_roundtrip
from p2_restore.model import fit_model
from p2_restore.profile_projection import (
    project_profiles,
    project_profiles_vectorized,
    public_endpoint_frame,
)
from p2_restore.regime_gate import (
    STATE_FEATURES,
    SoftRegimeGate,
    build_public_state_features,
    fit_simplex_weights,
    fit_soft_gate,
    predict_soft_gate,
)
from p2_restore.research import (
    append_public_dynamics,
    append_public_m2_harmonics,
    select_lean_m2_dynamics,
)
from p2_restore.score_optimization import P2ScoreRouterModel
from p2_restore.state_conditional import fit_state_conditional_lean

Progress = Callable[[dict[str, Any]], None]
TARGET_LAYERS = (2, 3, 4)
KST = ZoneInfo("Asia/Seoul")


@dataclass(frozen=True)
class DeepSelection:
    component: str
    seed: int
    best_epoch: int
    best_rmse: float
    parameter_count: int
    history: tuple[dict[str, float | int], ...]
    prediction: np.ndarray


@dataclass(frozen=True)
class DeepFixedFit:
    component: str
    seed: int
    epochs: int
    parameter_count: int
    final_train_mse_c: float
    prediction: np.ndarray


@dataclass(frozen=True)
class RouterContext:
    base: FeatureTable
    lean: FeatureTable
    phase: FeatureTable
    public_state: pd.DataFrame
    joint_rows: np.ndarray
    times: pd.DatetimeIndex


def _emit(progress: Progress | None, **payload: Any) -> None:
    if progress is not None:
        progress(payload)


def _now_kst() -> str:
    return datetime.now(KST).isoformat()


def _fraction_token(fraction: float) -> str:
    mapping = {0.4: "040", 0.55: "055", 0.7: "070", 0.85: "085", 1.0: "100"}
    try:
        return mapping[float(fraction)]
    except KeyError as exc:
        raise ValueError("unregistered prefix fraction") from exc


def _derived_seed(base_seed: int, *labels: object) -> int:
    text = "|".join((str(base_seed), *(str(label) for label in labels)))
    value = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    return value % 2_147_483_646 + 1


def build_execution_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete fit graph without touching data or importing a runner."""

    recipe = config["training_recipe"]
    folds = [fold["name"] for fold in recipe["outer_folds"]]
    fractions = list(recipe["prefix_fractions"])
    seeds = list(recipe["complete_pipeline_seed_ids"])
    components = list(recipe["deep_training"]["components"])
    inner_splits = len(recipe["inner_oof"]["validation_blocks"])
    cells = len(folds) * len(fractions)
    deep_jobs = cells * len(seeds) * len(components) * (inner_splits + 1)
    router_jobs = cells * len(seeds) * (inner_splits + 1)
    return {
        "schema_version": "p2_architecture_matched_stage_a_execution.plan.v2",
        "problem": "P2",
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "folds": folds,
        "prefix_fractions": fractions,
        "complete_pipeline_seeds": seeds,
        "deep_components": components,
        "inner_splits_per_cell": inner_splits,
        "outer_prefix_cells": cells,
        "deep_training_jobs": deep_jobs,
        "router_training_jobs": router_jobs,
        "challenger_jobs": 0,
        "full_fit_jobs": 0,
        "submission_predictions": 0,
        "uploads": 0,
    }


def _utc(value: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise ValueError("all fold boundaries must be timezone-aware")
    return parsed.tz_convert("UTC")


def _time_mask(times: pd.DatetimeIndex, selected: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(times.isin(selected), dtype=bool)


def _official_weighted_rmse(
    truth: np.ndarray,
    prediction: np.ndarray,
    layer: np.ndarray,
    counts: Mapping[str, int],
) -> float:
    target = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    layers = np.asarray(layer, dtype=int)
    if target.shape != estimate.shape or target.shape != layers.shape or not len(target):
        raise ValueError("metric vectors are empty or misaligned")
    weighted = 0.0
    total_weight = 0
    for current in TARGET_LAYERS:
        keep = layers == current
        if not keep.any():
            raise ValueError(f"metric has no rows for target layer {current}")
        mse = float(np.mean((estimate[keep] - target[keep]) ** 2))
        weight = int(counts[str(current)])
        weighted += weight * mse
        total_weight += weight
    return float(np.sqrt(weighted / total_weight))


def _joint_masked_panel(observations: pd.DataFrame) -> P2Panel:
    panel = build_panel(observations)
    selected = observations.loc[
        observations["layer"].isin(TARGET_LAYERS),
        ["time", "layer", "temp", "psal"],
    ].copy()
    selected["time"] = pd.to_datetime(selected["time"], utc=True)
    if selected.duplicated(["time", "layer"]).any():
        raise ValueError("target-layer time/layer keys are not unique")
    temp = selected.pivot(index="time", columns="layer", values="temp").reindex(panel.times)
    psal = selected.pivot(index="time", columns="layer", values="psal").reindex(panel.times)
    joint = np.column_stack(
        [
            np.isfinite(temp.get(layer, pd.Series(index=panel.times, dtype=float)))
            & np.isfinite(psal.get(layer, pd.Series(index=panel.times, dtype=float)))
            for layer in TARGET_LAYERS
        ]
    )
    effective = panel.target_mask & joint
    if not effective.any() or np.any(effective & ~panel.target_mask):
        raise ValueError("joint temp+psal target mask is invalid")
    return replace(panel, target_mask=effective)


def _build_router_context(observations: pd.DataFrame) -> RouterContext:
    base = build_training_features(observations)
    dynamic = append_public_dynamics(base, observations)
    lean = select_lean_m2_dynamics(base, dynamic)
    phase = append_public_m2_harmonics(lean, observations)
    keys = ["station", "layer", "time"]
    if not base.frame[keys].equals(lean.frame[keys]) or not base.frame[keys].equals(
        phase.frame[keys]
    ):
        raise ValueError("router feature arms are not key-aligned")
    availability = observations.loc[
        observations["layer"].isin(TARGET_LAYERS),
        ["station", "layer", "time", "temp", "psal"],
    ].copy()
    availability["_time"] = pd.to_datetime(availability["time"], utc=True)
    availability["_joint"] = np.isfinite(availability["temp"]) & np.isfinite(
        availability["psal"]
    )
    if availability.duplicated(["station", "layer", "_time"]).any():
        raise ValueError("router joint-mask keys are duplicated")
    keyed = base.frame.loc[:, keys].copy()
    keyed["_row"] = np.arange(len(keyed))
    keyed["_time"] = pd.to_datetime(keyed["time"], utc=True)
    merged = keyed.merge(
        availability.loc[:, ["station", "layer", "_time", "_joint"]],
        on=["station", "layer", "_time"],
        how="left",
        validate="one_to_one",
    ).sort_values("_row")
    joint_rows = merged["_joint"].fillna(False).to_numpy(bool)
    times = pd.DatetimeIndex(pd.to_datetime(base.frame["time"], utc=True))
    public_state = build_public_state_features(observations, base.frame[["time", "layer"]])
    if len(public_state) != len(base.frame):
        raise ValueError("public-state features are not router-row aligned")
    return RouterContext(base, lean, phase, public_state, joint_rows, times)


def _subset(table: FeatureTable, rows: np.ndarray) -> FeatureTable:
    selected = np.asarray(rows, dtype=bool)
    if selected.shape != (len(table.frame),):
        raise ValueError("feature-table row mask is misaligned")
    return FeatureTable(table.frame.loc[selected].reset_index(drop=True), table.feature_columns)


def _router_rows(context: RouterContext, times: pd.DatetimeIndex) -> np.ndarray:
    return context.joint_rows & np.asarray(context.times.isin(times), dtype=bool)


def _fit_predict_router(
    context: RouterContext,
    *,
    train_times: pd.DatetimeIndex,
    prediction_times: pd.DatetimeIndex,
    seed: int,
    layer_arms: Mapping[str, str],
) -> tuple[pd.DataFrame, np.ndarray]:
    train_rows = _router_rows(context, train_times)
    prediction_rows = _router_rows(context, prediction_times)
    if train_rows.sum() < 300 or prediction_rows.sum() < 3:
        raise ValueError("router train/prediction split is too small")
    arms = {int(layer): str(arm) for layer, arm in layer_arms.items()}
    fitted = P2ScoreRouterModel(
        base_model=fit_model(context.base, train_rows, seed=seed),
        phase_model=fit_model(context.phase, train_rows, seed=seed),
        state_lean_model=fit_state_conditional_lean(context.lean, train_rows, seed=seed),
        layer_arms=arms,
    )
    base = _subset(context.base, prediction_rows)
    lean = _subset(context.lean, prediction_rows)
    phase = _subset(context.phase, prediction_rows)
    prediction = fitted.predict_components(base, lean, phase)["router"]
    frame = base.frame.loc[:, ["station", "layer", "time", "target"]].rename(
        columns={"target": "truth"}
    )
    frame["router_400"] = csv_float_roundtrip(prediction)
    positions = np.flatnonzero(prediction_rows)
    state = context.public_state.iloc[positions].reset_index(drop=True)
    if not frame[["time", "layer"]].equals(state[["time", "layer"]]):
        left = frame[["time", "layer"]].copy()
        right = state[["time", "layer"]].copy()
        left["time"] = pd.to_datetime(left["time"], utc=True)
        right["time"] = pd.to_datetime(right["time"], utc=True)
        if not left.equals(right):
            raise ValueError("router and public-state keys are not aligned")
    frame = pd.concat(
        [
            frame.reset_index(drop=True),
            state.loc[:, STATE_FEATURES].reset_index(drop=True),
        ],
        axis=1,
    )
    return frame, prediction_rows


def _deep_config(
    recipe: Mapping[str, Any], component: str, seed: int, *, max_epochs: int
) -> TrainingConfig:
    common = recipe["deep_training"]
    current = common["components"][component]
    return TrainingConfig(
        model=component,
        learning_rate=float(current["learning_rate"]),
        weight_decay=float(current["weight_decay"]),
        max_epochs=int(max_epochs),
        patience=max_epochs,
        chunk_length=int(common["chunk_length"]),
        chunk_stride=int(common["chunk_stride"]),
        batch_size=int(common["batch_size"]),
        seed=int(seed),
        evaluation_interval=1,
        diffusion_samples=int(common["diffusion_samples"]),
    )


def _prepare_deep_training(
    panel: P2Panel,
    *,
    train_times: np.ndarray,
    config: TrainingConfig,
    minimum_values: int,
) -> tuple[
    PanelNormalizer,
    np.ndarray,
    tuple[tuple[int, int], ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    selected = np.asarray(train_times, dtype=bool)
    if selected.shape != (len(panel.times),) or not selected.any():
        raise ValueError("deep training time mask is invalid")
    normalizer = PanelNormalizer.fit(panel, selected)
    inputs = normalizer.transform_inputs(panel.inputs)
    target, target_mask = normalizer.transform_targets(panel)
    training_mask = target_mask & selected[:, None]
    all_bounds = make_chunk_bounds(
        panel.segment_ids,
        length=config.chunk_length,
        stride=config.chunk_stride,
    )
    train_bounds = tuple(
        bound
        for bound in all_bounds
        if training_mask[bound[0] : bound[1]].sum() >= minimum_values
    )
    if not train_bounds:
        raise RuntimeError("no joint-mask supervised chunks are available")
    chunk_x, chunk_y, chunk_mask = _materialize_chunks(
        inputs,
        target,
        training_mask,
        train_bounds,
        config.chunk_length,
    )
    return normalizer, inputs, all_bounds, chunk_x, chunk_y, chunk_mask


def _train_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    chunk_x: torch.Tensor,
    chunk_y: torch.Tensor,
    chunk_mask: torch.Tensor,
    order: np.ndarray,
    config: TrainingConfig,
    layer_weights: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    weight_sum = 0.0
    for begin in range(0, len(order), config.batch_size):
        ids = torch.from_numpy(order[begin : begin + config.batch_size]).long()
        x = chunk_x[ids].to(device, non_blocking=True)
        y = chunk_y[ids].to(device, non_blocking=True)
        mask = chunk_mask[ids].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if isinstance(model, ConditionalDiffusion):
                loss = model.training_loss(x, y, mask, layer_weights)
            else:
                predicted = model(x)
                squared = (predicted - y).square() * layer_weights.view(1, 1, 3)
                loss = (squared * mask).sum() / mask.sum().clamp_min(1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        weight = float(mask.sum().item())
        loss_sum += float(loss.detach().item()) * weight
        weight_sum += weight
    return loss_sum / max(weight_sum, 1.0)


def _panel_metric(
    panel: P2Panel,
    prediction: np.ndarray,
    selected_times: np.ndarray,
    counts: Mapping[str, int],
) -> float:
    truths: list[np.ndarray] = []
    estimates: list[np.ndarray] = []
    layers: list[np.ndarray] = []
    for offset, layer in enumerate(TARGET_LAYERS):
        keep = selected_times & panel.target_mask[:, offset]
        truths.append(panel.target[keep, offset])
        estimates.append(prediction[keep, offset])
        layers.append(np.full(int(keep.sum()), layer, dtype=int))
    return _official_weighted_rmse(
        np.concatenate(truths),
        np.concatenate(estimates),
        np.concatenate(layers),
        counts,
    )


def _train_select_deep(
    panel: P2Panel,
    *,
    optimization_times: pd.DatetimeIndex,
    calibration_times: pd.DatetimeIndex,
    component: str,
    seed: int,
    recipe: Mapping[str, Any],
    progress: Progress | None,
    context: Mapping[str, Any],
) -> DeepSelection:
    grid = tuple(int(epoch) for epoch in recipe["epoch_selection"]["epoch_grid"])
    config = _deep_config(recipe, component, seed, max_epochs=max(grid))
    set_deterministic_seed(seed)
    optimization_mask = _time_mask(panel.times, optimization_times)
    calibration_mask = _time_mask(panel.times, calibration_times)
    minimum = int(recipe["deep_training"]["minimum_supervised_values_per_chunk"])
    normalizer, inputs, all_bounds, chunk_x, chunk_y, chunk_mask = _prepare_deep_training(
        panel,
        train_times=optimization_mask,
        config=config,
        minimum_values=minimum,
    )
    device = _device()
    model = build_model(config.model, inputs.shape[1]).to(device)
    parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs,
        eta_min=config.learning_rate * 0.05,
    )
    layer_weights = torch.tensor(
        normalizer.residual_scale**2,
        device=device,
        dtype=torch.float32,
    )
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.max_epochs + 1):
        train_mse = _train_epoch(
            model=model,
            optimizer=optimizer,
            chunk_x=chunk_x,
            chunk_y=chunk_y,
            chunk_mask=chunk_mask,
            order=rng.permutation(len(chunk_x)),
            config=config,
            layer_weights=layer_weights,
            device=device,
        )
        scheduler.step()
        if epoch not in grid:
            continue
        normalized = _predict_chunks(
            model,
            inputs,
            all_bounds,
            length=config.chunk_length,
            batch_size=config.batch_size,
            diffusion_samples=config.diffusion_samples,
            seed=seed + epoch,
        )
        physical = normalizer.inverse_predictions(panel, normalized)
        score = _panel_metric(
            panel,
            physical,
            calibration_mask,
            recipe["metric"]["official_layer_counts"],
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_mse_c": float(train_mse),
            "calibration_rmse_c": float(score),
        }
        history.append(record)
        _emit(
            progress,
            event="deep_epoch_grid_score",
            component=component,
            epoch=epoch,
            max_epoch=config.max_epochs,
            **context,
        )
        if score < best_score - 1e-12:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None or best_epoch not in grid or not np.isfinite(best_score):
        raise RuntimeError("deep checkpoint selection did not produce a finite result")
    model.load_state_dict(best_state)
    normalized = _predict_chunks(
        model,
        inputs,
        all_bounds,
        length=config.chunk_length,
        batch_size=config.batch_size,
        diffusion_samples=config.diffusion_samples,
        seed=seed + best_epoch,
    )
    physical = normalizer.inverse_predictions(panel, normalized)
    result = DeepSelection(
        component=component,
        seed=seed,
        best_epoch=best_epoch,
        best_rmse=float(best_score),
        parameter_count=parameters,
        history=tuple(history),
        prediction=physical,
    )
    del model, optimizer, scheduler, best_state
    torch.cuda.empty_cache()
    return result


def _train_fixed_deep(
    panel: P2Panel,
    *,
    train_times: pd.DatetimeIndex,
    component: str,
    epochs: int,
    seed: int,
    recipe: Mapping[str, Any],
    progress: Progress | None,
    context: Mapping[str, Any],
) -> DeepFixedFit:
    config = _deep_config(recipe, component, seed, max_epochs=epochs)
    set_deterministic_seed(seed)
    train_mask = _time_mask(panel.times, train_times)
    minimum = int(recipe["deep_training"]["minimum_supervised_values_per_chunk"])
    normalizer, inputs, all_bounds, chunk_x, chunk_y, chunk_mask = _prepare_deep_training(
        panel,
        train_times=train_mask,
        config=config,
        minimum_values=minimum,
    )
    device = _device()
    model = build_model(config.model, inputs.shape[1]).to(device)
    parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=config.learning_rate * 0.05,
    )
    layer_weights = torch.tensor(
        normalizer.residual_scale**2,
        device=device,
        dtype=torch.float32,
    )
    rng = np.random.default_rng(seed)
    final_mse = float("nan")
    for epoch in range(1, epochs + 1):
        final_mse = _train_epoch(
            model=model,
            optimizer=optimizer,
            chunk_x=chunk_x,
            chunk_y=chunk_y,
            chunk_mask=chunk_mask,
            order=rng.permutation(len(chunk_x)),
            config=config,
            layer_weights=layer_weights,
            device=device,
        )
        scheduler.step()
        if epoch == 1 or epoch == epochs or epoch % 4 == 0:
            _emit(
                progress,
                event="deep_outer_refit_epoch",
                component=component,
                epoch=epoch,
                max_epoch=epochs,
                **context,
            )
    normalized = _predict_chunks(
        model,
        inputs,
        all_bounds,
        length=config.chunk_length,
        batch_size=config.batch_size,
        diffusion_samples=config.diffusion_samples,
        seed=seed + epochs,
    )
    physical = normalizer.inverse_predictions(panel, normalized)
    result = DeepFixedFit(
        component=component,
        seed=seed,
        epochs=epochs,
        parameter_count=parameters,
        final_train_mse_c=float(final_mse),
        prediction=physical,
    )
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return result


def _deep_on_router_rows(
    panel: P2Panel,
    prediction: np.ndarray,
    frame: pd.DataFrame,
) -> np.ndarray:
    positions = panel.times.get_indexer(pd.to_datetime(frame["time"], utc=True))
    layers = frame["layer"].to_numpy(int)
    if (positions < 0).any() or not set(layers).issubset(TARGET_LAYERS):
        raise ValueError("deep prediction keys are absent from the panel")
    values = prediction[positions, layers - 2]
    if not np.isfinite(values).all():
        raise ValueError("deep prediction contains non-finite values")
    return values


def _prefix_times(
    panel: P2Panel,
    *,
    outer_start: pd.Timestamp,
    embargo_days: int,
    fraction: float,
) -> pd.DatetimeIndex:
    eligible = panel.times[
        (panel.times < outer_start - pd.Timedelta(days=embargo_days))
        & panel.target_mask.any(axis=1)
    ]
    eligible = pd.DatetimeIndex(eligible.unique()).sort_values()
    count = int(math.ceil(len(eligible) * fraction))
    if count < 1 or count > len(eligible):
        raise ValueError("prefix timestamp count is invalid")
    return eligible[:count]


def _inner_splits(
    prefix: pd.DatetimeIndex,
    *,
    edges: Sequence[float],
    calibration_fraction: float,
    embargo_days: int,
) -> list[dict[str, Any]]:
    indices = [int(math.floor(len(prefix) * float(edge))) for edge in edges[:-1]] + [
        len(prefix)
    ]
    if indices != sorted(indices) or len(set(indices)) != len(indices):
        raise ValueError("inner chronological edge indices are degenerate")
    result: list[dict[str, Any]] = []
    for number, (start, stop) in enumerate(zip(indices[:-1], indices[1:], strict=True), 1):
        validation = prefix[start:stop]
        if not len(validation):
            raise ValueError("inner validation block is empty")
        train_cutoff = validation[0] - pd.Timedelta(days=embargo_days)
        inner_train = prefix[prefix < train_cutoff]
        calibration_count = int(math.ceil(len(inner_train) * calibration_fraction))
        if calibration_count < 1 or calibration_count >= len(inner_train):
            raise ValueError("inner calibration block is degenerate")
        calibration = inner_train[-calibration_count:]
        optimization_cutoff = calibration[0] - pd.Timedelta(days=embargo_days)
        optimization = inner_train[inner_train < optimization_cutoff]
        if not len(optimization):
            raise ValueError("inner optimization block is empty after embargo")
        result.append(
            {
                "name": f"inner_{number}",
                "validation": validation,
                "inner_train": inner_train,
                "calibration": calibration,
                "optimization": optimization,
            }
        )
    return result


def _fit_stack(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    weights: dict[str, dict[str, float]] = {}
    for layer in TARGET_LAYERS:
        keep = frame["layer"].to_numpy(int) == layer
        vector = fit_simplex_weights(
            frame.loc[keep, CONTRIBUTORS].to_numpy(float),
            frame.loc[keep, "truth"].to_numpy(float),
        )
        weights[str(layer)] = {
            name: float(value) for name, value in zip(CONTRIBUTORS, vector, strict=True)
        }
    return weights


def _weighted_stack(
    frame: pd.DataFrame, weights: Mapping[str, Mapping[str, float]]
) -> np.ndarray:
    result = np.full(len(frame), np.nan, dtype=np.float64)
    for layer in TARGET_LAYERS:
        keep = frame["layer"].to_numpy(int) == layer
        vector = np.array([weights[str(layer)][name] for name in CONTRIBUTORS])
        result[keep] = frame.loc[keep, CONTRIBUTORS].to_numpy(float) @ vector
    if not np.isfinite(result).all():
        raise ValueError("layer stack produced non-finite predictions")
    return result


def _gate_receipt(gate: SoftRegimeGate) -> dict[str, Any]:
    return {
        "feature_names": list(gate.feature_names),
        "prediction_columns": list(gate.prediction_columns),
        "regularization": float(gate.regularization),
        "layers": {
            str(layer): {
                "prior": [float(value) for value in fitted.prior],
                "coefficient_sha256": hashlib.sha256(
                    np.asarray(fitted.coefficients, dtype="<f8").tobytes()
                ).hexdigest(),
                "optimizer_iterations": int(fitted.optimizer_iterations),
                "objective_mse": float(fitted.objective_mse),
            }
            for layer, fitted in sorted(gate.layers.items())
        },
    }


def _compose_prediction(
    frame: pd.DataFrame,
    *,
    endpoints: pd.DataFrame,
    stack_weights: Mapping[str, Mapping[str, float]],
    gate: SoftRegimeGate,
    layer_factors: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, int | float]]:
    component_frame = frame.copy()
    component_frame["router_400"] = csv_float_roundtrip(component_frame["router_400"])
    deep_stack = csv_float_roundtrip(_weighted_stack(component_frame, stack_weights))
    keys = component_frame[["station", "time", "layer"]]
    base_raw = project_profiles(keys, deep_stack, endpoints)
    base_prediction = csv_float_roundtrip(base_raw.prediction)
    raw_gate = csv_float_roundtrip(predict_soft_gate(gate, component_frame))
    routed_input = base_prediction.copy()
    use_gate = component_frame["layer"].isin((2, 4)).to_numpy()
    routed_input[use_gate] = raw_gate[use_gate]
    routed = project_profiles_vectorized(keys, routed_input, endpoints).prediction
    layers = component_frame["layer"].to_numpy(int)
    scale = np.array([float(layer_factors[str(layer)]) for layer in layers])
    final_input = base_prediction + scale * (routed - base_prediction)
    final_projection = project_profiles_vectorized(keys, final_input, endpoints)
    final = csv_float_roundtrip(final_projection.prediction)
    if not np.isfinite(final).all():
        raise ValueError("final architecture-matched prediction is non-finite")
    return final, {
        "rows": int(len(final)),
        "deep_projection_active_rows": int(base_raw.active_mask.sum()),
        "soft_route_rows": int(use_gate.sum()),
        "final_projection_active_rows": int(final_projection.active_mask.sum()),
        "minimum": float(final.min()),
        "maximum": float(final.max()),
    }


def _run_cell_seed(
    *,
    panel: P2Panel,
    router: RouterContext,
    endpoints: pd.DataFrame,
    recipe: Mapping[str, Any],
    layer_factors: Mapping[str, float],
    fold: Mapping[str, Any],
    fraction: float,
    pipeline_seed: int,
    progress: Progress | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_name = str(fold["name"])
    outer_start = _utc(str(fold["start_kst"]))
    outer_stop = _utc(str(fold["stop_kst"]))
    embargo_days = int(recipe["embargo_days"])
    prefix = _prefix_times(
        panel,
        outer_start=outer_start,
        embargo_days=embargo_days,
        fraction=fraction,
    )
    outer_times = panel.times[(panel.times >= outer_start) & (panel.times < outer_stop)]
    splits = _inner_splits(
        prefix,
        edges=recipe["inner_oof"]["validation_fraction_edges"],
        calibration_fraction=float(recipe["epoch_selection"]["calibration_fraction"]),
        embargo_days=embargo_days,
    )
    inner_frames: list[pd.DataFrame] = []
    selected_epochs: dict[str, list[int]] = {
        component: [] for component in recipe["deep_training"]["components"]
    }
    inner_receipts: list[dict[str, Any]] = []
    for split in splits:
        context = {
            "fold": fold_name,
            "fraction": fraction,
            "pipeline_seed": pipeline_seed,
            "split": split["name"],
        }
        deep_results: dict[str, DeepSelection] = {}
        for component in recipe["deep_training"]["components"]:
            component_seed = _derived_seed(
                pipeline_seed,
                fold_name,
                _fraction_token(fraction),
                split["name"],
                component,
            )
            result = _train_select_deep(
                panel,
                optimization_times=split["optimization"],
                calibration_times=split["calibration"],
                component=component,
                seed=component_seed,
                recipe=recipe,
                progress=progress,
                context=context,
            )
            deep_results[component] = result
            selected_epochs[component].append(result.best_epoch)
        router_seed = _derived_seed(
            pipeline_seed,
            fold_name,
            _fraction_token(fraction),
            split["name"],
            "router_400",
        )
        frame, _ = _fit_predict_router(
            router,
            train_times=split["inner_train"],
            prediction_times=split["validation"],
            seed=router_seed,
            layer_arms=recipe["router_training"]["layer_arms"],
        )
        for component, result in deep_results.items():
            frame[component] = _deep_on_router_rows(panel, result.prediction, frame)
        inner_frames.append(frame)
        inner_receipts.append(
            {
                "split": split["name"],
                "optimization_timestamp_count": len(split["optimization"]),
                "calibration_timestamp_count": len(split["calibration"]),
                "inner_train_timestamp_count": len(split["inner_train"]),
                "validation_timestamp_count": len(split["validation"]),
                "validation_row_count": len(frame),
                "selected_epoch": {
                    component: result.best_epoch
                    for component, result in deep_results.items()
                },
                "calibration_rmse_c": {
                    component: result.best_rmse
                    for component, result in deep_results.items()
                },
                "router_seed": router_seed,
            }
        )
        _emit(progress, event="inner_split_complete", **context)
    inner = pd.concat(inner_frames, ignore_index=True)
    if inner.duplicated(["station", "layer", "time"]).any():
        raise ValueError("inner component OOF keys are duplicated")
    stack_weights = _fit_stack(inner)
    gate = fit_soft_gate(
        inner,
        prediction_columns=CONTRIBUTORS,
        regularization=float(recipe["meta_training"]["gate_regularization"]),
    )
    outer_epochs = {
        component: sorted(epochs)[len(epochs) // 2]
        for component, epochs in selected_epochs.items()
    }
    outer_deep: dict[str, DeepFixedFit] = {}
    for component, epochs in outer_epochs.items():
        component_seed = _derived_seed(
            pipeline_seed,
            fold_name,
            _fraction_token(fraction),
            "outer_refit",
            component,
        )
        outer_deep[component] = _train_fixed_deep(
            panel,
            train_times=prefix,
            component=component,
            epochs=epochs,
            seed=component_seed,
            recipe=recipe,
            progress=progress,
            context={
                "fold": fold_name,
                "fraction": fraction,
                "pipeline_seed": pipeline_seed,
                "split": "outer_refit",
            },
        )
    router_seed = _derived_seed(
        pipeline_seed,
        fold_name,
        _fraction_token(fraction),
        "outer_refit",
        "router_400",
    )
    outer_frame, _ = _fit_predict_router(
        router,
        train_times=prefix,
        prediction_times=outer_times,
        seed=router_seed,
        layer_arms=recipe["router_training"]["layer_arms"],
    )
    for component, result in outer_deep.items():
        outer_frame[component] = _deep_on_router_rows(panel, result.prediction, outer_frame)
    prediction, diagnostics = _compose_prediction(
        outer_frame,
        endpoints=endpoints,
        stack_weights=stack_weights,
        gate=gate,
        layer_factors=layer_factors,
    )
    result_frame = outer_frame.loc[:, ["station", "layer", "time", "truth"]].copy()
    result_frame.insert(0, "fold", fold_name)
    result_frame["prediction"] = prediction
    receipt = {
        "fold": fold_name,
        "fraction": fraction,
        "pipeline_seed": pipeline_seed,
        "prefix_timestamp_count": len(prefix),
        "outer_timestamp_count": len(outer_times),
        "outer_row_count": len(result_frame),
        "inner_splits": inner_receipts,
        "outer_selected_epochs": outer_epochs,
        "outer_router_seed": router_seed,
        "stack_weights": stack_weights,
        "gate": _gate_receipt(gate),
        "postprocess_diagnostics": diagnostics,
        "guards": {
            "joint_temp_psal_mask_applied": True,
            "outer_labels_used_for_fit": False,
            "future_target_labels_used_for_fit": False,
            "frozen_stack_reused": False,
            "frozen_gate_reused": False,
        },
    }
    _emit(
        progress,
        event="outer_cell_seed_complete",
        fold=fold_name,
        fraction=fraction,
        pipeline_seed=pipeline_seed,
    )
    return result_frame, receipt


def _curve_metric(
    frame: pd.DataFrame,
    prediction_column: str,
    counts: Mapping[str, int],
) -> tuple[float, dict[str, float], dict[str, float]]:
    fold_mse: list[float] = []
    by_fold: dict[str, float] = {}
    for fold, current in frame.groupby("fold", sort=False):
        score = _official_weighted_rmse(
            current["truth"].to_numpy(float),
            current[prediction_column].to_numpy(float),
            current["layer"].to_numpy(int),
            counts,
        )
        by_fold[str(fold)] = score
        fold_mse.append(score**2)
    aggregate = float(np.sqrt(np.mean(fold_mse)))
    by_layer = {
        str(layer): float(
            np.sqrt(
                np.mean(
                    (
                        frame.loc[frame["layer"].eq(layer), prediction_column].to_numpy(float)
                        - frame.loc[frame["layer"].eq(layer), "truth"].to_numpy(float)
                    )
                    ** 2
                )
            )
        )
        for layer in TARGET_LAYERS
    }
    return aggregate, by_fold, by_layer


def _merge_seed_predictions(
    by_seed: Mapping[int, pd.DataFrame], counts: Mapping[str, int]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["fold", "station", "layer", "time"]
    ordered_seeds = list(PIPELINE_SEEDS)
    reference = by_seed[ordered_seeds[0]].sort_values(keys).reset_index(drop=True)
    output = reference.loc[:, [*keys, "truth"]].copy()
    seed_metrics: list[float] = []
    for seed in ordered_seeds:
        current = by_seed[seed].sort_values(keys).reset_index(drop=True)
        if not reference[keys].equals(current[keys]) or not np.array_equal(
            reference["truth"].to_numpy(float), current["truth"].to_numpy(float)
        ):
            raise ValueError("three-seed outer OOF keys or truths differ")
        column = f"seed_{seed}"
        output[column] = current["prediction"].to_numpy(float)
        metric, _, _ = _curve_metric(
            current.rename(columns={"prediction": column}),
            column,
            counts,
        )
        seed_metrics.append(metric)
    seed_columns = [f"seed_{seed}" for seed in ordered_seeds]
    output["prediction_mean"] = output[seed_columns].to_numpy(float).mean(axis=1)
    metric, by_fold, by_layer = _curve_metric(output, "prediction_mean", counts)
    aggregate = {
        "rows": len(output),
        "prediction_mean_metric": metric,
        "seed_metrics": seed_metrics,
        "fold_metrics": by_fold,
        "layer_metrics": by_layer,
    }
    return output, aggregate


def _csv_bytes(frame: pd.DataFrame, columns: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    frame.loc[:, columns].to_csv(
        buffer,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    return buffer.getvalue().encode("utf-8")


def _pin(path: Path, output: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(output).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _verify_data_pins(data_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    directory = data_dir.resolve(strict=True)
    result: dict[str, Any] = {}
    for name, expected in config["data_contract"]["source_pins"].items():
        path = (directory / name).resolve(strict=True)
        if path.parent != directory:
            raise ValueError("data source path escaped the explicit data directory")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"data source SHA mismatch: {name}")
        result[name] = {"sha256": actual, "bytes": path.stat().st_size}
    return result


def _verify_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = config["runtime_contract"]
    observed = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": importlib.metadata.version("numpy"),
        "pandas": importlib.metadata.version("pandas"),
        "scipy": importlib.metadata.version("scipy"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "lightgbm": importlib.metadata.version("lightgbm"),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }
    exact_keys = (
        "python",
        "numpy",
        "pandas",
        "scipy",
        "scikit_learn",
        "lightgbm",
        "torch",
        "torch_cuda",
        "cuda_available",
        "cudnn_benchmark",
        "cudnn_deterministic",
    )
    mismatches = [key for key in exact_keys if observed[key] != expected[key]]
    if expected["gpu_name_contains"] not in observed["gpu_name"]:
        mismatches.append("gpu_name")
    if mismatches:
        raise RuntimeError(f"Stage-A runtime contract mismatch: {sorted(mismatches)}")
    return observed


def execute_stage_a(
    *,
    root: Path,
    data_dir: Path,
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    attempt_lock: Path,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run and seal the complete reference curve; never fit a challenger."""

    workspace = root.resolve(strict=True)
    canonical = load_canonical_config(workspace, supplied_config=config)
    paths = stage_paths(workspace, canonical)
    if attempt_lock.resolve(strict=True) != paths["attempt_lock"].resolve(strict=True):
        raise PermissionError("engine did not receive the canonical consumed attempt lock")
    _qa, qa_sha256 = verify_pre_execution_qa(workspace, canonical)
    _authorization, authorization_sha256 = verify_execution_authorization(
        workspace,
        canonical,
        qa_sha256=qa_sha256,
        require_unconsumed=False,
    )
    verify_consumed_attempt_lock(
        workspace,
        canonical,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
    )
    if preflight.get("status") != "PASS_STATIC_IMPLEMENTATION_ONLY":
        raise PermissionError("engine requires the successful static preflight")
    if preflight.get("implementation_pins") != implementation_pins(workspace):
        raise PermissionError("implementation bytes changed after preflight")
    expected_implementation_pins = preflight["implementation_pins"]
    if paths["output"].exists():
        raise FileExistsError("append-only Stage-A output already exists")
    set_deterministic_seed(PIPELINE_SEEDS[0])
    runtime = _verify_runtime(canonical)
    data_pins = _verify_data_pins(data_dir, canonical)
    output = paths["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output)
    started = _now_kst()
    exclusive_json(
        contained_path(output, canonical["stage_a_reference_contract"]["artifacts"]["architecture_manifest"]),
        canonical["architecture_reference"],
    )
    exclusive_json(
        contained_path(output, canonical["stage_a_reference_contract"]["artifacts"]["training_recipe"]),
        canonical["training_recipe"],
    )

    data = load_p2_data(data_dir)
    panel = _joint_masked_panel(data.observations)
    router = _build_router_context(data.observations)
    endpoints = public_endpoint_frame(data.observations)
    recipe = canonical["training_recipe"]
    plan = build_execution_plan(canonical)
    by_fraction: dict[float, dict[int, list[pd.DataFrame]]] = {
        fraction: {seed: [] for seed in PIPELINE_SEEDS} for fraction in PREFIX_FRACTIONS
    }
    cell_receipts: list[dict[str, Any]] = []
    for fraction in PREFIX_FRACTIONS:
        for fold in recipe["outer_folds"]:
            for pipeline_seed in PIPELINE_SEEDS:
                frame, receipt = _run_cell_seed(
                    panel=panel,
                    router=router,
                    endpoints=endpoints,
                    recipe=recipe,
                    layer_factors=canonical["architecture_reference"][
                        "layer_extrapolation_factors"
                    ],
                    fold=fold,
                    fraction=fraction,
                    pipeline_seed=pipeline_seed,
                    progress=progress,
                )
                by_fraction[fraction][pipeline_seed].append(frame)
                cell_receipts.append(receipt)

    artifacts = canonical["stage_a_reference_contract"]["artifacts"]
    oof_roles = {
        0.4: "reference_oof_040",
        0.55: "reference_oof_055",
        0.7: "reference_oof_070",
        0.85: "reference_oof_085",
        1.0: "reference_oof_100",
    }
    curve_points: list[dict[str, Any]] = []
    oof_paths: dict[float, Path] = {}
    output_columns = canonical["stage_a_reference_contract"]["reference_oof_columns"]
    for fraction in PREFIX_FRACTIONS:
        seed_frames = {
            seed: pd.concat(parts, ignore_index=True)
            for seed, parts in by_fraction[fraction].items()
        }
        merged, aggregate = _merge_seed_predictions(
            seed_frames,
            recipe["metric"]["official_layer_counts"],
        )
        row_artifact = merged.drop(columns="truth")
        if set(row_artifact.columns) != set(output_columns):
            raise RuntimeError("target-free OOF schema is incomplete")
        path = contained_path(output, artifacts[oof_roles[fraction]])
        exclusive_bytes(path, _csv_bytes(row_artifact, output_columns))
        oof_paths[fraction] = path
        curve_points.append({"fraction": fraction, **aggregate})

    curve_metrics = {
        "schema_version": "p2_architecture_matched_reference.curve_metrics.v2",
        "problem": "P2",
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "seed_aggregation": "PREDICTION_MEAN_THEN_METRIC",
        "metric": recipe["metric"],
        "points": curve_points,
        "local_qualification_only": True,
        "official_promotion_allowed": False,
        "uploads": 0,
    }
    exclusive_json(contained_path(output, artifacts["reference_curve_metrics"]), curve_metrics)
    training_receipt = {
        "schema_version": "p2_architecture_matched_reference.training_receipt.v2",
        "started_at_kst": started,
        "completed_at_kst": _now_kst(),
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "plan": plan,
        "runtime": runtime,
        "cells": cell_receipts,
        "guard_summary": {
            "joint_temp_psal_mask_applied_before_all_label_use": True,
            "outer_and_future_target_labels_used_for_fit": False,
            "frozen_stack_or_gate_reused": False,
            "all_five_prefixes_completed_before_seal": True,
            "challenger_import_fit_or_score_count": 0,
            "full_fit_count": 0,
            "submission_prediction_count": 0,
            "upload_count": 0,
        },
    }
    exclusive_json(contained_path(output, artifacts["training_receipt"]), training_receipt)

    # A long Stage-A run must not be sealed if any preregistered source, model
    # graph, data file, QA receipt, authorization, lock, or implementation byte
    # drifted while training was in progress.
    end_preflight = static_preflight(workspace, data_dir)
    if end_preflight["implementation_pins"] != expected_implementation_pins:
        raise PermissionError("implementation bytes changed during Stage-A execution")
    if _verify_data_pins(data_dir, canonical) != data_pins:
        raise PermissionError("data source bytes changed during Stage-A execution")
    _qa_end, qa_sha256_end = verify_pre_execution_qa(workspace, canonical)
    if qa_sha256_end != qa_sha256:
        raise PermissionError("QA receipt changed during Stage-A execution")
    _authorization_end, authorization_sha256_end = verify_execution_authorization(
        workspace,
        canonical,
        qa_sha256=qa_sha256,
        require_unconsumed=False,
        require_output_absent=False,
    )
    if authorization_sha256_end != authorization_sha256:
        raise PermissionError("authorization changed during Stage-A execution")
    verify_consumed_attempt_lock(
        workspace,
        canonical,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
    )

    artifact_pins: dict[str, dict[str, Any]] = {}
    for role, relative in artifacts.items():
        if role in {"manifest", "seal"}:
            continue
        path = contained_path(output, relative)
        artifact_pins[role] = _pin(path, output)
    manifest = {
        "schema_version": "p2_architecture_matched_reference.manifest.v2",
        "created_at_kst": _now_kst(),
        "append_only": True,
        "problem": "P2",
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "implementation_pins": expected_implementation_pins,
        "runtime": runtime,
        "data_source_pins": data_pins,
        "architecture_manifest_sha256": canonical_mapping_sha256(
            canonical["architecture_reference"]
        ),
        "training_recipe_sha256": canonical_mapping_sha256(recipe),
        "artifacts": artifact_pins,
        "challenger_import_fit_or_score_count": 0,
        "official_promotion_allowed": False,
        "uploads": 0,
    }
    manifest_path = contained_path(output, artifacts["manifest"])
    exclusive_json(manifest_path, manifest)
    reference_by_fraction = {
        str(fraction): artifact_pins[oof_roles[fraction]] for fraction in PREFIX_FRACTIONS
    }
    seal = {
        "schema_version": "p2_architecture_matched_reference.seal.v2",
        "complete": True,
        "all_five_prefixes_sealed": True,
        "challenger_import_fit_or_score_count_before_seal": 0,
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "official_promotion_allowed": False,
        "upload_count": 0,
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "manifest": _pin(manifest_path, output),
        "reference_oof_by_fraction": reference_by_fraction,
    }
    seal_path = contained_path(output, artifacts["seal"])
    exclusive_json(seal_path, seal)
    return {
        "schema_version": "p2_architecture_matched_stage_a_execution.result.v2",
        "status": "COMPLETE_SEALED_ARCHITECTURE_MATCHED_REFERENCE",
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "official_promotion_allowed": False,
        "output": output.relative_to(workspace).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "seal_sha256": sha256_file(seal_path),
        "curve_metrics_sha256": sha256_file(
            contained_path(output, artifacts["reference_curve_metrics"])
        ),
        "challenger_fits": 0,
        "submission_predictions": 0,
        "uploads": 0,
    }


__all__ = ["build_execution_plan", "execute_stage_a"]
