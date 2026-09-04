"""Aggregate-only matched-budget comparison utilities for P2.

The module operates exclusively on historical local OOF predictions and the
organizer observations outside the hidden target interval.  It has no test,
submission, or inference-output path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from p2_restore.dynamic_sigmoid_profile import effective_depth
from p2_restore.features import PUBLIC_LAYERS, _nearest_public_baseline, _wide
from p2_restore.profile_projection import project_profiles_vectorized, public_endpoint_frame
from p2_restore.public_layer_causal_residual import (
    CausalResidualSpec,
    build_public_residual_state,
    correction_for_rows,
)

TARGET_LAYERS = (2, 3, 4)
HIDDEN_START = pd.Timestamp("2025-09-01", tz="Asia/Seoul")
HIDDEN_STOP = pd.Timestamp("2025-11-01", tz="Asia/Seoul")


@dataclass(frozen=True)
class LocalContext:
    baseline_depth_interpolation: pd.DataFrame
    truth: pd.DataFrame
    endpoints: pd.DataFrame
    causal_state: pd.DataFrame


def build_local_context(
    observations: pd.DataFrame,
    spec: CausalResidualSpec,
) -> LocalContext:
    """Build historical truth and label-blind public-layer context lookups."""

    required = {
        "station",
        "layer",
        "time",
        "temp",
        "psal",
        "depth",
        "nominal_depth",
    }
    if missing := required.difference(observations.columns):
        raise ValueError(f"observations missing columns: {sorted(missing)}")

    public = observations.loc[
        observations["layer"].isin(PUBLIC_LAYERS), list(required)
    ].copy()
    public_temp = _wide(public, "temp").reindex(columns=PUBLIC_LAYERS)
    public_nominal = _wide(public, "nominal_depth").reindex(
        index=public_temp.index, columns=PUBLIC_LAYERS
    )
    # Round A was defined against the organizer interpolation baseline.  Its
    # historical OOF stores the same nominal-depth interpolation exactly; using
    # the observed (moving) depth here would silently define a different arm.
    public_baseline_depth = public_nominal.to_numpy(float)
    target_depth_rows = observations.loc[
        observations["layer"].isin(TARGET_LAYERS),
        ["time", "layer", "depth", "nominal_depth"],
    ]
    target_depth = _wide(target_depth_rows, "depth").reindex(
        index=public_temp.index, columns=TARGET_LAYERS
    )
    target_nominal = _wide(target_depth_rows, "nominal_depth").reindex(
        index=public_temp.index, columns=TARGET_LAYERS
    )
    target_effective_depth = effective_depth(
        target_depth.to_numpy(float), target_nominal.to_numpy(float)
    )
    target_baseline_depth = target_nominal.to_numpy(float)
    times = pd.to_datetime(public_temp.index, utc=True)
    baseline_parts: list[pd.DataFrame] = []
    for position, layer in enumerate(TARGET_LAYERS):
        interpolated = _nearest_public_baseline(
            public_temp.to_numpy(float),
            public_baseline_depth,
            target_baseline_depth[:, position],
        )
        baseline_parts.append(
            pd.DataFrame(
                {
                    "station": "S-ORS",
                    "layer": layer,
                    "_time_key": times,
                    "target_depth": target_effective_depth[:, position],
                    "local_depth_interpolation": interpolated,
                }
            )
        )
    baseline = pd.concat(baseline_parts, ignore_index=True)
    if baseline.duplicated(["station", "layer", "_time_key"]).any():
        raise AssertionError("local baseline keys are not unique")

    target = observations.loc[
        observations["layer"].isin(TARGET_LAYERS),
        ["station", "layer", "time", "temp"],
    ].copy()
    target["_time_key"] = pd.to_datetime(target["time"], utc=True)
    target_kst = target["_time_key"].dt.tz_convert("Asia/Seoul")
    allowed = ~target_kst.ge(HIDDEN_START) | target_kst.ge(HIDDEN_STOP)
    truth = target.loc[allowed, ["station", "layer", "_time_key", "temp"]].rename(
        columns={"temp": "source_truth"}
    )
    truth = truth.loc[np.isfinite(truth["source_truth"])].copy()
    if truth.duplicated(["station", "layer", "_time_key"]).any():
        raise AssertionError("historical truth keys are not unique")
    return LocalContext(
        baseline_depth_interpolation=baseline,
        truth=truth,
        endpoints=public_endpoint_frame(public),
        causal_state=build_public_residual_state(public, spec),
    )


def prepare_exact_frozen_surface(oof: pd.DataFrame, context: LocalContext) -> pd.DataFrame:
    required = {"time", "layer", "truth", "block", "prediction"}
    if missing := required.difference(oof.columns):
        raise ValueError(f"exact frozen OOF missing columns: {sorted(missing)}")
    frame = oof.loc[:, ["time", "layer", "truth", "block", "prediction"]].copy()
    frame.insert(0, "station", "S-ORS")
    frame["_time_key"] = pd.to_datetime(frame["time"], utc=True)
    frame["fold"] = frame.pop("block").astype(str)
    frame = frame.merge(
        context.baseline_depth_interpolation,
        on=["station", "layer", "_time_key"],
        how="left",
        validate="one_to_one",
    ).merge(
        context.truth,
        on=["station", "layer", "_time_key"],
        how="left",
        validate="one_to_one",
    )
    if (
        frame[["truth", "prediction", "local_depth_interpolation", "target_depth"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("exact frozen OOF alignment contains missing values")
    truth_error = np.max(
        np.abs(frame["truth"].to_numpy(float) - frame["source_truth"].to_numpy(float))
    )
    if truth_error > 1e-12:
        raise ValueError("exact frozen OOF truth does not reproduce source observations")
    kst = frame["_time_key"].dt.tz_convert("Asia/Seoul")
    if (kst.ge(HIDDEN_START) & kst.lt(HIDDEN_STOP)).any():
        raise ValueError("exact frozen OOF overlaps hidden target interval")
    frame["base_frozen"] = frame.pop("prediction").astype(float)
    return frame


def prepare_forward_surrogate_surface(
    oof: pd.DataFrame,
    context: LocalContext,
    seed_columns: tuple[str, ...],
) -> pd.DataFrame:
    required = {"fold", "station", "layer", "time", *seed_columns, "prediction_mean"}
    if missing := required.difference(oof.columns):
        raise ValueError(f"forward surrogate OOF missing columns: {sorted(missing)}")
    frame = oof.copy()
    frame["_time_key"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.merge(
        context.baseline_depth_interpolation,
        on=["station", "layer", "_time_key"],
        how="left",
        validate="one_to_one",
    ).merge(
        context.truth,
        on=["station", "layer", "_time_key"],
        how="left",
        validate="one_to_one",
    )
    needed = [
        *seed_columns,
        "prediction_mean",
        "local_depth_interpolation",
        "target_depth",
        "source_truth",
    ]
    if frame[needed].isna().any().any():
        raise ValueError("forward surrogate OOF alignment contains missing values")
    kst = frame["_time_key"].dt.tz_convert("Asia/Seoul")
    if (kst.ge(HIDDEN_START) & kst.lt(HIDDEN_STOP)).any():
        raise ValueError("forward surrogate OOF overlaps hidden target interval")
    frame["truth"] = frame["source_truth"].astype(float)
    return frame


def _project(frame: pd.DataFrame, values: np.ndarray, endpoints: pd.DataFrame) -> np.ndarray:
    projection_frame = frame.loc[:, ["station", "layer", "time"]]
    key_columns = ["station", "time", "layer"]
    if not projection_frame.duplicated(key_columns).any():
        return project_profiles_vectorized(
            projection_frame, values, endpoints
        ).prediction
    if "prefix_fraction" not in frame:
        raise ValueError("projection keys are duplicated without a sealed surface id")
    source = np.asarray(values, dtype=float)
    output = np.empty_like(source)
    fractions = frame["prefix_fraction"].to_numpy(float)
    for fraction in sorted(frame["prefix_fraction"].unique()):
        selected = np.isclose(fractions, float(fraction))
        output[selected] = project_profiles_vectorized(
            projection_frame.loc[selected].reset_index(drop=True),
            source[selected],
            endpoints,
        ).prediction
    return output


def materialize_settings(
    frame: pd.DataFrame,
    base_columns: tuple[str, ...],
    context: LocalContext,
    spec: CausalResidualSpec,
) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]], dict[str, Any]]:
    """Apply every sealed setting to each available prediction seed."""

    correction = correction_for_rows(
        frame,
        frame["target_depth"].to_numpy(float),
        context.causal_state,
        spec,
    )
    per_seed: dict[str, list[np.ndarray]] = {
        setting: []
        for setting in (
            "INCUMBENT_NOOP",
            "STACK_W0500",
            "STACK_W0625",
            "STACK_W0750",
            "CAUSAL_RESIDUAL_SCALE025",
            "FALLBACK_BLEND50_A0625",
            "CAUSAL_ON_FALLBACK",
        )
    }
    baseline = frame["local_depth_interpolation"].to_numpy(float)
    idempotence_error = 0.0
    for column in base_columns:
        raw_base = frame[column].to_numpy(float)
        incumbent = _project(frame, raw_base, context.endpoints)
        idempotence_error = max(
            idempotence_error, float(np.max(np.abs(incumbent - raw_base), initial=0.0))
        )
        stack: dict[float, np.ndarray] = {}
        for weight in (0.5, 0.625, 0.75):
            stack[weight] = _project(
                frame,
                baseline + weight * (incumbent - baseline),
                context.endpoints,
            )
        causal = _project(frame, incumbent + correction.correction, context.endpoints)
        fallback = _project(frame, 0.5 * incumbent + 0.5 * stack[0.625], context.endpoints)
        causal_fallback = _project(
            frame, fallback + correction.correction, context.endpoints
        )
        per_seed["INCUMBENT_NOOP"].append(incumbent)
        per_seed["STACK_W0500"].append(stack[0.5])
        per_seed["STACK_W0625"].append(stack[0.625])
        per_seed["STACK_W0750"].append(stack[0.75])
        per_seed["CAUSAL_RESIDUAL_SCALE025"].append(causal)
        per_seed["FALLBACK_BLEND50_A0625"].append(fallback)
        per_seed["CAUSAL_ON_FALLBACK"].append(causal_fallback)
    means = {
        setting: np.mean(np.vstack(predictions), axis=0)
        for setting, predictions in per_seed.items()
    }
    diagnostics = {
        "base_seed_count": len(base_columns),
        "incumbent_reprojection_max_abs_error_c": idempotence_error,
        "causal_correction": correction.diagnostics,
    }
    return means, per_seed, diagnostics


def metric_report(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    values = np.asarray(prediction, dtype=float)
    truth = frame["truth"].to_numpy(float)
    if values.shape != truth.shape or not np.isfinite(values).all():
        raise ValueError("metric prediction is invalid")
    work = frame.loc[:, ["fold", "layer"]].copy()
    work["error2"] = (values - truth) ** 2
    by_fold: dict[str, Any] = {}
    fold_layer_mse: list[float] = []
    for fold, part in work.groupby("fold", sort=True):
        layer_mse: list[float] = []
        layer_report: dict[str, float] = {}
        for layer in TARGET_LAYERS:
            selected = part["layer"].astype(int).eq(layer)
            if not selected.any():
                raise ValueError(f"fold {fold} lacks layer {layer}")
            mse = float(part.loc[selected, "error2"].mean())
            layer_mse.append(mse)
            fold_layer_mse.append(mse)
            layer_report[str(layer)] = float(np.sqrt(mse))
        by_fold[str(fold)] = {
            "rows": int(len(part)),
            "row_pooled_rmse_c": float(np.sqrt(part["error2"].mean())),
            "layer_equal_rmse_c": float(np.sqrt(np.mean(layer_mse))),
            "by_layer_rmse_c": layer_report,
        }
    by_layer = {
        str(layer): float(
            np.sqrt(work.loc[work["layer"].astype(int).eq(layer), "error2"].mean())
        )
        for layer in TARGET_LAYERS
    }
    return {
        "rows": int(len(frame)),
        "fold_equal_layer_equal_rmse_c": float(np.sqrt(np.mean(fold_layer_mse))),
        "fixed_historical_row_weighted_rmse_c": float(np.sqrt(work["error2"].mean())),
        "by_fold": by_fold,
        "by_layer_rmse_c": by_layer,
        "maximum_absolute_error_c": float(np.max(np.abs(values - truth), initial=0.0)),
    }


def build_bootstrap_plan(
    frame: pd.DataFrame, *, replicates: int, seed: int
) -> dict[str, np.ndarray]:
    work = frame.loc[:, ["fold", "_time_key"]].copy()
    work["day"] = work["_time_key"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    rng = np.random.default_rng(seed)
    plan: dict[str, np.ndarray] = {}
    for fold, part in work.groupby("fold", sort=True):
        days = np.array(sorted(part["day"].unique()), dtype=object)
        positions = rng.integers(0, len(days), size=(replicates, len(days)))
        plan[str(fold)] = days[positions]
    return plan


def paired_day_bootstrap(
    frame: pd.DataFrame,
    reference: np.ndarray,
    candidate: np.ndarray,
    plan: dict[str, np.ndarray],
    *,
    interval: float,
) -> dict[str, Any]:
    work = frame.loc[:, ["fold", "layer", "_time_key", "truth"]].copy()
    work["day"] = work["_time_key"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    truth = work["truth"].to_numpy(float)
    work["reference_sse"] = (np.asarray(reference, dtype=float) - truth) ** 2
    work["candidate_sse"] = (np.asarray(candidate, dtype=float) - truth) ** 2
    reference_cells: list[np.ndarray] = []
    candidate_cells: list[np.ndarray] = []
    for fold, part in work.groupby("fold", sort=True):
        draws = plan[str(fold)]
        for layer in TARGET_LAYERS:
            selected = part.loc[part["layer"].astype(int).eq(layer)]
            aggregate = selected.groupby("day", sort=False).agg(
                reference_sse=("reference_sse", "sum"),
                candidate_sse=("candidate_sse", "sum"),
                rows=("reference_sse", "size"),
            )
            reference_sse = (
                aggregate["reference_sse"]
                .reindex(draws.ravel(), fill_value=0.0)
                .to_numpy()
                .reshape(draws.shape)
            )
            candidate_sse = (
                aggregate["candidate_sse"]
                .reindex(draws.ravel(), fill_value=0.0)
                .to_numpy()
                .reshape(draws.shape)
            )
            counts = (
                aggregate["rows"]
                .reindex(draws.ravel(), fill_value=0)
                .to_numpy()
                .reshape(draws.shape)
            )
            sampled_count = counts.sum(axis=1)
            if np.any(sampled_count == 0):
                raise ValueError("bootstrap sampled an empty fold-layer cell")
            reference_cells.append(reference_sse.sum(axis=1) / sampled_count)
            candidate_cells.append(candidate_sse.sum(axis=1) / sampled_count)
    reference_rmse = np.sqrt(np.mean(np.vstack(reference_cells), axis=0))
    candidate_rmse = np.sqrt(np.mean(np.vstack(candidate_cells), axis=0))
    delta = candidate_rmse - reference_rmse
    tail = (1.0 - interval) / 2.0
    return {
        "replicates": int(len(delta)),
        "delta_rmse_c": float(delta.mean()),
        "ci90_c": [float(np.quantile(delta, tail)), float(np.quantile(delta, 1.0 - tail))],
        "probability_candidate_improves": float(np.mean(delta < 0.0)),
    }


def complementarity_report(
    frame: pd.DataFrame,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    endpoints: pd.DataFrame,
) -> dict[str, float]:
    truth = frame["truth"].to_numpy(float)
    incumbent_residual = np.asarray(incumbent, dtype=float) - truth
    candidate_residual = np.asarray(candidate, dtype=float) - truth
    if np.std(incumbent_residual) == 0.0 or np.std(candidate_residual) == 0.0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(incumbent_residual, candidate_residual)[0, 1])
    blend = _project(
        frame,
        0.5 * np.asarray(incumbent, dtype=float) + 0.5 * np.asarray(candidate, dtype=float),
        endpoints,
    )
    return {
        "paired_residual_pearson": correlation,
        "residual_disagreement_rms_c": float(
            np.sqrt(np.mean((candidate_residual - incumbent_residual) ** 2))
        ),
        "fixed_50_50_blend_primary_rmse_c": metric_report(frame, blend)[
            "fold_equal_layer_equal_rmse_c"
        ],
    }
