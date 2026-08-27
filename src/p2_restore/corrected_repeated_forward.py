"""Corrected repeated-forward training utilities for P2.

This generation deliberately reuses the already fixed low-complexity model
family.  Target-layer temperature and salinity are jointly removed from the
entire feature context before any feature is calculated.  Temperature labels
remain only in the returned population table and are selected by chronological
training masks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from p2_restore.features import (
    PUBLIC_LAYERS,
    TARGET_LAYERS,
    FeatureTable,
    _common_features,
    _finalize,
    _nearest_public_baseline,
    _wide,
)
from p2_restore.model import fit_model
from p2_restore.profile_projection import (
    project_profiles_vectorized,
    public_endpoint_frame,
)
from p2_restore.research import (
    P2ResearchBlendModel,
    append_public_dynamics,
    select_lean_m2_dynamics,
)

KST = "Asia/Seoul"


@dataclass(frozen=True)
class JointMaskAudit:
    target_context_rows: int
    target_temp_non_null_after_mask: int
    target_psal_non_null_after_mask: int
    hidden_target_rows: int
    hidden_temp_non_null_before_mask: int
    hidden_psal_non_null_before_mask: int


def joint_mask_target_context(
    observations: pd.DataFrame,
    *,
    hidden_start: str = "2025-09-01T00:00:00+09:00",
    hidden_stop: str = "2025-11-01T00:00:00+09:00",
) -> tuple[pd.DataFrame, JointMaskAudit]:
    """Return a copy with target-layer temp and psal jointly nulled.

    The distributed hidden interval must already contain no target temperature
    or salinity.  A populated value fails closed rather than being silently
    consumed as a label.
    """

    required = {"layer", "time", "temp", "psal"}
    if missing := required.difference(observations.columns):
        raise ValueError(f"observations are missing joint-mask columns: {sorted(missing)}")
    target = observations["layer"].isin(TARGET_LAYERS)
    time = pd.to_datetime(observations["time"], utc=True)
    left = pd.Timestamp(hidden_start).tz_convert("UTC")
    right = pd.Timestamp(hidden_stop).tz_convert("UTC")
    hidden = target & time.ge(left) & time.lt(right)
    hidden_temp_non_null = int(observations.loc[hidden, "temp"].notna().sum())
    hidden_psal_non_null = int(observations.loc[hidden, "psal"].notna().sum())
    if hidden_temp_non_null or hidden_psal_non_null:
        raise ValueError("hidden target-layer temp/psal unexpectedly contain values")

    masked = observations.copy()
    masked.loc[target, ["temp", "psal"]] = np.nan
    audit = JointMaskAudit(
        target_context_rows=int(target.sum()),
        target_temp_non_null_after_mask=int(masked.loc[target, "temp"].notna().sum()),
        target_psal_non_null_after_mask=int(masked.loc[target, "psal"].notna().sum()),
        hidden_target_rows=int(hidden.sum()),
        hidden_temp_non_null_before_mask=hidden_temp_non_null,
        hidden_psal_non_null_before_mask=hidden_psal_non_null,
    )
    if audit.target_temp_non_null_after_mask or audit.target_psal_non_null_after_mask:
        raise AssertionError("target-layer context was not jointly masked")
    return masked, audit


def build_joint_masked_population(
    observations: pd.DataFrame, masked_context: pd.DataFrame
) -> FeatureTable:
    """Build public-only features for every eligible target-layer grid row.

    Unlike ``build_training_features``, rows with missing target temperature are
    retained.  That lets physical projection operate on a complete three-layer
    profile before the finite-label scoring subset is selected.
    """

    if len(observations) != len(masked_context):
        raise ValueError("label source and masked context row counts differ")
    if not observations[["station", "year", "layer", "time"]].equals(
        masked_context[["station", "year", "layer", "time"]]
    ):
        raise ValueError("label source and masked context keys differ")
    target_context = masked_context["layer"].isin(TARGET_LAYERS)
    if masked_context.loc[target_context, ["temp", "psal"]].notna().any().any():
        raise ValueError("target-layer values remain in feature context")

    times, common = _common_features(masked_context)
    target_temp = _wide(observations, "temp").reindex(times)
    target_nominal = _wide(observations, "nominal_depth").reindex(times)
    public_temp = np.column_stack([common[f"temp_{layer}"] for layer in PUBLIC_LAYERS])
    public_depth = np.column_stack([common[f"nominal_{layer}"] for layer in PUBLIC_LAYERS])
    rows: list[pd.DataFrame] = []
    for layer in TARGET_LAYERS:
        truth = target_temp.get(layer, pd.Series(index=times, dtype=float)).to_numpy(float)
        depth = target_nominal.get(layer, pd.Series(index=times, dtype=float)).to_numpy(float)
        baseline = _nearest_public_baseline(public_temp, public_depth, depth)
        keep = (
            np.isfinite(depth)
            & np.isfinite(baseline)
            & (np.asarray(common["public_temp_count"], dtype=float) >= 2)
        )
        part = pd.DataFrame({name: values[keep] for name, values in common.items()})
        part["station"] = "S-ORS"
        part["layer"] = layer
        part["time"] = times[keep].astype(str)
        part["target_depth"] = depth[keep]
        part["baseline"] = baseline[keep]
        part["target"] = truth[keep]
        part["residual"] = truth[keep] - baseline[keep]
        rows.append(part)
    population = _finalize(pd.concat(rows, ignore_index=True))
    forbidden = {f"temp_{layer}" for layer in TARGET_LAYERS} | {
        f"psal_{layer}" for layer in TARGET_LAYERS
    }
    if forbidden.intersection(population.feature_columns):
        raise AssertionError("target-layer feature leakage detected")
    return population


def build_fixed_lean_arm(base: FeatureTable, masked_context: pd.DataFrame) -> FeatureTable:
    """Append the frozen 20 public-temperature 6h/M2 dynamics features."""

    dynamic = append_public_dynamics(base, masked_context)
    lean = select_lean_m2_dynamics(base, dynamic)
    if not base.frame[["station", "layer", "time"]].equals(
        lean.frame[["station", "layer", "time"]]
    ):
        raise AssertionError("base and lean feature rows are not aligned")
    return lean


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(KST)
    return timestamp.tz_convert("UTC")


def window_mask(frame: pd.DataFrame, start: str, stop: str) -> np.ndarray:
    time = pd.to_datetime(frame["time"], utc=True)
    return (time.ge(_utc(start)) & time.lt(_utc(stop))).to_numpy(bool)


def forward_training_mask(
    frame: pd.DataFrame, validation_start: str, *, embargo_days: int
) -> tuple[np.ndarray, pd.Timestamp]:
    """Select finite labels strictly before a KST validation boundary embargo."""

    if embargo_days < 1:
        raise ValueError("a positive target-label embargo is required")
    cutoff = _utc(validation_start) - pd.Timedelta(days=embargo_days)
    time = pd.to_datetime(frame["time"], utc=True)
    finite_label = np.isfinite(frame["residual"].to_numpy(float))
    selected = finite_label & time.lt(cutoff).to_numpy(bool)
    if not selected.any():
        raise ValueError("forward training split has no finite target labels")
    if not time.loc[selected].lt(cutoff).all():
        raise AssertionError("forward split crossed the embargo cutoff")
    return selected, cutoff


def fit_fixed_blend(
    base: FeatureTable,
    lean: FeatureTable,
    selected: np.ndarray,
    *,
    seed: int,
) -> P2ResearchBlendModel:
    """Fit the frozen two-arm 50:50 LightGBM family on selected rows."""

    rows = np.asarray(selected, dtype=bool)
    if rows.shape != (len(base.frame),) or int(rows.sum()) < 1_000:
        raise ValueError("fixed blend needs at least 1,000 aligned training rows")
    if not base.frame[["station", "layer", "time"]].equals(
        lean.frame[["station", "layer", "time"]]
    ):
        raise ValueError("base and lean training rows differ")
    if not np.isfinite(base.frame.loc[rows, "residual"]).all():
        raise ValueError("selected training labels contain non-finite residuals")
    return P2ResearchBlendModel(
        base_model=fit_model(base, rows, seed=seed),
        lean_model=fit_model(lean, rows, seed=seed),
        weight=0.5,
    )


def predict_scored_window(
    model: P2ResearchBlendModel,
    base: FeatureTable,
    lean: FeatureTable,
    endpoints: pd.DataFrame,
    *,
    start: str,
    stop: str,
    fold: str,
    stage: str,
) -> pd.DataFrame:
    """Predict and project a complete window, then retain finite target labels."""

    selected = window_mask(base.frame, start, stop)
    if not selected.any():
        raise ValueError(f"{stage} window {fold} has no eligible feature rows")
    base_window = FeatureTable(base.frame.loc[selected].reset_index(drop=True), base.feature_columns)
    lean_window = FeatureTable(lean.frame.loc[selected].reset_index(drop=True), lean.feature_columns)
    unprojected = model.predict(base_window, lean_window)
    projection = project_profiles_vectorized(base_window.frame, unprojected, endpoints)
    frame = base_window.frame.loc[:, ["station", "layer", "time", "target", "baseline"]].copy()
    frame = frame.rename(columns={"target": "truth"})
    frame["blend_prediction"] = unprojected
    frame["prediction"] = projection.prediction
    frame["projection_eligible"] = projection.eligible_mask
    frame["projection_active"] = projection.active_mask
    frame["fold"] = fold
    frame["stage"] = stage
    frame["kst_day"] = (
        pd.to_datetime(frame["time"], utc=True).dt.tz_convert(KST).dt.strftime("%Y-%m-%d")
    )
    scored = frame.loc[np.isfinite(frame["truth"])].reset_index(drop=True)
    if scored.empty or not np.isfinite(
        scored[["truth", "baseline", "blend_prediction", "prediction"]]
    ).all().all():
        raise ValueError(f"{stage} window {fold} has invalid scored predictions")
    if scored.duplicated(["station", "layer", "time"]).any():
        raise ValueError(f"{stage} window {fold} has duplicate scored keys")
    return scored


def nominal_target_rows(start: str, stop: str) -> int:
    minutes = (_utc(stop) - _utc(start)).total_seconds() / 60.0
    steps = int(round(minutes / 10.0))
    return steps * len(TARGET_LAYERS)


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(truth) - np.asarray(prediction)) ** 2)))


def normalized_layer_weights(layer_counts: Mapping[str | int, int]) -> dict[int, float]:
    counts = {int(layer): int(count) for layer, count in layer_counts.items()}
    if set(counts) != set(TARGET_LAYERS) or any(count <= 0 for count in counts.values()):
        raise ValueError("official layer counts must be positive for layers 2, 3, and 4")
    total = float(sum(counts.values()))
    return {layer: counts[layer] / total for layer in TARGET_LAYERS}


def metric_report(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    official_layer_counts: Mapping[str | int, int],
) -> dict[str, object]:
    """Report raw pooled and fold-equal official-layer-weighted RMSE."""

    required = {"fold", "layer", "truth", prediction_column}
    if missing := required.difference(frame.columns):
        raise ValueError(f"metric frame is missing columns: {sorted(missing)}")
    weights = normalized_layer_weights(official_layer_counts)
    folds: dict[str, object] = {}
    fold_mse: list[float] = []
    for fold, group in frame.groupby("fold", sort=False):
        by_layer: dict[str, object] = {}
        weighted_mse = 0.0
        for layer in TARGET_LAYERS:
            selected = group["layer"].eq(layer).to_numpy(bool)
            if not selected.any():
                raise ValueError(f"fold {fold} has no scored layer {layer}")
            truth = group.loc[selected, "truth"].to_numpy(float)
            prediction = group.loc[selected, prediction_column].to_numpy(float)
            mse = float(np.mean((truth - prediction) ** 2))
            weighted_mse += weights[layer] * mse
            by_layer[str(layer)] = {"rows": int(selected.sum()), "rmse_c": float(np.sqrt(mse))}
        fold_mse.append(weighted_mse)
        folds[str(fold)] = {
            "rows": int(len(group)),
            "official_layer_weighted_rmse_c": float(np.sqrt(weighted_mse)),
            "by_layer": by_layer,
        }
    truth = frame["truth"].to_numpy(float)
    prediction = frame[prediction_column].to_numpy(float)
    return {
        "rows": int(len(frame)),
        "folds": int(len(fold_mse)),
        "raw_pooled_rmse_c": _rmse(truth, prediction),
        "fold_equal_official_layer_weighted_rmse_c": float(np.sqrt(np.mean(fold_mse))),
        "by_fold": folds,
    }


def paired_fold_day_bootstrap(
    frame: pd.DataFrame,
    *,
    reference_column: str,
    candidate_column: str,
    official_layer_counts: Mapping[str | int, int],
    replicates: int,
    seed: int,
    interval: float,
) -> dict[str, object]:
    """Paired KST-day bootstrap preserving equal outer-fold weighting."""

    if replicates < 100 or not 0.0 < interval < 1.0:
        raise ValueError("invalid bootstrap settings")
    weights = normalized_layer_weights(official_layer_counts)
    rng = np.random.default_rng(seed)
    prepared: list[dict[str, object]] = []
    for fold, group in frame.groupby("fold", sort=False):
        days = np.asarray(sorted(group["kst_day"].astype(str).unique()))
        if len(days) < 2:
            raise ValueError(f"fold {fold} has fewer than two KST days")
        layer_arrays: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for layer in TARGET_LAYERS:
            part = group.loc[group["layer"].eq(layer)].copy()
            part["reference_sse"] = (
                part["truth"].to_numpy(float) - part[reference_column].to_numpy(float)
            ) ** 2
            part["candidate_sse"] = (
                part["truth"].to_numpy(float) - part[candidate_column].to_numpy(float)
            ) ** 2
            aggregate = part.groupby("kst_day", sort=False).agg(
                reference_sse=("reference_sse", "sum"),
                candidate_sse=("candidate_sse", "sum"),
                rows=("truth", "size"),
            )
            aggregate = aggregate.reindex(days, fill_value=0)
            layer_arrays[layer] = (
                aggregate["reference_sse"].to_numpy(float),
                aggregate["candidate_sse"].to_numpy(float),
                aggregate["rows"].to_numpy(float),
            )
        prepared.append({"fold": str(fold), "days": days, "layers": layer_arrays})

    deltas = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        reference_fold_mse: list[float] = []
        candidate_fold_mse: list[float] = []
        for fold in prepared:
            days = fold["days"]
            sampled = rng.integers(0, len(days), size=len(days))
            reference_mse = 0.0
            candidate_mse = 0.0
            layers = fold["layers"]
            for layer in TARGET_LAYERS:
                reference_sse, candidate_sse, rows = layers[layer]
                denominator = float(rows[sampled].sum())
                if denominator <= 0:
                    raise AssertionError("bootstrap sampled an empty layer")
                reference_mse += weights[layer] * float(reference_sse[sampled].sum()) / denominator
                candidate_mse += weights[layer] * float(candidate_sse[sampled].sum()) / denominator
            reference_fold_mse.append(reference_mse)
            candidate_fold_mse.append(candidate_mse)
        deltas[replicate] = float(
            np.sqrt(np.mean(candidate_fold_mse)) - np.sqrt(np.mean(reference_fold_mse))
        )

    reference = metric_report(
        frame,
        prediction_column=reference_column,
        official_layer_counts=official_layer_counts,
    )["fold_equal_official_layer_weighted_rmse_c"]
    candidate = metric_report(
        frame,
        prediction_column=candidate_column,
        official_layer_counts=official_layer_counts,
    )["fold_equal_official_layer_weighted_rmse_c"]
    alpha = (1.0 - interval) / 2.0
    return {
        "unit": "KST calendar day sampled within each fold",
        "fold_weighting": "equal",
        "layer_weighting": "official test_index counts",
        "replicates": int(replicates),
        "seed": int(seed),
        "reference_rmse_c": float(reference),
        "candidate_rmse_c": float(candidate),
        "delta_rmse_c": float(candidate - reference),
        "delta_interval": [
            float(np.quantile(deltas, alpha)),
            float(np.quantile(deltas, 1.0 - alpha)),
        ],
        "interval_mass": float(interval),
        "probability_candidate_improves": float(np.mean(deltas < 0.0)),
    }


def public_endpoints_from_masked_context(masked_context: pd.DataFrame) -> pd.DataFrame:
    """Expose the label-blind endpoint frame through this generation's API."""

    return public_endpoint_frame(masked_context)
