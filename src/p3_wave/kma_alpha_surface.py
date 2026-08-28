"""Diagnostic KMA correction-surface utilities for the P3 long leads.

The functions in this module operate only on previously sealed OOF predictions.
They do not read the official test context, create submissions, or upload files.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


ACTIVE_LEADS = (18, 24)
ALL_LEADS = (3, 6, 9, 12, 18, 24)
PAIR_KEYS = ("fold", "anchor_id", "station", "lead_h")


class KMAAlphaSurfaceError(ValueError):
    """Raised when the diagnostic input violates the frozen OOF contract."""


@dataclass(frozen=True)
class AlphaFit:
    analytic: float
    clipped: float
    grid: float
    grid_rmse: float
    rows: int


def rmse(truth: Sequence[float], prediction: Sequence[float]) -> float:
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.size == 0:
        raise KMAAlphaSurfaceError("RMSE arrays are empty or differ in shape")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise KMAAlphaSurfaceError("RMSE arrays contain non-finite values")
    return float(np.sqrt(np.mean(np.square(p - y))))


def make_alpha_grid(start: float, stop: float, step: float) -> np.ndarray:
    if not (np.isfinite(start) and np.isfinite(stop) and np.isfinite(step)):
        raise KMAAlphaSurfaceError("alpha grid bounds must be finite")
    if step <= 0.0 or stop < start:
        raise KMAAlphaSurfaceError("alpha grid bounds are invalid")
    count = int(round((stop - start) / step))
    grid = start + step * np.arange(count + 1, dtype=np.float64)
    if not np.isclose(grid[-1], stop, atol=1e-12, rtol=0.0):
        raise KMAAlphaSurfaceError("alpha grid does not land on stop")
    return np.round(grid, 12)


def prepare_oof_frame(blind: pd.DataFrame, evaluated: pd.DataFrame) -> pd.DataFrame:
    """Join sealed KMA predictions to the matching evaluated incumbent OOF rows."""

    blind_required = {
        *PAIR_KEYS,
        "incumbent_final",
        "calibrated_source",
    }
    evaluated_required = {*PAIR_KEYS, "target_hs", "prediction"}
    for role, frame, required in (
        ("blind", blind, blind_required),
        ("evaluated", evaluated, evaluated_required),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise KMAAlphaSurfaceError(f"{role} frame missing columns: {sorted(missing)}")
        if frame.empty or frame.duplicated(list(PAIR_KEYS)).any():
            raise KMAAlphaSurfaceError(f"{role} frame is empty or has duplicate keys")

    left = blind.loc[:, [*PAIR_KEYS, "incumbent_final", "calibrated_source"]].copy()
    right = evaluated.loc[:, [*PAIR_KEYS, "target_hs", "prediction"]].copy()
    merged = left.merge(right, on=list(PAIR_KEYS), how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise KMAAlphaSurfaceError("blind and evaluated OOF memberships differ")
    merged = merged.drop(columns="_merge")
    incumbent = merged["incumbent_final"].to_numpy(dtype=np.float64)
    evaluated_incumbent = merged["prediction"].to_numpy(dtype=np.float64)
    if not np.array_equal(incumbent, evaluated_incumbent):
        maximum = float(np.max(np.abs(incumbent - evaluated_incumbent)))
        raise KMAAlphaSurfaceError(f"incumbent prediction drifted; max abs delta={maximum}")
    leads = merged["lead_h"].to_numpy(dtype=np.int64)
    if not np.isin(leads, ALL_LEADS).all():
        raise KMAAlphaSurfaceError("OOF frame contains an unknown lead")
    lead_sets = merged.groupby(["fold", "anchor_id"], observed=True)["lead_h"].agg(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    if not lead_sets.map(lambda values: values == ALL_LEADS).all():
        raise KMAAlphaSurfaceError("an OOF case does not contain all six leads")
    merged["base"] = incumbent
    merged["delta"] = (
        merged["calibrated_source"].to_numpy(dtype=np.float64) - incumbent
    )
    inactive = ~merged["lead_h"].isin(ACTIVE_LEADS)
    if not np.array_equal(
        merged.loc[inactive, "delta"].to_numpy(dtype=np.float64),
        np.zeros(int(inactive.sum()), dtype=np.float64),
    ):
        # The stored calibrated source is not used on short leads. Force the
        # correction axis to an exact no-op there for every downstream check.
        merged.loc[inactive, "delta"] = 0.0
    numeric = merged[["target_hs", "base", "delta"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise KMAAlphaSurfaceError("prepared OOF frame contains non-finite values")
    return merged.sort_values(list(PAIR_KEYS), kind="mergesort").reset_index(drop=True)


def fit_alpha(
    frame: pd.DataFrame,
    grid: np.ndarray,
    *,
    mask: np.ndarray | pd.Series | None = None,
) -> AlphaFit:
    if mask is not None:
        selected = frame.loc[np.asarray(mask, dtype=bool)]
    else:
        selected = frame
    if selected.empty:
        raise KMAAlphaSurfaceError("cannot fit alpha on an empty frame")
    truth = selected["target_hs"].to_numpy(dtype=np.float64)
    base = selected["base"].to_numpy(dtype=np.float64)
    delta = selected["delta"].to_numpy(dtype=np.float64)
    denominator = float(np.dot(delta, delta))
    if denominator <= 0.0:
        analytic = 0.0
    else:
        analytic = float(np.dot(delta, truth - base) / denominator)
    clipped = float(np.clip(analytic, float(grid[0]), float(grid[-1])))
    residual = base - truth
    sse = (
        np.dot(residual, residual)
        + 2.0 * grid * np.dot(residual, delta)
        + np.square(grid) * denominator
    )
    index = int(np.argmin(sse))
    return AlphaFit(
        analytic=analytic,
        clipped=clipped,
        grid=float(grid[index]),
        grid_rmse=float(np.sqrt(sse[index] / len(selected))),
        rows=int(len(selected)),
    )


def fit_group_alphas(
    frame: pd.DataFrame,
    grid: np.ndarray,
    group_columns: Sequence[str],
) -> tuple[dict[tuple[object, ...], float], dict[str, dict[str, float | int]]]:
    mapping: dict[tuple[object, ...], float] = {}
    diagnostics: dict[str, dict[str, float | int]] = {}
    grouped = frame.loc[frame["lead_h"].isin(ACTIVE_LEADS)].groupby(
        list(group_columns), observed=True, sort=True
    )
    for raw_key, group in grouped:
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        fitted = fit_alpha(group, grid)
        mapping[key] = fitted.grid
        label = "|".join(str(value) for value in key)
        diagnostics[label] = {
            "analytic_alpha": fitted.analytic,
            "clipped_alpha": fitted.clipped,
            "grid_alpha": fitted.grid,
            "grid_rmse": fitted.grid_rmse,
            "rows": fitted.rows,
        }
    if not mapping:
        raise KMAAlphaSurfaceError("no active long-lead groups were fitted")
    return mapping, diagnostics


def predict_with_mapping(
    frame: pd.DataFrame,
    mapping: Mapping[tuple[object, ...], float],
    group_columns: Sequence[str],
) -> np.ndarray:
    prediction = frame["base"].to_numpy(dtype=np.float64).copy()
    active = frame["lead_h"].isin(ACTIVE_LEADS).to_numpy()
    for index in np.flatnonzero(active):
        key = tuple(frame.iloc[index][column] for column in group_columns)
        if key not in mapping:
            raise KMAAlphaSurfaceError(f"missing alpha mapping for {key}")
        prediction[index] += float(mapping[key]) * float(frame.iloc[index]["delta"])
    prediction = np.clip(prediction, 0.0, 30.0)
    if not np.isfinite(prediction).all():
        raise KMAAlphaSurfaceError("mapped prediction contains non-finite values")
    if not np.array_equal(
        prediction[~active], frame.loc[~active, "base"].to_numpy(dtype=np.float64)
    ):
        raise AssertionError("short-lead mapped prediction is not an exact no-op")
    return prediction


def blend_mappings(
    coarse: Mapping[tuple[object, ...], float],
    fine: Mapping[tuple[object, ...], float],
    *,
    shrink: float,
) -> dict[tuple[object, ...], float]:
    if not 0.0 <= shrink <= 1.0:
        raise KMAAlphaSurfaceError("shrink must be within [0,1]")
    result: dict[tuple[object, ...], float] = {}
    for key, fine_alpha in fine.items():
        lead_key = (key[-1],)
        if lead_key not in coarse:
            raise KMAAlphaSurfaceError(f"missing coarse alpha for {lead_key}")
        result[key] = (1.0 - shrink) * float(coarse[lead_key]) + shrink * float(fine_alpha)
    return result


def crossfit_predictions(
    frame: pd.DataFrame,
    grid: np.ndarray,
    *,
    strategy: str,
    shrink: float = 0.0,
) -> tuple[np.ndarray, dict[str, object]]:
    folds = sorted(str(value) for value in frame["fold"].unique())
    if len(folds) < 2:
        raise KMAAlphaSurfaceError("cross-fit requires at least two folds")
    output = np.full(len(frame), np.nan, dtype=np.float64)
    fitted_by_fold: dict[str, object] = {}
    for heldout in folds:
        train = frame.loc[frame["fold"].astype(str).ne(heldout)].copy()
        test_mask = frame["fold"].astype(str).eq(heldout).to_numpy()
        test = frame.loc[test_mask].copy()
        if strategy == "uniform":
            fit = fit_alpha(train, grid, mask=train["lead_h"].isin(ACTIVE_LEADS))
            mapping = {(18,): fit.grid, (24,): fit.grid}
            group_columns = ("lead_h",)
            diagnostics: object = {"uniform": fit.grid}
        elif strategy == "lead":
            mapping, diagnostics = fit_group_alphas(train, grid, ("lead_h",))
            group_columns = ("lead_h",)
        elif strategy in {"station_lead", "hierarchical"}:
            lead_mapping, lead_diagnostics = fit_group_alphas(train, grid, ("lead_h",))
            station_mapping, station_diagnostics = fit_group_alphas(
                train, grid, ("station", "lead_h")
            )
            mapping = (
                station_mapping
                if strategy == "station_lead"
                else blend_mappings(lead_mapping, station_mapping, shrink=shrink)
            )
            group_columns = ("station", "lead_h")
            diagnostics = {
                "lead": lead_diagnostics,
                "station_lead": station_diagnostics,
                "shrink": float(1.0 if strategy == "station_lead" else shrink),
            }
        else:
            raise KMAAlphaSurfaceError(f"unknown strategy: {strategy}")
        output[test_mask] = predict_with_mapping(test, mapping, group_columns)
        fitted_by_fold[heldout] = diagnostics
    if not np.isfinite(output).all():
        raise KMAAlphaSurfaceError("cross-fit left unfilled predictions")
    return output, fitted_by_fold


def metric_breakdown(frame: pd.DataFrame, prediction: Sequence[float]) -> dict[str, object]:
    work = frame.copy()
    work["candidate"] = np.asarray(prediction, dtype=np.float64)
    base_rmse = rmse(work["target_hs"], work["base"])
    candidate_rmse = rmse(work["target_hs"], work["candidate"])
    result: dict[str, object] = {
        "base_rmse": base_rmse,
        "candidate_rmse": candidate_rmse,
        "delta_rmse": candidate_rmse - base_rmse,
        "rows": int(len(work)),
        "cases": int(work.groupby(["fold", "anchor_id"], observed=True).ngroups),
    }
    for column, name in (("fold", "by_fold"), ("station", "by_station"), ("lead_h", "by_lead")):
        slices: dict[str, dict[str, float | int]] = {}
        for key, group in work.groupby(column, observed=True, sort=True):
            before = rmse(group["target_hs"], group["base"])
            after = rmse(group["target_hs"], group["candidate"])
            slices[str(key)] = {
                "base_rmse": before,
                "candidate_rmse": after,
                "delta_rmse": after - before,
                "rows": int(len(group)),
            }
        result[name] = slices
    return result


def paired_case_bootstrap(
    frame: pd.DataFrame,
    prediction: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    work = frame.copy()
    work["candidate"] = np.asarray(prediction, dtype=np.float64)
    grouped = list(work.groupby(["fold", "anchor_id"], observed=True, sort=True))
    if not grouped or replicates <= 0:
        raise KMAAlphaSurfaceError("bootstrap configuration is invalid")
    base_sse = np.asarray(
        [np.square(group["base"] - group["target_hs"]).sum() for _, group in grouped],
        dtype=np.float64,
    )
    candidate_sse = np.asarray(
        [np.square(group["candidate"] - group["target_hs"]).sum() for _, group in grouped],
        dtype=np.float64,
    )
    counts = np.asarray([len(group) for _, group in grouped], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        draw = rng.integers(0, len(grouped), size=len(grouped))
        denominator = float(counts[draw].sum())
        deltas[index] = np.sqrt(candidate_sse[draw].sum() / denominator) - np.sqrt(
            base_sse[draw].sum() / denominator
        )
    return {
        "replicates": int(replicates),
        "case_count": int(len(grouped)),
        "mean_delta_rmse": float(deltas.mean()),
        "median_delta_rmse": float(np.median(deltas)),
        "ci90_lower": float(np.quantile(deltas, 0.05)),
        "ci90_upper": float(np.quantile(deltas, 0.95)),
        "probability_improvement": float(np.mean(deltas < 0.0)),
    }


def exhaustive_lead_surface(
    frame: pd.DataFrame, grid: np.ndarray
) -> dict[str, float | int | list[dict[str, float]]]:
    truth = frame["target_hs"].to_numpy(dtype=np.float64)
    base = frame["base"].to_numpy(dtype=np.float64)
    delta = frame["delta"].to_numpy(dtype=np.float64)
    residual = base - truth
    fixed_mask = ~frame["lead_h"].isin(ACTIVE_LEADS).to_numpy()
    fixed_sse = float(np.square(residual[fixed_mask]).sum())
    lead_sse: dict[int, np.ndarray] = {}
    for lead in ACTIVE_LEADS:
        mask = frame["lead_h"].eq(lead).to_numpy()
        lead_residual = residual[mask]
        lead_delta = delta[mask]
        lead_sse[lead] = (
            np.square(lead_residual).sum()
            + 2.0 * grid * np.dot(lead_residual, lead_delta)
            + np.square(grid) * np.square(lead_delta).sum()
        )
    surface = fixed_sse + lead_sse[18][:, None] + lead_sse[24][None, :]
    flat_order = np.argsort(surface, axis=None, kind="stable")
    top: list[dict[str, float]] = []
    for flat_index in flat_order[:20]:
        i, j = np.unravel_index(int(flat_index), surface.shape)
        top.append(
            {
                "alpha_18": float(grid[i]),
                "alpha_24": float(grid[j]),
                "rmse": float(np.sqrt(surface[i, j] / len(frame))),
            }
        )
    best = top[0]
    return {
        "grid_size_each_axis": int(len(grid)),
        "evaluated_pairs": int(surface.size),
        "best_alpha_18": best["alpha_18"],
        "best_alpha_24": best["alpha_24"],
        "best_rmse": best["rmse"],
        "top20": top,
    }


def fold_robust_lead_surface(frame: pd.DataFrame, grid: np.ndarray) -> dict[str, object]:
    """Exhaustively locate lead pairs that are stable across all frozen folds."""

    fold_delta_surfaces: list[np.ndarray] = []
    folds: list[str] = []
    for fold, group in frame.groupby("fold", observed=True, sort=True):
        truth = group["target_hs"].to_numpy(dtype=np.float64)
        base = group["base"].to_numpy(dtype=np.float64)
        delta = group["delta"].to_numpy(dtype=np.float64)
        residual = base - truth
        fixed = ~group["lead_h"].isin(ACTIVE_LEADS).to_numpy()
        fixed_sse = float(np.square(residual[fixed]).sum())
        lead_sse: dict[int, np.ndarray] = {}
        for lead in ACTIVE_LEADS:
            mask = group["lead_h"].eq(lead).to_numpy()
            lead_residual = residual[mask]
            lead_delta = delta[mask]
            lead_sse[lead] = (
                np.square(lead_residual).sum()
                + 2.0 * grid * np.dot(lead_residual, lead_delta)
                + np.square(grid) * np.square(lead_delta).sum()
            )
        mse_surface = (
            fixed_sse + lead_sse[18][:, None] + lead_sse[24][None, :]
        ) / len(group)
        rmse_surface = np.sqrt(np.maximum(mse_surface, 0.0))
        fold_delta_surfaces.append(rmse_surface - rmse(truth, base))
        folds.append(str(fold))
    stacked = np.stack(fold_delta_surfaces, axis=0)
    all_improve = np.all(stacked < 0.0, axis=0)
    non_degrade = np.all(stacked <= 1e-15, axis=0)
    maximum_degradation = np.max(stacked, axis=0)
    mean_delta = np.mean(stacked, axis=0)
    minimax_index = np.unravel_index(int(np.argmin(maximum_degradation)), maximum_degradation.shape)
    robust_records: list[dict[str, object]] = []
    if all_improve.any():
        robust_indices = np.argwhere(all_improve)
        order = np.argsort(
            np.asarray([mean_delta[tuple(index)] for index in robust_indices]), kind="stable"
        )
        for order_index in order[:20]:
            i, j = (int(value) for value in robust_indices[int(order_index)])
            robust_records.append(
                {
                    "alpha_18": float(grid[i]),
                    "alpha_24": float(grid[j]),
                    "mean_fold_delta_rmse": float(mean_delta[i, j]),
                    "worst_fold_delta_rmse": float(maximum_degradation[i, j]),
                    "delta_by_fold": {
                        fold: float(stacked[fold_index, i, j])
                        for fold_index, fold in enumerate(folds)
                    },
                }
            )
    i, j = minimax_index
    return {
        "folds": folds,
        "evaluated_pairs": int(grid.size * grid.size),
        "strictly_improves_every_fold_pairs": int(all_improve.sum()),
        "non_degrades_every_fold_pairs": int(non_degrade.sum()),
        "best_all_fold_improvement_pairs": robust_records,
        "minimax_pair": {
            "alpha_18": float(grid[i]),
            "alpha_24": float(grid[j]),
            "mean_fold_delta_rmse": float(mean_delta[i, j]),
            "worst_fold_delta_rmse": float(maximum_degradation[i, j]),
            "delta_by_fold": {
                fold: float(stacked[fold_index, i, j])
                for fold_index, fold in enumerate(folds)
            },
        },
    }


def apply_official_correction(
    current: pd.DataFrame,
    old: pd.DataFrame,
    kma_alpha40: pd.DataFrame,
    *,
    alpha_by_lead: Mapping[int, float],
    reference_alpha: float = 0.4,
) -> pd.DataFrame:
    columns = ["case_id", "station", "lead_h", "hs_pred"]
    for role, frame in (("current", current), ("old", old), ("kma", kma_alpha40)):
        if list(frame.columns) != columns:
            raise KMAAlphaSurfaceError(f"{role} official frame schema changed")
        if frame.empty or frame.duplicated(columns[:3]).any():
            raise KMAAlphaSurfaceError(f"{role} official frame is empty or duplicated")
    if not current[columns[:3]].equals(old[columns[:3]]) or not current[columns[:3]].equals(
        kma_alpha40[columns[:3]]
    ):
        raise KMAAlphaSurfaceError("official frame key order differs")
    if reference_alpha <= 0.0:
        raise KMAAlphaSurfaceError("reference alpha must be positive")
    result = current.copy()
    current_value = current["hs_pred"].to_numpy(dtype=np.float64)
    correction = (
        kma_alpha40["hs_pred"].to_numpy(dtype=np.float64)
        - old["hs_pred"].to_numpy(dtype=np.float64)
    )
    values = current_value.copy()
    for lead, alpha in alpha_by_lead.items():
        if int(lead) not in ACTIVE_LEADS or not np.isfinite(alpha):
            raise KMAAlphaSurfaceError("official alpha mapping contains an invalid entry")
        mask = current["lead_h"].eq(int(lead)).to_numpy()
        values[mask] += float(alpha) / float(reference_alpha) * correction[mask]
    values = np.clip(values, 0.0, 30.0)
    inactive = ~current["lead_h"].isin(ACTIVE_LEADS).to_numpy()
    if not np.array_equal(values[inactive], current_value[inactive]):
        raise AssertionError("official short-lead values are not exact no-ops")
    if not np.isfinite(values).all():
        raise KMAAlphaSurfaceError("official candidate contains non-finite values")
    result["hs_pred"] = values
    return result


__all__ = [
    "ACTIVE_LEADS",
    "ALL_LEADS",
    "AlphaFit",
    "KMAAlphaSurfaceError",
    "apply_official_correction",
    "blend_mappings",
    "crossfit_predictions",
    "exhaustive_lead_surface",
    "fold_robust_lead_surface",
    "fit_alpha",
    "fit_group_alphas",
    "make_alpha_grid",
    "metric_breakdown",
    "paired_case_bootstrap",
    "predict_with_mapping",
    "prepare_oof_frame",
    "rmse",
]
