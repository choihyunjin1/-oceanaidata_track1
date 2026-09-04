"""Research-only P2 shrinkage conditional-profile probe.

The probe uses observations.csv only.  For each already-exposed 2025
two-month block it masks target layers 2--4, fits an OAS-shrunk joint Gaussian
model to public T/S, calendar harmonics, and hidden-layer T/S outside the
block, and evaluates the closed-form conditional mean.  It never reads the
official answer and never creates a submission.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import OAS

from p2_restore.profile_projection import (
    project_profiles_vectorized,
    public_endpoint_frame,
)


DATA = Path(r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore\observations.csv")
REFERENCE = Path("artifacts/p2_extrapolated_soft_gate_v2/oof.parquet")
OUTPUT = Path(
    os.environ.get(
        "P2_OAS_OUTPUT",
        "artifacts/p2_oas_conditional_profile_20260827_v3/result.json",
    )
)
PUBLIC = tuple(
    int(value) for value in os.environ.get("P2_OAS_PUBLIC_LAYERS", "1,5,6,7").split(",")
)
TARGET = (2, 3, 4)
BLOCKS = {
    "outer_2024_sep_oct": ("2024-09-01", "2024-11-01"),
    "outer_2025_jul_aug": ("2025-07-01", "2025-09-01"),
    "outer_2025_nov_dec": ("2025-11-01", "2026-01-01"),
}
ALL_SCORED: list[pd.DataFrame] = []


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def build_panel(observations: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    temp = observations.pivot(index="time", columns="layer", values="temp").sort_index()
    psal = observations.pivot(index="time", columns="layer", values="psal").sort_index()
    panel = pd.DataFrame(index=temp.index)
    x_columns: list[str] = []
    y_columns: list[str] = []
    for layer in PUBLIC:
        for name, values in (("temp", temp), ("psal", psal)):
            column = f"{name}_{layer}"
            panel[column] = values[layer]
            x_columns.append(column)
    local = panel.index.tz_convert("Asia/Seoul")
    minute = local.hour.to_numpy() * 60 + local.minute.to_numpy()
    doy = local.dayofyear.to_numpy() + minute / 1440.0
    for harmonic in (1, 2, 3, 4):
        for kind, fn in (("sin", np.sin), ("cos", np.cos)):
            column = f"doy_{kind}_{harmonic}"
            panel[column] = fn(2 * np.pi * harmonic * doy / 365.2425)
            x_columns.append(column)
    for layer in TARGET:
        for name, values in (("temp", temp), ("psal", psal)):
            column = f"{name}_{layer}"
            panel[column] = values[layer]
            y_columns.append(column)
    return panel, x_columns, y_columns


def fit_fold(
    name: str,
    start_text: str,
    stop_text: str,
    panel: pd.DataFrame,
    x_columns: list[str],
    y_columns: list[str],
    reference: pd.DataFrame,
) -> dict[str, object]:
    start = pd.Timestamp(start_text, tz="Asia/Seoul").tz_convert("UTC")
    stop = pd.Timestamp(stop_text, tz="Asia/Seoul").tz_convert("UTC")
    in_block = (panel.index >= start) & (panel.index < stop)
    evaluate = panel.loc[in_block, x_columns + y_columns].copy()
    nx = len(x_columns)
    x = evaluate[x_columns].to_numpy(float)
    finite_patterns = np.isfinite(x)
    yhat_z = np.full((len(evaluate), len(y_columns)), np.nan, dtype=float)
    yhat = np.full_like(yhat_z, np.nan)
    local_eval = evaluate.index.tz_convert("Asia/Seoul")
    season_bins = ((local_eval.dayofyear.to_numpy() - 1) // 14).astype(int)
    train_index = panel.index[~in_block]
    train_doy = train_index.tz_convert("Asia/Seoul").dayofyear.to_numpy(float)
    fit_receipts: list[dict[str, float | int]] = []
    for season_bin in np.unique(season_bins):
        center = float(season_bin * 14 + 7.5)
        distance = np.abs(train_doy - center)
        distance = np.minimum(distance, 365.2425 - distance)
        train = panel.loc[
            train_index[distance <= 60.0], x_columns + y_columns
        ].dropna()
        values = train.to_numpy(float)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale == 0] = 1.0
        standardized = (values - mean) / scale
        estimator = OAS(store_precision=False, assume_centered=False).fit(standardized)
        covariance = estimator.covariance_
        sigma_xx = covariance[:nx, :nx]
        sigma_yx = covariance[nx:, :nx]
        bin_rows = np.flatnonzero(season_bins == season_bin)
        for pattern in np.unique(finite_patterns[bin_rows], axis=0):
            local_keep = np.all(finite_patterns[bin_rows] == pattern, axis=1)
            row_ids = bin_rows[local_keep]
            observed = np.flatnonzero(pattern)
            if len(observed) == 0:
                yhat_z[row_ids] = 0.0
            else:
                conditional = sigma_yx[:, observed] @ np.linalg.pinv(
                    sigma_xx[np.ix_(observed, observed)], rcond=1e-10
                )
                xz = (x[np.ix_(row_ids, observed)] - mean[observed]) / scale[observed]
                yhat_z[row_ids] = xz @ conditional.T
            yhat[row_ids] = mean[nx:] + yhat_z[row_ids] * scale[nx:]
        fit_receipts.append(
            {
                "season_bin": int(season_bin),
                "center_doy": center,
                "train_timestamps": len(train),
                "oas_shrinkage": float(estimator.shrinkage_),
            }
        )

    rows: list[pd.DataFrame] = []
    for target_position, layer in enumerate(TARGET):
        truth_column = y_columns.index(f"temp_{layer}")
        part = pd.DataFrame(
            {
                "time": evaluate.index,
                "layer": layer,
                "truth": evaluate[f"temp_{layer}"].to_numpy(float),
                "prediction": yhat[:, truth_column],
                "public_gradient_1_5": (
                    evaluate["temp_1"] - evaluate["temp_5"]
                ).to_numpy(float),
            }
        )
        rows.append(part)
    scored = pd.concat(rows, ignore_index=True).dropna(subset=["truth"])
    ref = reference.loc[
        reference["block"] == name.removeprefix("outer_"),
        ["time", "layer", "prediction"],
    ].copy()
    ref = ref.rename(columns={"prediction": "reference_prediction"})
    ref["time"] = pd.to_datetime(ref["time"], utc=True)
    scored = scored.merge(ref, on=["time", "layer"], how="inner", validate="one_to_one")
    truth = scored["truth"].to_numpy(float)
    candidate = scored["prediction"].to_numpy(float)
    incumbent = scored["reference_prediction"].to_numpy(float)
    direction = candidate - incumbent
    scored["fold"] = name
    scored["reference"] = incumbent
    scored["candidate"] = candidate
    ALL_SCORED.append(scored.copy())
    denom = float(np.mean(direction**2))
    alpha = float(-np.mean((incumbent - truth) * direction) / denom) if denom else 0.0
    by_layer: dict[str, object] = {}
    for layer in TARGET:
        keep = scored["layer"].to_numpy(int) == layer
        layer_direction = direction[keep]
        layer_denom = float(np.mean(layer_direction**2))
        layer_alpha = (
            float(-np.mean((incumbent[keep] - truth[keep]) * layer_direction) / layer_denom)
            if layer_denom
            else 0.0
        )
        by_layer[str(layer)] = {
            "rows": int(keep.sum()),
            "reference_rmse": rmse(truth[keep], incumbent[keep]),
            "candidate_rmse": rmse(truth[keep], candidate[keep]),
            "blend_0.1_rmse": rmse(
                truth[keep], incumbent[keep] + 0.1 * layer_direction
            ),
            "oracle_alpha": layer_alpha,
            "oracle_blend_rmse": rmse(
                truth[keep], incumbent[keep] + layer_alpha * layer_direction
            ),
        }

    scored["week"] = (
        (scored["time"].dt.tz_convert("Asia/Seoul") - start.tz_convert("Asia/Seoul"))
        .dt.days.floordiv(7)
    )
    scored["gradient_bin"] = pd.cut(
        scored["public_gradient_1_5"],
        bins=[-np.inf, -1.0, 0.0, 1.0, 2.0, np.inf],
        labels=["lt_-1", "-1_to_0", "0_to_1", "1_to_2", "ge_2"],
    )

    def grouped_diagnostics(column: str) -> dict[str, object]:
        diagnostics: dict[str, object] = {}
        for value, group in scored.groupby(column, observed=True):
            ids = group.index.to_numpy(int)
            group_direction = direction[ids]
            group_denom = float(np.mean(group_direction**2))
            group_alpha = (
                float(
                    -np.mean((incumbent[ids] - truth[ids]) * group_direction)
                    / group_denom
                )
                if group_denom
                else 0.0
            )
            diagnostics[str(value)] = {
                "rows": len(ids),
                "reference_rmse": rmse(truth[ids], incumbent[ids]),
                "blend_0.1_rmse": rmse(
                    truth[ids], incumbent[ids] + 0.1 * group_direction
                ),
                "oracle_alpha": group_alpha,
            }
        return diagnostics
    return {
        "seasonal_window_days": 60,
        "season_bin_days": 14,
        "fit_receipts": fit_receipts,
        "rows": len(scored),
        "reference_rmse": rmse(truth, incumbent),
        "candidate_rmse": rmse(truth, candidate),
        "fixed_blends": {
            str(weight): rmse(truth, incumbent + weight * direction)
            for weight in (0.05, 0.1, 0.25, 0.5)
        },
        "oracle_alpha": alpha,
        "oracle_blend_rmse": rmse(truth, incumbent + alpha * direction),
        "direction_rms": float(np.sqrt(denom)),
        "by_layer": by_layer,
        "by_week": grouped_diagnostics("week"),
        "by_public_gradient_1_5": grouped_diagnostics("gradient_bin"),
    }


def main() -> None:
    observations = pd.read_csv(DATA)
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    panel, x_columns, y_columns = build_panel(observations)
    reference = pd.read_parquet(REFERENCE)
    fold_results = {
        name: fit_fold(name, start, stop, panel, x_columns, y_columns, reference)
        for name, (start, stop) in BLOCKS.items()
    }
    scored = pd.concat(ALL_SCORED, ignore_index=True)
    truth = scored["truth"].to_numpy(float)
    incumbent = scored["reference"].to_numpy(float)
    direction = scored["candidate"].to_numpy(float) - incumbent
    denom = float(np.mean(direction**2))
    alpha = float(-np.mean((incumbent - truth) * direction) / denom)
    fixed = incumbent + 0.1 * direction
    endpoints = public_endpoint_frame(observations)
    projection = project_profiles_vectorized(
        scored[["time", "layer"]], fixed, endpoints
    )
    fixed_projected = projection.prediction
    scored["day"] = scored["time"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    daily = scored.assign(
        se_reference=(incumbent - truth) ** 2,
        se_fixed=(fixed - truth) ** 2,
    ).groupby("day", sort=True).agg(
        rows=("truth", "size"),
        se_reference=("se_reference", "sum"),
        se_fixed=("se_fixed", "sum"),
    )
    rng = np.random.default_rng(20260827)
    draws = np.empty(5000, dtype=float)
    daily_values = daily.to_numpy(float)
    for draw in range(len(draws)):
        ids = rng.integers(0, len(daily_values), size=len(daily_values))
        sample = daily_values[ids]
        count = sample[:, 0].sum()
        draws[draw] = np.sqrt(sample[:, 2].sum() / count) - np.sqrt(
            sample[:, 1].sum() / count
        )
    result = {
        "schema_version": "p2.oas_conditional_profile.research.20260827.v3",
        "reference": "frozen P2_EXTRAPOLATED_SOFT_GATE_V2 OOF",
        "status": "RESEARCH_ONLY_EXPOSED_BLOCKS_NO_SUBMISSION",
        "hypothesis": (
            "An OAS-shrunk joint T/S conditional mean can recover stable vertical "
            "profile information from the contemporaneously observed public layers."
        ),
        "feature_contract": {"x": x_columns, "y": y_columns},
        "folds": fold_results,
        "aggregate": {
            "rows": len(scored),
            "reference_rmse": rmse(truth, incumbent),
            "blend_0.1_rmse": rmse(truth, fixed),
            "blend_0.1_delta_rmse": rmse(truth, fixed) - rmse(truth, incumbent),
            "blend_0.1_projected_rmse": rmse(truth, fixed_projected),
            "blend_0.1_projected_delta_rmse": (
                rmse(truth, fixed_projected) - rmse(truth, incumbent)
            ),
            "projection_active_rows": int(projection.active_mask.sum()),
            "projection_active_share": float(projection.active_mask.mean()),
            "oracle_alpha": alpha,
            "oracle_blend_rmse": rmse(truth, incumbent + alpha * direction),
        },
        "paired_kst_day_bootstrap_blend_0.1": {
            "replicates": len(draws),
            "days": len(daily),
            "mean_delta_rmse": float(draws.mean()),
            "ci90_low": float(np.quantile(draws, 0.05)),
            "ci90_high": float(np.quantile(draws, 0.95)),
            "probability_improved": float(np.mean(draws < 0)),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(OUTPUT.parent / "oof.parquet", index=False)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
