"""Research-only P2 probe for a boundary-registered previous-year prior.

This script never reads the official answer or writes a submission.  It masks
two 2025 two-month blocks, uses only observations outside each block plus the
continuously observed public layers, and compares a previous-calendar-year
profile registered to the hidden-layer observations immediately before/after
the block.  The experiment is intentionally fixed to a seven-day boundary
window and two already-defined outer blocks; results are research evidence,
not a fresh confirmation surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


DATA = Path(r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore\observations.csv")
REFERENCE = Path("artifacts/p2_architecture_matched_reference_v3/reference_oof_100.csv")
OUTPUT = Path("artifacts/p2_boundary_registered_prior_20260827_v1/result.json")
PUBLIC = (1, 5, 6, 7, 8)
TARGET = (2, 3, 4)
BLOCKS = {
    "outer_2025_may_jun": ("2025-05-01", "2025-07-01"),
    "outer_2025_jul_aug": ("2025-07-01", "2025-09-01"),
}


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def previous_year(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index - pd.DateOffset(years=1)


def take_previous(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    out = frame.reindex(previous_year(index)).copy()
    out.index = index
    return out


def build_feature_rows(
    index: pd.DatetimeIndex,
    layer: int,
    temp: pd.DataFrame,
    psal: pd.DataFrame,
    registered: pd.Series,
) -> pd.DataFrame:
    now_t = temp.reindex(index)
    now_s = psal.reindex(index)
    old_t = take_previous(temp, index)
    old_s = take_previous(psal, index)
    local = index.tz_convert("Asia/Seoul")
    minute = local.hour.to_numpy() * 60 + local.minute.to_numpy()
    doy = local.dayofyear.to_numpy() + minute / 1440.0
    values: dict[str, object] = {
        "layer": np.full(len(index), layer, dtype=float),
        "prior_target_t": old_t[layer].to_numpy(float),
        "prior_target_s": old_s[layer].to_numpy(float),
        "registered_prior_t": registered.reindex(index).to_numpy(float),
        "doy_sin": np.sin(2 * np.pi * doy / 365.2425),
        "doy_cos": np.cos(2 * np.pi * doy / 365.2425),
        "hour_sin": np.sin(2 * np.pi * minute / 1440.0),
        "hour_cos": np.cos(2 * np.pi * minute / 1440.0),
    }
    for public_layer in PUBLIC:
        values[f"now_t_{public_layer}"] = now_t[public_layer].to_numpy(float)
        values[f"now_s_{public_layer}"] = now_s[public_layer].to_numpy(float)
        values[f"old_t_{public_layer}"] = old_t[public_layer].to_numpy(float)
        values[f"old_s_{public_layer}"] = old_s[public_layer].to_numpy(float)
        values[f"delta_t_{public_layer}"] = (
            now_t[public_layer] - old_t[public_layer]
        ).to_numpy(float)
        values[f"delta_s_{public_layer}"] = (
            now_s[public_layer] - old_s[public_layer]
        ).to_numpy(float)
    return pd.DataFrame(values, index=index)


def registered_prior(
    *,
    index: pd.DatetimeIndex,
    layer: int,
    temp: pd.DataFrame,
    start: pd.Timestamp,
    stop: pd.Timestamp,
) -> tuple[pd.Series, dict[str, float]]:
    prior = take_previous(temp[[layer]], index)[layer]
    window = pd.Timedelta(days=7)
    left = temp.index[(temp.index >= start - window) & (temp.index < start)]
    right = temp.index[(temp.index >= stop) & (temp.index < stop + window)]
    left_delta = temp.loc[left, layer] - take_previous(temp[[layer]], left)[layer]
    right_delta = temp.loc[right, layer] - take_previous(temp[[layer]], right)[layer]
    left_values = left_delta.to_numpy(float)
    right_values = right_delta.to_numpy(float)
    left_bias = float(np.nanmedian(left_values)) if np.isfinite(left_values).any() else 0.0
    right_bias = float(np.nanmedian(right_values)) if np.isfinite(right_values).any() else 0.0
    fraction = ((index - start) / (stop - start)).to_numpy(float)
    bias = left_bias + fraction * (right_bias - left_bias)
    return prior + bias, {"left_bias": left_bias, "right_bias": right_bias}


def fit_fold(
    name: str,
    start_text: str,
    stop_text: str,
    temp: pd.DataFrame,
    psal: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict[str, object]:
    start = pd.Timestamp(start_text, tz="Asia/Seoul").tz_convert("UTC")
    stop = pd.Timestamp(stop_text, tz="Asia/Seoul").tz_convert("UTC")
    block_index = temp.index[(temp.index >= start) & (temp.index < stop)]
    train_index = temp.index[
        (temp.index.year == 2025) & ~((temp.index >= start) & (temp.index < stop))
    ]

    train_parts: list[pd.DataFrame] = []
    eval_parts: list[pd.DataFrame] = []
    boundary_meta: dict[str, object] = {}
    for layer in TARGET:
        registered_all, biases = registered_prior(
            index=temp.index[temp.index.year == 2025],
            layer=layer,
            temp=temp,
            start=start,
            stop=stop,
        )
        registered_train = registered_all.reindex(train_index)
        registered_eval = registered_all.reindex(block_index)
        boundary_meta[str(layer)] = biases

        train = build_feature_rows(
            train_index, layer, temp, psal, registered_train
        )
        train["truth"] = temp.reindex(train_index)[layer].to_numpy(float)
        train["time"] = train_index.astype(str)
        train_parts.append(train)

        evaluate = build_feature_rows(
            block_index, layer, temp, psal, registered_eval
        )
        evaluate["truth"] = temp.reindex(block_index)[layer].to_numpy(float)
        evaluate["time"] = block_index.astype(str)
        eval_parts.append(evaluate)

    train = pd.concat(train_parts, ignore_index=True)
    evaluate = pd.concat(eval_parts, ignore_index=True)
    feature_columns = [
        column for column in train.columns if column not in {"truth", "time"}
    ]
    train_keep = np.isfinite(train["truth"].to_numpy(float)) & np.isfinite(
        train["prior_target_t"].to_numpy(float)
    )
    public_count = np.column_stack(
        [evaluate[f"now_t_{layer}"].to_numpy(float) for layer in PUBLIC]
    )
    eval_keep = (
        np.isfinite(evaluate["truth"].to_numpy(float))
        & np.isfinite(evaluate["prior_target_t"].to_numpy(float))
        & (np.isfinite(public_count).sum(axis=1) >= 2)
    )

    model = lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=7,
        min_child_samples=200,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=20260827,
        n_jobs=8,
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )
    model.fit(train.loc[train_keep, feature_columns], train.loc[train_keep, "truth"])
    evaluate = evaluate.loc[eval_keep].copy()
    evaluate["prediction"] = model.predict(evaluate[feature_columns])

    evaluate = evaluate.reset_index(drop=True)
    keys = evaluate[["time", "layer"]].copy()
    keys["time"] = pd.to_datetime(keys["time"], utc=True)
    ref = reference.loc[reference["fold"] == name, ["time", "layer", "prediction_mean"]].copy()
    ref["time"] = pd.to_datetime(ref["time"], utc=True)
    keys["row_id"] = np.arange(len(keys))
    merged = keys.merge(ref, on=["time", "layer"], how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError(f"reference coverage is empty for {name}")
    evaluate = evaluate.iloc[merged["row_id"].to_numpy(int)].reset_index(drop=True)
    evaluate["reference"] = merged["prediction_mean"].to_numpy(float)

    truth = evaluate["truth"].to_numpy(float)
    candidate = evaluate["prediction"].to_numpy(float)
    incumbent = evaluate["reference"].to_numpy(float)
    direction = candidate - incumbent
    denom = float(np.mean(direction**2))
    alpha_star = (
        float(-np.mean((incumbent - truth) * direction) / denom) if denom > 0 else 0.0
    )
    blended = incumbent + alpha_star * direction
    by_layer: dict[str, object] = {}
    for layer in TARGET:
        keep = evaluate["layer"].to_numpy(int) == layer
        by_layer[str(layer)] = {
            "rows": int(keep.sum()),
            "reference_rmse": rmse(truth[keep], incumbent[keep]),
            "candidate_rmse": rmse(truth[keep], candidate[keep]),
            "oracle_blend_rmse": rmse(truth[keep], blended[keep]),
        }
    return {
        "rows": len(evaluate),
        "train_rows": int(train_keep.sum()),
        "reference_rmse": rmse(truth, incumbent),
        "registered_prior_rmse": rmse(
            truth, evaluate["registered_prior_t"].to_numpy(float)
        ),
        "conditional_model_rmse": rmse(truth, candidate),
        "fixed_blends": {
            str(weight): rmse(truth, incumbent + weight * direction)
            for weight in (0.1, 0.25, 0.5)
        },
        "oracle_alpha": alpha_star,
        "oracle_blend_rmse": rmse(truth, blended),
        "direction_rms": float(np.sqrt(denom)),
        "by_layer": by_layer,
        "boundary_biases": boundary_meta,
        "feature_count": len(feature_columns),
    }


def main() -> None:
    observations = pd.read_csv(DATA)
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    temp = observations.pivot(index="time", columns="layer", values="temp").sort_index()
    psal = observations.pivot(index="time", columns="layer", values="psal").sort_index()
    reference = pd.read_csv(REFERENCE)
    result = {
        "schema_version": "p2.boundary_registered_prior.research.20260827.v1",
        "status": "RESEARCH_ONLY_EXPOSED_BLOCKS_NO_SUBMISSION",
        "hypothesis": (
            "A 61-day hidden-layer gap can be reconstructed by conditioning a previous-year "
            "thermohaline profile on current public layers and registering it to seven-day "
            "hidden-layer observations on both sides of the gap."
        ),
        "folds": {
            name: fit_fold(name, start, stop, temp, psal, reference)
            for name, (start, stop) in BLOCKS.items()
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
