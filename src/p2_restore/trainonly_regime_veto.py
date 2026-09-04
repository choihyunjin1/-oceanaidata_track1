"""Train-only seasonal conditional-benefit veto utilities for P2."""

from __future__ import annotations

import numpy as np
import pandas as pd


def kst_calendar_fields(times: pd.Series | pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Return KST day-of-year and stable calendar-day labels."""

    index = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).tz_convert("Asia/Seoul")
    return index.dayofyear.to_numpy(dtype=np.int16), index.normalize().asi8


def circular_day_distance(day_of_year: np.ndarray, center: float) -> np.ndarray:
    """Circular distance on a fixed 366-day calendar, preserving leap-day support."""

    values = np.asarray(day_of_year, dtype=np.float64)
    direct = np.abs(values - float(center))
    return np.minimum(direct, 366.0 - direct)


def season_bin(times: pd.Series | pd.DatetimeIndex, width_days: int) -> np.ndarray:
    if width_days <= 0:
        raise ValueError("season bin width must be positive")
    day_of_year, _ = kst_calendar_fields(times)
    return ((day_of_year.astype(np.int64) - 1) // int(width_days)).astype(np.int16)


def paired_day_bootstrap_delta_rmse(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap candidate-minus-reference RMSE by KST calendar day."""

    required = {"truth", "reference", "candidate", "kst_day"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"bootstrap frame missing: {sorted(missing)}")
    grouped = frame.assign(
        reference_se=np.square(frame["truth"].to_numpy() - frame["reference"].to_numpy()),
        candidate_se=np.square(frame["truth"].to_numpy() - frame["candidate"].to_numpy()),
    ).groupby("kst_day", sort=True)[["reference_se", "candidate_se"]].agg(["sum", "count"])
    reference_sum = grouped[("reference_se", "sum")].to_numpy(dtype=np.float64)
    candidate_sum = grouped[("candidate_se", "sum")].to_numpy(dtype=np.float64)
    counts = grouped[("reference_se", "count")].to_numpy(dtype=np.float64)
    days = len(grouped)
    if days == 0 or replicates <= 0:
        raise ValueError("bootstrap needs days and positive replicates")
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draw = rng.integers(0, days, size=days)
        count = float(counts[draw].sum())
        baseline = np.sqrt(float(reference_sum[draw].sum()) / count)
        candidate = np.sqrt(float(candidate_sum[draw].sum()) / count)
        deltas[index] = candidate - baseline
    observed = float(
        np.sqrt(np.mean(np.square(frame["truth"] - frame["candidate"])))
        - np.sqrt(np.mean(np.square(frame["truth"] - frame["reference"])))
    )
    return {
        "days": days,
        "replicates": int(replicates),
        "observed_delta_rmse": observed,
        "ci90_low": float(np.quantile(deltas, 0.05)),
        "ci90_high": float(np.quantile(deltas, 0.95)),
        "probability_improved": float(np.mean(deltas < 0.0)),
    }


def trainonly_regime_decisions(
    inner_oof: pd.DataFrame,
    query_times: pd.DatetimeIndex,
    *,
    bin_days: int,
    window_days: float,
    minimum_source_blocks: int,
    minimum_profiles: int,
    minimum_kst_days: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    ci90_upper_below: float,
) -> tuple[dict[int, bool], dict[str, dict[str, float | int | bool | None]]]:
    """Learn a seasonal abstention map from outer-train inner-OOF losses only."""

    required = {"time", "source_block", "truth", "reference", "candidate"}
    if missing := required.difference(inner_oof.columns):
        raise ValueError(f"inner OOF missing: {sorted(missing)}")
    source = inner_oof.copy()
    source["time"] = pd.to_datetime(source["time"], utc=True)
    source_doy, source_day = kst_calendar_fields(source["time"])
    source["kst_doy"] = source_doy
    source["kst_day"] = source_day
    query_bins = season_bin(query_times, bin_days)
    decisions: dict[int, bool] = {}
    receipts: dict[str, dict[str, float | int | bool | None]] = {}
    for current_bin in sorted(map(int, np.unique(query_bins))):
        center = current_bin * int(bin_days) + (int(bin_days) + 1) / 2.0
        selected = source.loc[circular_day_distance(source["kst_doy"].to_numpy(), center) <= window_days]
        profiles = int(selected["time"].nunique())
        days = int(selected["kst_day"].nunique())
        blocks = int(selected["source_block"].nunique())
        support_ok = (
            blocks >= minimum_source_blocks
            and profiles >= minimum_profiles
            and days >= minimum_kst_days
        )
        bootstrap: dict[str, float | int] | None = None
        if support_ok:
            bootstrap = paired_day_bootstrap_delta_rmse(
                selected,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + current_bin,
            )
        enabled = bool(support_ok and bootstrap is not None and bootstrap["ci90_high"] < ci90_upper_below)
        decisions[current_bin] = enabled
        receipts[str(current_bin)] = {
            "center_kst_doy": float(center),
            "source_blocks": blocks,
            "profiles": profiles,
            "kst_days": days,
            "support_ok": bool(support_ok),
            "ci90_low": None if bootstrap is None else float(bootstrap["ci90_low"]),
            "ci90_high": None if bootstrap is None else float(bootstrap["ci90_high"]),
            "observed_delta_rmse": None if bootstrap is None else float(bootstrap["observed_delta_rmse"]),
            "enabled": enabled,
        }
    return decisions, receipts


__all__ = [
    "circular_day_distance",
    "kst_calendar_fields",
    "paired_day_bootstrap_delta_rmse",
    "season_bin",
    "trainonly_regime_decisions",
]
