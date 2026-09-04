"""Audit whether the frozen P3 local OOF represents the official hidden test.

This audit is deliberately label-free on the official 200 cases.  It reads only
the published context and index, and it never fits a model, writes predictions,
touches a submission, or performs an upload.  Local target access is restricted
to the already-scored frozen OOF and its immutable anchor cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
LEADS = (3, 6, 9, 12, 18, 24)
STATIONS = ("G-ORS", "I-ORS", "S-ORS")
WINDOWS = (
    ("2024_h2_storm", "2024-07-01", "2024-11-01"),
    ("winter_transition", "2024-11-01", "2025-03-01"),
    ("2025_h1", "2025-03-01", "2025-06-25"),
)
WAVE_COLUMNS = ("hs", "tp", "hmax", "wvdir")
ATMOS_COLUMNS = ("wspd", "gust", "wdir", "airt", "relh", "caph")
OOF_RELATIVE = Path("artifacts/p3/long_persistence_shrink/oof.parquet")
ANCHORS_RELATIVE = Path("artifacts/p3/features_all20_v1/train_anchors.parquet")
OUTPUT_RELATIVE = Path("artifacts/validation_system_audit_20260822/p3.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, data_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        try:
            return f"official_source/{resolved.relative_to(data_dir.resolve()).as_posix()}"
        except ValueError as exc:  # pragma: no cover - fail-closed guard
            raise ValueError(f"input lies outside registered roots: {path.name}") from exc


def _input_record(path: Path, *, data_dir: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "identity": _identity(path, data_dir=data_dir),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _rmse(truth: Iterable[float], prediction: Iterable[float]) -> float:
    y = np.asarray(list(truth), dtype=np.float64)
    p = np.asarray(list(prediction), dtype=np.float64)
    if y.shape != p.shape or y.size == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("RMSE inputs are empty, non-finite, or shape-mismatched")
    return float(np.sqrt(np.mean(np.square(p - y))))


def _metric_slices(frame: pd.DataFrame, prediction: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rows": int(len(frame)),
        "cases": int(frame["anchor_id"].nunique()),
        "rmse_m": _rmse(frame["target_hs"], frame[prediction]),
        "bias_m": float(
            np.mean(frame[prediction].to_numpy(float) - frame["target_hs"].to_numpy(float))
        ),
        "mae_m": float(
            np.mean(np.abs(frame[prediction].to_numpy(float) - frame["target_hs"].to_numpy(float)))
        ),
    }
    for column, name in (("fold", "by_fold"), ("station", "by_station"), ("lead_h", "by_lead")):
        result[name] = {
            str(key): {
                "rows": int(len(group)),
                "rmse_m": _rmse(group["target_hs"], group[prediction]),
            }
            for key, group in frame.groupby(column, observed=True, sort=True)
        }
    return result


def _case_bootstrap(frame: pd.DataFrame, *, replicates: int = 10_000) -> dict[str, Any]:
    case_ids = np.sort(frame["anchor_id"].unique())
    grouped = {int(case): frame.loc[frame["anchor_id"].eq(case)] for case in case_ids}
    incumbent_sq = np.asarray(
        [
            np.mean(np.square(group["prediction"] - group["target_hs"]))
            for group in grouped.values()
        ],
        dtype=np.float64,
    )
    persistence_sq = np.asarray(
        [
            np.mean(np.square(group["persistence"] - group["target_hs"]))
            for group in grouped.values()
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260822)
    sample = rng.integers(0, len(case_ids), size=(replicates, len(case_ids)))
    incumbent = np.sqrt(np.mean(incumbent_sq[sample], axis=1))
    persistence = np.sqrt(np.mean(persistence_sq[sample], axis=1))
    delta = incumbent - persistence
    return {
        "unit": "case_anchor_with_all_six_leads",
        "case_count": int(len(case_ids)),
        "replicates": int(replicates),
        "seed": 20260822,
        "incumbent_rmse_ci90_m": [float(x) for x in np.quantile(incumbent, [0.05, 0.95])],
        "incumbent_minus_persistence_delta_ci90_m": [
            float(x) for x in np.quantile(delta, [0.05, 0.95])
        ],
        "incumbent_minus_persistence_delta_median_m": float(np.median(delta)),
        "probability_incumbent_beats_persistence_descriptive": float(np.mean(delta < 0.0)),
        "interpretation": (
            "Descriptive uncertainty on the reused 182-case sample; it is not an independent "
            "promotion interval and does not include validation-selection uncertainty."
        ),
    }


def _greedy_ids(
    anchors: pd.DataFrame,
    *,
    start: str,
    end: str,
    require_distinct_episode: bool,
) -> np.ndarray:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    chosen: list[int] = []
    for _, group in anchors.groupby("station", sort=True, observed=True):
        eligible = group.loc[
            group["anchor_time"].ge(start_ts) & group["anchor_time"].lt(end_ts)
        ].sort_values(["anchor_time", "anchor_id"])
        next_time: pd.Timestamp | None = None
        used_episodes: set[int] = set()
        for row in eligible.itertuples(index=False):
            timestamp = pd.Timestamp(row.anchor_time)
            episode = int(row.episode_id)
            if require_distinct_episode and episode in used_episodes:
                continue
            if next_time is None or timestamp >= next_time:
                chosen.append(int(row.anchor_id))
                used_episodes.add(episode)
                next_time = timestamp + pd.Timedelta(hours=78)
    return np.asarray(sorted(chosen), dtype=np.int64)


def _assign_raw_wave_episodes(anchors: pd.DataFrame, wave: pd.DataFrame) -> pd.DataFrame:
    source = wave[["station", "time", "hs"]].copy()
    source["time"] = pd.to_datetime(source["time"], utc=True, errors="raise")
    if source.duplicated(["station", "time"]).any():
        raise ValueError("duplicate train-wave station/time key")
    parts: list[pd.DataFrame] = []
    next_episode = 0
    for _, group in source.groupby("station", sort=True, observed=True):
        group = group.sort_values("time").copy()
        high = group["hs"].ge(1.5) & group["hs"].notna()
        contiguous = group["time"].diff().eq(pd.Timedelta(minutes=20))
        start = high & (~high.shift(fill_value=False) | ~contiguous)
        local = start.cumsum().astype(np.int64) - 1 + next_episode
        selected = group.loc[high, ["station", "time", "hs"]].copy()
        selected["episode_id"] = local.loc[high].to_numpy(np.int64)
        parts.append(selected)
        if high.any():
            next_episode = int(local.loc[high].max()) + 1
    mapping = pd.concat(parts, ignore_index=True).rename(
        columns={"time": "anchor_time", "hs": "raw_current_hs"}
    )
    result = anchors.merge(
        mapping,
        on=["station", "anchor_time"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if result["episode_id"].isna().any():
        raise ValueError("eligible anchor missing raw-wave episode")
    if not np.allclose(result["current_hs"], result["raw_current_hs"], atol=1e-12, rtol=0.0):
        raise ValueError("anchor current_hs differs from immutable train_wave")
    result["episode_id"] = result["episode_id"].astype(np.int64)
    return result.drop(columns="raw_current_hs")


def _gap_and_episode_audit(anchors: pd.DataFrame, oof: pd.DataFrame) -> dict[str, Any]:
    case = (
        oof[["anchor_id", "fold", "station"]]
        .drop_duplicates()
        .merge(
            anchors[["anchor_id", "anchor_time", "episode_id"]],
            on="anchor_id",
            validate="one_to_one",
        )
    )
    recomputed: dict[str, np.ndarray] = {}
    episode_distinct: dict[str, np.ndarray] = {}
    for name, start, end in WINDOWS:
        recomputed[name] = _greedy_ids(
            anchors, start=start, end=end, require_distinct_episode=False
        )
        episode_distinct[name] = _greedy_ids(
            anchors, start=start, end=end, require_distinct_episode=True
        )
    expected_pairs = {
        (name, int(anchor_id)) for name, ids in recomputed.items() for anchor_id in ids
    }
    actual_pairs = set(zip(case["fold"], case["anchor_id"].astype(int), strict=True))
    gaps_within_fold: list[float] = []
    for _, group in case.groupby(["fold", "station"], observed=True):
        gap = group.sort_values("anchor_time")["anchor_time"].diff().dropna()
        gaps_within_fold.extend(gap.dt.total_seconds().div(3600).tolist())
    gaps_all: list[float] = []
    cross_window_below_78 = 0
    footprint_overlap = 0
    gap_by_station: dict[str, Any] = {}
    transition_counts: dict[str, int] = {}
    for station, group in case.groupby("station", observed=True):
        ordered = group.sort_values("anchor_time")
        gap = ordered["anchor_time"].diff().dt.total_seconds().div(3600)
        gaps_all.extend(gap.dropna().tolist())
        prior_fold = ordered["fold"].shift()
        cross = prior_fold.notna() & prior_fold.ne(ordered["fold"])
        cross_window_below_78 += int((cross & gap.lt(78.0)).sum())
        footprint_overlap += int(gap.lt(72.0).sum())
        gap_by_station[str(station)] = {
            "minimum_hours": float(gap.dropna().min()),
            "pairs_below_78h": int(gap.lt(78.0).sum()),
            "pairs_below_72h": int(gap.lt(72.0).sum()),
        }
        for index in ordered.index[cross & gap.lt(78.0)]:
            key = f"{prior_fold.loc[index]}->{ordered.loc[index, 'fold']}"
            transition_counts[key] = transition_counts.get(key, 0) + 1

    unique_episode = case[["station", "episode_id"]].drop_duplicates()
    duplicated_episode_case_count = int(
        case.duplicated(["station", "episode_id"], keep=False).sum()
    )
    shared_train_validation: dict[str, Any] = {}
    for name, start, _ in WINDOWS:
        validation = case.loc[case["fold"].eq(name)]
        train_end = pd.Timestamp(start, tz="UTC") - pd.Timedelta(hours=78)
        train = anchors.loc[anchors["anchor_time"].lt(train_end)]
        train_episode = set(zip(train["station"], train["episode_id"].astype(int), strict=True))
        validation_episode = set(
            zip(validation["station"], validation["episode_id"].astype(int), strict=True)
        )
        shared = train_episode.intersection(validation_episode)
        affected = validation.loc[
            [
                (str(row.station), int(row.episode_id)) in shared
                for row in validation.itertuples(index=False)
            ]
        ]
        shared_train_validation[name] = {
            "shared_station_episode_count": int(len(shared)),
            "affected_validation_case_count": int(len(affected)),
        }

    alt_ids = np.concatenate(list(episode_distinct.values()))
    actual_ids = case["anchor_id"].to_numpy(np.int64)
    return {
        "implementation": {
            "source_function": "src/p3_wave/data.py::select_independent_validation",
            "rule": "station-by-window chronological first eligible, then next timestamp >= previous+78h",
            "storm_episode_disjoint_enforced_by_original_splitter": False,
            "fold_membership_exactly_recomputed": actual_pairs == expected_pairs,
            "missing_pair_count": int(len(expected_pairs - actual_pairs)),
            "extra_pair_count": int(len(actual_pairs - expected_pairs)),
        },
        "spacing": {
            "within_fold_station_min_gap_hours": float(min(gaps_within_fold)),
            "all_folds_station_min_gap_hours": float(min(gaps_all)),
            "cross_window_adjacent_pairs_below_78h": int(cross_window_below_78),
            "context48_plus_target24_footprint_overlap_pairs_below_72h": int(footprint_overlap),
            "by_station": gap_by_station,
            "below_78h_fold_transition_counts": transition_counts,
            "official_test_spacing_independently_verifiable_from_anonymized_files": False,
        },
        "episode_structure": {
            "validation_cases": int(len(case)),
            "unique_station_episode_count": int(len(unique_episode)),
            "cases_in_repeated_station_episode": duplicated_episode_case_count,
            "repeated_station_episode_count": int(
                (case.groupby(["station", "episode_id"], observed=True).size() > 1).sum()
            ),
            "repeated_station_episode_count_by_station": {
                str(station): int((group.groupby("episode_id", observed=True).size() > 1).sum())
                for station, group in case.groupby("station", observed=True)
            },
            "train_validation_episode_overlap_by_fold": shared_train_validation,
            "episode_distinct_greedy_case_count": int(len(alt_ids)),
            "original_vs_episode_distinct_membership_overlap_count": int(
                len(np.intersect1d(actual_ids, alt_ids))
            ),
            "membership_changes_if_episode_distinct_rule_applied": int(
                len(np.setxor1d(actual_ids, alt_ids))
            ),
        },
        "assessment": (
            "The original OOF exactly follows its implemented window-local 78h greedy rule. "
            "That is fidelity to code, but not evidence that it also satisfies the later, stronger "
            "station-storm-episode-disjoint validation contract."
        ),
    }


def _make_local_context(
    case: pd.DataFrame, wave: pd.DataFrame, atmos: pd.DataFrame
) -> pd.DataFrame:
    wave_parts = {
        station: group.set_index("time").sort_index()[list(WAVE_COLUMNS)]
        for station, group in wave.groupby("station", observed=True)
    }
    atmos_parts = {
        station: group.set_index("time").sort_index()[list(ATMOS_COLUMNS)]
        for station, group in atmos.groupby("station", observed=True)
    }
    records: list[pd.DataFrame] = []
    steps = np.arange(-2880, 1, 10, dtype=np.int64)
    for row in case.itertuples(index=False):
        timestamp = pd.Timestamp(row.anchor_time)
        grid = pd.date_range(timestamp - pd.Timedelta(hours=48), timestamp, freq="10min")
        part = pd.DataFrame({"step_minute": steps})
        wave_values = wave_parts[str(row.station)].reindex(grid).reset_index(drop=True)
        atmos_values = atmos_parts[str(row.station)].reindex(grid).reset_index(drop=True)
        for column in WAVE_COLUMNS:
            part[column] = wave_values[column].to_numpy()
        for column in ATMOS_COLUMNS:
            part[column] = atmos_values[column].to_numpy()
        part.insert(0, "station", str(row.station))
        part.insert(0, "case_key", int(row.anchor_id))
        records.append(part)
    return pd.concat(records, ignore_index=True)


def _at_step(group: pd.DataFrame, column: str, step: int) -> float:
    values = group.loc[group["step_minute"].eq(step), column]
    if len(values) != 1 or pd.isna(values.iloc[0]):
        return math.nan
    return float(values.iloc[0])


def _case_features(context: pd.DataFrame, *, case_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, group in context.groupby(case_column, sort=True, observed=True):
        group = group.sort_values("step_minute")
        if len(group) != 289 or not np.array_equal(
            group["step_minute"].to_numpy(np.int64), np.arange(-2880, 1, 10, dtype=np.int64)
        ):
            raise ValueError("case context does not match the 289-row grid")
        station_values = group["station"].astype(str).unique()
        if len(station_values) != 1:
            raise ValueError("case spans multiple stations")
        record: dict[str, Any] = {"case_key": str(case_id), "station": station_values[0]}
        current_hs = _at_step(group, "hs", 0)
        record["hs_current"] = current_hs
        wave = group.loc[group["step_minute"].mod(20).eq(0)]
        for hours in (3, 6, 12, 24, 48):
            recent = wave.loc[wave["step_minute"].ge(-60 * hours), "hs"].dropna()
            record[f"hs_mean_{hours}h"] = float(recent.mean()) if len(recent) else math.nan
            record[f"hs_std_{hours}h"] = float(recent.std(ddof=0)) if len(recent) else math.nan
            record[f"hs_range_{hours}h"] = (
                float(recent.max() - recent.min()) if len(recent) else math.nan
            )
            lag = _at_step(group, "hs", -60 * hours)
            record[f"hs_delta_{hours}h"] = current_hs - lag if np.isfinite(lag) else math.nan
        hs_valid = wave["hs"].notna()
        record["hs_valid_fraction_48h"] = float(hs_valid.mean())
        finite_wave = wave.loc[hs_valid, ["step_minute", "hs"]]
        record["hs_peak_offset_hours"] = (
            float(finite_wave.loc[finite_wave["hs"].idxmax(), "step_minute"] / 60.0)
            if len(finite_wave)
            else math.nan
        )
        for column in ("tp", "hmax"):
            record[f"{column}_current"] = _at_step(group, column, 0)
            record[f"{column}_mean_12h"] = float(
                wave.loc[wave["step_minute"].ge(-720), column].mean()
            )
        for column in ("wspd", "gust", "airt", "relh", "caph"):
            record[f"{column}_current"] = _at_step(group, column, 0)
            recent = group.loc[group["step_minute"].ge(-720), column]
            record[f"{column}_mean_12h"] = float(recent.mean())
            lag = _at_step(group, column, -180)
            current = record[f"{column}_current"]
            record[f"{column}_delta_3h"] = (
                float(current - lag) if np.isfinite(current) and np.isfinite(lag) else math.nan
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _coverage(context: pd.DataFrame, *, case_column: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for column in (*WAVE_COLUMNS, *ATMOS_COLUMNS):
        expected = (
            context["step_minute"].mod(20).eq(0)
            if column in WAVE_COLUMNS
            else pd.Series(True, index=context.index)
        )
        for station in STATIONS:
            mask = context["station"].eq(station) & expected
            rows.append(
                {
                    "variable": column,
                    "station": station,
                    "observed": int(context.loc[mask, column].notna().sum()),
                    "expected": int(mask.sum()),
                    "coverage": float(context.loc[mask, column].notna().mean()),
                }
            )
        mask = expected
        rows.append(
            {
                "variable": column,
                "station": "ALL",
                "observed": int(context.loc[mask, column].notna().sum()),
                "expected": int(mask.sum()),
                "coverage": float(context.loc[mask, column].notna().mean()),
            }
        )
    return {"case_count": int(context[case_column].nunique()), "rows": rows}


def _shift_table(local: pd.DataFrame, test: pd.DataFrame) -> list[dict[str, Any]]:
    features = [column for column in local.columns if column not in {"case_key", "station"}]
    rows: list[dict[str, Any]] = []
    for feature in features:
        left = local[feature].dropna().to_numpy(np.float64)
        right = test[feature].dropna().to_numpy(np.float64)
        pooled = np.concatenate([left, right])
        scale = float(np.std(pooled, ddof=0)) if len(pooled) else math.nan
        coverage_delta = float(test[feature].notna().mean() - local[feature].notna().mean())
        smd = (
            float((np.mean(right) - np.mean(left)) / scale)
            if len(left) and len(right) and scale > 0.0
            else None
        )
        ks = float(ks_2samp(left, right).statistic) if len(left) and len(right) else None
        normalized_wasserstein = (
            float(wasserstein_distance(left, right) / scale)
            if len(left) and len(right) and scale > 0.0
            else None
        )
        severity_score = max(
            abs(coverage_delta) / 0.15,
            abs(smd or 0.0) / 0.50,
            (ks or 0.0) / 0.25,
        )
        severity = "high" if severity_score >= 1.0 else "medium" if severity_score >= 0.5 else "low"
        rows.append(
            {
                "feature": feature,
                "local_finite_share": float(local[feature].notna().mean()),
                "official_test_finite_share": float(test[feature].notna().mean()),
                "finite_share_delta_test_minus_local": coverage_delta,
                "local_median": float(np.median(left)) if len(left) else None,
                "official_test_median": float(np.median(right)) if len(right) else None,
                "standardized_mean_difference_test_minus_local": smd,
                "ks_statistic": ks,
                "normalized_wasserstein": normalized_wasserstein,
                "severity": severity,
            }
        )
    return sorted(
        rows,
        key=lambda row: max(
            abs(row["finite_share_delta_test_minus_local"]) / 0.15,
            abs(row["standardized_mean_difference_test_minus_local"] or 0.0) / 0.50,
            (row["ks_statistic"] or 0.0) / 0.25,
        ),
        reverse=True,
    )


def _station_mix(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    count = (
        frame.groupby("station", observed=True)[column].nunique().reindex(STATIONS, fill_value=0)
    )
    total = int(count.sum())
    return {
        station: {"cases": int(count.loc[station]), "share": float(count.loc[station] / total)}
        for station in STATIONS
    }


def _current_hs_bins(features: pd.DataFrame) -> dict[str, Any]:
    labels = ("1.50-1.70", "1.70-2.00", "2.00-2.50", "2.50+")
    binned = pd.cut(
        features["hs_current"],
        bins=[1.5 - 1e-12, 1.7, 2.0, 2.5, np.inf],
        labels=labels,
        include_lowest=True,
        right=False,
    )
    count = binned.value_counts(sort=False).reindex(labels, fill_value=0)
    return {
        str(label): {
            "cases": int(count.loc[label]),
            "share": float(count.loc[label] / len(features)),
        }
        for label in labels
    }


def _oracle_diagnostics(oof: pd.DataFrame) -> dict[str, Any]:
    components = ("single_prediction", "multi_prediction", "persistence")
    truth = oof["target_hs"].to_numpy(np.float64)
    values = oof[list(components)].to_numpy(np.float64)
    row_choice = np.argmin(np.abs(values - truth[:, None]), axis=1)
    row_oracle = values[np.arange(len(values)), row_choice]
    case_parts: list[np.ndarray] = []
    lead_parts: list[np.ndarray] = []
    for _, group in oof.groupby("anchor_id", sort=False, observed=True):
        error = [np.mean(np.square(group[column] - group["target_hs"])) for column in components]
        case_parts.append(group[components[int(np.argmin(error))]].to_numpy(np.float64))
    for _, group in oof.groupby("lead_h", sort=False, observed=True):
        error = [np.mean(np.square(group[column] - group["target_hs"])) for column in components]
        lead_parts.append(group[components[int(np.argmin(error))]].to_numpy(np.float64))
    case_oracle = np.concatenate(case_parts)
    lead_oracle = np.concatenate(lead_parts)
    # Group concatenation changes order, but RMSE is permutation invariant.
    return {
        "component_set": list(components),
        "component_rmse_m": {column: _rmse(oof["target_hs"], oof[column]) for column in components},
        "row_oracle_rmse_m_unimplementable": _rmse(truth, row_oracle),
        "case_oracle_rmse_m_unimplementable": float(
            np.sqrt(
                np.mean(
                    np.concatenate(
                        [
                            np.square(
                                group[
                                    components[
                                        int(
                                            np.argmin(
                                                [
                                                    np.mean(
                                                        np.square(group[c] - group["target_hs"])
                                                    )
                                                    for c in components
                                                ]
                                            )
                                        )
                                    ]
                                ].to_numpy(np.float64)
                                - group["target_hs"].to_numpy(np.float64)
                            )
                            for _, group in oof.groupby("anchor_id", sort=False, observed=True)
                        ]
                    )
                )
            )
        ),
        "lead_oracle_rmse_m_unimplementable": float(
            np.sqrt(
                np.mean(
                    np.concatenate(
                        [
                            np.square(
                                group[
                                    components[
                                        int(
                                            np.argmin(
                                                [
                                                    np.mean(
                                                        np.square(group[c] - group["target_hs"])
                                                    )
                                                    for c in components
                                                ]
                                            )
                                        )
                                    ]
                                ].to_numpy(np.float64)
                                - group["target_hs"].to_numpy(np.float64)
                            )
                            for _, group in oof.groupby("lead_h", sort=False, observed=True)
                        ]
                    )
                )
            )
        ),
        "unused_order_only_arrays_rows": {
            "case_oracle_rows": int(len(case_oracle)),
            "lead_oracle_rows": int(len(lead_oracle)),
        },
        "interpretation": (
            "Oracle values use the same labels to choose a component and are intentionally "
            "unimplementable. They show local component headroom, not hidden-test performance."
        ),
    }


def _same_key_oof_inventory(frozen: pd.DataFrame) -> dict[str, Any]:
    key_columns = ["anchor_id", "station", "lead_h"]
    frozen_keys = frozen[key_columns].sort_values(key_columns).reset_index(drop=True)
    matches: list[dict[str, Any]] = []
    for path in sorted((ROOT / "artifacts" / "p3").rglob("*oof*.parquet")):
        try:
            schema = pd.io.parquet.get_engine("auto").api.parquet.read_schema(path).names
        except Exception:
            # PyArrow is available in the registered runtime; pandas metadata APIs vary.
            import pyarrow.parquet as pq

            schema = pq.ParquetFile(path).schema.names
        if not set(key_columns).issubset(schema):
            continue
        candidate = pd.read_parquet(path, columns=key_columns)
        if len(candidate) != len(frozen_keys):
            continue
        candidate = candidate.sort_values(key_columns).reset_index(drop=True)
        if candidate.equals(frozen_keys):
            matches.append(
                {
                    "identity": path.relative_to(ROOT).as_posix(),
                    "sha256": _sha256(path),
                    "target_column_present_in_schema_but_not_read": "target_hs" in schema,
                }
            )
    return {
        "exact_same_key_oof_artifact_count": int(len(matches)),
        "target_bearing_exact_same_key_artifact_count": int(
            sum(bool(row["target_column_present_in_schema_but_not_read"]) for row in matches)
        ),
        "artifacts": matches,
        "interpretation": (
            "Concrete lower bound from persisted Parquet key sets. It omits metric-only runs and "
            "does not claim that every artifact was an independent model family."
        ),
    }


def _contains_number_and_rmse(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return bool(re.search(r"(?<![0-9])1092(?![0-9])", text)) and "rmse" in text.lower()


def _exposure_history(oof: pd.DataFrame) -> dict[str, Any]:
    ledger_path = ROOT / "artifacts/experiment_locks/p3_outer_truth_ledger.jsonl"
    ledger_entries = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    explicit_scoring = [
        entry
        for entry in ledger_entries
        if entry.get("designated_outer_scoring_open_attempt")
        or entry.get("purpose") == "one_shot_designated_outer_target_open_after_final_blind_seal"
    ]
    metric_docs: list[str] = []
    for path in sorted((ROOT / "artifacts").rglob("*.json")):
        if path.resolve() == (ROOT / OUTPUT_RELATIVE).resolve():
            continue
        if "p3" not in path.as_posix().lower():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if _contains_number_and_rmse(payload):
            metric_docs.append(path.relative_to(ROOT).as_posix())
    same_key = _same_key_oof_inventory(oof)
    adaptive_evidence_paths = [
        "artifacts/p3/final_ensemble_validation/metrics.json",
        "artifacts/p3/long_persistence_shrink/metrics.json",
        "artifacts/p3_kma_calibrated_longlead_blend_v2/posthoc_adaptive_global_alpha_grid_v1.json",
        "artifacts/p3_causal_forcing_analog_outer_research_v4/outer_one_shot/result.json",
    ]
    adaptive = []
    for relative in adaptive_evidence_paths:
        path = ROOT / relative
        if path.is_file():
            adaptive.append({"identity": relative, "sha256": _sha256(path)})
    return {
        "central_outer_truth_ledger": {
            "identity": ledger_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(ledger_path),
            "entries": int(len(ledger_entries)),
            "explicit_designated_scoring_events": int(len(explicit_scoring)),
            "experiment_ids": [str(entry.get("experiment_id")) for entry in explicit_scoring],
            "coverage_caveat": (
                "The ledger was introduced after earlier P3 model development and is not a "
                "complete history of all reads of the openly stored OOF targets."
            ),
        },
        "persisted_same_key_oof_lower_bound": same_key,
        "same_1092_grain_rmse_json_document_count": int(len(metric_docs)),
        "same_1092_grain_rmse_json_documents": metric_docs,
        "explicit_adaptive_or_posthoc_stage_count": int(len(adaptive)),
        "explicit_adaptive_or_posthoc_evidence": adaptive,
        "incumbent_origin": {
            "status": "adaptive_local_research_candidate_not_uploaded",
            "selection": "persistence_weight=0.20 chosen after local diagnostics",
            "bounded_sensitivity_weights_seen": [0.15, 0.20, 0.25],
            "virgin_holdout": False,
        },
        "conclusion": (
            "The 182-case labels are a repeatedly reused adaptive research surface. Pairwise "
            "results on it remain descriptive, but nominal bootstrap intervals omit selection "
            "multiplicity and must not be read as independent promotion evidence."
        ),
    }


def _official_contract(test_context: pd.DataFrame, test_index: pd.DataFrame) -> dict[str, Any]:
    keys = ["case_id", "station", "lead_h"]
    expected_steps = np.arange(-2880, 1, 10, dtype=np.int64)
    context_case_station = test_context[["case_id", "station"]].drop_duplicates()
    index_case_station = test_index[["case_id", "station"]].drop_duplicates()
    per_case_steps_exact = all(
        np.array_equal(
            group.sort_values("step_minute")["step_minute"].to_numpy(np.int64), expected_steps
        )
        for _, group in test_context.groupby("case_id", observed=True)
    )
    per_case_leads = test_index.groupby("case_id", observed=True)["lead_h"].agg(
        lambda values: tuple(sorted(map(int, values)))
    )
    current = test_context.loc[test_context["step_minute"].eq(0)]
    return {
        "context_rows": int(len(test_context)),
        "context_cases": int(test_context["case_id"].nunique()),
        "rows_per_case_exactly_289": bool(test_context.groupby("case_id").size().eq(289).all()),
        "step_grid_exact": bool(per_case_steps_exact),
        "context_duplicate_case_step_keys": int(
            test_context.duplicated(["case_id", "step_minute"]).sum()
        ),
        "index_rows": int(len(test_index)),
        "index_cases": int(test_index["case_id"].nunique()),
        "index_duplicate_keys": int(test_index.duplicated(keys).sum()),
        "six_leads_per_case_exact": bool(per_case_leads.map(lambda value: value == LEADS).all()),
        "context_index_case_station_exact": bool(
            context_case_station.sort_values(["case_id", "station"])
            .reset_index(drop=True)
            .equals(index_case_station.sort_values(["case_id", "station"]).reset_index(drop=True))
        ),
        "current_hs_all_finite_and_at_least_1p5": bool(
            current["hs"].notna().all() and current["hs"].ge(1.5).all()
        ),
        "wave_columns_null_on_structural_intermediate_10min_rows": bool(
            test_context.loc[test_context["step_minute"].mod(20).ne(0), list(WAVE_COLUMNS)]
            .isna()
            .all()
            .all()
        ),
        "absolute_case_times_available": False,
        "hidden_target_values_available_or_read": False,
    }


def _trust_matrix(
    *, persistence_gap: float, high_shift_features: int, explicit_scoring_events: int
) -> dict[str, Any]:
    return {
        "scale": "0-100 judgment score; not a probability",
        "relative_local_ranking": {
            "score": 55,
            "rating": "medium_with_strong_adaptive_caveat",
            "basis": (
                "Exact same-row pairing and deterministic recomputation support descriptive local "
                "comparisons, but the labels and candidate choices were repeatedly reused."
            ),
        },
        "absolute_hidden_score": {
            "score": 25,
            "rating": "low",
            "basis": (
                f"No hidden labels exist locally; local persistence differs from official B by "
                f"{persistence_gap:.6f}m and {high_shift_features} audited covariates/coverage features "
                "meet the prespecified high-shift threshold."
            ),
        },
        "submission_choice": {
            "score": 60,
            "rating": "medium_conservative_action",
            "basis": (
                f"Keeping the immutable incumbent is a defensible risk-control action after "
                f"{explicit_scoring_events} ledgered outer scoring events and multiple challenger "
                "failures; it is not proof that the incumbent is optimal on the official hidden set."
            ),
        },
    }


def build_audit(data_dir: Path) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    input_paths = {
        "official_readme": data_dir / "README.md",
        "official_score_script": data_dir / "score.py",
        "train_wave": data_dir / "train_wave.csv",
        "train_atmos": data_dir / "train_atmos.csv",
        "test_context_label_free": data_dir / "test_context.parquet",
        "test_index_label_free": data_dir / "test_index.csv",
        "frozen_oof": ROOT / OOF_RELATIVE,
        "anchor_cache": ROOT / ANCHORS_RELATIVE,
        "incumbent_metrics": ROOT / "artifacts/p3/long_persistence_shrink/metrics.json",
        "splitter_source": ROOT / "src/p3_wave/data.py",
        "fold_source": ROOT / "src/p3_wave/validation.py",
        "episode_splitter_source": ROOT / "src/p3_wave/revin_patch.py",
        "outer_ledger": ROOT / "artifacts/experiment_locks/p3_outer_truth_ledger.jsonl",
        "v4_outer_result": ROOT
        / "artifacts/p3_causal_forcing_analog_outer_research_v4/outer_one_shot/result.json",
        "v4_outer_manifest": ROOT
        / "artifacts/p3_causal_forcing_analog_outer_research_v4/outer_one_shot/manifest.json",
    }
    inputs = {name: _input_record(path, data_dir=data_dir) for name, path in input_paths.items()}

    oof = pd.read_parquet(input_paths["frozen_oof"])
    # Column projection is a policy boundary: do not materialize target_* vault columns.
    anchors = pd.read_parquet(
        input_paths["anchor_cache"],
        columns=["anchor_id", "station", "anchor_time", "grid_position", "current_hs"],
    )
    wave = pd.read_csv(input_paths["train_wave"])
    atmos = pd.read_csv(input_paths["train_atmos"])
    test_context = pd.read_parquet(input_paths["test_context_label_free"])
    test_index = pd.read_csv(input_paths["test_index_label_free"])
    wave["time"] = pd.to_datetime(wave["time"], utc=True, errors="raise")
    atmos["time"] = pd.to_datetime(atmos["time"], utc=True, errors="raise")
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], utc=True, errors="raise")
    anchors = _assign_raw_wave_episodes(anchors, wave)

    required_oof = {
        "anchor_id",
        "station",
        "lead_h",
        "current_hs",
        "target_hs",
        "fold",
        "single_prediction",
        "multi_prediction",
        "persistence",
        "prediction",
    }
    if not required_oof.issubset(oof.columns):
        raise ValueError(f"frozen OOF schema missing {sorted(required_oof - set(oof.columns))}")
    if len(oof) != 1092 or oof["anchor_id"].nunique() != 182:
        raise ValueError("frozen OOF grain drift")
    if oof.duplicated(["anchor_id", "station", "lead_h"]).any():
        raise ValueError("frozen OOF duplicate key")
    if not (
        oof.groupby("anchor_id")["lead_h"]
        .agg(lambda x: tuple(sorted(map(int, x))))
        .map(lambda value: value == LEADS)
        .all()
    ):
        raise ValueError("frozen OOF lead coverage drift")

    # Reconcile only the 1,092 already-scored OOF labels against immutable train_wave.
    paired = oof.merge(
        anchors[["anchor_id", "station", "anchor_time", "current_hs"]],
        on=["anchor_id", "station"],
        validate="many_to_one",
        suffixes=("", "_anchor"),
    )
    paired["target_time"] = paired["anchor_time"] + pd.to_timedelta(paired["lead_h"], unit="h")
    paired = paired.merge(
        wave[["station", "time", "hs"]].rename(
            columns={"time": "target_time", "hs": "raw_oof_target_hs"}
        ),
        on=["station", "target_time"],
        validate="many_to_one",
        how="left",
    )
    if paired["raw_oof_target_hs"].isna().any():
        raise ValueError("already-scored OOF target missing from immutable train_wave")
    target_error = float(np.max(np.abs(paired["target_hs"] - paired["raw_oof_target_hs"])))
    current_error = float(np.max(np.abs(paired["current_hs"] - paired["current_hs_anchor"])))
    persistence_error = float(np.max(np.abs(oof["persistence"] - oof["current_hs"])))
    if max(target_error, current_error, persistence_error) > 1e-12:
        raise ValueError("OOF target/current/persistence reconciliation failed")

    case = (
        oof[["anchor_id", "fold", "station", "current_hs"]]
        .drop_duplicates()
        .merge(anchors[["anchor_id", "anchor_time"]], on="anchor_id", validate="one_to_one")
    )
    local_context = _make_local_context(case, wave, atmos)
    local_features = _case_features(local_context, case_column="case_key")
    test_features = _case_features(test_context, case_column="case_id")
    feature_shift = _shift_table(local_features, test_features)
    high_shift_count = sum(row["severity"] == "high" for row in feature_shift)
    exposure = _exposure_history(oof)

    incumbent_metrics = _metric_slices(oof, "prediction")
    persistence_metrics = _metric_slices(oof, "persistence")
    official_b = 0.769455
    local_b = float(persistence_metrics["rmse_m"])
    gap = local_b - official_b
    exact_v4 = json.loads(input_paths["v4_outer_result"].read_text(encoding="utf-8"))
    if exact_v4.get("decision") != "NO_GO_KEEP_FROZEN_INCUMBENT":
        raise ValueError("v4 outer decision drift")

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "audit_id": "validation_system_audit_20260822_p3",
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "decision": "KEEP_FROZEN_INCUMBENT_AS_RISK_CONTROL_ONLY__HIDDEN_GENERALIZATION_UNVERIFIED",
        "executive_conclusion": {
            "answer": (
                "The frozen .7801609 OOF is useful for relative, same-row local comparisons, but "
                "it is not a calibrated estimate of the official hidden score. KEEP_INCUMBENT is "
                "therefore a conservative decision on a reused adaptive outer surface, not an "
                "independent validation endorsement."
            ),
            "freeze_definition": "immutable baseline and risk control; not validation endorsement",
            "v4_outer_decision": exact_v4["decision"],
            "official_hidden_score_known": False,
        },
        "scope": {
            "question": "How representative is the frozen local OOF of the official 200-case hidden test?",
            "permitted_test_reads": [
                "test_context.parquet label-free covariates",
                "test_index.csv keys",
            ],
            "local_label_scope": "already-scored frozen OOF only",
            "prohibited": [
                "model fitting",
                "prediction generation or write",
                "test target read",
                "submission mutation",
                "upload",
            ],
            "target_access": {
                "already_scored_frozen_oof_target_rows_read": 1092,
                "anchor_cache_columns_read": [
                    "anchor_id",
                    "station",
                    "anchor_time",
                    "grid_position",
                    "current_hs",
                ],
                "anchor_cache_target_columns_materialized": 0,
                "non_oof_target_rows_selected_for_reconciliation": 0,
                "execution_history_disclosure": {
                    "superseded_failed_draft_runs_materializing_full_anchor_cache": 1,
                    "superseded_draft_target_columns_materialized": 6,
                    "non_oof_target_values_used_in_final_metrics": 0,
                    "raw_target_values_persisted_or_reported": 0,
                    "remediation": "Canonical code now projects only non-target anchor columns and fail-closes on 1,092 OOF-only reconciliation rows.",
                },
            },
        },
        "inputs": inputs,
        "official_index_and_context_fidelity": _official_contract(test_context, test_index),
        "frozen_oof_fidelity": {
            "rows": int(len(oof)),
            "cases": int(oof["anchor_id"].nunique()),
            "duplicate_keys": int(oof.duplicated(["anchor_id", "station", "lead_h"]).sum()),
            "lead_set": list(LEADS),
            "six_leads_per_case_exact": True,
            "anchor_target_max_abs_error_m": target_error,
            "anchor_current_max_abs_error_m": current_error,
            "persistence_equals_current_max_abs_error_m": persistence_error,
            "all_predictions_finite": bool(np.isfinite(oof["prediction"].to_numpy(float)).all()),
            "all_predictions_within_0_30m": bool(oof["prediction"].between(0.0, 30.0).all()),
            "station_mix": _station_mix(case, "anchor_id"),
            "fold_station_case_mix": {
                str(fold): {
                    str(station): int(count)
                    for station, count in group.groupby("station", observed=True).size().items()
                }
                for fold, group in case.groupby("fold", observed=True)
            },
            "lead_rows": {
                str(int(lead)): int(count)
                for lead, count in oof.groupby("lead_h", observed=True).size().items()
            },
            "anchor_period_utc": {
                "min": case["anchor_time"].min().isoformat(),
                "max": case["anchor_time"].max().isoformat(),
            },
        },
        "sampling_and_event_audit": _gap_and_episode_audit(anchors, oof),
        "local_performance": {
            "incumbent": incumbent_metrics,
            "persistence": persistence_metrics,
            "incumbent_minus_persistence_rmse_m": float(
                incumbent_metrics["rmse_m"] - persistence_metrics["rmse_m"]
            ),
            "case_bootstrap": _case_bootstrap(oof),
            "oracle_diagnostics": _oracle_diagnostics(oof),
        },
        "official_baseline_calibration": {
            "local_persistence_rmse_m": local_b,
            "official_full_persistence_B_rmse_m": official_b,
            "local_minus_official_B_m": gap,
            "local_over_official_B_ratio": float(local_b / official_b),
            "official_public_B_rmse_m": 0.750046,
            "official_private_B_rmse_m": 0.778838,
            "official_T": {
                "value": 0.624165,
                "meaning": "published policy/scoring constant corresponding to 70% problem score",
                "is_hidden_model_score": False,
            },
            "interpretation": (
                "Persistence is label-free at prediction time, but these RMSE values use different "
                "target samples. The gap is a direct difficulty-calibration warning, not a correction "
                "formula for the incumbent hidden RMSE."
            ),
        },
        "label_free_local_vs_official_test": {
            "local_cases": int(len(local_features)),
            "official_test_cases": int(len(test_features)),
            "station_mix": {
                "local": _station_mix(local_features, "case_key"),
                "official_test": _station_mix(test_features, "case_key"),
            },
            "current_hs_bins": {
                "local": _current_hs_bins(local_features),
                "official_test": _current_hs_bins(test_features),
            },
            "coverage": {
                "local": _coverage(local_context, case_column="case_key"),
                "official_test": _coverage(test_context, case_column="case_id"),
            },
            "feature_shift": feature_shift,
            "high_shift_feature_count": int(high_shift_count),
            "domain_classifier_fit_or_predictions": 0,
            "limitations": [
                "Official case timestamps are anonymized, so seasonality and the stated 78h test spacing cannot be independently recomputed.",
                "Covariate similarity cannot establish target-conditional similarity or hidden RMSE calibration.",
                "Finite-only distribution tests are accompanied by explicit coverage deltas; imputation was not used.",
            ],
        },
        "adaptive_exposure": exposure,
        "trust": _trust_matrix(
            persistence_gap=gap,
            high_shift_features=high_shift_count,
            explicit_scoring_events=exposure["central_outer_truth_ledger"][
                "explicit_designated_scoring_events"
            ],
        ),
        "findings": [
            {
                "id": "P3-VAL-01",
                "severity": "high",
                "confidence": "high",
                "finding": "The .7801609 value is not an official hidden-score estimate.",
                "evidence": "Official hidden targets are unavailable; local and official persistence baselines differ materially.",
            },
            {
                "id": "P3-VAL-02",
                "severity": "high",
                "confidence": "high",
                "finding": "The 182-case outer labels have been reused adaptively.",
                "evidence": "Ledgered scoring events, exact-key OOF inventory, and explicit posthoc/adaptive artifacts.",
            },
            {
                "id": "P3-VAL-03",
                "severity": "high",
                "confidence": "high",
                "finding": "Window-local splitter resets create one cross-window pair below 78h and below the 72h context-plus-target footprint.",
                "evidence": "Global station-time gap reconstruction; within-window minimum remains exactly 78h.",
            },
            {
                "id": "P3-VAL-04",
                "severity": "medium",
                "confidence": "high",
                "finding": "Original greedy validation fidelity is exact, but the original splitter did not enforce storm-episode disjointness.",
                "evidence": "Source inspection plus raw-wave episode reconstruction.",
            },
            {
                "id": "P3-VAL-05",
                "severity": "medium",
                "confidence": "high",
                "finding": "Local and official label-free case mixes/coverage are not exchangeable without caveats.",
                "evidence": "Station mix, expected-slot coverage, SMD, KS, and normalized Wasserstein diagnostics.",
            },
        ],
        "remediation": [
            {
                "priority": 1,
                "action": "Treat the current 182-case surface as development data; never call another comparison on it independent validation.",
            },
            {
                "priority": 2,
                "action": "Create a new immutable episode-disjoint, globally 78h-spaced temporal holdout or repeated forward-chaining evaluation before promotion.",
            },
            {
                "priority": 3,
                "action": "Report skill versus persistence by fold/station/lead with selection-adjusted uncertainty, not only absolute pooled RMSE.",
            },
            {
                "priority": 4,
                "action": "Match official station proportions and raw observation coverage, or use explicit reweighting/sensitivity bounds without reading hidden labels.",
            },
            {
                "priority": 5,
                "action": "Keep the incumbent frozen unless a challenger passes a genuinely untouched validation surface; freeze is risk control, not endorsement.",
            },
        ],
        "operation_counters": {
            "model_fits": 0,
            "prediction_generations": 0,
            "prediction_writes": 0,
            "official_test_target_reads": 0,
            "already_scored_frozen_oof_target_rows_read": 1092,
            "canonical_run_anchor_cache_target_columns_materialized": 0,
            "canonical_run_non_oof_local_target_rows_selected": 0,
            "superseded_failed_draft_run_target_columns_materialized": 6,
            "superseded_draft_non_oof_target_values_used_in_final_analysis": 0,
            "submission_reads": 0,
            "submission_writes": 0,
            "uploads": 0,
            "source_mutations": 0,
        },
    }
    canonical = json.dumps(
        result, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    result["payload_sha256_before_integrity_field"] = hashlib.sha256(canonical).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT_RELATIVE)
    args = parser.parse_args()
    output = args.output.resolve()
    expected_output = (ROOT / OUTPUT_RELATIVE).resolve()
    if output != expected_output:
        raise ValueError(f"output is frozen at {OUTPUT_RELATIVE.as_posix()}")
    result = build_audit(args.data_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": OUTPUT_RELATIVE.as_posix(),
                "sha256": _sha256(output),
                "decision": result["decision"],
                "high_shift_feature_count": result["label_free_local_vs_official_test"][
                    "high_shift_feature_count"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
