"""Corrected, station-global repeated-forward validation primitives for P3.

This module is intentionally append-only.  It does not replace the historical splitter;
instead it defines the stronger research contract requested after the validation-system audit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .loss_router import COMPONENTS, ComponentLossRouter, RouterConfig, route_row_predictions
from .validation import rmse

OFFICIAL_LEADS = (3, 6, 9, 12, 18, 24)
ACTIVE_ROUTER_LEADS = (12, 18, 24)
CASE_GAP_HOURS = 78
FOOTPRINT_HOURS = 72


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class CorrectedFold:
    name: str
    train_ids: np.ndarray
    validation_ids: np.ndarray
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def normalize_windows(windows: Sequence[Sequence[str]]) -> tuple[WindowSpec, ...]:
    result: list[WindowSpec] = []
    for item in windows:
        if len(item) != 3:
            raise ValueError("each window must contain name, start, end")
        name, start, end = map(str, item)
        window = WindowSpec(name=name, start=_utc(start), end=_utc(end))
        if window.start >= window.end:
            raise ValueError(f"invalid validation window: {name}")
        result.append(window)
    if len({window.name for window in result}) != len(result):
        raise ValueError("validation window names must be unique")
    ordered = sorted(result, key=lambda window: window.start)
    for left, right in zip(ordered, ordered[1:], strict=False):
        if left.end > right.start:
            raise ValueError("validation windows may not overlap")
    return tuple(ordered)


def select_station_global_validation(
    anchors: pd.DataFrame,
    *,
    windows: Sequence[Sequence[str]],
    gap_hours: int = CASE_GAP_HOURS,
    footprint_hours: int = FOOTPRINT_HOURS,
) -> pd.DataFrame:
    """Select first-eligible cases once across all windows for each station.

    The greedy state and used-episode set never reset at a window boundary.  A selected
    anchor represents the closed 48-hour context and following 24-hour target footprint;
    the 78-hour anchor gap therefore leaves at least six hours between footprints.
    """

    if gap_hours != CASE_GAP_HOURS or footprint_hours != FOOTPRINT_HOURS:
        raise ValueError("P3 corrected selection is frozen at 78h gap and 72h footprint")
    required = {"anchor_id", "station", "anchor_time", "episode_id"}
    missing = required.difference(anchors.columns)
    if missing:
        raise ValueError(f"anchor metadata is missing: {sorted(missing)}")
    if anchors["anchor_id"].duplicated().any():
        raise ValueError("anchor_id must be unique")

    specifications = normalize_windows(windows)
    source = anchors.loc[:, list(required)].copy()
    source["anchor_time"] = pd.to_datetime(source["anchor_time"], utc=True, errors="raise")
    blocks: list[pd.DataFrame] = []
    for window in specifications:
        block = source.loc[
            source["anchor_time"].ge(window.start) & source["anchor_time"].lt(window.end)
        ].copy()
        block["fold"] = window.name
        blocks.append(block)
    eligible = pd.concat(blocks, ignore_index=True)
    if eligible.empty or eligible["anchor_id"].duplicated().any():
        raise ValueError("validation windows are empty or overlap on anchor ids")

    chosen_rows: list[dict[str, Any]] = []
    for station, group in eligible.groupby("station", sort=True, observed=True):
        ordered = group.sort_values(["anchor_time", "anchor_id"])
        next_allowed: pd.Timestamp | None = None
        used_episodes: set[int] = set()
        for row in ordered.itertuples(index=False):
            timestamp = pd.Timestamp(row.anchor_time)
            episode_id = int(row.episode_id)
            if episode_id in used_episodes:
                continue
            if next_allowed is not None and timestamp < next_allowed:
                continue
            chosen_rows.append(
                {
                    "anchor_id": int(row.anchor_id),
                    "station": str(station),
                    "anchor_time": timestamp,
                    "episode_id": episode_id,
                    "fold": str(row.fold),
                }
            )
            used_episodes.add(episode_id)
            next_allowed = timestamp + pd.Timedelta(hours=gap_hours)

    selected = (
        pd.DataFrame(chosen_rows)
        .sort_values(["anchor_time", "station", "anchor_id"])
        .reset_index(drop=True)
    )
    if selected.empty or selected["anchor_id"].duplicated().any():
        raise ValueError("corrected validation selection is empty or duplicated")
    if selected.duplicated(["station", "episode_id"]).any():
        raise AssertionError("a station storm episode was selected more than once")
    for _, group in selected.groupby("station", sort=True, observed=True):
        gaps = group.sort_values("anchor_time")["anchor_time"].diff().dropna()
        if not gaps.ge(pd.Timedelta(hours=gap_hours)).all():
            raise AssertionError("station-global validation gap is below 78 hours")
        if not gaps.ge(pd.Timedelta(hours=footprint_hours)).all():
            raise AssertionError("validation context-plus-target footprints overlap")
    return selected


def _station_episode_keys(frame: pd.DataFrame) -> set[tuple[str, int]]:
    return set(zip(frame["station"].astype(str), frame["episode_id"].astype(int), strict=True))


def _minimum_cross_gap_hours(train: pd.DataFrame, validation: pd.DataFrame) -> float:
    minima: list[float] = []
    for station, current in validation.groupby("station", sort=True, observed=True):
        train_times = train.loc[train["station"].eq(station), "anchor_time"]
        if train_times.empty:
            continue
        for timestamp in current["anchor_time"]:
            distance = (train_times - pd.Timestamp(timestamp)).abs()
            minima.append(float(distance.min().total_seconds() / 3600.0))
    if not minima:
        raise ValueError("no common station exists between train and validation")
    return float(min(minima))


def build_corrected_repeated_forward_folds(
    anchors: pd.DataFrame,
    *,
    windows: Sequence[Sequence[str]],
    gap_hours: int = CASE_GAP_HOURS,
    footprint_hours: int = FOOTPRINT_HOURS,
) -> tuple[tuple[CorrectedFold, ...], pd.DataFrame, dict[str, Any]]:
    """Build expanding folds around one globally selected validation surface."""

    selected = select_station_global_validation(
        anchors,
        windows=windows,
        gap_hours=gap_hours,
        footprint_hours=footprint_hours,
    )
    source = anchors.copy()
    source["anchor_time"] = pd.to_datetime(source["anchor_time"], utc=True, errors="raise")
    folds: list[CorrectedFold] = []
    summaries: dict[str, Any] = {}
    for window in normalize_windows(windows):
        validation = selected.loc[selected["fold"].eq(window.name)].copy()
        if validation.empty:
            raise ValueError(f"corrected fold has no validation cases: {window.name}")
        validation_ids = np.sort(validation["anchor_id"].to_numpy(dtype=np.int64))
        train_end = window.start - pd.Timedelta(hours=gap_hours)
        train = source.loc[source["anchor_time"].lt(train_end)].copy()
        before_episode_filter = len(train)
        validation_episodes = _station_episode_keys(validation)
        keep = np.asarray(
            [
                (str(row.station), int(row.episode_id)) not in validation_episodes
                for row in train.itertuples(index=False)
            ],
            dtype=bool,
        )
        train = train.loc[keep]
        train_ids = np.sort(train["anchor_id"].to_numpy(dtype=np.int64))
        if not len(train_ids) or np.intersect1d(train_ids, validation_ids).size:
            raise ValueError(f"empty or overlapping train partition: {window.name}")
        if _station_episode_keys(train).intersection(validation_episodes):
            raise AssertionError("a validation storm episode appears in outer training")
        minimum_gap = _minimum_cross_gap_hours(train, validation)
        if minimum_gap < gap_hours:
            raise AssertionError("train/validation anchor gap is below 78 hours")
        if minimum_gap < footprint_hours:
            raise AssertionError("train/validation 72-hour footprints overlap")
        folds.append(
            CorrectedFold(
                name=window.name,
                train_ids=train_ids,
                validation_ids=validation_ids,
                validation_start=window.start,
                validation_end=window.end,
            )
        )
        summaries[window.name] = {
            "train_anchor_count": int(len(train_ids)),
            "validation_case_count": int(len(validation_ids)),
            "validation_by_station": {
                str(key): int(value)
                for key, value in validation.groupby("station", observed=True).size().items()
            },
            "removed_same_episode_train_anchors": int(before_episode_filter - len(train)),
            "shared_train_validation_station_episode_count": 0,
            "minimum_train_validation_anchor_gap_hours": minimum_gap,
            "minimum_train_validation_footprint_separation_hours": minimum_gap - footprint_hours,
        }

    station_gap: dict[str, float] = {}
    for station, group in selected.groupby("station", sort=True, observed=True):
        gaps = (
            group.sort_values("anchor_time")["anchor_time"]
            .diff()
            .dropna()
            .dt.total_seconds()
            .div(3600.0)
        )
        station_gap[str(station)] = float(gaps.min())
    audit = {
        "selection_rule": "station-global chronological first eligible across all fixed windows",
        "validation_case_count": int(len(selected)),
        "validation_row_count": int(len(selected) * len(OFFICIAL_LEADS)),
        "unique_station_episode_count": int(
            selected[["station", "episode_id"]].drop_duplicates().shape[0]
        ),
        "repeated_station_episode_count": 0,
        "station_global_minimum_gap_hours": station_gap,
        "cross_window_pairs_below_78h": 0,
        "context48_plus_target24_footprint_overlap_pairs": 0,
        "footprint_definition": "[anchor-48h, anchor+24h] within station",
        "folds": summaries,
    }
    return tuple(folds), selected, audit


def fixed_prequential_lead_router(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    components: np.ndarray,
    row_losses: np.ndarray,
    *,
    fold_order: Sequence[str],
    config: RouterConfig,
    active_leads: Iterable[int] = ACTIVE_ROUTER_LEADS,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Refit one fixed router configuration using completed folds only."""

    values = np.asarray(components, dtype=float)
    losses = np.asarray(row_losses, dtype=float)
    if values.shape != (len(metadata), len(COMPONENTS)) or losses.shape != values.shape:
        raise ValueError("router rows are not aligned")
    if len(features) != len(metadata):
        raise ValueError("router feature rows are not aligned")
    active = tuple(int(value) for value in active_leads)
    if not active or not set(active).issubset(OFFICIAL_LEADS):
        raise ValueError("active router leads must be official P3 leads")
    unknown = set(metadata["fold"].astype(str)).difference(fold_order)
    if unknown:
        raise ValueError(f"unknown fold labels: {sorted(unknown)}")

    weights = np.full_like(values, np.nan, dtype=float)
    receipts: list[dict[str, Any]] = []
    completed: list[str] = []
    for fold in fold_order:
        current = metadata["fold"].astype(str).eq(fold).to_numpy()
        past = metadata["fold"].astype(str).isin(completed).to_numpy()
        if not current.any():
            raise ValueError(f"router fold has no rows: {fold}")
        if not past.any():
            weights[current] = np.array([0.5, 0.5, 0.0])
            fitted_rows = fitted_cases = 0
            action = "exact_equal_component_no_op"
        else:
            router = ComponentLossRouter(config).fit(
                features.loc[past].reset_index(drop=True), losses[past]
            )
            weights[current] = router.predict_weights(features.loc[current].reset_index(drop=True))
            inactive = current & ~metadata["lead_h"].astype(int).isin(active).to_numpy()
            weights[inactive] = np.array([0.5, 0.5, 0.0])
            fitted_rows = int(past.sum())
            fitted_cases = int(metadata.loc[past, "anchor_id"].nunique())
            action = "fixed_config_refit_on_completed_corrected_oof_only"
        receipts.append(
            {
                "fold": str(fold),
                "action": action,
                "past_fit_rows": fitted_rows,
                "past_fit_cases": fitted_cases,
                "current_fold_target_used_for_router": False,
                "config": {
                    "name": config.name,
                    "alpha": config.alpha,
                    "temperature_multiplier": config.temperature_multiplier,
                    "strength": config.strength,
                },
                "active_leads": list(active),
            }
        )
        completed.append(str(fold))
    if not np.isfinite(weights).all():
        raise RuntimeError("router failed to assign every row")
    prediction = route_row_predictions(values, weights)
    return prediction, weights, receipts


def paired_case_bootstrap(
    frame: pd.DataFrame,
    *,
    candidate_column: str,
    baseline_column: str,
    replicates: int = 5_000,
    seed: int = 20260822,
) -> dict[str, Any]:
    """Paired bootstrap over complete six-lead forecast cases."""

    required = {"fold", "anchor_id", "target_hs", candidate_column, baseline_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"bootstrap frame is missing: {sorted(missing)}")
    work = frame.reset_index(drop=True)
    blocks = [
        group.index.to_numpy(dtype=np.int64)
        for _, group in work.groupby(["fold", "anchor_id"], sort=False, observed=True)
    ]
    if not blocks or any(len(block) != len(OFFICIAL_LEADS) for block in blocks):
        raise ValueError("bootstrap blocks must be complete six-lead cases")
    truth = work["target_hs"].to_numpy(dtype=float)
    candidate = work[candidate_column].to_numpy(dtype=float)
    baseline = work[baseline_column].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    delta = np.empty(replicates, dtype=float)
    for number in range(replicates):
        selected = np.concatenate(
            [blocks[index] for index in rng.integers(0, len(blocks), size=len(blocks))]
        )
        delta[number] = rmse(truth[selected], candidate[selected]) - rmse(
            truth[selected], baseline[selected]
        )
    return {
        "unit": "complete_six_lead_case",
        "cases": int(len(blocks)),
        "replicates": int(replicates),
        "seed": int(seed),
        "delta_candidate_minus_persistence_ci90_m": [
            float(value) for value in np.quantile(delta, [0.05, 0.95])
        ],
        "delta_candidate_minus_persistence_median_m": float(np.median(delta)),
        "probability_candidate_improves_descriptive": float(np.mean(delta < 0.0)),
    }


def evaluate_candidate_gate(
    frame: pd.DataFrame,
    *,
    bootstrap: dict[str, Any],
    contract_checks: dict[str, bool],
    minimum_improved_folds: int = 2,
) -> dict[str, Any]:
    """Apply the preregistered corrected-evidence gate against persistence."""

    truth = frame["target_hs"].to_numpy(dtype=float)
    candidate = frame["final_prediction"].to_numpy(dtype=float)
    persistence = frame["persistence"].to_numpy(dtype=float)
    fold_delta = {
        str(name): rmse(group["target_hs"], group["final_prediction"])
        - rmse(group["target_hs"], group["persistence"])
        for name, group in frame.groupby("fold", sort=True, observed=True)
    }
    improved_folds = int(sum(value < 0.0 for value in fold_delta.values()))
    ci_upper = float(bootstrap["delta_candidate_minus_persistence_ci90_m"][1])
    checks = {
        "all_split_and_integrity_contracts_pass": bool(all(contract_checks.values())),
        "pooled_rmse_below_persistence": rmse(truth, candidate) < rmse(truth, persistence),
        "paired_case_bootstrap_ci90_upper_below_zero": ci_upper < 0.0,
        "at_least_two_of_three_folds_improve": improved_folds >= minimum_improved_folds,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "contract_checks": contract_checks,
        "candidate_rmse_m": rmse(truth, candidate),
        "persistence_rmse_m": rmse(truth, persistence),
        "delta_candidate_minus_persistence_m": rmse(truth, candidate) - rmse(truth, persistence),
        "fold_delta_candidate_minus_persistence_m": fold_delta,
        "improved_fold_count": improved_folds,
        "minimum_improved_fold_count": int(minimum_improved_folds),
        "consequence": (
            "allow_full_train_and_local_candidate_generation"
            if all(checks.values())
            else "stop_before_test_inference"
        ),
    }
