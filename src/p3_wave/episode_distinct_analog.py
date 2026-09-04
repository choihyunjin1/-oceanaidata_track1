"""Episode-distinct raw-shape analog forecasting for P3.

The module is deliberately model-free.  Retrieval sees only a 145-point past
``hs`` history and storm-episode identifiers.  Future residuals are projected
only after neighbour indices have been fixed, which keeps the similarity
search label-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

LEADS = (3, 6, 9, 12, 18, 24)
SHORT_LEADS = (3, 6, 9)
ACTIVE_LEADS = (12, 18, 24)
HISTORY_POINTS = 145


class EpisodeAnalogError(ValueError):
    """Raised when the frozen analog contract is violated."""


@dataclass(frozen=True)
class PreparedHistories:
    """Interpolated, current-centred and robustly scaled histories."""

    normalized: np.ndarray
    scale: np.ndarray
    coverage: np.ndarray
    eligible: np.ndarray


@dataclass(frozen=True)
class NeighborSelection:
    """Local library row indices selected without looking at future labels."""

    indices: np.ndarray
    distances: np.ndarray
    episodes: np.ndarray
    evaluated_candidates: int
    total_candidates: int


def prepare_histories(
    history: np.ndarray,
    *,
    minimum_coverage: float = 0.95,
    mad_floor: float = 0.1,
) -> PreparedHistories:
    """Interpolate past-only gaps and apply current-centred MAD scaling."""

    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != HISTORY_POINTS:
        raise EpisodeAnalogError(f"history must have shape (n, {HISTORY_POINTS})")
    if not 0.0 < minimum_coverage <= 1.0:
        raise EpisodeAnalogError("minimum_coverage must be in (0, 1]")
    if mad_floor <= 0.0:
        raise EpisodeAnalogError("mad_floor must be positive")

    finite = np.isfinite(values)
    coverage = finite.mean(axis=1)
    eligible = (coverage >= minimum_coverage) & finite[:, -1]
    filled = np.full_like(values, np.nan)
    x = np.arange(HISTORY_POINTS, dtype=np.float64)
    for row in np.flatnonzero(eligible):
        valid = finite[row]
        # All interpolation uses observations at or before the forecast origin.
        filled[row] = np.interp(x, x[valid], values[row, valid])

    scale = np.full(len(values), np.nan, dtype=np.float64)
    normalized = np.full_like(values, np.nan)
    if eligible.any():
        valid_values = filled[eligible]
        median = np.median(valid_values, axis=1)
        mad = np.median(np.abs(valid_values - median[:, None]), axis=1)
        robust_scale = np.maximum(mad, mad_floor)
        current = valid_values[:, -1]
        scale[eligible] = robust_scale
        normalized[eligible] = (valid_values - current[:, None]) / robust_scale[:, None]
    return PreparedHistories(
        normalized=normalized,
        scale=scale,
        coverage=coverage,
        eligible=eligible,
    )


def banded_dtw_distances(
    query: np.ndarray,
    candidates: np.ndarray,
    *,
    radius_steps: int,
) -> np.ndarray:
    """Return exact Sakoe-Chiba DTW distances for equal-length series.

    Squared path cost is divided by the fixed series length.  Since all series
    have the same length, this preserves the exact DTW ranking and gives a
    lower-bound-compatible RMS scale.
    """

    query_array = np.asarray(query, dtype=np.float64)
    candidate_array = np.asarray(candidates, dtype=np.float64)
    if query_array.shape != (HISTORY_POINTS,):
        raise EpisodeAnalogError(f"query must have shape ({HISTORY_POINTS},)")
    if candidate_array.ndim != 2 or candidate_array.shape[1] != HISTORY_POINTS:
        raise EpisodeAnalogError(
            f"candidates must have shape (n, {HISTORY_POINTS})"
        )
    if radius_steps < 0:
        raise EpisodeAnalogError("radius_steps must be non-negative")
    if not np.isfinite(query_array).all() or not np.isfinite(candidate_array).all():
        raise EpisodeAnalogError("DTW inputs must be finite")
    count = len(candidate_array)
    if count == 0:
        return np.empty(0, dtype=np.float64)

    width = HISTORY_POINTS + 1
    previous = np.full((count, width), np.inf, dtype=np.float64)
    previous[:, 0] = 0.0
    for i in range(1, HISTORY_POINTS + 1):
        current = np.full((count, width), np.inf, dtype=np.float64)
        start = max(1, i - radius_steps)
        stop = min(HISTORY_POINTS, i + radius_steps)
        for j in range(start, stop + 1):
            local = np.square(candidate_array[:, j - 1] - query_array[i - 1])
            predecessor = np.minimum(
                np.minimum(previous[:, j], current[:, j - 1]),
                previous[:, j - 1],
            )
            current[:, j] = local + predecessor
        previous = current
    result = np.sqrt(previous[:, HISTORY_POINTS] / float(HISTORY_POINTS))
    if not np.isfinite(result).all():
        raise EpisodeAnalogError("DTW returned a non-finite distance")
    return result


def lb_keogh_distances(
    query: np.ndarray,
    candidates: np.ndarray,
    *,
    radius_steps: int,
) -> np.ndarray:
    """Compute an admissible LB_Keogh bound on ``banded_dtw_distances``."""

    query_array = np.asarray(query, dtype=np.float64)
    candidate_array = np.asarray(candidates, dtype=np.float64)
    if query_array.shape != (HISTORY_POINTS,):
        raise EpisodeAnalogError(f"query must have shape ({HISTORY_POINTS},)")
    if candidate_array.ndim != 2 or candidate_array.shape[1] != HISTORY_POINTS:
        raise EpisodeAnalogError(
            f"candidates must have shape (n, {HISTORY_POINTS})"
        )
    if radius_steps < 0:
        raise EpisodeAnalogError("radius_steps must be non-negative")
    if not np.isfinite(query_array).all() or not np.isfinite(candidate_array).all():
        raise EpisodeAnalogError("LB_Keogh inputs must be finite")

    lower = np.empty(HISTORY_POINTS, dtype=np.float64)
    upper = np.empty(HISTORY_POINTS, dtype=np.float64)
    for index in range(HISTORY_POINTS):
        start = max(0, index - radius_steps)
        stop = min(HISTORY_POINTS, index + radius_steps + 1)
        window = query_array[start:stop]
        lower[index] = np.min(window)
        upper[index] = np.max(window)
    above = np.maximum(candidate_array - upper[None, :], 0.0)
    below = np.maximum(lower[None, :] - candidate_array, 0.0)
    return np.sqrt(np.sum(np.square(above) + np.square(below), axis=1) / HISTORY_POINTS)


def _episode_best(
    candidate_indices: np.ndarray,
    distances: np.ndarray,
    episode_ids: np.ndarray,
) -> dict[int, tuple[float, int]]:
    best: dict[int, tuple[float, int]] = {}
    for index, distance in zip(candidate_indices, distances, strict=True):
        episode = int(episode_ids[index])
        value = (float(distance), int(index))
        incumbent = best.get(episode)
        if incumbent is None or value < incumbent:
            best[episode] = value
    return best


class EpisodeAnalogIndex:
    """A same-station library supporting exact episode-distinct retrieval."""

    def __init__(
        self,
        *,
        anchor_ids: np.ndarray,
        episode_ids: np.ndarray,
        normalized_histories: np.ndarray,
        radius_steps: int = 6,
        neighbor_count: int = 8,
        batch_size: int = 1024,
    ) -> None:
        self.anchor_ids = np.asarray(anchor_ids, dtype=np.int64)
        self.episode_ids = np.asarray(episode_ids, dtype=np.int64)
        self.histories = np.asarray(normalized_histories, dtype=np.float64)
        self.radius_steps = int(radius_steps)
        self.neighbor_count = int(neighbor_count)
        self.batch_size = int(batch_size)
        if self.histories.shape != (len(self.anchor_ids), HISTORY_POINTS):
            raise EpisodeAnalogError("library history shape differs from anchor ids")
        if self.episode_ids.shape != self.anchor_ids.shape:
            raise EpisodeAnalogError("episode ids differ from anchor ids")
        if len(np.unique(self.anchor_ids)) != len(self.anchor_ids):
            raise EpisodeAnalogError("library anchor ids are duplicated")
        if (self.episode_ids < 0).any():
            raise EpisodeAnalogError("library contains an invalid episode id")
        if not np.isfinite(self.histories).all():
            raise EpisodeAnalogError("library histories must be finite")
        if self.radius_steps != 6:
            raise EpisodeAnalogError("the frozen +/-2h band is exactly six 20-minute steps")
        if self.neighbor_count != 8:
            raise EpisodeAnalogError("the frozen analog uses exactly eight episodes")
        if self.batch_size < 1:
            raise EpisodeAnalogError("batch_size must be positive")
        if len(np.unique(self.episode_ids)) < self.neighbor_count:
            raise EpisodeAnalogError("library has fewer than eight distinct episodes")

        order = np.lexsort((self.anchor_ids, self.episode_ids))
        episodes = self.episode_ids[order]
        unique, start = np.unique(episodes, return_index=True)
        self._episode_rows = {
            int(episode): order[begin:end]
            for episode, begin, end in zip(
                unique,
                start,
                np.r_[start[1:], len(order)],
                strict=True,
            )
        }
        self.unique_episodes = unique.astype(np.int64)

    def select_nearest(self, query: np.ndarray) -> NeighborSelection:
        """Select the exact nearest anchor from each of the nearest 8 episodes."""

        query_array = np.asarray(query, dtype=np.float64)
        lower_bound = lb_keogh_distances(
            query_array,
            self.histories,
            radius_steps=self.radius_steps,
        )
        order = np.lexsort((self.anchor_ids, lower_bound))
        best: dict[int, tuple[float, int]] = {}
        evaluated = 0
        cursor = 0
        while cursor < len(order):
            if len(best) >= self.neighbor_count:
                threshold = sorted(best.values())[: self.neighbor_count][-1][0]
                if float(lower_bound[order[cursor]]) > threshold:
                    break
            stop = min(cursor + self.batch_size, len(order))
            rows = order[cursor:stop]
            distances = banded_dtw_distances(
                query_array,
                self.histories[rows],
                radius_steps=self.radius_steps,
            )
            for row, distance in zip(rows, distances, strict=True):
                episode = int(self.episode_ids[row])
                value = (float(distance), int(row))
                incumbent = best.get(episode)
                if incumbent is None or value < incumbent:
                    best[episode] = value
            evaluated += len(rows)
            cursor = stop

        ranked = sorted(
            (distance, episode, row)
            for episode, (distance, row) in best.items()
        )[: self.neighbor_count]
        if len(ranked) != self.neighbor_count:
            raise EpisodeAnalogError("exact search returned fewer than eight episodes")
        return NeighborSelection(
            indices=np.asarray([row for _, _, row in ranked], dtype=np.int64),
            distances=np.asarray([distance for distance, _, _ in ranked], dtype=np.float64),
            episodes=np.asarray([episode for _, episode, _ in ranked], dtype=np.int64),
            evaluated_candidates=int(evaluated),
            total_candidates=int(len(self.anchor_ids)),
        )

    def select_random_panels(
        self,
        query: np.ndarray,
        *,
        seed: int,
        query_key: str,
        panel_count: int,
    ) -> tuple[NeighborSelection, ...]:
        """Build deterministic uniform-random episode panels.

        Episode sampling is independent of histories and labels.  Within each
        sampled episode, its nearest past-shape anchor is used, matching the
        distinct-episode rule of the proposed estimator.
        """

        if panel_count < 1:
            raise EpisodeAnalogError("panel_count must be positive")
        digest = sha256(f"{seed}|{query_key}".encode()).digest()
        local_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        generator = np.random.default_rng(local_seed)
        panels = [
            np.sort(
                generator.choice(
                    self.unique_episodes,
                    size=self.neighbor_count,
                    replace=False,
                ).astype(np.int64)
            )
            for _ in range(panel_count)
        ]
        needed = np.unique(np.concatenate(panels))
        rows = np.concatenate([self._episode_rows[int(episode)] for episode in needed])
        distances = banded_dtw_distances(
            np.asarray(query, dtype=np.float64),
            self.histories[rows],
            radius_steps=self.radius_steps,
        )
        distance_by_row = {int(row): float(value) for row, value in zip(rows, distances, strict=True)}
        result: list[NeighborSelection] = []
        for episodes in panels:
            chosen: list[tuple[float, int, int]] = []
            evaluated = 0
            for episode in episodes:
                episode_rows = self._episode_rows[int(episode)]
                evaluated += len(episode_rows)
                ranked = min(
                    (
                        distance_by_row[int(row)],
                        int(self.anchor_ids[row]),
                        int(row),
                    )
                    for row in episode_rows
                )
                chosen.append((ranked[0], int(episode), ranked[2]))
            chosen.sort()
            result.append(
                NeighborSelection(
                    indices=np.asarray([row for _, _, row in chosen], dtype=np.int64),
                    distances=np.asarray([distance for distance, _, _ in chosen]),
                    episodes=np.asarray([episode for _, episode, _ in chosen], dtype=np.int64),
                    evaluated_candidates=int(evaluated),
                    total_candidates=int(len(self.anchor_ids)),
                )
            )
        return tuple(result)


def project_normalized_residual(
    neighbor_residuals: np.ndarray,
    distances: np.ndarray,
    *,
    distance_floor: float = 1e-6,
) -> np.ndarray:
    """Inverse-distance project a six-lead normalized future residual."""

    residuals = np.asarray(neighbor_residuals, dtype=np.float64)
    distance = np.asarray(distances, dtype=np.float64)
    if residuals.ndim != 2 or residuals.shape[1] != len(LEADS):
        raise EpisodeAnalogError("neighbor residuals must have six columns")
    if distance.shape != (len(residuals),):
        raise EpisodeAnalogError("distance shape differs from neighbor residuals")
    if not np.isfinite(residuals).all() or not np.isfinite(distance).all():
        raise EpisodeAnalogError("analog projection inputs must be finite")
    if distance_floor <= 0.0:
        raise EpisodeAnalogError("distance_floor must be positive")
    weight = 1.0 / np.maximum(distance, distance_floor)
    weight /= weight.sum()
    return np.sum(residuals * weight[:, None], axis=0)


def blend_candidate(
    control_prediction: np.ndarray,
    analog_prediction: np.ndarray,
    lead_h: np.ndarray,
    *,
    alpha: float = 0.2,
) -> np.ndarray:
    """Keep 3/6/9h exact and blend the analog only at 12/18/24h."""

    control = np.asarray(control_prediction, dtype=np.float64)
    analog = np.asarray(analog_prediction, dtype=np.float64)
    lead = np.asarray(lead_h, dtype=np.int64)
    if control.shape != analog.shape or control.shape != lead.shape:
        raise EpisodeAnalogError("candidate arrays must have identical shapes")
    if not np.isclose(alpha, 0.2, rtol=0.0, atol=0.0):
        raise EpisodeAnalogError("the frozen blend alpha is exactly 0.2")
    result = control.copy()
    active = np.isin(lead, ACTIVE_LEADS) & np.isfinite(analog)
    result[active] = (1.0 - alpha) * control[active] + alpha * analog[active]
    return np.clip(result, 0.0, 30.0)


def rmse(truth: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series) -> float:
    truth_array = np.asarray(truth, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(prediction_array - truth_array))))


def evaluate_b_precheck(
    case_rows: pd.DataFrame,
    *,
    maximum_fold_mse_ratio: float = 0.9,
    minimum_passing_folds: int = 2,
) -> dict[str, Any]:
    """Evaluate the fixed nearest-versus-random normalized-residual check."""

    required = {"fold", "nearest_normalized_mse", "random_normalized_mse"}
    if not required.issubset(case_rows):
        raise EpisodeAnalogError(f"B precheck lacks columns: {sorted(required - set(case_rows))}")
    if case_rows.empty:
        raise EpisodeAnalogError("B precheck received no eligible cases")
    fold_result: dict[str, Any] = {}
    passing = 0
    for fold, group in case_rows.groupby("fold", sort=True, observed=True):
        nearest = float(group["nearest_normalized_mse"].mean())
        random = float(group["random_normalized_mse"].mean())
        ratio = nearest / random if random > 0.0 else np.inf
        passed = bool(ratio <= maximum_fold_mse_ratio)
        passing += int(passed)
        fold_result[str(fold)] = {
            "eligible_cases": int(len(group)),
            "nearest_normalized_mse": nearest,
            "random_normalized_mse": random,
            "ratio": float(ratio),
            "pass": passed,
        }
    pooled_nearest = float(case_rows["nearest_normalized_mse"].mean())
    pooled_random = float(case_rows["random_normalized_mse"].mean())
    return {
        "pass": bool(passing >= minimum_passing_folds),
        "passing_folds": int(passing),
        "required_passing_folds": int(minimum_passing_folds),
        "maximum_fold_mse_ratio": float(maximum_fold_mse_ratio),
        "pooled_nearest_normalized_mse": pooled_nearest,
        "pooled_random_normalized_mse": pooled_random,
        "pooled_ratio": float(pooled_nearest / pooled_random),
        "by_fold": fold_result,
    }


def _metric_deltas(
    rows: pd.DataFrame,
    *,
    group_column: str,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for value, group in rows.groupby(group_column, sort=True, observed=True):
        control = rmse(group["target_hs"], group["control_final"])
        candidate = rmse(group["target_hs"], group["candidate_final"])
        result[str(value)] = {
            "control_rmse_m": control,
            "candidate_rmse_m": candidate,
            "delta_m": candidate - control,
        }
    return result


def evaluate_inner_gate(
    rows: pd.DataFrame,
    *,
    maximum_pooled_delta_m: float = -0.005,
    minimum_improved_folds: int = 2,
    maximum_station_degradation_m: float = 0.01,
) -> dict[str, Any]:
    """Evaluate the preregistered inner-only promotion gate."""

    required = {
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "target_hs",
        "control_final",
        "candidate_final",
    }
    if not required.issubset(rows):
        raise EpisodeAnalogError(f"inner gate lacks columns: {sorted(required - set(rows))}")
    if rows.empty or rows.duplicated(["fold", "anchor_id", "lead_h"]).any():
        raise EpisodeAnalogError("inner gate rows are empty or duplicated")
    control = rmse(rows["target_hs"], rows["control_final"])
    candidate = rmse(rows["target_hs"], rows["candidate_final"])
    pooled_delta = candidate - control
    by_fold = _metric_deltas(rows, group_column="fold")
    by_station = _metric_deltas(rows, group_column="station")
    by_lead = _metric_deltas(rows, group_column="lead_h")
    improved_folds = sum(item["delta_m"] < 0.0 for item in by_fold.values())
    maximum_station_delta = max(item["delta_m"] for item in by_station.values())
    lead_18_ok = by_lead["18"]["delta_m"] <= 0.0
    lead_24_ok = by_lead["24"]["delta_m"] <= 0.0
    checks = {
        "pooled_delta_at_most_minus_0_005m": pooled_delta <= maximum_pooled_delta_m,
        "at_least_two_folds_improve": improved_folds >= minimum_improved_folds,
        "no_station_degrades_over_0_01m": maximum_station_delta <= maximum_station_degradation_m,
        "lead_18_non_degrading": lead_18_ok,
        "lead_24_non_degrading": lead_24_ok,
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "control_rmse_m": control,
        "candidate_rmse_m": candidate,
        "pooled_delta_m": pooled_delta,
        "improved_folds": int(improved_folds),
        "maximum_station_delta_m": float(maximum_station_delta),
        "by_fold": by_fold,
        "by_station": by_station,
        "by_lead": by_lead,
    }


__all__ = [
    "ACTIVE_LEADS",
    "EpisodeAnalogError",
    "EpisodeAnalogIndex",
    "HISTORY_POINTS",
    "LEADS",
    "NeighborSelection",
    "PreparedHistories",
    "SHORT_LEADS",
    "banded_dtw_distances",
    "blend_candidate",
    "evaluate_b_precheck",
    "evaluate_inner_gate",
    "lb_keogh_distances",
    "prepare_histories",
    "project_normalized_residual",
    "rmse",
]
