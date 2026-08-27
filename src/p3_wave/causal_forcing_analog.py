"""Causal forcing-conditioned extension of the P3 episode analog.

Only the neighbour geometry changes from v1.  The raw 145-point hs DTW,
same-station scope, episode distinctness, k=8 and inverse-distance residual
projection remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p3_wave.episode_distinct_analog import (
    HISTORY_POINTS,
    EpisodeAnalogError,
    EpisodeAnalogIndex,
    NeighborSelection,
    banded_dtw_distances,
    lb_keogh_distances,
)

FORCING_COLUMNS = (
    "wind_input_proxy_mean_6h",
    "wind_input_proxy_slope_6h",
    "wind_wave_alignment_mean_6h",
    "gust_excess_mean_6h",
    "caph_slope_6h",
    "tp_slope_6h",
)
FORCING_DIMENSIONS = len(FORCING_COLUMNS)


@dataclass(frozen=True)
class ForcingScaler:
    """Library-only componentwise median/IQR scaler."""

    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> ForcingScaler:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != FORCING_DIMENSIONS:
            raise EpisodeAnalogError(
                f"forcing values must have {FORCING_DIMENSIONS} columns"
            )
        if len(matrix) == 0 or not np.isfinite(matrix).all():
            raise EpisodeAnalogError("forcing scaler requires finite complete rows")
        center = np.median(matrix, axis=0)
        q25 = np.quantile(matrix, 0.25, axis=0)
        q75 = np.quantile(matrix, 0.75, axis=0)
        scale = q75 - q25
        scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
        return cls(center=center, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape[-1] != FORCING_DIMENSIONS:
            raise EpisodeAnalogError(
                f"forcing values must have {FORCING_DIMENSIONS} columns"
            )
        if not np.isfinite(matrix).all():
            raise EpisodeAnalogError("forcing transform received a non-finite value")
        return (matrix - self.center) / self.scale


@dataclass(frozen=True)
class ConditionedSelection:
    """A neighbour selection plus its deterministic fallback audit."""

    neighbors: NeighborSelection
    conditioning_used: bool
    fallback_reason: str | None
    forcing_distance_mean: float | None
    forcing_distance_max: float | None
    conditioned_library_anchors: int
    conditioned_library_episodes: int


class ForcingConditionedAnalogIndex:
    """Exact combined hs-DTW and causal-forcing episode retrieval."""

    def __init__(
        self,
        *,
        anchor_ids: np.ndarray,
        episode_ids: np.ndarray,
        normalized_histories: np.ndarray,
        forcing_state: np.ndarray,
        radius_steps: int = 6,
        neighbor_count: int = 8,
        batch_size: int = 1024,
    ) -> None:
        self.anchor_ids = np.asarray(anchor_ids, dtype=np.int64)
        self.episode_ids = np.asarray(episode_ids, dtype=np.int64)
        self.histories = np.asarray(normalized_histories, dtype=np.float64)
        self.forcing_state = np.asarray(forcing_state, dtype=np.float64)
        self.radius_steps = int(radius_steps)
        self.neighbor_count = int(neighbor_count)
        self.batch_size = int(batch_size)
        if self.forcing_state.shape != (len(self.anchor_ids), FORCING_DIMENSIONS):
            raise EpisodeAnalogError("forcing-state rows differ from library anchors")
        self.base = EpisodeAnalogIndex(
            anchor_ids=self.anchor_ids,
            episode_ids=self.episode_ids,
            normalized_histories=self.histories,
            radius_steps=self.radius_steps,
            neighbor_count=self.neighbor_count,
            batch_size=self.batch_size,
        )

        complete = np.isfinite(self.forcing_state).all(axis=1)
        self.complete_rows = np.flatnonzero(complete).astype(np.int64)
        complete_episodes = np.unique(self.episode_ids[self.complete_rows])
        self.conditioned_library_episodes = int(len(complete_episodes))
        self.scaler: ForcingScaler | None = None
        self.scaled_forcing = np.empty((0, FORCING_DIMENSIONS), dtype=np.float64)
        if self.conditioned_library_episodes >= self.neighbor_count:
            self.scaler = ForcingScaler.fit(self.forcing_state[self.complete_rows])
            self.scaled_forcing = self.scaler.transform(
                self.forcing_state[self.complete_rows]
            )

    def select_nearest(
        self,
        query_history: np.ndarray,
        query_forcing_state: np.ndarray,
    ) -> ConditionedSelection:
        query_state = np.asarray(query_forcing_state, dtype=np.float64)
        if query_state.shape != (FORCING_DIMENSIONS,):
            raise EpisodeAnalogError(
                f"query forcing state must have shape ({FORCING_DIMENSIONS},)"
            )
        if not np.isfinite(query_state).all():
            return self._fallback(query_history, "query_forcing_nonfinite")
        if self.scaler is None:
            return self._fallback(query_history, "fewer_than_8_complete_forcing_episodes")

        query = np.asarray(query_history, dtype=np.float64)
        if query.shape != (HISTORY_POINTS,) or not np.isfinite(query).all():
            raise EpisodeAnalogError(
                f"query history must be finite with shape ({HISTORY_POINTS},)"
            )
        query_scaled = self.scaler.transform(query_state)
        forcing_distance = np.sqrt(
            np.mean(np.square(self.scaled_forcing - query_scaled[None, :]), axis=1)
        )
        complete_histories = self.histories[self.complete_rows]
        hs_lower_bound = lb_keogh_distances(
            query,
            complete_histories,
            radius_steps=self.radius_steps,
        )
        combined_lower_bound = np.sqrt(
            np.square(hs_lower_bound) + np.square(forcing_distance)
        )
        complete_anchor_ids = self.anchor_ids[self.complete_rows]
        order = np.lexsort((complete_anchor_ids, combined_lower_bound))
        best: dict[int, tuple[float, int, int, float]] = {}
        evaluated = 0
        cursor = 0
        while cursor < len(order):
            if len(best) >= self.neighbor_count:
                threshold = sorted(best.values())[: self.neighbor_count][-1][0]
                if float(combined_lower_bound[order[cursor]]) > threshold:
                    break
            stop = min(cursor + self.batch_size, len(order))
            local_rows = order[cursor:stop]
            hs_distance = banded_dtw_distances(
                query,
                complete_histories[local_rows],
                radius_steps=self.radius_steps,
            )
            combined = np.sqrt(
                np.square(hs_distance) + np.square(forcing_distance[local_rows])
            )
            for local_row, distance, forcing in zip(
                local_rows,
                combined,
                forcing_distance[local_rows],
                strict=True,
            ):
                full_row = int(self.complete_rows[local_row])
                episode = int(self.episode_ids[full_row])
                value = (
                    float(distance),
                    int(self.anchor_ids[full_row]),
                    full_row,
                    float(forcing),
                )
                incumbent = best.get(episode)
                if incumbent is None or value[:2] < incumbent[:2]:
                    best[episode] = value
            evaluated += len(local_rows)
            cursor = stop

        ranked = sorted(
            (distance, episode, anchor_id, row, forcing)
            for episode, (distance, anchor_id, row, forcing) in best.items()
        )[: self.neighbor_count]
        if len(ranked) != self.neighbor_count:
            raise EpisodeAnalogError("combined search returned fewer than eight episodes")
        force = np.asarray([item[4] for item in ranked], dtype=np.float64)
        neighbors = NeighborSelection(
            indices=np.asarray([item[3] for item in ranked], dtype=np.int64),
            distances=np.asarray([item[0] for item in ranked], dtype=np.float64),
            episodes=np.asarray([item[1] for item in ranked], dtype=np.int64),
            evaluated_candidates=int(evaluated),
            total_candidates=int(len(self.complete_rows)),
        )
        return ConditionedSelection(
            neighbors=neighbors,
            conditioning_used=True,
            fallback_reason=None,
            forcing_distance_mean=float(force.mean()),
            forcing_distance_max=float(force.max()),
            conditioned_library_anchors=int(len(self.complete_rows)),
            conditioned_library_episodes=self.conditioned_library_episodes,
        )

    def select_random_panels(
        self,
        query_history: np.ndarray,
        *,
        seed: int,
        query_key: str,
        panel_count: int,
    ) -> tuple[NeighborSelection, ...]:
        """Delegate exactly to the v1 unconditioned random reference."""

        return self.base.select_random_panels(
            query_history,
            seed=seed,
            query_key=query_key,
            panel_count=panel_count,
        )

    def _fallback(self, query_history: np.ndarray, reason: str) -> ConditionedSelection:
        neighbors = self.base.select_nearest(query_history)
        return ConditionedSelection(
            neighbors=neighbors,
            conditioning_used=False,
            fallback_reason=reason,
            forcing_distance_mean=None,
            forcing_distance_max=None,
            conditioned_library_anchors=int(len(self.complete_rows)),
            conditioned_library_episodes=self.conditioned_library_episodes,
        )


__all__ = [
    "ConditionedSelection",
    "FORCING_COLUMNS",
    "FORCING_DIMENSIONS",
    "ForcingConditionedAnalogIndex",
    "ForcingScaler",
]
