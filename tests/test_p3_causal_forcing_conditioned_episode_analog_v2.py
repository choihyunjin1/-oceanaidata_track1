from __future__ import annotations

import numpy as np
import pytest

from p3_wave.causal_forcing_analog import (
    FORCING_COLUMNS,
    FORCING_DIMENSIONS,
    ForcingConditionedAnalogIndex,
    ForcingScaler,
)
from p3_wave.episode_distinct_analog import HISTORY_POINTS, banded_dtw_distances


def _histories(episodes: int, anchors_per_episode: int = 1) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, HISTORY_POINTS)
    values: list[np.ndarray] = []
    labels: list[int] = []
    for episode in range(episodes):
        for anchor in range(anchors_per_episode):
            values.append(x + 0.01 * episode + 0.002 * anchor * np.sin(np.pi * x))
            labels.append(episode)
    return np.vstack(values), np.asarray(labels, dtype=np.int64)


def test_forcing_column_contract_is_fixed_and_past_only() -> None:
    assert FORCING_COLUMNS == (
        "wind_input_proxy_mean_6h",
        "wind_input_proxy_slope_6h",
        "wind_wave_alignment_mean_6h",
        "gust_excess_mean_6h",
        "caph_slope_6h",
        "tp_slope_6h",
    )
    assert all("target" not in column and "future" not in column for column in FORCING_COLUMNS)


def test_forcing_scaler_uses_componentwise_median_and_iqr() -> None:
    values = np.tile(np.arange(5, dtype=np.float64)[:, None], (1, FORCING_DIMENSIONS))
    scaler = ForcingScaler.fit(values)

    assert scaler.center == pytest.approx(np.full(FORCING_DIMENSIONS, 2.0))
    assert scaler.scale == pytest.approx(np.full(FORCING_DIMENSIONS, 2.0))
    assert scaler.transform(np.full(FORCING_DIMENSIONS, 4.0)) == pytest.approx(
        np.ones(FORCING_DIMENSIONS)
    )


def test_forcing_scaler_uses_one_for_zero_iqr() -> None:
    values = np.full((10, FORCING_DIMENSIONS), 3.0)
    scaler = ForcingScaler.fit(values)

    assert scaler.scale == pytest.approx(np.ones(FORCING_DIMENSIONS))
    assert scaler.transform(values).sum() == pytest.approx(0.0)


def test_combined_pruned_search_matches_exhaustive_episode_search() -> None:
    generator = np.random.default_rng(20260822)
    histories, episodes = _histories(12, anchors_per_episode=2)
    forcing = generator.normal(size=(len(histories), FORCING_DIMENSIONS))
    query_history = generator.normal(scale=0.2, size=HISTORY_POINTS)
    query_forcing = generator.normal(size=FORCING_DIMENSIONS)
    index = ForcingConditionedAnalogIndex(
        anchor_ids=np.arange(100, 100 + len(histories)),
        episode_ids=episodes,
        normalized_histories=histories,
        forcing_state=forcing,
        batch_size=5,
    )
    selected = index.select_nearest(query_history, query_forcing)
    hs_distance = banded_dtw_distances(query_history, histories, radius_steps=6)
    scaled = index.scaler.transform(forcing)
    query_scaled = index.scaler.transform(query_forcing)
    forcing_distance = np.sqrt(np.mean(np.square(scaled - query_scaled), axis=1))
    combined = np.sqrt(np.square(hs_distance) + np.square(forcing_distance))
    exhaustive: list[tuple[float, int, int, int]] = []
    anchor_ids = index.anchor_ids
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        best = min(
            (float(combined[row]), int(anchor_ids[row]), int(row)) for row in rows
        )
        exhaustive.append((best[0], int(episode), best[1], best[2]))
    exhaustive.sort()

    assert selected.conditioning_used is True
    assert selected.neighbors.indices.tolist() == [row for _, _, _, row in exhaustive[:8]]
    assert len(np.unique(selected.neighbors.episodes)) == 8


def test_query_missing_forcing_falls_back_exactly_to_v1() -> None:
    histories, episodes = _histories(12)
    forcing = np.tile(np.arange(12, dtype=np.float64)[:, None], (1, FORCING_DIMENSIONS))
    index = ForcingConditionedAnalogIndex(
        anchor_ids=np.arange(12),
        episode_ids=episodes,
        normalized_histories=histories,
        forcing_state=forcing,
    )
    query = histories[4]
    selected = index.select_nearest(query, np.full(FORCING_DIMENSIONS, np.nan))
    expected = index.base.select_nearest(query)

    assert selected.conditioning_used is False
    assert selected.fallback_reason == "query_forcing_nonfinite"
    assert selected.neighbors.indices.tolist() == expected.indices.tolist()
    assert selected.neighbors.distances == pytest.approx(expected.distances)


def test_fewer_than_eight_complete_episodes_falls_back_to_v1() -> None:
    histories, episodes = _histories(12)
    forcing = np.full((12, FORCING_DIMENSIONS), np.nan)
    forcing[:7] = np.arange(7, dtype=np.float64)[:, None]
    index = ForcingConditionedAnalogIndex(
        anchor_ids=np.arange(12),
        episode_ids=episodes,
        normalized_histories=histories,
        forcing_state=forcing,
    )
    selected = index.select_nearest(histories[0], np.zeros(FORCING_DIMENSIONS))

    assert selected.conditioning_used is False
    assert selected.fallback_reason == "fewer_than_8_complete_forcing_episodes"
    assert selected.conditioned_library_episodes == 7


def test_forcing_state_changes_ranking_when_hs_shapes_tie() -> None:
    history = np.linspace(-1.0, 1.0, HISTORY_POINTS)
    histories = np.tile(history, (12, 1))
    episodes = np.arange(12, dtype=np.int64)
    forcing = np.tile(np.arange(12, dtype=np.float64)[:, None], (1, FORCING_DIMENSIONS))
    index = ForcingConditionedAnalogIndex(
        anchor_ids=np.arange(12),
        episode_ids=episodes,
        normalized_histories=histories,
        forcing_state=forcing,
    )
    selected = index.select_nearest(history, np.full(FORCING_DIMENSIONS, 10.0))

    assert selected.conditioning_used is True
    assert selected.neighbors.episodes[0] == 10
    assert 0 not in selected.neighbors.episodes


def test_random_reference_delegates_exactly_to_v1() -> None:
    histories, episodes = _histories(12)
    forcing = np.tile(np.arange(12, dtype=np.float64)[:, None], (1, FORCING_DIMENSIONS))
    index = ForcingConditionedAnalogIndex(
        anchor_ids=np.arange(12),
        episode_ids=episodes,
        normalized_histories=histories,
        forcing_state=forcing,
    )
    actual = index.select_random_panels(
        histories[0], seed=20260822, query_key="fold|1", panel_count=4
    )
    expected = index.base.select_random_panels(
        histories[0], seed=20260822, query_key="fold|1", panel_count=4
    )

    assert [item.indices.tolist() for item in actual] == [
        item.indices.tolist() for item in expected
    ]
    assert np.array_equal(
        np.vstack([item.distances for item in actual]),
        np.vstack([item.distances for item in expected]),
    )
