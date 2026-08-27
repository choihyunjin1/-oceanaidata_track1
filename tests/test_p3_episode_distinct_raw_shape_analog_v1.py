from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p3_wave.episode_distinct_analog import (
    HISTORY_POINTS,
    LEADS,
    EpisodeAnalogIndex,
    banded_dtw_distances,
    blend_candidate,
    evaluate_b_precheck,
    evaluate_inner_gate,
    lb_keogh_distances,
    prepare_histories,
    project_normalized_residual,
)


def _wave(pulse_position: int, amplitude: float = 1.0) -> np.ndarray:
    values = np.zeros(HISTORY_POINTS, dtype=np.float64)
    values[pulse_position] = amplitude
    return values


def test_prepare_histories_uses_current_center_and_raw_mad_floor() -> None:
    ramp = np.linspace(1.0, 3.0, HISTORY_POINTS)
    flat = np.full(HISTORY_POINTS, 2.5)
    values = np.vstack([ramp, flat])
    prepared = prepare_histories(values, minimum_coverage=0.95, mad_floor=0.1)

    assert prepared.eligible.tolist() == [True, True]
    assert prepared.normalized[0, -1] == pytest.approx(0.0)
    assert prepared.normalized[1].tolist() == pytest.approx([0.0] * HISTORY_POINTS)
    assert prepared.scale[1] == pytest.approx(0.1)


def test_prepare_histories_rejects_low_coverage_but_interpolates_short_gaps() -> None:
    values = np.vstack(
        [
            np.linspace(1.0, 2.0, HISTORY_POINTS),
            np.linspace(1.0, 2.0, HISTORY_POINTS),
        ]
    )
    values[0, 20:24] = np.nan
    values[1, :20] = np.nan
    prepared = prepare_histories(values, minimum_coverage=0.95, mad_floor=0.1)

    assert prepared.eligible.tolist() == [True, False]
    assert np.isfinite(prepared.normalized[0]).all()
    assert np.isnan(prepared.normalized[1]).all()


def test_banded_dtw_zero_radius_equals_euclidean_rms() -> None:
    query = np.linspace(-1.0, 1.0, HISTORY_POINTS)
    candidates = np.vstack([query, query + 0.25])
    distance = banded_dtw_distances(query, candidates, radius_steps=0)
    expected = np.sqrt(np.mean(np.square(candidates - query[None, :]), axis=1))

    assert distance == pytest.approx(expected)


def test_lb_keogh_is_admissible_for_banded_dtw() -> None:
    generator = np.random.default_rng(20260822)
    query = generator.normal(size=HISTORY_POINTS)
    candidates = generator.normal(size=(24, HISTORY_POINTS))
    lower = lb_keogh_distances(query, candidates, radius_steps=6)
    exact = banded_dtw_distances(query, candidates, radius_steps=6)

    assert np.all(lower <= exact + 1e-12)


def test_episode_distinct_pruned_search_matches_exhaustive_search() -> None:
    query = _wave(72)
    histories: list[np.ndarray] = []
    episodes: list[int] = []
    for episode in range(12):
        histories.extend(
            [
                _wave(66 + episode % 7, 1.0 + 0.01 * episode),
                _wave(67 + episode % 7, 1.1 + 0.01 * episode),
            ]
        )
        episodes.extend([episode, episode])
    matrix = np.vstack(histories)
    episode_array = np.asarray(episodes, dtype=np.int64)
    anchor_ids = np.arange(100, 100 + len(matrix), dtype=np.int64)
    index = EpisodeAnalogIndex(
        anchor_ids=anchor_ids,
        episode_ids=episode_array,
        normalized_histories=matrix,
        radius_steps=6,
        neighbor_count=8,
        batch_size=5,
    )
    selected = index.select_nearest(query)
    distance = banded_dtw_distances(query, matrix, radius_steps=6)
    exhaustive: list[tuple[float, int, int]] = []
    for episode in np.unique(episode_array):
        rows = np.flatnonzero(episode_array == episode)
        best = min((float(distance[row]), int(anchor_ids[row]), int(row)) for row in rows)
        exhaustive.append((best[0], int(episode), best[2]))
    exhaustive.sort()

    assert selected.indices.tolist() == [row for _, _, row in exhaustive[:8]]
    assert len(np.unique(selected.episodes)) == 8


def test_random_episode_panels_are_deterministic_and_distinct() -> None:
    histories = np.vstack([_wave(60 + episode % 13) for episode in range(16)])
    index = EpisodeAnalogIndex(
        anchor_ids=np.arange(16),
        episode_ids=np.arange(16),
        normalized_histories=histories,
        radius_steps=6,
        neighbor_count=8,
    )
    first = index.select_random_panels(
        _wave(72), seed=7, query_key="fold|1", panel_count=3
    )
    second = index.select_random_panels(
        _wave(72), seed=7, query_key="fold|1", panel_count=3
    )

    assert [item.episodes.tolist() for item in first] == [
        item.episodes.tolist() for item in second
    ]
    assert all(len(np.unique(item.episodes)) == 8 for item in first)


def test_inverse_distance_projection_and_fixed_candidate_route() -> None:
    residual = np.vstack([np.zeros(6), np.full(6, 2.0)])
    projected = project_normalized_residual(residual, np.asarray([1.0, 3.0]))
    control = np.arange(6, dtype=np.float64)
    analog = control + projected
    candidate = blend_candidate(
        control,
        analog,
        np.asarray(LEADS, dtype=np.int64),
        alpha=0.2,
    )

    assert projected == pytest.approx(np.full(6, 0.5))
    assert np.array_equal(candidate[:3], control[:3])
    assert candidate[3:] == pytest.approx(control[3:] + 0.1)


def test_b_precheck_requires_two_passing_folds() -> None:
    rows = pd.DataFrame(
        {
            "fold": ["a", "a", "b", "b", "c", "c"],
            "nearest_normalized_mse": [0.8, 0.8, 0.85, 0.85, 0.95, 0.95],
            "random_normalized_mse": [1.0] * 6,
        }
    )
    result = evaluate_b_precheck(rows)

    assert result["pass"] is True
    assert result["passing_folds"] == 2


def test_inner_gate_enforces_short_no_op_and_long_improvement_metrics() -> None:
    rows: list[dict[str, object]] = []
    anchor_id = 0
    for fold, station in zip(("a", "b", "c"), ("G-ORS", "I-ORS", "S-ORS"), strict=True):
        for lead in LEADS:
            rows.append(
                {
                    "fold": fold,
                    "anchor_id": anchor_id,
                    "station": station,
                    "lead_h": lead,
                    "target_hs": 0.0,
                    "control_final": 1.0,
                    "candidate_final": 1.0 if lead in (3, 6, 9) else 0.9,
                }
            )
        anchor_id += 1
    result = evaluate_inner_gate(pd.DataFrame(rows))

    assert result["pass"] is True
    assert result["pooled_delta_m"] <= -0.005
    assert result["by_lead"]["3"]["delta_m"] == pytest.approx(0.0)
    assert result["by_lead"]["18"]["delta_m"] < 0.0
