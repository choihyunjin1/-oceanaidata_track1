from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p3_trajectory import (
    HISTORY_POINTS,
    PATH_STEPS,
    ClosedFormTrajectoryRegressor,
    attach_official_targets,
    build_blind_prediction_frame,
    build_trajectory_dataset,
    event_balanced_weights,
    select_lattice_phase,
)


def _wave(rows: int = 420) -> pd.DataFrame:
    time = pd.date_range("2024-01-01", periods=rows, freq="20min", tz="UTC")
    hs = 2.0 + 0.25 * np.sin(np.arange(rows) / 19.0)
    return pd.DataFrame({"station": "G-ORS", "time": time, "hs": hs})


def test_builds_72_step_complete_paths_and_official_targets() -> None:
    wave = _wave()
    dataset = build_trajectory_dataset(wave)
    assert dataset.history.shape == (len(dataset.anchors), HISTORY_POINTS)
    assert dataset.path_target.shape == (len(dataset.anchors), PATH_STEPS)
    assert dataset.official_target.shape == (len(dataset.anchors), 6)
    assert dataset.complete_path.all()
    first_position = int(dataset.anchors.iloc[0]["grid_position_20m"])
    assert dataset.path_target[0, 0] == pytest.approx(wave.loc[first_position + 1, "hs"])
    assert dataset.official_target[0, 0] == pytest.approx(wave.loc[first_position + 9, "hs"])


def test_event_weight_is_inverse_sqrt_count_and_mean_normalized() -> None:
    anchors = pd.DataFrame(
        {
            "anchor_id": np.arange(5),
            "episode_id": [10, 10, 10, 10, 20],
        }
    )
    weight = event_balanced_weights(anchors, np.arange(5))
    assert np.isclose(weight.mean(), 1.0)
    assert np.isclose(weight[-1] / weight[0], 2.0)


def test_lattice_phase_is_frozen_and_station_local() -> None:
    time = pd.date_range("2025-01-01", periods=48, freq="2h", tz="UTC")
    anchors = pd.DataFrame(
        {
            "anchor_id": np.arange(96),
            "station": ["G-ORS"] * 48 + ["I-ORS"] * 48,
            "anchor_time": list(time) * 2,
        }
    )
    chosen = select_lattice_phase(
        anchors,
        start="2025-01-01",
        end="2025-01-05",
        phase_hours=10,
        gap_hours=20,
    )
    selected = anchors.set_index("anchor_id").loc[chosen]
    assert selected["anchor_time"].min() == pd.Timestamp("2025-01-01 10:00", tz="UTC")
    for _, group in selected.groupby("station", observed=True):
        assert group["anchor_time"].diff().dropna().ge(pd.Timedelta(hours=20)).all()


def test_fixed_models_are_deterministic_and_emit_full_path() -> None:
    rng = np.random.default_rng(20260821)
    history = rng.normal(size=(220, HISTORY_POINTS)).cumsum(axis=1).astype(np.float32)
    station = np.asarray(["G-ORS", "I-ORS", "S-ORS", "G-ORS"] * 55)
    slope = history[:, -1] - history[:, -7]
    target = np.column_stack(
        [slope * (step / PATH_STEPS) for step in range(1, PATH_STEPS + 1)]
    )
    weight = np.ones(len(history))
    for variant in ("nlinear", "dlinear_trend"):
        first = ClosedFormTrajectoryRegressor(variant=variant).fit(
            history, station, target, weight
        )
        second = ClosedFormTrajectoryRegressor(variant=variant).fit(
            history, station, target, weight
        )
        prediction = first.predict_delta(history[:8], station[:8])
        assert prediction.shape == (8, PATH_STEPS)
        assert np.isfinite(prediction).all()
        assert np.array_equal(prediction, second.predict_delta(history[:8], station[:8]))
        assert np.all(first.loss_weights[np.asarray([9, 18, 27, 36, 54, 72]) - 1] == 1.0)
        assert np.count_nonzero(first.loss_weights == 0.1) == PATH_STEPS - 6


def test_holdout_frame_is_blind_until_explicit_target_attach() -> None:
    dataset = build_trajectory_dataset(_wave())
    ids = dataset.anchors["anchor_id"].to_numpy(dtype=np.int64)[:3]
    blind = build_blind_prediction_frame(
        dataset,
        ids,
        np.zeros((3, PATH_STEPS)),
        fold="synthetic",
        phase="holdout",
        variant="nlinear",
    )
    assert "target_hs" not in blind.columns
    evaluated = attach_official_targets(dataset, blind)
    assert "target_hs" in evaluated.columns
    assert np.isfinite(evaluated["target_hs"]).all()
