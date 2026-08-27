from __future__ import annotations

import numpy as np
import pandas as pd

from p3_wave.corrected_repeated_forward import (
    build_corrected_repeated_forward_folds,
    fixed_prequential_lead_router,
    select_station_global_validation,
)
from p3_wave.loss_router import RouterConfig

WINDOWS = (
    ("f1", "2024-01-01", "2024-01-05"),
    ("f2", "2024-01-05", "2024-01-10"),
)


def _anchors() -> pd.DataFrame:
    rows = []
    anchor_id = 0
    for station in ("A", "B"):
        for hour in range(-300, 240):
            timestamp = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=hour)
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "station": station,
                    "anchor_time": timestamp,
                    "episode_id": hour // 12 + (0 if station == "A" else 10_000),
                }
            )
            anchor_id += 1
    return pd.DataFrame(rows)


def test_global_selection_does_not_reset_at_window_boundary() -> None:
    anchors = _anchors()
    selected = select_station_global_validation(anchors, windows=WINDOWS)
    for _, group in selected.groupby("station"):
        gap = group.sort_values("anchor_time")["anchor_time"].diff().dropna()
        assert gap.ge(pd.Timedelta(hours=78)).all()
    assert not selected.duplicated(["station", "episode_id"]).any()
    assert set(selected["fold"]) == {"f1", "f2"}


def test_corrected_folds_remove_validation_episodes_and_footprint_overlap() -> None:
    anchors = _anchors()
    folds, selected, audit = build_corrected_repeated_forward_folds(anchors, windows=WINDOWS)
    lookup = anchors.set_index("anchor_id")
    for fold in folds:
        train = lookup.loc[fold.train_ids]
        valid = lookup.loc[fold.validation_ids]
        train_episode = set(zip(train["station"], train["episode_id"], strict=True))
        valid_episode = set(zip(valid["station"], valid["episode_id"], strict=True))
        assert not train_episode.intersection(valid_episode)
        assert audit["folds"][fold.name]["minimum_train_validation_anchor_gap_hours"] >= 78
    assert audit["validation_case_count"] == len(selected)
    assert audit["context48_plus_target24_footprint_overlap_pairs"] == 0


def test_fixed_router_never_uses_current_fold_losses() -> None:
    cases_per_fold = 12
    rows = cases_per_fold * 2 * 6
    metadata = pd.DataFrame(
        {
            "fold": np.repeat(["f1", "f2"], cases_per_fold * 6),
            "anchor_id": np.repeat(np.arange(cases_per_fold * 2), 6),
            "station": np.tile(np.repeat(["A", "B"], 3), cases_per_fold * 2),
            "lead_h": np.tile([3, 6, 9, 12, 18, 24], cases_per_fold * 2),
        }
    )
    features = pd.DataFrame(
        {
            "station": metadata["station"],
            "lead_h": metadata["lead_h"].astype(str),
            "x": np.linspace(0.0, 1.0, rows),
        }
    )
    components = np.column_stack(
        [np.linspace(1.0, 2.0, rows), np.linspace(1.1, 2.1, rows), np.ones(rows)]
    )
    losses = np.square(components - 1.5)
    config = RouterConfig(10.0, 2.0, 0.5, "smooth_medium")
    first = fixed_prequential_lead_router(
        features,
        metadata,
        components,
        losses,
        fold_order=("f1", "f2"),
        config=config,
    )
    changed = losses.copy()
    changed[metadata["fold"].eq("f2").to_numpy()] += 100.0
    second = fixed_prequential_lead_router(
        features,
        metadata,
        components,
        changed,
        fold_order=("f1", "f2"),
        config=config,
    )
    current = metadata["fold"].eq("f2").to_numpy()
    np.testing.assert_allclose(first[0][current], second[0][current])
    np.testing.assert_allclose(first[1][current], second[1][current])
    inactive = ~metadata["lead_h"].isin([12, 18, 24]).to_numpy()
    np.testing.assert_allclose(
        first[1][inactive],
        np.broadcast_to([0.5, 0.5, 0.0], first[1][inactive].shape),
    )
