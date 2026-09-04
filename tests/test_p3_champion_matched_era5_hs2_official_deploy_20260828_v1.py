from __future__ import annotations

import numpy as np
import pandas as pd

from p3_wave.champion_matched_era5_hs2_official_deploy import (
    align_transfer_predictions,
    build_relative_test_features,
    make_candidate,
)
from p3_wave.era5_context_transfer import LEADS, common_feature_columns


def _official_shape_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks: list[pd.DataFrame] = []
    index_rows: list[dict[str, object]] = []
    steps = np.arange(-2880, 1, 10)
    for number in range(200):
        case_id = f"case_{number:03d}"
        station = ("G-ORS", "I-ORS", "S-ORS")[number % 3]
        phase = np.arange(len(steps), dtype=np.float64) / 30.0
        block = pd.DataFrame(
            {
                "case_id": case_id,
                "station": station,
                "step_minute": steps,
                "hs": 2.0 + 0.01 * np.sin(phase),
                "tp": 8.0 + 0.01 * np.cos(phase),
                "hmax": 3.0 + 0.01 * np.sin(phase),
                "wvdir": np.full(len(steps), 180.0),
                "wspd": np.full(len(steps), 6.0),
                "gust": np.full(len(steps), 8.0),
                "wdir": np.full(len(steps), 190.0),
                "airt": np.full(len(steps), 15.0),
                "relh": np.full(len(steps), 70.0),
                "caph": np.full(len(steps), 1010.0),
            }
        )
        blocks.append(block)
        index_rows.extend(
            {"case_id": case_id, "station": station, "lead_h": lead} for lead in LEADS
        )
    return pd.concat(blocks, ignore_index=True), pd.DataFrame(index_rows)


def test_relative_feature_builder_has_frozen_surface_and_order() -> None:
    context, test_index = _official_shape_fixture()
    features, metadata = build_relative_test_features(context, test_index)
    assert tuple(features.columns) == ("case_id", *common_feature_columns())
    assert features.shape == (200, 287)
    assert metadata.shape == (200, 3)
    assert not any("time" in column or "station" in column for column in common_feature_columns())


def test_alignment_respects_immutable_test_row_order() -> None:
    _, test_index = _official_shape_fixture()
    metadata = test_index[["case_id", "station"]].drop_duplicates().reset_index(drop=True)
    metadata["current_hs"] = 2.0
    matrix = np.arange(200 * 6, dtype=np.float64).reshape(200, 6)
    aligned = align_transfer_predictions(test_index, metadata, matrix)
    assert np.array_equal(aligned, matrix.reshape(-1))


def test_candidate_changes_only_18_and_24_hour_rows() -> None:
    leads = np.tile(np.asarray(LEADS), 200)
    champion = np.linspace(1.0, 3.0, 1200)
    transfer = champion + 0.5
    candidate, active = make_candidate(champion, transfer, leads)
    assert int(active.sum()) == 400
    assert np.array_equal(candidate[~active], champion[~active])
    assert np.all(candidate[active] > champion[active])
