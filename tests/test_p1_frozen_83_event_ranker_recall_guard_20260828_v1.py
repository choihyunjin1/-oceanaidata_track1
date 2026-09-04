from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p1_qc.p1_conditional_real_event_donor_20260828_v1 import EventSpan
from p1_qc.p1_frozen_83_event_ranker_recall_guard_20260828_v1 import (
    chronological_split,
    mask_from_proposals,
    proposal_feature_matrix,
    select_recall_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_frozen_83_event_ranker_recall_guard_20260828_v1"


def event(start_day: int, rows: np.ndarray) -> EventSpan:
    start = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=start_day)
    return EventSpan(
        station="S-ORS",
        layer=2,
        start_time=start,
        end_time=start + pd.Timedelta(minutes=10 * (len(rows) - 1)),
        rows=rows,
    )


def test_chronological_split_enforces_forward_embargo() -> None:
    proposals = [event(day, np.arange(index * 3, index * 3 + 3)) for index, day in enumerate(range(0, 100, 10))]
    split = chronological_split(
        proposals,
        first_fraction=0.4,
        second_cumulative_fraction=0.7,
        purge_days=15,
    )
    assert len(split.train) == 4
    assert all(
        proposals[int(index)].start_time >= split.first_boundary + pd.Timedelta(days=15)
        for index in split.calibration
    )
    assert all(
        proposals[int(index)].start_time >= split.second_boundary + pd.Timedelta(days=15)
        for index in split.qualification
    )


def test_recall_threshold_uses_largest_qualifying_observed_score() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.2])
    matches = np.asarray(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [False, False, False],
        ]
    )
    threshold, record = select_recall_threshold(scores, matches, minimum_retention=0.66)
    assert threshold == 0.8
    assert record["retained_matched_truth_events"] == 2
    assert record["selected_proposals"] == 2


def test_feature_matrix_is_event_level_without_identifier_columns() -> None:
    keys = pd.DataFrame(
        {
            "station": ["S-ORS"] * 300,
            "year": [2024] * 300,
            "layer": [2] * 300,
            "time": pd.date_range("2024-01-01", periods=300, freq="10min", tz="UTC"),
        }
    )
    values = np.column_stack(
        [np.linspace(0.0, 1.0, 300), np.linspace(1.0, 0.0, 300)]
    )
    proposal = EventSpan(
        "S-ORS",
        2,
        pd.Timestamp(keys.iloc[140]["time"]),
        pd.Timestamp(keys.iloc[160]["time"]),
        np.arange(140, 161),
    )
    matrix, names, support = proposal_feature_matrix(
        keys,
        values,
        ["temp_abs_diff_1", "depth_diff_1"],
        [proposal],
        np.linspace(0.0, 1.0, 300),
        np.arange(0, 140),
        frozen_threshold=0.5,
        context_rows_each_side=12,
        minimum_normality_group_rows=10,
    )
    assert matrix.shape == (1, len(names))
    assert not any(name in {"station", "layer"} for name in names)
    assert support["primary_reference_coverage"] == 1.0


def test_mask_from_proposals_is_exact_zero_add_or_union() -> None:
    proposals = [event(0, np.asarray([1, 2, 3])), event(1, np.asarray([6, 7]))]
    zero = mask_from_proposals(10, proposals, [False, False])
    selected = mask_from_proposals(10, proposals, [True, False])
    assert np.array_equal(zero, np.zeros(10, dtype=np.int8))
    assert selected.sum() == 3


def test_prereg_and_check_only_seal_frozen_archive() -> None:
    config_path = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["frozen_generator"]["expected_calibration_proposals"] == 83
    assert config["frozen_generator"]["regeneration_allowed"] is False
    assert config["model"]["fit_count"] == 1
    assert config["model"]["cpu_threads"] <= 2
    assert config["threshold"]["selection_count"] == 1
    runner_path = ROOT / "scripts" / f"run_{EXPERIMENT_ID}.py"
    spec = importlib.util.spec_from_file_location("frozen_83_runner", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    result = runner.check_only()
    assert result["status"] == "READY_CHECK_ONLY"
    assert result["model_fit_count"] == 0
    assert result["frozen_archive"]["calibration_addition_rows"] == 10039
    assert result["q2_truth_rows_read"] == 0
