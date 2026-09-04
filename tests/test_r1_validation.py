from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p1_qc.r1_validation import (
    IntervalCandidate,
    LeakageError,
    SelectionProvenance,
    apply_interval_candidate,
    apply_selected_interval_grid,
    candidate_output_to_mask,
    intervals_to_mask,
    select_interval_grid,
)


def _frame(group_sizes: tuple[int, ...] = (8, 2)) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for group_number, size in enumerate(group_sizes):
        parts.append(
            pd.DataFrame(
                {
                    "station": [chr(ord("A") + group_number)] * size,
                    "layer": [1] * size,
                    "time": pd.date_range(
                        "2025-01-01", periods=size, freq="10min", tz="Asia/Seoul"
                    ).astype(str),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _provenance(rows: int, *, scope: str = "inner_validation") -> SelectionProvenance:
    return SelectionProvenance(
        fit_rows=np.arange(100, 110),
        inner_validation_rows=np.arange(200, 200 + rows),
        outer_validation_rows=np.arange(300, 305),
        label_scope=scope,
        generator_id="unit-test-generator",
    )


def test_intervals_project_to_rows_and_reject_group_or_gap_crossing() -> None:
    frame = _frame((3, 3))
    mask = intervals_to_mask(frame, [IntervalCandidate(0, 2)])
    assert mask.tolist() == [True, True, False, False, False, False]
    with pytest.raises(ValueError, match="crosses a group or time gap"):
        intervals_to_mask(frame, [(2, 4)])

    gap_frame = _frame((4,))
    gap_frame.loc[3, "time"] = "2025-01-01T01:00:00+09:00"
    with pytest.raises(ValueError, match="crosses a group or time gap"):
        intervals_to_mask(gap_frame, [(1, 4)])


def test_numpy_interval_contract_and_bounds_are_checked() -> None:
    frame = _frame((5,))
    mask = candidate_output_to_mask(frame, np.asarray([[1, 3], [3, 5]]))
    assert mask.tolist() == [False, True, True, True, True]
    with pytest.raises(IndexError):
        intervals_to_mask(frame, [(0, 6)])
    with pytest.raises(ValueError, match="length 5"):
        candidate_output_to_mask(frame, np.zeros(4, dtype=bool))
    with pytest.raises(ValueError, match="finite integer"):
        candidate_output_to_mask(frame, np.asarray([[0.5, 2.0]]))


def test_apply_unions_base_and_preserves_spike_and_plateau_rows() -> None:
    frame = _frame((8,))
    base = np.zeros(8, dtype=np.int8)
    base[6] = 1
    spike = np.zeros(8, dtype=bool)
    spike[0] = True
    plateau = np.zeros(8, dtype=bool)
    plateau[7] = True
    prediction = apply_interval_candidate(
        frame,
        base,
        [(2, 5)],
        spike_protected=spike,
        plateau_protected=plateau,
    )
    assert prediction.tolist() == [1, 0, 1, 1, 1, 0, 1, 1]


def test_provenance_overlap_is_blocked_before_candidate_callback() -> None:
    frame = _frame((4,))
    called = False

    def factory(_: dict[str, object]) -> np.ndarray:
        nonlocal called
        called = True
        return np.zeros(len(frame), dtype=bool)

    provenance = SelectionProvenance(
        fit_rows=[1, 2, 3],
        inner_validation_rows=[3, 4, 5, 6],
        label_scope="inner_validation",
    )
    with pytest.raises(LeakageError, match="overlap"):
        select_interval_grid(
            frame,
            np.zeros(len(frame)),
            np.zeros(len(frame)),
            [{}],
            factory,
            provenance=provenance,
        )
    assert not called


@pytest.mark.parametrize("scope", ["outer_validation", "test", "unknown"])
def test_outer_or_unknown_labels_fail_closed_before_callback(scope: str) -> None:
    frame = _frame((4,))
    called = False

    def factory(_: dict[str, object]) -> np.ndarray:
        nonlocal called
        called = True
        return np.zeros(len(frame), dtype=bool)

    with pytest.raises(LeakageError, match="inner_validation"):
        select_interval_grid(
            frame,
            [9, 9, 9, 9],  # label parsing must never be reached
            np.zeros(len(frame)),
            [{}],
            factory,
            provenance=_provenance(len(frame), scope=scope),
        )
    assert not called


def test_micro_and_test_share_weighted_f1_can_select_different_candidates() -> None:
    frame = _frame((8, 2))
    truth = np.asarray([1, 1, 1, 1, 0, 0, 0, 0, 1, 1], dtype=np.int8)
    masks = {
        "large_group": np.asarray([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool),
        "test_heavy_group": np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=bool),
    }

    def factory(parameters: dict[str, str]) -> np.ndarray:
        return masks[parameters["candidate"]]

    grid = [{"candidate": "test_heavy_group"}, {"candidate": "large_group"}]
    kwargs = {
        "frame": frame,
        "truth": truth,
        "base_prediction": np.zeros(len(frame)),
        "parameter_grid": grid,
        "candidate_factory": factory,
        "provenance": _provenance(len(frame)),
        "group_weights": {("A", 1): 0.1, ("B", 1): 0.9},
    }
    micro = select_interval_grid(**kwargs, primary_metric="micro_f1")
    weighted = select_interval_grid(**kwargs, primary_metric="weighted_f1")
    assert micro.parameters == {"candidate": "large_group"}
    assert weighted.parameters == {"candidate": "test_heavy_group"}
    assert weighted.diagnostics["selected"]["weighted"]["f1"] > 0.9


def test_precision_breaks_exact_f1_tie_and_grid_order_is_irrelevant() -> None:
    frame = _frame((6,))
    truth = np.asarray([1, 1, 1, 1, 0, 0], dtype=np.int8)
    masks = {
        "high_precision": np.asarray([1, 1, 0, 0, 0, 0], dtype=bool),
        "low_precision": np.asarray([1, 1, 1, 0, 1, 1], dtype=bool),
    }

    def factory(parameters: dict[str, str]) -> np.ndarray:
        return masks[parameters["candidate"]]

    forward = [{"candidate": "high_precision"}, {"candidate": "low_precision"}]
    reverse = list(reversed(forward))
    first = select_interval_grid(
        frame,
        truth,
        np.zeros(len(frame)),
        forward,
        factory,
        provenance=_provenance(len(frame)),
    )
    second = select_interval_grid(
        frame,
        truth,
        np.zeros(len(frame)),
        reverse,
        factory,
        provenance=_provenance(len(frame)),
    )
    assert first.parameters == {"candidate": "high_precision"}
    assert second.parameters == first.parameters
    assert first.diagnostics == second.diagnostics
    assert np.array_equal(first.prediction, second.prediction)


def test_protected_masks_are_preserved_during_selection_and_outer_apply() -> None:
    inner = _frame((6,))
    truth = np.asarray([1, 0, 1, 0, 0, 1], dtype=np.int8)
    spike = np.asarray([1, 0, 0, 0, 0, 0], dtype=bool)
    plateau = np.asarray([0, 0, 0, 0, 0, 1], dtype=bool)

    def factory(parameters: dict[str, int]) -> np.ndarray:
        mask = np.zeros(6, dtype=bool)
        mask[parameters["row"]] = True
        return mask

    result = select_interval_grid(
        inner,
        truth,
        np.zeros(6),
        [{"row": 1}, {"row": 2}],
        factory,
        provenance=_provenance(len(inner)),
        spike_protected=spike,
        plateau_protected=plateau,
    )
    assert result.parameters == {"row": 2}
    assert result.prediction[[0, 5]].tolist() == [1, 1]

    outer = _frame((6,))
    applied = apply_selected_interval_grid(
        outer,
        np.zeros(6),
        result,
        factory,
        spike_protected=spike,
        plateau_protected=plateau,
    )
    assert applied.tolist() == [1, 0, 1, 0, 0, 1]


def test_grid_must_be_finite_nonempty_and_have_finite_parameters() -> None:
    frame = _frame((3,))

    def factory(_: dict[str, object]) -> np.ndarray:
        return np.zeros(len(frame), dtype=bool)

    common = (frame, np.zeros(3), np.zeros(3))
    with pytest.raises(TypeError, match="finite sequence"):
        select_interval_grid(
            *common,
            ({"x": value} for value in range(3)),
            factory,
            provenance=_provenance(len(frame)),
        )
    with pytest.raises(ValueError, match="cannot be empty"):
        select_interval_grid(
            *common,
            [],
            factory,
            provenance=_provenance(len(frame)),
        )
    with pytest.raises(ValueError, match="must be finite"):
        select_interval_grid(
            *common,
            [{"x": float("nan")}],
            factory,
            provenance=_provenance(len(frame)),
        )
