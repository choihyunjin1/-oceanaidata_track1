from __future__ import annotations

import pandas as pd
import pytest

from p2_restore.score_optimization import (
    TARGET_RELEVANT_BLOCKS,
    align_score_oof,
    leave_one_relevant_block_out,
    route_predictions,
    select_layer_router,
)


def _oof() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for block_number, block in enumerate(TARGET_RELEVANT_BLOCKS):
        for layer in (2, 3, 4):
            for offset in range(4):
                truth = float(block_number + layer + offset / 10)
                phase_error = 0.1 if layer in (2, 3) else 0.4
                state_error = 0.4 if layer in (2, 3) else 0.1
                rows.append(
                    {
                        "time": pd.Timestamp("2024-01-01", tz="UTC")
                        + pd.Timedelta(minutes=10 * len(rows)),
                        "layer": layer,
                        "truth": truth,
                        "current_blend50": truth + 0.5,
                        "phase_blend50": truth + phase_error,
                        "state_blend50": truth + state_error,
                        "block": block,
                    }
                )
    state = pd.DataFrame(rows)
    phase = state.drop(columns=["state_blend50", "block"]).copy()
    return phase, state


def test_alignment_and_router_selection() -> None:
    phase, state = _oof()
    aligned = align_score_oof(phase, state)
    selected = select_layer_router(aligned)["selected"]
    assert selected["layer_arms"] == {"2": "phase", "3": "phase", "4": "state"}
    assert selected["target_relevant_rmse"] == pytest.approx(0.1)


def test_leave_one_block_out_recovers_stable_router() -> None:
    aligned = align_score_oof(*_oof())
    result = leave_one_relevant_block_out(aligned)
    assert result["rmse"] == pytest.approx(0.1)
    assert result["rmse"] < result["phase_rmse"]
    assert result["rmse"] < result["state_rmse"]


def test_router_rejects_invalid_layer_contract() -> None:
    aligned = align_score_oof(*_oof())
    with pytest.raises(ValueError, match="layers 2, 3, and 4"):
        route_predictions(aligned, {2: "phase", 3: "phase"})


def test_alignment_fails_on_truth_mismatch() -> None:
    phase, state = _oof()
    phase.loc[0, "truth"] += 1
    with pytest.raises(ValueError, match="truth differs"):
        align_score_oof(phase, state)
