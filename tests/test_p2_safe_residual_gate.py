from __future__ import annotations

import numpy as np
import pandas as pd

from p2_restore.safe_residual_gate import apply_safe_calibrator, fit_safe_calibrator


def _frame() -> pd.DataFrame:
    rows = []
    for block, start in (("a", "2024-01-01"), ("b", "2024-02-01")):
        for day in range(8):
            for layer in (2, 3, 4):
                rows.append(
                    {
                        "time": pd.Timestamp(start, tz="Asia/Seoul") + pd.Timedelta(days=day),
                        "layer": layer,
                        "truth": 9.0,
                        "block": block,
                        "abs_t1_t5": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_beneficial_correction_is_bounded_and_enabled() -> None:
    frame = _frame()
    baseline = np.full(len(frame), 10.0)
    raw = np.full(len(frame), 8.0)
    calibrator = fit_safe_calibrator(frame, baseline, raw)
    prediction = apply_safe_calibrator(calibrator, frame, baseline, raw)
    assert np.allclose(prediction, 9.0)
    active = [cell for cell in calibrator.cells.values() if cell.alpha > 0]
    assert len(active) == 3
    assert all(np.isclose(cell.alpha, 0.5) for cell in active)


def test_disagreement_or_insufficient_support_falls_back_to_exact_noop() -> None:
    frame = _frame()
    baseline = np.full(len(frame), 10.0)
    raw = np.full(len(frame), 8.0)
    raw[frame["block"].eq("b").to_numpy()] = 12.0
    calibrator = fit_safe_calibrator(frame, baseline, raw)
    prediction = apply_safe_calibrator(calibrator, frame, baseline, raw)
    assert np.array_equal(prediction, baseline)


def test_unseen_public_state_uses_exact_noop() -> None:
    frame = _frame()
    baseline = np.full(len(frame), 10.0)
    raw = np.full(len(frame), 8.0)
    calibrator = fit_safe_calibrator(frame, baseline, raw)
    target = frame.iloc[:3].copy()
    target["abs_t1_t5"] = np.nan
    prediction = apply_safe_calibrator(calibrator, target, baseline[:3], raw[:3])
    assert np.array_equal(prediction, baseline[:3])
