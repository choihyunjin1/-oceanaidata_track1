from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

import scripts.run_p3_corrected_fixed_long_shrink_v4 as runner
from p3_wave.corrected_fixed_long_shrink import FixedLongLeadShrinkCalibrator


def test_fixed_calibrator_is_short_lead_no_op_and_long_lead_quarter_shrink() -> None:
    routed = np.array([2.0] * 6)
    persistence = np.array([1.0] * 6)
    leads = np.array([3, 6, 9, 12, 18, 24])
    result = FixedLongLeadShrinkCalibrator().predict(routed, persistence, leads)
    np.testing.assert_array_equal(result[:3], routed[:3])
    np.testing.assert_allclose(result[3:], 1.75, rtol=0.0, atol=0.0)


def test_calibrator_rejects_nonsealed_parameters() -> None:
    try:
        FixedLongLeadShrinkCalibrator(persistence_weight=0.2)
    except ValueError as error:
        assert "exactly 0.25" in str(error)
    else:
        raise AssertionError("nonsealed calibrator weight was accepted")


def test_runner_has_no_config_output_or_coefficient_override() -> None:
    assert set(inspect.signature(runner.run_experiment).parameters) == {"root", "data_dir"}
    source = inspect.getsource(runner.run_experiment)
    assert "evaluate_identical_oof" in source
    assert source.index('if not evaluation["gate"]["passed"]') < source.index("_infer_candidate")


def test_identical_oof_winner_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    oof, evaluation = runner.evaluate_identical_oof(root)
    assert len(oof) == 1086
    assert evaluation["gate"]["passed"] is True
    assert evaluation["candidate"]["rmse_m"] < runner.INCUMBENT_RMSE
    assert evaluation["strictly_improved_fold_count"] >= 2
    assert max(evaluation["station_delta_m"].values()) <= 0.01
    assert evaluation["lead_18_24_combined_delta_m"] <= 0.01
