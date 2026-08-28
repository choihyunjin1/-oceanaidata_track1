from __future__ import annotations

from p1_qc.low_fidelity_gate import evaluate_low_fidelity_screen


def _result(q3: float, q4: float, pooled: float) -> dict:
    by_station = {
        "G-ORS": {"candidate_added_rows": 2},
        "I-ORS": {"candidate_added_rows": 3},
        "S-ORS": {"candidate_added_rows": 5},
    }
    return {
        "metrics": {
            "q3": {"delta_f1": q3, "by_station": by_station},
            "q4": {"delta_f1": q4, "by_station": by_station},
        },
        "pooled": {"delta_f1": pooled},
        "official_test_rows_read": 0,
        "submission_created": False,
        "upload_performed": False,
    }


def test_both_windows_and_pooled_must_improve() -> None:
    assert evaluate_low_fidelity_screen(_result(0.01, 0.02, 0.015))["decision"] == "PROMOTE_TO_FULL_FIDELITY"
    stopped = evaluate_low_fidelity_screen(_result(-0.001, 0.02, 0.001))
    assert stopped["decision"] == "STOP_BEFORE_FULL_FIDELITY"
    assert "q3_delta_f1_gt_0" in stopped["failed_or_unavailable_checks"]


def test_missing_station_counts_fails_closed() -> None:
    result = _result(0.01, 0.02, 0.015)
    del result["metrics"]["q4"]["by_station"]["S-ORS"]["candidate_added_rows"]
    stopped = evaluate_low_fidelity_screen(result)
    assert stopped["decision"] == "STOP_BEFORE_FULL_FIDELITY"
    assert stopped["checks"]["maximum_station_addition_share_lt_0_8"] is None
