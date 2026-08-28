"""Fail-closed promotion gate for staged P1 low-fidelity screens."""

from __future__ import annotations

from typing import Any


def evaluate_low_fidelity_screen(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    q3_delta = float(metrics["q3"]["delta_f1"])
    q4_delta = float(metrics["q4"]["delta_f1"])
    pooled_delta = float(result["pooled"]["delta_f1"])
    station_additions: dict[str, int] = {}
    station_counts_available = True
    for phase in ("q3", "q4"):
        for station, values in metrics[phase]["by_station"].items():
            if "candidate_added_rows" not in values:
                station_counts_available = False
                continue
            station_additions[station] = station_additions.get(station, 0) + int(
                values["candidate_added_rows"]
            )
    total_additions = sum(station_additions.values())
    maximum_share = (
        max(station_additions.values()) / total_additions
        if station_counts_available and total_additions
        else None
    )
    checks: dict[str, bool | None] = {
        "q3_delta_f1_gt_0": q3_delta > 0.0,
        "q4_delta_f1_gt_0": q4_delta > 0.0,
        "pooled_delta_f1_gt_0": pooled_delta > 0.0,
        "maximum_station_addition_share_lt_0_8": (
            maximum_share < 0.8 if maximum_share is not None else None
        ),
        "official_test_rows_read_eq_0": int(result["official_test_rows_read"]) == 0,
        "submission_not_created": result["submission_created"] is False,
        "upload_not_performed": result["upload_performed"] is False,
    }
    required = (
        "q3_delta_f1_gt_0",
        "q4_delta_f1_gt_0",
        "pooled_delta_f1_gt_0",
        "maximum_station_addition_share_lt_0_8",
        "official_test_rows_read_eq_0",
        "submission_not_created",
        "upload_not_performed",
    )
    passed = all(checks[name] is True for name in required)
    return {
        "decision": "PROMOTE_TO_FULL_FIDELITY" if passed else "STOP_BEFORE_FULL_FIDELITY",
        "checks": checks,
        "metrics": {
            "q3_delta_f1": q3_delta,
            "q4_delta_f1": q4_delta,
            "pooled_delta_f1": pooled_delta,
            "station_additions": station_additions,
            "maximum_station_addition_share": maximum_share,
        },
        "failed_or_unavailable_checks": [
            name for name in required if checks[name] is not True
        ],
    }


__all__ = ["evaluate_low_fidelity_screen"]
