from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_original_training_structure_20260830_v1.py"
)
SPEC = importlib.util.spec_from_file_location("original_training_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_p3_leads_match_public_contract() -> None:
    assert MODULE.P3_LEADS_H == (3, 6, 9, 12, 18, 24)
    assert MODULE.P3_DENSE_SPACING_MINUTES == 60
    assert MODULE.P3_DENSE_SPACING_MINUTES % MODULE.P3_WAVE_CADENCE_MINUTES == 0


def test_greedy_spacing_is_per_station_and_78_hours() -> None:
    frame = pd.DataFrame(
        {
            "station": ["A", "A", "A", "B", "B"],
            "time": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-04T05:00:00Z",
                    "2026-01-04T06:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-04T06:00:00Z",
                ],
                utc=True,
            ),
        }
    )

    selected = MODULE._greedy_78h(frame)

    assert selected.groupby("station").size().to_dict() == {"A": 2, "B": 1}
    assert selected.loc[selected["station"].eq("A"), "time"].tolist() == [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-04T06:00:00Z"),
    ]


def test_finite_summary_drops_nonfinite_values() -> None:
    result = MODULE.finite_summary([1.0, float("nan"), float("inf"), 3.0])
    assert result == {"count": 2, "q10": 1.2, "median": 2.0, "q90": 2.8}
