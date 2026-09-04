from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v20r1 as cycle  # noqa: E402

from src.p1_qc.robust_student_t_llr import derive_causal_gap_minutes  # noqa: E402


def test_repair_contract_preserves_scientific_parameters() -> None:
    contract = cycle.load_contract()
    assert contract["candidate"] == "P1_1_ROBUST_STUDENT_T_CLASS_LLR_ADDONLY"
    assert contract["model"]["degrees_of_freedom"] == 4.0
    assert contract["inner_calibration"]["fit_fraction"] == 0.75
    assert contract["fit_budget"]["maximum"] == 2


def test_gap_is_previous_key_distance_and_restores_order() -> None:
    frame = pd.DataFrame({"station": ["A", "A", "A", "B"], "layer": [1, 1, 1, 1], "year": [2025] * 4, "time": pd.to_datetime(["2025-01-01 00:20Z", "2025-01-01 00:00Z", "2025-01-01 00:10Z", "2025-01-01 00:05Z"])})
    np.testing.assert_array_equal(derive_causal_gap_minutes(frame), [10.0, 0.0, 10.0, 0.0])


def test_schema_preflight_is_schema_only() -> None:
    result = cycle.historical_schema_preflight()
    assert result["status"] == "PASS"
    assert result["historical_truth_reads"] == 0
    assert result["official_reads"] == 0
