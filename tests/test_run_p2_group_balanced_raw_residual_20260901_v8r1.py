from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p2_group_balanced_raw_residual_20260901_v8r1.py"
SPEC = importlib.util.spec_from_file_location("p2_v8r1", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_mixed_datetime_units_canonicalize_to_same_ns_key() -> None:
    us = pd.DatetimeIndex(["2025-07-01T00:00:00Z"]).as_unit("us")
    ns = pd.DatetimeIndex(["2025-07-01T00:00:00Z"]).as_unit("ns")
    assert MODULE.engine.canonical_time_ns(us).tolist() == MODULE.engine.canonical_time_ns(
        ns
    ).tolist()


def test_zero_operation_preflight_is_deterministic() -> None:
    first = MODULE.preflight()
    second = MODULE.preflight()
    assert first == second
    assert first["data_rows_read"] == 0
    assert first["model_fits"] == 0
    assert first["artifacts_written"] == 0
    assert first["official_rows_read"] == 0


def test_repair_keeps_v8_candidate_order_and_seeds() -> None:
    result = MODULE.preflight()
    assert result["candidate_names"] == [
        "P2_V8_GROUP_BALANCED_L2_RAW_RESIDUAL",
        "P2_V8_GROUP_BALANCED_L1_RAW_RESIDUAL",
    ]
    assert result["seeds"] == [20260823, 20260824, 20260825]
