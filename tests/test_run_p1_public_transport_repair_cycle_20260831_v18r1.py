from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v18r1 as recovery  # noqa: E402


def test_us_datetime_series_becomes_distinct_ten_minute_values() -> None:
    values = pd.Series(pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:10:00Z"]))
    assert str(values.dtype) == "datetime64[us, UTC]"
    minutes = recovery.fixed_utc_minutes(values)
    np.testing.assert_array_equal(np.diff(minutes), np.asarray([10]))


def test_recovery_preserves_base_method_hash_and_zero_change_counts() -> None:
    contract = recovery.load_recovery_contract()
    assert contract["base_method_config_sha256"] == recovery.original.sha256(recovery.BASE_CONFIG)
    assert contract["method_changes"] == 0
    assert contract["feature_changes"] == 0
    assert contract["model_changes"] == 0
    assert contract["threshold_changes"] == 0
    assert contract["gate_changes"] == 0
    assert contract["original_fit_count"] == 0


def test_validate_only_keeps_new_artifact_absent() -> None:
    before = recovery.ARTIFACT.exists()
    payload = recovery.validate_only()
    assert payload["status"] == "VALID"
    assert payload["datetime_resolution_regression"] == "PASS"
    assert recovery.ARTIFACT.exists() is before
