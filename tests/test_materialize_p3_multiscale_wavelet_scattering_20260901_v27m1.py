from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/materialize_p3_multiscale_wavelet_scattering_20260901_v27m1.py"
SPEC = importlib.util.spec_from_file_location("p3_v27m1", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_zero_official_preflight_or_consumed() -> None:
    if MODULE.ARTIFACT.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight()
    else:
        value = MODULE.preflight()
        assert value["status"] == "ZERO_OFFICIAL_ROW_PREFLIGHT_PASS"
        assert value["official_rows_read"] == 0
        assert value["submission_csv_created"] == 0


def test_candidate_is_exactly_frozen() -> None:
    config = MODULE.load_config()
    assert config["candidate"]["ridge_alpha"] == 1024.0
    assert config["candidate"]["additive_residual_weight"] == 0.10
    assert config["candidate"]["row_deletion"] == 0
    assert not config["operation_limits"]["posthoc_routing"]


def test_guard_accepts_small_safe_action() -> None:
    rows = 18
    frame = pd.DataFrame({"station": np.repeat(["G-ORS", "I-ORS", "S-ORS"], 6), "lead_h": np.tile([3, 6, 9, 12, 18, 24], 3)})
    champion = np.full(rows, 2.0)
    candidate = champion + 0.01
    _, checks = MODULE.geometry(frame, candidate, champion, MODULE.load_config())
    assert not checks["rows_exact"]
    assert all(value for name, value in checks.items() if name != "rows_exact")


def test_guard_rejects_large_action() -> None:
    frame = pd.DataFrame({"station": np.repeat(["G-ORS", "I-ORS", "S-ORS"], 6), "lead_h": np.tile([3, 6, 9, 12, 18, 24], 3)})
    champion = np.full(18, 2.0)
    candidate = champion + 0.25
    _, checks = MODULE.geometry(frame, candidate, champion, MODULE.load_config())
    assert not checks["action_max"]


def test_internal_hash_contract() -> None:
    result, qa = MODULE.verify_internal(MODULE.load_config())
    assert result["decision"] == "PASS_CANDIDATE_AVAILABLE"
    assert qa["decision"] == "PASS"
