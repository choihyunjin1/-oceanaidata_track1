from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v13 as base  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v13r1 as recovery  # noqa: E402


def test_utc_dtype_normalization_allows_truth_attachment(monkeypatch) -> None:
    historical = pd.DataFrame(
        {
            "station": ["S"],
            "year": [2025],
            "layer": [1],
            "time": ["2025-01-01T00:00:00Z"],
            "fold": ["2025_q3"],
            "label_base": [1],
        }
    )
    anchor = pd.DataFrame(
        {
            "station": ["S"],
            "year": [2025],
            "layer": [1],
            "time": pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True),
            "fold": ["2025_q3"],
            "current_router_prediction": [0],
        }
    )
    monkeypatch.setattr(base.base.source_cycle, "p1_frame", lambda: (historical, None))
    attached, candidate = base.attach_truth(anchor, np.array([1], dtype=np.int8))
    assert len(attached) == 1
    assert candidate.tolist() == [1]
    assert str(attached["time"].dtype).endswith("UTC]")


def test_reused_proposal_hashes_are_bit_exact() -> None:
    _anchor, additions, candidate, seal = recovery.load_reused_proposal()
    assert base.sha256_bool(additions) == seal["additions_sha256"]
    assert base.sha256_bool(candidate) == seal["candidate_sha256"]
    assert base.sha256_file(recovery.SOURCE_PROPOSAL) == seal["npz_sha256"]


def test_recovery_contract_does_not_change_candidate_or_gate() -> None:
    config = base.load_contract()
    assert config["candidate"]["distinct_other_layer_quorum"] == 2
    assert config["candidate"]["lookback_minutes_inclusive"] == 10
    assert config["fit_budget"]["maximum"] == 0
    assert np.isclose(
        config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"],
        0.015383691373120248,
    )
