from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PATH = Path(__file__).resolve().parents[1] / "scripts/run_p3_public_transport_robust_retrain_cycle_20260831_v10.py"
SPEC = importlib.util.spec_from_file_location("p3_v10_runner", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_candidate_contract_is_three_distinct_robust_families() -> None:
    assert len(MODULE.SPECS) == 3
    assert len({item.name for item in MODULE.SPECS}) == 3
    assert {item.family for item in MODULE.SPECS} == {"huber", "hist_gbdt", "ridge"}


def test_train_only_weights_downweight_without_row_deletion() -> None:
    frame = pd.DataFrame(
        {
            "target_hs": [0.0, 0.0, 0.0, 10.0],
            "reference": [0.0, 0.0, 0.0, 0.0],
        }
    )
    weights, receipt = MODULE.train_only_weights(frame)
    assert weights.shape == (4,)
    assert np.isfinite(weights).all()
    assert (weights > 0).all()
    assert receipt["validation_rows_deleted"] == 0
    assert receipt["downweighted_rows"] >= 1


def test_prior_audit_excludes_official_hidden_submission_paths() -> None:
    audit = MODULE.audit_prior_results()
    assert audit["excluded_official_hidden_submission_paths"]
    assert audit["files_read"] > 0


def test_transport_gate_constants_match_authoritative_calibration() -> None:
    assert MODULE.load_penalty() == 0.3219056897594759
    assert MODULE.MIN_CALIBRATED_POINTS == 0.01
    assert MODULE.MAX_STATION_LEAD_REGRESSION_M == 0.01
