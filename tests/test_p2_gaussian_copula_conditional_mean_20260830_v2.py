from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p2_gaussian_copula_conditional_mean_20260830_v2.py"
SPEC = importlib.util.spec_from_file_location("p2_copula_v2", RUNNER_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_repair_uses_exact_noop_for_incomplete_profile_time() -> None:
    times = pd.to_datetime(["2024-09-01T00:00:00Z", "2024-09-02T00:00:00Z"])
    query = pd.DataFrame(
        {
            "time": [times[0], times[0], times[0], times[1], times[1]],
            "layer": [2, 3, 4, 2, 4],
        }
    )
    prediction = np.asarray([[0.1, 0.2, 0.3]])
    correction = RUNNER.repaired_row_correction(query, times[:1], prediction)
    assert np.allclose(correction[:3], prediction[0])
    assert np.allclose(correction[3:], 0.0)


def test_repair_overlay_pins_v1_and_changes_no_model_setting() -> None:
    overlay = json.loads(RUNNER.CONFIG.read_text(encoding="utf-8"))
    assert overlay["classification"] == "COMPLETION_ONLY_PROFILE_SUPPORT_CONTRACT_REPAIR"
    assert overlay["repair_contract"]["model_hyperparameters_changed"] is False
    assert overlay["repair_contract"]["shrinkage_search_changed"] is False
    assert overlay["repair_contract"]["outer_folds_changed"] is False
    assert overlay["repair_contract"]["result_based_tuning"] is False


def test_repair_forbids_official_access_and_retry() -> None:
    policy = json.loads(RUNNER.CONFIG.read_text(encoding="utf-8"))["execution_policy"]
    assert policy["maximum_executions"] == 1
    assert not any(value for key, value in policy.items() if key != "maximum_executions")
