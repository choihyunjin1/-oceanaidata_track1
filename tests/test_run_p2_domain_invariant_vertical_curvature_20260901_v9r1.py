from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT / "scripts" / "run_p2_domain_invariant_vertical_curvature_20260901_v9r1.py"
)
CONFIG_PATH = (
    ROOT / "configs" / "experiments" / "p2_domain_invariant_vertical_curvature_20260901_v9r1.json"
)
SPEC = importlib.util.spec_from_file_location("p2_v9r1_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_repair_contract_preserves_science_and_has_exact_allow_list() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    repair = config["contract_repair"]
    assert config["experiment_id"] == runner.EXPERIMENT_ID
    assert repair["predecessor_status"].startswith("INVALID_ZERO_FIT")
    assert set(repair["all_missing_columns"]) == runner.EXPECTED_ALL_MISSING
    assert repair["candidate_feature_split_alpha_blend_gate_winsor_changed"] is False
    assert config["training"]["ridge_alpha"] == 25.0
    assert config["training"]["champion_preserving_weight"] == 0.8
    assert config["training"]["model_weight"] == 0.2
    assert config["training"]["target_winsor_mad_multiplier"] == 4.0
    assert config["operation_limits"]["maximum_fit_count"] == 6


def test_all_missing_allow_list_becomes_zero() -> None:
    frame = pd.DataFrame(
        {
            "temp_offset_l8": [np.nan, np.nan],
            "psal_anomaly_l8": [np.nan, np.nan],
            "depth_offset_l8": [np.nan, np.nan],
            "nominal_offset_l8": [np.nan, np.nan],
            "ordinary": [1.0, 3.0],
        }
    )
    median = runner.deterministic_finite_column_median(frame)
    assert median.loc[list(runner.EXPECTED_ALL_MISSING)].eq(0.0).all()
    assert median["ordinary"] == 2.0


def test_unexpected_all_missing_column_fails_closed() -> None:
    frame = pd.DataFrame({"unexpected": [np.nan, np.nan]})
    try:
        runner.deterministic_finite_column_median(frame)
    except runner.engine.ContractError as error:
        assert "unexpected all-missing" in str(error)
    else:  # pragma: no cover
        raise AssertionError("unexpected all-missing column did not fail closed")


def test_preflight_is_byte_stable_and_zero_operation() -> None:
    first = runner.preflight()
    second = runner.preflight()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["status"] == "ZERO_OPERATION_PREFLIGHT_PASS"
    assert first["data_rows_read"] == 0
    assert first["model_fits"] == 0
    assert first["official_rows_read"] == 0
    assert first["submission_csv_created"] == 0
    assert first["uploads"] == 0
    assert first["contract_repair"]["scientific_contract_changed"] is False
