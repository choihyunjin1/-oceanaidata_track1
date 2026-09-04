from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p2_prefix_safe_domain_balanced_deepset_20260901_v13.py"
SPEC = importlib.util.spec_from_file_location("p2_v13", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_contract_is_one_candidate_nine_fits() -> None:
    config = MODULE.load_config()
    assert config["operation_limits"]["maximum_candidate_count"] == 1
    assert config["training"]["maximum_fit_count"] == 9
    assert config["training"]["model_weight"] == 0.2
    assert config["training"]["row_deletion"] is False


def test_preflight_is_byte_identical() -> None:
    first = MODULE.preflight()
    second = MODULE.preflight()
    assert first == second
    assert first["preflight_sha256"] == second["preflight_sha256"]
    assert first["model_fits"] == 0
    assert first["official_rows_read"] == 0


def test_domain_balanced_groups_have_equal_mass() -> None:
    time = pd.to_datetime(
        [
            "2024-05-01T00:00:00+09:00",
            "2024-05-01T00:10:00+09:00",
            "2024-05-02T00:00:00+09:00",
            "2024-06-01T00:00:00+09:00",
            "2024-05-01T00:00:00+09:00",
            "2024-06-01T00:00:00+09:00",
        ],
        utc=True,
    ).tz_convert("Asia/Seoul")
    layer = np.asarray([2, 2, 2, 2, 3, 3])
    weights, receipt = MODULE.domain_balanced_weights(layer, time)
    assert np.isclose(weights.mean(), 1.0)
    masses = [value["raw_weight_sum"] for value in receipt["groups"].values()]
    assert np.allclose(masses, masses[0])


def test_safety_gate_requires_every_fold() -> None:
    record = {
        "by_fold": {
            "a": {"delta_rmse": -0.1},
            "b": {"delta_rmse": -0.1},
            "c": {"delta_rmse": 0.01},
        },
        "by_month": {"m": {"delta_rmse": -0.1}},
        "by_layer": {"2": {"delta_rmse": -0.1}},
        "bootstrap": {"ci90_high": -0.01},
    }
    assert MODULE.safety_gate(record, 0.2)["all_three_folds_non_harm"] is False


def test_v12_permutation_contract_is_retained() -> None:
    assert MODULE.v12.permutation_invariance_receipt()["maximum_abs_error"] <= 1e-6
