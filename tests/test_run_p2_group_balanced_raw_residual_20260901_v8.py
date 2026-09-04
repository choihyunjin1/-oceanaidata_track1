from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p2_group_balanced_raw_residual_20260901_v8.py"
SPEC = importlib.util.spec_from_file_location("p2_v8", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_group_balanced_weights_equalize_six_groups() -> None:
    layers = []
    windows = []
    dates = []
    for layer in (2, 3, 4):
        for window in ("a", "b"):
            for day, count in (("d1", 2), ("d2", 5)):
                layers.extend([layer] * count)
                windows.extend([window] * count)
                dates.extend([day] * count)
    weights, receipt = MODULE.group_balanced_weights(
        np.asarray(layers), np.asarray(windows), np.asarray(dates)
    )
    assert np.isclose(weights.mean(), 1.0)
    assert np.isclose(receipt["raw_weight_sum"], 1.0)
    assert all(
        np.isclose(item["raw_weight_sum"], 1.0 / 6.0)
        for item in receipt["groups"].values()
    )


def test_strict_gate_requires_transport_and_all_stability_checks() -> None:
    by_fold = {
        "2024_sep_oct": {"delta_rmse": -0.02},
        "2025_jul_aug": {"delta_rmse": -0.01},
        "2025_nov_dec": {"delta_rmse": 0.001},
    }
    by_layer = {str(layer): {"delta_rmse": 0.002} for layer in (2, 3, 4)}
    checks = MODULE.strict_gate(
        pooled_delta=-0.01,
        by_fold=by_fold,
        by_layer=by_layer,
        bootstrap_ci90_high=-0.001,
        calibrated_expected_points=0.02,
    )
    assert all(checks.values())
    checks["calibrated_expected_points_gte_0_01"] = False
    assert not all(checks.values())


def test_runner_has_no_materializer_or_official_input_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8").lower()
    assert "to_csv(" not in text
    assert "upload" in text
    assert "official_test_index_rows_read\": 0" in text
