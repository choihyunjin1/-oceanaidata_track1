from __future__ import annotations

import inspect

import numpy as np

from scripts import run_p1_public_transport_repair_cycle_20260831_v30 as runner
from src.p1_qc.label_free_reliability_cap import (
    apply_label_free_day_cap,
    fit_label_free_group_reliability,
    reliability_margin_lower_bound,
)


def test_group_reliability_uses_scores_not_labels() -> None:
    for function in (
        fit_label_free_group_reliability,
        reliability_margin_lower_bound,
        apply_label_free_day_cap,
    ):
        assert not any("label" in name for name in inspect.signature(function).parameters)
    n = 1000
    station = np.where(np.arange(n) < 900, "A", "B")
    layer = np.ones(n, dtype=np.int8)
    source = np.full((n, 3), 0.8)
    calibrated = np.full(n, 0.805)
    calibrated[station == "B"] = 0.99
    receipts = fit_label_free_group_reliability(
        calibrated,
        source,
        station,
        layer,
        minimum_group_rows=64,
    )
    assert receipts["A|1"].eligible
    assert not receipts["B|1"].eligible


def test_margin_lower_bound_fail_closes_unknown_and_unreliable_groups() -> None:
    n = 900
    station = np.array(["A"] * n)
    layer = np.ones(n, dtype=np.int8)
    source = np.full((n, 3), 0.9)
    receipts = fit_label_free_group_reliability(
        np.full(n, 0.9),
        source,
        station,
        layer,
        minimum_group_rows=256,
    )
    outer_station = np.array(["A", "C"])
    outer_layer = np.ones(2, dtype=np.int8)
    lower = reliability_margin_lower_bound(
        np.array([0.95, 0.95]),
        0.8,
        np.full((2, 3), 0.9),
        outer_station,
        outer_layer,
        receipts,
    )
    assert lower[0] > 0
    assert np.isneginf(lower[1])


def test_day_cap_is_half_percent_and_stably_margin_ranked() -> None:
    n = 1000
    proposed = np.ones(n, dtype=bool)
    margin = np.arange(n, dtype=np.float64)
    day = np.zeros(n, dtype=np.int8)
    kept = apply_label_free_day_cap(proposed, margin, day)
    assert kept.sum() == 5
    assert np.flatnonzero(kept).tolist() == [995, 996, 997, 998, 999]
    ties = apply_label_free_day_cap(proposed, np.ones(n), day)
    assert np.flatnonzero(ties).tolist() == [0, 1, 2, 3, 4]


def test_contract_preserves_v28_and_is_not_v29_label_gate() -> None:
    config = runner.load_contract()
    assert config["model"]["C"] == 0.1
    assert config["em"]["maximum_iterations"] == 200
    assert config["label_free_reliability"]["outer_truth_or_failed_slice_inputs"] == 0
    assert config["authorization"]["historical_execution"] is True
    assert config["authorization"]["attempt_lock_creation"] is True


def test_synthetic_preflight_passes_with_zero_access() -> None:
    result = runner.preflight()
    assert result["status"] == "PASS"
    assert result["historical_model_fits_executed"] == 0
    assert all(value == 0 for value in result["access"].values())
    if runner.ARTIFACT.exists():
        assert (runner.ARTIFACT / "attempt_lock.json").is_file()
        assert (runner.ARTIFACT / "result.json").is_file()
