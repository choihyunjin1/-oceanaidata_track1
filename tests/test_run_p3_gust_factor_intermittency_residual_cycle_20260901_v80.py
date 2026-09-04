from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_gust_factor_intermittency_residual_cycle_20260901_v80.py"
SPEC = importlib.util.spec_from_file_location("p3_v80", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preflight_or_consumed() -> None:
    if MODULE.ARTIFACT.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight_payload()
    else:
        value = MODULE.preflight_payload()
        assert value["status"] in {"READY_EXACTLY_ONCE", "STOP_SUPPORT_GATE_ZERO_FIT"}
        assert value["prior_outputs_used"] is False
        assert value["official_used_for_features_gates_selection"] is False


def test_gust_burst_and_scale_guards() -> None:
    receipt = MODULE.synthetic_receipt()
    assert receipt["burst_max_increased"] is True
    assert receipt["burst_mean_excess_increased"] is True
    assert receipt["burst_exceedance_share_increased"] is True
    assert receipt["common_scale_invariant"] is True


def test_calm_and_lag_correlation_bounds() -> None:
    assert np.array_equal(
        MODULE.gust_window_statistics(np.zeros(145), np.zeros(145)), np.zeros(8)
    )
    alternating = np.tile([0.0, 1.0], 73)[:145]
    value = MODULE.lag1_correlation(alternating)
    assert -1.0 <= value <= 1.0


def test_feature_shape_determinism_and_future_isolation() -> None:
    sequence = np.zeros((289, 10), dtype=np.float64)
    axis = np.linspace(0.0, 8.0, 289)
    sequence[:, MODULE.WIND_COLUMN] = 6.0 + 0.5 * np.sin(axis)
    sequence[:, MODULE.GUST_COLUMN] = sequence[:, MODULE.WIND_COLUMN] + 1.0
    sequence[1::7, (MODULE.WIND_COLUMN, MODULE.GUST_COLUMN)] = np.nan
    direct = MODULE.gust_features(sequence)
    assert direct.shape == (16,) and np.isfinite(direct).all()
    assert np.array_equal(direct, MODULE.gust_features(sequence.copy()))
    assert np.array_equal(
        direct, MODULE.gust_features(np.vstack([sequence, np.full((12, 10), 1e9)]))
    )


def test_sealed_contract_and_novelty_boundary() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["raw_column_indices"] == [4, 5]
    assert config["encoder"]["feature_count"] == 16
    assert config["validation"]["maximum_total_fits"] == 12
    assert "v20" in config["duplication_audit"]["distinction"]
    assert all(value == 0 for value in config["official_policy"].values())
