from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_extreme_wave_ratio_residual_cycle_20260901_v81.py"
SPEC = importlib.util.spec_from_file_location("p3_v81", RUNNER)
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


def test_extreme_ratio_regime_and_scale_guards() -> None:
    receipt = MODULE.synthetic_receipt()
    assert receipt["late_extreme_ratio_variance_q90_endpoint_slope_increased"] is True
    assert receipt["common_scale_invariant"] is True
    assert receipt["calm_path_zero"] is True


def test_exact_ratio_and_constant_geometry() -> None:
    hs = np.full(145, 2.0)
    stats = MODULE.ratio_statistics(hs, 1.6 * hs)
    assert np.isclose(stats[0], 1.6)
    assert np.isclose(stats[2], 1.6)
    assert np.allclose(stats[[1, 3, 6, 7]], 0.0, atol=1e-12, rtol=0.0)


def test_feature_shape_determinism_and_future_isolation() -> None:
    sequence = np.zeros((289, 10), dtype=np.float64)
    axis = np.linspace(0.0, 8.0, 289)
    sequence[:, MODULE.HS_COLUMN] = 2.0 + 0.25 * np.sin(axis)
    sequence[:, MODULE.HMAX_COLUMN] = 1.6 * sequence[:, MODULE.HS_COLUMN]
    sequence[1::7, (MODULE.HS_COLUMN, MODULE.HMAX_COLUMN)] = np.nan
    direct = MODULE.ratio_features(sequence)
    assert direct.shape == (16,) and np.isfinite(direct).all()
    assert np.array_equal(direct, MODULE.ratio_features(sequence.copy()))
    assert np.array_equal(direct, MODULE.ratio_features(np.vstack([sequence, np.full((12, 10), 1e9)])))


def test_sealed_contract_and_novelty_boundary() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["raw_column_indices"] == [0, 2]
    assert config["encoder"]["feature_count"] == 16
    assert config["validation"]["maximum_total_fits"] == 12
    assert "v21" in config["duplication_audit"]["distinction"]
    assert all(value == 0 for value in config["official_policy"].values())
