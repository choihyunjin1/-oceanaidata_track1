from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_station_graph_laplacian_residual_cycle_20260901_v24.py"
SPEC = importlib.util.spec_from_file_location("p3_v24", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def synthetic_sequence() -> np.ndarray:
    base = np.linspace(0.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    sequence[2::11, (0, 3, 6)] = np.nan
    return sequence


def test_config_is_sealed_and_official_zero() -> None:
    config = MODULE.load_config()
    assert config["status"] == "SEALED_BEFORE_OUTER_SCORING"
    assert all(value == 0 for value in config["official_policy"].values())
    assert config["validation"]["maximum_total_fits"] == 12


def test_trajectory_features_are_fixed_and_deterministic() -> None:
    first = MODULE.trajectory_features(synthetic_sequence())
    second = MODULE.trajectory_features(synthetic_sequence())
    assert first.shape == (104,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_graph_laplacian_is_symmetric_psd_and_connected() -> None:
    laplacian, receipt = MODULE.station_laplacian(MODULE.load_config())
    assert np.allclose(laplacian, laplacian.T)
    values = np.linalg.eigvalsh(laplacian)
    assert values[0] >= -1e-10
    assert values[1] > 0
    assert len(receipt["distances_km"]) == 3


def test_duplicate_axes_are_explicitly_stopped() -> None:
    audit = MODULE.load_config()["duplication_audit"]
    assert audit["concurrent_cross_station_propagation"] == "STOP_NO_DEPLOYMENT_CONTRACT"
    assert audit["directional_wind_sea_wave_age"] == "STOP_SEMANTIC_DUPLICATE"
    assert audit["semantic_verdict"] == "NON_DUPLICATE_ARCHITECTURE_AXIS"


def test_zero_op_preflight_is_access_zero_and_bounded() -> None:
    if MODULE.ARTIFACT_DIR.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="already consumed"):
            MODULE.preflight_payload()
    else:
        payload = MODULE.preflight_payload()
        assert payload["status"] == "READY_EXACTLY_ONCE"
        assert payload["maximum_model_fits"] == 12
        assert payload["official_access"] == 0
        assert payload["csv_materializations"] == 0
        assert payload["uploads"] == 0
