from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_fixed_simplex_stacking_cycle_20260901_v25.py"
SPEC = importlib.util.spec_from_file_location("p3_v25", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_config_is_sealed_and_access_zero() -> None:
    config = MODULE.load_config()
    assert config["status"] == "SEALED_BEFORE_OUTER_SCORING"
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())


def test_revin_is_explicit_duplicate_stop() -> None:
    audit = MODULE.load_config()["duplication_audit"]
    assert audit["revin_style_joint_residual"] == "STOP_SEMANTIC_DUPLICATE"
    assert audit["semantic_verdict"] == "NON_DUPLICATE_ARCHITECTURE_AXIS"


def test_simplex_fit_recovers_feasible_weights_deterministically() -> None:
    x = np.column_stack([np.linspace(0, 1, 30), np.linspace(1, 0, 30), np.ones(30), np.zeros(30), np.linspace(0.2, 0.8, 30)])
    y = 0.25 * x[:, 0] + 0.75 * x[:, 1]
    first, _ = MODULE.simplex_fit(x, y)
    second, _ = MODULE.simplex_fit(x, y)
    assert np.array_equal(first, second)
    assert np.all(first >= 0)
    assert np.isclose(first.sum(), 1.0)
    assert np.allclose(x @ first, y, atol=1e-8)


def test_base_surface_is_key_aligned_and_finite() -> None:
    cases, truth, reference, matrix, profile = MODULE.base_surface()
    assert len(cases) == 182
    assert truth.shape == reference.shape == (182, 6)
    assert matrix.shape == (1092, 5)
    assert np.isfinite(matrix).all()
    assert profile["base_matrix"]["target_alignment_max_abs"] == 0.0


def test_preflight_or_consumed_namespace_is_deterministic() -> None:
    if MODULE.ARTIFACT_DIR.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="already consumed"):
            MODULE.preflight_payload()
    else:
        first = MODULE.preflight_payload()
        second = MODULE.preflight_payload()
        assert first == second
        assert first["maximum_model_fits"] == 12
        assert first["official_access"] == first["csv_materializations"] == first["uploads"] == 0
