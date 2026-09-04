from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_multiscale_wavelet_scattering_residual_cycle_20260901_v27.py"
SPEC = importlib.util.spec_from_file_location("p3_v27", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_config_is_sealed_and_official_zero() -> None:
    config = json.loads(MODULE.CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "SEALED_BEFORE_OUTER_SCORING"
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())


def test_scattering_is_deterministic_and_finite() -> None:
    sequence = np.arange(2890, dtype=float).reshape(289, 10) / 100
    sequence[::13, 3] = np.nan
    first = MODULE.scattering_features(sequence)
    second = MODULE.scattering_features(sequence)
    assert first.shape == (336,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_constant_path_has_zero_scattering() -> None:
    sequence = np.ones((289, 10), dtype=float)
    feature = MODULE.scattering_features(sequence)
    assert np.allclose(feature, 0.0)


def test_novelty_audit_is_explicit() -> None:
    config = json.loads(MODULE.CONFIG.read_text(encoding="utf-8"))
    audit = config["duplication_audit"]
    assert audit["semantic_verdict"] == "NON_DUPLICATE_FIXED_SCATTERING_AXIS"
    assert audit["repository_exact_hits"] == 0


def test_preflight_or_consumed_namespace() -> None:
    if MODULE.ARTIFACT_DIR.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight_payload()
    else:
        payload = MODULE.preflight_payload()
        assert payload["status"] == "READY_EXACTLY_ONCE"
        assert payload["official_access"] == 0
