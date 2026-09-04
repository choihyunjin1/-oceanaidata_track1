from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p2_v12_source_regime_coverage_audit_20260901_v12a.py"
SPEC = importlib.util.spec_from_file_location("p2_v12a", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_audit_is_zero_fit() -> None:
    config = MODULE.load_config()
    assert config["operation_limits"]["model_fits"] == 0
    assert config["operation_limits"]["official_rows_read"] == 0


def test_descriptor_is_permutation_invariant() -> None:
    rng = np.random.default_rng(7)
    tokens = rng.normal(size=(6, 5, 8)).astype(np.float32)
    mask = np.ones((6, 5), dtype=np.float32)
    context = rng.normal(size=(6, 11)).astype(np.float32)
    order = [4, 1, 3, 0, 2]
    expected = MODULE.descriptor(tokens, mask, context)
    observed = MODULE.descriptor(tokens[:, order], mask[:, order], context)
    np.testing.assert_allclose(expected, observed, atol=2e-7, rtol=0.0)


def test_robust_distance_receipt_is_finite() -> None:
    train = np.arange(230, dtype=float).reshape(10, 23)
    query = train[:3] + 0.5
    receipt = MODULE.robust_distance_receipt(train, query, 0.05)
    assert receipt["rows"] == 3
    assert np.isfinite(list(receipt.values())).all()
