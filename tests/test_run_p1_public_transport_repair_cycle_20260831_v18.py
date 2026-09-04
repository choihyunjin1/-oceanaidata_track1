from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v18 as cycle  # noqa: E402

from src.p1_qc.causal_soft_symbolic import (  # noqa: E402
    build_spec,
    causal_robust_windows,
    soft_symbolic_transform,
)


def test_contract_is_exactly_frozen() -> None:
    contract = cycle.load_contract()
    assert contract["representation"]["paa_segments"] == 12
    assert len(contract["representation"]["soft_symbol_centers"]) == 5
    assert contract["representation"]["feature_count"] == 347
    assert contract["model"]["C"] == 0.1
    assert contract["model"]["probability_threshold_inclusive"] == 0.99
    assert contract["fit_budget"] == {"pipeline_fits": 2, "maximum": 2}
    assert contract["authorization"]["historical_execution"] is True
    assert contract["authorization"]["attempt_lock_creation"] is True


def test_future_append_invariance_and_no_long_gap_fill() -> None:
    times = np.arange(0, 8001, 10, dtype=np.int64)
    values = np.sin(times / 100.0)
    queries = np.asarray([7200])
    before, before_mask = causal_robust_windows(times, values, queries)
    after, after_mask = causal_robust_windows(np.append(times, 8010), np.append(values, 999.0), queries)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(before_mask, after_mask)

    sparse_times = np.asarray([0, 10, 300, 310, 7200])
    sparse_values = np.arange(len(sparse_times), dtype=float)
    _, sparse_mask = causal_robust_windows(sparse_times, sparse_values, queries)
    assert (~sparse_mask).any()


def test_soft_symbolic_transform_is_deterministic_finite_and_exact_width() -> None:
    rng = np.random.default_rng(20260831)
    windows = rng.normal(size=(7, 145)).astype(np.float32)
    observed = rng.random((7, 145)) > 0.05
    first = soft_symbolic_transform(windows, observed)
    second = soft_symbolic_transform(windows, observed, build_spec())
    assert first.shape == (7, 347)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_pointwise_robust_normalization_is_offset_invariant() -> None:
    times = np.arange(0, 10_001, 10, dtype=np.int64)
    values = np.cos(times / 300.0) + 0.002 * times
    queries = np.asarray([7200, 9000])
    original, mask = causal_robust_windows(times, values, queries)
    shifted, shifted_mask = causal_robust_windows(times, values + 1234.5, queries)
    np.testing.assert_array_equal(mask, shifted_mask)
    np.testing.assert_allclose(original, shifted, atol=1.0e-5, rtol=0.0)


def test_vectorized_prefix_normalization_matches_reference() -> None:
    times = np.arange(0, 1500, 10, dtype=np.int64)
    values = np.sin(times / 170.0) + 0.001 * times
    vectorized = cycle._robust_prefix_z(times, values, device="cpu", chunk_size=37)
    reference_windows, reference_mask = causal_robust_windows(times, values, times)
    expected = np.where(reference_mask[:, -1], reference_windows[:, -1], np.nan)
    np.testing.assert_allclose(vectorized, expected, atol=1.0e-6, rtol=0.0, equal_nan=True)


def test_validate_only_has_no_artifact_side_effect() -> None:
    before = cycle.ARTIFACT.exists()
    payload = cycle.validate_only()
    assert payload["status"] == "VALID"
    assert payload["fit_budget"] == 2
    assert cycle.ARTIFACT.exists() is before
