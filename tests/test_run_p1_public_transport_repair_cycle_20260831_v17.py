from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v17 as cycle  # noqa: E402

from src.p1_qc.causal_minirocket_lite import (  # noqa: E402
    build_spec,
    causal_trailing_robust_z,
    causal_windows,
)


def test_contract_is_exactly_sealed() -> None:
    config = cycle.load_contract()
    assert config["representation"]["feature_count"] == 512
    assert config["model"]["probability_threshold_inclusive"] == 0.99
    assert config["fit_budget"] == {"pipeline_fits": 2, "maximum": 2}
    assert config["resource_caps"] == {"wall_seconds": 5400, "rss_bytes": 17179869184, "vram_bytes": 8589934592}
    assert config["model"]["retuning"] is False


def test_future_rows_do_not_change_prior_windows() -> None:
    times = np.arange(0, 2001, 15)
    values = np.sin(times).astype(np.float32)
    before = causal_windows(times, values, np.asarray([1800]))
    after = causal_windows(np.append(times, 2010), np.append(values, 999), np.asarray([1800]))
    np.testing.assert_array_equal(before, after)


def test_gap_over_two_hours_is_not_interpolated() -> None:
    window = causal_windows(np.asarray([0, 15, 300, 315, 1800]), np.arange(5), np.asarray([1800]))
    assert (window[0, 1] == 0).any()


def test_spec_is_deterministic_and_exact() -> None:
    first, second = build_spec(), build_spec()
    assert first.weights.shape == (512, 9)
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.dilations, second.dilations)
    assert int((first.channels == 0).sum()) == 384
    assert int((first.channels == 1).sum()) == 128


def test_trailing_robust_z_is_prefix_only_and_clipped() -> None:
    times = np.arange(0, 600, 15)
    values = np.sin(times / 50).astype(np.float32)
    before = causal_trailing_robust_z(times, values)
    after = causal_trailing_robust_z(np.append(times, [615, 630]), np.append(values, [999, -999]))
    np.testing.assert_array_equal(before, after[: len(before)])
    assert np.abs(after).max() <= 12


def test_execution_authorized_but_exactly_once() -> None:
    config = cycle.load_contract()
    assert config["authorization"]["historical_execution"] is True
    assert config["authorization"]["attempt_lock_creation"] is True
