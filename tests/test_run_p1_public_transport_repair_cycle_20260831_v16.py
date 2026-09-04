from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v16 as cycle  # noqa: E402


def test_contract_fixes_gce_threshold_and_two_fits() -> None:
    config = cycle.load_contract()
    assert config["model"]["gce_q"] == 0.7
    assert config["model"]["l2"] == 0.001
    assert config["model"]["probability_threshold_inclusive"] == 0.95
    assert config["fit_budget"]["maximum"] == 2


def test_leverage_weights_downweight_without_deleting() -> None:
    scaled = np.array([[0.0, 1.0], [8.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    weights = cycle.leverage_weights(scaled)
    assert np.allclose(weights, [1.0, 1.0, 0.8])
    assert (weights >= 0.25).all()


def test_gce_analytic_gradient_matches_finite_difference() -> None:
    design = np.array([[1.0, -0.5], [0.2, 0.3], [-0.7, 1.1]], dtype=np.float64)
    labels = np.array([1, 0, 1], dtype=np.int8)
    weights = np.array([1.0, 0.8, 1.0], dtype=np.float64)
    parameters = np.array([0.1, -0.2, 0.05], dtype=np.float64)
    value, gradient = cycle.gce_objective_gradient(
        parameters,
        design,
        labels,
        weights,
        q=0.7,
        l2=0.001,
    )
    finite = np.empty_like(parameters)
    epsilon = 1e-6
    for index in range(len(parameters)):
        right = parameters.copy()
        left = parameters.copy()
        right[index] += epsilon
        left[index] -= epsilon
        right_value = cycle.gce_objective_gradient(
            right, design, labels, weights, q=0.7, l2=0.001
        )[0]
        left_value = cycle.gce_objective_gradient(
            left, design, labels, weights, q=0.7, l2=0.001
        )[0]
        finite[index] = (right_value - left_value) / (2 * epsilon)
    assert np.isfinite(value)
    assert np.allclose(gradient, finite, atol=1e-7)
