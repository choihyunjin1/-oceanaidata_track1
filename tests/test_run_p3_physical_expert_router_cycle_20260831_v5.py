from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_p3_physical_expert_router_cycle_20260831_v5.py"
)
SPEC = importlib.util.spec_from_file_location("p3_router_v5", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_frozen_candidate_count_and_families() -> None:
    assert len(MODULE.SPECS) == 3
    assert {spec.family for spec in MODULE.SPECS} == {"catboost", "extra_trees", "logistic"}


def test_router_policies_are_physical_and_finite() -> None:
    probability = np.array([0.0, 0.39, 0.5, 0.75, 1.0])
    for spec in MODULE.SPECS:
        alpha = MODULE.route_alpha(probability, spec.policy)
        assert np.isfinite(alpha).all()
        assert np.all(alpha >= 0.0)
        assert np.all(alpha <= MODULE.REFERENCE_ALPHA)


def test_sample_weight_is_bounded() -> None:
    import pandas as pd

    weight = MODULE._sample_weight(pd.Series([-100.0, -1.0, 0.0, 1.0, 100.0]))
    assert np.all(weight >= 0.25)
    assert np.all(weight <= 4.0)
