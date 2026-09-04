from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p3_inner_lcb_router_cycle_20260831_v6 as module  # noqa: E402


def test_candidates_are_frozen_and_unique() -> None:
    assert len(module.SPECS) == 3
    assert len({spec.name for spec in module.SPECS}) == 3
    assert module.LCB_RESIDUAL_QUANTILE == 0.80
    assert module.INTERNAL_UNIQUE_FITS == 72
    assert module.MAX_UNIQUE_FITS == 86


def test_candidate_route_obeys_physical_bounds_and_consensus() -> None:
    et = np.asarray([-0.1, 0.2, 0.3])
    cb = np.asarray([0.2, -0.1, 0.4])
    first = module.candidate_route(module.SPECS[0], et, cb)
    second = module.candidate_route(module.SPECS[1], et, cb)
    consensus = module.candidate_route(module.SPECS[2], et, cb)
    assert np.array_equal(first, [0.0, module.REFERENCE_ALPHA, module.REFERENCE_ALPHA])
    assert np.array_equal(second, [module.REFERENCE_ALPHA, 0.0, module.REFERENCE_ALPHA])
    assert np.array_equal(consensus, [0.0, 0.0, module.REFERENCE_ALPHA])
    assert consensus.min() >= 0.0
    assert consensus.max() <= module.ALPHA_MAX


def test_score_translation_is_ordered_and_not_a_gate() -> None:
    translated = module.conditional_score_translation(-0.01, -0.02, 0.005)
    scenarios = translated["scenarios"]
    assert scenarios["conservative"]["projected_points"] < scenarios["central"]["projected_points"]
    assert scenarios["central"]["projected_points"] < scenarios["optimistic"]["projected_points"]
    assert translated["status"] == "CONDITIONAL_PLANNING_ESTIMATE_NOT_A_PROMOTION_GATE"
