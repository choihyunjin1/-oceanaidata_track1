from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts/run_p3_sors_longlead_episode_selector_cycle_20260831_v11.py"
SPEC = importlib.util.spec_from_file_location("p3_v11_runner", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_sealed_candidate_and_support_contract() -> None:
    assert len(MODULE.SPECS) == 3
    assert {item.policy for item in MODULE.SPECS} == {"extra_trees", "hist_gbdt", "consensus"}
    assert MODULE.INTERVENTION_BUDGET == 0.20
    assert MODULE.LCB_QUANTILE == 0.80


def test_feature_basis_contains_requested_physical_signals() -> None:
    assert {"wave_energy", "base_rise_18_24", "kma_disagreement_abs"}.issubset(MODULE.FEATURES)


def test_transport_threshold_is_inclusive_point01() -> None:
    assert MODULE.MIN_CALIBRATED_POINTS == 0.01
    assert MODULE.MAX_WORST_STATION_LEAD_M == 0.01
