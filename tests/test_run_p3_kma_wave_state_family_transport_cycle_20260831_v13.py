from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts/run_p3_kma_wave_state_family_transport_cycle_20260831_v13.py"
SPEC = importlib.util.spec_from_file_location("p3_v13_runner", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_family_registration_is_frozen() -> None:
    fixed, high, low = MODULE.SPECS
    assert fixed.family_id == "P3_FIXED_KMA_LONGLEAD_FACTOR"
    assert fixed.penalty == 0.04958605409228893
    assert high.tier_id == low.tier_id == "HARD_CONDITIONAL_ROUTER"
    assert high.penalty == low.penalty == 0.3219056897594759


def test_fixed_and_energy_policy_contract() -> None:
    assert [item.alpha_24 for item in MODULE.SPECS] == [0.6, 0.6, 0.2]
    assert [item.quantile for item in MODULE.SPECS] == [None, 0.67, 0.33]
    assert MODULE.REFERENCE_ALPHA == 0.425


def test_family_gate_is_inclusive_point01() -> None:
    assert MODULE.MIN_CALIBRATED_POINTS == 0.01
    assert abs(MODULE.SPECS[0].raw_threshold - MODULE.SPECS[0].penalty - 0.01) < 1e-12
