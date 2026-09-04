from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts/run_p3_sors_era5_orthogonal_prequential_cycle_20260831_v12.py"
SPEC = importlib.util.spec_from_file_location("p3_v12_runner", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_prequential_contract_is_frozen() -> None:
    assert MODULE.FOLD_ORDER == ("2024_h2_storm", "winter_transition", "2025_h1")
    assert MODULE.ENERGY_QUANTILE == 0.67
    assert MODULE.CORRECTION_CAP_M == 0.25


def test_candidate_contract_is_three_distinct_estimators() -> None:
    assert len(MODULE.SPECS) == 3
    assert {item.family for item in MODULE.SPECS} == {"analytic_huber", "ridge", "huber"}


def test_public_transport_rmse_equivalent_is_exact() -> None:
    assert MODULE.CI_TARGET_M == -0.020913058224751535
    assert MODULE.MIN_CALIBRATED_POINTS == 0.01
