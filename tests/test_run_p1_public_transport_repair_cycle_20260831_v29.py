import numpy as np

from scripts import run_p1_public_transport_repair_cycle_20260831_v29 as runner
from src.p1_qc.inner_group_day_guard import apply_group_guard, day_cap_mask, eligible_groups


def test_group_guard_requires_support_precision_and_nonnegative_delta():
    n = 100
    y = np.r_[np.ones(30), np.zeros(70)].astype(np.int8)
    anchor = np.zeros(n, dtype=np.int8)
    proposed = np.r_[np.ones(25), np.zeros(75)].astype(bool)
    station = np.array(["A"] * n)
    layer = np.ones(n, dtype=int)
    allowed = eligible_groups(y, anchor, proposed, station, layer, minimum_support=20)
    assert allowed == {("A", 1)}
    assert apply_group_guard(proposed, station, layer, allowed).sum() == 25


def test_day_cap_is_half_percent_floor_and_stable():
    n = 1000
    proposed = np.ones(n, dtype=bool)
    score = np.ones(n)
    day = np.zeros(n, dtype=int)
    kept = day_cap_mask(proposed, score, day)
    assert kept.sum() == 5
    assert np.flatnonzero(kept).tolist() == [0, 1, 2, 3, 4]


def test_contract_preserves_v28_parameters_and_authorizes_exact_once():
    cfg = runner.load_contract()
    assert cfg["model"]["C"] == 0.1
    assert cfg["em"]["maximum_iterations"] == 200
    assert cfg["authorization"]["historical_execution"] is True


def test_synthetic_preflight_passes():
    assert runner.preflight()["status"] == "PASS"
