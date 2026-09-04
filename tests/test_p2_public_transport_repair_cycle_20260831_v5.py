import importlib.util
from pathlib import Path

import numpy as np

P = Path(__file__).parents[1] / "scripts/run_p2_public_transport_repair_cycle_20260831_v5.py"
S = importlib.util.spec_from_file_location("p2v5", P)
M = importlib.util.module_from_spec(S)
S.loader.exec_module(M)


def test_rmse():
    assert M.rmse(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0


def test_gate_constants():
    assert M.PENALTY == 0.12168209161000616 and M.SCALE > 12


def test_unique_prefix():
    assert M.EXP == "p2_public_transport_repair_cycle_20260831_v5"
