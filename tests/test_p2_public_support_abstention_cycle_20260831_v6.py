import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).parents[1] / "scripts/run_p2_public_support_abstention_cycle_20260831_v6.py"
SPEC = importlib.util.spec_from_file_location("p2v6", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_constants() -> None:
    assert MODULE.PENALTY == 0.12168209161000616


def test_rmse() -> None:
    assert MODULE.rmse(np.array([1.0]), np.array([1.0])) == 0.0
