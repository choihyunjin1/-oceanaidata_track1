import importlib.util
from pathlib import Path

PATH = Path(__file__).parents[1] / "scripts/run_p2_public_feature_benefit_gate_cycle_20260831_v7.py"
SPEC = importlib.util.spec_from_file_location("p2v7", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_gate() -> None:
    assert MODULE.PENALTY == 0.12168209161000616
