import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/qa_forward_cycle_20260906_v1.py"
SPEC = importlib.util.spec_from_file_location("root_qa", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_f1_counts():
    result = MODULE.metric([1, 1, 0, 0], [1, 0, 1, 0], "f1")
    assert result == {"value": 0.5, "tp": 1, "fp": 1, "fn": 1, "n": 4}


def test_rmse_pooled():
    result = MODULE.metric([0, 0, 0, 0], [0, 0, 0, 4], "rmse")
    assert result == {"value": 2.0, "sse": 16.0, "n": 4}


@pytest.mark.parametrize("y,p,name", [([], [], "rmse"), ([1], [1, 2], "rmse"),
                                      ([1], [float("nan")], "rmse"), ([1], [0.2], "f1")])
def test_reject_invalid(y, p, name):
    with pytest.raises(ValueError):
        MODULE.metric(y, p, name)
