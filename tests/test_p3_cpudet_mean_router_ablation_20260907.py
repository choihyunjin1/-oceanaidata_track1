import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("mean_ablation", SCRIPTS / "p3_cpudet_mean_router_ablation_20260907_v1.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_holds_shrink_fixed_and_averages():
    assert np.array_equal(MODULE.mean_prediction([1., 1.], [3., 3.], [7., 7.], [3, 12]), [2., 3.])


def test_component_order_symmetric():
    assert np.array_equal(MODULE.mean_prediction([1.], [3.], [7.], [24]),
                          MODULE.mean_prediction([3.], [1.], [7.], [24]))


def test_invalid_components_rejected():
    with pytest.raises(ValueError):
        MODULE.mean_prediction([np.nan], [3.], [7.], [24])
    with pytest.raises(ValueError):
        MODULE.mean_prediction([1., 2.], [3.], [7.], [24])
