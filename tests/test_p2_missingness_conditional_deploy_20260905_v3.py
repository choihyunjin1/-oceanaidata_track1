import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_p2_missingness_conditional_deploy_20260905_v3 as module  # noqa: E402


def test_nontrigger_exact_and_changed_count():
    c, r = np.array([1., 2., 3.]), np.array([5., 2., 4.])
    result = module.validate_route(c, r, np.array([False, True, True]), np.array([1., 2., 4.]))
    assert result["trigger_rows"] == 2 and result["changed_rows"] == 1
    assert result["nontrigger_exact_C"]


def test_wrong_rule_rejected():
    with pytest.raises(ValueError):
        module.validate_route(np.array([1.]), np.array([2.]), np.array([False]), np.array([2.]))


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        module.validate_route(np.array([1.]), np.array([np.nan]), np.array([False]), np.array([1.]))
