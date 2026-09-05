import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_p2_missingness_conditional_3seed_20260905_v3 as module  # noqa: E402


def test_fixed_contract():
    cfg = module.read_config()
    assert cfg["maximum_new_historical_fits"] + cfg["maximum_new_full_fits"] == 9
    assert cfg["maximum_new_rule_fits"] == 0
    assert cfg["official_access_rows"] == cfg["csv_written"] == cfg["upload"] == 0


def test_three_seed_mean_not_best_seed():
    c = module.component_mean([np.array([1., 4.]), np.array([3., 6.]), np.array([5., 8.])])
    r = module.component_mean([np.array([2., 3.]), np.array([4., 5.]), np.array([6., 7.])])
    np.testing.assert_array_equal(module.prior.conditional(c, r, np.array([False, True])), [3., 5.])


@pytest.mark.parametrize("components", [[np.array([1.])], [np.array([1.]), np.array([np.nan]), np.array([2.])]])
def test_mean_requires_three_finite_components(components):
    with pytest.raises(ValueError):
        module.component_mean(components)


def test_public_missingness_four_cases_target_independence():
    frame = pd.DataFrame({"temp_5": [np.nan, 1., np.nan, 1.], "psal_5": [1., np.nan, np.nan, 1.], "temp_2": [1., 1., 1., np.nan], "psal_2": [1., 1., 1., np.nan]})
    np.testing.assert_array_equal(module.prior.route_trigger(frame), [True, True, True, False])
    frame[["temp_2", "psal_2"]] = 999999.
    np.testing.assert_array_equal(module.prior.route_trigger(frame), [True, True, True, False])


def test_same_episode_and_support_policy():
    cfg = module.prior.read_config()
    assert len(cfg["episodes"]) == 10
    assert sum(not e["development"] for e in cfg["episodes"]) == 9
    assert module.read_config()["unsupported_policy"] == cfg["unsupported_policy"]
    assert cfg["feature_dependency_hours"] == 0
