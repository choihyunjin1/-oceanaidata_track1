import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/p2_final_day_projection_20260907_v1.py"
spec = importlib.util.spec_from_file_location("p2_final_projection", PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def test_exact_clip_before_pava_and_incomplete():
    frame = pd.DataFrame({"time": ["2025-01-01"]*3+["2025-01-02"]*2, "layer": [2,3,4,2,3]})
    ep = pd.DataFrame({"time": ["2025-01-01", "2025-01-02"], "temp_1": [0.,0.], "temp_5": [10.,10.]})
    result = m.project_profiles_vectorized(frame, np.array([100.,0.,0.,100.,-10.]), ep)
    np.testing.assert_allclose(result.prediction, [10/3]*3+[100.,-10.])
    np.testing.assert_array_equal(result.eligible_mask, [True]*3+[False]*2)


def test_fallback_missing_equal_inverted_and_reference():
    obs = pd.DataFrame({"time": ["2025-01-01"]*5, "layer": [1,5,6,7,8], "temp": [10., np.nan, np.inf,2.,1.]})
    ep = m.public_endpoint_frame(obs)
    assert ep.temp_5.iloc[0] == 2.
    f = pd.DataFrame({"time": ["2025-01-01"]*3, "layer": [4,2,3]})
    for endpoints in [(10.,2.), (4.,4.), (np.nan,2.)]:
        ep["temp_1"], ep["temp_5"] = endpoints
        p = np.array([9.,-1.,8.])
        a,b = m.project_profiles(f,p,ep),m.project_profiles_vectorized(f,p,ep)
        np.testing.assert_allclose(a.prediction,b.prediction,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(b.prediction,p)


def test_duplicates_rejected():
    f = pd.DataFrame({"time": ["2025-01-01"]*3, "layer": [2,2,3]})
    with pytest.raises(ValueError):
        m.project_profiles_vectorized(f, np.ones(3), pd.DataFrame())
