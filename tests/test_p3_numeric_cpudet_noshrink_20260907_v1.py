import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("no_shrink_internal", Path(__file__).parents[1] /
    "scripts/p3_numeric_cpudet_noshrink_20260907_v1_internal.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_inverse_only_long_leads():
    lead = np.array([3, 6, 9, 12, 18, 24])
    raw = np.array([1., 2., 3., 4., 5., 6.])
    persistence = np.full(6, 2.)
    base = raw.copy()
    base[3:] = .8*raw[3:] + .2*persistence[3:]
    np.testing.assert_allclose(m.undo(base, persistence, lead), raw, rtol=0, atol=1e-14)
    assert np.array_equal(m.undo(base, persistence, lead)[:3], base[:3])


def test_paired_sse_and_block_bootstrap():
    frame = pd.DataFrame({"station": ["A"]*4, "episode_id": [1, 1, 2, 2],
        "target_hs": [0., 0., 0., 0.], "final_prediction": [2., 2., 2., 2.],
        "noshrink": [1., 1., 1., 1.]})
    assert m.compare(frame)["delta"] == -1
    b = m.bootstrap(frame, {"seed": 1, "resamples": 30})
    assert b["blocks"] == 2 and b["p_improve"] == 1 and b["ci90"] == [-1., -1.]
