import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ocean_evaluation_contract_v5 as cv
from p3_forward_independent_statistics_20260906_v2 import independent_bootstrap


def test_unequal_cluster_denominator_and_paired_order():
    frame = pd.DataFrame(
        {
            "station": ["A"] * 8,
            "episode_id": [1, 2, 2, 3, 3, 3, 3, 3],
            "target_hs": [2.0] * 8,
            "control": np.arange(8) / 4,
            "candidate": np.arange(8) / 5,
        }
    )
    cfg = cv.load_contract()
    a = independent_bootstrap(frame, cfg["common"]["bootstrap"])
    b = cv.paired_bootstrap(
        frame.target_hs,
        frame.control,
        frame.candidate,
        cv.bootstrap_groups(frame, "P3", cfg),
        "rmse",
        cfg,
    )
    np.testing.assert_allclose(a["ci90"], b["ci90"], atol=1e-12, rtol=0)
    assert a["p_improve"] == b["p_improve"] and a["n_rows"] == 8 and a["n_clusters"] == 3
