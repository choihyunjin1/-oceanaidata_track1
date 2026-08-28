from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_p3_chronos2_full_nested_v2 import (
    _cluster_bootstrap,
    _slice_deltas,
)


def test_cluster_bootstrap_keeps_whole_cases() -> None:
    frame = pd.DataFrame(
        {
            "anchor_id": np.repeat([1, 2, 3], 2),
            "prediction": [1.0] * 6,
            "comparator": [2.0] * 6,
            "target_hs": [1.0] * 6,
        }
    )
    result = _cluster_bootstrap(
        frame,
        "comparator",
        "anchor_id",
        replicates=100,
        seed=7,
    )
    assert result["clusters"] == 3
    assert result["observed_delta_m"] == -1.0
    assert result["ci90_m"] == [-1.0, -1.0]


def test_slice_deltas_are_candidate_minus_comparator() -> None:
    candidate = {
        "pooled_rmse_m": 1.0,
        "by_fold_rmse_m": {"f": 2.0},
        "by_station_rmse_m": {"s": 3.0},
        "by_lead_rmse_m": {"3": 4.0},
    }
    comparator = {
        "pooled_rmse_m": 2.0,
        "by_fold_rmse_m": {"f": 3.0},
        "by_station_rmse_m": {"s": 4.0},
        "by_lead_rmse_m": {"3": 5.0},
    }
    result = _slice_deltas(candidate, comparator)
    assert result["pooled_delta_m"] == -1.0
    assert result["by_fold_delta_m"]["f"] == -1.0
