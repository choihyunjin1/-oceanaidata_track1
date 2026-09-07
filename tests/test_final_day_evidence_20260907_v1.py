import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_final_day_evidence_20260907_v1 import cluster_bootstrap, pava, rmse


def test_pava_independent_known_blocks():
    np.testing.assert_array_equal(pava([3, 1, 2]), [2, 2, 2])
    np.testing.assert_array_equal(pava([1, 3, 2]), [1, 2.5, 2.5])
    np.testing.assert_array_equal(pava([1, 3, 2], False), [2, 2, 2])
    np.testing.assert_array_equal(pava([3]), [3])


def test_clipping_order_is_not_interchangeable():
    data = np.array([100, -100, 5])
    before = pava(np.clip(data, 0, 10))
    after = np.clip(pava(data), 0, 10)
    assert not np.array_equal(before, after)
    np.testing.assert_array_equal(before, [5, 5, 5])


def test_rmse_pooled_not_mean_of_block_scores():
    assert rmse([0, 0, 0, 0], [0, 0, 0, 2]) == 1
    with pytest.raises(ValueError):
        rmse([0], [float('nan')])


def test_bootstrap_station_episode_and_perfect_improvement():
    frame = pd.DataFrame({'station': ['S', 'S', 'I', 'I'], 'episode_id': [0, 0, 0, 0], 'target_hs': [1., 2., 1., 2.]})
    truth = frame.target_hs.to_numpy()
    result = cluster_bootstrap(frame, truth+1, truth, {'seed': 1, 'resamples': 20, 'ci_quantiles': [.05, .95]})
    assert result['clusters'] == 2
    assert result['p_improve'] == 1
    assert result['ci90'] == [-1., -1.]
