import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/run_p3_episode_weight_20260905_v2.py"
SPEC = importlib.util.spec_from_file_location("p3_episode_weight_tests", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def anchors():
    return pd.DataFrame({"anchor_id": [0, 1, 2, 3, 4], "station": ["A"] * 5, "current_hs": [2.0] * 5, "episode_id": [0, 0, 0, 0, 1]})


def test_inverse_sqrt_episode_mass_and_normalization():
    w = M.training_weights(anchors(), np.arange(5), "episode_weight")
    np.testing.assert_allclose(w, np.array([0.5, 0.5, 0.5, 0.5, 1]) / 0.6)
    assert w.mean() == pytest.approx(1)


def test_outer_train_only_counts_ignore_future_rows():
    data = anchors()
    first = M.training_weights(data, [0, 4], "episode_weight")
    changed = pd.concat([data, pd.DataFrame({"anchor_id": np.arange(10, 110), "station": "A", "current_hs": 20, "episode_id": 1})], ignore_index=True)
    second = M.training_weights(changed, [0, 4], "episode_weight")
    np.testing.assert_array_equal(first, second)


def test_control_weight_is_unchanged_threshold_weight():
    data = anchors()
    data.current_hs = [1.5, 2, 3, 4, 8]
    np.testing.assert_array_equal(M.training_weights(data, np.arange(5), "control"), M.threshold_case_weights(data.current_hs.to_numpy()))


def test_background_is_not_a_giant_episode():
    times = pd.date_range("2024-01-01", periods=6, freq="20min", tz="UTC")
    wave = pd.DataFrame({"station": "A", "time": times, "hs": [2, 2, 0.5, 0.6, 2, 2]})
    selected = pd.DataFrame({"anchor_id": [0, 1, 4, 5], "station": "A", "anchor_time": times[[0, 1, 4, 5]], "current_hs": 2.0})
    result = M.assign_storm_episodes_from_wave(selected, wave)
    assert result.episode_id.nunique() == 2
    assert result.loc[0, "episode_id"] != result.loc[2, "episode_id"]
    selected.loc[0, "current_hs"] = 0.5
    with pytest.raises(ValueError):
        M.assign_storm_episodes_from_wave(selected, wave)


def test_target_eligibility_holes_do_not_split_one_physical_storm():
    times = pd.date_range("2024-01-01", periods=6, freq="20min", tz="UTC")
    wave = pd.DataFrame({"station": "A", "time": times, "hs": 2.0})
    selected = pd.DataFrame({"anchor_id": [0, 5], "station": "A", "anchor_time": times[[0, 5]], "current_hs": 2.0})
    result = M.assign_storm_episodes_from_wave(selected, wave)
    assert result.episode_id.nunique() == 1


def component_frame(offset=0):
    frame = pd.DataFrame({"anchor_id": np.repeat([0, 1], 6), "station": "A", "lead_h": np.tile(M.LEADS, 2), "fold": "a", "current_hs": 2.0, "target_hs": 3.0, "single_prediction": 2.0 + offset, "multi_prediction": 4.0 + offset, "persistence": 2.0})
    return frame


def test_average_is_key_aligned_before_router():
    out = M.average_components(component_frame(), component_frame(2).iloc[::-1])
    np.testing.assert_array_equal(out.single_prediction, np.full(12, 3.0))
    np.testing.assert_array_equal(out.multi_prediction, np.full(12, 5.0))
    np.testing.assert_array_equal(out.equal_prediction, np.full(12, 4.0))
    assert "routed_prediction" not in out.columns


def test_average_rejects_wrong_truth_and_duplicate_keys():
    changed = component_frame(2)
    changed.loc[0, "target_hs"] = 99
    with pytest.raises(ValueError):
        M.average_components(component_frame(), changed)
    with pytest.raises(ValueError):
        M.average_components(component_frame(), pd.concat([component_frame(), component_frame()]))


def test_six_lead_weight_expansion_uses_lead_major_order():
    data = anchors()
    for lead in M.LEADS:
        data[f"target_{lead}"] = 2.1
    features = pd.DataFrame({"anchor_id": np.arange(5), "station": "A", "x": np.arange(5, dtype=float)})
    _, _, meta = M.expand_leads(features, data, np.arange(5), ["x"])
    w = M.training_weights(data, np.arange(5), "episode_weight")
    actual = pd.Series(w, index=np.arange(5)).loc[meta.anchor_id].to_numpy()
    np.testing.assert_array_equal(actual, np.tile(w, 6))
    np.testing.assert_allclose(pd.Series(actual).groupby(meta.anchor_id).sum(), 6*w)
