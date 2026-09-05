import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/run_p3_episode_policy_fulltrain_20260905_v2.py"
SPEC = importlib.util.spec_from_file_location("p3_policy_full_tests", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class Single:
    def __init__(self, value):
        self.value = value

    def predict(self, matrix, **kwargs):
        assert kwargs["task_type"] == "CPU" and kwargs["thread_count"] == 2
        assert "target_hs" not in matrix
        return np.full(len(matrix), self.value)


class Multi(Single):
    def predict(self, matrix, **kwargs):
        assert kwargs["task_type"] == "CPU" and kwargs["thread_count"] == 2
        assert "target_hs" not in matrix
        return np.full((len(matrix), 6), self.value)


class Router:
    def predict_weights(self, frame):
        assert "target_hs" not in frame
        return np.tile([0.1, 0.2, 0.7], (len(frame), 1))


def cases():
    frame = pd.DataFrame({name: [2.0, 3.0] for name in M.OBSERVED_FEATURES})
    frame["anchor_id"] = [9, 3]
    frame["station"] = ["S-ORS", "G-ORS"]
    return frame


def test_two_seed_mean_precedes_router_and_short_noop():
    frame = cases()
    output = M.predict_cases(frame, list(M.OBSERVED_FEATURES), [(Single(1), Multi(2)), (Single(3), Multi(4))], Router())
    current = np.repeat(frame.hs_current.to_numpy(), 6)
    np.testing.assert_array_equal(output.anchor_id, np.repeat([9, 3], 6))
    np.testing.assert_array_equal(output.single_prediction, current+2)
    np.testing.assert_array_equal(output.multi_prediction, current+3)
    short = output.lead_h.le(9)
    np.testing.assert_allclose(output.loc[short, "final_prediction"], (current+2.5)[short])
    np.testing.assert_allclose(output.loc[~short, "final_prediction"], (current+0.64)[~short])


def test_clip_happens_per_seed_before_ensemble():
    frame = cases()
    frame["hs_current"] = 29
    output = M.predict_cases(frame, list(M.OBSERVED_FEATURES), [(Single(5), Multi(-50)), (Single(-5), Multi(5))], Router())
    np.testing.assert_array_equal(output.single_prediction, np.full(12, 27.0))
    np.testing.assert_array_equal(output.multi_prediction, np.full(12, 15.0))
    assert output.final_prediction.between(0, 30).all()


def test_target_values_not_used_in_inference():
    frame = cases()
    models = [(Single(1), Multi(2)), (Single(3), Multi(4))]
    before = M.predict_cases(frame, list(M.OBSERVED_FEATURES), models, Router())
    frame["target_hs"] = [999.0, -999.0]
    after = M.predict_cases(frame, list(M.OBSERVED_FEATURES), models, Router())
    pd.testing.assert_frame_equal(before, after)


def test_duplicate_case_keys_rejected():
    frame = cases()
    frame["anchor_id"] = 1
    with pytest.raises(ValueError):
        M.predict_cases(frame, list(M.OBSERVED_FEATURES), [(Single(1), Multi(2))]*2, Router())


def test_frozen_full_fit_budget_seeds_and_official_zero():
    config = json.loads(M.CONFIG.read_text())
    assert config["full_seeds"] == [20260817, 20260917]
    assert config["full_backbone_fit_budget"] == 4
    assert config["full_router_fit_budget"] == 1
    assert config["official_input_rows"] == 0
    assert config["submission_csv"] is False and config["upload"] is False
