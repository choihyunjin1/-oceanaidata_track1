"""Synthetic target-masking, trigger and exact episode boundary tests."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p2_missingness_v3", ROOT / "scripts/run_p2_missingness_conditional_validation_20260905_v3.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def observations():
    return pd.DataFrame([{"station": "S-ORS", "time": t, "layer": layer, "nominal_depth": layer*5., "depth": layer*5.+.5, "temp": 20-layer*.3, "psal": 30+layer*.1} for t in pd.date_range("2024-09-03", periods=8, freq="10min", tz="UTC") for layer in range(1, 9)])


def test_target_temp_and_psal_poison_masked_before_features_and_trigger():
    obs = observations()
    frame, _ = M.base.public_frame(obs)
    poisoned = obs.copy()
    poisoned.loc[poisoned.layer.isin([2, 3, 4]), ["temp", "psal"]] = [-999, 999]
    second, _ = M.base.public_frame(poisoned)
    for a, b in zip(M.base.arrays(frame), M.base.arrays(second), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(M.route_trigger(frame), M.route_trigger(second))
    assert not M.route_trigger(frame).any()


def test_trigger_only_public_layer_five_availability():
    frame, _ = M.base.public_frame(observations())
    frame["target"] = np.nan
    assert not M.route_trigger(frame).any()
    frame.loc[0, "psal_5"] = np.nan
    frame.loc[1, "temp_5"] = np.inf
    assert M.route_trigger(frame).sum() == 2


@pytest.mark.parametrize("missing,expected", [(["temp_5"], True), (["psal_5"], True), (["temp_5", "psal_5"], True), (["target", "temp_2", "psal_2"], False)])
def test_four_availability_cases(missing, expected):
    frame, _ = M.base.public_frame(observations())
    for name in missing:
        frame[name] = np.nan
    assert bool(M.route_trigger(frame).all()) == expected


def test_onset_inclusive_offset_exclusive_no_pre_post_lag_leak():
    frame, _ = M.base.public_frame(observations())
    episode = {"start": "2024-09-03T00:20:00+00:00", "stop": "2024-09-03T00:50:00+00:00"}
    changed, selected = M.episode_frame(frame, episode)
    assert selected.sum() == 9
    assert changed.loc[selected, ["temp_5", "psal_5"]].isna().all().all()
    for original, masked in zip(M.base.arrays(frame), M.base.arrays(changed), strict=True):
        np.testing.assert_array_equal(original[~selected], masked[~selected])
    assert M.route_trigger(changed).sum() == 9
    assert not M.base.arrays(changed)[1][selected, 1].any()


def test_rule_is_exact_C_when_available_R_when_missing():
    c, r = np.array([1., 2., 3.]), np.array([8., 9., 10.])
    np.testing.assert_array_equal(M.conditional(c, r, np.array([False, True, False])), [1, 9, 3])
    with pytest.raises(ValueError):
        M.conditional(c, r[:1], np.zeros(3, bool))


def test_episode_dates_nonoverlap_new_vs_old_and_fit_zero():
    cfg = M.read_config()
    for i, left in enumerate(cfg["episodes"]):
        assert M.base.utc(left["start"]) < M.base.utc(left["stop"])
        for right in cfg["episodes"][i+1:]:
            assert M.base.utc(left["stop"]) <= M.base.utc(right["start"]) or M.base.utc(right["stop"]) <= M.base.utc(left["start"])
    assert cfg["maximum_new_backbone_fits"] == cfg["maximum_rule_fits"] == 0


def test_metric_same_row_sse_and_no_unsupported_deletion():
    result = M.panel_metrics(np.zeros(3), {"C": np.array([0, 0, 3]), "R": np.zeros(3), "conditional": np.array([0, 0, 3])}, np.array(["2024_sep_oct"]*3), np.zeros(3, bool))
    assert result["autumn_primary"]["n"] == 3
    assert result["autumn_primary"]["metrics"]["C"]["rmse"] == pytest.approx(np.sqrt(3))
    assert result["trigger"] == {"n": 0, "metrics": None}


def test_runner_has_no_training_or_official_generation():
    code = (ROOT / "scripts/run_p2_missingness_conditional_validation_20260905_v3.py").read_text(encoding="utf-8")
    assert "fit_model(" not in code and ".fit(" not in code and ".cuda(" not in code
    assert "to_csv(" not in code and "pd.read_parquet(" not in code
