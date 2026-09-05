import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor

SPEC = importlib.util.spec_from_file_location(
    "numeric_lead", Path(__file__).resolve().parents[1] / "scripts/run_p3_numeric_lead_forward_20260906_v1.py"
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def recipe():
    return json.loads((mod.ROOT / "configs/experiments/p3_corrected_repeated_forward_catboost_v2.json").read_text())


def synthetic_matrix():
    rng = np.random.default_rng(123)
    rows = 72
    frame = pd.DataFrame({"station": ["A", "B"] * 36, "lead_h": np.tile(mod.LEADS, 12), "current_hs_for_residual": 2.0, "x": rng.normal(size=rows)})
    y = frame.lead_h.to_numpy() / 10 + frame.x.to_numpy()
    return frame, y


def test_single_only_dtype_and_category_change():
    frame, _ = synthetic_matrix()
    categorical, numeric = mod.matrix(frame, False), mod.matrix(frame, True)
    assert categorical.drop(columns="lead_h").equals(numeric.drop(columns="lead_h"))
    assert pd.api.types.is_string_dtype(categorical.lead_h.dtype)
    assert numeric.lead_h.dtype == np.float64
    np.testing.assert_array_equal(numeric.lead_h, categorical.lead_h.astype(float))


@pytest.mark.parametrize("bad", [0, 7, np.nan])
def test_unknown_lead_rejected(bad):
    frame, _ = synthetic_matrix()
    frame.loc[0, "lead_h"] = bad
    with pytest.raises(ValueError):
        mod.matrix(frame, True)


@pytest.mark.parametrize("kind", ["categorical_single", "numeric_single", "multi"])
def test_cpu_native_compatibility_save_load_and_callback(tmp_path, kind):
    frame, y = synthetic_matrix()
    is_multi = kind == "multi"
    if is_multi:
        x = frame.drop(columns=["lead_h", "current_hs_for_residual"])
        y = np.column_stack([y + lead / 100 for lead in mod.LEADS])
        cats = ["station"]
    else:
        x = mod.matrix(frame, kind == "numeric_single")
        cats = ["station"] if kind == "numeric_single" else ["station", "lead_h"]
    params = mod.parameters(recipe(), "multi" if is_multi else "single", 13, synthetic=True)
    assert params["task_type"] == "CPU" and params["thread_count"] == 2 and "devices" not in params
    model = CatBoostRegressor(**params)
    model.fit(x, y, cat_features=cats, sample_weight=np.ones(len(x)), callbacks=[mod.Deadline(time.perf_counter() + 60)])
    assert model.tree_count_ == 3
    path = tmp_path / f"{kind}.cbm"
    model.save_model(path)
    restored = CatBoostRegressor().load_model(path)
    np.testing.assert_array_equal(model.predict(x, thread_count=2), restored.predict(x, thread_count=2))


def test_purge_prior_oof_even_if_from_previous_fold():
    contract = mod.cv.load_contract()
    times = pd.to_datetime(["2024-06-01T00:00Z", "2024-06-30T00:00Z", "2024-07-02T00:00Z"])
    anchors = pd.DataFrame({"anchor_id": [0, 1, 2], "station": "A", "anchor_time": times, "episode_id": [1, 2, 3]})
    meta = anchors.loc[[0, 1]].assign(fold="Q2_2024", lead_h=24)
    np.testing.assert_array_equal(mod.safe_meta_mask(meta, anchors, "Q3_2024", contract), [True, False])


def test_episode_not_broken_by_future_missing_targets():
    times = pd.date_range("2024-01-01", periods=1000, freq="20min", tz="UTC")
    wave = pd.DataFrame({"station": "A", "time": times, "hs": 2.0})
    wave.loc[250, "hs"] = np.nan
    anchors = mod.wave_anchors(wave)
    before = anchors.loc[(anchors.anchor_time > times[144]) & (anchors.anchor_time < times[250])]
    assert before.episode_id.nunique() == 1
    assert before.anchor_time.diff().max() > pd.Timedelta(minutes=20)


def test_raw_context_future_or_past_outside_window_irrelevant():
    times = pd.date_range("2024-01-01", periods=600, freq="10min", tz="UTC")
    frame = pd.DataFrame({"station": "A", "time": times})
    for index, column in enumerate((*mod.BASE_COLUMNS, *mod.DIRECTION_COLUMNS)):
        frame[column] = 1.0 + index + np.arange(len(frame)) / 1000
    anchor = pd.Series({"station": "A", "anchor_time": times[400]})
    before = mod.summarize_context(mod.raw_context(frame, anchor))
    out = (frame.time > times[400]) | (frame.time < times[112])
    frame.loc[out, [*mod.BASE_COLUMNS, *mod.DIRECTION_COLUMNS]] = -1e6
    after = mod.summarize_context(mod.raw_context(frame, anchor))
    np.testing.assert_array_equal(list(before.values()), list(after.values()))


def test_budget_and_no_official_modes():
    cfg = json.loads(mod.CONFIG.read_text())
    assert cfg["budget"]["backbone_fits"] == 15 and cfg["budget"]["router_fits"] == 8
    assert cfg["budget"]["gpu"] == 0 and cfg["budget"]["wall_seconds"] == 5400
    assert not any(cfg["permissions"].values())
    assert sum(cfg["expected_validation"]) * 6 == cfg["expected_oof_rows"]
    assert cfg["onset"] == "NOT_ENABLED"
    assert not any("oof" in path or "models/" in path for path in cfg["inputs"])


def test_preexisting_output_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "OUT", tmp_path)
    with pytest.raises(FileExistsError):
        mod.preflight({}, tmp_path)
