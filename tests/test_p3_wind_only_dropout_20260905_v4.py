"""Synthetic-only contracts; no historical data, official input or GPU."""

import importlib.util
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from p3_wave.features import BASE_COLUMNS, DIRECTION_COLUMNS, summarize_context

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p3_wind_v4", ROOT / "scripts/run_p3_wind_only_dropout_20260905_v4.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def context():
    rng = np.random.default_rng(591)
    data = pd.DataFrame({c: rng.uniform(1, 4, 289) for c in BASE_COLUMNS + DIRECTION_COLUMNS})
    data.loc[::2, ["hs", "tp", "hmax", "wvdir"]] = np.nan
    data.loc[40:70, ["wspd", "gust", "wdir"]] = np.nan
    return data


def test_all1275_features_match_raw_wind_mask_recomputation():
    raw = context()
    before = pd.DataFrame([summarize_context(raw)])
    raw.loc[:, ["wspd", "gust", "wdir"]] = np.nan
    after = pd.DataFrame([summarize_context(raw)])
    pd.testing.assert_frame_equal(M.mask_wind(before), after)
    assert len(before.columns) == 1275


def test_nonwind_and_wave_fields_unchanged():
    before = pd.DataFrame([summarize_context(context())])
    before["target_hs"] = 3.0
    after = M.mask_wind(before)
    columns = [c for c in before if c not in M.wind_columns(before)]
    pd.testing.assert_frame_equal(before[columns], after[columns])
    assert any(c.startswith("caph_") for c in columns)
    assert after.caph_current.notna().all()
    assert "gust_excess_current" in M.wind_columns(before)


def test_mixed_missing_six_leads_preserve_each_source_mass_and_target():
    observed = pd.concat([pd.DataFrame([summarize_context(context())])] * 6, ignore_index=True)
    observed["lead_h"] = M.LEADS
    missing = M.mask_wind(observed)
    frame = pd.concat([observed, missing], ignore_index=True)
    y, weight = np.arange(12, dtype=float), np.linspace(.1, 2, 12)
    augmented, target, weights, receipt = M.augment_wind(frame, y, weight)
    assert len(augmented) == 18
    np.testing.assert_array_equal(target, np.r_[y, y[:6]])
    np.testing.assert_allclose(weights[:6] + weights[12:], weight[:6], rtol=0, atol=0)
    np.testing.assert_array_equal(weights[6:12], weight[6:12])
    assert not M.wind_observed(augmented.iloc[12:]).any()
    assert receipt["max_per_original_row_weight_error"] == 0


def test_unknown_categories_fail():
    with pytest.raises(ValueError, match="unknown"):
        M.categorical(pd.DataFrame({"station": ["X"], "lead_h": [3]}))


def test_no_wind_support_columns_fail():
    with pytest.raises(ValueError, match="support"):
        M.wind_observed(pd.DataFrame({"hs_current": [2.0]}))


def test_shape_mismatch_fails():
    with pytest.raises(ValueError, match="shape"):
        M.augment_wind(pd.DataFrame([summarize_context(context())]), np.zeros(2), np.ones(1))


def test_fixed_cpu_budget_and_no_public_fit():
    cfg = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    assert cfg["fit_budget"] == 6 and cfg["new_router_fits"] == 0
    assert len(cfg["windows"]) * len(cfg["arms"]) == 6
    assert cfg["model"]["n_jobs"] == 2 and cfg["model"]["device_type"] == "cpu"
    assert cfg["public_inverse"] == cfg["official_access"] == cfg["csv"] == cfg["upload"] == 0
    assert cfg["selection"]["policies"] == M.POLICIES


def test_synthetic_lgbm_fit_and_native_replay(tmp_path):
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"station": ["G-ORS"] * 60, "lead_h": M.LEADS * 10, "x": rng.normal(size=60)})
    matrix = M.categorical(frame)
    model = lgb.LGBMRegressor(n_estimators=3, n_jobs=2, device_type="cpu", verbosity=-1, min_child_samples=3)
    model.fit(matrix, rng.normal(size=60), categorical_feature=["station", "lead_h"])
    path = tmp_path / "synthetic.txt"
    model.booster_.save_model(str(path))
    np.testing.assert_array_equal(model.predict(matrix), lgb.Booster(model_file=str(path)).predict(matrix, num_threads=2))


def test_key_alignment_rejects_missing_duplicate_truth_mismatch():
    frame = pd.DataFrame({"fold": ["a", "a"], "anchor_id": [1, 2], "station": ["G", "G"], "lead_h": [3, 3], "target_hs": [1., 2.], "p": [1.1, 2.1]})
    np.testing.assert_array_equal(M.aligned(frame, frame.iloc[::-1], ["p"]).p, frame.p)
    with pytest.raises(ValueError, match="population"):
        M.aligned(frame, frame.iloc[:1], ["p"])
    with pytest.raises(ValueError, match="duplicate"):
        M.aligned(frame, pd.concat([frame, frame]), ["p"])
    corrupt = frame.copy()
    corrupt.loc[0, "target_hs"] = 2
    with pytest.raises(ValueError, match="truth"):
        M.aligned(frame, corrupt, ["p"])
