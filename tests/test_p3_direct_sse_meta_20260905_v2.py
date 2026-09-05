import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/run_p3_direct_sse_meta_20260905_v2.py"
SPEC = importlib.util.spec_from_file_location("p3_direct_sse_test", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def frame():
    rng = np.random.default_rng(7)
    data = pd.DataFrame({"anchor_id": np.repeat(np.arange(9), 6), "station": "G-ORS", "lead_h": np.tile(M.LEADS, 9), "fold": np.repeat(["a", "b", "c"], 18)})
    for col in M.COMPONENTS:
        data[col] = rng.uniform(1.5, 3, len(data))
    data["target_hs"] = data[M.COMPONENTS].to_numpy() @ [0.2, 0.3, 0.5]
    data["final_prediction"] = data.single_prediction * 0.5 + data.multi_prediction * 0.5
    return data


def test_simplex_recovers_known_weights():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 3))
    w = M.simplex_fit(x, x @ [0.2, 0.3, 0.5])
    np.testing.assert_allclose(w, [0.2, 0.3, 0.5], atol=1e-10)


def test_simplex_vertex_and_duplicate_columns():
    x = np.column_stack([np.arange(20), np.arange(20), np.zeros(20)])
    w = M.simplex_fit(x, np.zeros(20))
    assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-12
    np.testing.assert_allclose(x @ w, 0, atol=1e-9)


def test_simplex_rejects_invalid_input():
    with pytest.raises(ValueError):
        M.simplex_fit(np.full((3, 3), np.nan), np.ones(3))


def test_first_fold_noop_and_short_leads_untouched():
    data = frame()
    out, fits = M.prequential(data, ["a", "b", "c"])
    for policy in out:
        np.testing.assert_array_equal(out[policy][:18], data.final_prediction.to_numpy()[:18])
    short = data.lead_h.isin([3, 6, 9]).to_numpy()
    np.testing.assert_array_equal(out["long_simplex"][short], data.final_prediction.to_numpy()[short])
    assert len(fits) == 4
    assert [r["fit_cases"] for r in fits] == [3, 3, 6, 6]


def test_future_truth_never_changes_past_or_current_predictions():
    data = frame()
    before, fits1 = M.prequential(data, ["a", "b", "c"])
    mutated = data.copy()
    mutated.loc[mutated.fold.eq("c"), "target_hs"] += 50
    after, fits2 = M.prequential(mutated, ["a", "b", "c"])
    for policy in before:
        np.testing.assert_array_equal(before[policy], after[policy])
    assert fits1 == fits2


def test_bias_is_past_mean_and_range_clip():
    data = frame()
    param = M.fit_policy(data, "global_bias")
    assert param["bias_m"] == pytest.approx((data.target_hs - data.final_prediction).mean())
    pred = M.apply_policy(data, {"policy": "global_bias", "bias_m": 100})
    np.testing.assert_array_equal(pred, np.full(len(data), 30.0))


def test_pooled_rmse_not_mean_fold_rmse():
    y = np.array([0.0, 0, 0, 0, 10])
    p = np.zeros(5)
    assert M.rmse(y, p) == np.sqrt(20)
    assert M.rmse(y, p) != (M.rmse(y[:4], p[:4]) + M.rmse(y[4:], p[4:])) / 2
