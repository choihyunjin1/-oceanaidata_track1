"""Synthetic-only validation of the P3 weather ablation contract."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_wave.features import BASE_COLUMNS, DIRECTION_COLUMNS, summarize_context

PATH = Path(__file__).resolve().parents[1] / "scripts/run_p3_score_repair_20260905_v1.py"
SPEC = importlib.util.spec_from_file_location("p3_score_repair", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def context():
    rng = np.random.default_rng(72)
    frame = pd.DataFrame({c: rng.uniform(1, 4, 289) for c in BASE_COLUMNS + DIRECTION_COLUMNS})
    frame.loc[::2, ["hs", "tp", "hmax", "wvdir"]] = np.nan
    frame.loc[200:230, ["wspd", "gust", "wdir"]] = np.nan
    return frame


def test_mask_matches_raw_context_recomputation_all_1275_features():
    raw = context()
    before = pd.DataFrame([summarize_context(raw)])
    raw.loc[:, ["wspd", "gust", "wdir", "airt", "relh", "caph"]] = np.nan
    after = pd.DataFrame([summarize_context(raw)])
    pd.testing.assert_frame_equal(MODULE.mask_weather(before), after)


def test_mask_preserves_wave_and_residual_fields():
    frame = pd.DataFrame([summarize_context(context())])
    frame["current_hs_for_residual"] = 2.0
    frame["target_hs"] = 3.0
    masked = MODULE.mask_weather(frame)
    cols = [c for c in frame if c not in MODULE.weather_columns(frame)]
    pd.testing.assert_frame_equal(frame[cols], masked[cols])


def test_weights_target_and_six_leads_preserved():
    frame = pd.concat([pd.DataFrame([summarize_context(context())])] * 6, ignore_index=True)
    frame["lead_h"] = [3, 6, 9, 12, 18, 24]
    target = np.arange(6, dtype=float)
    weight = np.full(6, 2.0)
    augmented, y, weights, receipt = MODULE.augment_weather(frame, target, weight)
    assert len(augmented) == 12
    np.testing.assert_array_equal(y[:6], y[6:])
    np.testing.assert_array_equal(weights[:6] + weights[6:], weight)
    assert receipt["observed_rows"] == 6
    assert not MODULE.weather_observed(augmented.iloc[6:]).any()


def test_unobserved_rows_not_duplicated():
    frame = MODULE.mask_weather(pd.DataFrame([summarize_context(context())]))
    result, target, weight, _ = MODULE.augment_weather(frame, np.array([1.0]), np.array([2.0]))
    assert len(result) == 1 and target[0] == 1.0 and weight[0] == 2.0


def small_oof():
    return pd.DataFrame(
        {
            "fold": ["a", "a"],
            "anchor_id": [1, 2],
            "station": ["G", "G"],
            "lead_h": [3, 3],
            "target_hs": [1.0, 2.0],
            "p": [1.2, 2.2],
        }
    )


def test_alignment_reorders_but_rejects_population_or_truth_mismatch():
    ref = small_oof()
    np.testing.assert_allclose(MODULE.aligned(ref, ref.iloc[::-1], ["p"]).p, ref.p)
    with pytest.raises(ValueError, match="population"):
        MODULE.aligned(ref, ref.iloc[:1], ["p"])
    corrupt = ref.copy()
    corrupt.loc[0, "target_hs"] += 0.1
    with pytest.raises(ValueError, match="truth"):
        MODULE.aligned(ref, corrupt, ["p"])


def test_alignment_rejects_duplicate_and_nonfinite():
    ref = small_oof()
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.aligned(ref, pd.concat([ref, ref]), ["p"])
    corrupt = ref.copy()
    corrupt.loc[0, "p"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        MODULE.aligned(ref, corrupt, ["p"])


def test_pooled_rmse_not_mean_fold_rmse():
    frame = small_oof()
    frame.loc[1, "fold"] = "b"
    metric = MODULE.sliced(frame, np.array([1.0, 4.0]))
    assert metric["rmse"] == pytest.approx(np.sqrt(2.0))
    assert metric["rmse"] != np.mean(list(metric["by_fold"].values()))
