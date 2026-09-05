"""Synthetic-only contract and leakage regression checks; no distributed data I/O."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p1_repair", ROOT / "scripts/run_p1_score_repair_20260905_v1.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
CONFIG = json.loads((ROOT / "configs/experiments/p1_score_repair_20260905_v1.json").read_text())


def sample(n=100):
    return pd.DataFrame(
        {
            "station": "S-ORS",
            "year": 2025,
            "layer": 1,
            "time": pd.date_range("2025-01-01", periods=n, freq="10min", tz="Asia/Seoul").astype(
                str
            ),
            "temp": np.sin(np.arange(n) / 7),
            "psal": np.cos(np.arange(n) / 7),
            "depth": 10.0,
            "label": 0,
            "anomaly_type": "",
            "row_id": np.arange(n),
        }
    )


def short_config():
    return {**CONFIG, "flank_inner_hours": 1, "flank_outer_hours": 4}


def test_metric_counts_and_micro():
    result = runner.metric([1, 1, 0, 0], [1, 0, 1, 0])
    assert result["tp"] == result["fp"] == result["fn"] == 1
    assert result["f1"] == 0.5


def test_gap_segments():
    frame = sample().drop(index=30).reset_index(drop=True)
    assert runner.segments(frame).nunique() == 2


def test_flank_excludes_center():
    frame = sample()
    cfg = short_config()
    before = runner.flank_features(frame, cfg)
    altered = frame.copy()
    altered.loc[45:55, "temp"] += 100
    after = runner.flank_features(altered, cfg)
    # delta changes by current value; the flank medians remain unchanged.
    for side in ["left", "right"]:
        column = f"flank24_168_temp_{side}_delta"
        assert after.loc[50, column] - before.loc[50, column] == pytest.approx(100, abs=1e-5)


def test_flank_cannot_cross_gap():
    frame = sample().drop(index=49).reset_index(drop=True)
    before = runner.flank_features(frame, short_config())
    altered = frame.copy()
    altered.loc[:48, "temp"] += 1000
    after = runner.flank_features(altered, short_config())
    np.testing.assert_allclose(before.iloc[49:], after.iloc[49:], equal_nan=True)


def test_flank_labels_irrelevant():
    frame = sample()
    before = runner.flank_features(frame, short_config())
    frame["label"], frame["anomaly_type"] = 1, "offset"
    pd.testing.assert_frame_equal(before, runner.flank_features(frame, short_config()))


def test_fragmentation_label_independent():
    cfg = {**CONFIG, "fragmentation": {**CONFIG["fragmentation"], "start_probability": 0.1}}
    frame = sample()
    _, first = runner.fragmented(frame, cfg)
    frame["label"] = 1
    _, second = runner.fragmented(frame, cfg)
    np.testing.assert_array_equal(first, second)
    assert (~first).any()


def test_train_slice_retreats_without_future_truth():
    frame = sample()
    frame.loc[35:55, "label"] = 1
    cutoff = pd.Timestamp(frame.time.iloc[50])
    first = runner.train_slice(frame, cutoff)
    frame.loc[50:, "label"] = 0
    second = runner.train_slice(frame, cutoff)
    pd.testing.assert_frame_equal(first, second)
    assert first.row_id.max() == 34


def test_feature_statistics_use_training_depth():
    training = sample()
    evaluation = sample()
    evaluation["depth"] = 999
    stats = runner.stats_fit(training)
    base, extended = runner.feature_pair(evaluation, stats, short_config())
    assert base.frame.nominal_depth_m.eq(10).all()
    assert base.frame.depth_raw.eq(999).all()
    assert len(extended.feature_columns) == len(base.feature_columns) + 25
    assert not {"label", "anomaly_type", "row_id"} & set(extended.feature_columns)


def test_calibration_union_is_explicit_and_inner_only():
    frame = sample(20)
    frame.loc[10:, "label"] = 1
    probability = np.r_[np.repeat(0.1, 10), np.repeat(0.9, 10)]
    rules = (np.zeros(20, bool), np.zeros(20, bool))
    cfg = {**CONFIG, "minimum_positive_run": 1}
    _, single = runner.calibrate(frame, probability, rules, cfg)
    _, union = runner.calibrate(frame, probability, rules, cfg, np.ones(20, np.int8))
    assert runner.metric(frame.label, single)["f1"] == 1
    assert union.min() == 1
    assert (single < union).any()


def test_old_station_router_is_not_eligible():
    code = (ROOT / "scripts/run_p1_score_repair_20260905_v1.py").read_text(encoding="utf-8")
    assert "historical_router" not in code


def test_diagnostic_counts_partial_events_without_changing_rows():
    frame = sample(20)
    frame.loc[5:10, "label"] = 1
    bits = np.zeros(20, dtype=np.int8)
    bits[5] = 1
    result = runner.diagnostic(frame, bits, np.zeros(20, dtype=np.int8), bits * .8)
    assert result["rows"] == 20
    assert result["partial_event_missing_rows"] == 5
    assert result["added_tp"] == 1


def test_fit_contract_and_official_boundary():
    assert CONFIG["screen_max_lgbm_fits"] == 12
    assert CONFIG["screen_max_xgboost_fits"] == 6
    assert CONFIG["training_input"] == "train.csv"
    assert not CONFIG["official_access_authorized"]
    assert CONFIG["threads"] == 4
    code = (ROOT / "scripts/run_p1_score_repair_20260905_v1.py").read_text(encoding="utf-8")
    assert "test.csv" not in code and "sample_submission.csv" not in code
    assert "read_parquet" not in code
