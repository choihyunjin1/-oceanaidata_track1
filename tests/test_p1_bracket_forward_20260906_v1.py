"""Synthetic contracts only; no distributed data reads or model fitting."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "p1_bracket_v1", ROOT / "scripts/run_p1_bracket_forward_20260906_v1.py"
)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


@pytest.fixture
def contract():
    return m.load_contract()


def toy(n=1200, begin="2025-01-01"):
    t = np.arange(n)
    return pd.DataFrame(
        {
            "station": "S-ORS",
            "year": pd.Timestamp(begin).year,
            "layer": 1,
            "time": pd.date_range(begin, periods=n, freq="10min", tz="Asia/Seoul").astype(str),
            "temp": 10.0 + np.sin(t / 20),
            "psal": 32.0 + 0.1 * np.sin(t / 20),
            "depth": 5.0,
        }
    )


def test_sealed_budget_no_official(contract):
    cfg, _, _ = contract
    assert cfg["fit_budget"] == {
        "determinism_full_original": 2,
        "baseline_inner_outer": 12,
        "bracket_inner_outer": 6,
    }
    assert cfg["threads"] == 2 and cfg["max_fits"] == 20 and not cfg["gpu"]
    assert cfg["wall_cap_seconds"] == 5400
    assert not any(cfg[k] for k in ("official_access", "csv_generation", "upload"))
    assert not cfg["determinism"]["quality_metric"]


def test_bracket_27_features_no_target(contract):
    cfg, _, _ = contract
    frame = toy()
    features = m.bracket_features(frame, cfg)
    assert features.shape == (len(frame), 27)
    assert not set(features) & {"label", "anomaly_type", "station", "year", "layer"}
    with pytest.raises(ValueError, match="raw observation"):
        m.bracket_features(frame.assign(label=0), cfg)


def test_bracket_radius_37h_by_sentinel(contract):
    cfg, _, _ = contract
    frame = toy()
    center, radius = 600, 37 * 6
    changed = frame.copy()
    changed.loc[
        (np.arange(len(frame)) < center - radius) | (np.arange(len(frame)) > center + radius),
        ["temp", "psal", "depth"],
    ] = 999.0
    before, after = m.bracket_features(frame, cfg), m.bracket_features(changed, cfg)
    np.testing.assert_array_equal(before.iloc[center], after.iloc[center])
    changed.loc[center - radius, "temp"] = -999.0
    # Time coverage uses exactly the preregistered +/-37h footprint, not row-dropping compression.
    assert before.loc[center, "bracket_72h_coverage"] == 1.0


def test_gap_and_other_station_cannot_change_segment(contract):
    cfg, _, _ = contract
    first = toy(800)
    second = toy(500, "2025-02-01")
    frame = pd.concat([first, second], ignore_index=True)
    changed = frame.copy()
    changed.loc[800:, "temp"] += 9999
    pd.testing.assert_frame_equal(
        m.bracket_features(frame, cfg).iloc[:800], m.bracket_features(changed, cfg).iloc[:800]
    )
    other = second.assign(station="Z-ORS", layer=5)
    joined = pd.concat([first, other], ignore_index=True)
    pd.testing.assert_frame_equal(
        m.bracket_features(joined, cfg).iloc[:800], m.bracket_features(first, cfg)
    )


def test_offset_bracket_entry_exit_and_return(contract):
    cfg, _, _ = contract
    raw = toy(800)
    raw["temp"] = 10.0
    center = 400
    raw.loc[center - 18 : center + 18, "temp"] = 12.0
    feat = m.bracket_features(raw, cfg)
    assert feat.loc[center, "bracket_6h_entry_jump"] == 2
    assert feat.loc[center, "bracket_6h_exit_jump"] == -2
    assert feat.loc[center, "bracket_6h_jump_cancellation"] == 0
    assert feat.loc[center, "bracket_6h_outside_return_abs"] == 0
    assert feat.loc[center, "bracket_6h_interior_minus_left"] == 2


def test_original80_equivalence_to_canonical(contract):
    cfg, frozen, _ = contract
    raw = toy()
    stats = m.old.stats_fit(raw)
    own = m.base_bundle(raw, stats, frozen)
    old = m.old.feature_pair(raw, stats, frozen)[0]
    assert own.frame.shape[1] == 80
    pd.testing.assert_frame_equal(own.frame, old.frame)
    assert m.bundle(raw, stats, frozen, cfg, True).frame.shape[1] == 107


def test_base_context_bounded_with_fixed_train_stats(contract):
    _, frozen, _ = contract
    frame = toy(10000)
    center = 5000
    stats = m.old.stats_fit(frame)
    changed = frame.copy()
    # Conservative +/-15-day envelope includes nested rolling and capped plateau evidence.
    outside = (np.arange(len(frame)) < center - 15 * 144) | (
        np.arange(len(frame)) > center + 15 * 144
    )
    changed.loc[outside, ["temp", "psal", "depth"]] = 9999.0
    before = m.base_bundle(frame, stats, frozen).frame.iloc[[center]]
    after = m.base_bundle(changed, stats, frozen).frame.iloc[[center]]
    pd.testing.assert_frame_equal(before, after)


@pytest.mark.parametrize("bracket", [False, True])
def test_partition_sentinels_cannot_reach_stats_encoder_features_rules_decoder(contract, bracket):
    cfg, frozen, _ = contract
    frame = pd.concat(
        [toy(400, "2024-01-01"), toy(400, "2024-06-01"), toy(400, "2025-01-01")], ignore_index=True
    )
    frame["label"] = np.tile(([0] * 180 + [1] * 40 + [0] * 180), 3)
    frame["anomaly_type"] = np.where(frame.label.eq(1), "offset", "")
    train = np.arange(1200) < 400
    val = np.arange(1200) >= 800
    altered = frame.copy()
    altered.loc[~(train | val), ["temp", "psal", "depth"]] = 99999.0
    altered.loc[~(train | val), "station"] = "UNSEEN_SENTINEL"
    a = m.partition_inputs(frame, train, val, frozen, cfg, bracket)
    b = m.partition_inputs(altered, train, val, frozen, cfg, bracket)
    assert a["stats"] == b["stats"]
    assert a["encoder"].category_maps == b["encoder"].category_maps
    np.testing.assert_array_equal(a["x"], b["x"])
    np.testing.assert_array_equal(a["xe"], b["xe"])
    for ar, br in zip(a["rules"], b["rules"], strict=True):
        np.testing.assert_array_equal(ar, br)
    probability = np.linspace(0.05, 0.9, len(a["validation"]))
    np.testing.assert_array_equal(
        m.old.decode(a["validation"], probability, a["rules"], frozen, 0.2),
        m.old.decode(b["validation"], probability, b["rules"], frozen, 0.2),
    )
    # Outer observations/labels cannot affect fitted stats/encoder or training features.
    altered.loc[val, "temp"] += 111
    altered.loc[val, "label"] = 1 - altered.loc[val, "label"]
    c = m.partition_inputs(altered, train, val, frozen, cfg, bracket)
    assert a["stats"] == c["stats"] and a["encoder"].category_maps == c["encoder"].category_maps
    np.testing.assert_array_equal(a["x"], c["x"])


def test_hysteresis_has_partition_not_21day_radius(contract):
    _, frozen, _ = contract
    frame = toy(4000)
    probability = np.full(4000, 0.15)
    rule = (np.zeros(4000, dtype=bool), np.zeros(4000, dtype=bool))
    before = m.old.decode(frame, probability, rule, frozen, 0.2)
    probability[-1] = 0.3
    after = m.old.decode(frame, probability, rule, frozen, 0.2)
    assert before.sum() == 0 and after.sum() == 4000


def test_no_existing_model_inputs_or_official_code():
    source = (ROOT / "scripts/run_p1_bracket_forward_20260906_v1.py").read_text(encoding="utf-8")
    assert ' / "test.csv"' not in source and ' / "sample_submission.csv"' not in source
    assert ".to_csv(" not in source
    assert "predict_proba(x[:4096])" in source


def test_all_forward_inner_masks_have_ns_boundaries_and_no_future_train(contract):
    cfg, _, ev = contract
    timestamps = pd.date_range("2024-01-01", "2025-12-31", freq="2D", tz="Asia/Seoul")
    raw = toy(len(timestamps))
    raw["time"] = timestamps.astype(str)
    raw["year"] = timestamps.year
    raw["label"] = (np.arange(len(raw)) % 5 == 0).astype(int)
    masks, support = m.all_split_masks(raw, ev, cfg)
    assert len(masks) == len(support) == 6
    for record in support:
        train, validation = masks[(record["fold"], record["stage"])]
        assert not np.any(train & validation)
        assert timestamps[train].max() < pd.Timestamp(record["train_cutoff_exclusive"])
        assert timestamps[validation].min() >= pd.Timestamp(record["validation_start"])
