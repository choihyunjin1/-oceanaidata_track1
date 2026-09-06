"""Synthetic only: no training, datasets, model reloads or official inputs."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/p1_champion_reconstruction_20260906_v1/tree.py"
SPEC = importlib.util.spec_from_file_location("champion_tree", PATH)
t = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(t)


def frame(n=130):
    return pd.DataFrame({"station": "S-ORS", "year": 2025, "layer": 1,
        "time": pd.date_range("2025-01-01", periods=n, freq="10min", tz="Asia/Seoul").astype(str),
        "temp": np.sin(np.arange(n) / 9), "psal": np.cos(np.arange(n) / 7),
        "depth": 10.0, "label": np.r_[np.ones(20, dtype=int), np.zeros(n - 20, dtype=int)],
        "row_id": np.arange(n)})


def test_recipe_and_budget_exact():
    c = t.config()
    assert 3 * 2 * (1 + len(c["B_seeds"])) == c["historical_fits"] == 24
    assert c["full_fits"] == 4 and c["threads"] == 4 and not c["gpu"]
    assert c["B_seeds"] == [20260813, 20260829, 20260847]
    original = t.core.load_config(ROOT / "configs/p1.toml", env={})
    assert c["xgboost_parameters"] == original.raw["models"]["xgboost"]
    recipe = json.loads((ROOT / "configs/p1_meaningful_learning_curve_generation_v1.json").read_text())
    assert c["lightgbm_parameters"] == recipe["lightgbm_parameters"]
    for seed in c["B_seeds"]:
        p = t.parameters(c, "B", seed)
        assert p["n_estimators"] == 700 and p["num_leaves"] == 63 and p["min_child_samples"] == 60
        assert p["n_jobs"] == 4 and p["deterministic"] and p["force_row_wise"]
        assert all(p[k] == seed for k in ["random_state", "feature_fraction_seed", "bagging_seed",
                                        "data_random_seed", "extra_seed"])


def test_active_feature_parity_label_independence_and_depth():
    f = frame()
    cfg = t.config()
    stats = t.core.stats_fit(f)
    actual = t.base_features(f, stats, cfg)
    expected = t.core.feature_pair(f, stats, {**cfg, "flank_outer_hours": 168,
                                    "flank_inner_hours": 24, "flank_min_fraction": .25})[0]
    pd.testing.assert_frame_equal(actual.frame, expected.frame)
    assert actual.feature_columns == expected.feature_columns and len(actual.feature_columns) == 80
    f.label = 1 - f.label
    pd.testing.assert_frame_equal(actual.frame, t.base_features(f, stats, cfg).frame)
    f.depth = 900
    assert t.base_features(f, stats, cfg).frame.nominal_depth_m.eq(10).all()
    f.year = 2026
    assert t.base_features(f, stats, cfg).frame.nominal_depth_m.isna().all()


def test_partition_sentinel_cannot_enter_features_encoder_rules():
    train = frame()
    allowed = frame().assign(time=lambda x: pd.to_datetime(x.time) + pd.Timedelta(days=40))
    outside = frame().assign(time=lambda x: pd.to_datetime(x.time) + pd.Timedelta(days=80))
    complete = pd.concat([allowed, outside], ignore_index=True)
    boundary = pd.Timestamp("2025-03-01", tz="Asia/Seoul")

    def extract(source):
        valid = source.loc[pd.to_datetime(source.time) < boundary].reset_index(drop=True)
        stats = t.core.stats_fit(train)
        tb = t.base_features(train, stats, t.config())
        vb = t.base_features(valid, stats, t.config())
        encoder = t.core.TabularEncoder().fit(tb, np.arange(len(train)))
        return encoder.transform(vb), t.core.rule_masks(valid, stats)

    a, ar = extract(complete)
    complete.loc[len(allowed):, ["temp", "psal", "depth", "label"]] = 100000
    b, br = extract(complete)
    np.testing.assert_array_equal(a, b)
    for x, y in zip(ar, br, strict=True):
        np.testing.assert_array_equal(x, y)


def toy_split():
    f = frame(130)
    f.label = 0
    f.loc[2:4, "label"] = 1
    f.loc[64:68, "label"] = 1  # crosses start: owned by earlier region; excluded here
    f.loc[84:92, "label"] = 1  # crosses end: belongs wholly to validation
    cfg = {**t.config(), "purge_days": 1 / 144, "inner_days": 20 / 144,
        "folds": [{"id": "toy", "start": f.time.iloc[66], "end": f.time.iloc[90]}]}
    return f, cfg


def test_run_start_ownership_excludes_crossing_and_keeps_end_run():
    f, cfg = toy_split()
    masks, _ = t.split_masks(f, cfg)
    train, valid = masks[("toy", "outer")]
    assert not valid[64:69].any() and valid[84:93].all()
    assert not train[64:69].any() and not (train & valid).any()
    assert not valid[93:].any()


def test_split_contract_matches_active_helper_run_formulas():
    f, cfg = toy_split()
    masks, support = t.split_masks(f, cfg)
    assert len(masks) == len(support) == 2
    assert support[1]["validation_rows"] == int(masks[("toy", "outer")][1].sum())
    assert support[1]["train_cutoff_exclusive"] == (
        pd.Timestamp(cfg["folds"][0]["start"]) - pd.Timedelta(days=cfg["purge_days"])).isoformat()


def test_empty_support_and_duplicate_keys_fail_closed():
    f, cfg = toy_split()
    with pytest.raises(ValueError):
        t.split_masks(f.assign(label=0), cfg)
    with pytest.raises(ValueError):
        t.split_masks(pd.concat([f, f.iloc[:1]], ignore_index=True), cfg)


def test_generic_cells_tie_B_and_unseen_B(monkeypatch):
    f = frame(40)
    y = f.label.to_numpy()
    monkeypatch.setattr(t.core, "calibrate", lambda frame, p, rules, cfg:
                        ({"threshold": .2}, p.astype(np.int8)))
    monkeypatch.setattr(t.core, "decode", lambda frame, p, rules, cfg, threshold: p.astype(np.int8))
    policy = t.select_policy(f, {"O": y, "B": y}, None, t.config())
    assert policy["cells"][0]["policy"] == "B"
    original = y.copy()
    original[0] = 0
    policy = t.select_policy(f, {"O": original, "B": y}, None, t.config())
    assert policy["cells"][0]["policy"] == "B"
    out = t.predict_policy(f.assign(station="NEW"), {"O": original, "B": y}, None, t.config(), policy)
    np.testing.assert_array_equal(out["router"], y)
    assert set(out) == set(t.ARMS)


def test_inner_policy_invariant_to_outer_labels(monkeypatch):
    f = frame(40)
    monkeypatch.setattr(t.core, "calibrate", lambda frame, p, rules, cfg:
                        ({"threshold": .2}, p.astype(np.int8)))
    monkeypatch.setattr(t.core, "decode", lambda frame, p, rules, cfg, threshold: p.astype(np.int8))
    probs = {"O": f.label.to_numpy(), "B": np.zeros(len(f), dtype=np.int8)}
    selector = t.select_policy(f, probs, None, t.config())
    first = t.predict_policy(f, probs, None, t.config(), selector)
    second = t.predict_policy(f.assign(label=1 - f.label), probs, None, t.config(), selector)
    for arm in t.ARMS:
        np.testing.assert_array_equal(first[arm], second[arm])


@pytest.mark.parametrize("y,p", [([], []), ([0, 1], [0]), ([0, 1], [0, np.nan]), ([0, 1], [0, 2])])
def test_metric_rejects_invalid(y, p):
    with pytest.raises(ValueError):
        t.metric(y, p)


def test_weights_permutation_event_day_and_class_normalization():
    f = frame()
    w = t.core._event_day_weight(f, f.label.to_numpy())
    assert np.isfinite(w).all() and (w > 0).all()
    assert w[f.label == 0].mean() == pytest.approx(1)
    assert w[f.label == 1].mean() == pytest.approx(np.sqrt(110 / 20))
    order = np.random.default_rng(8).permutation(len(f))
    shuffled = f.iloc[order].reset_index(drop=True)
    np.testing.assert_array_equal(t.core._event_day_weight(shuffled, shuffled.label), w[order])


def test_no_official_or_legacy_full_entrypoints():
    source = PATH.read_text()
    assert "test.csv" not in source and "sample_submission.csv" not in source
    assert "_full_fit_and_reproduce(" not in source
    assert "to_csv(" not in source and "--worker" in source
    assert t.config()["deployment_selector"] == "2025_q4_inner"


def test_isolated_cli_help_from_unrelated_directory(tmp_path):
    result = subprocess.run([sys.executable, "-I", str(PATH), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "preflight" in result.stdout
