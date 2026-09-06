"""Synthetic contract checks, no distributed inputs or model fits."""

import importlib.util
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/p1_core_tuning_20260906_v1/run.py"
SPEC = importlib.util.spec_from_file_location("p1_core_tuning", PATH)
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def frame(n=130):
    return pd.DataFrame({"station": "S-ORS", "year": 2025, "layer": 1,
        "time": pd.date_range("2025-01-01", periods=n, freq="10min", tz="Asia/Seoul").astype(str),
        "temp": np.sin(np.arange(n) / 9), "psal": np.cos(np.arange(n) / 7),
        "depth": 10., "label": np.r_[np.ones(20, dtype=int), np.zeros(n - 20, dtype=int)],
        "row_id": np.arange(n)})


def test_actual_fit_budget_for_every_possible_selection():
    c, b = r.config(), r.base_config()
    counts = [r.fit_count(p, c, b) for p in itertools.product(c["policies"], repeat=3)]
    assert min(counts) == 39 and max(counts) == 48
    assert sum(len(r.seeds(k, b)) for k in c["components"]) == 9
    assert sum(len(r.seeds(k, b)) for k in r.BASE_COMPONENTS) == 4
    assert r.outer_components("control", c) == list(r.BASE_COMPONENTS)
    assert "B_slow" not in c["components"] and not c["baseline_reuse"]


def test_parameter_differences_are_exact_and_cpu2():
    c, b = r.config(), r.base_config()
    for name, spec in c["components"].items():
        for seed in r.seeds(name, b):
            p = r.parameters(name, seed, b, c)
            baseline = r.t.parameters(b, spec["arm"], seed)
            assert p == {**baseline, **spec["overrides"]}
            assert p["n_estimators"] == (1400 if name == "O_slow" else 700)
            if spec["arm"] == "B":
                assert p["n_jobs"] == 2
                assert all(p[k] == seed for k in ("random_state", "feature_fraction_seed",
                           "bagging_seed", "data_random_seed", "extra_seed"))
    assert r.seeds("B_regular", b) == [20260813, 20260829, 20260847]


def test_loaded_native_threads_capped():
    with threadpool_limits(limits=2):
        assert r.check_threads() and all(x["num_threads"] <= 2 for x in r.check_threads())


def test_policy_tie_control_and_outer_label_invariance(monkeypatch):
    f = frame(40)
    c, b = r.config(), r.base_config()
    monkeypatch.setattr(r.core, "calibrate", lambda frame, p, rules, cfg:
                        ({"threshold": .2}, p.astype(np.int8)))
    monkeypatch.setattr(r.core, "decode", lambda frame, p, rules, cfg, threshold: p.astype(np.int8))
    probs = {component: f.label.to_numpy() for component in c["components"]}
    chosen = r.select_inner(f, probs, None, b, c)
    assert chosen["selected"] == "control"
    first = r.policy_bits(f, probs, None, b, c, chosen)
    second = r.policy_bits(f.assign(label=1 - f.label), probs, None, b, c, chosen)
    for arm in first:
        np.testing.assert_array_equal(first[arm], second[arm])


def test_inner_policy_selects_complete_union_not_component(monkeypatch):
    f = frame(40)
    c, b = r.config(), r.base_config()
    monkeypatch.setattr(r.core, "calibrate", lambda frame, p, rules, cfg:
                        ({"threshold": .3}, p.astype(np.int8)))
    monkeypatch.setattr(r.core, "decode", lambda frame, p, rules, cfg, threshold: p.astype(np.int8))
    p = {name: np.zeros(len(f), dtype=np.int8) for name in c["components"]}
    p["B_regular"] = f.label.to_numpy()
    chosen = r.select_inner(f, p, None, b, c)
    assert chosen["selected"] == "B_regular"
    np.testing.assert_array_equal(r.policy_bits(f, p, None, b, c, chosen)["candidate"], f.label)


def test_threshold_tie_remains_higher(monkeypatch):
    f = frame(40)
    monkeypatch.setattr(r.core, "decode", lambda frame, p, rules, cfg, threshold: p.astype(np.int8))
    p = {name: f.label.to_numpy() for name in r.config()["components"]}
    selected = r.select_inner(f, p, None, r.base_config(), r.config())
    assert set(selected["thresholds"].values()) == {.8}


def test_exact_active_features_and_no_label_effect():
    f, b = frame(), r.base_config()
    stats = r.core.stats_fit(f)
    actual = r.t.base_features(f, stats, b)
    reference = r.core.feature_pair(f, stats, {**b, "flank_outer_hours": 168,
                              "flank_inner_hours": 24, "flank_min_fraction": .25})[0]
    pd.testing.assert_frame_equal(actual.frame, reference.frame)
    assert len(actual.feature_columns) == 80
    pd.testing.assert_frame_equal(actual.frame, r.t.base_features(f.assign(label=1 - f.label), stats, b).frame)
    assert r.t.base_features(f.assign(year=2026), stats, b).frame.nominal_depth_m.isna().all()


def test_partition_feature_encoder_rule_sentinel():
    training = frame()
    valid = frame().assign(time=lambda x: pd.to_datetime(x.time) + pd.Timedelta(days=40))
    outside = frame().assign(time=lambda x: pd.to_datetime(x.time) + pd.Timedelta(days=80))
    whole = pd.concat([valid, outside], ignore_index=True)
    b = r.base_config()
    def extract(source):
        permitted = source.loc[pd.to_datetime(source.time) < pd.Timestamp("2025-03-01", tz="Asia/Seoul")].reset_index(drop=True)
        stats = r.core.stats_fit(training)
        encoder = r.core.TabularEncoder().fit(r.t.base_features(training, stats, b), np.arange(len(training)))
        return encoder.transform(r.t.base_features(permitted, stats, b)), r.core.rule_masks(permitted, stats)
    x, rules = extract(whole)
    whole.loc[len(valid):, ["temp", "psal", "depth", "label"]] = 999999
    x2, rules2 = extract(whole)
    np.testing.assert_array_equal(x, x2)
    for a, z in zip(rules, rules2, strict=True):
        np.testing.assert_array_equal(a, z)


def test_whole_run_boundary_ownership_and_purge():
    f = frame()
    f.label = 0
    f.loc[2:4, "label"] = 1
    f.loc[64:68, "label"] = 1
    f.loc[84:92, "label"] = 1
    b = {**r.base_config(), "purge_days": 1 / 144, "inner_days": 20 / 144,
         "folds": [{"id": "toy", "start": f.time.iloc[66], "end": f.time.iloc[90]}]}
    masks, _ = r.t.split_masks(f, b)
    train, valid = masks[("toy", "outer")]
    assert not valid[64:69].any() and valid[84:93].all()
    assert not train[64:69].any() and not (train & valid).any()


@pytest.mark.parametrize("y,p", [([], []), ([0, 1], [0]), ([0, 1], [0, np.nan]), ([0, 1], [0, 2])])
def test_metric_fail_closed(y, p):
    with pytest.raises(ValueError):
        r.t.metric(y, p)


def test_source_no_official_or_legacy_full_paths():
    source = PATH.read_text()
    assert "test.csv" not in source and "sample_submission.csv" not in source
    assert "to_csv(" not in source and "_full_fit_and_reproduce(" not in source
    assert "timer" in source and '"/T", "/F"' in source


def test_isolated_entrypoint(tmp_path):
    result = subprocess.run([sys.executable, "-I", str(PATH), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "preflight" in result.stdout and "replay" in result.stdout
