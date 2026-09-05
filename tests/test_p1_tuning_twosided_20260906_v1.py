"""Synthetic contracts only; four small native700/prefix1400 toy fits are counted separately."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_p1_tuning_twosided_20260906_v1 as m  # noqa: E402


@pytest.fixture
def contract():
    return m.contract()


def toy(n=1000, start="2024-01-01"):
    t = np.arange(n)
    dates = pd.date_range(start, periods=n, freq="10min", tz="Asia/Seoul")
    return pd.DataFrame({"station": "S-ORS", "year": dates.year, "layer": 1,
                         "time": dates.astype(str), "temp": 10 + np.sin(t / 20),
                         "psal": 32 + np.cos(t / 20) / 10, "depth": 5.0,
                         "label": (t % 150 < 20).astype(int), "anomaly_type": "offset"})


def test_contract_no_official_and_separate_diagnostic(contract):
    cfg, _, _, _ = contract
    assert cfg["outer_iterations"] == 700 and cfg["inner_max_iterations"] == 1400
    assert cfg["diagnostic_threshold"] == 0.5 and not cfg["diagnostic_changes_outer"]
    assert cfg["max_new_fits"] == 18 and cfg["threads"] == 2 and cfg["wall_cap_seconds"] == 3600
    assert not cfg["official_access"] and not cfg["csv_generation"] and not cfg["upload"]


@pytest.mark.parametrize("arm", ["original", "balanced"])
def test_native700_equals_1400_prefix_exact(contract, arm):
    cfg, _, frozen, _ = contract
    rng = np.random.default_rng(20260906)
    x = rng.normal(size=(1000, 8)).astype(np.float32)
    data = toy()
    data["label"] = (x[:, 0] + 0.3 * x[:, 1] > 0.7).astype(np.int8)
    short = m.fit(arm, x, data, frozen, cfg, 700)
    long = m.fit(arm, x, data, frozen, cfg, 1400)
    probe = rng.normal(size=(111, 8)).astype(np.float32)
    a, b = m.probability(short, probe, arm, 700), m.probability(long, probe, arm, 700)
    np.testing.assert_array_equal(a, b)
    # Correct native facade semantics must match the prefix path too.
    np.testing.assert_array_equal(a, short.predict_proba(probe)[:, 1])


def test_nested_twosided_purge_and_outer_exclusion(contract):
    cfg, _, _, ev = contract
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="2D", tz="Asia/Seoul")
    frame = toy(len(dates))
    frame["time"], frame["year"] = dates.astype(str), dates.year
    frame["label"] = (np.arange(len(frame)) % 7 == 0).astype(int)
    masks, counts = m.nested_masks(frame, ev, cfg)
    assert len(counts) == 6
    for fold in m.FOLDS:
        itr, iva = masks[(fold, "inner")]
        otr, ova = masks[(fold, "outer")]
        assert not (itr & iva).any() and not ((itr | iva) & ~otr).any()
        assert not ((itr | iva) & ova).any()
        row = next(x for x in counts if x["fold"] == fold and x["stage"] == "inner")
        left = pd.Timestamp(row["inner_start"]) - pd.Timedelta(days=21)
        right = pd.Timestamp(row["inner_end"]) + pd.Timedelta(days=21)
        assert ((dates[itr] < left) | (dates[itr] >= right)).all()
    assert dates[masks[("H1_2025", "inner")][0]].max() > pd.Timestamp("2025-07-22", tz="Asia/Seoul")


@pytest.mark.parametrize("bracket", [False, True])
def test_twosided_partition_sentinels(contract, bracket):
    _, prior, frozen, _ = contract
    frame = pd.concat([toy(400, start=d) for d in ("2024-01-01", "2024-06-01", "2024-08-01", "2025-02-01")], ignore_index=True)
    pos = np.arange(len(frame))
    train, val = (pos < 400) | (pos >= 1200), (pos >= 800) & (pos < 1200)
    altered = frame.copy()
    altered.loc[~(train | val), ["temp", "psal", "depth"]] = 9999.0
    a = m.base.partition_inputs(frame, train, val, frozen, prior, bracket)
    b = m.base.partition_inputs(altered, train, val, frozen, prior, bracket)
    assert a["stats"] == b["stats"] and a["encoder"].category_maps == b["encoder"].category_maps
    np.testing.assert_array_equal(a["x"], b["x"])
    np.testing.assert_array_equal(a["xe"], b["xe"])
    for ar, br in zip(a["rules"], b["rules"], strict=True):
        np.testing.assert_array_equal(ar, br)
    p = np.linspace(.05, .9, val.sum())
    np.testing.assert_array_equal(m.old.decode(a["validation"], p, a["rules"], frozen, .5),
                                  m.old.decode(b["validation"], p, b["rules"], frozen, .5))
    altered.loc[val, "temp"] += 999
    altered.loc[val, "label"] = 1 - altered.loc[val, "label"]
    c = m.base.partition_inputs(altered, train, val, frozen, prior, bracket)
    assert a["stats"] == c["stats"] and a["encoder"].category_maps == c["encoder"].category_maps
    np.testing.assert_array_equal(a["x"], c["x"])
    # No continuous run/window is manufactured across the removed middle period.
    assert m.old.segments(a["train"]).nunique() == 2


def test_diagnostic_uses_fixed_threshold_pooled_counts_and_tie_smaller():
    records = []
    for arm in m.ARMS:
        for iteration in (100, 200, 400, 700, 1000, 1400):
            for fold in m.FOLDS:
                records.append({"arm": arm, "fold": fold, "iteration": iteration,
                                "rows": 100, "tp": 10, "fp": 2, "fn": 5, "logloss_sum": 20})
    result = m.diagnostic_summary(records)
    assert result["original"]["best_fixed_decoder_iteration"] == 100
    assert result["balanced"]["curve"][0]["rows"] == 300
    assert result["balanced"]["curve"][0]["f1"] == 60 / 81


def test_no_official_or_answer_code():
    source = Path(m.__file__).read_text(encoding="utf-8")
    assert ' / "test.csv"' not in source and "sample_submission" not in source
    assert ".to_csv(" not in source
