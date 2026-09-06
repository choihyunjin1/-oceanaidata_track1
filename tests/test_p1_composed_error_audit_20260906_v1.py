"""Synthetic labels/types and exact joins; no model or distributed inputs."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p1_composed_audit", ROOT / "scripts/p1_composed_error_audit_20260906_v1/run.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def fixture():
    n = 80
    f = pd.DataFrame({"station": "S", "year": 2025, "layer": 1,
        "time": pd.date_range("2025-01-01", periods=n, freq="10min", tz="Asia/Seoul").astype(str),
        "label": 0, "anomaly_type": pd.Series([None] * n, dtype=object)})
    f.loc[20:59, "label"] = 1
    f.loc[20:59, "anomaly_type"] = "offset"
    p = f[a.KEYS + ["label"]].copy()
    p["fold"] = "q3"
    p["union"] = 0
    p["proposal"] = 0
    p.loc[20:29, "union"] = 1
    p.loc[30:39, "proposal"] = 1
    p.loc[5, "union"] = 1
    p.loc[6, "proposal"] = 1
    p.loc[7, ["union", "proposal"]] = 1
    p["union_OR_e150"] = p.union | p.proposal
    return f, p


def test_truth_runs_edge_lengths_and_actual_time_distance():
    f, _ = fixture()
    s = a.annotate_source(f)
    assert s.loc[20:59, "truth_run_rows"].eq(40).all()
    assert s.loc[20:59, "truth_run_edge"].sum() == 24
    assert s.normal_near_anomaly_2h.iloc[8] and not s.normal_near_anomaly_2h.iloc[7]
    assert s.normal_near_anomaly_2h.iloc[71] and not s.normal_near_anomaly_2h.iloc[72]
    assert s.loc[20:59, "truth_duration"].eq("over6_to_24h").all()


def test_types_overlap_and_label_missingness():
    assert a.type_group(1, "offset+drift") == "mixed"
    assert a.type_group(1, "flatline+flatline") == "flatline"
    assert a.type_group(0, np.nan) == "normal"
    for label, kind in ((1, None), (0, "spike"), (1, "madeup")):
        with pytest.raises(ValueError):
            a.type_group(label, kind)


def test_exact_composition_confusion_and_disjoint_attribution():
    f, p = fixture()
    out = a.exact_join(p.iloc[::-1].reset_index(drop=True), a.annotate_source(f))
    c = a.counts(out)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (20, 3, 20, 37)
    assert c["positive_neither"] == 20 and c["normal_tree_only"] == c["normal_MS_only"] == c["normal_both"] == 1
    events, _ = a.events(out)
    assert len(events) == 1 and events.partially_missed.iloc[0] and events.miss_rows.iloc[0] == 20
    assert sum(v["error_rows"] for v in a.error_bursts(out, "both_FN")) == 20
    assert sum(v["error_rows"] for v in a.error_bursts(out, "composed_FP")) == 3


def test_gap_breaks_truth_run_and_no_false_temporal_nearness():
    f, _ = fixture()
    f.loc[40:, "time"] = (pd.to_datetime(f.loc[40:, "time"]) + pd.Timedelta(days=2)).astype(str)
    s = a.annotate_source(f)
    assert s.loc[20:59, "truth_run_rows"].eq(20).all()
    assert s.truth_run.iloc[39] != s.truth_run.iloc[40]


@pytest.mark.parametrize("change", ["missing_key", "duplicate", "wrong_label", "wrong_composition"])
def test_exact_join_fails_closed(change):
    f, p = fixture()
    if change == "missing_key":
        p.loc[0, "station"] = "missing"
    elif change == "duplicate":
        p = pd.concat([p, p.iloc[:1]], ignore_index=True)
    elif change == "wrong_label":
        p.loc[0, "label"] = 1
    else:
        p.loc[0, "union_OR_e150"] = 1
    with pytest.raises(ValueError):
        a.exact_join(p, a.annotate_source(f))


def test_positive_event_partial_support_is_not_silently_counted():
    f, p = fixture()
    p = p.drop(index=20).reset_index(drop=True)
    out = a.exact_join(p, a.annotate_source(f))
    with pytest.raises(ValueError):
        a.events(out)


def test_no_fit_official_observation_or_prediction_mutation_entrypoints():
    source = Path(a.__file__).read_text()
    assert "test.csv" not in source and "sample_submission" not in source and "to_csv(" not in source
    assert '"temp"' not in source and '"psal"' not in source and '"depth"' not in source
    assert "model.fit(" not in source and "ATTEMPT_LOCK" not in source
