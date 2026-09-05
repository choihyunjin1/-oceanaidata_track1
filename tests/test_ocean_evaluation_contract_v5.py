"""Synthetic-only contract checks: no distributed or official inputs, models or fits."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ocean_eval_v5", ROOT / "scripts/ocean_evaluation_contract_v5.py"
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


@pytest.fixture
def contract():
    return m.load_contract()


def kst(day):
    return pd.Timestamp(day, tz="Asia/Seoul")


def p1_frame(times, labels):
    return pd.DataFrame({"station": "S", "layer": 1, "time": times, "label": labels})


def value_frame(values, keys=None):
    return pd.DataFrame(
        {"key": list(range(len(values))) if keys is None else keys, "value": np.asarray(values, dtype=float)}
    )


def test_config_status_is_pending_support_not_execution_ready(contract):
    assert contract["status"] == m.STATUS
    assert contract["execution"]["fit_budget"] == 0
    assert not contract["execution"]["official_input_access"]
    assert not contract["execution"]["model_execution_ready"]
    assert contract["execution"]["support_audit"] == "NOT_RUN"
    assert "not the roadmap verbatim" in contract["interpretation"]


@pytest.mark.parametrize(
    "change", ["status", "direction", "dates", "warmup", "purge", "gate", "fit", "greedy"]
)
def test_contract_mutations_fail_closed(contract, change):
    c = copy.deepcopy(contract)
    if change == "status":
        c["status"] = "EXECUTION_READY"
    elif change == "direction":
        c["P1"]["direction"] = "two_sided"
    elif change == "dates":
        c["P2"]["folds"][6]["start"] = "2025-08-02T00:00:00+09:00"
    elif change == "warmup":
        c["P3"]["folds"][0]["warmup"] = False
    elif change == "purge":
        c["P1"]["purge_days"] = 0
    elif change == "gate":
        c["common"]["selection"]["bootstrap_probability_hard_gate"] = 0.8
    elif change == "fit":
        c["execution"]["fit_budget"] = 1
    else:
        c["P3"]["diagnostics"]["greedy"] = "ENABLED"
    with pytest.raises(ValueError):
        m.validate_contract(c)


def test_metrics_use_pooled_counts_and_sse_not_fold_means():
    assert m.pooled_metric([1, 1, 0], [1, 0, 1], "f1") == 0.5
    assert m.pooled_metric([0, 0], [0, 0], "f1") == 0
    pooled = m.pooled_metric(np.zeros(9), [2, 0, 0, 0, 0, 0, 0, 0, 0], "rmse")
    assert pooled == pytest.approx(2 / 3)
    assert pooled != (2 + 0) / 2  # One-row fold versus eight-row fold.


@pytest.mark.parametrize(
    "truth,prediction,metric",
    [
        ([], [], "rmse"),
        ([1], [np.nan], "rmse"),
        ([np.inf], [1], "rmse"),
        ([1, 0], [1], "f1"),
        ([1], [0.2], "f1"),
        ([[1]], [[1]], "rmse"),
        ([1], [1], "unknown"),
    ],
)
def test_invalid_evaluation_values_fail_closed(truth, prediction, metric):
    with pytest.raises(ValueError):
        m.pooled_metric(truth, prediction, metric)


@pytest.mark.parametrize(
    "fault", ["empty", "duplicate", "missing_value", "nonfinite", "order", "keys", "missing_key"]
)
def test_alignment_never_silently_drops_rows(fault):
    truth = value_frame([0, 1, 2])
    candidate = truth.copy()
    if fault == "empty":
        candidate = candidate.iloc[:0]
    elif fault == "duplicate":
        candidate.loc[2, "key"] = 1
    elif fault == "missing_value":
        candidate = candidate.drop(columns="value")
    elif fault == "nonfinite":
        candidate.loc[1, "value"] = np.inf
    elif fault == "order":
        candidate = candidate.iloc[::-1]
    elif fault == "keys":
        candidate.loc[2, "key"] = 10
    else:
        candidate.loc[2, "key"] = np.nan
    with pytest.raises(ValueError):
        m.aligned_values(truth, truth, candidate, ["key"])


def test_aligned_keys_and_values():
    truth = value_frame([0, 1])
    y, b, c = m.aligned_values(truth, value_frame([1, 1]), value_frame([0, 1]), ["key"])
    np.testing.assert_array_equal(y, c)
    np.testing.assert_array_equal(b, [1, 1])


def test_p1_year_crossing_run_owned_once_calendar_primary_differs(contract):
    frame = p1_frame(
        [kst("2024-01-01"), kst("2024-12-31 23:50"), kst("2025-01-01"), kst("2025-01-01 00:10")],
        [0, 1, 1, 0],
    )
    old = m.p1_split(frame, "H2_2024", contract)
    new = m.p1_split(frame, "H1_2025", contract)
    assert old["owner"][1] == old["owner"][2] == "H2_2024"
    assert old["validation"].tolist() == [False, True, True, False]
    assert new["validation"].tolist() == [False, False, False, True]
    assert not np.any(old["validation"] & new["validation"])
    assert m.primary_mask(frame, "P1", contract).tolist() == [False, False, True, True]


def test_p1_entire_run_crossing_purge_is_excluded_not_only_prefix(contract):
    frame = p1_frame(
        [kst("2024-12-01"), kst("2024-12-10 23:50"), kst("2024-12-11"), kst("2025-01-01")],
        [0, 1, 1, 0],
    )
    split = m.p1_split(frame, "H1_2025", contract)
    assert split["train"].tolist() == [True, False, False, False]


def test_p1_gap_station_layer_break_runs_but_year_does_not(contract):
    frame = p1_frame([kst("2024-12-31 23:50"), kst("2025-01-01 00:10")], [1, 1])
    split = m.p1_run_metadata(frame, contract)
    assert split["owner"].tolist() == ["H2_2024", "H1_2025"]
    extra = frame.iloc[[0]].assign(layer=2)
    meta = m.p1_run_metadata(pd.concat([frame, extra], ignore_index=True), contract)
    assert len(set(meta["run_id"])) == 3


def test_p1_warmup_is_not_scored_and_empty_support_rejected(contract):
    frame = p1_frame([kst("2024-01-01"), kst("2025-01-01")], [0, 0])
    with pytest.raises(ValueError, match="warm-up"):
        m.p1_split(frame, "H1_2024", contract)
    with pytest.raises(ValueError, match="empty training"):
        m.p1_split(frame.iloc[[1]], "H1_2025", contract)


@pytest.mark.parametrize(
    "times",
    [
        ["2024-01-01", "2025-01-01"],
        ["2024-01-01T00:00:00+09:00", "2023-12-31T15:00:00Z"],
    ],
)
def test_naive_and_duplicate_normalized_times_rejected(contract, times):
    with pytest.raises(ValueError):
        m.p1_run_metadata(p1_frame(times, [0, 0]), contract)


def test_p2_exact_dates_halfopen_purge_and_last17days(contract):
    times = [
        kst(day)
        for day in [
            "2024-08-24",
            "2024-08-25",
            "2024-09-01",
            "2024-10-14",
            "2024-10-15",
            "2024-11-01",
            "2024-11-07",
            "2024-11-08",
        ]
    ]
    frame = pd.DataFrame({"time": times, "layer": 2})
    split = m.p2_split(frame, "B3", contract, ["time", "layer"])
    assert split["train"].tolist() == [True, False, False, False, False, False, False, True]
    assert split["validation"].tolist() == [False, False, True, True, True, False, False, False]
    assert split["outage"].tolist() == [False, False, False, False, True, False, False, False]
    assert m.primary_mask(frame, "P2", contract).tolist() == split["validation"].tolist()
    assert len(contract["P2"]["folds"]) == 8
    assert contract["P2"]["folds"][6]["start"].startswith("2025-08-01")
    assert contract["P2"]["folds"][6]["end"].startswith("2025-09-01")


def test_p2_joint_mask_does_not_mutate_source_or_mask_other_channel(contract):
    frame = pd.DataFrame(
        {
            "time": [kst("2024-10-14"), kst("2024-10-15"), kst("2024-10-15"), kst("2024-11-01")],
            "channel": ["T5", "T5", "T4", "T5"],
            "temp": [10.0] * 4,
            "psal": [30.0] * 4,
        }
    )
    original = frame.copy(deep=True)
    masked = m.p2_joint_t5_mask(frame, "B3", contract)
    pd.testing.assert_frame_equal(frame, original)
    assert masked[["temp", "psal"]].isna().all(axis=1).tolist() == [False, True, False, False]


def test_p3_forward_78h_and_station_episode_exclusion(contract):
    frame = pd.DataFrame(
        {
            "station": ["S", "S", "S", "S", "G"],
            "anchor_time": pd.to_datetime(
                [
                    "2024-06-01T00:00Z",
                    "2024-06-02T00:00Z",
                    "2024-06-27T18:00Z",
                    "2024-07-01T00:00Z",
                    "2024-06-02T00:00Z",
                ]
            ),
            "episode_id": ["past", "cross", "near", "cross", "cross"],
        }
    )
    split = m.p3_split(frame, "Q3_2024", contract)
    assert split["train"].tolist() == [True, False, False, False, True]
    assert split["validation"].tolist() == [False, False, False, True, False]
    frame.loc[0, "episode_id"] = None
    with pytest.raises(ValueError, match="episode"):
        m.p3_split(frame, "Q3_2024", contract)


def test_p3_six_leads_are_one_complete_case():
    frame = pd.DataFrame(
        {
            "station": "S",
            "anchor_time": "2024-07-01T00:00Z",
            "lead_hours": [3, 6, 9, 12, 18, 24],
            "value": np.zeros(6),
        }
    )
    keys = ["station", "anchor_time", "lead_hours"]
    m.aligned_values(frame, frame, frame, keys, p3=True)
    partial = frame.iloc[:-1]
    with pytest.raises(ValueError, match="all six"):
        m.aligned_values(partial, partial, partial, keys, p3=True)


@pytest.mark.parametrize("episode", [np.inf, -np.inf, np.nan, ""])
def test_p3_unknown_or_nonfinite_episode_rejected(contract, episode):
    frame = pd.DataFrame(
        {
            "station": ["S", "S"],
            "anchor_time": ["2024-05-01T00:00Z", "2024-07-01T00:00Z"],
            "episode_id": [episode, "valid"],
        }
    )
    with pytest.raises(ValueError):
        m.p3_split(frame, "Q3_2024", contract)
    with pytest.raises(ValueError):
        m.bootstrap_groups(frame, "P3", contract)


def test_bootstrap_calendar_units_and_station_episode(contract):
    frame = pd.DataFrame(
        {"time": [kst("2024-01-01"), kst("2024-01-07"), kst("2024-01-08"), kst("2024-01-08 23:59")]}
    )
    assert m.bootstrap_groups(frame, "P2", contract) == [0, 0, 1, 1]
    assert m.bootstrap_groups(frame, "P1", contract)[-2:] == ["2024-01-08"] * 2
    episodes = pd.DataFrame({"station": ["S", "S", "G"], "episode_id": [1, 1, 1]})
    assert m.bootstrap_groups(episodes, "P3", contract) == [("S", 1), ("S", 1), ("G", 1)]


def test_paired_bootstrap_fixed_seed_mean_retention_without_point8_gate(contract):
    # One good and one mildly bad cluster: positive mean need not reach P(improve)=.8.
    args = ([0, 0], [2, 0], [0, 1.9], ["a", "b"], "rmse", contract)
    result = m.paired_bootstrap(*args)
    assert result == m.paired_bootstrap(*args)
    assert result["candidate_retained"]
    assert 0.6 < result["p_improve"] < 0.8
    assert result["resamples"] == 2000 and result["seed"] == 20260906
    assert result["probability_hard_gate"] is None
    assert result["automatic_promotion"] is False


def test_bootstrap_ties_are_not_improvement_and_one_cluster_not_estimable(contract):
    result = m.paired_bootstrap([1, 0], [1, 0], [1, 0], ["a", "b"], "f1", contract)
    assert result["p_improve"] == 0 and result["ci90"] == [0, 0]
    assert not result["candidate_retained"]
    small = m.paired_bootstrap([1, 0], [0, 0], [1, 0], ["a", "a"], "f1", contract)
    assert small["candidate_retained"]
    assert small["bootstrap_status"] == "NOT_ESTIMABLE"
    assert small["p_improve"] is None and small["resamples"] == 0


@pytest.mark.parametrize("metric", ["f1", "rmse"])
def test_cluster_preaggregation_equals_raw_row_resampling_on_toy(contract, metric):
    y = np.array([0, 1, 1, 0, 1], dtype=float)
    b = np.array([1, 0, 1, 0, 0], dtype=float)
    c = np.array([0, 1, 1, 1, 0], dtype=float)
    if metric == "rmse":
        b, c = b + 0.25, c + 0.5
    groups = ["a", "a", "b", "c", "c"]
    result = m.paired_bootstrap(y, b, c, groups, metric, contract)
    rng = np.random.default_rng(20260906)
    indices = [np.array([0, 1]), np.array([2]), np.array([3, 4])]
    raw_deltas = []
    for _ in range(2000):
        rows = np.concatenate([indices[i] for i in rng.integers(0, 3, 3)])
        raw_deltas.append(
            m.pooled_metric(y[rows], c[rows], metric) - m.pooled_metric(y[rows], b[rows], metric)
        )
    raw_deltas = np.array(raw_deltas)
    np.testing.assert_allclose(result["ci90"], np.quantile(raw_deltas, [0.05, 0.95]))
    expected = raw_deltas > 0 if metric == "f1" else raw_deltas < 0
    assert result["p_improve"] == expected.mean()


def test_absent_primary_values_fail_closed(contract):
    with pytest.raises(ValueError, match="primary evaluation values missing"):
        m.primary_mask(pd.DataFrame({"time": [kst("2024-01-01")]}), "P1", contract)
