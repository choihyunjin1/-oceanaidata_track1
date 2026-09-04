from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_tolerance_recalibration_and_failure_replay_20260830_v2.py"
SPEC = importlib.util.spec_from_file_location("tolerance_failure_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _payloads() -> dict[str, dict]:
    return {
        name: MODULE._read_json(path)
        for name, path in MODULE.INPUTS.items()
        if path.suffix.lower() == ".json"
    }


def _policy() -> dict:
    return json.loads(MODULE.POLICY.read_text(encoding="utf-8"))


def test_directional_classifier_has_no_arbitrary_nonzero_margin() -> None:
    classify = MODULE.classify
    assert classify(0.001, (0.0001, 0.002)) == "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY"
    assert classify(0.001, (-0.0001, 0.002)) == "INCONCLUSIVE_RESEARCH_ONLY"
    assert classify(-0.001, (-0.002, -0.0001)) == "PRIMARY_HARM_RESEARCH_ONLY"
    assert classify(0.000001, None) == "REOPEN_FROZEN_CONFIRMATION_ONLY"
    assert classify(-0.01, None, decisive=True) == "PRIMARY_HARM_RESEARCH_ONLY"
    assert _policy()["tolerance_layers"]["scientific_effect"]["universal_nonzero_raw_metric_margin"] is None


def test_all_historical_ledgers_are_cross_checked_once() -> None:
    replay = MODULE.build_replay(_payloads(), _policy())
    families = replay["historical_family_replay"]
    groups = replay["canonical_group_replay"]
    assert len(families) == 48
    assert {row["problem"] for row in families} == {"P1", "P2", "P3"}
    assert len({row["family_id"] for row in families}) == 48
    assert replay["summary"]["historical_families_by_problem"] == {
        "P1": 17,
        "P2": 19,
        "P3": 12,
    }
    assert len(groups) == 35
    assert len({(row["problem"], row["group"]) for row in groups}) == 35


def test_old_raw_metric_gates_have_official_false_negative_counterexamples() -> None:
    replay = MODULE.build_replay(_payloads(), _policy())
    official = replay["official_false_negative_cases"]
    assert replay["summary"]["official_false_negative_proven_problem_count"] == 3
    assert abs(official["P1"]["observed_official_metric_improvement"] - 0.0003) < 1e-12
    assert abs(official["P2"]["observed_official_metric_improvement"] - 0.001002) < 1e-12
    assert abs(official["P3"]["observed_official_metric_improvement"] - 0.002409) < 1e-12
    assert all(official[p]["false_negative_proven"] for p in ("P1", "P2", "P3"))


def test_key_reclassifications_are_metric_aligned() -> None:
    replay = MODULE.build_replay(_payloads(), _policy())
    cases = {row["candidate"]: row for row in replay["key_case_replay"]}
    for name in (
        "supervised_rank1",
        "crossfit_rank1_v2",
        "nested_pls",
        "gaussian_copula_v2",
        "state_conditioned_copula",
    ):
        assert cases[name]["new_state"] == "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY"
        assert cases[name]["benefit_ci90"][0] > 0.0
    assert cases["lead_continuous"]["new_state"] == "EXPLORATORY_CHALLENGER_RESEARCH_ONLY"
    assert cases["sparse_gp_abstention"]["new_state"] == "INCONCLUSIVE_RESEARCH_ONLY"
    assert cases["availability_aware_copula_v2"]["new_state"] == "PRIMARY_HARM_RESEARCH_ONLY"
    assert cases["catboost_repaired_confirmation"]["new_state"] == "PRIMARY_HARM_RESEARCH_ONLY"
    assert cases["selection_matched_masked_ssl"]["new_state"] == "PRIMARY_HARM_RESEARCH_ONLY"


def test_replay_has_zero_execution_and_official_interface_access() -> None:
    summary = MODULE.build_replay(_payloads(), _policy())["summary"]
    assert summary["model_fits"] == 0
    assert summary["raw_training_or_prediction_rows_read"] == 0
    assert summary["official_test_sample_submission_hidden_or_query_rows_read"] == 0
    assert summary["csv_created"] == 0
    assert summary["uploads"] == 0
