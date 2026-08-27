from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from ocean_goal.meaningful_score import (
    ContractError,
    append_ledger_event,
    evaluate_goal_completion,
    evaluate_learning_curve,
    evaluate_official_score,
    load_contract,
    validate_ledger,
    validate_upload_approval,
    verify_initial_pins,
)


@pytest.fixture
def contract() -> dict:
    base = {
        "metric": "metric",
        "absolute_effect_threshold": 0.02,
        "maximum_critical_slice_regression": 0.005,
        "required_critical_slices": ["a", "b"],
        "bootstrap_cluster": "cluster",
    }
    return {
        "learning_curve": {
            "training_prefix_fractions": [0.4, 0.55, 0.7, 0.85, 1.0],
            "late_fractions": [0.7, 0.85, 1.0],
            "fixed_stochastic_seed_count": 3,
            "paired_cluster_bootstrap_replicates": 5000,
            "minimum_improved_outer_folds": 2,
        },
        "problems": {
            "P1": {**base, "direction": "higher"},
            "P2": {
                **base,
                "direction": "lower",
                "absolute_effect_threshold": 0.03,
                "maximum_critical_slice_regression": 0.0075,
            },
        },
    }


def evidence(problem: str = "P1", *, direction: str = "higher") -> dict:
    points = []
    for fraction, improvement in zip(
        [0.4, 0.55, 0.7, 0.85, 1.0],
        [0.005, 0.01, 0.015, 0.022, 0.025],
        strict=True,
    ):
        if direction == "higher":
            challenger = 0.8 + improvement
            interval = [improvement - 0.004, improvement + 0.004]
        else:
            challenger = 1.0 - improvement * 2
            delta = challenger - 1.0
            interval = [delta - 0.004, delta + 0.004]
        points.append(
            {
                "fraction": fraction,
                "incumbent": 0.8 if direction == "higher" else 1.0,
                "challenger": challenger,
                "delta_ci90": interval,
                "incumbent_seed_metrics": [0.8, 0.8, 0.8],
                "challenger_seed_metrics": [challenger, challenger, challenger],
            }
        )
    return {
        "problem": problem,
        "preregistration": {
            "generation_id": "g1",
            "config_path": "configs/g1.json",
            "config_sha256": "a" * 64,
            "hypothesis_count": 3,
            "created_before_first_fit": True,
            "score_derived_tuning": False,
        },
        "curve_protocol": {
            "prefix_fractions": [0.4, 0.55, 0.7, 0.85, 1.0],
            "seed_ids": [1, 2, 3],
            "seed_aggregation": "PREDICTION_MEAN_THEN_METRIC",
            "bootstrap_replicates": 5000,
            "bootstrap_cluster": "cluster",
            "incumbent_fresh_refit_each_prefix": True,
            "challenger_fresh_refit_each_prefix": True,
            "same_fold_keys_metric_postprocess": True,
            "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": True,
        },
        "points": points,
        "fold_deltas_candidate_minus_incumbent": [0.02, 0.01, -0.001]
        if direction == "higher"
        else [-0.04, -0.03, 0.001],
        "slice_deltas_candidate_minus_incumbent": {"a": 0.01, "b": -0.002}
        if direction == "higher"
        else {"a": -0.03, "b": 0.004},
        "leakage_checks": {"time_safe": True, "target_unknown": True},
        "reproducibility_checks": {"same_keys": True, "same_metric": True},
    }


def test_higher_direction_curve_passes_both_required_gates(contract: dict) -> None:
    result = evaluate_learning_curve(contract, evidence())
    assert result["passed"] is True
    assert result["decision"] == "CURVE_QUALIFIED"
    assert result["gates"]["full_effect_meets_absolute_threshold"] is True


def test_lower_direction_curve_passes(contract: dict) -> None:
    result = evaluate_learning_curve(contract, evidence("P2", direction="lower"))
    assert result["passed"] is True
    assert result["full_fraction_improvement"] == pytest.approx(0.05)


def test_small_decimal_improvement_is_research_only(contract: dict) -> None:
    raw = evidence()
    for point in raw["points"]:
        point["challenger"] = point["incumbent"] + 0.001
        point["delta_ci90"] = [0.0005, 0.0015]
    result = evaluate_learning_curve(contract, raw)
    assert result["passed"] is False
    assert result["gates"]["full_effect_meets_absolute_threshold"] is False


def test_one_lucky_full_data_point_is_rejected(contract: dict) -> None:
    raw = evidence()
    raw["points"][2]["challenger"] = 0.79
    raw["points"][2]["delta_ci90"] = [-0.02, -0.001]
    result = evaluate_learning_curve(contract, raw)
    assert result["passed"] is False
    assert result["gates"]["late_fractions_all_improve"] is False


def test_missing_critical_slice_fails_closed(contract: dict) -> None:
    raw = evidence()
    raw["slice_deltas_candidate_minus_incumbent"].pop("b")
    with pytest.raises(ContractError, match="critical-slice"):
        evaluate_learning_curve(contract, raw)


def test_official_small_gain_updates_score_only(contract: dict) -> None:
    curve = evaluate_learning_curve(contract, evidence())
    official = {
        "problem": "P1",
        "incumbent_scoring_version": "v1",
        "challenger_scoring_version": "v1",
        "incumbent_split": "public",
        "challenger_split": "public",
        "incumbent_points": 10.0,
        "challenger_points": 10.1,
        "incumbent_raw_metric": 0.8,
        "challenger_raw_metric": 0.805,
        "incumbent_sha256": "a" * 64,
        "challenger_sha256": "b" * 64,
        "user_approval_recorded": True,
        "saved_model_byte_reproduces": True,
        "schema_key_order_valid": True,
        "final_or_private_confirmed": False,
    }
    result = evaluate_official_score(contract, curve, official)
    assert result["score_incumbent_updates"] is True
    assert result["meaningful_incumbent_updates"] is False
    assert result["decision"] == "SCORE_INCUMBENT_ONLY_SMALL_GAIN"


def test_official_meaningful_final_promotion(contract: dict) -> None:
    curve = evaluate_learning_curve(contract, evidence())
    official = {
        "problem": "P1",
        "incumbent_scoring_version": "v1",
        "challenger_scoring_version": "v1",
        "incumbent_split": "private",
        "challenger_split": "private",
        "incumbent_points": 10.0,
        "challenger_points": 11.0,
        "incumbent_raw_metric": 0.8,
        "challenger_raw_metric": 0.825,
        "incumbent_sha256": "a" * 64,
        "challenger_sha256": "b" * 64,
        "user_approval_recorded": True,
        "saved_model_byte_reproduces": True,
        "schema_key_order_valid": True,
        "final_or_private_confirmed": True,
    }
    result = evaluate_official_score(contract, curve, official)
    assert result["meaningful_incumbent_updates"] is True
    assert result["goal_problem_milestone_complete"] is True


def test_changed_split_rejects_official_comparison(contract: dict) -> None:
    curve = evaluate_learning_curve(contract, evidence())
    official = {
        "problem": "P1",
        "incumbent_scoring_version": "v1",
        "challenger_scoring_version": "v1",
        "incumbent_split": "public-a",
        "challenger_split": "public-b",
        "incumbent_points": 10.0,
        "challenger_points": 12.0,
        "incumbent_raw_metric": 0.8,
        "challenger_raw_metric": 0.85,
        "incumbent_sha256": "a" * 64,
        "challenger_sha256": "b" * 64,
        "user_approval_recorded": True,
        "saved_model_byte_reproduces": True,
        "schema_key_order_valid": True,
        "final_or_private_confirmed": False,
    }
    result = evaluate_official_score(contract, curve, official)
    assert result["score_incumbent_updates"] is False
    assert result["decision"] == "OFFICIAL_REJECTED_OR_INCOMPARABLE"


def test_no_user_approval_never_promotes(contract: dict) -> None:
    curve = evaluate_learning_curve(contract, evidence())
    official = {
        "problem": "P1",
        "incumbent_scoring_version": "v1",
        "challenger_scoring_version": "v1",
        "incumbent_split": "public",
        "challenger_split": "public",
        "incumbent_points": 10.0,
        "challenger_points": 12.0,
        "incumbent_raw_metric": 0.8,
        "challenger_raw_metric": 0.85,
        "incumbent_sha256": "a" * 64,
        "challenger_sha256": "b" * 64,
        "user_approval_recorded": False,
        "saved_model_byte_reproduces": True,
        "schema_key_order_valid": True,
        "final_or_private_confirmed": False,
    }
    result = evaluate_official_score(contract, curve, official)
    assert result["score_incumbent_updates"] is False


def test_append_only_hash_chain_detects_tampering(tmp_path: Path) -> None:
    ledger = tmp_path / "registry.jsonl"
    first = append_ledger_event(ledger, event_type="GOAL_INITIALIZED", payload={"x": 1})
    second = append_ledger_event(ledger, event_type="CURVE_RESULT", payload={"x": 2})
    records = validate_ledger(ledger)
    assert [item["seq"] for item in records] == [1, 2]
    assert second["previous_event_sha256"] == first["event_sha256"]

    tampered = deepcopy(records)
    tampered[0]["payload"]["x"] = 99
    ledger.write_text("\n".join(json.dumps(item) for item in tampered) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="hash mismatch"):
        validate_ledger(ledger)


def test_event_hash_is_canonical(tmp_path: Path) -> None:
    ledger = tmp_path / "registry.jsonl"
    record = append_ledger_event(ledger, event_type="GOAL_INITIALIZED", payload={"b": 2, "a": 1})
    base = {key: value for key, value in record.items() if key != "event_sha256"}
    expected = hashlib.sha256(
        json.dumps(base, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert record["event_sha256"] == expected


def test_canonical_contract_and_current_pins_are_exact() -> None:
    root = Path(__file__).resolve().parents[1]
    relative = "configs/goals/meaningful_score_maximization_v2.json"
    contract = load_contract(root, relative)
    pins = verify_initial_pins(root, contract)

    assert contract["problems"]["P1"]["absolute_effect_threshold"] == 0.02
    assert contract["problems"]["P2"]["absolute_effect_threshold"] == 0.03
    assert contract["problems"]["P3"]["absolute_effect_threshold"] == 0.03
    assert all(len(pins[problem]) == 2 for problem in ("P1", "P2", "P3"))
    assert contract["official_scoring"]["daily_upload_limit_team_wide"] == 3
    assert "daily_upload_limit_per_problem" not in contract["official_scoring"]
    assert pins["OFFICIAL_UI_OBSERVATION"]["source"]["sha256"] == (
        "b2aaf91646b2166fe224fdef64e4c69afa01e33f2f06f71f25a6b35155ba8ecc"
    )


def test_upload_approval_requires_exact_user_confirmation(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.csv"
    candidate.write_text("x\n1\n", encoding="utf-8")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    local_contract = {
        "official_scoring": {
            "daily_upload_limit_team_wide": 3,
            "reserve_one_daily_slot": True,
        },
        "problems": {
            "P1": {
                "immutable_baseline": {
                    "path": "candidate.csv",
                    "sha256": digest,
                }
            }
        },
    }
    receipt = {
        "problem": "P1",
        "role": "IMMUTABLE_BASELINE_ANCHOR",
        "candidate_path": "candidate.csv",
        "candidate_sha256": digest,
        "confirmation": "not-approved",
        "user_message_reference": "turn-1",
        "scoring_window_open": True,
        "scoring_version": "v1",
        "split_identity": "public-v1",
        "final_model_already_designated": False,
        "team_daily_upload_count_before": 0,
    }
    with pytest.raises(ContractError, match="explicit_user_confirmation"):
        validate_upload_approval(tmp_path, local_contract, receipt)


def test_baseline_upload_readiness_never_performs_upload(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.csv"
    candidate.write_text("x\n1\n", encoding="utf-8")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    local_contract = {
        "official_scoring": {
            "daily_upload_limit_team_wide": 3,
            "reserve_one_daily_slot": True,
        },
        "problems": {
            "P1": {
                "immutable_baseline": {
                    "path": "candidate.csv",
                    "sha256": digest,
                }
            }
        },
    }
    receipt = {
        "problem": "P1",
        "role": "IMMUTABLE_BASELINE_ANCHOR",
        "candidate_path": "candidate.csv",
        "candidate_sha256": digest,
        "confirmation": f"APPROVE_OFFICIAL_UPLOAD:P1:{digest}",
        "user_message_reference": "turn-1",
        "scoring_window_open": True,
        "scoring_version": "v1",
        "split_identity": "public-v1",
        "final_model_already_designated": False,
        "team_daily_upload_count_before": 0,
    }
    result = validate_upload_approval(tmp_path, local_contract, receipt)
    assert result["status"] == "UPLOAD_READY_PENDING_SEPARATE_PLATFORM_ACTION"
    assert result["upload_performed"] is False


def test_curve_challenger_cannot_use_reserved_third_slot_without_override(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.csv"
    candidate.write_text("x\n1\n", encoding="utf-8")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    local_contract = {
        "official_scoring": {
            "daily_upload_limit_team_wide": 3,
            "reserve_one_daily_slot": True,
        },
        "problems": {"P1": {"immutable_baseline": {"path": "base.csv", "sha256": "a" * 64}}},
    }
    receipt = {
        "problem": "P1",
        "role": "CURVE_QUALIFIED_CHALLENGER",
        "candidate_path": "candidate.csv",
        "candidate_sha256": digest,
        "confirmation": f"APPROVE_OFFICIAL_UPLOAD:P1:{digest}",
        "user_message_reference": "turn-1",
        "scoring_window_open": True,
        "scoring_version": "v1",
        "split_identity": "public-v1",
        "final_model_already_designated": False,
        "team_daily_upload_count_before": 2,
    }
    with pytest.raises(ContractError, match="reserve_slot_policy"):
        validate_upload_approval(
            tmp_path,
            local_contract,
            receipt,
            curve_decision={"problem": "P1", "passed": True},
        )


def _completion_evidence() -> dict:
    return {
        "problems": {
            problem: {
                "meaningful_promotion_count": 1,
                "initial_points": 10.0,
                "final_points": 11.0,
                "final_score_incumbent_is_problem_best": True,
                "final_private_improvement_maintained": True,
                "scored_csv_byte_reproduced": True,
                "saved_model_package_reproduced": True,
                "organizer_verification_confirmed": True,
                "scoring_version": "final-v1",
                "final_split": "private-v1",
            }
            for problem in ("P1", "P2", "P3")
        },
        "all_final_model_submissions_confirmed": True,
    }


def test_completion_requires_all_three_meaningful_promotions(contract: dict) -> None:
    three_problem_contract = deepcopy(contract)
    three_problem_contract["problems"]["P3"] = deepcopy(three_problem_contract["problems"]["P2"])
    raw = _completion_evidence()
    raw["problems"]["P3"]["meaningful_promotion_count"] = 0
    result = evaluate_goal_completion(three_problem_contract, raw)
    assert result["goal_complete"] is False
    assert result["decision"] == "NOT_COMPLETE"


def test_completion_passes_only_after_final_reproduction_and_verification(contract: dict) -> None:
    three_problem_contract = deepcopy(contract)
    three_problem_contract["problems"]["P3"] = deepcopy(three_problem_contract["problems"]["P2"])
    result = evaluate_goal_completion(three_problem_contract, _completion_evidence())
    assert result["goal_complete"] is True
    assert result["total_point_gain"] == pytest.approx(3.0)
