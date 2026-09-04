from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from ocean_goal.meaningful_score_v3 import (
    ARCHITECTURE_MODE,
    ARCHITECTURE_PASS_DECISION,
    CONTRACT_RELATIVE,
    EXACT_MODE,
    GENESIS_IMPLEMENTATION_ROLES,
    V2_LEDGER_HEAD_SHA256,
    ContractError,
    append_goal_ledger_event,
    evaluate_learning_curve,
    evaluate_official_score,
    initialize_goal_ledger,
    load_contract,
    validate_goal_ledger,
    verify_initial_pins,
    verify_official_evidence_pins,
)


@pytest.fixture
def contract() -> dict:
    base = {
        "metric": "metric",
        "absolute_effect_threshold": 0.03,
        "maximum_critical_slice_regression": 0.0075,
        "required_critical_slices": ["a", "b"],
        "bootstrap_cluster": "cluster",
        "immutable_baseline": {"path": "baseline.csv", "sha256": "a" * 64},
    }
    return {
        "learning_curve": {
            "training_prefix_fractions": [0.4, 0.55, 0.7, 0.85, 1.0],
            "late_fractions": [0.7, 0.85, 1.0],
            "fixed_stochastic_seed_count": 3,
            "paired_cluster_bootstrap_replicates": 5000,
            "minimum_improved_outer_folds": 2,
        },
        "comparison_modes": {
            EXACT_MODE: {"allowed_problems": ["P1", "P2", "P3"]},
            ARCHITECTURE_MODE: {"allowed_problems": ["P2"]},
        },
        "problems": {
            "P1": {**base, "direction": "higher", "allowed_comparison_modes": [EXACT_MODE]},
            "P2": {
                **base,
                "direction": "lower",
                "allowed_comparison_modes": [EXACT_MODE, ARCHITECTURE_MODE],
            },
            "P3": {**base, "direction": "lower", "allowed_comparison_modes": [EXACT_MODE]},
        },
    }


def architecture_evidence() -> dict:
    points = []
    for fraction, improvement in zip(
        [0.4, 0.55, 0.7, 0.85, 1.0],
        [0.01, 0.02, 0.04, 0.045, 0.05],
        strict=True,
    ):
        challenger = 1.0 - improvement
        delta = challenger - 1.0
        points.append(
            {
                "fraction": fraction,
                "incumbent": 1.0,
                "challenger": challenger,
                "delta_ci90": [delta - 0.005, delta + 0.005],
                "incumbent_seed_metrics": [1.0, 1.0, 1.0],
                "challenger_seed_metrics": [challenger, challenger, challenger],
            }
        )
    binding = {
        role: {"path": f"artifacts/reference/{role}.json", "sha256": character * 64}
        for role, character in zip(
            (
                "stage_a_config",
                "deployed_graph_manifest",
                "training_recipe",
                "reference_oof_100",
                "reference_seal",
            ),
            "abcde",
            strict=True,
        )
    }
    return {
        "problem": "P2",
        "comparison_mode": ARCHITECTURE_MODE,
        "baseline_identity": {
            "comparison_mode": ARCHITECTURE_MODE,
            "explicitly_not_exact_official_incumbent": True,
            "training_recipe_origin": "NEW_PREREGISTERED_TIME_SAFE_RECIPE",
            "immutable_csv_used_only_for_official_paired_ab": True,
        },
        "reference_binding": binding,
        "preregistration": {
            "generation_id": "p2-gen2",
            "config_path": "configs/p2-gen2.json",
            "config_sha256": "f" * 64,
            "hypothesis_count": 1,
            "created_before_first_fit": True,
            "score_derived_tuning": False,
        },
        "curve_protocol": {
            "comparison_mode": ARCHITECTURE_MODE,
            "prefix_fractions": [0.4, 0.55, 0.7, 0.85, 1.0],
            "seed_ids": [1, 2, 3],
            "seed_aggregation": "PREDICTION_MEAN_THEN_METRIC",
            "bootstrap_replicates": 5000,
            "bootstrap_cluster": "cluster",
            "incumbent_fresh_refit_each_prefix": True,
            "architecture_matched_reference_fresh_refit_each_prefix": True,
            "challenger_fresh_refit_each_prefix": True,
            "same_fold_keys_metric_postprocess": True,
            "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": False,
            "deployed_inference_graph_sha_pinned": True,
            "nested_chronological_component_oof": True,
            "prefix_local_epoch_selection": True,
            "three_complete_pipeline_seeds": True,
            "reference_100_percent_oof_sealed_before_challenger_scoring": True,
        },
        "points": points,
        "fold_deltas_candidate_minus_incumbent": [-0.04, -0.03, 0.001],
        "slice_deltas_candidate_minus_incumbent": {"a": -0.03, "b": 0.004},
        "leakage_checks": {"forward_only": True, "joint_mask": True},
        "reproducibility_checks": {"same_keys": True, "stage_a_sealed_first": True},
    }


def official_evidence(*, with_pair: bool) -> dict:
    result = {
        "problem": "P2",
        "comparison_mode": ARCHITECTURE_MODE,
        "incumbent_scoring_version": "official-v1",
        "challenger_scoring_version": "official-v1",
        "incumbent_split": "public-v1",
        "challenger_split": "public-v1",
        "incumbent_points": 10.0,
        "challenger_points": 11.0,
        "incumbent_raw_metric": 1.0,
        "challenger_raw_metric": 0.95,
        "incumbent_sha256": "a" * 64,
        "challenger_sha256": "b" * 64,
        "user_approval_recorded": True,
        "saved_model_byte_reproduces": True,
        "schema_key_order_valid": True,
        "final_or_private_confirmed": False,
    }
    if with_pair:
        result["official_paired_ab"] = {
            "baseline_submission_sha256": "a" * 64,
            "challenger_submission_sha256": "b" * 64,
            "baseline_scored_before_challenger": True,
            "baseline_receipt_id": "receipt-baseline",
            "challenger_receipt_id": "receipt-challenger",
            "baseline_receipt_path": "artifacts/official/receipt-baseline.json",
            "challenger_receipt_path": "artifacts/official/receipt-challenger.json",
            "baseline_receipt_sha256": "c" * 64,
            "challenger_receipt_sha256": "d" * 64,
            "same_scoring_version_and_split": True,
            "team_wide_upload_accounting_recorded": True,
        }
    return result


def test_p2_architecture_curve_is_local_only(contract: dict) -> None:
    decision = evaluate_learning_curve(contract, architecture_evidence())
    assert decision["decision"] == ARCHITECTURE_PASS_DECISION
    assert decision["local_qualification"] is True
    assert decision["passed"] is False
    assert decision["exact_official_incumbent_comparison"] is False
    assert decision["curve_alone_can_promote"] is False


@pytest.mark.parametrize("problem", ["P1", "P3"])
def test_architecture_mode_is_rejected_for_p1_and_p3(contract: dict, problem: str) -> None:
    evidence = architecture_evidence()
    evidence["problem"] = problem
    with pytest.raises(ContractError, match="forbidden"):
        evaluate_learning_curve(contract, evidence)


def test_architecture_local_curve_without_official_pair_cannot_promote(contract: dict) -> None:
    curve = evaluate_learning_curve(contract, architecture_evidence())
    decision = evaluate_official_score(contract, curve, official_evidence(with_pair=False))
    assert decision["score_incumbent_updates"] is False
    assert decision["meaningful_incumbent_updates"] is False
    assert decision["official_promotion"] is False


def test_actual_immutable_csv_pair_and_003_effect_are_both_required(contract: dict) -> None:
    curve = evaluate_learning_curve(contract, architecture_evidence())
    official = official_evidence(with_pair=True)
    decision = evaluate_official_score(contract, curve, official)
    assert decision["meaningful_incumbent_updates"] is True
    assert decision["official_raw_improvement"] == pytest.approx(0.05)
    assert decision["meaningful_checks"]["actual_immutable_csv_paired_ab"] is True

    wrong_baseline = deepcopy(official)
    wrong_baseline["official_paired_ab"]["baseline_submission_sha256"] = "e" * 64
    rejected = evaluate_official_score(contract, curve, wrong_baseline)
    assert rejected["meaningful_incumbent_updates"] is False

    too_small = deepcopy(official)
    too_small["challenger_raw_metric"] = 0.98
    rejected = evaluate_official_score(contract, curve, too_small)
    assert rejected["meaningful_incumbent_updates"] is False


def test_official_pair_receipt_paths_sha_and_payload_are_verified(
    tmp_path: Path, contract: dict
) -> None:
    evidence = official_evidence(with_pair=True)
    pair = evidence["official_paired_ab"]
    for role, surface_role in (("baseline", "incumbent"), ("challenger", "challenger")):
        payload = {
            "official_receipt_id": pair[f"{role}_receipt_id"],
            "submission_sha256": pair[f"{role}_submission_sha256"],
            "scoring_version": evidence[f"{surface_role}_scoring_version"],
            "split_identity": evidence[f"{surface_role}_split"],
        }
        path = tmp_path / f"{role}.json"
        encoded = json.dumps(payload).encode()
        path.write_bytes(encoded)
        pair[f"{role}_receipt_path"] = path.name
        pair[f"{role}_receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
    verified = verify_official_evidence_pins(tmp_path, contract, evidence)
    assert set(verified) == {"baseline", "challenger"}

    (tmp_path / "challenger.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ContractError, match="SHA mismatch"):
        verify_official_evidence_pins(tmp_path, contract, evidence)


def test_canonical_v3_pins_and_p1_p3_exact_only() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = load_contract(root, CONTRACT_RELATIVE)
    pins = verify_initial_pins(root, contract)
    assert contract["problems"]["P1"]["allowed_comparison_modes"] == [EXACT_MODE]
    assert contract["problems"]["P3"]["allowed_comparison_modes"] == [EXACT_MODE]
    assert contract["problems"]["P2"]["absolute_effect_threshold"] == 0.03
    assert pins["V2_PREDECESSOR"]["sha256"] == (
        "8fc6b03aba3b88b07b954030759f351644513490f7a9c030b7d7f1b950023549"
    )
    assert pins["V2_EVALUATOR"]["sha256"] == (
        "69b9dc1168a47e0d1b1a50e5590c3d2f0966f2885f0f92730d4c38c4ea92800c"
    )
    assert pins["P2_EXACT_PREFIX_AUDIT"]["sha256"] == (
        "8484b73deeb508f1471e0c0baf7f0cad9a92fbcd8d2a569e94ab22b0b11727ae"
    )
    assert pins["V2_LEDGER"]["sha256"] == (
        "1decc63e63a3ac5a732035402cb00f8750e1afb64c835f17d8f972a6526fb233"
    )


def test_noncanonical_v3_contract_path_is_rejected() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ContractError, match="canonical v3"):
        load_contract(root, "configs/goals/meaningful_score_maximization_v2.json")


def test_v3_ledger_cross_chain_genesis_and_second_init_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ocean_goal import meaningful_score_v3 as module

    monkeypatch.setattr(module, "_verify_genesis_payload", lambda *_args: None)
    ledger = tmp_path / "registry.jsonl"
    first = initialize_goal_ledger(tmp_path, ledger, payload={"sealed": True})
    original = ledger.read_bytes()
    assert first["previous_event_sha256"] == V2_LEDGER_HEAD_SHA256
    assert validate_goal_ledger(tmp_path, ledger) == [first]
    with pytest.raises(FileExistsError):
        initialize_goal_ledger(tmp_path, ledger, payload={"sealed": True})
    assert ledger.read_bytes() == original


def test_v3_ledger_requires_genesis_and_forbids_duplicate_genesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ocean_goal import meaningful_score_v3 as module

    monkeypatch.setattr(module, "_verify_genesis_payload", lambda *_args: None)
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        append_goal_ledger_event(tmp_path, missing, event_type="CURVE_RESULT", payload={"x": 1})
    ledger = tmp_path / "registry.jsonl"
    initialize_goal_ledger(tmp_path, ledger, payload={"sealed": True})
    with pytest.raises(ContractError, match="duplicate"):
        append_goal_ledger_event(tmp_path, ledger, event_type="GOAL_INITIALIZED", payload={})
    second = append_goal_ledger_event(tmp_path, ledger, event_type="CURVE_RESULT", payload={"x": 2})
    assert second["seq"] == 2
    assert len(validate_goal_ledger(tmp_path, ledger)) == 2


def test_v3_append_rejects_outside_workspace_before_lock(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ContractError, match="escapes workspace"):
        append_goal_ledger_event(root, outside, event_type="CURVE_RESULT", payload={"x": 1})
    assert not (tmp_path / "outside.jsonl.append.lock").exists()


def test_v3_genesis_implementation_role_set_is_exact() -> None:
    assert GENESIS_IMPLEMENTATION_ROLES == {
        "V3_CONTRACT",
        "V3_EVALUATOR",
        "V3_CLI",
        "P2_ARCHITECTURE_CONFIG",
        "P2_ARCHITECTURE_GUARDS",
        "P2_STAGE_A_RUNNER",
        "P2_STAGE_B_RUNNER",
        "V3_TESTS",
        "P2_ARCHITECTURE_TESTS",
    }
