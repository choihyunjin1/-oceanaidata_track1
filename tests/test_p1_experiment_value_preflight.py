from __future__ import annotations

from p1_qc.experiment_value_preflight import evaluate_experiment_value


def _registry() -> dict:
    return {
        "duplicate_similarity_threshold": 0.67,
        "minimum_score": 9,
        "closed_experiments": [
            {
                "experiment_id": "closed_calibrator",
                "intervention_layer": "score_calibration",
                "mechanisms": ["station_layer_intercept", "logit_calibration", "anchor_union"],
            }
        ],
    }


def _valid() -> dict:
    return {
        "intervention_layer": "training_objective",
        "mechanisms": ["station_layer_environment_balance", "worst_environment_proxy"],
        "evidence": {"primary_sources": ["one", "two"]},
        "evaluation": {
            "primary_metric": "binary_row_f1",
            "confirmation_windows": ["q3", "q4"],
            "required_station_slices": ["G-ORS", "I-ORS", "S-ORS"],
            "retrospective_confirmation": True,
        },
        "budget": {
            "low_fidelity_epochs": 10,
            "stop_conditions": ["q3", "q4", "pooled"],
        },
        "scope": {"official_test_labels": False, "submission_created": False},
    }


def test_novel_staged_proposal_passes_only_to_low_fidelity() -> None:
    decision = evaluate_experiment_value(_valid(), _registry())
    assert decision.decision == "PASS_TO_LOW_FIDELITY"
    assert decision.score == 10
    assert decision.hard_failures == ()
    assert decision.warnings


def test_closed_family_repackaging_is_rejected() -> None:
    proposal = _valid()
    proposal["intervention_layer"] = "score_calibration"
    proposal["mechanisms"] = ["station_layer_intercept", "logit_calibration", "anchor_union"]
    decision = evaluate_experiment_value(proposal, _registry())
    assert decision.decision == "REJECT_BEFORE_COMPUTE"
    assert "not_duplicate_of_closed_family" in decision.hard_failures


def test_wrong_metric_and_single_window_are_rejected() -> None:
    proposal = _valid()
    proposal["evaluation"]["primary_metric"] = "event_f1_iou"
    proposal["evaluation"]["confirmation_windows"] = ["q3"]
    decision = evaluate_experiment_value(proposal, _registry())
    assert decision.decision == "REJECT_BEFORE_COMPUTE"
    assert "binary_row_f1_primary" in decision.hard_failures
    assert "at_least_two_confirmation_windows" in decision.hard_failures
