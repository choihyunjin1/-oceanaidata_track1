"""Synthetic claim accounting only; no package, model, or data access."""

import copy

import pytest
from compare_recovery import validate_claims


def inputs():
    cold = {
        "status": "TRAINING_COMPLETE_FROM_EMPTY_MODEL",
        "empty_03_model_verified": True,
        "source_sha256": {"x": "a"},
    }
    recovery = {
        "status": "TRAINING_COMPLETE_RECOVERED_FRESH_MODELS",
        "empty_03_model_verified": False,
        "source_sha256": {"x": "a"},
        "new_backbone_fits": 1,
        "new_full_router_fits": 1,
        "inherited_successful_backbone_fits": 7,
        "inherited_interrupted_backbone_attempts": 1,
        "repeated_historical_backbone_fits": 0,
        "repeated_prequential_router_fits": 0,
    }
    return cold, recovery


def test_honest_interrupted_count_passes():
    validate_claims(*inputs())


@pytest.mark.parametrize(
    "key,value",
    [
        ("empty_03_model_verified", True),
        ("inherited_interrupted_backbone_attempts", 0),
        ("repeated_prequential_router_fits", 2),
        ("source_sha256", {"x": "b"}),
    ],
)
def test_false_claim_or_contract_mismatch_is_blocked(key, value):
    cold, recovered = inputs()
    recovered = copy.deepcopy(recovered)
    recovered[key] = value
    with pytest.raises(ValueError):
        validate_claims(cold, recovered)
