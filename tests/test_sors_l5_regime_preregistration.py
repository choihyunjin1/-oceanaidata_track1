from __future__ import annotations

import copy
from pathlib import Path

import pytest

from p1_qc.sors_l5_regime_preregistration import (
    REFERENCE_HASHES,
    SORSL5PreregistrationError,
    load_sors_l5_preregistration,
    validate_sors_l5_preregistration,
)

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "configs" / "experiments" / "p1_sors_l5_regime_invariance_draft.json"


def _payload() -> dict:
    return load_sors_l5_preregistration(PREREGISTRATION)


def test_draft_is_valid_implementation_only_and_records_budget_no_go() -> None:
    receipt = validate_sors_l5_preregistration(_payload(), observed_hashes=REFERENCE_HASHES)
    assert receipt["status"] == "valid_implementation_only"
    assert receipt["outer_execution_authorized"] is False
    assert receipt["outer_decision"] == "no_go_due_experiment_budget"
    assert (
        receipt["recoverable_test_share_weighted_f1_upper_bound"]
        < receipt["promotion_delta_required"]
    )
    assert receipt["no_virgin_holdout_remains"] is True


def test_outer_request_always_fails_closed_even_if_authorization_is_tampered() -> None:
    with pytest.raises(SORSL5PreregistrationError, match="outer CV denied"):
        validate_sors_l5_preregistration(_payload(), require_outer_authorized=True)

    payload = copy.deepcopy(_payload())
    payload["authorization"]["outer_cv"] = True
    payload["comparison"]["outer_execution_authorized"] = True
    with pytest.raises(SORSL5PreregistrationError):
        validate_sors_l5_preregistration(payload, require_outer_authorized=True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["hypothesis"]["exactly_one_change"].__setitem__("layer", 4),
        lambda p: p["hypothesis"]["exactly_one_change"].__setitem__(
            "numeric_depth_unchanged", False
        ),
        lambda p: p["outer_budget_audit"].__setitem__(
            "recoverable_test_share_weighted_f1_upper_bound", 0.005
        ),
        lambda p: p["outer_budget_audit"]["fold_positive_support"].__setitem__("2025_q3", 5),
    ],
)
def test_contract_drift_fails_closed(mutation) -> None:
    payload = copy.deepcopy(_payload())
    mutation(payload)
    with pytest.raises(SORSL5PreregistrationError):
        validate_sors_l5_preregistration(payload)


def test_hashes_must_be_complete_and_exact() -> None:
    hashes = dict(REFERENCE_HASHES)
    hashes["test"] = "0" * 64
    with pytest.raises(SORSL5PreregistrationError, match="observed test hash"):
        validate_sors_l5_preregistration(_payload(), observed_hashes=hashes)
