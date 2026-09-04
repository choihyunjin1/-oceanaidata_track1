from __future__ import annotations

import copy
from pathlib import Path

import pytest

from p1_qc.gors_depth_invariance_preregistration import (
    FAMILY_ID,
    REFERENCE_HASHES,
    GORSDepthPreregistrationError,
    canonical_gors_depth_sha256,
    load_gors_depth_preregistration,
    validate_gors_depth_preregistration,
)

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "configs" / "experiments" / "p1_gors_depth_invariance_draft.json"


def _payload() -> dict:
    return load_gors_depth_preregistration(PREREGISTRATION)


def test_draft_is_valid_but_outer_execution_is_not_authorized() -> None:
    receipt = validate_gors_depth_preregistration(_payload(), observed_hashes=REFERENCE_HASHES)
    assert receipt["status"] == "valid"
    assert receipt["outer_execution_authorized"] is False
    assert receipt["family_outer_exposure"] == 0
    assert receipt["additional_hyperparameters"] == 0
    assert receipt["inner_reselection"] is False


def test_runner_authorization_requires_all_three_matching_fields() -> None:
    payload = _payload()
    with pytest.raises(GORSDepthPreregistrationError, match="not authorized"):
        validate_gors_depth_preregistration(payload, require_outer_authorized=True)

    payload["status"] = "authorized_one_shot"
    payload["authorization"]["outer_cv"] = True
    with pytest.raises(GORSDepthPreregistrationError, match="agreement"):
        validate_gors_depth_preregistration(payload, require_outer_authorized=True)

    payload["comparison"]["outer_execution_authorized"] = True
    ledger = [
        {
            "event": "preregistered",
            "experiment_id": "P1_gors_depth_invariance_v1",
            "family_id": FAMILY_ID,
            "preregistration_sha256": canonical_gors_depth_sha256(payload),
            "outer_result_count": 0,
        }
    ]
    receipt = validate_gors_depth_preregistration(
        payload, ledger_rows=ledger, require_outer_authorized=True
    )
    assert receipt["outer_execution_authorized"] is True


def test_any_contract_drift_fails_closed() -> None:
    cases = [
        ("trees", lambda p: p["baseline"]["fold_iteration_counts"].__setitem__(0, 699)),
        (
            "postprocess",
            lambda p: p["baseline"]["fold_postprocess"]["2025_q2"].__setitem__(
                "high_threshold", 0.2
            ),
        ),
        (
            "gate",
            lambda p: p["one_shot_evaluation_after_separate_authorization"][
                "primary_promotion_gates"
            ].__setitem__("gors_group_f1_delta_min", 0.01),
        ),
        (
            "transform",
            lambda p: p["hypothesis"]["exactly_one_change"].__setitem__("station", "S-ORS"),
        ),
    ]
    for _, mutation in cases:
        payload = copy.deepcopy(_payload())
        mutation(payload)
        with pytest.raises(GORSDepthPreregistrationError):
            validate_gors_depth_preregistration(payload)


def test_observed_hashes_must_be_complete_and_exact() -> None:
    hashes = dict(REFERENCE_HASHES)
    hashes["metrics"] = "0" * 64
    with pytest.raises(GORSDepthPreregistrationError, match="observed metrics hash"):
        validate_gors_depth_preregistration(_payload(), observed_hashes=hashes)

    hashes = dict(REFERENCE_HASHES)
    hashes.pop("test")
    with pytest.raises(GORSDepthPreregistrationError, match="exactly"):
        validate_gors_depth_preregistration(_payload(), observed_hashes=hashes)


def test_any_prior_family_outer_exposure_blocks_the_run() -> None:
    ledger = [
        {
            "family_id": FAMILY_ID,
            "event": "outer_evaluated",
            "outer_result_count": 1,
        }
    ]
    with pytest.raises(GORSDepthPreregistrationError, match="already has outer exposure"):
        validate_gors_depth_preregistration(_payload(), ledger_rows=ledger)
