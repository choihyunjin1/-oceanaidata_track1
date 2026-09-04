from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from p1_qc.preregistration import (
    PreregistrationError,
    load_preregistration,
    read_experiment_ledger,
    record_ledger_entry,
    validate_preregistration,
)

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "configs" / "experiments" / "p1_next_single_hypothesis.json"
LEDGER = ROOT / "reports" / "EXPERIMENT_LEDGER.jsonl"


def valid_payload() -> dict[str, object]:
    return load_preregistration(PREREGISTRATION)


def test_repository_preregistration_schema_passes_without_execution_ledger() -> None:
    receipt = validate_preregistration(valid_payload())
    assert receipt["status"] == "valid"
    assert receipt["candidate_count"] == 2
    assert receipt["minimum_outer_weighted_f1"] == pytest.approx(0.8183155525620019)
    assert receipt["outer_is_independent_holdout"] is False
    assert receipt["shadow_holdout_available"] is False


def test_repository_ledger_closure_blocks_exact_experiment_rerun() -> None:
    with pytest.raises(PreregistrationError, match="family is closed; rerun is prohibited"):
        validate_preregistration(valid_payload(), ledger_rows=read_experiment_ledger(LEDGER))


def test_repository_ledger_closure_blocks_a_new_experiment_in_same_family() -> None:
    payload = valid_payload()
    payload["experiment_id"] = "P1_fixed24h_peer_coherence_v2_forbidden"
    with pytest.raises(PreregistrationError, match="family-wise outer reuse"):
        validate_preregistration(payload, ledger_rows=read_experiment_ledger(LEDGER))


def test_candidate_growth_and_adaptive_search_fail_closed() -> None:
    payload = valid_payload()
    payload["comparison"]["arm_count"] = 3
    payload["comparison"]["arms"].append({"id": "tuned_48h", "change": "new"})
    with pytest.raises(PreregistrationError, match="exactly 2"):
        validate_preregistration(payload)

    payload = valid_payload()
    payload["comparison"]["additional_hyperparameters"] = ["window_hours"]
    with pytest.raises(PreregistrationError, match="must remain empty"):
        validate_preregistration(payload)


def test_outer_label_selection_and_weak_promotion_gates_are_rejected() -> None:
    payload = valid_payload()
    payload["inner_nuisance_selection"]["scope"] = "outer_validation"
    with pytest.raises(PreregistrationError, match="inner-validation"):
        validate_preregistration(payload)

    payload = valid_payload()
    payload["outer_evaluation"]["promotion_gate"]["weighted_f1_delta_min"] = 0.001
    with pytest.raises(PreregistrationError, match=r"at least \+0.005"):
        validate_preregistration(payload)


def test_inner_nuisance_selection_cannot_select_the_candidate_family() -> None:
    payload = valid_payload()
    payload["comparison"]["family_selection_allowed"] = True
    with pytest.raises(PreregistrationError, match="must be False"):
        validate_preregistration(payload)

    payload = valid_payload()
    payload["inner_nuisance_selection"]["selected_only"].append("feature_bundle")
    with pytest.raises(PreregistrationError, match="limited"):
        validate_preregistration(payload)


def test_fixed_bundle_cannot_be_tuned_after_registration() -> None:
    payload = valid_payload()
    payload["hypothesis"]["exactly_one_change"]["fixed_parameters"]["window_hours"] = 48
    with pytest.raises(PreregistrationError, match="fixed_parameters"):
        validate_preregistration(payload)


def test_prior_family_outer_result_blocks_reuse() -> None:
    payload = valid_payload()
    prior = [
        {
            "experiment_id": "earlier_peer_gate",
            "family_id": payload["family_id"],
            "event": "outer_evaluated",
            "outer_result_count": 1,
        }
    ]
    with pytest.raises(PreregistrationError, match="family-wise outer reuse"):
        validate_preregistration(payload, ledger_rows=prior)


def test_missing_shadow_holdout_cannot_be_relabelled_available() -> None:
    payload = valid_payload()
    payload["shadow_holdout"]["available"] = True
    with pytest.raises(PreregistrationError, match="must be False"):
        validate_preregistration(payload)


def test_record_ledger_entry_is_append_only_and_rejects_duplicate_event(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    entry = {
        "experiment_id": "unit",
        "family_id": "unit_family",
        "event": "preregistered",
        "outer_result_count": 0,
        "recorded_at_kst": "2026-08-13T20:30:00+09:00",
    }
    record_ledger_entry(ledger, entry)
    assert read_experiment_ledger(ledger) == [entry]
    with pytest.raises(PreregistrationError, match="duplicate ledger event"):
        record_ledger_entry(ledger, entry)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":"1.0","schema_version":"2.0"}', encoding="utf-8")
    with pytest.raises(PreregistrationError, match="duplicate JSON key"):
        load_preregistration(path)


def test_payload_hash_is_stable_under_key_order() -> None:
    payload = valid_payload()
    reversed_payload = {key: payload[key] for key in reversed(payload)}
    first = validate_preregistration(payload)["preregistration_sha256"]
    second = validate_preregistration(copy.deepcopy(reversed_payload))["preregistration_sha256"]
    assert first == second


def test_ledger_lines_are_valid_json() -> None:
    for raw in LEDGER.read_text(encoding="utf-8").splitlines():
        assert isinstance(json.loads(raw), dict)
