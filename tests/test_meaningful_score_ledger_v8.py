from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from ocean_goal import meaningful_score_ledger_v5 as ledger_v5
from ocean_goal import meaningful_score_ledger_v7 as ledger_v7
from ocean_goal import meaningful_score_ledger_v8 as ledger_v8

ROOT = Path(__file__).resolve().parents[1]
P2_EVIDENCE = (
    ROOT
    / "artifacts/p2_architecture_matched_stage_b_parser_correction_r1/learning_curve_evidence.json"
)
EXACT_CURVE_EVIDENCE = (
    "artifacts/p1_meaningful_learning_curve_generation_v1/learning_curve_evidence.json",
    "artifacts/p3_meaningful_learning_curve_20260823_v1/learning_curve_evidence.json",
    "artifacts/p3_causal_forcing_sequence_residual_20260823_v1/learning_curve_evidence.json",
    "artifacts/p1_station_layer_temporal_convolution_event_v2/learning_curve_evidence.json",
    "artifacts/p3_causal_spectral_kernel_20260823_v1/learning_curve_evidence.json",
    "artifacts/p1_binary_event_tcn_dense_natural_v3/learning_curve_evidence.json",
    "artifacts/p1_masked_pretrain_binary_event_v4r4/learning_curve_evidence.json",
    "artifacts/p3_station_stable_energy_state_space_20260823_v1/learning_curve_evidence.json",
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _completion_contract() -> dict[str, Any]:
    return {"completion_lineage_policy": ledger_v8._expected_completion_policy()}


def _claimed_complete_payload(root: Path, evidence_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "meaningful_score_ledger_v5.goal_completion.v1",
        "evidence": ledger_v8._file_pin_for_path(root, evidence_path),
        "decision": {
            "schema_version": "goal_completion_decision.v1",
            "decision": "COMPLETE",
            "goal_complete": True,
            "initial_total_points": 1.0,
            "final_total_points": 2.0,
            "total_point_gain": 1.0,
            "problem_checks": {
                problem: {
                    "meaningful_promotion_at_least_once": True,
                    "other_check": True,
                }
                for problem in ("P1", "P2", "P3")
            },
            "global_checks": {
                "all_problem_checks_pass": True,
                "portfolio_total_strictly_improves": True,
                "all_final_model_submissions_confirmed": True,
            },
        },
        "upload_performed": False,
    }


def _prior_record(
    seq: int,
    event_type: str,
    *,
    decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if decision is not None:
        payload["decision"] = decision
    return {
        "seq": seq,
        "event_type": event_type,
        "event_sha256": format(seq, "064x"),
        "payload": payload,
    }


def _patch_completion_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ledger_v8, "_verify_inherited_policy", lambda _root: _completion_contract())
    monkeypatch.setattr(
        ledger_v8.ledger_v5,
        "build_goal_completion_payload",
        _claimed_complete_payload,
    )


def test_v8_pins_v5_seq9_and_frozen_uninitialized_v7() -> None:
    contract = ledger_v8.load_contract(ROOT)
    verified = ledger_v8.verify_predecessor(ROOT, contract)
    assert contract["predecessor_ledger"] == {
        "path": ledger_v8.V5_LEDGER_RELATIVE,
        "sha256": ledger_v8.V5_LEDGER_SHA256,
        "bytes": ledger_v8.V5_LEDGER_BYTES,
        "event_count": 9,
        "head_event_sha256": ledger_v8.V5_LEDGER_HEAD_SHA256,
    }
    assert verified["V7_STATIC_IMPLEMENTATION"] == ledger_v8.V7_IMPLEMENTATION
    for relative in (
        ledger_v7.LEDGER_RELATIVE,
        ledger_v7.PRE_INIT_QA_RELATIVE,
        f"{ledger_v7.LEDGER_RELATIVE}.append.lock",
    ):
        assert not (ROOT / relative).exists()


def test_v7_four_file_bytes_are_unchanged() -> None:
    expected = {
        "V7_CONTRACT": (
            7644,
            "bd71902bd4d00fa925b6fddb15e76a5d3872d242ccbdd1c8f15fd9d7309efddc",
        ),
        "V7_EVALUATOR": (
            34661,
            "6cea123ccc181915667643ff1a2d320368255997bec442ba847bb13d9ec29382",
        ),
        "V7_CLI": (
            7866,
            "b898fead7ab7b6cd1edb1896c9d33b5baeb3c963fa5177c0d2390ef56eae2b49",
        ),
        "V7_TESTS": (
            16840,
            "b6d165d6313957f8aa27018c83123da960cccdc6284848123e439c4d579c2ad0",
        ),
    }
    for role, pin in ledger_v8.V7_IMPLEMENTATION.items():
        path = ROOT / pin["path"]
        assert (path.stat().st_size, ledger_v8.sha256_file(path)) == expected[role]


def test_contract_forbids_current_or_self_asserted_completion_authority() -> None:
    contract = ledger_v8.load_contract(ROOT)
    policy = contract["completion_lineage_policy"]
    assert policy["standalone_completion_evidence_authoritative"] is False
    assert policy["replay_with_all_prior_v8_events_required"] is True
    assert policy["p2_architecture_events_cannot_satisfy_meaningful_milestone"] is True
    assert policy["p2_meaningful_milestone_requires_distinct_event_type"] == (
        ledger_v8.REQUIRED_CONFIRMATION_EVENT_TYPE
    )
    assert policy["required_event_type_currently_allowlisted"] is False
    assert policy["required_event_type_currently_policy_authorized"] is False
    assert ledger_v8.REQUIRED_CONFIRMATION_EVENT_TYPE not in ledger_v8.ALL_EVENT_TYPES


@pytest.mark.parametrize("relative", EXACT_CURVE_EVIDENCE)
def test_p1_p3_noncompletion_curve_payloads_remain_bit_for_bit_v5(relative: str) -> None:
    path = ROOT / relative
    assert ledger_v8.build_curve_payload(ROOT, path) == ledger_v5.build_curve_payload(ROOT, path)


def test_p2_curve_and_v7_policy_closures_are_inherited_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert ledger_v8.build_curve_payload(ROOT, P2_EVIDENCE) == ledger_v7.build_curve_payload(
        ROOT, P2_EVIDENCE
    )
    evidence = tmp_path / "evidence.json"
    curve = tmp_path / "curve.json"
    receipt = tmp_path / "receipt.json"
    for path in (evidence, curve, receipt):
        _write_json(path, {"value": path.name})
    official = {"v7": "official-policy-closed"}
    upload = {"v7": "upload-policy-closed"}
    monkeypatch.setattr(ledger_v8, "_verify_inherited_policy", lambda _root: {})
    monkeypatch.setattr(
        ledger_v8.ledger_v7,
        "build_official_score_payload",
        lambda *_args: official,
    )
    monkeypatch.setattr(
        ledger_v8.ledger_v7,
        "build_upload_readiness_payload",
        lambda *_args: upload,
    )
    assert ledger_v8.build_official_score_payload(tmp_path, evidence, curve) is official
    assert ledger_v8.build_upload_readiness_payload(tmp_path, receipt, curve) is upload


def test_forged_standalone_complete_is_forced_not_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "completion.json"
    _write_json(evidence, {"organizer_claim": "all complete"})
    _patch_completion_builder(monkeypatch)
    payload = ledger_v8.build_goal_completion_payload(tmp_path, evidence)
    assert payload["standalone_evidence_claims"] == {
        "goal_complete": True,
        "p2_meaningful_promotion_at_least_once": True,
        "authoritative": False,
    }
    assert payload["decision"]["decision"] == "NOT_COMPLETE"
    assert payload["decision"]["goal_complete"] is False
    assert payload["decision"]["p2_milestone_authorized"] is False
    assert payload["decision"]["problem_checks"]["P2"][
        "meaningful_promotion_at_least_once"
    ] is False
    assert payload["decision"]["problem_checks"]["P2"][
        "exact_official_confirmation_event_authorized"
    ] is False
    assert payload["decision"]["global_checks"]["all_problem_checks_pass"] is False
    assert payload["lineage"]["prior_event_count"] == 0
    assert payload["lineage"]["p2_milestone_authorized"] is False


def test_architecture_lineage_and_exact_score_cannot_authorize_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "completion.json"
    _write_json(evidence, {"organizer_claim": "all complete"})
    _patch_completion_builder(monkeypatch)
    prior = [
        _prior_record(1, "GOAL_INITIALIZED"),
        _prior_record(
            2,
            "CURVE_RESULT",
            decision={
                "problem": "P2",
                "comparison_mode": ledger_v8.ARCHITECTURE_MODE,
                "local_qualification": True,
            },
        ),
        _prior_record(
            3,
            "OFFICIAL_SCORE_RESULT",
            decision={
                "problem": "P2",
                "comparison_mode": ledger_v8.ARCHITECTURE_MODE,
                "meaningful_incumbent_updates": False,
            },
        ),
        _prior_record(
            4,
            "OFFICIAL_SCORE_RESULT",
            decision={
                "problem": "P2",
                "comparison_mode": ledger_v8.scoring.EXACT_MODE,
                "meaningful_incumbent_updates": True,
            },
        ),
    ]
    payload = ledger_v8.build_goal_completion_payload(
        tmp_path, evidence, prior_records=prior
    )
    assert payload["lineage"]["p2_architecture_event_seqs"] == [2, 3]
    assert payload["lineage"]["p2_exact_official_score_event_seqs"] == [4]
    assert payload["lineage"]["required_confirmation_event_count"] == 0
    assert payload["lineage"]["p2_milestone_authorized"] is False
    assert payload["decision"]["goal_complete"] is False
    assert ledger_v8.recompute_later_payload(
        tmp_path,
        "GOAL_COMPLETION",
        payload,
        prior_records=prior,
    ) == payload

    forged = copy.deepcopy(payload)
    forged["decision"]["decision"] = "COMPLETE"
    forged["decision"]["goal_complete"] = True
    forged["decision"]["problem_checks"]["P2"][
        "meaningful_promotion_at_least_once"
    ] = True
    with pytest.raises(ledger_v8.ContractError, match="differs from replay"):
        ledger_v8.recompute_later_payload(
            tmp_path,
            "GOAL_COMPLETION",
            forged,
            prior_records=prior,
        )


def test_completion_replay_without_prior_lineage_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "completion.json"
    _write_json(evidence, {"organizer_claim": "all complete"})
    _patch_completion_builder(monkeypatch)
    payload = ledger_v8.build_goal_completion_payload(tmp_path, evidence)
    with pytest.raises(ledger_v8.ContractError, match="requires all prior v8 records"):
        ledger_v8.recompute_later_payload(tmp_path, "GOAL_COMPLETION", payload)


def test_forged_future_confirmation_event_is_not_a_valid_lineage_record() -> None:
    forged = _prior_record(1, ledger_v8.REQUIRED_CONFIRMATION_EVENT_TYPE)
    with pytest.raises(ledger_v8.ContractError, match="not a valid v8 prefix"):
        ledger_v8._completion_lineage([forged])


def test_stateful_ledger_replay_rejects_rehashed_complete_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "completion.json"
    _write_json(evidence, {"organizer_claim": "all complete"})
    _patch_completion_builder(monkeypatch)
    monkeypatch.setattr(ledger_v8, "load_contract", lambda _root: {})
    monkeypatch.setattr(ledger_v8, "verify_predecessor", lambda *_args: {})
    monkeypatch.setattr(ledger_v8, "verify_p2_stage_a_v3_lineage", lambda *_args: {})
    monkeypatch.setattr(ledger_v8, "_verify_genesis_payload", lambda *_args: None)
    ledger = tmp_path / ledger_v8.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    genesis = ledger_v8.initialize_ledger(tmp_path, ledger, payload={"genesis": True})
    payload = ledger_v8.build_goal_completion_payload(
        tmp_path, evidence, prior_records=[genesis]
    )
    ledger_v8.append_ledger_event(
        tmp_path,
        ledger,
        event_type="GOAL_COMPLETION",
        payload=payload,
    )
    records = ledger_v8.validate_ledger(tmp_path, ledger)
    assert records[-1]["payload"]["decision"]["goal_complete"] is False

    forged = copy.deepcopy(records[-1])
    forged["payload"]["decision"]["decision"] = "COMPLETE"
    forged["payload"]["decision"]["goal_complete"] = True
    base = {key: value for key, value in forged.items() if key != "event_sha256"}
    forged["event_sha256"] = hashlib.sha256(
        ledger_v8.canonical_json_bytes(base)
    ).hexdigest()
    ledger.write_bytes(
        ledger_v8.canonical_json_bytes(records[0])
        + b"\n"
        + ledger_v8.canonical_json_bytes(forged)
        + b"\n"
    )
    with pytest.raises(ledger_v8.ContractError, match="differs from replay"):
        ledger_v8.validate_ledger(tmp_path, ledger)


def _patch_synthetic_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ledger_v8, "load_contract", lambda _root: {})
    monkeypatch.setattr(ledger_v8, "verify_predecessor", lambda *_args: {})
    monkeypatch.setattr(ledger_v8, "verify_p2_stage_a_v3_lineage", lambda *_args: {})
    monkeypatch.setattr(ledger_v8, "_verify_genesis_payload", lambda *_args: None)
    monkeypatch.setattr(
        ledger_v8,
        "recompute_later_payload",
        lambda _root, _event_type, payload, **_kwargs: dict(payload),
    )


def test_robust_binary_write_loop_survives_partial_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_synthetic_chain(monkeypatch)
    ledger = tmp_path / ledger_v8.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    real_write = os.write

    def partial_write(descriptor: int, data: bytes | memoryview) -> int:
        return real_write(descriptor, bytes(data[:7]))

    monkeypatch.setattr(ledger_v8.os, "write", partial_write)
    ledger_v8.initialize_ledger(tmp_path, ledger, payload={"genesis": "x" * 100})
    ledger_v8.append_ledger_event(
        tmp_path,
        ledger,
        event_type="CURVE_RESULT",
        payload={"curve": "y" * 100},
    )
    raw = ledger.read_bytes()
    assert raw.count(b"\n") == 2
    assert b"\r" not in raw
    assert len(ledger_v8.validate_ledger(tmp_path, ledger)) == 2


def test_all_three_open_sites_are_binary_o_excl_and_write_loop_only() -> None:
    source = Path(ledger_v8.__file__).read_text(encoding="utf-8")
    assert source.count("os.open(") == 3
    assert source.count("| O_BINARY") == 3
    assert source.count("os.O_EXCL | O_BINARY") == 2
    assert source.count("_write_all(descriptor") == 4
    assert "os.write(descriptor, encoded)" not in source


def test_containment_unknown_event_and_duplicate_init_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.jsonl"
    monkeypatch.setattr(
        ledger_v8,
        "_verify_genesis_payload",
        lambda *_args: pytest.fail("payload ran before containment"),
    )
    with pytest.raises(ledger_v8.ContractError, match="escapes workspace"):
        ledger_v8.initialize_ledger(root, outside, payload={})
    with pytest.raises(ledger_v8.ContractError, match="escapes workspace"):
        ledger_v8.append_ledger_event(root, outside, event_type="CURVE_RESULT", payload={})
    with pytest.raises(ledger_v8.ContractError, match="not allowlisted"):
        ledger_v8.append_ledger_event(
            root,
            root / ledger_v8.LEDGER_RELATIVE,
            event_type=ledger_v8.REQUIRED_CONFIRMATION_EVENT_TYPE,
            payload={},
        )
    ledger = root / ledger_v8.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"existing\n")
    before = ledger.read_bytes()
    monkeypatch.setattr(ledger_v8, "_verify_genesis_payload", lambda *_args: None)
    with pytest.raises(FileExistsError):
        ledger_v8.initialize_ledger(root, ledger, payload={"valid": True})
    assert ledger.read_bytes() == before
    assert not outside.exists()


def test_static_v8_has_no_receipt_ledger_lock_or_personal_path() -> None:
    for relative in (
        ledger_v8.LEDGER_RELATIVE,
        ledger_v8.PRE_INIT_QA_RELATIVE,
        f"{ledger_v8.LEDGER_RELATIVE}.append.lock",
    ):
        assert not (ROOT / relative).exists()
    paths = [
        ROOT / ledger_v8.CONTRACT_RELATIVE,
        Path(ledger_v8.__file__),
        ROOT / "scripts/run_meaningful_score_ledger_v8.py",
        Path(__file__),
    ]
    unix_home = "/" + "home/"
    personal = re.compile(
        rf"(?i)(?:[a-z]:[\\/]+users[\\/]+[^\\/]+|{re.escape(unix_home)}[^/]+)"
    )
    assert all(personal.search(path.read_text(encoding="utf-8")) is None for path in paths)
