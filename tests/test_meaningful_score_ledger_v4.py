from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ocean_goal import meaningful_score_ledger_v4 as ledger_v4


def _write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(value, sort_keys=True).encode("utf-8")
    path.write_bytes(encoded)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }


@pytest.fixture
def replay_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    curve_evidence = tmp_path / "curve.json"
    official_evidence = tmp_path / "official.json"
    curve_decision = tmp_path / "curve_decision.json"
    upload_receipt = tmp_path / "upload_receipt.json"
    completion_evidence = tmp_path / "completion.json"
    _write_json(curve_evidence, {"value": 1})
    _write_json(official_evidence, {"value": 2})
    _write_json(curve_decision, {"curve": "qualified"})
    _write_json(upload_receipt, {"approval": "recorded"})
    _write_json(completion_evidence, {"value": 3})

    monkeypatch.setattr(ledger_v4, "_scoring_contract", lambda _root: {"contract": "v3"})
    monkeypatch.setattr(
        ledger_v4.scoring,
        "verify_curve_evidence_pins",
        lambda _root, evidence: {"curve_source_value": evidence["value"]},
    )
    monkeypatch.setattr(
        ledger_v4.scoring,
        "evaluate_learning_curve",
        lambda _contract, evidence: {"recomputed_curve": evidence["value"]},
    )
    monkeypatch.setattr(
        ledger_v4.scoring,
        "verify_official_evidence_pins",
        lambda _root, _contract, evidence: {"official_source_value": evidence["value"]},
    )
    monkeypatch.setattr(
        ledger_v4.scoring,
        "evaluate_official_score",
        lambda _contract, curve, evidence: {
            "recomputed_score": evidence["value"],
            "curve": curve["curve"],
        },
    )
    monkeypatch.setattr(
        ledger_v4.scoring,
        "validate_upload_approval",
        lambda _root, _contract, receipt, curve_decision=None: {
            "recomputed_readiness": receipt["approval"],
            "curve": None if curve_decision is None else curve_decision["curve"],
        },
    )
    monkeypatch.setattr(
        ledger_v4.scoring,
        "evaluate_goal_completion",
        lambda _contract, evidence: {"recomputed_completion": evidence["value"]},
    )
    return {
        "root": tmp_path,
        "curve_evidence": curve_evidence,
        "official_evidence": official_evidence,
        "curve_decision": curve_decision,
        "upload_receipt": upload_receipt,
        "completion_evidence": completion_evidence,
    }


@pytest.mark.parametrize(
    ("event_type", "builder", "tamper_key"),
    [
        ("CURVE_RESULT", "curve", "decision"),
        ("OFFICIAL_SCORE_RESULT", "score", "decision"),
        ("UPLOAD_READINESS", "upload", "readiness"),
        ("GOAL_COMPLETION", "completion", "decision"),
    ],
)
def test_all_later_events_recompute_and_reject_forged_payloads(
    replay_root: dict[str, Any],
    event_type: str,
    builder: str,
    tamper_key: str,
) -> None:
    root = replay_root["root"]
    if builder == "curve":
        payload = ledger_v4.build_curve_payload(root, replay_root["curve_evidence"])
    elif builder == "score":
        payload = ledger_v4.build_official_score_payload(
            root,
            replay_root["official_evidence"],
            replay_root["curve_decision"],
        )
    elif builder == "upload":
        payload = ledger_v4.build_upload_readiness_payload(
            root,
            replay_root["upload_receipt"],
            replay_root["curve_decision"],
        )
    else:
        payload = ledger_v4.build_goal_completion_payload(
            root, replay_root["completion_evidence"]
        )
    assert ledger_v4.recompute_later_payload(root, event_type, payload) == payload

    forged = copy.deepcopy(payload)
    forged[tamper_key] = {"forged": True}
    with pytest.raises(ledger_v4.ContractError, match="differs from replay"):
        ledger_v4.recompute_later_payload(root, event_type, forged)


def test_evidence_drift_is_rejected_by_replay(replay_root: dict[str, Any]) -> None:
    root = replay_root["root"]
    evidence = replay_root["curve_evidence"]
    payload = ledger_v4.build_curve_payload(root, evidence)
    evidence.write_text('{"value": 999}', encoding="utf-8")
    with pytest.raises(ledger_v4.ContractError, match="SHA or size mismatch"):
        ledger_v4.recompute_later_payload(root, "CURVE_RESULT", payload)


def test_forged_direct_append_fails_before_lock_or_write(tmp_path: Path) -> None:
    ledger = tmp_path / ledger_v4.LEDGER_RELATIVE
    lock = ledger.with_name(f"{ledger.name}.append.lock")
    with pytest.raises(ledger_v4.ContractError, match="official evidence pin is missing"):
        ledger_v4.append_ledger_event(
            tmp_path,
            ledger,
            event_type="OFFICIAL_SCORE_RESULT",
            payload={"decision": {"meaningful_incumbent_updates": True}},
        )
    assert not ledger.exists()
    assert not lock.exists()


def test_validly_pinned_but_forged_decision_fails_before_lock(
    replay_root: dict[str, Any]
) -> None:
    root = replay_root["root"]
    ledger = root / ledger_v4.LEDGER_RELATIVE
    lock = ledger.with_name(f"{ledger.name}.append.lock")
    payload = ledger_v4.build_curve_payload(root, replay_root["curve_evidence"])
    payload["decision"] = {"forged": True}
    with pytest.raises(ledger_v4.ContractError, match="differs from replay"):
        ledger_v4.append_ledger_event(
            root,
            ledger,
            event_type="CURVE_RESULT",
            payload=payload,
        )
    assert not ledger.exists()
    assert not lock.exists()


def test_unknown_event_fails_before_lock_or_write(tmp_path: Path) -> None:
    ledger = tmp_path / ledger_v4.LEDGER_RELATIVE
    lock = ledger.with_name(f"{ledger.name}.append.lock")
    with pytest.raises(ledger_v4.ContractError, match="not allowlisted"):
        ledger_v4.append_ledger_event(
            tmp_path,
            ledger,
            event_type="FORGED_EVENT",
            payload={},
        )
    assert not ledger.exists()
    assert not lock.exists()


def test_no_init_fails_before_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / ledger_v4.LEDGER_RELATIVE
    lock = ledger.with_name(f"{ledger.name}.append.lock")
    monkeypatch.setattr(
        ledger_v4,
        "recompute_later_payload",
        lambda _root, _event_type, payload: dict(payload),
    )
    monkeypatch.setattr(ledger_v4, "validate_ledger", lambda _root, _path: [])
    with pytest.raises(ledger_v4.ContractError, match="must be initialized"):
        ledger_v4.append_ledger_event(
            tmp_path,
            ledger,
            event_type="CURVE_RESULT",
            payload={"valid": True},
        )
    assert not ledger.exists()
    assert not lock.exists()


def test_duplicate_init_is_o_excl_and_preserves_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / ledger_v4.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"existing-genesis\n")
    original = ledger.read_bytes()
    monkeypatch.setattr(ledger_v4, "_verify_genesis_payload", lambda *_args: None)
    with pytest.raises(FileExistsError):
        ledger_v4.initialize_ledger(tmp_path, ledger, payload={"valid": True})
    assert ledger.read_bytes() == original


def test_containment_rejected_before_payload_or_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.jsonl"
    monkeypatch.setattr(
        ledger_v4,
        "_verify_genesis_payload",
        lambda *_args: pytest.fail("payload verification ran before containment"),
    )
    with pytest.raises(ledger_v4.ContractError, match="escapes workspace"):
        ledger_v4.initialize_ledger(root, outside, payload={})
    with pytest.raises(ledger_v4.ContractError, match="escapes workspace"):
        ledger_v4.append_ledger_event(
            root,
            outside,
            event_type="CURVE_RESULT",
            payload={},
        )
    assert not outside.exists()
    assert not (tmp_path / "outside.jsonl.append.lock").exists()


def test_predecessor_drift_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    contract = ledger_v4.load_contract(root)
    original_sha256_file = ledger_v4.sha256_file
    predecessor = (root / ledger_v4.V3_LEDGER_RELATIVE).resolve(strict=True)

    def drift_one_file(path: Path) -> str:
        if path.resolve(strict=True) == predecessor:
            return "0" * 64
        return original_sha256_file(path)

    monkeypatch.setattr(ledger_v4, "sha256_file", drift_one_file)
    with pytest.raises(ledger_v4.ContractError, match="predecessor-ledger bytes drifted"):
        ledger_v4.verify_predecessor(root, contract)


def test_existing_append_lock_blocks_concurrent_writer_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / ledger_v4.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"immutable-ledger\n")
    original = ledger.read_bytes()
    lock = ledger.with_name(f"{ledger.name}.append.lock")
    lock.write_bytes(b"other-writer")
    fake_record = {"event_sha256": "a" * 64}
    monkeypatch.setattr(
        ledger_v4,
        "recompute_later_payload",
        lambda _root, _event_type, payload: dict(payload),
    )
    monkeypatch.setattr(
        ledger_v4,
        "validate_ledger",
        lambda _root, _path: [fake_record],
    )
    with pytest.raises(FileExistsError):
        ledger_v4.append_ledger_event(
            tmp_path,
            ledger,
            event_type="CURVE_RESULT",
            payload={"valid": True},
        )
    assert ledger.read_bytes() == original
    assert lock.read_bytes() == b"other-writer"


def test_head_race_after_lock_fails_and_releases_owned_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / ledger_v4.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"immutable-ledger\n")
    original = ledger.read_bytes()
    first = {"event_sha256": "a" * 64}
    raced = [{"event_sha256": "a" * 64}, {"event_sha256": "b" * 64}]
    calls = iter(([first], raced))
    monkeypatch.setattr(
        ledger_v4,
        "recompute_later_payload",
        lambda _root, _event_type, payload: dict(payload),
    )
    monkeypatch.setattr(
        ledger_v4,
        "validate_ledger",
        lambda _root, _path: next(calls),
    )
    with pytest.raises(ledger_v4.ContractError, match="changed before lock"):
        ledger_v4.append_ledger_event(
            tmp_path,
            ledger,
            event_type="CURVE_RESULT",
            payload={"valid": True},
        )
    assert ledger.read_bytes() == original
    assert not ledger.with_name(f"{ledger.name}.append.lock").exists()


def test_lock_setup_failure_releases_owned_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / ledger_v4.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"immutable-ledger\n")
    original = ledger.read_bytes()
    fake_record = {"event_sha256": "a" * 64}
    monkeypatch.setattr(
        ledger_v4,
        "recompute_later_payload",
        lambda _root, _event_type, payload: dict(payload),
    )
    monkeypatch.setattr(
        ledger_v4,
        "validate_ledger",
        lambda _root, _path: [fake_record],
    )
    monkeypatch.setattr(ledger_v4.os, "write", lambda _descriptor, _data: 0)
    with pytest.raises(OSError, match="short v4 append-lock write"):
        ledger_v4.append_ledger_event(
            tmp_path,
            ledger,
            event_type="CURVE_RESULT",
            payload={"valid": True},
        )
    assert ledger.read_bytes() == original
    assert not ledger.with_name(f"{ledger.name}.append.lock").exists()


def test_validation_replays_every_later_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / ledger_v4.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    genesis = ledger_v4._ledger_record(
        seq=1,
        previous=ledger_v4.V3_LEDGER_HEAD_SHA256,
        event_type=ledger_v4.GENESIS_EVENT_TYPE,
        payload={"genesis": True},
    )
    later = ledger_v4._ledger_record(
        seq=2,
        previous=genesis["event_sha256"],
        event_type="CURVE_RESULT",
        payload={"curve": True},
    )
    ledger.write_bytes(
        ledger_v4.canonical_json_bytes(genesis)
        + b"\n"
        + ledger_v4.canonical_json_bytes(later)
        + b"\n"
    )
    replayed: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(ledger_v4, "load_contract", lambda _root: {})
    monkeypatch.setattr(ledger_v4, "verify_predecessor", lambda *_args: {})
    monkeypatch.setattr(ledger_v4, "_verify_genesis_payload", lambda *_args: None)

    def replay(_root: Path, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        replayed.append((event_type, dict(payload)))
        return dict(payload)

    monkeypatch.setattr(ledger_v4, "recompute_later_payload", replay)
    assert ledger_v4.validate_ledger(tmp_path, ledger) == [genesis, later]
    assert replayed == [("CURVE_RESULT", {"curve": True})]


def test_noncanonical_qa_receipt_is_rejected(tmp_path: Path) -> None:
    receipt = tmp_path / "wrong.json"
    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(ledger_v4.ContractError, match="canonical"):
        ledger_v4._canonical_qa_pin(tmp_path, receipt)


def test_canonical_qa_receipt_deep_binds_genesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_pin = {
        "path": ledger_v4.CONTRACT_RELATIVE,
        "sha256": "1" * 64,
        "bytes": 10,
    }
    predecessor_pins = {"V3_PREDECESSOR_LEDGER": {"sha256": "2" * 64}}
    implementation_pins = {
        role: {"path": f"{role}.py", "sha256": "3" * 64, "bytes": 1}
        for role in ledger_v4.IMPLEMENTATION_ROLES
    }
    contract = {
        "ledger_id": "ledger-v4",
        "initial_state": {
            "status": "active",
            "official_uploads": 0,
            "score_promotions": {"P1": False, "P2": False, "P3": False},
            "meaningful_promotions": {"P1": False, "P2": False, "P3": False},
            "execution_counts": {
                "stage_a_fit": 0,
                "stage_a_prediction": 0,
                "stage_b_fit": 0,
                "stage_b_prediction": 0,
                "upload": 0,
            },
        },
    }
    monkeypatch.setattr(ledger_v4, "load_contract", lambda _root: contract)
    monkeypatch.setattr(
        ledger_v4, "verify_predecessor", lambda _root, _contract: predecessor_pins
    )
    monkeypatch.setattr(
        ledger_v4, "current_implementation_pins", lambda _root: implementation_pins
    )
    monkeypatch.setattr(ledger_v4, "_ledger_contract_pin", lambda _root: contract_pin)
    qa_path = tmp_path / ledger_v4.PRE_INIT_QA_RELATIVE
    qa_path.parent.mkdir(parents=True)
    receipt = {
        "schema_version": "meaningful_score_ledger_v4.pre_init_qa.v1",
        "decision": "GO_INITIALIZE_V4_LEDGER",
        "p0_count": 0,
        "p1_count": 0,
        "ledger_contract": contract_pin,
        "predecessor_ledger_anchor": ledger_v4._predecessor_anchor(),
        "implementation_pins": implementation_pins,
    }
    qa_path.write_text(json.dumps(receipt), encoding="utf-8")
    payload = ledger_v4.build_genesis_payload(tmp_path, qa_path)
    assert payload["independent_pre_init_qa"] == ledger_v4._file_pin_for_path(
        tmp_path, qa_path
    )
    assert payload["implementation_pins"] == implementation_pins

    receipt["p1_count"] = 1
    qa_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ledger_v4.ContractError, match="does not bind"):
        ledger_v4.build_genesis_payload(tmp_path, qa_path)


def test_ledger_requires_canonical_lf_termination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / ledger_v4.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"{}")
    monkeypatch.setattr(ledger_v4, "load_contract", lambda _root: {})
    monkeypatch.setattr(ledger_v4, "verify_predecessor", lambda *_args: {})
    with pytest.raises(ledger_v4.ContractError, match="LF-terminated"):
        ledger_v4.validate_ledger(tmp_path, ledger)


def test_canonical_contract_and_predecessor_are_current() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = ledger_v4.load_contract(root)
    pins = ledger_v4.verify_predecessor(root, contract)
    assert contract["event_protocol"]["later_event_types"] == [
        "CURVE_RESULT",
        "OFFICIAL_SCORE_RESULT",
        "UPLOAD_READINESS",
        "GOAL_COMPLETION",
    ]
    assert ledger_v4.ALL_EVENT_TYPES == {
        "GOAL_INITIALIZED",
        "CURVE_RESULT",
        "OFFICIAL_SCORE_RESULT",
        "UPLOAD_READINESS",
        "GOAL_COMPLETION",
    }
    assert pins["V3_PREDECESSOR_LEDGER"] == {
        "path": ledger_v4.V3_LEDGER_RELATIVE,
        "sha256": ledger_v4.V3_LEDGER_SHA256,
        "bytes": ledger_v4.V3_LEDGER_BYTES,
    }
