from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import pytest

from ocean_goal import meaningful_score_ledger_v5 as ledger_v5


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


@pytest.fixture
def replay_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    paths = {
        "curve": tmp_path / "curve.json",
        "official": tmp_path / "official.json",
        "decision": tmp_path / "decision.json",
        "receipt": tmp_path / "receipt.json",
        "completion": tmp_path / "completion.json",
    }
    for number, path in enumerate(paths.values(), 1):
        _write_json(path, {"value": number})
    monkeypatch.setattr(ledger_v5, "_scoring_contract", lambda _root: {"v": 3})
    monkeypatch.setattr(
        ledger_v5.scoring,
        "verify_curve_evidence_pins",
        lambda _root, evidence: {"pin": evidence["value"]},
    )
    monkeypatch.setattr(
        ledger_v5.scoring,
        "evaluate_learning_curve",
        lambda _contract, evidence: {"curve": evidence["value"]},
    )
    monkeypatch.setattr(
        ledger_v5.scoring,
        "verify_official_evidence_pins",
        lambda _root, _contract, evidence: {"pin": evidence["value"]},
    )
    monkeypatch.setattr(
        ledger_v5.scoring,
        "evaluate_official_score",
        lambda _contract, curve, evidence: {
            "score": evidence["value"],
            "curve": curve["value"],
        },
    )
    monkeypatch.setattr(
        ledger_v5.scoring,
        "validate_upload_approval",
        lambda _root, _contract, receipt, curve_decision=None: {
            "ready": receipt["value"],
            "curve": None if curve_decision is None else curve_decision["value"],
        },
    )
    monkeypatch.setattr(
        ledger_v5.scoring,
        "evaluate_goal_completion",
        lambda _contract, evidence: {"complete": evidence["value"]},
    )
    return {"root": tmp_path, **paths}


@pytest.mark.parametrize(
    ("event_type", "builder", "tamper"),
    [
        ("CURVE_RESULT", "curve", "decision"),
        ("OFFICIAL_SCORE_RESULT", "official", "decision"),
        ("UPLOAD_READINESS", "upload", "readiness"),
        ("GOAL_COMPLETION", "completion", "decision"),
    ],
)
def test_all_typed_payloads_replay_and_reject_forgery(
    replay_root: dict[str, Any], event_type: str, builder: str, tamper: str
) -> None:
    root = replay_root["root"]
    if builder == "curve":
        payload = ledger_v5.build_curve_payload(root, replay_root["curve"])
    elif builder == "official":
        payload = ledger_v5.build_official_score_payload(
            root, replay_root["official"], replay_root["decision"]
        )
    elif builder == "upload":
        payload = ledger_v5.build_upload_readiness_payload(
            root, replay_root["receipt"], replay_root["decision"]
        )
    else:
        payload = ledger_v5.build_goal_completion_payload(root, replay_root["completion"])
    assert ledger_v5.recompute_later_payload(root, event_type, payload) == payload
    forged = copy.deepcopy(payload)
    forged[tamper] = {"forged": True}
    with pytest.raises(ledger_v5.ContractError, match="differs from replay"):
        ledger_v5.recompute_later_payload(root, event_type, forged)


def _patch_synthetic_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ledger_v5, "load_contract", lambda _root: {})
    monkeypatch.setattr(ledger_v5, "verify_predecessor", lambda *_args: {})
    monkeypatch.setattr(ledger_v5, "verify_failed_v4_lineage", lambda *_args: {})
    monkeypatch.setattr(ledger_v5, "_verify_genesis_payload", lambda *_args: None)
    monkeypatch.setattr(
        ledger_v5,
        "recompute_later_payload",
        lambda _root, _event_type, payload: dict(payload),
    )


@pytest.mark.skipif(os.name != "nt", reason="real Windows CRT binary-mode integration")
def test_real_windows_raw_bytes_init_and_one_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_synthetic_chain(monkeypatch)
    ledger = tmp_path / ledger_v5.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    genesis = ledger_v5.initialize_ledger(tmp_path, ledger, payload={"genesis": True})
    appended = ledger_v5.append_ledger_event(
        tmp_path,
        ledger,
        event_type="CURVE_RESULT",
        payload={"curve": True},
    )
    raw = ledger.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    assert raw.count(b"\n") == 2
    records = ledger_v5.validate_ledger(tmp_path, ledger)
    assert records == [genesis, appended]


def test_robust_write_loop_survives_partial_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_synthetic_chain(monkeypatch)
    ledger = tmp_path / ledger_v5.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    real_write = os.write

    def partial_write(descriptor: int, data: bytes | memoryview) -> int:
        return real_write(descriptor, bytes(data[:7]))

    monkeypatch.setattr(ledger_v5.os, "write", partial_write)
    ledger_v5.initialize_ledger(tmp_path, ledger, payload={"genesis": "x" * 100})
    ledger_v5.append_ledger_event(
        tmp_path,
        ledger,
        event_type="CURVE_RESULT",
        payload={"curve": "y" * 100},
    )
    raw = ledger.read_bytes()
    assert raw.count(b"\n") == 2
    assert b"\r" not in raw
    assert len(ledger_v5.validate_ledger(tmp_path, ledger)) == 2


def test_all_three_open_sites_are_binary_and_write_loop_only() -> None:
    source = Path(ledger_v5.__file__).read_text(encoding="utf-8")
    assert source.count("os.open(") == 3
    assert source.count("| O_BINARY") == 3
    assert source.count("_write_all(descriptor") == 4  # definition + three call sites
    assert "os.write(descriptor, encoded)" not in source


def test_duplicate_init_o_excl_preserves_failed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / ledger_v5.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"existing\n")
    before = ledger.read_bytes()
    monkeypatch.setattr(ledger_v5, "_verify_genesis_payload", lambda *_args: None)
    with pytest.raises(FileExistsError):
        ledger_v5.initialize_ledger(tmp_path, ledger, payload={"valid": True})
    assert ledger.read_bytes() == before


def test_no_init_and_unknown_event_fail_before_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / ledger_v5.LEDGER_RELATIVE
    lock = ledger.with_name(f"{ledger.name}.append.lock")
    with pytest.raises(ledger_v5.ContractError, match="not allowlisted"):
        ledger_v5.append_ledger_event(
            tmp_path, ledger, event_type="FORGED_EVENT", payload={}
        )
    monkeypatch.setattr(
        ledger_v5,
        "recompute_later_payload",
        lambda _root, _event, payload: dict(payload),
    )
    monkeypatch.setattr(ledger_v5, "validate_ledger", lambda *_args: [])
    with pytest.raises(ledger_v5.ContractError, match="must be initialized"):
        ledger_v5.append_ledger_event(
            tmp_path, ledger, event_type="CURVE_RESULT", payload={"valid": True}
        )
    assert not ledger.exists()
    assert not lock.exists()


def test_containment_fails_before_payload_or_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.jsonl"
    monkeypatch.setattr(
        ledger_v5,
        "_verify_genesis_payload",
        lambda *_args: pytest.fail("payload ran before containment"),
    )
    with pytest.raises(ledger_v5.ContractError, match="escapes workspace"):
        ledger_v5.initialize_ledger(root, outside, payload={})
    with pytest.raises(ledger_v5.ContractError, match="escapes workspace"):
        ledger_v5.append_ledger_event(
            root, outside, event_type="CURVE_RESULT", payload={}
        )
    assert not outside.exists()


def test_existing_lock_blocks_writer_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / ledger_v5.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"immutable\n")
    before = ledger.read_bytes()
    lock = ledger.with_name(f"{ledger.name}.append.lock")
    lock.write_bytes(b"other")
    monkeypatch.setattr(
        ledger_v5,
        "recompute_later_payload",
        lambda _root, _event, payload: dict(payload),
    )
    monkeypatch.setattr(
        ledger_v5, "validate_ledger", lambda *_args: [{"event_sha256": "a" * 64}]
    )
    with pytest.raises(FileExistsError):
        ledger_v5.append_ledger_event(
            tmp_path, ledger, event_type="CURVE_RESULT", payload={"valid": True}
        )
    assert ledger.read_bytes() == before
    assert lock.read_bytes() == b"other"


def test_head_race_releases_owned_lock_without_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / ledger_v5.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"immutable\n")
    before = ledger.read_bytes()
    first = [{"event_sha256": "a" * 64}]
    raced = [*first, {"event_sha256": "b" * 64}]
    calls = iter((first, raced))
    monkeypatch.setattr(
        ledger_v5,
        "recompute_later_payload",
        lambda _root, _event, payload: dict(payload),
    )
    monkeypatch.setattr(ledger_v5, "validate_ledger", lambda *_args: next(calls))
    with pytest.raises(ledger_v5.ContractError, match="changed before lock"):
        ledger_v5.append_ledger_event(
            tmp_path, ledger, event_type="CURVE_RESULT", payload={"valid": True}
        )
    assert ledger.read_bytes() == before
    assert not ledger.with_name(f"{ledger.name}.append.lock").exists()


def test_forged_direct_payload_fails_before_lock(replay_root: dict[str, Any]) -> None:
    root = replay_root["root"]
    ledger = root / ledger_v5.LEDGER_RELATIVE
    payload = ledger_v5.build_curve_payload(root, replay_root["curve"])
    payload["decision"] = {"forged": True}
    with pytest.raises(ledger_v5.ContractError, match="differs from replay"):
        ledger_v5.append_ledger_event(
            root, ledger, event_type="CURVE_RESULT", payload=payload
        )
    assert not ledger.exists()
    assert not ledger.with_name(f"{ledger.name}.append.lock").exists()


def test_canonical_contract_v3_predecessor_and_failed_v4_pins() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = ledger_v5.load_contract(root)
    predecessor = ledger_v5.verify_predecessor(root, contract)
    failed = ledger_v5.verify_failed_v4_lineage(root, contract)
    assert predecessor["V3_PREDECESSOR_LEDGER"] == {
        "path": ledger_v5.V3_LEDGER_RELATIVE,
        "sha256": ledger_v5.V3_LEDGER_SHA256,
        "bytes": ledger_v5.V3_LEDGER_BYTES,
    }
    assert failed["V4_FAILED_REGISTRY"] == ledger_v5.V4_FAILED_REGISTRY
    assert failed["V4_PRE_INIT_QA"] == ledger_v5.V4_PRE_INIT_QA
    assert failed["V4_FAILURE_RECEIPT"] == ledger_v5.V4_FAILURE_RECEIPT
    assert contract["failed_v4_lineage"]["v4_retry_allowed"] is False


def test_v5_canonical_ledger_and_qa_are_absent_before_independent_qa() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / ledger_v5.LEDGER_RELATIVE).exists()
    assert not (root / ledger_v5.PRE_INIT_QA_RELATIVE).exists()
