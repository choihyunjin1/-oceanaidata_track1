from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from ocean_goal import meaningful_score_ledger_v5 as ledger_v5
from ocean_goal import meaningful_score_ledger_v6 as ledger_v6

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


def test_canonical_v5_seq9_anchor_and_p2_stage_a_v3_lineage() -> None:
    contract = ledger_v6.load_contract(ROOT)
    predecessor = ledger_v6.verify_predecessor(ROOT, contract)
    lineage = ledger_v6.verify_p2_stage_a_v3_lineage(ROOT, contract)
    assert contract["predecessor_ledger"] == {
        "path": ledger_v6.V5_LEDGER_RELATIVE,
        "sha256": ledger_v6.V5_LEDGER_SHA256,
        "bytes": ledger_v6.V5_LEDGER_BYTES,
        "event_count": 9,
        "head_event_sha256": ledger_v6.V5_LEDGER_HEAD_SHA256,
    }
    assert predecessor["V5_PREDECESSOR_LEDGER"] == {
        "path": ledger_v6.V5_LEDGER_RELATIVE,
        "sha256": ledger_v6.V5_LEDGER_SHA256,
        "bytes": ledger_v6.V5_LEDGER_BYTES,
    }
    assert set(lineage) == ledger_v6.STAGE_A_ROLES
    assert lineage["SEAL"]["sha256"] == (
        "315e52da6b7f4ab7bf3a1970301b4391b9f31611da6187667760ed0d3a501ba2"
    )
    assert lineage["MANIFEST"]["sha256"] == (
        "bb21341f94b866be7e2d6dc0e0f0b78b9c84a1898d4844ff75637f95798f21c6"
    )
    assert {lineage[role]["sha256"] for role in ledger_v6.OOF_ROLES_BY_FRACTION.values()} == {
        "8042676cda285a92e01c286a871ef11d323a672836d1e597cd5aa749fb85e626",
        "870d57779a1c5ab66ac78bc8525baa55ba33a7b10544b00e87837f4e50335168",
        "58a62e0c1944eccfcc9bb909747f647ce11b6267346be1f91071fc37bd0fbaaf",
        "1bacd0972bae51a50f9f7e04dfd5ba2d36c798427048b6195f9ca196a1196fe8",
        "4d663b40a4053bff9d5fcbe5cdc6aaa58b58d75aefbc1a9a0010420e0672bb06",
    }


@pytest.mark.parametrize("relative", EXACT_CURVE_EVIDENCE)
def test_p1_p3_exact_curve_payloads_are_bit_for_bit_v5(relative: str) -> None:
    path = ROOT / relative
    assert ledger_v6.build_curve_payload(ROOT, path) == ledger_v5.build_curve_payload(ROOT, path)


def test_p2_v3_compatibility_replays_as_central_research_only() -> None:
    payload = ledger_v6.build_curve_payload(ROOT, P2_EVIDENCE)
    decision = payload["decision"]
    assert payload["schema_version"] == "meaningful_score_ledger_v6.curve_result.v1"
    assert decision["problem"] == "P2"
    assert decision["comparison_mode"] == ledger_v6.ARCHITECTURE_MODE
    assert decision["decision"] == "RESEARCH_ONLY"
    assert decision["passed"] is False
    assert decision["official_promotion"] is False
    assert isinstance(decision["local_qualification"], bool)
    assert payload["upload_performed"] is False
    assert set(payload["evidence_pins"]["p2_stage_a_v3_lineage"]) == ledger_v6.STAGE_A_ROLES
    assert ledger_v6.recompute_later_payload(ROOT, "CURVE_RESULT", payload) == payload


def test_p2_preserves_local_qualification_but_never_promotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ledger_v6.scoring.evaluate_learning_curve

    def qualify(contract: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
        decision = original(contract, evidence)
        return {**decision, "local_qualification": True, "passed": True}

    monkeypatch.setattr(ledger_v6.scoring, "evaluate_learning_curve", qualify)
    decision = ledger_v6.build_curve_payload(ROOT, P2_EVIDENCE)["decision"]
    assert decision["local_qualification"] is True
    assert decision["decision"] == "RESEARCH_ONLY"
    assert decision["passed"] is False
    assert decision["official_promotion"] is False


def test_p2_forged_reference_binding_and_schema_are_rejected() -> None:
    evidence = json.loads(P2_EVIDENCE.read_text(encoding="utf-8"))
    evidence["reference_binding"]["reference_seal"]["sha256"] = "0" * 64
    with pytest.raises(ledger_v6.ContractError, match="reference_binding differs"):
        ledger_v6._build_p2_architecture_curve_payload(ROOT, P2_EVIDENCE, evidence)

    evidence = json.loads(P2_EVIDENCE.read_text(encoding="utf-8"))
    evidence["schema_version"] = "p2_architecture_matched_stage_b.learning_curve_evidence.v4"
    with pytest.raises(ledger_v6.ContractError, match="must use Stage-B schema v3"):
        ledger_v6._build_p2_architecture_curve_payload(ROOT, P2_EVIDENCE, evidence)


def test_lineage_pin_forgery_is_rejected() -> None:
    contract = copy.deepcopy(ledger_v6.load_contract(ROOT))
    contract["p2_stage_a_v3_lineage"]["OOF_070"]["sha256"] = "0" * 64
    with pytest.raises(ledger_v6.ContractError, match="SHA or size mismatch"):
        ledger_v6.verify_p2_stage_a_v3_lineage(ROOT, contract)


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

    def pin(root: Path, path: Path) -> dict[str, Any]:
        return ledger_v6._file_pin_for_path(root, path)

    monkeypatch.setattr(
        ledger_v6.ledger_v5,
        "build_curve_payload",
        lambda root, path: {
            "schema_version": "v5.curve",
            "evidence": pin(root, path),
            "decision": {"value": json.loads(path.read_text())["value"]},
        },
    )
    monkeypatch.setattr(
        ledger_v6.ledger_v5,
        "build_official_score_payload",
        lambda root, evidence, decision: {
            "schema_version": "v5.official",
            "evidence": pin(root, evidence),
            "curve_decision": pin(root, decision),
            "decision": {"value": json.loads(evidence.read_text())["value"]},
        },
    )
    monkeypatch.setattr(
        ledger_v6.ledger_v5,
        "build_upload_readiness_payload",
        lambda root, receipt, decision: {
            "schema_version": "v5.upload",
            "receipt": pin(root, receipt),
            "curve_decision": None if decision is None else pin(root, decision),
            "readiness": {"value": json.loads(receipt.read_text())["value"]},
        },
    )
    monkeypatch.setattr(
        ledger_v6.ledger_v5,
        "build_goal_completion_payload",
        lambda root, path: {
            "schema_version": "v5.completion",
            "evidence": pin(root, path),
            "decision": {"value": json.loads(path.read_text())["value"]},
        },
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
        payload = ledger_v6.build_curve_payload(root, replay_root["curve"])
    elif builder == "official":
        payload = ledger_v6.build_official_score_payload(
            root, replay_root["official"], replay_root["decision"]
        )
    elif builder == "upload":
        payload = ledger_v6.build_upload_readiness_payload(
            root, replay_root["receipt"], replay_root["decision"]
        )
    else:
        payload = ledger_v6.build_goal_completion_payload(root, replay_root["completion"])
    assert ledger_v6.recompute_later_payload(root, event_type, payload) == payload
    forged = copy.deepcopy(payload)
    forged[tamper] = {"forged": True}
    with pytest.raises(ledger_v6.ContractError, match="differs from replay"):
        ledger_v6.recompute_later_payload(root, event_type, forged)


def _patch_synthetic_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ledger_v6, "load_contract", lambda _root: {})
    monkeypatch.setattr(ledger_v6, "verify_predecessor", lambda *_args: {})
    monkeypatch.setattr(ledger_v6, "verify_p2_stage_a_v3_lineage", lambda *_args: {})
    monkeypatch.setattr(ledger_v6, "_verify_genesis_payload", lambda *_args: None)
    monkeypatch.setattr(
        ledger_v6,
        "recompute_later_payload",
        lambda _root, _event_type, payload: dict(payload),
    )


def test_robust_binary_write_loop_survives_partial_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_synthetic_chain(monkeypatch)
    ledger = tmp_path / ledger_v6.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    real_write = os.write

    def partial_write(descriptor: int, data: bytes | memoryview) -> int:
        return real_write(descriptor, bytes(data[:7]))

    monkeypatch.setattr(ledger_v6.os, "write", partial_write)
    ledger_v6.initialize_ledger(tmp_path, ledger, payload={"genesis": "x" * 100})
    ledger_v6.append_ledger_event(
        tmp_path,
        ledger,
        event_type="CURVE_RESULT",
        payload={"curve": "y" * 100},
    )
    raw = ledger.read_bytes()
    assert raw.count(b"\n") == 2
    assert b"\r" not in raw
    assert len(ledger_v6.validate_ledger(tmp_path, ledger)) == 2


def test_all_three_open_sites_are_binary_o_excl_and_write_loop_only() -> None:
    source = Path(ledger_v6.__file__).read_text(encoding="utf-8")
    assert source.count("os.open(") == 3
    assert source.count("| O_BINARY") == 3
    assert source.count("os.O_EXCL | O_BINARY") == 2
    assert source.count("_write_all(descriptor") == 4
    assert "os.write(descriptor, encoded)" not in source


def test_duplicate_init_o_excl_preserves_existing_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / ledger_v6.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"existing\n")
    before = ledger.read_bytes()
    monkeypatch.setattr(ledger_v6, "_verify_genesis_payload", lambda *_args: None)
    with pytest.raises(FileExistsError):
        ledger_v6.initialize_ledger(tmp_path, ledger, payload={"valid": True})
    assert ledger.read_bytes() == before


def test_containment_and_unknown_event_fail_before_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.jsonl"
    monkeypatch.setattr(
        ledger_v6,
        "_verify_genesis_payload",
        lambda *_args: pytest.fail("payload ran before containment"),
    )
    with pytest.raises(ledger_v6.ContractError, match="escapes workspace"):
        ledger_v6.initialize_ledger(root, outside, payload={})
    with pytest.raises(ledger_v6.ContractError, match="escapes workspace"):
        ledger_v6.append_ledger_event(root, outside, event_type="CURVE_RESULT", payload={})
    with pytest.raises(ledger_v6.ContractError, match="not allowlisted"):
        ledger_v6.append_ledger_event(
            root,
            root / ledger_v6.LEDGER_RELATIVE,
            event_type="FORGED_EVENT",
            payload={},
        )
    assert not outside.exists()


def test_forged_p2_payload_fails_before_canonical_lock() -> None:
    ledger = ROOT / ledger_v6.LEDGER_RELATIVE
    lock = ledger.with_name(f"{ledger.name}.append.lock")
    assert not ledger.exists()
    assert not lock.exists()
    payload = ledger_v6.build_curve_payload(ROOT, P2_EVIDENCE)
    payload["decision"] = {"forged": True}
    with pytest.raises(ledger_v6.ContractError, match="differs from replay"):
        ledger_v6.append_ledger_event(
            ROOT, ledger, event_type="CURVE_RESULT", payload=payload
        )
    assert not ledger.exists()
    assert not lock.exists()


def test_strict_json_rejects_duplicate_and_nonfinite_values() -> None:
    with pytest.raises(ledger_v6.ContractError, match="duplicate JSON key"):
        ledger_v6._strict_json_loads('{"x":1,"x":2}')
    with pytest.raises(ledger_v6.ContractError, match="non-finite JSON"):
        ledger_v6._strict_json_loads('{"x":NaN}')


def test_static_revision_has_no_canonical_receipt_ledger_or_personal_path() -> None:
    assert not (ROOT / ledger_v6.LEDGER_RELATIVE).exists()
    assert not (ROOT / ledger_v6.PRE_INIT_QA_RELATIVE).exists()
    paths = [
        ROOT / ledger_v6.CONTRACT_RELATIVE,
        Path(ledger_v6.__file__),
        ROOT / "scripts/run_meaningful_score_ledger_v6.py",
        Path(__file__),
    ]
    unix_home = "/" + "home/"
    personal = re.compile(
        rf"(?i)(?:[a-z]:[\\/]+users[\\/]+[^\\/]+|{re.escape(unix_home)}[^/]+)"
    )
    assert all(personal.search(path.read_text(encoding="utf-8")) is None for path in paths)
    assert ledger_v6.ALL_EVENT_TYPES == {
        "GOAL_INITIALIZED",
        "CURVE_RESULT",
        "OFFICIAL_SCORE_RESULT",
        "UPLOAD_READINESS",
        "GOAL_COMPLETION",
    }
