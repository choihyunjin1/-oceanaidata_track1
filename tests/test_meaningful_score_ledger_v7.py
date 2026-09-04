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
from ocean_goal import meaningful_score_ledger_v7 as ledger_v7

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


def _architecture_curve(*, local_qualification: bool = True) -> dict[str, Any]:
    return {
        "problem": "P2",
        "comparison_mode": ledger_v7.ARCHITECTURE_MODE,
        "exact_official_incumbent_comparison": False,
        "decision": "RESEARCH_ONLY",
        "passed": False,
        "official_promotion": False,
        "curve_alone_can_promote": False,
        "local_qualification": local_qualification,
    }


def _architecture_official_evidence() -> dict[str, Any]:
    return {
        "problem": "P2",
        "comparison_mode": ledger_v7.ARCHITECTURE_MODE,
    }


def _architecture_upload_receipt() -> dict[str, Any]:
    return {
        "problem": "P2",
        "role": ledger_v7.ARCHITECTURE_UPLOAD_ROLE,
        "comparison_mode": ledger_v7.ARCHITECTURE_MODE,
    }


def _patch_policy_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = {"p2_architecture_matched_policy": ledger_v7._expected_p2_policy()}
    monkeypatch.setattr(
        ledger_v7,
        "_verify_policy_dependencies",
        lambda _root: (contract, {"SEAL": {"sha256": "s" * 64}}),
    )


def test_v7_pins_v5_seq9_and_frozen_uninitialized_v6() -> None:
    contract = ledger_v7.load_contract(ROOT)
    verified = ledger_v7.verify_predecessor(ROOT, contract)
    assert contract["predecessor_ledger"] == {
        "path": ledger_v7.V5_LEDGER_RELATIVE,
        "sha256": ledger_v7.V5_LEDGER_SHA256,
        "bytes": ledger_v7.V5_LEDGER_BYTES,
        "event_count": 9,
        "head_event_sha256": ledger_v7.V5_LEDGER_HEAD_SHA256,
    }
    assert verified["V6_STATIC_IMPLEMENTATION"] == ledger_v7.V6_IMPLEMENTATION
    for relative in (
        ledger_v6.LEDGER_RELATIVE,
        ledger_v6.PRE_INIT_QA_RELATIVE,
        f"{ledger_v6.LEDGER_RELATIVE}.append.lock",
    ):
        assert not (ROOT / relative).exists()


def test_v6_four_file_bytes_are_unchanged() -> None:
    expected = {
        "V6_CONTRACT": (
            6560,
            "a0aa5b04d57cc74552b58f94f6eb79c078ce48722e7351f21f3007f0ca8264f8",
        ),
        "V6_EVALUATOR": (
            36256,
            "2162695cb178aec1f639a66d9f07b9156977d4154cafc0449d281836ce7146b3",
        ),
        "V6_CLI": (
            7766,
            "2f5065db2e4300afa53375973cb09869623cb9b4fd15cedac890d9e7be392620",
        ),
        "V6_TESTS": (
            13921,
            "f0872cf214060d1185c7cbb8caeac82c10205ff4f3df4c3752f43570bd768d5a",
        ),
    }
    for role, pin in ledger_v7.V6_IMPLEMENTATION.items():
        path = ROOT / pin["path"]
        assert (path.stat().st_size, ledger_v7.sha256_file(path)) == expected[role]


def test_contract_separates_score_record_from_meaningful_promotion() -> None:
    policy = ledger_v7.load_contract(ROOT)["p2_architecture_matched_policy"]
    official = policy["official_score_result"]
    upload = policy["upload_readiness"]
    assert official["score_incumbent_policy"] == (
        "ALLOW_GENERAL_PAIRED_OFFICIAL_SCORE_RECORD_ONLY"
    )
    assert official["score_incumbent_updates_may_reflect_v3_operational_checks"] is True
    assert official["meaningful_incumbent_updates"] is False
    assert official["official_promotion"] is False
    assert official["goal_problem_milestone_complete"] is False
    assert upload["upload_ready"] is False
    assert upload["no_v3_readiness_delegation"] is True


@pytest.mark.parametrize("relative", EXACT_CURVE_EVIDENCE)
def test_p1_p3_curve_payloads_remain_bit_for_bit_v5(relative: str) -> None:
    path = ROOT / relative
    assert ledger_v7.build_curve_payload(ROOT, path) == ledger_v5.build_curve_payload(ROOT, path)


def test_p2_curve_payload_remains_exact_v6_compatibility_result() -> None:
    payload = ledger_v7.build_curve_payload(ROOT, P2_EVIDENCE)
    assert payload == ledger_v6.build_curve_payload(ROOT, P2_EVIDENCE)
    assert payload["decision"]["decision"] == "RESEARCH_ONLY"
    assert payload["decision"]["passed"] is False
    assert payload["decision"]["official_promotion"] is False


def test_forged_in_memory_official_promotion_proof_is_policy_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_path = tmp_path / "official.json"
    curve_path = tmp_path / "curve.json"
    _write_json(evidence_path, _architecture_official_evidence())
    _write_json(curve_path, _architecture_curve())
    _patch_policy_dependencies(monkeypatch)
    monkeypatch.setattr(ledger_v7.scoring, "load_contract", lambda _root: {})
    monkeypatch.setattr(
        ledger_v7.scoring,
        "verify_official_evidence_pins",
        lambda *_args: {"baseline": {"verified": True}, "challenger": {"verified": True}},
    )
    monkeypatch.setattr(
        ledger_v7.scoring,
        "evaluate_official_score",
        lambda *_args: {
            "problem": "P2",
            "comparison_mode": ledger_v7.ARCHITECTURE_MODE,
            "decision": "MEANINGFUL_PROMOTED_PROVISIONAL",
            "score_incumbent_updates": True,
            "meaningful_incumbent_updates": True,
            "goal_problem_milestone_complete": True,
            "official_promotion": True,
            "meaningful_checks": {"forged_all_true": True},
        },
    )
    payload = ledger_v7.build_official_score_payload(tmp_path, evidence_path, curve_path)
    decision = payload["decision"]
    assert decision["score_incumbent_updates"] is True
    assert decision["score_incumbent_policy"] == (
        "ALLOW_GENERAL_PAIRED_OFFICIAL_SCORE_RECORD_ONLY"
    )
    assert decision["decision"] == ledger_v7.P2_OFFICIAL_DECISION
    assert decision["meaningful_incumbent_updates"] is False
    assert decision["official_promotion"] is False
    assert decision["goal_problem_milestone_complete"] is False
    assert decision["meaningful_checks"][
        "v7_research_only_policy_allows_meaningful_promotion"
    ] is False
    assert payload["upload_performed"] is False
    assert ledger_v7.recompute_later_payload(
        tmp_path, "OFFICIAL_SCORE_RESULT", payload
    ) == payload

    forged = copy.deepcopy(payload)
    forged["decision"]["meaningful_incumbent_updates"] = True
    forged["decision"]["official_promotion"] = True
    forged["decision"]["decision"] = "MEANINGFUL_PROMOTED_PROVISIONAL"
    with pytest.raises(ledger_v7.ContractError, match="differs from replay"):
        ledger_v7.recompute_later_payload(tmp_path, "OFFICIAL_SCORE_RESULT", forged)


@pytest.mark.parametrize(
    ("evidence_mode", "curve_mode"),
    [
        (ledger_v7.ARCHITECTURE_MODE, ledger_v7.scoring.EXACT_MODE),
        (ledger_v7.scoring.EXACT_MODE, ledger_v7.ARCHITECTURE_MODE),
    ],
)
def test_official_mode_downgrade_cannot_reach_v5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence_mode: str,
    curve_mode: str,
) -> None:
    evidence = _architecture_official_evidence()
    curve = _architecture_curve()
    evidence["comparison_mode"] = evidence_mode
    curve["comparison_mode"] = curve_mode
    evidence_path = tmp_path / "official.json"
    curve_path = tmp_path / "curve.json"
    _write_json(evidence_path, evidence)
    _write_json(curve_path, curve)
    monkeypatch.setattr(
        ledger_v7.ledger_v5,
        "build_official_score_payload",
        lambda *_args: pytest.fail("downgraded architecture evidence reached v5"),
    )
    with pytest.raises(ledger_v7.ContractError, match="cannot be downgraded"):
        ledger_v7.build_official_score_payload(tmp_path, evidence_path, curve_path)


def test_architecture_upload_readiness_is_explicitly_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "receipt.json"
    curve_path = tmp_path / "curve.json"
    _write_json(receipt_path, _architecture_upload_receipt())
    _write_json(curve_path, _architecture_curve())
    _patch_policy_dependencies(monkeypatch)
    monkeypatch.setattr(
        ledger_v7.ledger_v5,
        "build_upload_readiness_payload",
        lambda *_args: pytest.fail("architecture upload readiness delegated to v5"),
    )
    payload = ledger_v7.build_upload_readiness_payload(tmp_path, receipt_path, curve_path)
    readiness = payload["readiness"]
    assert readiness["status"] == ledger_v7.P2_UPLOAD_STATUS
    assert readiness["upload_ready"] is False
    assert readiness["upload_performed"] is False
    assert payload["upload_performed"] is False
    assert ledger_v7.recompute_later_payload(tmp_path, "UPLOAD_READINESS", payload) == payload

    forged = copy.deepcopy(payload)
    forged["readiness"]["status"] = "UPLOAD_READY_PENDING_SEPARATE_PLATFORM_ACTION"
    forged["readiness"]["upload_ready"] = True
    with pytest.raises(ledger_v7.ContractError, match="differs from replay"):
        ledger_v7.recompute_later_payload(tmp_path, "UPLOAD_READINESS", forged)


def test_forged_curve_promotion_cannot_enter_architecture_post_curve_routes(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "official.json"
    receipt_path = tmp_path / "receipt.json"
    curve_path = tmp_path / "curve.json"
    _write_json(evidence_path, _architecture_official_evidence())
    _write_json(receipt_path, _architecture_upload_receipt())
    curve = _architecture_curve()
    curve["official_promotion"] = True
    curve["passed"] = True
    _write_json(curve_path, curve)
    with pytest.raises(ledger_v7.ContractError, match="research-only policy"):
        ledger_v7.build_official_score_payload(tmp_path, evidence_path, curve_path)
    with pytest.raises(ledger_v7.ContractError, match="research-only policy"):
        ledger_v7.build_upload_readiness_payload(tmp_path, receipt_path, curve_path)


def test_nonarchitecture_official_and_upload_routes_return_exact_v5_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_path = tmp_path / "official.json"
    receipt_path = tmp_path / "receipt.json"
    curve_path = tmp_path / "curve.json"
    _write_json(evidence_path, {"problem": "P1"})
    _write_json(receipt_path, {"problem": "P1", "role": "IMMUTABLE_BASELINE_ANCHOR"})
    _write_json(curve_path, {"problem": "P1", "comparison_mode": ledger_v7.scoring.EXACT_MODE})
    official_sentinel = {"v5": "official"}
    upload_sentinel = {"v5": "upload"}
    monkeypatch.setattr(
        ledger_v7.ledger_v5,
        "build_official_score_payload",
        lambda *_args: official_sentinel,
    )
    monkeypatch.setattr(
        ledger_v7.ledger_v5,
        "build_upload_readiness_payload",
        lambda *_args: upload_sentinel,
    )
    assert (
        ledger_v7.build_official_score_payload(tmp_path, evidence_path, curve_path)
        is official_sentinel
    )
    assert (
        ledger_v7.build_upload_readiness_payload(tmp_path, receipt_path, None)
        is upload_sentinel
    )


def _patch_synthetic_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ledger_v7, "load_contract", lambda _root: {})
    monkeypatch.setattr(ledger_v7, "verify_predecessor", lambda *_args: {})
    monkeypatch.setattr(ledger_v7, "verify_p2_stage_a_v3_lineage", lambda *_args: {})
    monkeypatch.setattr(ledger_v7, "_verify_genesis_payload", lambda *_args: None)
    monkeypatch.setattr(
        ledger_v7,
        "recompute_later_payload",
        lambda _root, _event_type, payload: dict(payload),
    )


def test_robust_binary_write_loop_survives_partial_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_synthetic_chain(monkeypatch)
    ledger = tmp_path / ledger_v7.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    real_write = os.write

    def partial_write(descriptor: int, data: bytes | memoryview) -> int:
        return real_write(descriptor, bytes(data[:7]))

    monkeypatch.setattr(ledger_v7.os, "write", partial_write)
    ledger_v7.initialize_ledger(tmp_path, ledger, payload={"genesis": "x" * 100})
    ledger_v7.append_ledger_event(
        tmp_path,
        ledger,
        event_type="CURVE_RESULT",
        payload={"curve": "y" * 100},
    )
    raw = ledger.read_bytes()
    assert raw.count(b"\n") == 2
    assert b"\r" not in raw
    assert len(ledger_v7.validate_ledger(tmp_path, ledger)) == 2


def test_all_three_open_sites_are_binary_o_excl_and_write_loop_only() -> None:
    source = Path(ledger_v7.__file__).read_text(encoding="utf-8")
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
        ledger_v7,
        "_verify_genesis_payload",
        lambda *_args: pytest.fail("payload ran before containment"),
    )
    with pytest.raises(ledger_v7.ContractError, match="escapes workspace"):
        ledger_v7.initialize_ledger(root, outside, payload={})
    with pytest.raises(ledger_v7.ContractError, match="escapes workspace"):
        ledger_v7.append_ledger_event(root, outside, event_type="CURVE_RESULT", payload={})
    with pytest.raises(ledger_v7.ContractError, match="not allowlisted"):
        ledger_v7.append_ledger_event(
            root,
            root / ledger_v7.LEDGER_RELATIVE,
            event_type="FORGED_EVENT",
            payload={},
        )
    ledger = root / ledger_v7.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"existing\n")
    before = ledger.read_bytes()
    monkeypatch.setattr(ledger_v7, "_verify_genesis_payload", lambda *_args: None)
    with pytest.raises(FileExistsError):
        ledger_v7.initialize_ledger(root, ledger, payload={"valid": True})
    assert ledger.read_bytes() == before
    assert not outside.exists()


def test_superseded_v6_state_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = {
        "predecessor_ledger": ledger_v7._predecessor_anchor(),
        "p2_stage_a_v3_lineage": {},
    }
    monkeypatch.setattr(ledger_v7, "_verify_v6_implementation", lambda *_args: {})
    monkeypatch.setattr(
        ledger_v7.ledger_v6,
        "load_contract",
        lambda _root: {"p2_stage_a_v3_lineage": {}},
    )
    superseded = tmp_path / ledger_v6.LEDGER_RELATIVE
    superseded.parent.mkdir(parents=True)
    superseded.write_bytes(b"forbidden\n")
    with pytest.raises(ledger_v7.ContractError, match="must remain uninitialized"):
        ledger_v7.verify_predecessor(tmp_path, contract)


def test_static_v7_has_no_receipt_ledger_lock_or_personal_path() -> None:
    for relative in (
        ledger_v7.LEDGER_RELATIVE,
        ledger_v7.PRE_INIT_QA_RELATIVE,
        f"{ledger_v7.LEDGER_RELATIVE}.append.lock",
    ):
        assert not (ROOT / relative).exists()
    paths = [
        ROOT / ledger_v7.CONTRACT_RELATIVE,
        Path(ledger_v7.__file__),
        ROOT / "scripts/run_meaningful_score_ledger_v7.py",
        Path(__file__),
    ]
    unix_home = "/" + "home/"
    personal = re.compile(
        rf"(?i)(?:[a-z]:[\\/]+users[\\/]+[^\\/]+|{re.escape(unix_home)}[^/]+)"
    )
    assert all(personal.search(path.read_text(encoding="utf-8")) is None for path in paths)
    assert ledger_v7.ALL_EVENT_TYPES == {
        "GOAL_INITIALIZED",
        "CURVE_RESULT",
        "OFFICIAL_SCORE_RESULT",
        "UPLOAD_READINESS",
        "GOAL_COMPLETION",
    }
