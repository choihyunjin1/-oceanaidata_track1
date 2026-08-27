from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from ocean_goal import meaningful_score_ledger_v8 as ledger_v8
from ocean_goal import meaningful_score_ledger_v9 as ledger_v9
from scripts import run_meaningful_score_ledger_v9 as cli_v9

ROOT = Path(__file__).resolve().parents[1]
P3_EVIDENCE = ROOT / ledger_v9.P3_CORE_FILES["EVIDENCE"]["path"]
P1_EVIDENCE = (
    ROOT
    / "artifacts/p1_meaningful_learning_curve_generation_v1/learning_curve_evidence.json"
)
P2_EVIDENCE = (
    ROOT
    / "artifacts/p2_architecture_matched_stage_b_parser_correction_r1/learning_curve_evidence.json"
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_v9_pins_and_deep_replays_exact_initialized_v8_seq2() -> None:
    contract = ledger_v9.load_contract(ROOT)
    predecessor = ledger_v9.verify_predecessor(ROOT, contract)
    assert contract["predecessor_v8_ledger"] == {
        "path": ledger_v9.V8_LEDGER_RELATIVE,
        "sha256": ledger_v9.V8_LEDGER_SHA256,
        "bytes": 13231,
        "event_count": 2,
        "head_event_sha256": ledger_v9.V8_LEDGER_HEAD_SHA256,
    }
    assert predecessor["V8_LEDGER"]["event_count"] == 2
    assert predecessor["V8_LEDGER"]["head_event_sha256"] == (
        ledger_v9.V8_LEDGER_HEAD_SHA256
    )
    records = ledger_v8.validate_ledger(ROOT, ROOT / ledger_v9.V8_LEDGER_RELATIVE)
    assert len(records) == 2
    assert records[-1]["event_sha256"] == ledger_v9.V8_LEDGER_HEAD_SHA256


def test_frozen_v8_four_files_and_registry_are_byte_exact() -> None:
    expected = {
        "V8_CONTRACT": (
            4773,
            "269b3b3f42c6fe356cee9d5685e268131ff01950b84659b7869234579183f6d1",
        ),
        "V8_EVALUATOR": (
            32801,
            "c1f100d73e2c391ddaf7c61a76f12b2a7745a9ec5a420b9601e7b86de92f37bf",
        ),
        "V8_CLI": (
            7974,
            "322eb9775dc369d16cccb400c1c23121af0c91b44ea32f8e2d3341cb347a2b1c",
        ),
        "V8_TESTS": (
            16767,
            "e8bc6a9dc3012ebee0e4ebefa54ebff84dba0459b09be07ddbde6778ec54c4d7",
        ),
    }
    for role, pin in ledger_v9.V8_IMPLEMENTATION.items():
        path = ROOT / pin["path"]
        assert (path.stat().st_size, ledger_v9.sha256_file(path)) == expected[role]
    registry = ROOT / ledger_v9.V8_LEDGER_RELATIVE
    assert (registry.stat().st_size, ledger_v9.sha256_file(registry)) == (
        13231,
        ledger_v9.V8_LEDGER_SHA256,
    )


def test_exact_p3_239_file_lineage_is_verified() -> None:
    contract = ledger_v9.load_contract(ROOT)
    lineage = ledger_v9.verify_p3_gen5r4_lineage(ROOT, contract, P3_EVIDENCE)
    assert lineage["artifact_file_count"] == 239
    assert lineage["manifest_output_file_count"] == 237
    assert lineage["control_file_count"] == 3
    assert lineage["manifest_implementation_pin_count"] == 71
    assert lineage["manifest_input_pin_count"] == 26
    assert lineage["core_files"] == ledger_v9.P3_CORE_FILES
    assert lineage["execution_counts"] == {
        "models": 45,
        "blind_predictions": 45,
        "blind_raw_deltas": 45,
        "cell_commitments": 45,
        "raw_cell_commitments": 45,
        "fold_commitments": 3,
        "raw_fold_commitments": 3,
        "fit_cells": 45,
        "optimizer_steps": 10260,
    }
    assert lineage["candidate_created"] is False
    assert lineage["test_prediction_created"] is False
    assert lineage["official_upload_count"] == 0


def test_p3_curve_route_is_forced_research_only() -> None:
    payload = ledger_v9.build_curve_payload(ROOT, P3_EVIDENCE)
    assert payload["schema_version"] == (
        "meaningful_score_ledger_v9.p3_structure_matched_curve_result.v1"
    )
    assert payload["evidence"] == ledger_v9.P3_CORE_FILES["EVIDENCE"]
    decision = payload["decision"]
    assert decision["problem"] == "P3"
    assert decision["comparison_mode"] == ledger_v9.STRUCTURE_MODE
    assert decision["exact_official_incumbent_comparison"] is False
    assert decision["decision"] == "RESEARCH_ONLY"
    assert decision["local_qualification"] is False
    assert decision["passed"] is False
    assert decision["official_promotion"] is False
    assert decision["curve_alone_can_promote"] is False
    assert decision["official_score_route_allowed"] is False
    assert decision["upload_readiness_route_allowed"] is False
    assert payload["upload_performed"] is False


def test_p3_curve_values_are_preserved_without_promotion_reinterpretation() -> None:
    payload = ledger_v9.build_curve_payload(ROOT, P3_EVIDENCE)
    points = payload["decision"]["points"]
    assert [point["delta_candidate_minus_incumbent"] for point in points] == pytest.approx(
        [
            0.003914455833476493,
            0.01820222027063878,
            0.04931969456255847,
            0.06153022481517356,
            0.06729476403367496,
        ],
        abs=1e-15,
    )
    assert points[-1]["delta_ci90_candidate_minus_incumbent"] == pytest.approx(
        [0.032683423577773194, 0.10164674575032215], abs=1e-15
    )
    assert payload["decision"]["fold_deltas_candidate_minus_incumbent"] == pytest.approx(
        [0.10827323057022131, 0.04351888205695076, 0.06686365161621466],
        abs=1e-15,
    )


@pytest.mark.parametrize("path", [P1_EVIDENCE, P2_EVIDENCE])
def test_existing_curve_routes_return_v8_payload_unchanged(path: Path) -> None:
    assert ledger_v9.build_curve_payload(ROOT, path) == ledger_v8.build_curve_payload(ROOT, path)


def test_delegated_curve_object_is_not_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence.json"
    _write_json(evidence, {"problem": "P1", "comparison_mode": "EXACT_OFFICIAL_PREFIX_REFIT"})
    sentinel = {"v8": "same-object"}
    monkeypatch.setattr(ledger_v9, "_verify_inherited_policy", lambda _root: {})
    monkeypatch.setattr(ledger_v9.ledger_v8, "build_curve_payload", lambda *_args: sentinel)
    assert ledger_v9.build_curve_payload(tmp_path, evidence) is sentinel


def test_p3_structure_official_and_upload_routes_fail_before_delegation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure = tmp_path / "structure.json"
    other = tmp_path / "other.json"
    _write_json(structure, {"problem": "P3", "comparison_mode": ledger_v9.STRUCTURE_MODE})
    _write_json(other, {"problem": "P3", "comparison_mode": "EXACT_OFFICIAL_PREFIX_REFIT"})
    monkeypatch.setattr(
        ledger_v9.ledger_v8,
        "build_official_score_payload",
        lambda *_args: pytest.fail("v8 official delegate must not run"),
    )
    monkeypatch.setattr(
        ledger_v9.ledger_v8,
        "build_upload_readiness_payload",
        lambda *_args: pytest.fail("v8 upload delegate must not run"),
    )
    with pytest.raises(ledger_v9.ContractError, match="official-score route is forbidden"):
        ledger_v9.build_official_score_payload(tmp_path, structure, other)
    with pytest.raises(ledger_v9.ContractError, match="upload-readiness route is forbidden"):
        ledger_v9.build_upload_readiness_payload(tmp_path, structure, None)
    with pytest.raises(ledger_v9.ContractError, match="upload-readiness route is forbidden"):
        ledger_v9.build_upload_readiness_payload(tmp_path, other, structure)


def test_exact_official_and_upload_routes_return_v8_objects_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence.json"
    curve = tmp_path / "curve.json"
    receipt = tmp_path / "receipt.json"
    for path in (evidence, curve, receipt):
        _write_json(path, {"problem": "P1", "comparison_mode": "EXACT_OFFICIAL_PREFIX_REFIT"})
    official = {"v8": "official"}
    upload = {"v8": "upload"}
    monkeypatch.setattr(ledger_v9, "_verify_inherited_policy", lambda _root: {})
    monkeypatch.setattr(
        ledger_v9.ledger_v8, "build_official_score_payload", lambda *_args: official
    )
    monkeypatch.setattr(
        ledger_v9.ledger_v8, "build_upload_readiness_payload", lambda *_args: upload
    )
    assert ledger_v9.build_official_score_payload(tmp_path, evidence, curve) is official
    assert ledger_v9.build_upload_readiness_payload(tmp_path, receipt, curve) is upload


def test_completion_delegates_v8_policy_over_v8_plus_v9_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "completion.json"
    _write_json(evidence, {"claim": "complete"})
    inherited = [{"seq": 1}, {"seq": 2}]
    local = [{"seq": 3}, {"seq": 4}]
    sentinel = {"v8": "completion-object"}
    captured: dict[str, Any] = {}

    def builder(
        _root: Path,
        _evidence: Path,
        *,
        prior_records: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        captured["prior"] = prior_records
        return sentinel

    monkeypatch.setattr(ledger_v9, "_verify_inherited_policy", lambda _root: {})
    monkeypatch.setattr(ledger_v9, "_validated_v8_records", lambda _root: inherited)
    monkeypatch.setattr(ledger_v9.ledger_v8, "build_goal_completion_payload", builder)
    result = ledger_v9.build_goal_completion_payload(
        tmp_path, evidence, prior_records=local
    )
    assert result is sentinel
    assert captured["prior"] == tuple(inherited + local)


def test_replay_rejects_forged_p3_promotion_fields() -> None:
    payload = ledger_v9.build_curve_payload(ROOT, P3_EVIDENCE)
    forged = copy.deepcopy(payload)
    forged["decision"]["decision"] = "PROMOTE"
    forged["decision"]["passed"] = True
    forged["decision"]["official_promotion"] = True
    forged["decision"]["curve_alone_can_promote"] = True
    with pytest.raises(ledger_v9.ContractError, match="differs from replay"):
        ledger_v9.recompute_later_payload(ROOT, "CURVE_RESULT", forged)


def test_alternate_copy_of_exact_p3_evidence_is_not_canonical(tmp_path: Path) -> None:
    copied = tmp_path / "evidence.json"
    copied.write_bytes(P3_EVIDENCE.read_bytes())
    with pytest.raises(ledger_v9.ContractError, match="only the canonical P3"):
        ledger_v9.build_curve_payload(ROOT, copied)


def test_point_schema_and_check_truth_mutations_fail_closed() -> None:
    evidence = json.loads(P3_EVIDENCE.read_text(encoding="utf-8"))
    bad_point = copy.deepcopy(evidence)
    bad_point["points"][0]["extra"] = 1
    with pytest.raises(ledger_v9.ContractError, match="point schema"):
        ledger_v9._validate_point_evidence(bad_point)
    bad_check = copy.deepcopy(evidence)
    bad_check["leakage_checks"][next(iter(bad_check["leakage_checks"]))] = False
    with pytest.raises(ledger_v9.ContractError, match="failed check"):
        ledger_v9._validate_point_evidence(bad_check)


def _patch_synthetic_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ledger_v9, "load_contract", lambda _root: {})
    monkeypatch.setattr(ledger_v9, "verify_predecessor", lambda *_args: {})
    monkeypatch.setattr(ledger_v9, "verify_p3_gen5r4_lineage", lambda *_args: {})
    monkeypatch.setattr(ledger_v9, "_verify_genesis_payload", lambda *_args: None)
    monkeypatch.setattr(
        ledger_v9,
        "recompute_later_payload",
        lambda _root, _event_type, payload, **_kwargs: dict(payload),
    )


def test_v9_chain_starts_at_seq3_and_partial_writes_are_robust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_synthetic_chain(monkeypatch)
    ledger = tmp_path / ledger_v9.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    real_write = os.write

    def partial_write(descriptor: int, data: bytes | memoryview) -> int:
        return real_write(descriptor, bytes(data[:7]))

    monkeypatch.setattr(ledger_v9.os, "write", partial_write)
    genesis = ledger_v9.initialize_ledger(tmp_path, ledger, payload={"genesis": "x" * 100})
    assert genesis["seq"] == 3
    assert genesis["previous_event_sha256"] == ledger_v9.V8_LEDGER_HEAD_SHA256
    event = ledger_v9.append_ledger_event(
        tmp_path,
        ledger,
        event_type="CURVE_RESULT",
        payload={"curve": "y" * 100},
    )
    assert event["seq"] == 4
    raw = ledger.read_bytes()
    assert raw.count(b"\n") == 2
    assert b"\r" not in raw
    assert len(ledger_v9.validate_ledger(tmp_path, ledger)) == 2


def test_containment_unknown_event_and_duplicate_init_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.jsonl"
    monkeypatch.setattr(
        ledger_v9,
        "_verify_genesis_payload",
        lambda *_args: pytest.fail("payload ran before containment"),
    )
    with pytest.raises(ledger_v9.ContractError, match="escapes workspace"):
        ledger_v9.initialize_ledger(root, outside, payload={})
    with pytest.raises(ledger_v9.ContractError, match="escapes workspace"):
        ledger_v9.append_ledger_event(root, outside, event_type="CURVE_RESULT", payload={})
    with pytest.raises(ledger_v9.ContractError, match="not allowlisted"):
        ledger_v9.append_ledger_event(root, root / ledger_v9.LEDGER_RELATIVE, event_type="X", payload={})
    ledger = root / ledger_v9.LEDGER_RELATIVE
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"existing\n")
    before = ledger.read_bytes()
    monkeypatch.setattr(ledger_v9, "_verify_genesis_payload", lambda *_args: None)
    with pytest.raises(FileExistsError):
        ledger_v9.initialize_ledger(root, ledger, payload={"valid": True})
    assert ledger.read_bytes() == before
    assert not outside.exists()


def test_all_open_sites_are_binary_o_excl_and_robust_write_only() -> None:
    source = Path(ledger_v9.__file__).read_text(encoding="utf-8")
    assert source.count("os.open(") == 3
    assert source.count("| O_BINARY") == 3
    assert source.count("os.O_EXCL | O_BINARY") == 2
    assert source.count("_write_all(descriptor") == 4
    assert "os.write(descriptor, encoded)" not in source


def test_strict_json_rejects_duplicate_and_nonfinite_values() -> None:
    with pytest.raises(ledger_v9.ContractError, match="duplicate JSON key"):
        ledger_v9._strict_json_loads('{"a":1,"a":2}')
    with pytest.raises(ledger_v9.ContractError, match="non-finite"):
        ledger_v9._strict_json_loads('{"a":NaN}')


def test_cli_check_is_read_only_and_reports_zero_operations() -> None:
    report = cli_v9.check(ROOT)
    assert report["status"] == "PASS_READ_ONLY"
    assert report["ledger_exists"] is False
    assert report["pre_init_qa_exists"] is False
    assert report["global_ledger_event_count"] == 2
    assert report["ledger_head_event_sha256"] == ledger_v9.V8_LEDGER_HEAD_SHA256
    assert report["p3_gen5r4_lineage"]["artifact_file_count"] == 239
    assert report["p3_structure_curve_decision"] == "RESEARCH_ONLY"
    assert report["p3_structure_official_score_allowed"] is False
    assert report["p3_structure_upload_readiness_allowed"] is False
    assert (report["writes"], report["fits"], report["predictions"], report["uploads"]) == (
        0,
        0,
        0,
        0,
    )


def test_static_v9_has_no_receipt_ledger_lock_or_personal_path() -> None:
    for relative in (
        ledger_v9.LEDGER_RELATIVE,
        ledger_v9.PRE_INIT_QA_RELATIVE,
        f"{ledger_v9.LEDGER_RELATIVE}.append.lock",
    ):
        assert not (ROOT / relative).exists()
    paths = [
        ROOT / ledger_v9.CONTRACT_RELATIVE,
        Path(ledger_v9.__file__),
        ROOT / "scripts/run_meaningful_score_ledger_v9.py",
        Path(__file__),
    ]
    unix_home = "/" + "home/"
    personal = re.compile(
        rf"(?i)(?:[a-z]:[\\/]+users[\\/]+[^\\/]+|{re.escape(unix_home)}[^/]+)"
    )
    assert all(personal.search(path.read_text(encoding="utf-8")) is None for path in paths)


def test_compiled_contract_pin_matches_current_bytes() -> None:
    path = ROOT / ledger_v9.CONTRACT_RELATIVE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ledger_v9.CONTRACT_SHA256
