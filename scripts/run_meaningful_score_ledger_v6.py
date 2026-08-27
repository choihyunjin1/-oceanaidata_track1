"""Canonical read/replay/append CLI for the P2 Stage-A v3 compatibility ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ocean_goal.meaningful_score_ledger_v6 import (
    CONTRACT_RELATIVE,
    CONTRACT_SHA256,
    LEDGER_RELATIVE,
    PRE_INIT_QA_RELATIVE,
    ContractError,
    append_ledger_event,
    build_curve_payload,
    build_genesis_payload,
    build_goal_completion_payload,
    build_official_score_payload,
    build_upload_readiness_payload,
    current_implementation_pins,
    initialize_ledger,
    load_contract,
    sha256_file,
    validate_ledger,
    verify_p2_stage_a_v3_lineage,
    verify_predecessor,
)


def _workspace(root: Path) -> Path:
    return root.resolve(strict=True)


def _ledger(root: Path) -> Path:
    return (_workspace(root) / LEDGER_RELATIVE).resolve(strict=False)


def _receipt(root: Path) -> Path:
    return (_workspace(root) / PRE_INIT_QA_RELATIVE).resolve(strict=False)


def _source_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else _workspace(root) / path


def check(root: Path) -> dict[str, Any]:
    contract = load_contract(root)
    predecessor = verify_predecessor(root, contract)
    lineage = verify_p2_stage_a_v3_lineage(root, contract)
    records = validate_ledger(root, _ledger(root))
    contract_path = (_workspace(root) / CONTRACT_RELATIVE).resolve(strict=True)
    return {
        "schema_version": "meaningful_score_ledger_v6.check.v1",
        "status": "PASS_READ_ONLY",
        "ledger_contract": {
            "path": CONTRACT_RELATIVE,
            "sha256": sha256_file(contract_path),
            "bytes": contract_path.stat().st_size,
        },
        "compiled_contract_sha256": CONTRACT_SHA256,
        "predecessor_ledger_anchor": contract["predecessor_ledger"],
        "verified_predecessor_pins": predecessor,
        "p2_stage_a_v3_lineage_pins": lineage,
        "implementation_pins": current_implementation_pins(root),
        "event_allowlist": contract["event_protocol"],
        "p2_compatibility": contract["revision"],
        "ledger_path": LEDGER_RELATIVE,
        "ledger_exists": _ledger(root).exists(),
        "ledger_event_count": len(records),
        "ledger_head_event_sha256": records[-1]["event_sha256"] if records else None,
        "pre_init_qa_path": PRE_INIT_QA_RELATIVE,
        "pre_init_qa_exists": _receipt(root).exists(),
        "pre_init_qa_schema": "meaningful_score_ledger_v6.pre_init_qa.v1",
        "pre_init_qa_required_decision": "GO_INITIALIZE_V6_LEDGER",
        "windows_binary_open_enforced": True,
        "robust_write_loop_enforced": True,
        "writes": 0,
        "fits": 0,
        "predictions": 0,
        "uploads": 0,
    }


def initialize(root: Path, *, append: bool, qa_receipt: Path | None) -> dict[str, Any]:
    report = check(root)
    if report["ledger_exists"]:
        if append:
            raise ContractError("v6 ledger is already initialized; second init forbidden")
        return {**report, "status": "ALREADY_INITIALIZED_READ_ONLY"}
    if not append:
        return {**report, "status": "INITIALIZATION_CHECK_ONLY_NOT_APPENDED"}
    if qa_receipt is None:
        raise ContractError("canonical independent v6 pre-init QA receipt is required")
    payload = build_genesis_payload(root, _source_path(root, qa_receipt))
    record = initialize_ledger(root, _ledger(root), payload=payload)
    return {
        "status": "INITIALIZED",
        "record": record,
        "fit_performed": False,
        "prediction_performed": False,
        "upload_performed": False,
    }


def evaluate_curve(root: Path, evidence_path: Path, *, append: bool) -> dict[str, Any]:
    payload = build_curve_payload(root, _source_path(root, evidence_path))
    record = (
        append_ledger_event(root, _ledger(root), event_type="CURVE_RESULT", payload=payload)
        if append
        else None
    )
    return {"payload": payload, "ledger_appended": append, "record": record, "upload_performed": False}


def evaluate_score(
    root: Path, evidence_path: Path, curve_decision_path: Path, *, append: bool
) -> dict[str, Any]:
    payload = build_official_score_payload(
        root, _source_path(root, evidence_path), _source_path(root, curve_decision_path)
    )
    record = (
        append_ledger_event(
            root, _ledger(root), event_type="OFFICIAL_SCORE_RESULT", payload=payload
        )
        if append
        else None
    )
    return {"payload": payload, "ledger_appended": append, "record": record, "upload_performed": False}


def evaluate_upload_readiness(
    root: Path,
    receipt_path: Path,
    curve_decision_path: Path | None,
    *,
    append: bool,
) -> dict[str, Any]:
    payload = build_upload_readiness_payload(
        root,
        _source_path(root, receipt_path),
        None if curve_decision_path is None else _source_path(root, curve_decision_path),
    )
    record = (
        append_ledger_event(root, _ledger(root), event_type="UPLOAD_READINESS", payload=payload)
        if append
        else None
    )
    return {"payload": payload, "ledger_appended": append, "record": record, "upload_performed": False}


def evaluate_completion(root: Path, evidence_path: Path, *, append: bool) -> dict[str, Any]:
    payload = build_goal_completion_payload(root, _source_path(root, evidence_path))
    record = (
        append_ledger_event(root, _ledger(root), event_type="GOAL_COMPLETION", payload=payload)
        if append
        else None
    )
    return {"payload": payload, "ledger_appended": append, "record": record, "upload_performed": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("check")
    init = subparsers.add_parser("init")
    init.add_argument("--append", action="store_true")
    init.add_argument("--qa-receipt", type=Path)
    curve = subparsers.add_parser("curve")
    curve.add_argument("--evidence", type=Path, required=True)
    curve.add_argument("--append", action="store_true")
    score = subparsers.add_parser("score")
    score.add_argument("--evidence", type=Path, required=True)
    score.add_argument("--curve-decision", type=Path, required=True)
    score.add_argument("--append", action="store_true")
    upload = subparsers.add_parser("upload-readiness")
    upload.add_argument("--receipt", type=Path, required=True)
    upload.add_argument("--curve-decision", type=Path)
    upload.add_argument("--append", action="store_true")
    completion = subparsers.add_parser("completion")
    completion.add_argument("--evidence", type=Path, required=True)
    completion.add_argument("--append", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.mode == "check":
        result = check(args.root)
    elif args.mode == "init":
        result = initialize(args.root, append=args.append, qa_receipt=args.qa_receipt)
    elif args.mode == "curve":
        result = evaluate_curve(args.root, args.evidence, append=args.append)
    elif args.mode == "score":
        result = evaluate_score(args.root, args.evidence, args.curve_decision, append=args.append)
    elif args.mode == "upload-readiness":
        result = evaluate_upload_readiness(
            args.root, args.receipt, args.curve_decision, append=args.append
        )
    else:
        result = evaluate_completion(args.root, args.evidence, append=args.append)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
