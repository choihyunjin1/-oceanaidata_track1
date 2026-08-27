"""Initialize and validate the three-problem meaningful-score goal ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ocean_goal.meaningful_score import (
    append_ledger_event,
    evaluate_goal_completion,
    evaluate_learning_curve,
    evaluate_official_score,
    load_contract,
    sha256_file,
    validate_ledger,
    validate_upload_approval,
    verify_curve_evidence_pins,
    verify_initial_pins,
)

CONTRACT_RELATIVE = "configs/goals/meaningful_score_maximization_v2.json"
LEDGER_RELATIVE = "artifacts/meaningful_score_goal_v2/registry.jsonl"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _paths(root: Path) -> tuple[Path, Path]:
    workspace = root.resolve(strict=True)
    contract_path = (workspace / CONTRACT_RELATIVE).resolve(strict=True)
    ledger_path = (workspace / LEDGER_RELATIVE).resolve(strict=False)
    return contract_path, ledger_path


def initialize(root: Path) -> dict[str, Any]:
    contract_path, ledger_path = _paths(root)
    contract = load_contract(root, CONTRACT_RELATIVE)
    existing = validate_ledger(ledger_path)
    if existing:
        if existing[0]["event_type"] != "GOAL_INITIALIZED":
            raise RuntimeError("existing ledger was not initialized by this contract")
        return {"status": "ALREADY_INITIALIZED", "records": len(existing), "tail": existing[-1]}
    pins = verify_initial_pins(root, contract)
    record = append_ledger_event(
        ledger_path,
        event_type="GOAL_INITIALIZED",
        payload={
            "goal_id": contract["goal_id"],
            "status": "WAITING_OFFICIAL_WINDOW_RESEARCH_ACTIVE",
            "contract_path": CONTRACT_RELATIVE,
            "contract_sha256": sha256_file(contract_path),
            "verified_candidate_pins": pins,
            "official_window": {
                "status": contract["official_scoring"]["window_status"],
                "scoring_start_kst_date": contract["official_scoring"][
                    "confirmed_scoring_start_kst_date"
                ],
                "safe_final_model_deadline_kst_date": contract["official_scoring"][
                    "safe_final_model_deadline_kst_date"
                ],
                "daily_upload_limit_scope": "TEAM_WIDE",
                "daily_upload_limit": contract["official_scoring"]["daily_upload_limit_team_wide"],
            },
            "official_uploads": 0,
            "meaningful_promotions": {"P1": False, "P2": False, "P3": False},
        },
    )
    return {"status": "INITIALIZED", "record": record}


def evaluate_curve(root: Path, evidence_path: Path, *, append: bool) -> dict[str, Any]:
    _, ledger_path = _paths(root)
    contract = load_contract(root, CONTRACT_RELATIVE)
    evidence = _json(evidence_path.resolve(strict=True))
    evidence_pins = verify_curve_evidence_pins(root, evidence)
    decision = evaluate_learning_curve(contract, evidence)
    if append:
        append_ledger_event(
            ledger_path,
            event_type="LEARNING_CURVE_DECISION",
            payload={
                "evidence_sha256": sha256_file(evidence_path),
                "evidence_pins": evidence_pins,
                "decision": decision,
            },
        )
    return decision


def evaluate_score(
    root: Path,
    evidence_path: Path,
    curve_decision_path: Path,
    *,
    append: bool,
) -> dict[str, Any]:
    _, ledger_path = _paths(root)
    contract = load_contract(root, CONTRACT_RELATIVE)
    evidence = _json(evidence_path.resolve(strict=True))
    curve_decision = _json(curve_decision_path.resolve(strict=True))
    decision = evaluate_official_score(contract, curve_decision, evidence)
    if append:
        append_ledger_event(
            ledger_path,
            event_type="OFFICIAL_SCORE_DECISION",
            payload={
                "evidence_sha256": sha256_file(evidence_path),
                "curve_decision_sha256": sha256_file(curve_decision_path),
                "decision": decision,
            },
        )
    return decision


def prepare_upload(
    root: Path,
    approval_path: Path,
    curve_decision_path: Path | None,
    *,
    append: bool,
) -> dict[str, Any]:
    _, ledger_path = _paths(root)
    contract = load_contract(root, CONTRACT_RELATIVE)
    approval = _json(approval_path.resolve(strict=True))
    curve_decision = (
        _json(curve_decision_path.resolve(strict=True)) if curve_decision_path is not None else None
    )
    readiness = validate_upload_approval(
        root,
        contract,
        approval,
        curve_decision=curve_decision,
    )
    if append:
        append_ledger_event(
            ledger_path,
            event_type="UPLOAD_READINESS_VALIDATED",
            payload={
                "approval_receipt_sha256": sha256_file(approval_path),
                "curve_decision_sha256": (
                    sha256_file(curve_decision_path) if curve_decision_path is not None else None
                ),
                "readiness": readiness,
            },
        )
    return readiness


def evaluate_completion(root: Path, evidence_path: Path, *, append: bool) -> dict[str, Any]:
    _, ledger_path = _paths(root)
    contract = load_contract(root, CONTRACT_RELATIVE)
    evidence = _json(evidence_path.resolve(strict=True))
    decision = evaluate_goal_completion(contract, evidence)
    if append:
        append_ledger_event(
            ledger_path,
            event_type="GOAL_COMPLETION_DECISION",
            payload={"evidence_sha256": sha256_file(evidence_path), "decision": decision},
        )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("initialize")
    subparsers.add_parser("status")
    curve = subparsers.add_parser("curve")
    curve.add_argument("--evidence", type=Path, required=True)
    curve.add_argument("--append", action="store_true")
    score = subparsers.add_parser("score")
    score.add_argument("--evidence", type=Path, required=True)
    score.add_argument("--curve-decision", type=Path, required=True)
    score.add_argument("--append", action="store_true")
    upload = subparsers.add_parser("prepare-upload")
    upload.add_argument("--approval", type=Path, required=True)
    upload.add_argument("--curve-decision", type=Path)
    upload.add_argument("--append", action="store_true")
    completion = subparsers.add_parser("completion")
    completion.add_argument("--evidence", type=Path, required=True)
    completion.add_argument("--append", action="store_true")
    args = parser.parse_args()

    if args.mode == "initialize":
        result = initialize(args.root)
    elif args.mode == "status":
        _, ledger_path = _paths(args.root)
        records = validate_ledger(ledger_path)
        result = {
            "status": "VALID" if records else "NOT_INITIALIZED",
            "records": len(records),
            "tail": records[-1] if records else None,
        }
    elif args.mode == "curve":
        result = evaluate_curve(args.root, args.evidence, append=args.append)
    elif args.mode == "score":
        result = evaluate_score(
            args.root,
            args.evidence,
            args.curve_decision,
            append=args.append,
        )
    elif args.mode == "prepare-upload":
        result = prepare_upload(
            args.root,
            args.approval,
            args.curve_decision,
            append=args.append,
        )
    else:
        result = evaluate_completion(args.root, args.evidence, append=args.append)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
