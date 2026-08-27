"""CLI for the append-only meaningful-score v3 contract.

The default/check paths are read-only.  Ledger writes require an explicit
``--append`` flag; uploads are never performed by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ocean_goal.meaningful_score_v3 import (
    CONTRACT_RELATIVE,
    CONTRACT_SHA256,
    PRE_INIT_QA_RELATIVE,
    ContractError,
    append_goal_ledger_event,
    evaluate_goal_completion,
    evaluate_learning_curve,
    evaluate_official_score,
    initialize_goal_ledger,
    load_contract,
    sha256_file,
    validate_goal_ledger,
    validate_upload_approval,
    verify_curve_evidence_pins,
    verify_initial_pins,
    verify_official_evidence_pins,
)

LEDGER_RELATIVE = "artifacts/meaningful_score_goal_v3/registry.jsonl"
EVALUATOR_RELATIVE = "src/ocean_goal/meaningful_score_v3.py"
EVALUATOR_SHA256 = "720839224209cf487e500650ecf34c1e8cd3e5fb26f2395c60e76d2050ed973a"
IMPLEMENTATION_RELATIVES = {
    "V3_CONTRACT": CONTRACT_RELATIVE,
    "V3_EVALUATOR": EVALUATOR_RELATIVE,
    "V3_CLI": "scripts/run_meaningful_score_goal_v3.py",
    "P2_ARCHITECTURE_CONFIG": (
        "configs/experiments/p2_architecture_matched_time_safe_baseline_v1.json"
    ),
    "P2_ARCHITECTURE_GUARDS": ("src/p2_restore/architecture_matched_prefix_refit.py"),
    "P2_STAGE_A_RUNNER": "scripts/run_p2_architecture_matched_reference_v1.py",
    "P2_STAGE_B_RUNNER": ("scripts/run_p2_meaningful_learning_curve_generation_v2.py"),
    "V3_TESTS": "tests/test_meaningful_score_v3.py",
    "P2_ARCHITECTURE_TESTS": ("tests/test_p2_architecture_matched_prefix_refit.py"),
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _workspace(root: Path) -> Path:
    return root.resolve(strict=True)


def _ledger(root: Path) -> Path:
    workspace = _workspace(root)
    ledger = (workspace / LEDGER_RELATIVE).resolve(strict=False)
    if not ledger.is_relative_to(workspace):
        raise ValueError("canonical ledger escapes workspace")
    return ledger


def _implementation_pins(root: Path) -> dict[str, dict[str, Any]]:
    workspace = _workspace(root)
    result: dict[str, dict[str, Any]] = {}
    for role, relative in IMPLEMENTATION_RELATIVES.items():
        path = (workspace / relative).resolve(strict=True)
        if not path.is_relative_to(workspace):
            raise ContractError(f"implementation path escapes workspace: {role}")
        result[role] = {
            "path": Path(relative).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return result


def _predecessor_anchor(contract: dict[str, Any]) -> dict[str, Any]:
    predecessor = contract["supersedes"]
    return {
        "path": predecessor["ledger"],
        "sha256": predecessor["ledger_sha256"],
        "bytes": predecessor["ledger_bytes"],
        "event_count": predecessor["ledger_event_count"],
        "head_event_sha256": predecessor["ledger_head_event_sha256"],
    }


def _qa_pin(root: Path, receipt_path: Path) -> dict[str, Any]:
    workspace = _workspace(root)
    receipt = receipt_path.resolve(strict=True)
    if not receipt.is_relative_to(workspace):
        raise ContractError("pre-init QA receipt must be workspace-relative")
    if receipt != (workspace / PRE_INIT_QA_RELATIVE).resolve(strict=True):
        raise ContractError("pre-init QA receipt path must be canonical")
    return {
        "path": receipt.relative_to(workspace).as_posix(),
        "sha256": sha256_file(receipt),
        "bytes": receipt.stat().st_size,
    }


def _require_initialized(root: Path) -> list[dict[str, Any]]:
    records = validate_goal_ledger(root, _ledger(root))
    if not records:
        raise ContractError("v3 ledger must be initialized before append")
    return records


def check(root: Path) -> dict[str, Any]:
    contract = load_contract(root)
    pins = verify_initial_pins(root, contract)
    contract_path = (_workspace(root) / CONTRACT_RELATIVE).resolve(strict=True)
    evaluator_path = (_workspace(root) / EVALUATOR_RELATIVE).resolve(strict=True)
    evaluator_sha = sha256_file(evaluator_path)
    if evaluator_sha != EVALUATOR_SHA256:
        raise RuntimeError("canonical v3 evaluator SHA mismatch")
    ledger_records = validate_goal_ledger(root, _ledger(root))
    return {
        "schema_version": "meaningful_score_goal_v3.check.v1",
        "status": "PASS_READ_ONLY",
        "contract_path": CONTRACT_RELATIVE,
        "contract_sha256": sha256_file(contract_path),
        "compiled_contract_sha256": CONTRACT_SHA256,
        "evaluator_path": EVALUATOR_RELATIVE,
        "evaluator_sha256": evaluator_sha,
        "ledger_path": LEDGER_RELATIVE,
        "ledger_exists": _ledger(root).exists(),
        "ledger_event_count": len(ledger_records),
        "ledger_head_event_sha256": (
            ledger_records[-1]["event_sha256"] if ledger_records else None
        ),
        "predecessor_ledger_anchor": _predecessor_anchor(contract),
        "implementation_pins": _implementation_pins(root),
        "comparison_modes": contract["comparison_modes"],
        "pins": pins,
        "writes": 0,
        "uploads": 0,
    }


def initialize(root: Path, *, append: bool, qa_receipt: Path | None = None) -> dict[str, Any]:
    report = check(root)
    if report["ledger_exists"]:
        if append:
            raise ContractError("v3 ledger is already initialized; second init forbidden")
        return {**report, "status": "ALREADY_INITIALIZED_READ_ONLY"}
    if not append:
        return {**report, "status": "INITIALIZATION_CHECK_ONLY_NOT_APPENDED"}
    if qa_receipt is None:
        raise ContractError("independent pre-init QA receipt is required")
    contract = load_contract(root)
    record = initialize_goal_ledger(
        root,
        _ledger(root),
        payload={
            "goal_id": contract["goal_id"],
            "status": contract["initial_state"]["status"],
            "contract_path": CONTRACT_RELATIVE,
            "contract_sha256": CONTRACT_SHA256,
            "verified_candidate_pins": report["pins"],
            "implementation_pins": report["implementation_pins"],
            "predecessor_ledger_anchor": report["predecessor_ledger_anchor"],
            "independent_pre_init_qa": _qa_pin(root, qa_receipt),
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
            "score_promotions": {"P1": False, "P2": False, "P3": False},
            "meaningful_promotions": {"P1": False, "P2": False, "P3": False},
            "execution_counts": {
                "stage_a_fit": 0,
                "stage_a_prediction": 0,
                "stage_b_fit": 0,
                "stage_b_prediction": 0,
                "upload": 0,
            },
            "daily_upload_limit_scope": "TEAM_WIDE",
            "upload_performed": False,
        },
    )
    return {"status": "INITIALIZED", "record": record, "upload_performed": False}


def evaluate_curve(root: Path, evidence_path: Path, *, append: bool) -> dict[str, Any]:
    if append:
        _require_initialized(root)
    contract = load_contract(root)
    evidence = _json(evidence_path.resolve(strict=True))
    pins = verify_curve_evidence_pins(root, evidence)
    decision = evaluate_learning_curve(contract, evidence)
    if append:
        append_goal_ledger_event(
            root,
            _ledger(root),
            event_type="CURVE_RESULT",
            payload={
                "evidence_sha256": sha256_file(evidence_path),
                "evidence_pins": pins,
                "decision": decision,
                "upload_performed": False,
            },
        )
    return {"decision": decision, "evidence_pins": pins, "ledger_appended": append}


def evaluate_score(
    root: Path,
    evidence_path: Path,
    curve_decision_path: Path,
    *,
    append: bool,
) -> dict[str, Any]:
    if append:
        _require_initialized(root)
    contract = load_contract(root)
    evidence = _json(evidence_path.resolve(strict=True))
    curve = _json(curve_decision_path.resolve(strict=True))
    evidence_pins = verify_official_evidence_pins(root, contract, evidence)
    decision = evaluate_official_score(contract, curve, evidence)
    if append:
        append_goal_ledger_event(
            root,
            _ledger(root),
            event_type="OFFICIAL_SCORE_RESULT",
            payload={
                "evidence_sha256": sha256_file(evidence_path),
                "curve_decision_sha256": sha256_file(curve_decision_path),
                "evidence_pins": evidence_pins,
                "decision": decision,
                "upload_performed": False,
            },
        )
    return {
        "decision": decision,
        "evidence_pins": evidence_pins,
        "ledger_appended": append,
        "upload_performed": False,
    }


def prepare_upload(
    root: Path,
    receipt_path: Path,
    curve_decision_path: Path | None,
    *,
    append: bool,
) -> dict[str, Any]:
    if append:
        _require_initialized(root)
    contract = load_contract(root)
    receipt = _json(receipt_path.resolve(strict=True))
    curve = (
        _json(curve_decision_path.resolve(strict=True)) if curve_decision_path is not None else None
    )
    readiness = validate_upload_approval(root, contract, receipt, curve_decision=curve)
    if append:
        append_goal_ledger_event(
            root,
            _ledger(root),
            event_type="UPLOAD_READINESS",
            payload={
                "receipt_sha256": sha256_file(receipt_path),
                "curve_decision_sha256": (
                    sha256_file(curve_decision_path) if curve_decision_path is not None else None
                ),
                "readiness": readiness,
                "upload_performed": False,
            },
        )
    return {"readiness": readiness, "ledger_appended": append, "upload_performed": False}


def evaluate_completion(root: Path, evidence_path: Path, *, append: bool) -> dict[str, Any]:
    if append:
        _require_initialized(root)
    decision = evaluate_goal_completion(
        load_contract(root), _json(evidence_path.resolve(strict=True))
    )
    if append:
        append_goal_ledger_event(
            root,
            _ledger(root),
            event_type="GOAL_COMPLETION_RESULT",
            payload={
                "evidence_sha256": sha256_file(evidence_path),
                "decision": decision,
                "upload_performed": False,
            },
        )
    return {"decision": decision, "ledger_appended": append, "upload_performed": False}


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
        result = initialize(
            args.root,
            append=args.append,
            qa_receipt=args.qa_receipt,
        )
    elif args.mode == "curve":
        result = evaluate_curve(args.root, args.evidence, append=args.append)
    elif args.mode == "score":
        result = evaluate_score(args.root, args.evidence, args.curve_decision, append=args.append)
    elif args.mode == "upload-readiness":
        result = prepare_upload(args.root, args.receipt, args.curve_decision, append=args.append)
    else:
        result = evaluate_completion(args.root, args.evidence, append=args.append)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
