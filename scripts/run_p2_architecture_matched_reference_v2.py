"""Fail-closed runner for executable P2 architecture-matched Stage A v2."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from p2_restore.architecture_matched_stage_a_contract_v2 import (
    CONFIG_RELATIVE,
    ENGINE_RELATIVE,
    STAGE,
    consume_attempt_lock,
    implementation_pins,
    load_canonical_config,
    static_preflight,
    verify_execution_authorization,
    verify_pre_execution_qa,
    verify_stage_a_seal,
)

ENGINE_MODULE = "p2_restore.architecture_matched_stage_a_execution_v2"


def check_only(
    *, root: Path, data_dir: Path, config_path: Path | None = None
) -> dict[str, Any]:
    report = static_preflight(root, data_dir, requested_config=config_path)
    return {
        **report,
        "stage": STAGE,
        "execution_engine_present": importlib.util.find_spec(ENGINE_MODULE) is not None,
        "execution_engine_imported": False,
        "independent_qa_verified": False,
        "execution_authorized": False,
        "attempt_lock_created": False,
        "reference_fit_started": False,
        "reference_prediction_started": False,
        "challenger_import_fit_or_score_count": 0,
        "submission_prediction_count": 0,
        "uploads": 0,
    }


def _progress(event: dict[str, Any]) -> None:
    print(json.dumps(event, sort_keys=True, ensure_ascii=False), file=sys.stderr, flush=True)


def run_authorized(
    *, root: Path, data_dir: Path, config_path: Path | None = None
) -> dict[str, Any]:
    """Run only after QA/auth; the numerical engine is deliberately late-imported."""

    workspace = root.resolve(strict=True)
    config = load_canonical_config(workspace, config_path)
    preflight = static_preflight(workspace, data_dir, requested_config=config_path)
    _qa, qa_sha256 = verify_pre_execution_qa(workspace, config)
    authorization, authorization_sha256 = verify_execution_authorization(
        workspace,
        config,
        qa_sha256=qa_sha256,
    )
    pins_before_lock = implementation_pins(workspace)
    if pins_before_lock != preflight["implementation_pins"]:
        raise PermissionError("implementation bytes changed between preflight and lock")
    lock = consume_attempt_lock(
        workspace,
        config,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
    )
    engine = importlib.import_module(ENGINE_MODULE)
    engine_file = Path(engine.__file__).resolve(strict=True)
    expected_engine = (workspace / ENGINE_RELATIVE).resolve(strict=True)
    if engine_file != expected_engine:
        raise RuntimeError("imported Stage-A engine is not the canonical implementation")
    execute = getattr(engine, "execute_stage_a", None)
    if not callable(execute):
        raise RuntimeError("canonical Stage-A engine has no execute_stage_a callable")
    result = execute(
        root=workspace,
        data_dir=data_dir.resolve(strict=True),
        config=config,
        preflight=preflight,
        attempt_lock=lock,
        progress=_progress,
    )
    seal = verify_stage_a_seal(workspace, config)
    return {
        "schema_version": "p2_architecture_matched_stage_a_run.v2",
        "status": "COMPLETE_SEALED_ARCHITECTURE_MATCHED_REFERENCE",
        "stage": STAGE,
        "authorization": authorization,
        "attempt_lock": lock.relative_to(workspace).as_posix(),
        "engine_result": result,
        "reference_seal": seal,
        "exact_official_incumbent_comparison": False,
        "official_promotion_allowed": False,
        "challenger_import_fit_or_score_count": 0,
        "submission_prediction_count": 0,
        "uploads": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(CONFIG_RELATIVE),
        help="copies/overrides are rejected; this must resolve to the canonical file",
    )
    parser.add_argument("--mode", choices=("check-only", "run"), default="check-only")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = args.root / config_path
    if args.mode == "check-only":
        result = check_only(root=args.root, data_dir=args.data_dir, config_path=config_path)
    else:
        result = run_authorized(root=args.root, data_dir=args.data_dir, config_path=config_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
