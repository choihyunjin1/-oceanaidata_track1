"""Check or execute the one-shot P3 Gen6 incumbent-preserving research curve."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

HELPER_MODULE = "p3_wave.gen6_incumbent_preserving_residual_calibrator"
HELPER_RELATIVE = "src/p3_wave/gen6_incumbent_preserving_residual_calibrator.py"
CONFIG_RELATIVE = (
    "configs/experiments/p3_gen6_incumbent_preserving_residual_calibrator_v1.json"
)


def _canonical_module(module: ModuleType, path: Path) -> None:
    if Path(str(module.__file__)).resolve(strict=True) != path.resolve(strict=True):
        raise PermissionError("Gen6 helper resolved outside the canonical workspace")


def _helper(root: Path) -> ModuleType:
    module = importlib.import_module(HELPER_MODULE)
    _canonical_module(module, root / HELPER_RELATIVE)
    return module


def check_only(
    *, root: Path, data_dir: Path, config_path: Path | None = None
) -> dict[str, Any]:
    workspace = root.resolve(strict=True)
    helper = _helper(workspace)
    preflight = helper.static_preflight(
        workspace,
        data_dir,
        requested_config=config_path or workspace / CONFIG_RELATIVE,
    )
    return {
        "schema_version": (
            "p3_gen6_incumbent_preserving_residual_calibrator.check_only_result.v1"
        ),
        "status": preflight["status"],
        "preflight": preflight,
        "qa_receipts_created": 0,
        "authorizations_created": 0,
        "attempt_locks_created": 0,
        "fits": 0,
        "predictions": 0,
        "scores": 0,
        "test_value_reads": 0,
        "candidate_files_created": 0,
        "registry_appends": 0,
        "uploads": 0,
    }


def run_once(
    *, root: Path, data_dir: Path, config_path: Path | None = None
) -> dict[str, Any]:
    workspace = root.resolve(strict=True)
    helper = _helper(workspace)
    canonical_config = (workspace / CONFIG_RELATIVE).resolve(strict=True)
    requested = (config_path or canonical_config).resolve(strict=True)
    if requested != canonical_config:
        raise PermissionError("non-canonical Gen6 config path is forbidden")
    preflight = helper.static_preflight(
        workspace,
        data_dir,
        requested_config=requested,
        allow_execution_documents=True,
    )
    config = json.loads(canonical_config.read_bytes())
    _, qa_sha, _, authorization_sha = helper.verify_execution_documents(workspace, config)
    lock_created = False
    try:
        lock, lock_sha = helper.create_attempt_lock(
            workspace,
            config,
            qa_sha256=qa_sha,
            authorization_sha256=authorization_sha,
        )
        lock_created = True
        capability = helper.issue_execution_capability(
            workspace,
            config,
            attempt_lock_sha256=lock_sha,
        )
        evaluated, metrics, evidence, receipts = helper.run_research_curve(
            capability,
            root=workspace,
            config=config,
        )
        published = helper.publish_research_output(
            capability,
            root=workspace,
            config=config,
            evaluated=evaluated,
            metrics=metrics,
            evidence=evidence,
        )
    except BaseException as exc:
        if lock_created:
            helper.write_failure_receipt(workspace, config, exception=exc)
        raise
    return {
        **published,
        "preflight_status": preflight["status"],
        "attempt_lock": lock.relative_to(workspace).as_posix(),
        "attempt_lock_sha256": lock_sha,
        "outer_application_cells": len(receipts),
        "official_promotion_allowed": False,
        "candidate_created": False,
        "test_prediction_created": False,
        "registry_appended": False,
        "uploads": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("check-only", "run"), default="check-only")
    return parser


def main() -> None:
    args = _parser().parse_args()
    data_dir = args.data_dir
    if data_dir is None:
        value = os.environ.get("P3_DATA_DIR")
        if not value:
            raise SystemExit("--data-dir or P3_DATA_DIR is required")
        data_dir = Path(value)
    operation = check_only if args.mode == "check-only" else run_once
    result = operation(root=args.root, data_dir=data_dir, config_path=args.config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
