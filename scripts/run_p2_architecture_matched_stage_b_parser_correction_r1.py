"""Check or execute the one-shot P2 Stage-B parser correction r1."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

GUARD_MODULE = "p2_restore.architecture_matched_stage_b_contract_r1"
GUARD_RELATIVE = "src/p2_restore/architecture_matched_stage_b_contract_r1.py"
ENGINE_MODULE = "p2_restore.architecture_matched_stage_b_execution_r1"
ENGINE_RELATIVE = "src/p2_restore/architecture_matched_stage_b_execution_r1.py"
CONFIG_RELATIVE = (
    "configs/experiments/p2_architecture_matched_stage_b_parser_correction_r1.json"
)


def _canonical_module(module: ModuleType, path: Path, *, name: str) -> None:
    observed = Path(str(module.__file__)).resolve(strict=True)
    expected = path.resolve(strict=True)
    if observed != expected:
        raise PermissionError(f"{name} resolved outside the canonical workspace")


def _progress(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def check_only(
    *,
    root: Path,
    data_dir: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    workspace = root.resolve(strict=True)
    guard = importlib.import_module(GUARD_MODULE)
    _canonical_module(guard, workspace / GUARD_RELATIVE, name="parser-correction guard")
    requested = config_path or workspace / CONFIG_RELATIVE
    preflight = guard.static_preflight(workspace, data_dir, requested_config=requested)
    paths = guard.stage_paths(workspace, guard.load_canonical_config(workspace, requested))
    return {
        "schema_version": "p2_architecture_matched_stage_b.parser_correction.check_only.r1",
        "status": preflight["status"],
        "stage": guard.STAGE,
        "preflight": preflight,
        "control_state": {key: path.exists() for key, path in paths.items()},
        "independent_qa_created": False,
        "authorization_created": False,
        "attempt_lock_created": False,
        "output_created": False,
        "fits": 0,
        "predictions": 0,
        "scores": 0,
        "test_predictions": 0,
        "uploads": 0,
    }


def run_once(
    *,
    root: Path,
    data_dir: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    workspace = root.resolve(strict=True)
    guard = importlib.import_module(GUARD_MODULE)
    _canonical_module(guard, workspace / GUARD_RELATIVE, name="parser-correction guard")
    requested = config_path or workspace / CONFIG_RELATIVE
    config = guard.load_canonical_config(workspace, requested)

    preflight = guard.static_preflight(
        workspace,
        data_dir,
        requested_config=requested,
        supplied_config=config,
    )
    if (
        preflight.get("status")
        != "PASS_STATIC_IMPLEMENTATION_ONLY_STAGE_A_SEALED_PARSER_CORRECTED"
    ):
        raise PermissionError("fresh parser-correction preflight did not pass")
    _qa, qa_sha256 = guard.verify_pre_execution_qa(
        workspace,
        config,
        parser_preflight_sha256=preflight["parser_preflight_sha256"],
    )
    authorization, authorization_sha256 = guard.verify_execution_authorization(
        workspace,
        config,
        qa_sha256=qa_sha256,
    )
    pins_before_lock = guard.implementation_pins(workspace)
    if pins_before_lock != preflight["implementation_pins"]:
        raise PermissionError("parser-correction implementation changed before lock")
    lock = guard.consume_attempt_lock(
        workspace,
        config,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
    )

    try:
        engine = importlib.import_module(ENGINE_MODULE)
        _canonical_module(engine, workspace / ENGINE_RELATIVE, name="parser-correction engine")
        execute = getattr(engine, "execute_stage_b", None)
        if not callable(execute):
            raise RuntimeError("canonical parser-correction engine lacks execute_stage_b")
        result = execute(
            root=workspace,
            data_dir=data_dir.resolve(strict=True),
            config=config,
            preflight=preflight,
            attempt_lock=lock,
            progress=_progress,
        )
        seal = guard.verify_stage_b_seal(workspace, config)
    except BaseException as exc:
        guard.write_run_failure_receipt(workspace, config, exception=exc)
        raise

    return {
        "schema_version": "p2_architecture_matched_stage_b.parser_correction.run.r1",
        "status": result["status"],
        "stage": guard.STAGE,
        "authorization": authorization,
        "attempt_lock": lock.relative_to(workspace).as_posix(),
        "engine_result": result,
        "stage_b_seal": seal,
        "comparison_mode": guard.MODE,
        "exact_official_incumbent_comparison": False,
        "official_promotion_allowed": False,
        "candidate_generated": False,
        "test_prediction_generated": False,
        "uploads": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("check-only", "run"), default="check-only")
    return parser


def main() -> None:
    args = _parser().parse_args()
    operation = check_only if args.mode == "check-only" else run_once
    result = operation(root=args.root, data_dir=args.data_dir, config_path=args.config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
