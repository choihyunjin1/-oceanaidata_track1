"""Check or execute the append-only P3 Gen5r3 dense72 correction."""

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

GUARD_MODULE = "p3_wave.hierarchical_residual_basis_dense72_contract_r3"
GUARD_RELATIVE = "src/p3_wave/hierarchical_residual_basis_dense72_contract_r3.py"
ENGINE_MODULE = "p3_wave.hierarchical_residual_basis_dense72_execution_r3"
ENGINE_RELATIVE = "src/p3_wave/hierarchical_residual_basis_dense72_execution_r3.py"
CONFIG_RELATIVE = "configs/experiments/p3_hierarchical_residual_basis_dense72_r3.json"


def _canonical_module(module: ModuleType, path: Path, *, name: str) -> None:
    if Path(str(module.__file__)).resolve(strict=True) != path.resolve(strict=True):
        raise PermissionError(f"{name} resolved outside the canonical workspace")


def check_only(
    *,
    root: Path,
    data_dir: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    workspace = root.resolve(strict=True)
    guard = importlib.import_module(GUARD_MODULE)
    _canonical_module(guard, workspace / GUARD_RELATIVE, name="Gen5r3 guard")
    report = guard.static_preflight(
        workspace,
        data_dir,
        requested_config=config_path or workspace / CONFIG_RELATIVE,
    )
    return {
        "schema_version": "p3_hierarchical_residual_basis.gen5r3_dense72.check_only.r1",
        "status": report["status"],
        "stage": guard.STAGE,
        "preflight": report,
        "independent_qa_created": False,
        "authorization_created": False,
        "attempt_lock_created": False,
        "capability_minted": False,
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
    _canonical_module(guard, workspace / GUARD_RELATIVE, name="Gen5r3 guard")
    config, preflight = guard.prepare_execution_preflight(
        workspace,
        data_dir,
        requested_config=config_path or workspace / CONFIG_RELATIVE,
    )
    lock_created = False
    capability = None
    try:
        lock, lock_payload, lock_sha256, _qa_sha, _authorization_sha = (
            guard.create_and_verify_attempt_lock(workspace, config, preflight)
        )
        lock_created = True
        capability = guard.issue_execution_capability(
            workspace,
            config,
            preflight,
            lock_sha256=lock_sha256,
        )
        engine = importlib.import_module(ENGINE_MODULE)
        _canonical_module(engine, workspace / ENGINE_RELATIVE, name="Gen5r3 engine")
        execute = getattr(engine, "execute_curve_stage", None)
        if not callable(execute):
            raise RuntimeError("canonical Gen5r3 engine lacks execute_curve_stage")
        result = execute(
            capability=capability,
            root=workspace,
            data_dir=data_dir.resolve(strict=True),
            config=config,
            preflight=preflight,
        )
    except BaseException as exc:
        if lock_created:
            guard.write_run_failure_receipt(workspace, config, exception=exc)
        raise
    finally:
        if capability is not None:
            guard.revoke_execution_capability(capability)
    return {
        **result,
        "attempt_lock": lock.relative_to(workspace).as_posix(),
        "attempt_lock_sha256": lock_sha256,
        "attempt_lock_status": lock_payload["status"],
        "capability_minted_after_verified_lock": True,
        "comparison_mode": guard.COMPARISON_MODE,
        "exact_official_incumbent_comparison": False,
        "official_promotion_allowed": False,
        "candidate_generated": False,
        "test_prediction_generated": False,
        "uploads": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
