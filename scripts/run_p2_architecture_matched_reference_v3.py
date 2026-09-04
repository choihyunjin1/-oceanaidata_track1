"""Fail-closed runner for executable P2 architecture-matched Stage A v3."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

CONFIG_RELATIVE = "configs/experiments/p2_architecture_matched_stage_a_execution_v3.json"
GUARD_MODULE = "p2_restore.architecture_matched_stage_a_contract_v3"
ENGINE_MODULE = "p2_restore.architecture_matched_stage_a_execution_v3"
GUARD_RELATIVE = "src/p2_restore/architecture_matched_stage_a_contract_v3.py"
ENGINE_RELATIVE = "src/p2_restore/architecture_matched_stage_a_execution_v3.py"
NUMERICAL_MODULE_PREFIXES = (
    "lightgbm",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
)


def _loaded_numerical_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in NUMERICAL_MODULE_PREFIXES
        )
    )


def _isolated_preflight(
    *,
    root: Path,
    data_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    workspace = root.resolve(strict=True)
    script = """
import json
import sys
from pathlib import Path
from p2_restore.architecture_matched_stage_a_contract_v3 import static_preflight
root = Path(sys.argv[1]).resolve(strict=True)
data_dir = Path(sys.argv[2]).resolve(strict=True)
config = Path(sys.argv[3]).resolve(strict=True)
print(json.dumps(static_preflight(root, data_dir, requested_config=config), sort_keys=True))
"""
    environment = os.environ.copy()
    source = str((workspace / "src").resolve(strict=True))
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not existing else os.pathsep.join((source, existing))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(workspace),
            str(data_dir.resolve(strict=True)),
            str(config_path.resolve(strict=True)),
        ],
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"isolated Stage-A v3 preflight failed: {detail}")
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("isolated Stage-A v3 preflight returned invalid JSON") from exc


def check_only(
    *,
    root: Path,
    data_dir: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run all checks in a child, keeping this process free of model imports."""

    workspace = root.resolve(strict=True)
    requested = (config_path or workspace / CONFIG_RELATIVE).resolve(strict=True)
    numerical_before = set(_loaded_numerical_modules())
    engine_before = ENGINE_MODULE in sys.modules
    guard_before = GUARD_MODULE in sys.modules
    report = _isolated_preflight(
        root=workspace,
        data_dir=data_dir,
        config_path=requested,
    )
    numerical_after = set(_loaded_numerical_modules())
    return {
        **report,
        "stage": "P2_ARCHITECTURE_MATCHED_REFERENCE_STAGE_A_V3",
        "execution_engine_present": (workspace / ENGINE_RELATIVE).is_file(),
        "check_only_parent_process": {
            "isolated_preflight": True,
            "numerical_modules_before": sorted(numerical_before),
            "numerical_modules_after": sorted(numerical_after),
            "new_numerical_modules": sorted(numerical_after - numerical_before),
            "guard_imported_before": guard_before,
            "guard_imported_after": GUARD_MODULE in sys.modules,
            "engine_imported_before": engine_before,
            "engine_imported_after": ENGINE_MODULE in sys.modules,
        },
        "execution_engine_imported_by_parent": ENGINE_MODULE in sys.modules,
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


def _canonical_module(module: Any, expected: Path, *, name: str) -> None:
    module_file = Path(module.__file__).resolve(strict=True)
    if module_file != expected.resolve(strict=True):
        raise RuntimeError(f"imported {name} is not the canonical implementation")


def run_authorized(
    *,
    root: Path,
    data_dir: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Verify QA/auth, consume the one-shot lock, then late-import the engine."""

    workspace = root.resolve(strict=True)
    guard = importlib.import_module(GUARD_MODULE)
    _canonical_module(guard, workspace / GUARD_RELATIVE, name="Stage-A v3 guard")
    requested = config_path or workspace / CONFIG_RELATIVE
    config = guard.load_canonical_config(workspace, requested)
    preflight = guard.static_preflight(workspace, data_dir, requested_config=requested)
    _qa, qa_sha256 = guard.verify_pre_execution_qa(workspace, config)
    authorization, authorization_sha256 = guard.verify_execution_authorization(
        workspace,
        config,
        qa_sha256=qa_sha256,
    )
    pins_before_lock = guard.implementation_pins(workspace)
    if pins_before_lock != preflight["implementation_pins"]:
        raise PermissionError("v3 implementation bytes changed between preflight and lock")
    lock = guard.consume_attempt_lock(
        workspace,
        config,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
    )
    engine = importlib.import_module(ENGINE_MODULE)
    _canonical_module(engine, workspace / ENGINE_RELATIVE, name="Stage-A v3 engine")
    execute = getattr(engine, "execute_stage_a", None)
    if not callable(execute):
        raise RuntimeError("canonical Stage-A v3 engine has no execute_stage_a callable")
    result = execute(
        root=workspace,
        data_dir=data_dir.resolve(strict=True),
        config=config,
        preflight=preflight,
        attempt_lock=lock,
        progress=_progress,
    )
    seal = guard.verify_stage_a_seal(workspace, config)
    return {
        "schema_version": "p2_architecture_matched_stage_a_run.v3",
        "status": "COMPLETE_SEALED_ARCHITECTURE_MATCHED_REFERENCE",
        "stage": guard.STAGE,
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
