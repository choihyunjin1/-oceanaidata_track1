"""Fail-closed CLI for the P2 joint-hydrographic Layer-4 r2 curve."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

CONFIG_RELATIVE = (
    "configs/experiments/p2_joint_hydrographic_multitask_layer4_execution_r2.json"
)
GUARD_MODULE = "p2_restore.joint_hydrographic_multitask_layer4_contract_r2"
ENGINE_MODULE = "p2_restore.joint_hydrographic_multitask_layer4_execution_r2"
GUARD_RELATIVE = "src/p2_restore/joint_hydrographic_multitask_layer4_contract_r2.py"
ENGINE_RELATIVE = "src/p2_restore/joint_hydrographic_multitask_layer4_execution_r2.py"
NUMERICAL_PREFIXES = ("numpy", "pandas", "scipy", "sklearn", "torch")


def _loaded_numerical_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in NUMERICAL_PREFIXES)
    )


def _isolated_preflight(
    *,
    root: Path,
    data_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    script = r"""
import importlib.util, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(strict=True)
data = Path(sys.argv[2]).resolve(strict=True)
config = Path(sys.argv[3]).resolve(strict=True)
guard_path = (root / sys.argv[4]).resolve(strict=True)
name = "p2_restore.joint_hydrographic_multitask_layer4_contract_r2"
spec = importlib.util.spec_from_file_location(name, guard_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load canonical Layer-4 guard")
guard = importlib.util.module_from_spec(spec)
sys.modules[name] = guard
spec.loader.exec_module(guard)
print(json.dumps(guard.static_preflight(root, data, requested_config=config), sort_keys=True))
"""
    workspace = root.resolve(strict=True)
    environment = os.environ.copy()
    source = str((workspace / "src").resolve(strict=True))
    prior = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not prior else os.pathsep.join((source, prior))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(workspace),
            str(data_dir.resolve(strict=True)),
            str(config_path.resolve(strict=True)),
            GUARD_RELATIVE,
        ],
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"isolated Layer-4 preflight failed: {detail}")
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("isolated Layer-4 preflight returned invalid JSON") from exc


def check_only(
    *,
    root: Path,
    data_dir: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Static-only verification in a disposable process; no control write."""

    workspace = root.resolve(strict=True)
    requested = (config_path or workspace / CONFIG_RELATIVE).resolve(strict=True)
    numerical_before = set(_loaded_numerical_modules())
    guard_before = GUARD_MODULE in sys.modules
    engine_before = ENGINE_MODULE in sys.modules
    report = _isolated_preflight(root=workspace, data_dir=data_dir, config_path=requested)
    numerical_after = set(_loaded_numerical_modules())
    return {
        **report,
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
        "model_fits": 0,
        "predictions": 0,
        "scores": 0,
        "candidate_predictions": 0,
        "test_predictions": 0,
        "uploads": 0,
    }


def _canonical_module(module: Any, expected: Path, *, label: str) -> None:
    path = Path(module.__file__).resolve(strict=True)
    if path != expected.resolve(strict=True):
        raise PermissionError(f"imported {label} is not the canonical implementation")


def _load_guard_without_package_import(workspace: Path) -> Any:
    """Load the guard bytes without executing p2_restore.__init__ pre-lock."""

    path = (workspace / GUARD_RELATIVE).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(GUARD_MODULE, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load canonical Layer-4 guard")
    module = importlib.util.module_from_spec(spec)
    sys.modules[GUARD_MODULE] = module
    spec.loader.exec_module(module)
    return module


def _progress(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)


def run_authorized(
    *,
    root: Path,
    data_dir: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Verify controls, consume the lock, mint, then late-import the engine."""

    if (
        GUARD_MODULE in sys.modules
        or ENGINE_MODULE in sys.modules
        or _loaded_numerical_modules()
    ):
        raise PermissionError(
            "authorized Layer-4 run requires a clean pre-lock process"
        )
    workspace = root.resolve(strict=True)
    resolved_data = data_dir.resolve(strict=True)
    guard = _load_guard_without_package_import(workspace)
    _canonical_module(guard, workspace / GUARD_RELATIVE, label="Layer-4 guard")
    requested = (config_path or workspace / CONFIG_RELATIVE).resolve(strict=True)
    config = guard.load_canonical_config(workspace, requested)

    first = guard.static_preflight(workspace, resolved_data, requested_config=requested)
    _qa_first, qa_sha_first = guard.verify_pre_execution_qa(workspace, config)
    _auth_first, auth_sha_first = guard.verify_execution_authorization(workspace, config)
    second = guard.static_preflight(workspace, resolved_data, requested_config=requested)
    _qa_second, qa_sha_second = guard.verify_pre_execution_qa(workspace, config)
    _auth_second, auth_sha_second = guard.verify_execution_authorization(workspace, config)
    if (
        first["summary_sha256"] != second["summary_sha256"]
        or qa_sha_first != qa_sha_second
        or auth_sha_first != auth_sha_second
    ):
        raise PermissionError("preflight, QA, or authorization changed before lock")

    # This is the first and only persistent mutation before model execution.
    guard.consume_attempt_lock(workspace, resolved_data, config)
    capability, locked_preflight = guard.issue_execution_capability(
        workspace,
        resolved_data,
        config,
    )
    if locked_preflight["summary_sha256"] != first["summary_sha256"]:
        raise PermissionError("post-lock operational snapshot changed")

    # Engine import occurs strictly after consumed-lock verification and mint.
    engine = importlib.import_module(ENGINE_MODULE)
    _canonical_module(engine, workspace / ENGINE_RELATIVE, label="Layer-4 engine")
    execute = getattr(engine, "execute_layer4_curve", None)
    if not callable(execute):
        raise RuntimeError("canonical Layer-4 engine lacks execute_layer4_curve")
    result = execute(
        capability=capability,
        root=workspace,
        data_dir=resolved_data,
        config=config,
        preflight=locked_preflight,
        progress=_progress,
    )
    sealed = guard.verify_seal(workspace, config)
    return {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.run.r2",
        "status": result["status"],
        "stage": guard.STAGE,
        "engine_result": result,
        "verified_seal": sealed,
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
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(CONFIG_RELATIVE),
        help="copies and overrides are rejected; only the canonical config resolves",
    )
    parser.add_argument("--mode", choices=("check-only", "run"), default="check-only")
    return parser


def main() -> int:
    args = _parser().parse_args()
    requested = args.config if args.config.is_absolute() else args.root / args.config
    if args.mode == "check-only":
        result = check_only(root=args.root, data_dir=args.data_dir, config_path=requested)
    else:
        result = run_authorized(root=args.root, data_dir=args.data_dir, config_path=requested)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
