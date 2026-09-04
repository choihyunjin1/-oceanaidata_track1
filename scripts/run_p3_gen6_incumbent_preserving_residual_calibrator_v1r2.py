"""Check or execute the one-shot P3 Gen6r2 research-only curve.

The default mode is read-only.  This module deliberately imports the NumPy-backed
contract and execution engine only after the canonical process environment has
been validated.  The engine is imported only after a verified O_EXCL attempt lock.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final

CONFIG_RELATIVE: Final = (
    "configs/experiments/p3_gen6_incumbent_preserving_residual_calibrator_v1r2.json"
)
CONTRACT_RELATIVE: Final = (
    "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_contract_r2.py"
)
ENGINE_RELATIVE: Final = (
    "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_execution_r2.py"
)
RUNNER_RELATIVE: Final = (
    "scripts/run_p3_gen6_incumbent_preserving_residual_calibrator_v1r2.py"
)
CONTRACT_MODULE: Final = (
    "p3_wave.gen6_incumbent_preserving_residual_calibrator_contract_r2"
)
ENGINE_MODULE: Final = (
    "p3_wave.gen6_incumbent_preserving_residual_calibrator_execution_r2"
)
THREAD_ENVIRONMENT: Final = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}
REPARSE_ATTRIBUTE: Final = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _has_reparse(path: Path) -> bool:
    info = path.lstat()
    return path.is_symlink() or bool(
        int(getattr(info, "st_file_attributes", 0)) & REPARSE_ATTRIBUTE
    )


def _canonical_environment(
    *,
    root: Path | None,
    data_dir: Path | None,
    config_path: Path | None,
) -> tuple[Path, Path, Path]:
    workspace_value = os.environ.get("P3_WORKSPACE_ROOT")
    data_value = os.environ.get("P3_DATA_DIR")
    if not workspace_value or not data_value:
        raise PermissionError("P3_WORKSPACE_ROOT and P3_DATA_DIR are both required")
    for key, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(key) != expected:
            raise PermissionError(f"canonical process environment differs: {key}")

    workspace_env = Path(workspace_value)
    source_env = Path(data_value)
    if not workspace_env.is_absolute() or not source_env.is_absolute():
        raise PermissionError("canonical workspace and data environment must be absolute")
    if _has_reparse(workspace_env) or _has_reparse(source_env):
        raise PermissionError("canonical workspace and data roots cannot be symlinks")
    workspace = workspace_env.resolve(strict=True)
    source = source_env.resolve(strict=True)
    if workspace_env.absolute() != workspace or source_env.absolute() != source:
        raise PermissionError("canonical environment roots cannot use aliases")
    if Path(__file__).resolve(strict=True) != (workspace / RUNNER_RELATIVE).resolve(
        strict=True
    ):
        raise PermissionError("runner resolved outside the canonical workspace")
    for canonical_path in (
        workspace / "src",
        workspace / "src/p3_wave",
        workspace / CONTRACT_RELATIVE,
        workspace / RUNNER_RELATIVE,
        workspace / CONFIG_RELATIVE,
    ):
        if _has_reparse(canonical_path):
            raise PermissionError("canonical code/config path contains a reparse object")

    if root is not None and (
        not root.is_absolute()
        or root.absolute() != workspace
        or root.resolve(strict=True) != workspace
    ):
        raise PermissionError("alternate --root resolution is forbidden")
    if data_dir is not None and (
        not data_dir.is_absolute()
        or data_dir.absolute() != source
        or data_dir.resolve(strict=True) != source
    ):
        raise PermissionError("alternate --data-dir resolution is forbidden")
    canonical_config = workspace / CONFIG_RELATIVE
    if config_path is not None and (
        not config_path.is_absolute()
        or config_path.absolute() != canonical_config
        or config_path.resolve(strict=True) != canonical_config.resolve(strict=True)
    ):
        raise PermissionError("alternate --config resolution is forbidden")
    return workspace, source, canonical_config


def _canonical_module(name: str, expected: Path) -> ModuleType:
    module = importlib.import_module(name)
    module_path = Path(str(module.__file__)).resolve(strict=True)
    if module_path != expected.resolve(strict=True):
        raise PermissionError(f"{name} resolved outside the canonical workspace")
    return module


def _contract(root: Path) -> ModuleType:
    source = root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return _canonical_module(CONTRACT_MODULE, root / CONTRACT_RELATIVE)


def check_only(
    *,
    root: Path | None = None,
    data_dir: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    workspace, source, canonical_config = _canonical_environment(
        root=root, data_dir=data_dir, config_path=config_path
    )
    guard = _contract(workspace)
    preflight = guard.static_preflight(
        workspace,
        source,
        requested_config=canonical_config,
        execution_documents_allowed=False,
    )
    return {
        "schema_version": (
            "p3_gen6_incumbent_preserving_residual_calibrator.check_only_result.r2.v1"
        ),
        "status": preflight["status"],
        "preflight": preflight,
        "independent_qa_receipts_created": 0,
        "authorizations_created": 0,
        "attempt_locks_created": 0,
        "fits": 0,
        "predictions": 0,
        "scores": 0,
        "target_scalar_decodes": 0,
        "test_value_reads": 0,
        "candidate_files_created": 0,
        "registry_appends": 0,
        "uploads": 0,
    }


def run_once(
    *,
    root: Path | None = None,
    data_dir: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    workspace, source, canonical_config = _canonical_environment(
        root=root, data_dir=data_dir, config_path=config_path
    )
    guard = _contract(workspace)
    preflight = guard.static_preflight(
        workspace,
        source,
        requested_config=canonical_config,
        execution_documents_allowed=True,
    )
    config, _raw = guard.load_canonical_config(workspace, canonical_config)
    capability: object | None = None
    lock_path = workspace / config["canonical_paths"]["attempt_lock"]
    try:
        lock, lock_sha, qa_sha, auth_sha, lineage = guard.create_attempt_lock(
            workspace, source, config
        )
        capability = guard.issue_execution_capability(
            root=workspace,
            data_dir=source,
            config=config,
            lock_payload=lock,
            lock_sha256=lock_sha,
            qa_sha256=qa_sha,
            authorization_sha256=auth_sha,
            static_lineage=lineage,
        )

        # Importing this module imports the sealed v1 scientific helper.  Keep it
        # strictly after the verified one-shot lock and capability issuance.
        engine = _canonical_module(ENGINE_MODULE, workspace / ENGINE_RELATIVE)
        state = engine.load_key_input_only(capability)
        engine.predict_and_commit_fold(capability, state, fold_index=0)
        engine.release_committed_fold_truth(capability, state, fold_index=0)
        engine.predict_and_commit_fold(capability, state, fold_index=1)
        engine.release_committed_fold_truth(capability, state, fold_index=1)
        engine.predict_and_commit_fold(capability, state, fold_index=2)
        engine.commit_predictions_complete(capability, state)
        engine.release_committed_fold_truth(capability, state, fold_index=2)
        scored = engine.score_and_write_core(capability, state)
        published = engine.publish_manifest_sidecar_seal(capability, state)
        verified = guard.verify_published_output(
            workspace, source, requested_config=canonical_config
        )
    except BaseException as exception:
        if capability is not None:
            guard.revoke_capability_after_failure(capability)
        if lock_path.exists():
            failure = workspace / config["canonical_paths"]["run_failure_receipt"]
            if not failure.exists():
                guard.write_failure_receipt(
                    root=workspace, config=config, exception=exception
                )
        raise
    return {
        **published,
        "preflight_status": preflight["status"],
        "attempt_lock_sha256": lock_sha,
        "score_status": scored["status"],
        "post_publish_verification": verified,
        "capability_revoked": True,
        "candidate_created": False,
        "test_prediction_created": False,
        "registry_appended": False,
        "uploads": 0,
    }


def verify_published(
    *,
    root: Path | None = None,
    data_dir: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run the registry-free read-only verifier, including from a later process."""

    workspace, source, canonical_config = _canonical_environment(
        root=root, data_dir=data_dir, config_path=config_path
    )
    guard = _contract(workspace)
    return guard.verify_published_output(
        workspace, source, requested_config=canonical_config
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--mode",
        choices=("check-only", "run", "verify-published"),
        default="check-only",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    operations = {
        "check-only": check_only,
        "run": run_once,
        "verify-published": verify_published,
    }
    operation = operations[args.mode]
    result = operation(
        root=args.root,
        data_dir=args.data_dir,
        config_path=args.config,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
