"""Stage-A runner for the P2 architecture-matched time-safe reference.

``check-only`` is the default and performs only contract/SHA/header/key checks.
Actual execution requires a separate O_EXCL authorization and a separately
installed execution engine; this static implementation never imports it during
preflight.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from p2_restore.architecture_matched_prefix_refit import (
    CONFIG_RELATIVE,
    STAGE_A,
    consume_attempt_lock,
    load_canonical_config,
    sha256_file,
    static_preflight,
    verify_execution_authorization,
    verify_stage_a_seal,
)

EXECUTION_ENGINE = "p2_restore.architecture_matched_stage_a_execution"


def check_only(*, root: Path, data_dir: Path, config_path: Path | None = None) -> dict[str, Any]:
    report = static_preflight(
        root,
        data_dir,
        requested_config=config_path,
    )
    return {
        **report,
        "stage": STAGE_A,
        "execution_engine_imported": False,
        "execution_authorized": False,
        "actual_execution_ready": False,
        "engine_present": importlib.util.find_spec(EXECUTION_ENGINE) is not None,
        "scaffolding_only": True,
        "attempt_lock_created": False,
        "reference_fit_started": False,
        "reference_prediction_started": False,
    }


def run_authorized(
    *, root: Path, data_dir: Path, config_path: Path | None = None
) -> dict[str, Any]:
    """Execute only after authorization; the engine is intentionally late-imported."""

    config = load_canonical_config(root, config_path)
    preflight = static_preflight(root, data_dir, requested_config=config_path)
    if config["execution_policy"].get("static_check_only_now") is not False:
        raise RuntimeError(
            "Stage-A is static-check-only; a new canonical config and a pinned "
            "execution engine are required before authorization or lock creation"
        )
    engine_spec = importlib.util.find_spec(EXECUTION_ENGINE)
    if engine_spec is None or engine_spec.origin is None:
        raise RuntimeError("authorized Stage-A execution engine is unavailable")
    authorization = verify_execution_authorization(root, config, STAGE_A)
    authorization_path = (
        root.resolve(strict=True) / config["canonical_paths"]["stage_a_authorization"]
    ).resolve(strict=True)
    lock = consume_attempt_lock(
        root,
        config,
        STAGE_A,
        authorization_sha256=sha256_file(authorization_path),
    )
    engine = importlib.import_module(EXECUTION_ENGINE)
    execute = getattr(engine, "execute_stage_a", None)
    if not callable(execute):
        raise RuntimeError("authorized Stage-A execution engine has no execute_stage_a")
    result = execute(root=root, data_dir=data_dir, config=config, preflight=preflight)
    seal = verify_stage_a_seal(root, config)
    return {
        "schema_version": "p2_architecture_matched_stage_a_run.v1",
        "stage": STAGE_A,
        "authorization": authorization,
        "attempt_lock": lock.relative_to(root.resolve(strict=True)).as_posix(),
        "engine_result": result,
        "reference_seal": seal,
        "upload_performed": False,
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
