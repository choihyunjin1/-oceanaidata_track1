"""Stage-B runner for the preregistered P2 structural challenger.

The Stage-A reference seal is verified before an attempt lock is consumed and
before the challenger execution module is imported.  ``check-only`` never
imports training/prediction code and never writes.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from p2_restore.architecture_matched_prefix_refit import (
    CONFIG_RELATIVE,
    STAGE_B,
    consume_attempt_lock,
    load_canonical_config,
    sha256_file,
    static_preflight,
    verify_execution_authorization,
    verify_stage_a_seal,
)

EXECUTION_ENGINE = "p2_restore.architecture_matched_stage_b_execution"


def check_only(
    *, root: Path, data_dir: Path, config_path: Path | None = None
) -> dict[str, Any]:
    config = load_canonical_config(root, config_path)
    preflight = static_preflight(root, data_dir, requested_config=config_path)
    try:
        reference = verify_stage_a_seal(root, config)
        status = "PASS_STATIC_CHECK_ONLY_REFERENCE_SEALED"
    except FileNotFoundError:
        reference = None
        status = "BLOCKED_REFERENCE_SEAL_MISSING"
    return {
        **preflight,
        "status": status,
        "stage": STAGE_B,
        "reference_seal": reference,
        "reference_seal_verified_before_challenger_import": reference is not None,
        "challenger_engine_imported": False,
        "execution_authorized": False,
        "attempt_lock_created": False,
        "challenger_fit_started": False,
        "challenger_prediction_started": False,
        "challenger_score_started": False,
    }


def run_authorized(
    *, root: Path, data_dir: Path, config_path: Path | None = None
) -> dict[str, Any]:
    """Verify Stage A first, then authorize, lock, and only then import Stage B."""

    config = load_canonical_config(root, config_path)
    reference = verify_stage_a_seal(root, config)
    preflight = static_preflight(root, data_dir, requested_config=config_path)
    authorization = verify_execution_authorization(root, config, STAGE_B)
    authorization_path = (
        root.resolve(strict=True) / config["canonical_paths"]["stage_b_authorization"]
    ).resolve(strict=True)
    lock = consume_attempt_lock(
        root,
        config,
        STAGE_B,
        authorization_sha256=sha256_file(authorization_path),
    )
    engine = importlib.import_module(EXECUTION_ENGINE)
    execute = getattr(engine, "execute_stage_b", None)
    if not callable(execute):
        raise RuntimeError("authorized Stage-B execution engine has no execute_stage_b")
    result = execute(
        root=root,
        data_dir=data_dir,
        config=config,
        preflight=preflight,
        reference_seal=reference,
    )
    return {
        "schema_version": "p2_meaningful_learning_curve_generation_v2.run.v1",
        "stage": STAGE_B,
        "comparison_mode": config["comparison_mode"],
        "exact_official_incumbent_comparison": False,
        "reference_seal": reference,
        "authorization": authorization,
        "attempt_lock": lock.relative_to(root.resolve(strict=True)).as_posix(),
        "engine_result": result,
        "local_pass_can_promote": False,
        "official_promotion_requires_actual_immutable_csv_paired_ab": True,
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
        result = run_authorized(
            root=args.root, data_dir=args.data_dir, config_path=config_path
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
