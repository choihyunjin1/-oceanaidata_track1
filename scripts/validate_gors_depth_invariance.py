"""Validate the exact G-ORS depth-invariance contract without running CV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from p1_qc.experiment import sha256_file  # noqa: E402
from p1_qc.gors_depth_invariance_preregistration import (  # noqa: E402
    GORSDepthPreregistrationError,
    validate_gors_depth_preregistration_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("configs/experiments/p1_gors_depth_invariance_draft.json"),
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("reports/EXPERIMENT_LEDGER.jsonl"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/p1.toml"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-run",
        type=Path,
        default=Path("artifacts/runs/20260813T153038+0900_cv_378a4e89"),
    )
    parser.add_argument("--require-outer-authorized", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve(strict=True)


def main() -> int:
    args = parse_args()
    try:
        data_dir = _resolve(args.data_dir)
        reference = _resolve(args.reference_run)
        paths = {
            "config": _resolve(args.config),
            "train": _resolve(data_dir / "train.csv"),
            "test": _resolve(data_dir / "test.csv"),
            "oof": _resolve(reference / "oof.parquet"),
            "metrics": _resolve(reference / "metrics.json"),
            "selection": _resolve(reference / "selection.json"),
            "deployment_stress": _resolve(
                PROJECT_ROOT / "artifacts" / "depth_shift_stress_20260813" / "result.json"
            ),
        }
        receipt = validate_gors_depth_preregistration_files(
            _resolve(args.preregistration),
            ledger_path=_resolve(args.ledger),
            observed_hashes={name: sha256_file(path) for name, path in paths.items()},
            require_outer_authorized=args.require_outer_authorized,
        )
    except (GORSDepthPreregistrationError, FileNotFoundError) as exc:
        raise SystemExit(f"G-ORS depth-invariance preregistration rejected: {exc}") from exc
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
