"""Audit the S-ORS layer-5 implementation; outer CV is permanently disabled."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from p1_qc.experiment import sha256_file  # noqa: E402
from p1_qc.sors_l5_regime_preregistration import (  # noqa: E402
    SORSL5PreregistrationError,
    validate_sors_l5_preregistration_files,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("configs/experiments/p1_sors_l5_regime_invariance_draft.json"),
    )
    parser.add_argument("--ledger", type=Path, default=Path("reports/EXPERIMENT_LEDGER.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/p1.toml"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--outer-cv",
        action="store_true",
        help="Request the prohibited outer run (always fails closed).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve(strict=True)
    hashes = {
        "config": sha256_file(args.config.resolve(strict=True)),
        "train": sha256_file((data_dir / "train.csv").resolve(strict=True)),
        "test": sha256_file((data_dir / "test.csv").resolve(strict=True)),
    }
    try:
        receipt = validate_sors_l5_preregistration_files(
            args.preregistration.resolve(strict=True),
            ledger_path=args.ledger.resolve(strict=True),
            observed_hashes=hashes,
            require_outer_authorized=args.outer_cv,
        )
    except (SORSL5PreregistrationError, FileNotFoundError) as exc:
        raise SystemExit(f"S-ORS L5 experiment rejected: {exc}") from exc

    # This runner intentionally stops at contract validation.  It never loads
    # labels, builds features, fits a model, or creates an outer result.
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
