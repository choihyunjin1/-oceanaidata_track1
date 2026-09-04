"""Validate the permanently stopped ring-residual audit contract.

The historical filename is retained for discoverability.  This command cannot
fit a model, predict, score, or inspect any prior evaluation artifact.
"""

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

from p1_qc.ring_inner_experiment import (  # noqa: E402
    assert_safe_audit_path,
    audit_runner_ast,
    load_ring_inner_contract,
    validate_ring_inner_contract,
    verify_coverage_artifact,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/experiments/p1_ring_residual_inner_only.json"),
    )
    parser.add_argument(
        "--coverage-artifact",
        type=Path,
        default=Path("artifacts/ring_coverage_audit_20260813/result.json"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("artifacts/ring_inner_audit/contract_receipt.json"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    audit_runner_ast(Path(__file__))
    receipt_path = assert_safe_audit_path(arguments.receipt)
    payload = load_ring_inner_contract(arguments.contract)
    receipt = validate_ring_inner_contract(payload)
    receipt["coverage_evidence"] = verify_coverage_artifact(arguments.coverage_artifact)
    receipt["decision"] = "STOP_PERMANENT_NO_GO"
    receipt["reason"] = "preregistered deployment coverage gates failed"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
