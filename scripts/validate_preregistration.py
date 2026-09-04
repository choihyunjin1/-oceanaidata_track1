"""Validate a P1 preregistration before any inner or outer experiment runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from p1_qc.preregistration import (
    PreregistrationError,
    load_preregistration,
    read_experiment_ledger,
    validate_preregistration,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    try:
        payload = load_preregistration(args.preregistration)
        ledger = read_experiment_ledger(args.ledger) if args.ledger else []
        receipt = validate_preregistration(payload, ledger_rows=ledger)
    except PreregistrationError as exc:
        parser.exit(2, f"preregistration rejected: {exc}\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
