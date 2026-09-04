"""Strictly validate a P1 submission without hidden labels."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

from p1_qc.submission import SubmissionValidationError, validate_submission

PROJECT_ROOT = Path(__file__).resolve().parents[1]
P1_REQUIRED_FILES = (
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "baseline_rule.csv",
    "README.md",
)


def default_test_path(
    *,
    env: Mapping[str, str] | None = None,
    search_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve test.csv from P1_DATA_DIR or one complete local P1 file set."""

    environment = os.environ if env is None else env
    configured = environment.get("P1_DATA_DIR")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        missing = [name for name in P1_REQUIRED_FILES if not (candidate / name).is_file()]
        if missing:
            raise FileNotFoundError(f"P1_DATA_DIR is missing {missing}: {candidate}")
        return candidate / "test.csv"

    candidates = {
        path.parent.resolve()
        for path in search_root.rglob("train.csv")
        if all((path.parent / name).is_file() for name in P1_REQUIRED_FILES)
    }
    if len(candidates) != 1:
        raise FileNotFoundError(
            "set P1_DATA_DIR or --test; repository fallback requires exactly one "
            f"complete P1 file set, found {len(candidates)}"
        )
    return next(iter(candidates)) / "test.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--test", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        test_path = args.test.expanduser().resolve() if args.test else default_test_path()
        report = validate_submission(args.submission, test_path)
    except (SubmissionValidationError, FileNotFoundError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print("PASS: P1 submission structure is valid")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
