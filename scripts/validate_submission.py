"""Validate a P1 submission without access to hidden labels.

This mirrors the strict input checks in the supplied score.py and adds a
reproducibility summary. It never edits the submission or source dataset.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["station", "year", "layer", "time"]
REQUIRED = KEY + ["label"]
ALLOWED_ORDERS = [REQUIRED, REQUIRED + ["anomaly_type"]]
DEFAULT_TEST = (
    Path(__file__).resolve().parents[1]
    / "데이터셋 원본"
    / "데이터셋_P1"
    / "P1_qc_anomaly"
    / "test.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def validate(submission_path: Path, test_path: Path) -> None:
    if not submission_path.is_file():
        fail(f"submission not found: {submission_path}")
    if not test_path.is_file():
        fail(f"test file not found: {test_path}")

    submission = pd.read_csv(submission_path)
    test = pd.read_csv(test_path)

    if list(submission.columns) not in ALLOWED_ORDERS:
        fail(
            f"column names/order are {list(submission.columns)}; "
            f"allowed orders are {ALLOWED_ORDERS}"
        )
    if list(test.columns) != KEY + ["temp", "psal", "depth"]:
        fail(f"unexpected test schema: {list(test.columns)}")
    if len(submission) != len(test):
        fail(f"row count is {len(submission):,}; expected {len(test):,}")
    if submission[KEY].isna().any().any():
        fail("submission keys contain missing values")
    duplicate_count = int(submission.duplicated(KEY).sum())
    if duplicate_count:
        fail(f"submission contains {duplicate_count:,} duplicate keys")
    if test.duplicated(KEY).any():
        fail("source test keys are not unique")

    try:
        merged = test[KEY].merge(
            submission[REQUIRED],
            on=KEY,
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
    except Exception as exc:
        fail(f"key validation failed: {exc}")
    mismatch_count = int(merged["_merge"].ne("both").sum())
    if mismatch_count:
        fail(f"key set differs from test by {mismatch_count:,} rows")

    numeric_label = pd.to_numeric(merged["label"], errors="coerce")
    values = numeric_label.to_numpy(dtype=float, na_value=np.nan)
    if numeric_label.isna().any() or not np.isfinite(values).all():
        fail("label must contain finite numeric values only")
    if not numeric_label.isin([0, 1]).all():
        invalid_preview = numeric_label[~numeric_label.isin([0, 1])].head().tolist()
        fail(f"label must be binary integer 0/1; examples: {invalid_preview}")

    order_matches = submission[KEY].equals(test[KEY])
    positive_count = int(numeric_label.sum())
    positive_rate = positive_count / len(submission)
    print("PASS: P1 submission structure is valid")
    print(f"path={submission_path.resolve()}")
    print(f"rows={len(submission):,}")
    print(f"positive={positive_count:,} ({positive_rate:.6%})")
    print(f"test_order_match={order_matches}")
    print(f"bytes={submission_path.stat().st_size:,}")
    print(f"sha256={sha256(submission_path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="candidate submission CSV")
    parser.add_argument(
        "--test",
        type=Path,
        default=DEFAULT_TEST,
        help=f"test CSV (default: {DEFAULT_TEST})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate(args.submission, args.test)
