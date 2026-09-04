"""Independently validate a P3 submission and emit an aggregate receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from p3_wave.data import resolve_p3_data_dir
from p3_wave.submission import validate_submission


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--data-dir")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    root = resolve_p3_data_dir(args.data_dir)
    test_index = pd.read_csv(root / "test_index.csv")
    sample = pd.read_csv(root / "sample_submission.csv")
    submission = pd.read_csv(args.submission)
    validate_submission(submission, test_index)
    if not sample[["case_id", "station", "lead_h"]].equals(
        test_index[["case_id", "station", "lead_h"]]
    ):
        raise ValueError("sample_submission and test_index key order differ")
    prediction = submission["hs_pred"].to_numpy(dtype=float)
    receipt = {
        "validated_at": datetime.now().astimezone().isoformat(),
        "status": "passed_local_schema_and_key_validation",
        "official_hidden_score_available": False,
        "rows": int(len(submission)),
        "cases": int(submission["case_id"].nunique()),
        "stations": submission["station"].value_counts().sort_index().to_dict(),
        "leads": submission["lead_h"].value_counts().sort_index().to_dict(),
        "prediction": {
            "minimum": float(np.min(prediction)),
            "median": float(np.median(prediction)),
            "maximum": float(np.max(prediction)),
            "mean": float(np.mean(prediction)),
        },
        "submission_sha256": _sha256(args.submission),
        "test_index_sha256": _sha256(root / "test_index.csv"),
        "sample_submission_sha256": _sha256(root / "sample_submission.csv"),
        "uploaded": False,
    }
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
