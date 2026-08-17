"""Build a ready-to-upload P3 submission from the frozen router and persistence files."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from p3_wave.data import resolve_p3_data_dir
from p3_wave.persistence_shrink import (
    FROZEN_LONG_PERSISTENCE_WEIGHT,
    LONG_LEADS,
    apply_long_lead_persistence_shrink,
)
from p3_wave.submission import validate_submission, write_submission

KEYS = ["case_id", "station", "lead_h"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--incumbent", default="submissions/p3_lead_long_loss_router/submission.csv"
    )
    parser.add_argument("--output-dir", default="submissions/p3_long_persistence_shrink")
    args = parser.parse_args()
    root = resolve_p3_data_dir(args.data_dir)
    incumbent_path = Path(args.incumbent)
    persistence_path = root / "baseline_persistence.csv"
    test_index = pd.read_csv(root / "test_index.csv")
    incumbent = pd.read_csv(incumbent_path)
    persistence = pd.read_csv(persistence_path)
    validate_submission(incumbent, test_index)
    if not persistence[KEYS].equals(test_index[KEYS]):
        raise ValueError("persistence keys/order differ from test_index")
    if "hs_pred" not in persistence:
        raise ValueError("baseline_persistence.csv is missing hs_pred")
    prediction = apply_long_lead_persistence_shrink(
        incumbent["hs_pred"].to_numpy(float),
        persistence["hs_pred"].to_numpy(float),
        test_index["lead_h"].to_numpy(int),
    )
    submission = test_index.copy()
    submission["hs_pred"] = prediction
    validate_submission(submission, test_index)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    submission_path = write_submission(submission, test_index, output / "submission.csv")
    short = test_index["lead_h"].isin([3, 6, 9])
    short_error = float(
        np.max(
            np.abs(
                submission.loc[short, "hs_pred"].to_numpy(float)
                - incumbent.loc[short, "hs_pred"].to_numpy(float)
            )
        )
    )
    if short_error != 0.0:
        raise ValueError("short-lead hidden predictions are not exact incumbent values")
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "local_candidate_not_uploaded",
        "problem": "P3_wave_forecast",
        "model": "frozen long-lead component-loss router shrunk toward persistence",
        "config": {
            "active_leads": list(LONG_LEADS),
            "persistence_weight": FROZEN_LONG_PERSISTENCE_WEIGHT,
        },
        "rows": int(len(submission)),
        "cases": int(submission["case_id"].nunique()),
        "prediction": {
            "minimum": float(np.min(prediction)),
            "median": float(np.median(prediction)),
            "maximum": float(np.max(prediction)),
            "short_lead_incumbent_max_abs_difference": short_error,
        },
        "sha256": {
            "submission": _sha256(submission_path),
            "incumbent_submission": _sha256(incumbent_path),
            "baseline_persistence": _sha256(persistence_path),
            "test_index": _sha256(root / "test_index.csv"),
        },
        "external_observations_used": 0,
        "hidden_test_labels_used": 0,
        "uploaded": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
