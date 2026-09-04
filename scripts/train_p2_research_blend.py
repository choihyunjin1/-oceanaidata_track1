"""Train and validate the research-only P2 lean-M2 blend candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.features import build_test_features, build_training_features
from p2_restore.research import (
    append_public_dynamics,
    fit_research_blend,
    select_lean_m2_dynamics,
)
from p2_restore.submission import build_submission, validate_submission


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/p2_blend50/model.joblib"))
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("submissions/p2/P2_RESEARCH_BLEND50.csv"),
    )
    args = parser.parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    data = load_p2_data(data_dir)

    train_base = build_training_features(data.observations)
    train_dynamic = append_public_dynamics(train_base, data.observations)
    train_lean = select_lean_m2_dynamics(train_base, train_dynamic)
    model = fit_research_blend(train_base, train_lean)

    test_base = build_test_features(data)
    test_dynamic = append_public_dynamics(test_base, data.observations)
    test_lean = select_lean_m2_dynamics(test_base, test_dynamic)
    prediction = model.predict(test_base, test_lean)
    submission = build_submission(data.test_index, prediction)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    submission.to_csv(args.submission, index=False, encoding="utf-8", lineterminator="\n")
    validation = validate_submission(args.submission, data.test_index)

    additions = [
        column for column in test_lean.feature_columns if column not in test_base.feature_columns
    ]
    coverage = test_lean.frame[additions].notna().mean()
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "research_only": True,
        "uploaded": False,
        "method": "fixed 50:50 v0 + lean public-temperature M2 dynamics",
        "seed": 20260816,
        "rows": validation["rows"],
        "prediction_min": validation["minimum"],
        "prediction_max": validation["maximum"],
        "prediction_mean": float(np.mean(prediction)),
        "lean_feature_count": len(additions),
        "test_feature_finite_rate_min": float(coverage.min()),
        "test_feature_finite_rate_median": float(coverage.median()),
        "model_sha256": _sha256(args.model),
        "submission_sha256": _sha256(args.submission),
        "source_hashes": {
            name: _sha256(data_dir / name)
            for name in ("observations.csv", "test_index.csv", "baseline_interp.csv")
        },
    }
    manifest_path = args.model.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
