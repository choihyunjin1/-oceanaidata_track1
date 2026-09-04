"""Independent reconciliation of the fixed P3 long-lead shrinkage artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from p3_wave.persistence_shrink import apply_long_lead_persistence_shrink
from p3_wave.validation import rmse

KEYS = ["fold", "anchor_id", "station", "lead_h"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="artifacts/p3/long_persistence_shrink/oof.parquet")
    parser.add_argument("--metrics", default="artifacts/p3/long_persistence_shrink/metrics.json")
    parser.add_argument("--incumbent", default="artifacts/p3/lead_long_loss_router/oof.parquet")
    parser.add_argument(
        "--output", default="artifacts/p3/long_persistence_shrink/independent_validation.json"
    )
    args = parser.parse_args()
    candidate_path = Path(args.candidate)
    metrics_path = Path(args.metrics)
    incumbent_path = Path(args.incumbent)
    candidate = pd.read_parquet(candidate_path).sort_values(KEYS).reset_index(drop=True)
    incumbent = pd.read_parquet(incumbent_path).sort_values(KEYS).reset_index(drop=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if len(candidate) != 1_092 or candidate.duplicated(KEYS).any():
        raise ValueError("candidate grain or key uniqueness failed")
    if not candidate[KEYS].equals(incumbent[KEYS]):
        raise ValueError("candidate and incumbent keys differ")
    for column in ("target_hs", "persistence"):
        if not np.array_equal(candidate[column].to_numpy(), incumbent[column].to_numpy()):
            raise ValueError(f"candidate changed frozen column: {column}")
    if not np.array_equal(
        candidate["incumbent_prediction"].to_numpy(), incumbent["prediction"].to_numpy()
    ):
        raise ValueError("incumbent prediction was not reproduced exactly")
    rebuilt = apply_long_lead_persistence_shrink(
        candidate["incumbent_prediction"].to_numpy(float),
        candidate["persistence"].to_numpy(float),
        candidate["lead_h"].to_numpy(int),
    )
    reconstruction_error = float(np.max(np.abs(rebuilt - candidate["prediction"])))
    if reconstruction_error > 1e-12:
        raise ValueError("candidate does not reconstruct from the frozen rule")
    short = candidate["lead_h"].isin([3, 6, 9])
    short_error = float(
        np.max(
            np.abs(
                candidate.loc[short, "prediction"] - candidate.loc[short, "incumbent_prediction"]
            )
        )
    )
    if short_error != 0.0:
        raise ValueError("short leads are not an exact no-op")
    truth = candidate["target_hs"].to_numpy(float)
    candidate_score = rmse(truth, candidate["prediction"])
    incumbent_score = rmse(truth, candidate["incumbent_prediction"])
    if not np.isclose(candidate_score, metrics["metrics"]["candidate_rmse"], atol=1e-12):
        raise ValueError("candidate RMSE does not reconcile")
    if not np.isclose(incumbent_score, metrics["metrics"]["incumbent_rmse"], atol=1e-12):
        raise ValueError("incumbent RMSE does not reconcile")
    fold_deltas = {
        str(fold): rmse(group["target_hs"], group["prediction"])
        - rmse(group["target_hs"], group["incumbent_prediction"])
        for fold, group in candidate.groupby("fold", sort=True)
    }
    station_deltas = {
        str(station): rmse(group["target_hs"], group["prediction"])
        - rmse(group["target_hs"], group["incumbent_prediction"])
        for station, group in candidate.groupby("station", sort=True)
    }
    if any(delta >= 0.0 for delta in (*fold_deltas.values(), *station_deltas.values())):
        raise ValueError("candidate must improve every observed fold and station")
    result = {
        "validated_at": datetime.now().astimezone().isoformat(),
        "status": "passed_independent_oof_reconciliation",
        "metrics": {
            "candidate_rmse": candidate_score,
            "incumbent_rmse": incumbent_score,
            "delta_rmse": candidate_score - incumbent_score,
            "fold_deltas": fold_deltas,
            "station_deltas": station_deltas,
        },
        "invariants": {
            "prediction_reconstruction_max_error": reconstruction_error,
            "short_lead_no_op_max_error": short_error,
            "raw_rows_written": 0,
            "hidden_test_labels_used": 0,
            "official_score": False,
        },
        "sha256": {
            "candidate_oof": _sha256(candidate_path),
            "metrics": _sha256(metrics_path),
            "incumbent_oof": _sha256(incumbent_path),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
