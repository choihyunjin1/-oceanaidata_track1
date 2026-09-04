"""Independent reconciliation for a P3 component-loss router OOF artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from p3_wave.validation import rmse

KEYS = ["fold", "anchor_id", "station", "lead_h"]
COMPONENT_COLUMNS = ["single_prediction", "multi_prediction", "persistence"]
WEIGHT_COLUMNS = ["weight_single", "weight_multi", "weight_persistence"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap(frame: pd.DataFrame, replicates: int, seed: int) -> dict[str, object]:
    blocks = [group for _, group in frame.groupby(["fold", "anchor_id"], sort=False)]
    rng = np.random.default_rng(seed)
    delta = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = pd.concat(
            [blocks[item] for item in rng.integers(0, len(blocks), len(blocks))],
            ignore_index=True,
        )
        truth = sampled["target_hs"].to_numpy(float)
        delta[index] = rmse(truth, sampled["prediction"]) - rmse(
            truth, sampled["frozen_prediction"]
        )
    return {
        "ci90": np.quantile(delta, [0.05, 0.95]).tolist(),
        "probability_improved": float(np.mean(delta < 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="artifacts/p3/lead_long_loss_router/oof.parquet")
    parser.add_argument("--metrics", default="artifacts/p3/lead_long_loss_router/metrics.json")
    parser.add_argument("--frozen", default="artifacts/p3/final_ensemble_validation/oof.parquet")
    parser.add_argument(
        "--output", default="artifacts/p3/lead_long_loss_router/independent_validation.json"
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    args = parser.parse_args()
    candidate_path = Path(args.candidate)
    metrics_path = Path(args.metrics)
    frozen_path = Path(args.frozen)
    candidate_raw = pd.read_parquet(candidate_path)
    candidate = candidate_raw.sort_values(KEYS).reset_index(drop=True)
    frozen = pd.read_parquet(frozen_path).sort_values(KEYS).reset_index(drop=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    if len(candidate) != 1_092 or candidate["anchor_id"].nunique() != 182:
        raise ValueError("candidate must contain 182 cases and 1,092 rows")
    if candidate.duplicated(KEYS).any() or frozen.duplicated(KEYS).any():
        raise ValueError("OOF keys must be unique")
    if not candidate[KEYS].equals(frozen[KEYS]):
        raise ValueError("candidate/frozen keys do not align")
    for column in ("target_hs", *COMPONENT_COLUMNS):
        if not np.array_equal(
            candidate[column].to_numpy(float), frozen[column].to_numpy(float), equal_nan=True
        ):
            raise ValueError(f"candidate changed frozen source column: {column}")
    if not np.array_equal(
        candidate["frozen_prediction"].to_numpy(float),
        frozen["prediction"].to_numpy(float),
    ):
        raise ValueError("candidate frozen_prediction does not exactly reproduce incumbent")

    weights = candidate[WEIGHT_COLUMNS].to_numpy(float)
    components = candidate[COMPONENT_COLUMNS].to_numpy(float)
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("router weights must be finite and non-negative")
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("router weights do not sum to one")
    reconstructed = np.sum(weights * components, axis=1)
    reconstruction_error = float(
        np.max(np.abs(reconstructed - candidate["prediction"].to_numpy(float)))
    )
    if reconstruction_error > 1e-12:
        raise ValueError("router prediction cannot be reconstructed from components and weights")

    first = candidate["fold"].eq("2024_h2_storm")
    first_error = float(
        np.max(
            np.abs(
                candidate.loc[first, "prediction"].to_numpy(float)
                - candidate.loc[first, "frozen_prediction"].to_numpy(float)
            )
        )
    )
    if first_error != 0.0:
        raise ValueError("first chronological fold is not an exact no-op")
    if metrics["grain"]["router_training_granularity"] == "lead_long":
        short = candidate["lead_h"].isin([3, 6, 9])
        short_error = float(
            np.max(
                np.abs(
                    candidate.loc[short, "prediction"].to_numpy(float)
                    - candidate.loc[short, "frozen_prediction"].to_numpy(float)
                )
            )
        )
        if short_error != 0.0:
            raise ValueError("short leads must be exact frozen predictions")
    else:
        short_error = None

    truth = candidate["target_hs"].to_numpy(float)
    candidate_rmse = rmse(truth, candidate["prediction"])
    frozen_rmse = rmse(truth, candidate["frozen_prediction"])
    if not np.isclose(
        candidate_rmse, metrics["metrics"]["candidate"]["rmse"], rtol=0.0, atol=1e-12
    ) or not np.isclose(frozen_rmse, metrics["metrics"]["frozen"]["rmse"], rtol=0.0, atol=1e-12):
        raise ValueError("headline RMSE does not reconcile")
    by_fold = {
        fold: {
            "rows": int(len(group)),
            "candidate_rmse": rmse(group["target_hs"], group["prediction"]),
            "frozen_rmse": rmse(group["target_hs"], group["frozen_prediction"]),
        }
        for fold, group in candidate.groupby("fold", observed=True)
    }
    by_lead = {
        str(int(lead)): {
            "rows": int(len(group)),
            "candidate_rmse": rmse(group["target_hs"], group["prediction"]),
            "frozen_rmse": rmse(group["target_hs"], group["frozen_prediction"]),
        }
        for lead, group in candidate.groupby("lead_h", observed=True)
    }
    bootstrap = _bootstrap(candidate_raw, args.bootstrap_replicates, 20260817)
    stated = metrics["paired_case_bootstrap"]
    if not np.allclose(bootstrap["ci90"], stated["ci90"], rtol=0.0, atol=1e-12):
        raise ValueError("paired bootstrap CI does not reconcile")

    result = {
        "validated_at": datetime.now().astimezone().isoformat(),
        "status": "passed_independent_oof_reconciliation",
        "confidence": "share_with_adaptive_research_caveat",
        "grain": {"cases": 182, "rows": 1_092, "duplicate_keys": 0},
        "metrics": {
            "candidate_rmse": candidate_rmse,
            "frozen_rmse": frozen_rmse,
            "delta_rmse": candidate_rmse - frozen_rmse,
            "by_fold": by_fold,
            "by_lead": by_lead,
        },
        "bootstrap": bootstrap,
        "invariants": {
            "frozen_source_columns_exact": True,
            "weight_sum_max_error": float(np.max(np.abs(weights.sum(axis=1) - 1.0))),
            "prediction_reconstruction_max_error": reconstruction_error,
            "first_fold_no_op_max_error": first_error,
            "short_lead_no_op_max_error": short_error,
            "raw_rows_written": 0,
            "external_observations_used": 0,
            "hidden_test_labels_used": 0,
        },
        "sha256": {
            "candidate_oof": _sha256(candidate_path),
            "metrics": _sha256(metrics_path),
            "frozen_oof": _sha256(frozen_path),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
