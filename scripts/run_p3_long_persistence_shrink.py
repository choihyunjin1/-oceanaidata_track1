"""Evaluate the fixed P3 long-lead persistence shrinkage on frozen OOF rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from p3_wave.persistence_shrink import (
    FROZEN_LONG_PERSISTENCE_WEIGHT,
    LONG_LEADS,
    LongLeadPersistenceShrink,
    apply_long_lead_persistence_shrink,
)
from p3_wave.validation import rmse

KEYS = ["fold", "anchor_id", "station", "lead_h"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group_metrics(frame: pd.DataFrame, column: str) -> dict[str, dict[str, float | int]]:
    return {
        str(key): {
            "rows": int(len(group)),
            "incumbent_rmse": rmse(group["target_hs"], group["incumbent_prediction"]),
            "candidate_rmse": rmse(group["target_hs"], group["prediction"]),
            "delta_rmse": rmse(group["target_hs"], group["prediction"])
            - rmse(group["target_hs"], group["incumbent_prediction"]),
        }
        for key, group in frame.groupby(column, sort=True, observed=True)
    }


def _paired_case_bootstrap(frame: pd.DataFrame, *, replicates: int, seed: int) -> dict[str, object]:
    blocks = [group.index.to_numpy() for _, group in frame.groupby(["fold", "anchor_id"])]
    rng = np.random.default_rng(seed)
    delta = np.empty(replicates, dtype=float)
    truth = frame["target_hs"].to_numpy(float)
    incumbent = frame["incumbent_prediction"].to_numpy(float)
    candidate = frame["prediction"].to_numpy(float)
    for replicate in range(replicates):
        indices = np.concatenate(
            [blocks[index] for index in rng.integers(0, len(blocks), len(blocks))]
        )
        delta[replicate] = rmse(truth[indices], candidate[indices]) - rmse(
            truth[indices], incumbent[indices]
        )
    return {
        "replicates": replicates,
        "seed": seed,
        "ci90": np.quantile(delta, [0.05, 0.95]).tolist(),
        "median": float(np.median(delta)),
        "probability_improved": float(np.mean(delta < 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", default="artifacts/p3/lead_long_loss_router/oof.parquet")
    parser.add_argument("--output-dir", default="artifacts/p3/long_persistence_shrink")
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    args = parser.parse_args()
    source = Path(args.incumbent)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source).sort_values(KEYS).reset_index(drop=True)
    required = {*KEYS, "target_hs", "prediction", "persistence"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"incumbent OOF is missing columns: {sorted(missing)}")
    if len(frame) != 1_092 or frame["anchor_id"].nunique() != 182 or frame.duplicated(KEYS).any():
        raise ValueError("incumbent OOF must contain 182 unique cases and 1,092 unique rows")

    frame = frame.rename(columns={"prediction": "incumbent_prediction"})
    config = LongLeadPersistenceShrink()
    frame["prediction"] = apply_long_lead_persistence_shrink(
        frame["incumbent_prediction"].to_numpy(float),
        frame["persistence"].to_numpy(float),
        frame["lead_h"].to_numpy(int),
        config=config,
    )
    frame["second_stage_persistence_weight"] = np.where(
        frame["lead_h"].isin(config.active_leads), config.weight, 0.0
    )
    truth = frame["target_hs"].to_numpy(float)
    incumbent_score = rmse(truth, frame["incumbent_prediction"])
    candidate_score = rmse(truth, frame["prediction"])
    sensitivity = {}
    for weight in (0.15, 0.20, 0.25):
        prediction = apply_long_lead_persistence_shrink(
            frame["incumbent_prediction"].to_numpy(float),
            frame["persistence"].to_numpy(float),
            frame["lead_h"].to_numpy(int),
            config=LongLeadPersistenceShrink(weight=weight),
        )
        sensitivity[f"{weight:.2f}"] = rmse(truth, prediction)
    bootstrap = _paired_case_bootstrap(frame, replicates=args.bootstrap_replicates, seed=20260817)
    metrics = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "adaptive_local_research_candidate_not_uploaded",
        "grain": {"cases": 182, "rows": 1_092, "leads": 6, "duplicate_keys": 0},
        "config": {
            "active_leads": list(LONG_LEADS),
            "persistence_weight": FROZEN_LONG_PERSISTENCE_WEIGHT,
            "selection_note": "single bounded scalar chosen after local diagnostics; not a virgin holdout result",
        },
        "metrics": {
            "incumbent_rmse": incumbent_score,
            "candidate_rmse": candidate_score,
            "delta_rmse": candidate_score - incumbent_score,
            "by_fold": _group_metrics(frame, "fold"),
            "by_station": _group_metrics(frame, "station"),
            "by_lead": _group_metrics(frame, "lead_h"),
        },
        "paired_case_bootstrap": bootstrap,
        "bounded_sensitivity_rmse": sensitivity,
        "invariants": {
            "short_lead_max_abs_difference": float(
                np.max(
                    np.abs(
                        frame.loc[frame["lead_h"].isin([3, 6, 9]), "prediction"]
                        - frame.loc[frame["lead_h"].isin([3, 6, 9]), "incumbent_prediction"]
                    )
                )
            ),
            "target_used_for_prediction": False,
            "external_observations_used": 0,
            "hidden_test_labels_used": 0,
            "official_score": False,
        },
    }
    oof_path = output / "oof.parquet"
    metrics_path = output / "metrics.json"
    frame.to_parquet(oof_path, index=False)
    metrics["sha256"] = {"source_oof": _sha256(source), "candidate_oof": _sha256(oof_path)}
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
