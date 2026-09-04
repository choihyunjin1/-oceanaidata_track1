"""Validate a two-arm P3 matched probe and compute paired case bootstrap uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from p3_wave.validation import rmse


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_bootstrap(
    base: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    keys = ["fold", "anchor_id", "station", "lead_h"]
    paired = base[keys + ["target_hs", "prediction"]].merge(
        candidate[keys + ["target_hs", "prediction"]],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_base", "_candidate"),
    )
    if len(paired) != 1_092 or not np.array_equal(
        paired["target_hs_base"].to_numpy(), paired["target_hs_candidate"].to_numpy()
    ):
        raise ValueError("matched probe keys or targets do not reconcile")
    case_groups = [
        group.index.to_numpy(dtype=np.int64)
        for _, group in paired.groupby(["fold", "anchor_id"], sort=False)
    ]
    if len(case_groups) != 182 or any(len(group) != 6 for group in case_groups):
        raise ValueError("expected 182 independent six-lead cases")
    truth = paired["target_hs_base"].to_numpy(float)
    base_prediction = paired["prediction_base"].to_numpy(float)
    candidate_prediction = paired["prediction_candidate"].to_numpy(float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled_cases = rng.integers(0, len(case_groups), len(case_groups))
        sampled_rows = np.concatenate([case_groups[item] for item in sampled_cases])
        deltas[index] = rmse(truth[sampled_rows], candidate_prediction[sampled_rows]) - rmse(
            truth[sampled_rows], base_prediction[sampled_rows]
        )
    return {
        "replicates": replicates,
        "seed": seed,
        "delta_rmse_candidate_minus_base": rmse(truth, candidate_prediction)
        - rmse(truth, base_prediction),
        "ci90": [float(np.quantile(deltas, 0.05)), float(np.quantile(deltas, 0.95))],
        "probability_candidate_improved": float(np.mean(deltas < 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--base-arm", required=True)
    parser.add_argument("--candidate-arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    frame = pd.read_parquet(args.oof)
    if set(frame["arm"].astype(str)) != {args.base_arm, args.candidate_arm}:
        raise ValueError("OOF arm set does not match requested base and candidate")
    base = frame.loc[frame["arm"].eq(args.base_arm)].copy()
    candidate = frame.loc[frame["arm"].eq(args.candidate_arm)].copy()
    bootstrap = _case_bootstrap(
        base,
        candidate,
        replicates=args.replicates,
        seed=args.seed,
    )
    fold_rows = []
    for fold in sorted(base["fold"].unique()):
        base_fold = base.loc[base["fold"].eq(fold)]
        candidate_fold = candidate.loc[candidate["fold"].eq(fold)]
        base_score = rmse(base_fold["target_hs"], base_fold["prediction"])
        candidate_score = rmse(candidate_fold["target_hs"], candidate_fold["prediction"])
        fold_rows.append(
            {
                "fold": str(fold),
                "rows": int(len(base_fold)),
                "base_rmse": base_score,
                "candidate_rmse": candidate_score,
                "delta_rmse": candidate_score - base_score,
            }
        )
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "passed",
        "base_arm": args.base_arm,
        "candidate_arm": args.candidate_arm,
        "grain": {"cases": 182, "rows_per_arm": 1_092, "leads_per_case": 6},
        "paired_case_bootstrap": bootstrap,
        "folds": fold_rows,
        "oof_sha256": _sha256(args.oof),
        "raw_rows_written": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
