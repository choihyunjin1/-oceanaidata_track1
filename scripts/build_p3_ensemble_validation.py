"""Reconcile the fixed 50:50 single/multi-output P3 ensemble on exact OOF keys."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from p3_wave.validation import metric_slices, rmse


def _bootstrap(frame: pd.DataFrame, replicates: int = 2_000) -> dict[str, object]:
    groups = [group for _, group in frame.groupby(["fold", "anchor_id"], sort=False)]
    rng = np.random.default_rng(20260817)
    ensemble_delta = np.empty(replicates, dtype=float)
    persistence_delta = np.empty(replicates, dtype=float)
    for index in range(replicates):
        block = pd.concat(
            [groups[item] for item in rng.integers(0, len(groups), len(groups))],
            ignore_index=True,
        )
        truth = block["target_hs"].to_numpy(float)
        ensemble = block["prediction"].to_numpy(float)
        single = block["single_prediction"].to_numpy(float)
        persistence = block["persistence"].to_numpy(float)
        ensemble_delta[index] = rmse(truth, ensemble) - rmse(truth, single)
        persistence_delta[index] = rmse(truth, ensemble) - rmse(truth, persistence)
    return {
        "replicates": replicates,
        "seed": 20260817,
        "ensemble_minus_single": {
            "ci90": np.quantile(ensemble_delta, [0.05, 0.95]).tolist(),
            "probability_improved": float(np.mean(ensemble_delta < 0.0)),
        },
        "ensemble_minus_persistence": {
            "ci90": np.quantile(persistence_delta, [0.05, 0.95]).tolist(),
            "probability_improved": float(np.mean(persistence_delta < 0.0)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-oof", default="artifacts/p3/initial_tournament_all20/oof.parquet")
    parser.add_argument("--multi-oof", default="artifacts/p3/multioutput_gpu_probe/oof.parquet")
    parser.add_argument("--output-dir", default="artifacts/p3/final_ensemble_validation")
    args = parser.parse_args()
    single = pd.read_parquet(args.single_oof)
    single = single.loc[single["backend"].eq("catboost")].copy()
    multi = pd.read_parquet(args.multi_oof)
    multi = multi.loc[multi["configuration"].eq("cat_multi_compact")].copy()
    keys = ["fold", "anchor_id", "station", "lead_h"]
    frame = single.merge(
        multi[keys + ["prediction"]],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_single", "_multi"),
    )
    if len(frame) != 1_092 or not np.array_equal(
        frame["target_hs"].to_numpy(),
        single["target_hs"].to_numpy(),
        equal_nan=True,
    ):
        raise ValueError("single/multi OOF keys or labels do not align exactly")
    frame = frame.rename(
        columns={
            "prediction_single": "single_prediction",
            "prediction_multi": "multi_prediction",
        }
    )
    frame["prediction"] = 0.5 * (frame["single_prediction"] + frame["multi_prediction"])
    metrics = {
        "ensemble": metric_slices(frame, frame["prediction"].to_numpy()),
        "single": metric_slices(frame, frame["single_prediction"].to_numpy()),
        "multi": metric_slices(frame, frame["multi_prediction"].to_numpy()),
        "persistence": metric_slices(frame, frame["persistence"].to_numpy()),
        "folds": {
            fold: metric_slices(group, group["prediction"].to_numpy())
            for fold, group in frame.groupby("fold", observed=True)
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    oof_path = output / "oof.parquet"
    frame.to_parquet(oof_path, index=False, compression="zstd")
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "research_only_fixed_equal_weight_ensemble",
        "selection_caveat": (
            "The equal-weight ensemble was evaluated after both component OOF results were "
            "visible; it is not a virgin model-family comparison."
        ),
        "weights": {"single": 0.5, "multi": 0.5},
        "metrics": metrics,
        "paired_case_bootstrap": _bootstrap(frame),
        "oof_sha256": hashlib.sha256(oof_path.read_bytes()).hexdigest(),
    }
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
