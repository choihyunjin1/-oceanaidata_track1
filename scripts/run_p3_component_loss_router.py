"""Run the chronological P3 component-loss soft-router experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from p3_wave.loss_router import (
    LEADS,
    build_case_router_data,
    expand_case_router_rows,
    run_prequential_lead_router,
    run_prequential_router,
)
from p3_wave.validation import metric_slices, rmse

FOLD_ORDER = ("2024_h2_storm", "winter_transition", "2025_h1")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_truth(oof: pd.DataFrame) -> np.ndarray:
    ordered = oof.sort_values(["fold", "anchor_id", "lead_h"])
    return np.stack(
        [
            group.sort_values("lead_h")["target_hs"].to_numpy(float)
            for _, group in ordered.groupby(["fold", "anchor_id"], sort=False)
        ]
    )


def _bootstrap(frame: pd.DataFrame, replicates: int, seed: int) -> dict[str, object]:
    cases = [group for _, group in frame.groupby(["fold", "anchor_id"], sort=False)]
    rng = np.random.default_rng(seed)
    delta = np.empty(replicates, dtype=float)
    for index in range(replicates):
        draw = pd.concat(
            [cases[item] for item in rng.integers(0, len(cases), len(cases))],
            ignore_index=True,
        )
        truth = draw["target_hs"].to_numpy(float)
        delta[index] = rmse(truth, draw["prediction"]) - rmse(truth, draw["frozen_prediction"])
    return {
        "replicates": replicates,
        "seed": seed,
        "delta_rmse_candidate_minus_frozen": float(
            rmse(frame["target_hs"], frame["prediction"])
            - rmse(frame["target_hs"], frame["frozen_prediction"])
        ),
        "ci90": np.quantile(delta, [0.05, 0.95]).tolist(),
        "probability_candidate_improved": float(np.mean(delta < 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="artifacts/p3/features_all20_v1")
    parser.add_argument("--oof", default="artifacts/p3/final_ensemble_validation/oof.parquet")
    parser.add_argument(
        "--frozen-metrics", default="artifacts/p3/final_ensemble_validation/metrics.json"
    )
    parser.add_argument("--granularity", choices=("case", "lead", "lead_long"), default="case")
    parser.add_argument("--output-dir")
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    args = parser.parse_args()
    cache = Path(args.cache_dir)
    oof_path = Path(args.oof)
    metrics_path = Path(args.frozen_metrics)
    default_output = {
        "case": "artifacts/p3/component_loss_router",
        "lead": "artifacts/p3/lead_loss_router",
        "lead_long": "artifacts/p3/lead_long_loss_router",
    }[args.granularity]
    output = Path(args.output_dir or default_output)
    output.mkdir(parents=True, exist_ok=True)

    oof = pd.read_parquet(oof_path)
    features = pd.read_parquet(cache / "train_features.parquet")
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    frozen = json.loads(metrics_path.read_text(encoding="utf-8"))
    if _sha256(oof_path) != frozen["oof_sha256"]:
        raise ValueError("frozen OOF hash does not match its metric manifest")
    observed_rmse = rmse(oof["target_hs"], oof["prediction"])
    if not np.isclose(
        observed_rmse, float(frozen["metrics"]["ensemble"]["rmse"]), rtol=0.0, atol=1e-12
    ):
        raise ValueError("frozen OOF RMSE does not reconcile")

    case_features, case_metadata, case_components, case_losses = build_case_router_data(
        oof, features, anchors
    )
    truth = _case_truth(oof)
    if args.granularity == "case":
        router_features = case_features
        metadata = case_metadata
        result = run_prequential_router(
            router_features,
            metadata,
            case_components,
            case_losses,
            truth,
            fold_order=FOLD_ORDER,
        )
        routed = pd.DataFrame(
            {
                "fold": np.repeat(metadata["fold"].to_numpy(str), len(LEADS)),
                "anchor_id": np.repeat(metadata["anchor_id"].to_numpy(np.int64), len(LEADS)),
                "lead_h": np.tile(np.asarray(LEADS, dtype=int), len(metadata)),
                "prediction": result.prediction.reshape(-1),
            }
        )
        weights = metadata[["fold", "anchor_id"]].copy()
        weight_keys = ["fold", "anchor_id"]
        weight_validation = "many_to_one"
    else:
        router_features, metadata, components, row_losses = expand_case_router_rows(
            case_features, case_metadata, case_components, truth
        )
        active_leads = LEADS if args.granularity == "lead" else (12, 18, 24)
        result = run_prequential_lead_router(
            router_features,
            metadata,
            components,
            row_losses,
            truth.reshape(-1),
            fold_order=FOLD_ORDER,
            active_leads=active_leads,
        )
        routed = metadata[["fold", "anchor_id", "lead_h"]].copy()
        routed["prediction"] = result.prediction
        weights = metadata[["fold", "anchor_id", "lead_h"]].copy()
        weight_keys = ["fold", "anchor_id", "lead_h"]
        weight_validation = "one_to_one"
    for index, name in enumerate(("single", "multi", "persistence")):
        weights[f"weight_{name}"] = result.weights[:, index]
    frame = oof.merge(
        routed,
        on=["fold", "anchor_id", "lead_h"],
        how="left",
        validate="one_to_one",
        suffixes=("_frozen", ""),
    ).rename(columns={"prediction_frozen": "frozen_prediction"})
    frame = frame.merge(weights, on=weight_keys, how="left", validate=weight_validation)
    if len(frame) != 1_092 or not np.isfinite(frame["prediction"]).all():
        raise ValueError("router OOF grain or prediction validity failed")
    first_fold = frame["fold"].eq(FOLD_ORDER[0])
    if not np.array_equal(
        frame.loc[first_fold, "prediction"].to_numpy(),
        frame.loc[first_fold, "frozen_prediction"].to_numpy(),
    ):
        raise ValueError("first fold must remain an exact no-op")

    metrics = {
        "candidate": metric_slices(frame, frame["prediction"].to_numpy(float)),
        "frozen": metric_slices(frame, frame["frozen_prediction"].to_numpy(float)),
        "folds": {
            fold: {
                "candidate": metric_slices(group, group["prediction"].to_numpy(float)),
                "frozen": metric_slices(group, group["frozen_prediction"].to_numpy(float)),
            }
            for fold, group in frame.groupby("fold", observed=True)
        },
    }
    metrics["delta_candidate_minus_frozen"] = {
        "rmse": float(metrics["candidate"]["rmse"] - metrics["frozen"]["rmse"]),
        "by_lead": {
            lead: float(metrics["candidate"]["by_lead"][lead] - metrics["frozen"]["by_lead"][lead])
            for lead in metrics["frozen"]["by_lead"]
        },
        "by_station": {
            station: float(
                metrics["candidate"]["by_station"][station]
                - metrics["frozen"]["by_station"][station]
            )
            for station in metrics["frozen"]["by_station"]
        },
    }
    oof_output = output / "oof.parquet"
    frame.to_parquet(oof_output, index=False, compression="zstd")
    bootstrap = _bootstrap(frame, args.bootstrap_replicates, 20260817)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": f"research_only_chronological_{args.granularity}_loss_soft_router",
        "adaptive_research_caveat": (
            "The router hypothesis was selected after inspecting the same outer OOF failure "
            "structure; current-fold labels are excluded from each fold's router selection."
        ),
        "metric": "pooled six-lead row RMSE_m",
        "grain": {
            "cases": int(case_metadata.shape[0]),
            "router_training_granularity": args.granularity,
            "rows": int(len(frame)),
            "leads": list(LEADS),
        },
        "router_features": {
            "count": int(router_features.shape[1]),
            "future_target_columns": 0,
            "inputs": "station + observed past-48h summaries + frozen component trajectories",
        },
        "selections": list(result.selections),
        "metrics": metrics,
        "paired_case_bootstrap": bootstrap,
        "weight_summary": {
            name: {
                "mean": float(weights[f"weight_{name}"].mean()),
                "p10": float(weights[f"weight_{name}"].quantile(0.10)),
                "p90": float(weights[f"weight_{name}"].quantile(0.90)),
            }
            for name in ("single", "multi", "persistence")
        },
        "promotion_gate": {
            "candidate_rmse_below_frozen": bool(
                metrics["candidate"]["rmse"] < metrics["frozen"]["rmse"]
            ),
            "ci90_upper_below_zero": bool(bootstrap["ci90"][1] < 0.0),
            "all_non_noop_folds_improve": all(
                metrics["folds"][fold]["candidate"]["rmse"]
                < metrics["folds"][fold]["frozen"]["rmse"]
                for fold in FOLD_ORDER[1:]
            ),
        },
        "provenance": {
            "frozen_oof_sha256": _sha256(oof_path),
            "feature_cache_sha256": _sha256(cache / "train_features.parquet"),
            "anchor_cache_sha256": _sha256(cache / "train_anchors.parquet"),
            "router_oof_sha256": _sha256(oof_output),
            "external_observations_used": 0,
            "hidden_test_labels_used": 0,
            "submission_uploaded": False,
        },
    }
    path = output / "metrics.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
