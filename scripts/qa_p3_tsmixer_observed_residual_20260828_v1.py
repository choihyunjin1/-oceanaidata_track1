"""Independent aggregate QA for the sealed P3 TSMixer v1 result."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p3_wave.tsmixer_residual import decision_gates, sha256_file


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rmse(truth: pd.Series, prediction: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(prediction.to_numpy() - truth.to_numpy()))))


def _slices(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    result: dict[str, Any] = {"pooled_rmse_m": _rmse(frame["target_hs"], frame[column])}
    for dimension in ("fold", "station", "lead_h"):
        result[f"by_{dimension}_rmse_m"] = {
            str(key): _rmse(group["target_hs"], group[column])
            for key, group in frame.groupby(dimension, observed=True)
        }
    return result


def _delta(candidate: dict[str, Any], incumbent: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pooled_delta_m": candidate["pooled_rmse_m"] - incumbent["pooled_rmse_m"]
    }
    for dimension in ("fold", "station", "lead_h"):
        name = f"by_{dimension}_rmse_m"
        result[f"by_{dimension}_delta_m"] = {
            key: candidate[name][key] - incumbent[name][key] for key in candidate[name]
        }
    return result


def _bootstrap(frame: pd.DataFrame, replicates: int, seed: int) -> dict[str, Any]:
    work = frame[["anchor_id"]].copy()
    work["candidate_sq"] = np.square(frame["candidate_prediction"] - frame["target_hs"])
    work["incumbent_sq"] = np.square(frame["incumbent_prediction"] - frame["target_hs"])
    work["rows"] = 1
    values = work.groupby("anchor_id", observed=True)[
        ["candidate_sq", "incumbent_sq", "rows"]
    ].sum().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        deltas[index] = np.sqrt(sampled[0] / sampled[2]) - np.sqrt(sampled[1] / sampled[2])
    return {
        "clusters": int(len(values)),
        "replicates": int(replicates),
        "ci90_m": [float(np.quantile(deltas, 0.05)), float(np.quantile(deltas, 0.95))],
        "probability_improved": float(np.mean(deltas < 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/p3_tsmixer_observed_residual_20260828_v1"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/p3_tsmixer_observed_residual_20260828_v1.json"),
    )
    args = parser.parse_args()
    artifact = args.artifact_dir
    result_path = artifact / "result.json"
    manifest_path = artifact / "manifest.json"
    seal_path = artifact / "PREDICTION_SEAL.json"
    evaluated_path = artifact / "evaluated_outer_predictions.parquet"
    result = _read(result_path)
    manifest = _read(manifest_path)
    seal = _read(seal_path)
    config = _read(args.config)
    checks: dict[str, bool] = {}
    checks["experiment_id"] = result["experiment_id"] == config["experiment_id"]
    checks["config_hash"] = result["config_sha256"] == sha256_file(args.config)
    checks["manifest_result_hash"] = (
        manifest["terminal_result"]["sha256"] == sha256_file(result_path)
    )
    checks["manifest_seal_hash"] = manifest["prediction_seal"]["sha256"] == sha256_file(
        seal_path
    )
    checks["evaluated_hash"] = result["evaluated_outer_sha256"] == sha256_file(evaluated_path)
    checks["truth_late"] = (
        result["truth_first_read_after_prediction_seal"] is True
        and result["outer_truth_open_count"] == 1
        and seal["outer_truth_read_before_seal"] is False
        and seal["official_inputs_read_before_gate"] is False
    )
    checks["nine_fold_seed_predictions"] = len(seal["fold_seed_prediction_files"]) == 9
    checkpoint_hashes_ok = True
    prediction_hashes_ok = True
    for record in seal["fold_seed_prediction_files"]:
        prediction_hashes_ok &= sha256_file(record["path"]) == record["sha256"]
        matching = [
            item
            for item in result["fold_seed_records"]
            if item["fold"] == record["fold"] and item["seed"] == record["seed"]
        ]
        if len(matching) != 1:
            checkpoint_hashes_ok = False
            continue
        checkpoint_hashes_ok &= (
            sha256_file(matching[0]["inner"]["checkpoint_path"])
            == record["inner_checkpoint_sha256"]
        )
        checkpoint_hashes_ok &= (
            sha256_file(matching[0]["outer"]["checkpoint_path"])
            == record["outer_checkpoint_sha256"]
        )
    checks["checkpoint_hashes"] = checkpoint_hashes_ok
    checks["prediction_hashes"] = prediction_hashes_ok

    frame = pd.read_parquet(evaluated_path)
    checks["grain"] = (
        len(frame) == 1086
        and frame["anchor_id"].nunique() == 181
        and not frame.duplicated(["fold", "anchor_id", "station", "lead_h"]).any()
    )
    early = frame["lead_h"].isin((3, 6, 9))
    checks["early_bit_exact"] = np.array_equal(
        frame.loc[early, "candidate_prediction"].to_numpy(),
        frame.loc[early, "incumbent_prediction"].to_numpy(),
    )
    candidate = _slices(frame, "candidate_prediction")
    incumbent = _slices(frame, "incumbent_prediction")
    delta = _delta(candidate, incumbent)
    bootstrap = _bootstrap(
        frame,
        int(config["evaluation"]["bootstrap_replicates"]),
        int(config["evaluation"]["bootstrap_seed"]),
    )
    reported = result["metrics"]
    checks["metrics_recomputed"] = (
        abs(candidate["pooled_rmse_m"] - reported["candidate"]["pooled_rmse_m"]) < 1e-12
        and abs(incumbent["pooled_rmse_m"] - reported["incumbent"]["pooled_rmse_m"]) < 1e-12
        and abs(delta["pooled_delta_m"] - reported["delta_vs_incumbent"]["pooled_delta_m"])
        < 1e-12
        and np.allclose(bootstrap["ci90_m"], reported["bootstrap_vs_incumbent"]["ci90_m"])
        and abs(
            bootstrap["probability_improved"]
            - reported["bootstrap_vs_incumbent"]["probability_improved"]
        )
        < 1e-12
    )
    novelty = _rmse(frame["incumbent_prediction"], frame["candidate_prediction"])
    gates = decision_gates(
        pooled_delta_m=float(delta["pooled_delta_m"]),
        fold_deltas_m=delta["by_fold_delta_m"],
        station_deltas_m=delta["by_station_delta_m"],
        lead_deltas_m=delta["by_lead_h_delta_m"],
        bootstrap_ci90_upper_m=float(bootstrap["ci90_m"][1]),
        probability_improved=float(bootstrap["probability_improved"]),
        novelty_rms_m=float(novelty),
        seed_rmse_spread_m=float(result["seed_rmse_spread_m"]),
        runtime_seconds=float(result["runtime_seconds_before_final_fit"]),
        maximum_seed_seconds=max(
            item["runtime_seconds"] for item in result["fold_seed_records"]
        ),
    )
    checks["gates_recomputed"] = gates == result["gates"]
    checks["protected_lineage"] = result["protected_lineage_hashes_before_after_identical"] is True
    for path, expected in result["protected_lineage_hashes"].items():
        checks["protected_lineage"] &= sha256_file(path) == expected

    if gates["official_info_go"]:
        candidate_record = result["final_candidate"]
        checks["official_branch_consistent"] = (
            candidate_record["created"] is True
            and candidate_record["uploaded"] is False
            and sha256_file(candidate_record["submission_path"])
            == candidate_record["submission_sha256"]
            and _read(Path(candidate_record["validator_receipt_path"]))["status"]
            == "passed_local_schema_and_key_validation"
        )
    else:
        checks["official_branch_consistent"] = (
            result["final_candidate"]["created"] is False
            and result["final_candidate"]["official_inputs_read"] is False
            and result["official_inputs_read_before_information_gate"] is False
        )

    passed = all(checks.values())
    qa = {
        "experiment_id": result["experiment_id"],
        "qa_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "recomputed": {
            "candidate": candidate,
            "incumbent": incumbent,
            "delta_vs_incumbent": delta,
            "bootstrap_vs_incumbent": bootstrap,
            "novelty_rms_m": novelty,
            "gates": gates,
        },
    }
    qa_path = artifact / "qa.json"
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    qa_manifest = {
        "qa_path": str(qa_path.resolve()),
        "qa_sha256": sha256_file(qa_path),
        "result_sha256": sha256_file(result_path),
        "evaluated_sha256": sha256_file(evaluated_path),
        "status": qa["status"],
    }
    (artifact / "qa_manifest.json").write_text(
        json.dumps(qa_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
