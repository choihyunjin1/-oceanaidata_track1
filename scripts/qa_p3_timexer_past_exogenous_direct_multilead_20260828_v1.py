"""Independent aggregate QA for the P3 direct TimeXer local-only experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p3_wave.timexer_direct_multilead import promotion_gates, sha256_file

EXPERIMENT_ID = "p3_timexer_past_exogenous_direct_multilead_20260828_v1"
DEFAULT_CONFIG = Path(
    "configs/experiments/p3_timexer_past_exogenous_direct_multilead_20260828_v1.json"
)
DEFAULT_ARTIFACT = Path(f"artifacts/{EXPERIMENT_ID}")
PAIR_KEYS = ["fold", "anchor_id", "station", "lead_h"]


def _load_runner() -> Any:
    path = Path(__file__).with_name(
        "run_p3_timexer_past_exogenous_direct_multilead_20260828_v1.py"
    )
    spec = importlib.util.spec_from_file_location("_timexer_runner_qa", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import TimeXer runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rmse(frame: pd.DataFrame, prediction: str) -> float:
    return float(
        np.sqrt(np.mean(np.square(frame[prediction] - frame["target_hs"])))
    )


def _slices(frame: pd.DataFrame, prediction: str) -> dict[str, Any]:
    result: dict[str, Any] = {"pooled_rmse_m": _rmse(frame, prediction)}
    for dimension in ("fold", "station", "lead_h"):
        result[f"by_{dimension}_rmse_m"] = {
            str(key): _rmse(group, prediction)
            for key, group in frame.groupby(dimension, observed=True)
        }
    return result


def _delta(candidate: dict[str, Any], incumbent: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pooled_delta_m": candidate["pooled_rmse_m"] - incumbent["pooled_rmse_m"]
    }
    for dimension in ("fold", "station", "lead_h"):
        key = f"by_{dimension}_rmse_m"
        result[f"by_{dimension}_delta_m"] = {
            name: candidate[key][name] - incumbent[key][name]
            for name in candidate[key]
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.check_only == args.final:
        raise SystemExit("choose exactly one of --check-only or --final")
    config = _read(args.config)
    RUNNER._validate_config(config)
    if args.check_only:
        payload = {
            "experiment_id": EXPERIMENT_ID,
            "status": "QA_CHECK_ONLY_PASS",
            "config_sha256": sha256_file(args.config),
            "artifact_exists": args.artifact_dir.exists(),
            "writes": 0,
            "official_inputs_read": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    result_path = args.artifact_dir / "result.json"
    manifest_path = args.artifact_dir / "manifest.json"
    result = _read(result_path)
    manifest = _read(manifest_path)
    checks: dict[str, bool] = {
        "experiment_id": result["experiment_id"] == EXPERIMENT_ID,
        "config_hash": result["config_sha256"] == sha256_file(args.config),
        "manifest_result_hash": manifest["result_sha256"] == sha256_file(result_path),
        "official_inputs_absent": result["official_inputs_read"] is False,
        "candidate_csv_absent": result["candidate_csv_created"] is False,
        "upload_absent": result["uploaded"] is False,
        "protected_lineage": result["protected_lineage_hashes_before_after_identical"] is True,
    }
    for path, expected in result["protected_lineage_hashes"].items():
        checks["protected_lineage"] &= sha256_file(path) == expected

    recomputed: dict[str, Any] = {}
    if result["status"] == "TERMINAL_NO_GO_INNER_PERSISTENCE":
        checks["truth_unopened_on_inner_stop"] = result["outer_truth_open_count"] == 0
        checks["inner_stop_has_no_outer_predictions"] = all(
            record["outer_prediction_created"] is False
            for record in result["fold_seed_records"]
        )
    else:
        seal_path = args.artifact_dir / "PREDICTION_SEAL.json"
        evaluated_path = args.artifact_dir / "evaluated_outer_predictions.parquet"
        seal = _read(seal_path)
        frame = pd.read_parquet(evaluated_path)
        checks["manifest_seal_hash"] = (
            manifest["prediction_seal_sha256"] == sha256_file(seal_path)
        )
        checks["manifest_evaluated_hash"] = (
            manifest["evaluated_outer_sha256"] == sha256_file(evaluated_path)
        )
        checks["truth_late"] = (
            result["truth_first_read_after_prediction_seal"] is True
            and result["outer_truth_open_count"] == 1
            and seal["outer_truth_read_before_seal"] is False
        )
        checks["grain"] = (
            len(frame) == 1086
            and frame["anchor_id"].nunique() == 181
            and not frame.duplicated(PAIR_KEYS).any()
        )
        checks["direct_all_six_leads"] = np.array_equal(
            frame["candidate_prediction"].to_numpy(),
            frame["timexer_prediction"].to_numpy(),
        )
        checkpoint_hashes = True
        prediction_hashes = True
        for record in seal["fold_seed_prediction_files"]:
            prediction_hashes &= sha256_file(record["path"]) == record["sha256"]
            matching = [
                item
                for item in result["fold_seed_records"]
                if item["fold"] == record["fold"] and item["seed"] == record["seed"]
            ]
            if len(matching) != 1:
                checkpoint_hashes = False
            else:
                checkpoint_hashes &= (
                    sha256_file(matching[0]["inner"]["checkpoint_path"])
                    == record["inner_checkpoint_sha256"]
                    and sha256_file(matching[0]["outer"]["checkpoint_path"])
                    == record["outer_checkpoint_sha256"]
                )
        checks["prediction_hashes"] = prediction_hashes
        checks["checkpoint_hashes"] = checkpoint_hashes
        candidate = _slices(frame, "candidate_prediction")
        incumbent = _slices(frame, "incumbent_prediction")
        delta = _delta(candidate, incumbent)
        bootstrap = RUNNER.BASE._case_bootstrap(
            frame,
            replicates=int(config["validation"]["bootstrap_replicates"]),
            seed=int(config["validation"]["bootstrap_seed"]),
        )
        gates = promotion_gates(
            pooled_delta_m=float(delta["pooled_delta_m"]),
            fold_deltas_m=delta["by_fold_delta_m"],
            station_deltas_m=delta["by_station_delta_m"],
            lead_deltas_m=delta["by_lead_h_delta_m"],
            bootstrap_ci90_upper_m=float(bootstrap["ci90_m"][1]),
        )
        checks["metrics_recomputed"] = (
            np.isclose(
                candidate["pooled_rmse_m"],
                result["metrics"]["candidate"]["pooled_rmse_m"],
                atol=1e-12,
            )
            and np.isclose(
                delta["pooled_delta_m"],
                result["metrics"]["delta_vs_incumbent"]["pooled_delta_m"],
                atol=1e-12,
            )
            and np.allclose(
                bootstrap["ci90_m"],
                result["metrics"]["bootstrap_vs_incumbent"]["ci90_m"],
                atol=1e-12,
            )
        )
        checks["gates_recomputed"] = gates == result["gates"]
        recomputed = {
            "candidate": candidate,
            "incumbent": incumbent,
            "delta_vs_incumbent": delta,
            "bootstrap_vs_incumbent": bootstrap,
            "gates": gates,
        }

    passed = all(checks.values())
    qa = {
        "experiment_id": EXPERIMENT_ID,
        "qa_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "recomputed": recomputed,
    }
    qa_path = args.artifact_dir / "independent_qa.json"
    RUNNER.BASE._atomic_json(qa_path, qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
