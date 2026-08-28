"""Independent aggregate-only QA for the sealed P2 heave experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from p2_restore.p2_public_heave_tangent_incumbent_20260828_v1 import (
    evaluate_gate,
    paired_kst_day_bootstrap,
    rmse,
)

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_public_heave_tangent_incumbent_20260828_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def decode_time(values: np.ndarray) -> pd.DatetimeIndex:
    raw = np.asarray(values, dtype=np.int64)
    unit = "ns" if int(np.max(np.abs(raw))) >= 10**17 else "us"
    return pd.DatetimeIndex(pd.to_datetime(raw, unit=unit, utc=True))


def prediction_frame(commitment: dict[str, object]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for block, output in commitment["prediction_outputs"].items():
        path = REPO / str(output["path"])
        require(sha256(path) == output["sha256"], f"prediction hash differs: {block}")
        with np.load(path, allow_pickle=False) as payload:
            frames.append(
                pd.DataFrame(
                    {
                        "time": decode_time(payload["time_ns"]),
                        "layer": payload["layer"].astype(int),
                        "block": block,
                        "reference": payload["reference"].astype(float),
                        "candidate": payload["candidate"].astype(float),
                        "correction": payload["correction"].astype(float),
                        "enabled": payload["enabled"].astype(bool),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def truth_frame(config: dict[str, object], selected: dict[str, object]) -> pd.DataFrame:
    specification = config["comparator_priority"][int(selected["priority_index"])]
    if specification["kind"] == "parquet":
        path = REPO / str(specification["path"])
        expected = str(specification["sha256"])
    else:
        path = REPO / str(specification["truth_anchor"]["path"])
        expected = str(specification["truth_anchor"]["sha256"])
    require(sha256(path) == expected, "truth lineage hash differs")
    frame = pd.read_parquet(path, columns=["time", "layer", "block", "truth"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["layer"] = pd.to_numeric(frame["layer"], errors="raise").astype(int)
    return frame


def metric(frame: pd.DataFrame) -> dict[str, float | int]:
    reference = rmse(frame["truth"].to_numpy(), frame["reference"].to_numpy())
    candidate = rmse(frame["truth"].to_numpy(), frame["candidate"].to_numpy())
    return {
        "rows": int(len(frame)),
        "incumbent_rmse_c": reference,
        "heave_candidate_rmse_c": candidate,
        "delta_rmse_c": candidate - reference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=REPO / "artifacts" / EXPERIMENT_ID,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs" / "experiments" / f"{EXPERIMENT_ID}.json",
    )
    args = parser.parse_args()
    artifact = args.artifact_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    commitment_path = artifact / "prediction_commitment.json"
    result_path = artifact / "result.json"
    manifest_path = artifact / "manifest.json"
    qa_path = artifact / "independent_qa.json"
    require(not qa_path.exists(), "independent QA is append-only")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["experiment_identity"] = all(
        value["experiment_id"] == EXPERIMENT_ID for value in (config, commitment, result, manifest)
    )
    checks["commitment_pretruth"] = (
        commitment["truth_metric_computed"] is False
        and commitment["validation_truth_loaded"] is False
    )
    checks["strongest_comparator_selected"] = (
        commitment["comparator_selected_before_prediction"]["name"]
        == "p2_extrapolated_soft_gate_v2"
        and commitment["comparator_selected_before_prediction"]["priority_index"] == 0
    )
    checks["result_hash"] = sha256(result_path) == manifest["outputs"]["result"]["sha256"]
    checks["commitment_hash"] = (
        sha256(commitment_path) == manifest["outputs"]["prediction_commitment"]["sha256"]
    )
    checks["source_hashes"] = all(
        sha256(REPO / relative) == expected for relative, expected in manifest["sources"].items()
    )
    predictions = prediction_frame(commitment)
    truth = truth_frame(config, commitment["comparator_selected_before_prediction"])
    scored = predictions.merge(truth, on=["time", "layer", "block"], validate="one_to_one")
    checks["row_key_contract"] = len(scored) == len(predictions) == 69_850
    disabled = ~scored["enabled"].to_numpy(bool)
    checks["bit_exact_noop"] = np.array_equal(
        scored.loc[disabled, "reference"].to_numpy(),
        scored.loc[disabled, "candidate"].to_numpy(),
    )
    checks["correction_identity"] = np.array_equal(
        scored["candidate"].to_numpy() - scored["reference"].to_numpy(),
        scored["correction"].to_numpy(),
    )
    aggregate = metric(scored)
    folds = {str(name): metric(group) for name, group in scored.groupby("block", sort=True)}
    layers = {str(int(name)): metric(group) for name, group in scored.groupby("layer", sort=True)}
    bootstrap = paired_kst_day_bootstrap(
        scored,
        replicates=int(config["model"]["bootstrap_replicates"]),
        seed=int(config["model"]["bootstrap_seed"]),
    )
    correction = scored["correction"].to_numpy(np.float64)
    enabled = scored["enabled"].to_numpy(bool)
    correction_rms = float(np.sqrt(np.mean(correction**2)))
    correction_p99 = float(np.quantile(np.abs(correction), 0.99))
    correction_maximum = float(np.max(np.abs(correction)))
    gate = evaluate_gate(
        aggregate_delta=float(aggregate["delta_rmse_c"]),
        ci90_high=float(bootstrap["ci90_high_c"]),
        fold_deltas={name: float(value["delta_rmse_c"]) for name, value in folds.items()},
        layer_deltas={name: float(value["delta_rmse_c"]) for name, value in layers.items()},
        active_fraction=float(enabled.mean()),
        correction_rms=correction_rms,
        correction_p99=correction_p99,
        correction_maximum=correction_maximum,
        thresholds=config["gate"],
    )
    checks["aggregate_metrics"] = all(
        np.isclose(aggregate[key], result["metrics"]["aggregate"][key], atol=1e-15)
        for key in ("incumbent_rmse_c", "heave_candidate_rmse_c", "delta_rmse_c")
    )
    checks["fold_metrics"] = all(
        np.isclose(
            value["delta_rmse_c"], result["metrics"]["by_fold"][name]["delta_rmse_c"], atol=1e-15
        )
        for name, value in folds.items()
    )
    checks["layer_metrics"] = all(
        np.isclose(
            value["delta_rmse_c"], result["metrics"]["by_layer"][name]["delta_rmse_c"], atol=1e-15
        )
        for name, value in layers.items()
    )
    checks["bootstrap_reproduction"] = all(
        np.isclose(bootstrap[key], result["paired_kst_day_bootstrap"][key], atol=1e-15)
        for key in ("mean_delta_rmse_c", "ci90_low_c", "ci90_high_c", "probability_improved")
    )
    checks["gate_reproduction"] = gate == result["gate"]
    checks["no_candidate_csv"] = manifest["outputs"]["candidate_csv"] is None
    checks["leakage_flags"] = all(
        result["leakage_audit"][key] is False
        for key in (
            "official_input_paths_read",
            "official_answer_or_mirror_read",
            "candidate_csv_generated",
            "official_upload_performed",
            "target_temp_psal_used_as_features",
            "new_pava_applied",
            "post_result_parameter_search_performed",
        )
    )
    passed = bool(all(checks.values()))
    qa = {
        "schema_version": "p2.public_heave_tangent_incumbent.independent_qa.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "passed": passed,
        "checks": checks,
        "recomputed": {
            "rows": int(len(scored)),
            "metrics": {"aggregate": aggregate, "by_fold": folds, "by_layer": layers},
            "paired_kst_day_bootstrap": bootstrap,
            "correction": {
                "enabled_rows": int(enabled.sum()),
                "enabled_fraction": float(enabled.mean()),
                "rms_c": correction_rms,
                "p99_absolute_c": correction_p99,
                "maximum_absolute_c": correction_maximum,
            },
            "gate": gate,
        },
        "candidate_csv_generated": False,
        "official_upload_performed": False,
    }
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    require(
        passed, f"independent QA failed: {[name for name, value in checks.items() if not value]}"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
