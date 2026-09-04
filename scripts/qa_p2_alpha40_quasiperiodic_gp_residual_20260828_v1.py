"""Independent QA for the committed P2 alpha40 quasi-periodic residual pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from p2_restore.p2_alpha40_quasiperiodic_gp_residual_20260828_v1 import (
    paired_kst_day_bootstrap,
    rmse,
)

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_alpha40_quasiperiodic_gp_residual_20260828_v1"
DEFAULT_ARTIFACT = REPO / "artifacts" / EXPERIMENT_ID


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def metric(frame: pd.DataFrame) -> tuple[float, float, float]:
    reference = rmse(frame["truth"].to_numpy(), frame["reference"].to_numpy())
    candidate = rmse(frame["truth"].to_numpy(), frame["candidate"].to_numpy())
    return reference, candidate, candidate - reference


def decode_committed_time(values: np.ndarray) -> pd.DatetimeIndex:
    current = np.asarray(values, dtype=np.int64)
    unit = "ns" if int(np.max(np.abs(current))) >= 10**17 else "us"
    decoded = pd.DatetimeIndex(pd.to_datetime(current, unit=unit, utc=True))
    require(decoded.min().year >= 2024 and decoded.max().year <= 2025, "time unit invalid")
    return decoded


def main() -> None:
    args = parse_args()
    artifact = args.artifact_dir.expanduser().resolve()
    qa_path = artifact / "independent_qa.json"
    require(not qa_path.exists(), "independent QA already exists")
    config_path = REPO / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    commitment_path = artifact / "prediction_commitment.json"
    result_path = artifact / "result.json"
    manifest_path = artifact / "manifest.json"
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}

    checks["commitment_predates_truth_metric"] = (
        commitment["truth_metric_computed"] is False
        and commitment["validation_truth_column_loaded"] is False
        and result["prediction_commitment"]["verified_before_truth_load"] is True
    )
    checks["no_candidate_csv"] = not any(artifact.rglob("*.csv")) and manifest["outputs"]["candidate_csv"] is None
    checks["no_official_path_contract"] = (
        commitment["official_test_sample_submission_paths_read"] is False
        and result["leakage_audit"]["official_test_sample_submission_paths_read"] is False
    )
    checks["no_parameter_search"] = commitment["hyperparameter_search_count"] == 0

    source_hashes_ok = True
    for relative, expected in manifest["sources"].items():
        source_hashes_ok &= sha256(REPO / relative) == expected
    checks["source_hashes"] = bool(source_hashes_ok)

    anchor_path = REPO / config["validation_anchor"]["path"]
    prediction_hashes_ok = True
    frames: list[pd.DataFrame] = []
    for fold_name, output in commitment["prediction_outputs"].items():
        path = REPO / output["path"]
        prediction_hashes_ok &= sha256(path) == output["sha256"]
        with np.load(path, allow_pickle=False) as payload:
            frame = pd.DataFrame(
                {
                    "time": decode_committed_time(payload["time_ns"]),
                    "layer": payload["layer"].astype(int),
                    "reference": payload["reference"].astype(float),
                    "candidate": payload["candidate"].astype(float),
                    "correction": payload["correction"].astype(float),
                    "enabled": payload["enabled"].astype(bool),
                }
            )
        truth = pd.read_parquet(
            anchor_path,
            columns=["time", "layer", "truth", "block"],
            filters=[("block", "==", fold_name)],
        )
        truth["time"] = pd.to_datetime(truth["time"], utc=True)
        truth["layer"] = truth["layer"].astype(int)
        frame = frame.merge(truth[["time", "layer", "truth"]], on=["time", "layer"], validate="one_to_one")
        frame["fold"] = fold_name
        frames.append(frame)
    checks["prediction_hashes"] = bool(prediction_hashes_ok)
    scored = pd.concat(frames, ignore_index=True)
    reference, candidate, delta = metric(scored)
    recorded = result["metrics"]["aggregate"]
    checks["aggregate_metrics"] = bool(
        np.isclose(reference, recorded["alpha40_reference_rmse"], atol=1e-15)
        and np.isclose(candidate, recorded["quasiperiodic_gp_candidate_rmse"], atol=1e-15)
        and np.isclose(delta, recorded["delta_rmse"], atol=1e-15)
    )
    fold_metrics_ok = True
    for fold_name, group in scored.groupby("fold", sort=True):
        current = metric(group)
        expected = result["metrics"]["by_fold"][fold_name]
        fold_metrics_ok &= all(
            np.isclose(value, expected[key], atol=1e-15)
            for value, key in zip(
                current,
                ("alpha40_reference_rmse", "quasiperiodic_gp_candidate_rmse", "delta_rmse"),
                strict=True,
            )
        )
    checks["fold_metrics"] = bool(fold_metrics_ok)
    layer_metrics_ok = True
    for layer, group in scored.groupby("layer", sort=True):
        current = metric(group)
        expected = result["metrics"]["by_layer"][str(int(layer))]
        layer_metrics_ok &= all(
            np.isclose(value, expected[key], atol=1e-15)
            for value, key in zip(
                current,
                ("alpha40_reference_rmse", "quasiperiodic_gp_candidate_rmse", "delta_rmse"),
                strict=True,
            )
        )
    checks["layer_metrics"] = bool(layer_metrics_ok)
    correction = scored["correction"].to_numpy(float)
    enabled = scored["enabled"].to_numpy(bool)
    checks["exact_noop_fallback"] = bool(np.array_equal(correction[~enabled], np.zeros((~enabled).sum())))
    checks["correction_rms_cap"] = float(np.sqrt(np.mean(correction**2))) <= float(config["gate"]["maximum_correction_rms_c"]) + 1e-12
    checks["correction_p99_cap"] = float(np.quantile(np.abs(correction), 0.99)) <= float(config["gate"]["maximum_correction_p99_c"]) + 1e-12
    bootstrap = paired_kst_day_bootstrap(
        scored,
        replicates=int(config["model"]["bootstrap_replicates"]),
        seed=int(config["model"]["bootstrap_seed"]),
    )
    checks["bootstrap_reproduction"] = all(
        np.isclose(bootstrap[key], result["paired_kst_day_bootstrap"][key], atol=1e-15)
        for key in ("mean_delta_rmse", "ci90_low", "ci90_high", "probability_improved")
    )
    checks["manifest_result_hash"] = sha256(result_path) == manifest["outputs"]["result"]["sha256"]
    passed = bool(all(checks.values()))
    qa = {
        "schema_version": "p2.alpha40_quasiperiodic_gp_residual.independent_qa.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "passed": passed,
        "checks": checks,
        "recomputed": {
            "rows": int(len(scored)),
            "reference_rmse": reference,
            "candidate_rmse": candidate,
            "delta_rmse": delta,
            "bootstrap": bootstrap,
        },
        "official_upload_performed": False,
    }
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    require(passed, f"independent QA failed: {[name for name, value in checks.items() if not value]}")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
