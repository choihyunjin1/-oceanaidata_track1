"""Append-only recovery QA for a 1.69e-15 C subtraction-roundoff mismatch."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from qa_p2_public_heave_tangent_incumbent_20260828_v1 import (
    EXPERIMENT_ID,
    REPO,
    metric,
    prediction_frame,
    sha256,
    truth_frame,
)

from p2_restore.p2_public_heave_tangent_incumbent_20260828_v1 import (
    evaluate_gate,
    paired_kst_day_bootstrap,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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
    output_path = artifact / "independent_qa_recovery1.json"
    recovery_manifest_path = artifact / "qa_recovery_manifest.json"
    require(not output_path.exists(), "recovery QA is append-only")
    require(not recovery_manifest_path.exists(), "recovery manifest is append-only")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    commitment_path = artifact / "prediction_commitment.json"
    result_path = artifact / "result.json"
    manifest_path = artifact / "manifest.json"
    failed_qa_path = artifact / "independent_qa.json"
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failed_qa = json.loads(failed_qa_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["original_qa_failure_is_narrow"] = [
        name for name, passed in failed_qa["checks"].items() if not passed
    ] == ["correction_identity"]
    checks["original_manifest_sources_unchanged"] = all(
        sha256(REPO / relative) == expected for relative, expected in manifest["sources"].items()
    )
    checks["sealed_outputs_unchanged"] = (
        sha256(commitment_path) == manifest["outputs"]["prediction_commitment"]["sha256"]
        and sha256(result_path) == manifest["outputs"]["result"]["sha256"]
    )
    predictions = prediction_frame(commitment)
    truth = truth_frame(config, commitment["comparator_selected_before_prediction"])
    scored = predictions.merge(truth, on=["time", "layer", "block"], validate="one_to_one")
    difference = (scored["candidate"] - scored["reference"] - scored["correction"]).to_numpy(
        np.float64
    )
    maximum_identity_roundoff = float(np.max(np.abs(difference)))
    checks["correction_identity_with_machine_roundoff"] = bool(
        np.allclose(
            scored["candidate"] - scored["reference"],
            scored["correction"],
            rtol=0.0,
            atol=2e-15,
        )
    )
    disabled = ~scored["enabled"].to_numpy(bool)
    checks["bit_exact_incumbent_noop"] = np.array_equal(
        scored.loc[disabled, "candidate"].to_numpy(),
        scored.loc[disabled, "reference"].to_numpy(),
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
    checks["aggregate_reproduction"] = all(
        np.isclose(aggregate[key], result["metrics"]["aggregate"][key], atol=1e-15)
        for key in ("incumbent_rmse_c", "heave_candidate_rmse_c", "delta_rmse_c")
    )
    checks["bootstrap_reproduction"] = all(
        np.isclose(bootstrap[key], result["paired_kst_day_bootstrap"][key], atol=1e-15)
        for key in (
            "mean_delta_rmse_c",
            "ci90_low_c",
            "ci90_high_c",
            "probability_improved",
        )
    )
    checks["gate_reproduction"] = gate == result["gate"]
    checks["no_candidate_csv"] = manifest["outputs"]["candidate_csv"] is None
    passed = bool(all(checks.values()))
    qa = {
        "schema_version": "p2.public_heave_tangent_incumbent.independent_qa_recovery.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "passed": passed,
        "recovery_scope": "Only the correction identity check changed from bit equality to rtol=0, atol=2e-15. No prediction, metric, threshold, gate, or lineage changed.",
        "maximum_correction_identity_roundoff_c": maximum_identity_roundoff,
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
    output_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    recovery_manifest = {
        "schema_version": "p2.public_heave_tangent_incumbent.qa_recovery_manifest.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "original_manifest_sha256": sha256(manifest_path),
        "prediction_commitment_sha256": sha256(commitment_path),
        "result_sha256": sha256(result_path),
        "failed_independent_qa_sha256": sha256(failed_qa_path),
        "recovery_qa_script": {
            "path": str(Path(__file__).resolve().relative_to(REPO)).replace("\\", "/"),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "recovery_qa_output": {
            "path": str(output_path.relative_to(REPO)).replace("\\", "/"),
            "sha256": sha256(output_path),
        },
        "passed": passed,
    }
    recovery_manifest_path.write_text(
        json.dumps(recovery_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    require(passed, f"recovery QA failed: {[name for name, value in checks.items() if not value]}")
    print(
        json.dumps({"qa": qa, "recovery_manifest": recovery_manifest}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
