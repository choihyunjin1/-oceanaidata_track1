"""Independent aggregate-only QA for the sealed P2 thermocline pilot."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_oas40_target_weighted_nonlinear_thermocline_residual_20260828_v1"
ARTIFACT = REPO / "artifacts" / EXPERIMENT_ID


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    output = ARTIFACT / "independent_qa.json"
    require(not output.exists(), "independent QA already exists")
    commitment_path = ARTIFACT / "prediction_commitment.json"
    result_path = ARTIFACT / "result.json"
    manifest_path = ARTIFACT / "manifest.json"
    report_path = ARTIFACT / "report.md"
    for path in (commitment_path, result_path, manifest_path, report_path):
        require(path.is_file(), f"missing output: {path.name}")
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "experiment_id": commitment["experiment_id"] == result["experiment_id"] == manifest["experiment_id"] == EXPERIMENT_ID,
        "execution_count_one": commitment["execution_count"] == result["execution_count"] == 1,
        "prediction_commitment_hash": sha256(commitment_path) == result["prediction_commitment"]["sha256"],
        "result_hash": sha256(result_path) == manifest["outputs"]["result"]["sha256"],
        "report_hash": sha256(report_path) == manifest["outputs"]["report"]["sha256"],
        "no_candidate_csv_manifest": manifest["outputs"]["candidate_csv"] is None,
        "no_csv_in_artifact": not any(ARTIFACT.rglob("*.csv")),
        "official_paths_not_read": result["leakage_audit"]["official_test_index_sample_submission_paths_read"] is False,
        "hidden_values_not_used": result["leakage_audit"]["hidden_interval_values_used"] is False,
        "no_upload": result["leakage_audit"]["official_upload_performed"] is False,
        "search_zero": commitment["hyperparameter_search_count"] == 0,
        "three_folds": len(commitment["prediction_outputs"]) == 3,
    }
    total_rows = 0
    enabled_rows = 0
    fallback_max = 0.0
    for fold, receipt in commitment["prediction_outputs"].items():
        path = REPO / receipt["path"]
        checks[f"prediction_hash_{fold}"] = sha256(path) == receipt["sha256"]
        with np.load(path, allow_pickle=False) as payload:
            rows = len(payload["layer"])
            total_rows += rows
            enabled = payload["enabled"].astype(bool)
            correction = payload["correction"].astype(float)
            enabled_rows += int(enabled.sum())
            require(np.isfinite(payload["reference"]).all(), f"reference non-finite: {fold}")
            require(np.isfinite(payload["candidate"]).all(), f"candidate non-finite: {fold}")
            require(np.isfinite(correction).all(), f"correction non-finite: {fold}")
            if (~enabled).any():
                fallback_max = max(fallback_max, float(np.max(np.abs(correction[~enabled]))))
    checks["row_count_matches_result"] = total_rows == result["metrics"]["aggregate"]["rows"]
    checks["enabled_count_matches_result"] = enabled_rows == result["correction"]["enabled_rows"]
    checks["fallback_exact_no_op"] = fallback_max == result["correction"]["fallback_maximum_absolute_c"] == 0.0
    checks["gate_decision_consistent"] = bool(result["gate"]["passed"]) == result["decision"].startswith("ORTHOGONAL_PROBE_READY")
    require(all(checks.values()), f"independent QA failed: {[key for key, value in checks.items() if not value]}")
    qa = {
        "schema_version": "p2.oas40_target_weighted_nonlinear_thermocline_residual.independent_qa.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "passed": True,
        "checks": checks,
        "aggregate": {
            "rows": total_rows,
            "enabled_rows": enabled_rows,
            "fallback_maximum_absolute_c": fallback_max,
            "decision": result["decision"],
        },
        "verified_hashes": {
            "prediction_commitment": sha256(commitment_path),
            "result": sha256(result_path),
            "manifest": sha256(manifest_path),
            "report": sha256(report_path),
        },
    }
    output.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "decision": result["decision"], "qa": str(output.relative_to(REPO))}, indent=2))


if __name__ == "__main__":
    main()
