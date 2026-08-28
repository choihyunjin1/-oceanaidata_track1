from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports" / "p2_submit_p1_p3_deep_research_20260828_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the frozen P2 candidate and research evidence.")
    parser.add_argument(
        "--submission",
        type=Path,
        required=True,
        help="Path to the frozen P2 submission CSV; the file is read-only.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    submission = parse_args().submission.expanduser().resolve()
    with submission.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)

    receipt = load_json(REPORT / "official_score_receipt.json")
    p1_support = load_json(
        REPO / "artifacts" / "p1_frozen_direct_event_verifier_blocked_20260828_v2" / "aggregate_metrics.json"
    )
    p1_ncad = load_json(
        REPO / "artifacts" / "p1_ncad_synthetic_long_event_20260828_v1" / "result.json"
    )
    p3_nlinear = load_json(
        REPO / "artifacts" / "p3_nlinear_station_ridge_residual_20260828_v1" / "result.json"
    )
    p3_era5 = load_json(REPO / "reports" / "p3_era5_context_transfer_terminal_20260828.json")

    checks = {
        "p2_file_exists": submission.exists(),
        "p2_sha_matches_receipt": sha256(submission) == receipt["submission_sha256"],
        "p2_rows_26061": rows == 26061,
        "p2_schema_exact": header == ["station", "layer", "time", "temp"],
        "p2_official_rmse_recorded": receipt["official_public_rmse_c"] == 0.445147,
        "p2_official_points_recorded": receipt["official_score_points"] == 27.747847,
        "p2_improved_previous_best": receipt["delta_rmse_c"] < 0 and receipt["delta_score_points"] > 0,
        "p1_support_status_no_go": p1_support["status"] == "NO_GO_SUPPORT_PREFLIGHT",
        "p1_support_fit_zero": p1_support["execution"]["model_fit_count"] == 0,
        "p1_support_counts_exact": (
            p1_support["support"]["train"]["utility_positive"] == 2
            and p1_support["support"]["calibration"]["utility_positive"] == 0
        ),
        "p1_ncad_status_no_go": p1_ncad["status"] == "NO_GO_CALIBRATION_SAFETY",
        "p1_ncad_official_surface_untouched": p1_ncad["access"]["official_test_rows_read"] == 0,
        "p1_ncad_checkpoint_hash_matches": sha256(
            REPO / "artifacts" / "p1_ncad_synthetic_long_event_20260828_v1" / "best_checkpoint.pt"
        ) == p1_ncad["hashes"]["checkpoint_sha256"],
        "p1_ncad_prediction_hash_matches": sha256(
            REPO / "artifacts" / "p1_ncad_synthetic_long_event_20260828_v1" / "sealed_split_predictions.npz"
        ) == p1_ncad["hashes"]["prediction_sha256"],
        "p3_nlinear_status_no_go": p3_nlinear["status"] == "TERMINAL_NO_GO",
        "p3_nlinear_official_surface_untouched": p3_nlinear["access"]["official_test_rows_read"] == 0,
        "p3_nlinear_sealed_hash_matches": sha256(
            REPO / "artifacts" / "p3_nlinear_station_ridge_residual_20260828_v1" / "sealed_outer_predictions.parquet"
        ) == p3_nlinear["hashes"]["sealed_prediction_sha256"],
        "p3_nlinear_evaluated_hash_matches": sha256(
            REPO / "artifacts" / "p3_nlinear_station_ridge_residual_20260828_v1" / "evaluated_outer_predictions.parquet"
        ) == p3_nlinear["hashes"]["evaluated_prediction_sha256"],
        "p3_era5_download_complete": p3_era5["download"]["raw_complete"] == 363,
        "p3_era5_preflight_passed": p3_era5["preflight"]["passed"] is True,
        "p3_era5_fit_zero_dependency_failure": (
            p3_era5["execution"]["model_fits_completed"] == 0
            and p3_era5["execution"]["failed_import"] == "catboost.CatBoostRegressor"
        ),
    }
    output = {
        "schema_version": "p2_submit_p1_p3_deep_research.independent_qa.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "summary": {
            "passed": sum(checks.values()),
            "total": len(checks),
            "p2_sha256": sha256(submission),
            "p2_rows": rows,
            "p2_header": header,
        },
    }
    (REPORT / "independent_qa.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if output["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
