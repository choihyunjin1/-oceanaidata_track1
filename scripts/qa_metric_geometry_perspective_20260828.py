"""Independent aggregate-only QA for the metric-geometry research artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = REPO / "reports" / "metric_geometry_perspective_deep_research_20260828_v12"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    report_dir = args.report_dir.expanduser().resolve()
    result_path = report_dir / "metric_geometry.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    official_path = REPO / "reports" / "p2_oas_alpha50_deployment_20260828_v13" / "official_score_receipt.json"
    official = json.loads(official_path.read_text(encoding="utf-8"))
    selected = result["decision"]["selected_next_probe"]
    checks = {
        "research_only_status": result["status"] == "RESEARCH_ONLY_NO_SUBMISSION_CREATED",
        "target_labels_not_read": result["method"]["target_labels_read"] is False,
        "candidate_csv_not_created": result["decision"]["candidate_csv_created"] is False,
        "upload_not_performed": result["decision"]["upload_performed"] is False,
        "report_directory_contains_no_csv": not any(report_dir.glob("*.csv")),
        "row_contract_26061": result["input_contract"]["rows"] == 26061,
        "lineage_reproduced": max(result["lineage_reproduction_max_abs"].values()) <= 1e-12,
        "selected_alpha_is_050": abs(float(selected["alpha"]) - 0.5) <= 1e-12,
        "robust_upper_beats_alpha40": float(selected["rmse_upper"]) < 0.445147,
        "minimum_improvement_gate": float(selected["guaranteed_improvement_vs_alpha40"]) >= 0.003,
        "single_probe_cap": result["decision"]["max_official_probes_before_reassessment"] == 1,
        "official_score_confirmed": official["status"] == "OFFICIAL_PUBLIC_SCORE_CONFIRMED_IMPROVEMENT",
        "official_result_inside_preregistered_interval": official["preregistered_geometry"]["observed_inside_interval"] is True,
        "official_rmse_improved": official["delta"]["rmse_improvement"] > 0,
        "automatic_followup_disabled": official["followup_gate"]["automatic_second_probe_allowed"] is False,
    }
    qa = {
        "schema_version": "metric_geometry_perspective.independent_qa.20260828.v2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metric_geometry_sha256": sha256(result_path),
        "official_score_receipt_sha256": sha256(official_path),
        "report_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(report_dir.glob("*.md"))
        },
        "answer_file_read": False,
        "submission_created_or_uploaded": True,
        "official_public_rmse": official["official_result"]["public_rmse"],
        "official_points": official["official_result"]["points"],
    }
    output = report_dir / "independent_qa.json"
    output.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    if qa["status"] != "PASS":
        raise SystemExit(json.dumps(qa, ensure_ascii=False, indent=2))
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
