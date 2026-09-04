"""Independent cross-problem QA for the 2026-08-28 v4 research cycle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "parallel_deep_research_execution_20260828_v4"
OUTPUT = REPORT_DIR / "integrated-independent-qa.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p1_dir = ROOT / "artifacts" / "p1_frozen_83_event_ranker_recall_guard_20260828_v1"
    p2_dir = ROOT / "artifacts" / "p2_public_heave_tangent_incumbent_20260828_v1"
    p3_dir = ROOT / "artifacts" / "p3_era5_incumbent_safe_advantage_router_20260828_v1"
    p1 = load_json(p1_dir / "result.json")
    p2 = load_json(p2_dir / "result.json")
    p3 = load_json(p3_dir / "result.json")
    p1_qa = load_json(p1_dir / "independent_qa.json")
    p2_qa = load_json(p2_dir / "independent_qa_recovery1.json")
    p3_qa = load_json(p3_dir / "independent_qa.json")
    p3_schema = set(pq.read_schema(p3_dir / "sealed_outer_predictions.parquet").names)
    docx = REPORT_DIR / "P1_P2_P3_병렬_딥리서치_결론보고_v4_20260828.docx"
    a11y = load_json(REPORT_DIR / "docx-a11y-audit.json")
    rendered_pages = sorted((REPORT_DIR / "rendered_v4").glob("page-*.png"))

    expected_hashes = {
        "p1_result": (
            p1_dir / "result.json",
            "8afca6fc57c7bd98e99478de235533d3c69aa7a994844a71aa466f1c61ec9f4e",
        ),
        "p1_manifest": (
            p1_dir / "manifest.json",
            "e611a024eaa5989b4d6437405e9a3f58a1f29cce8025cb46974038c63dd28a57",
        ),
        "p2_result": (
            p2_dir / "result.json",
            "f9626d17833a01f0ae2095eb0eaf2a9c055a16659ac31bc002c589413af52400",
        ),
        "p2_manifest": (
            p2_dir / "manifest.json",
            "e2c15a9ac24c89b430d73d2d53af3accf187717ac7e341c8de92025c3aa443c8",
        ),
        "p3_result": (
            p3_dir / "result.json",
            "aa9b4931e479c10ee7540a2b688072da433c4353375fa43a673a63946045ec3f",
        ),
        "p3_manifest": (
            p3_dir / "manifest.json",
            "42097c750d34cd9b85e480d6a632333b80aa93f4af7673b683e31f50c5076d61",
        ),
        "p3_predictions": (
            p3_dir / "sealed_outer_predictions.parquet",
            "cf98f7b507008b2150b31cbb992c699ebf88fc04fd80334b8ad7b56c09614cf7",
        ),
    }
    checks: dict[str, bool] = {
        f"hash::{name}": sha256(path) == expected
        for name, (path, expected) in expected_hashes.items()
    }
    checks.update(
        {
            "p1_terminal_no_fit": p1["status"] == "NO_GO_SUPPORT"
            and p1["model_fit_count"] == 0
            and p1["q2_truth_rows_read"] == 0,
            "p1_noop": p1["no_op"]["byte_equivalent"] is True,
            "p1_qa": p1_qa["status"] == "PASS",
            "p2_failed_gate": p2["decision"] == "FAIL_GATE_STOP_NO_CSV_NO_RESEARCH_LOOP"
            and p2["gate"]["passed"] is False,
            "p2_metric": abs(p2["metrics"]["aggregate"]["delta_rmse_c"] - 1.124089425097452e-05)
            < 1e-15,
            "p2_recovery_qa": p2_qa["passed"] is True
            and p2_qa["maximum_correction_identity_roundoff_c"] < 2e-15,
            "p3_failed_i4": p3["status"] == "NO_GO_INNER_I4_GATE",
            "p3_zero_outer_intervention": p3["metrics"]["intervention_coverage"] == 0.0,
            "p3_qa": p3_qa["verdict"] == "PASS",
            "p3_seal_has_no_truth": not p3_schema.intersection(
                {"truth", "target", "y_true", "observed_hs"}
            ),
            "report_source_present": (REPORT_DIR / "report-source.md").is_file(),
            "claim_ledger_present": (REPORT_DIR / "claim-ledger.json").is_file(),
            "docx_present": docx.is_file() and docx.stat().st_size > 0,
            "docx_a11y_clean": a11y["counts"] == {"high": 0, "medium": 0, "low": 0},
            "docx_four_pages_rendered": len(rendered_pages) == 4
            and all(path.stat().st_size > 0 for path in rendered_pages),
        }
    )
    # Path.rglob returns iterators; verify CSV absence explicitly without touching official paths.
    checks["no_candidate_csv_in_artifacts"] = all(
        next(directory.rglob("*.csv"), None) is None for directory in (p1_dir, p2_dir, p3_dir)
    )
    receipt = {
        "schema_version": "parallel.deep_research.execution.independent_qa.20260828.v4",
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_hashes": {
            "report_source": sha256(REPORT_DIR / "report-source.md"),
            "claim_ledger": sha256(REPORT_DIR / "claim-ledger.json"),
            "qa_script": sha256(Path(__file__)),
            "docx": sha256(docx),
            "docx_a11y_audit": sha256(REPORT_DIR / "docx-a11y-audit.json"),
        },
        "official_paths_accessed": False,
        "candidate_csv_generated": False,
        "upload_count": 0,
    }
    OUTPUT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
