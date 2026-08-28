"""Independent cross-problem QA for the 2026-08-28 parallel research cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "parallel_deep_research_execution_20260828_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"JSON root is not an object: {path}")
    return value


def run_qa(root: Path = ROOT) -> dict[str, Any]:
    p1_dir = root / "artifacts" / "p1_conditional_real_event_donor_20260828_v1"
    p2_dir = root / "artifacts" / "p2_alpha40_quasiperiodic_gp_residual_20260828_v1"
    p3_dir = root / "artifacts" / "p3_era5_context_transfer_dependency_recovery_20260828_v2"
    p1 = read_json(p1_dir / "result.json")
    p1_preflight = read_json(p1_dir / "preflight.json")
    p1_qa = read_json(p1_dir / "qa.json")
    p2 = read_json(p2_dir / "result.json")
    p2_qa = read_json(p2_dir / "independent_qa.json")
    p3 = read_json(p3_dir / "result.json")
    ledger = read_json(REPORT_DIR / "claim-ledger.json")
    seal_path = p3_dir / "sealed_historical_blind_predictions.parquet"
    seal = pq.ParquetFile(seal_path)

    checks = {
        "p1_support_pass": p1_preflight["support_gate_pass"] is True,
        "p1_calibration_fail": p1["status"] == "NO_GO_CALIBRATION"
        and p1["calibration"]["gate_pass"] is False,
        "p1_no_outer_or_official_access": p1["q2_truth_rows_read"] == 0
        and p1["q2_evaluated"] is False,
        "p1_qa_pass": p1_qa["contract_pass"] is True,
        "p2_gate_fail": p2["decision"] == "FAIL_GATE_STOP_NO_CSV_NO_RESEARCH_LOOP"
        and p2["gate"]["passed"] is False,
        "p2_near_noop": p2["correction"]["enabled_rows"] == 75
        and p2["correction"]["enabled_fraction"] < 0.002,
        "p2_no_official_access": not any(
            [
                p2["leakage_audit"]["official_test_sample_submission_paths_read"],
                p2["leakage_audit"]["candidate_csv_generated"],
                p2["leakage_audit"]["official_upload_performed"],
            ]
        ),
        "p2_independent_qa_pass": p2_qa["passed"] is True,
        "p3_source_pass": p3["source_gate"]["passed"] is True,
        "p3_viewpoint_pass": p3["viewpoint_signal_gate"]["passed"] is True,
        "p3_solution_fail": p3["local_gate"]["passed"] is False
        and p3["promotion"] == "no-go",
        "p3_seal_rows": seal.metadata.num_rows == 1086,
        "p3_seal_has_no_truth": "target_hs" not in seal.schema_arrow.names,
        "p3_seal_hash": sha256(seal_path)
        == p3["blind_seal"]["sha256"]
        == "25accc81915e95bebcf4e69cd313b73520c36969b88521a186f5be214c4ba2a7",
        "claim_ledger_count": len(ledger["claims"]) == 10,
        "all_local_claim_hashes_match": all(
            sha256(root / item["evidence"]) == item["sha256"]
            for item in ledger["claims"]
            if "evidence" in item
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"independent QA failed: {failed}")
    return {
        "schema_version": "parallel_deep_research_execution.independent_qa.20260828.v3",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "checks": checks,
        "result_hashes": {
            "p1": sha256(p1_dir / "result.json"),
            "p2": sha256(p2_dir / "result.json"),
            "p3": sha256(p3_dir / "result.json"),
            "p3_blind_seal": sha256(seal_path),
        },
        "official_candidate_decisions": {"P1": "NO_GO", "P2": "NO_GO", "P3": "NO_GO"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "independent-qa.json")
    args = parser.parse_args()
    result = run_qa(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
