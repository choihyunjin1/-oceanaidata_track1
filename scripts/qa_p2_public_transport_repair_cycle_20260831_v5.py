"""Independent artifact and gate QA for P2 v5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = "p2_public_transport_repair_cycle_20260831_v5"
ART = ROOT / "artifacts" / EXP
REP = ROOT / "reports" / EXP


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    result = json.loads((ART / "result.json").read_text(encoding="utf-8"))
    checks = {
        "terminal_complete_no_pass": result["status"] == "COMPLETE_NO_PASS",
        "two_preregistered_candidates": result["candidate_count"] == 2,
        "fit_count_12": result["fit_count"] == 12,
        "prediction_hashes_match": all(
            sha256_file(ART / f"{candidate['name']}.npz") == candidate["prediction_sha256"]
            for candidate in result["candidates"]
        ),
        "calibration_penalty_exact": all(
            candidate["transport_penalty_points"] == 0.12168209161000616
            for candidate in result["candidates"]
        ),
        "inclusive_gate_recomputed": all(
            candidate["gate_checks"]["calibrated_expected_points_gte_0_01"]
            == (candidate["calibrated_expected_points_delta"] >= 0.01)
            for candidate in result["candidates"]
        ),
        "no_candidate_passed": not any(candidate["pass"] for candidate in result["candidates"]),
        "official_test_index_rows_zero": result["official_test_index_rows_read"] == 0,
        "hidden_truth_rows_zero": result["hidden_truth_rows_read"] == 0,
        "submission_and_upload_zero": result["submission_count"] == result["upload_count"] == 0,
        "technical_recovery_chain_preserved": all(
            (ART / f"technical_recovery{suffix}.json").exists() for suffix in ("", "_2")
        ),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "runner_sha256": result["runner_sha256"],
        "result_sha256": sha256_file(ART / "result.json"),
    }
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "independent-qa.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
