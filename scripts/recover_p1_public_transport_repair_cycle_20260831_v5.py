"""Recover the complete scientific v5 receipt without rerunning any fit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v5"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main() -> int:
    partial_path = ARTIFACT / "result.json"
    failure_path = ARTIFACT / "terminal_failure.json"
    raw = partial_path.read_text(encoding="utf-8")
    marker = '  "independent_qa": {'
    if marker not in raw:
        raise RuntimeError("expected terminal-only truncated independent_qa marker")
    recovered = json.loads(raw.split(marker, 1)[0].rstrip().rstrip(",") + "\n}\n")
    checks = {
        "failure_is_terminal_json_serialization_only": (
            json.loads(failure_path.read_text(encoding="utf-8"))["error_type"]
            == "TypeError"
        ),
        "three_candidates_recovered": len(recovered["candidates"]) == 3,
        "all_12_historical_fits_completed": recovered["historical_fit_count"] == 12,
        "pass_count_zero": recovered["pass_count"] == 0,
        "all_candidate_deltas_zero": all(
            item["delta_f1"] == 0.0 for item in recovered["candidates"]
        ),
        "all_calibrated_deltas_below_gate": all(
            item["calibrated_conservative_expected_points_delta"] < 0.01
            for item in recovered["candidates"]
        ),
        "outputs_zero": recovered["outputs"] == [],
        "official_covariate_reads_zero": recovered["operations"][
            "official_covariate_reads"
        ]
        == 0,
        "hidden_truth_reads_zero": recovered["operations"]["hidden_truth_reads"] == 0,
        "uploads_zero": recovered["operations"]["uploads"] == 0,
        "prior_tie_excluded": recovered["prior_tie_disposition"]
        == "EXCLUDED_FROM_PASS_COUNT",
    }
    qa = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    recovered["status"] = "SCIENTIFIC_NO_PASS_TECHNICAL_RECEIPT_RECOVERED"
    recovered["independent_qa"] = qa
    recovered["recovery"] = {
        "fits_rerun": 0,
        "official_rows_reread": 0,
        "source_partial_result_sha256": sha256_file(partial_path),
        "terminal_failure_sha256": sha256_file(failure_path),
        "reason": "final JSON writer encountered numpy.bool_; all result fields before independent_qa were complete",
    }
    write_json(REPORT / "recovered-result.json", recovered)
    write_json(REPORT / "recovery-receipt.json", recovered["recovery"])
    write_json(REPORT / "independent-qa-recovered.json", qa)
    print(json.dumps({"status": qa["status"], "pass_count": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
