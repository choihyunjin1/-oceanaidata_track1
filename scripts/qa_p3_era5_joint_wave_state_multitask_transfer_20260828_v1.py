"""Independent aggregate-only QA for the sealed P3 joint-state experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_era5_joint_wave_state_multitask_transfer_20260828_v1"
OUTPUT = ROOT / "artifacts" / EXPERIMENT_ID


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    result_path = OUTPUT / "result.json"
    manifest_path = OUTPUT / "manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = pd.read_parquet(OUTPUT / "source_predictions_sealed.parquet")
    protected = seal["lead_h"].isin((3, 6, 9, 12))
    checks = {
        "terminal_source_no_go": result["status"] == "NO_GO_SOURCE_GATE",
        "source_gate_failed": result["source_gate"]["passed"] is False,
        "fit_count_two": int(result["fit_count"]) == 2,
        "held_rows_2952": len(seal) == 492 * 6,
        "held_cases_492": seal["anchor_id"].nunique() == 492,
        "keys_unique": not seal.duplicated(["fold", "anchor_id", "station", "lead_h"]).any(),
        "protected_leads_bit_exact": seal.loc[
            protected, "base_prediction"
        ].equals(seal.loc[protected, "candidate_prediction"]),
        "no_truth_in_seal": "target_hs" not in seal.columns,
        "shadow_not_opened": not (OUTPUT / "shadow_predictions_sealed.parquet").exists(),
        "official_rows_zero": int(result["official_test_sample_submission_rows_read"]) == 0,
        "no_submission": result["submission_generated_or_uploaded"] is False,
        "manifest_result_hash": manifest["result_sha256"] == sha256_file(result_path),
        "seal_hash": manifest["artifacts"]["source_prediction_seal"]
        == sha256_file(OUTPUT / "source_predictions_sealed.parquet"),
    }
    payload = {
        "schema_version": "p3.joint_wave_state.qa.v1",
        "experiment_id": EXPERIMENT_ID,
        "passed": all(checks.values()),
        "checks": checks,
        "result_sha256": sha256_file(result_path),
        "manifest_sha256": sha256_file(manifest_path),
        "seal_sha256": sha256_file(OUTPUT / "source_predictions_sealed.parquet"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
