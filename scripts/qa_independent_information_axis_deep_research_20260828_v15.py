"""Independent QA for the P1/P2/P3 Deep Research v15 one-shot cycle."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/independent_information_axis_deep_research_20260828_v15"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    artifacts = {
        "p1": ROOT / "artifacts/p1_temporally_fused_rpca_offset_drift_anchor_union_20260828_v1",
        "p2": ROOT / "artifacts/p2_alpha50_supervised_rank1_functional_residual_20260828_v1",
        "p3": ROOT / "artifacts/p3_past_only_wind_wave_memory_regime_increment_20260828_v1",
    }
    p1 = read_json(artifacts["p1"] / "result.json")
    p2 = read_json(artifacts["p2"] / "result.json")
    p3 = read_json(artifacts["p3"] / "result.json")
    p1_predictions = pd.read_parquet(artifacts["p1"] / "sealed_candidate_predictions.parquet")
    p2_commitment = read_json(artifacts["p2"] / "prediction_commitment.json")
    p3_seal = read_json(artifacts["p3"] / "shadow_prediction_seal.json")
    p3_predictions = pd.read_parquet(artifacts["p3"] / "shadow_predictions_sealed.parquet")

    p2_prediction_hashes = {
        fold: sha256_file(ROOT / record["path"])
        for fold, record in p2_commitment["outputs"].items()
    }
    checks = {
        "p1_terminal_no_go": p1["decision"] == "NO_GO_EXACT_NO_OUTPUT",
        "p1_exact_anchor_no_output": bool(
            np.array_equal(
                p1_predictions["candidate_prediction"].to_numpy(dtype=np.int8),
                p1_predictions["current_router_prediction"].to_numpy(dtype=np.int8),
            )
        ),
        "p1_zero_added_rows": int(p1["added_rows"]) == 0,
        "p1_truth_late": int(p1["commitment"]["truth_rows_read_before_commitment"]) == 0,
        "p2_terminal_no_go": p2["decision"] == "NO_GO_EXACT_NO_OUTPUT",
        "p2_comparator_disclosed": p2["comparator"] == "INCUMBENT_PROXY_VALIDATION",
        "p2_prediction_hashes_match": all(
            p2_prediction_hashes[fold] == record["sha256"]
            for fold, record in p2_commitment["outputs"].items()
        ),
        "p2_truth_late": (
            p2_commitment["truth_metric_computed"] is False
            and p2_commitment["validation_truth_column_loaded"] is False
        ),
        "p3_terminal_no_go": p3["decision"] == "NO_GO_SHADOW_GATE",
        "p3_shadow_hash_matches": p3_seal["prediction_sha256"]
        == hashlib.sha256(
            p3_predictions["candidate_prediction"].to_numpy(dtype=np.float64).tobytes()
        ).hexdigest(),
        "p3_outer_not_opened": not (artifacts["p3"] / "outer_predictions_sealed.parquet").exists(),
        "p3_outer_truth_not_read": int(p3["outer_truth_rows_read"]) == 0,
        "official_rows_zero_all": all(
            int(result["official_test_sample_submission_rows_read"]) == 0
            for result in (p1, p2, p3)
        ),
        "submission_generation_zero_all": all(
            result["submission_generated_or_uploaded"] is False for result in (p1, p2, p3)
        ),
        "no_csv_in_cycle_artifacts": not any(
            path.suffix.lower() == ".csv"
            for directory in artifacts.values()
            for path in directory.rglob("*")
            if path.is_file()
        ),
    }
    inventory = {
        str(path.relative_to(ROOT)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for directory in artifacts.values()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    result = {
        "schema_version": "deep_research.independent_qa.20260828.v15",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "summary": {
            "p1_decision": p1["decision"],
            "p2_decision": p2["decision"],
            "p3_decision": p3["decision"],
            "official_uploads": 0,
        },
        "artifact_inventory": inventory,
    }
    atomic_json(REPORT_DIR / "independent-qa.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
