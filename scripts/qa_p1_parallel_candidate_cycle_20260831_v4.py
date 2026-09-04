"""Independently recompute the P1 v4 gate arithmetic without refitting models."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_full_internal_submission_cycle_20260831_v2 as prior_cycle  # noqa: E402

RESULT = ROOT / "artifacts/p1_parallel_candidate_cycle_20260831_v4/result.json"
OUTPUT = ROOT / "reports/p1_parallel_candidate_cycle_20260831_v4/independent-recompute.json"


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def main() -> int:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    frame, _ = prior_cycle.p1_frame()
    checks: dict[str, bool] = {}
    evidence: dict[str, Any] = {}
    pooled_mask = frame["fold"].isin(["2025_q3", "2025_q4"]).to_numpy()
    truth = frame["label_base"].to_numpy(np.int8)
    anchor = frame["e150_prediction"].to_numpy(np.int8)
    checks["historical_key_unique"] = not frame.duplicated(
        ["station", "year", "layer", "time", "fold"]
    ).any()
    checks["pooled_rows_287862"] = int(pooled_mask.sum()) == 287_862
    for candidate in result["candidates"]:
        fold_expected = {}
        pooled_tp_add = 0
        pooled_fp_add = 0
        for fold, record in candidate["by_fold"].items():
            mask = frame["fold"].eq(fold).to_numpy()
            tp = int(((anchor == 1) & (truth == 1) & mask).sum())
            fp = int(((anchor == 1) & (truth == 0) & mask).sum())
            fn = int(((anchor == 0) & (truth == 1) & mask).sum())
            expected_reference = f1_from_counts(tp, fp, fn)
            tp_add = int(record["true_positive_additions"])
            fp_add = int(record["false_positive_additions"])
            expected_candidate = f1_from_counts(
                tp + tp_add, fp + fp_add, fn - tp_add
            )
            prefix = f"{candidate['name']}_{fold}"
            checks[f"{prefix}_reference_f1"] = bool(
                np.isclose(expected_reference, record["reference_f1"], atol=1e-12)
            )
            checks[f"{prefix}_candidate_f1"] = bool(
                np.isclose(expected_candidate, record["candidate_f1"], atol=1e-12)
            )
            checks[f"{prefix}_addition_accounting"] = (
                tp_add + fp_add == record["additions"]
            )
            checks[f"{prefix}_anchor_removal_zero"] = record["anchor_removals"] == 0
            pooled_tp_add += tp_add
            pooled_fp_add += fp_add
            fold_expected[fold] = {
                "reference_f1": expected_reference,
                "candidate_f1": expected_candidate,
            }
        tp = int(((anchor == 1) & (truth == 1) & pooled_mask).sum())
        fp = int(((anchor == 1) & (truth == 0) & pooled_mask).sum())
        fn = int(((anchor == 0) & (truth == 1) & pooled_mask).sum())
        expected_reference = f1_from_counts(tp, fp, fn)
        expected_candidate = f1_from_counts(
            tp + pooled_tp_add, fp + pooled_fp_add, fn - pooled_tp_add
        )
        prefix = candidate["name"]
        checks[f"{prefix}_pooled_reference_f1"] = bool(
            np.isclose(expected_reference, candidate["reference_f1"], atol=1e-12)
        )
        checks[f"{prefix}_pooled_candidate_f1"] = bool(
            np.isclose(expected_candidate, candidate["candidate_f1"], atol=1e-12)
        )
        expected_pass = bool(
            expected_candidate > expected_reference
            and all(item["candidate_f1"] >= item["reference_f1"] for item in fold_expected.values())
            and candidate["additions"] > 0
        )
        checks[f"{prefix}_gate"] = expected_pass == candidate["strict_internal_pass"]
        evidence[prefix] = {
            "fold_expected": fold_expected,
            "pooled_reference_f1": expected_reference,
            "pooled_candidate_f1": expected_candidate,
            "expected_pass": expected_pass,
        }
    checks["no_failed_candidate_materialized"] = not result["outputs"]
    checks["official_reads_zero"] = (
        result["operations"]["official_covariate_reads_after_internal_scoring"] == 0
    )
    checks["hidden_truth_reads_zero"] = result["operations"]["hidden_truth_reads"] == 0
    checks["uploads_zero"] = result["operations"]["uploads"] == 0
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "evidence": evidence,
        "refits": 0,
        "official_reads": 0,
    }
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
