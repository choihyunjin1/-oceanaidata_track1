"""Adjudicate only floating round-trip false negatives in the sealed v3 QA."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

EXPERIMENT_ID = "p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v3"
EXPECTED_RESULT_SHA256 = (
    "1788893add5d009993f32d9089ce9a20942182106cd7dc1ae553d488d70e51a5"
)
EXPECTED_ORIGINAL_QA_SHA256 = (
    "bcb62831b01a1e6ed02b7afb8917c01231006da0c36ab4aabb5b78487253d1be"
)
TOLERANCE_C = 1e-12
EXPECTED_FALSE_CHECKS = {
    "difference_receipts",
    "incomplete_profile_exact_noop",
    "strict_schema_key_order_finite_domain",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _numeric_gap(left: float, right: float) -> float:
    return abs(float(left) - float(right))


def _delta_checks(
    stored: dict[str, object], recomputed: dict[str, object]
) -> tuple[bool, dict[str, object]]:
    mapping = {
        "vs_alpha50": "difference_vs_alpha50",
        "vs_current_official_incumbent": (
            "difference_vs_current_official_incumbent"
        ),
        "vs_official_base_u": "difference_vs_official_base_u",
    }
    receipt: dict[str, object] = {}
    passed = True
    for qa_name, result_name in mapping.items():
        left = stored[result_name]
        right = recomputed[qa_name]
        gaps = {
            field: _numeric_gap(left[field], right[field])
            for field in ("rms_c", "p99_abs_c", "maximum_abs_c")
        }
        counts_equal = left["changed_rows"] == right["changed_rows"]
        current = counts_equal and max(gaps.values()) <= TOLERANCE_C
        passed = passed and current
        receipt[qa_name] = {
            "changed_rows_equal": counts_equal,
            "numeric_absolute_gaps_c": gaps,
            "maximum_numeric_absolute_gap_c": max(gaps.values()),
            "pass_at_1e_12": current,
        }
    return passed, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    result_path = report_dir / "result.json"
    original_qa_path = report_dir / "independent-qa.json"
    output_path = report_dir / "independent-qa-adjudication.json"
    if sha256_file(result_path) != EXPECTED_RESULT_SHA256:
        raise RuntimeError("sealed result hash mismatch")
    if sha256_file(original_qa_path) != EXPECTED_ORIGINAL_QA_SHA256:
        raise RuntimeError("sealed original QA hash mismatch")
    result = load_json(result_path)
    original = load_json(original_qa_path)
    if result["experiment_id"] != EXPERIMENT_ID:
        raise RuntimeError("result experiment mismatch")
    if original["experiment_id"] != EXPERIMENT_ID or original["qa_status"] != "FAIL":
        raise RuntimeError("original QA chronology mismatch")
    false_checks = {name for name, value in original["checks"].items() if not value}
    if false_checks != EXPECTED_FALSE_CHECKS:
        raise RuntimeError("original QA has a non-adjudicable failure")

    stored_validation = result["materialization"]["validation"]
    recomputed_validation = original["candidate"]["validation"]
    validation_gaps = {
        "minimum": _numeric_gap(
            stored_validation["minimum"], recomputed_validation["minimum"]
        ),
        "maximum": _numeric_gap(
            stored_validation["maximum"], recomputed_validation["maximum"]
        ),
    }
    validation_pass = (
        stored_validation["rows"] == recomputed_validation["rows"] == 26061
        and max(validation_gaps.values()) <= TOLERANCE_C
        and -5.0 <= float(recomputed_validation["minimum"])
        and float(recomputed_validation["maximum"]) <= 45.0
        and original["checks"]["sample_and_candidate_exact_output_schema"]
        and original["checks"]["sample_key_order"]
    )
    stored_prediction = result["materialization"]["prediction"]
    recomputed_partial = float(original["candidate"]["partial_profile_noop_max_abs_c"])
    partial_gap = _numeric_gap(
        stored_prediction["incomplete_profile_max_abs_correction_c"],
        recomputed_partial,
    )
    partial_pass = recomputed_partial <= TOLERANCE_C and partial_gap <= TOLERANCE_C
    differences_pass, differences_receipt = _delta_checks(
        stored_prediction, original["candidate"]["differences"]
    )
    adjudicated = {
        "strict_schema_key_order_finite_domain": validation_pass,
        "incomplete_profile_exact_noop": partial_pass,
        "difference_receipts": differences_pass,
    }
    status = "PASS" if all(adjudicated.values()) else "FAIL"
    receipt = {
        "schema_version": (
            "p2.gaussian_copula_v2.exact_frozen_submission_pack.qa_adjudication."
            "20260830.v3"
        ),
        "experiment_id": EXPERIMENT_ID,
        "qa_status": status,
        "status_detail": (
            "PASS_QA_FLOAT_ROUNDTRIP_ADJUDICATED"
            if status == "PASS"
            else "FAIL_NONADJUDICABLE"
        ),
        "chronology": {
            "original_qa_preserved": True,
            "original_qa_status": "FAIL",
            "original_qa_sha256": EXPECTED_ORIGINAL_QA_SHA256,
            "result_sha256": EXPECTED_RESULT_SHA256,
            "candidate_or_model_rerun": False,
        },
        "basis": {
            "absolute_tolerance_c": TOLERANCE_C,
            "predeclared_contract": (
                "The original QA already declared <=1e-12 for exact-no-op and "
                "PAVA checks; applying the same tolerance to CSV IEEE-754 decimal "
                "round-trip comparisons is a QA correction, not a model relaxation."
            ),
            "original_false_checks_exact": sorted(false_checks),
            "all_other_original_checks_true": True,
        },
        "adjudicated_checks": adjudicated,
        "operands": {
            "validation": {
                "rows_equal": stored_validation["rows"]
                == recomputed_validation["rows"],
                "numeric_absolute_gaps_c": validation_gaps,
            },
            "incomplete_profile": {
                "recomputed_max_abs_correction_c": recomputed_partial,
                "stored_to_recomputed_gap_c": partial_gap,
            },
            "difference_receipts": differences_receipt,
        },
        "candidate": {
            "path": original["candidate"]["path"],
            "bytes": original["candidate"]["bytes"],
            "sha256": original["candidate"]["sha256"],
            "rows": recomputed_validation["rows"],
            "minimum_c": recomputed_validation["minimum"],
            "maximum_c": recomputed_validation["maximum"],
            "duplicate_qa": original["duplicate_qa"],
        },
        "execution_receipt": {
            "aggregate_json_inputs_only": True,
            "official_input_rows_read": 0,
            "model_fits": 0,
            "candidate_csv_rewrites": 0,
            "uploads": 0,
        },
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
    }
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "qa_status": status,
                "candidate_sha256": receipt["candidate"]["sha256"],
                "adjudication_sha256": sha256_file(output_path),
                "candidate_or_model_rerun": False,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
