"""Independent aggregate-only QA for the frozen-83 P1 event-ranker run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

EXPERIMENT_ID = "p1_frozen_83_event_ranker_recall_guard_20260828_v1"
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def evaluate() -> dict[str, object]:
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((ARTIFACT_DIR / "result.json").read_text(encoding="utf-8"))
    qa = json.loads((ARTIFACT_DIR / "qa.json").read_text(encoding="utf-8"))
    preflight = json.loads((ARTIFACT_DIR / "preflight.json").read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "experiment_id": manifest["experiment_id"] == EXPERIMENT_ID
        and result["experiment_id"] == EXPERIMENT_ID,
        "manifest_prohibition_counters": manifest["q3_q4_rows_read"] == 0
        and manifest["official_test_sample_submission_rows_read"] == 0
        and manifest["submission_generated_or_uploaded"] is False,
        "result_prohibition_counters": result["q3_q4_rows_read"] == 0
        and result["official_test_sample_submission_rows_read"] == 0
        and result["submission_generated_or_uploaded"] is False,
        "model_fit_count": int(result["model_fit_count"]) in {0, 1},
        "runner_contract_pass": bool(qa["contract_pass"]),
        "anchor_rows_removed": int(qa["anchor_rows_removed"]) == 0,
        "preflight_status_consistent": bool(preflight["support_gate_pass"])
        == (result["status"] != "NO_GO_SUPPORT"),
    }
    for relative, expected in manifest["sources"].items():
        checks[f"source_hash::{relative}"] = sha256_file(ROOT / relative) == expected
    for name, record in manifest["immutable_inputs"].items():
        config = json.loads(
            (
                ROOT
                / "configs"
                / "experiments"
                / f"{EXPERIMENT_ID}.json"
            ).read_text(encoding="utf-8")
        )
        path = ROOT / config["immutable_inputs"][name]["path"]
        checks[f"input_hash::{name}"] = (
            path.stat().st_size == int(record["bytes"])
            and sha256_file(path) == record["sha256"]
        )
    for name, record in manifest["artifacts"].items():
        path = ARTIFACT_DIR / name
        checks[f"artifact_hash::{name}"] = (
            path.is_file()
            and path.stat().st_size == int(record["bytes"])
            and sha256_file(path) == record["sha256"]
        )
    if int(result["model_fit_count"]) == 0:
        checks["terminal_before_fit"] = result["status"] == "NO_GO_SUPPORT"
        checks["no_prediction_commitment_without_fit"] = not (
            ARTIFACT_DIR / "prediction_commitment.npz"
        ).exists()
        checks["q2_truth_closed_without_fit"] = result["q2_truth_rows_read"] == 0
    else:
        receipt_path = ARTIFACT_DIR / "prediction_commitment_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        checks["single_threshold"] = result["threshold_selection_count"] == 1
        checks["prediction_commitment_hash"] = sha256_file(
            ARTIFACT_DIR / receipt["prediction_path"]
        ) == receipt["prediction_sha256"]
        checks["commitment_before_q2_truth"] = receipt["q2_truth_rows_read_before_commitment"] == 0
    if result["status"] in {"NO_GO_SUPPORT", "NO_GO_QUALIFICATION"}:
        checks["failure_exact_no_op"] = bool(result["no_op"]["byte_equivalent"])
        checks["failure_q2_truth_closed"] = result["q2_truth_rows_read"] == 0
    passed = all(checks.values())
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "result_status": result["status"],
        "model_fit_count": result["model_fit_count"],
        "manifest_sha256": sha256_file(ARTIFACT_DIR / "manifest.json"),
        "qa_script_sha256": sha256_file(Path(__file__)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    record = evaluate()
    if args.execute:
        atomic_json(ARTIFACT_DIR / "independent_qa.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
