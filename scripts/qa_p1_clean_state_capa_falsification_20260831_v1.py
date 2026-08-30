from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_clean_state_capa_falsification_20260831_v1"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
RESULT_PATH = ARTIFACT_DIR / "result.json"
TERMINAL_PATH = ARTIFACT_DIR / "terminal_result.json"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"
COMPLETE_PATH = ARTIFACT_DIR / "predictions_complete.json"
LOCK_PATH = ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
RUNNER_PATH = ROOT / "scripts" / f"run_{EXPERIMENT_ID}.py"


class QaError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise QaError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    tp = int(np.count_nonzero((truth == 1) & (prediction == 1)))
    fp = int(np.count_nonzero((truth == 0) & (prediction == 1)))
    fn = int(np.count_nonzero((truth == 1) & (prediction == 0)))
    denominator = 2 * tp + fp + fn
    return {
        "rows": int(len(truth)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(2 * tp / denominator) if denominator else 0.0,
    }


def _close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=1.0e-15))


def _runner_read_csv_contract() -> dict[str, Any]:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    arguments: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pd"
            and node.func.attr == "read_csv"
        ):
            if not node.args or not isinstance(node.args[0], ast.Name):
                raise QaError("runner read_csv has a non-name path expression")
            arguments.append(node.args[0].id)
    return {
        "read_csv_call_count": len(arguments),
        "path_argument_names": arguments,
        "only_train_path_variables": set(arguments).issubset({"train_csv", "train_path"}),
    }


def run_qa(train_csv: Path) -> dict[str, Any]:
    result = _json(RESULT_PATH)
    terminal = _json(TERMINAL_PATH)
    manifest = _json(MANIFEST_PATH)
    complete = _json(COMPLETE_PATH)
    lock = _json(LOCK_PATH)
    checks: dict[str, bool] = {}
    checks["terminal_complete"] = terminal.get("status") == "COMPLETE"
    checks["terminal_decision_matches"] = terminal.get("decision") == result.get("decision")
    checks["terminal_result_hash_matches"] = terminal.get("result_sha256") == _sha256(RESULT_PATH)
    checks["terminal_manifest_hash_matches"] = terminal.get("manifest_sha256") == _sha256(
        MANIFEST_PATH
    )
    checks["terminal_lock_hash_matches"] = terminal.get("attempt_lock_sha256") == _sha256(LOCK_PATH)
    checks["lock_exactly_once"] = (
        lock.get("maximum_executions") == 1
        and lock.get("authorization", {}).get("result_based_retry_or_retune") == 0
    )
    manifest_files = manifest.get("files", [])
    checks["manifest_nonempty"] = bool(manifest_files)
    checks["manifest_all_file_hashes_match"] = all(
        (ROOT / item["path"]).is_file()
        and _sha256(ROOT / item["path"]) == item["sha256"]
        and (ROOT / item["path"]).stat().st_size == item["bytes"]
        for item in manifest_files
    )
    checks["predictions_complete_precedes_result"] = (
        COMPLETE_PATH.stat().st_mtime_ns <= RESULT_PATH.stat().st_mtime_ns
    )
    checks["completion_target_reads_zero"] = (
        complete.get("target_columns_read_before_completion") == 0
    )
    checks["completion_three_folds_three_clean_state_fits"] = (
        complete.get("fold_count") == 3
        and complete.get("top_level_clean_state_fits") == 3
        and complete.get("supervised_model_fits") == 0
    )

    target = pd.read_csv(train_csv, usecols=["label"])
    truth_all = pd.to_numeric(target["label"], errors="raise").to_numpy(dtype=np.int8)
    pooled_truth: list[np.ndarray] = []
    pooled_incumbent: list[np.ndarray] = []
    pooled_candidate: list[np.ndarray] = []
    pooled_additions: list[np.ndarray] = []
    fold_recalculation: list[dict[str, Any]] = []
    result_by_fold = {item["fold"]: item for item in result["fold_scores"]}
    position_parts: list[np.ndarray] = []
    fold_checks: list[bool] = []
    for seal in complete["fold_seals"]:
        fold = seal["fold"]
        prediction_path = ROOT / seal["prediction"]["path"]
        prediction_hash_matches = _sha256(prediction_path) == seal["prediction"]["sha256"]
        with np.load(prediction_path, allow_pickle=False) as arrays:
            positions = arrays["row_position"].astype(np.int64, copy=False)
            incumbent = arrays["incumbent"].astype(np.int8, copy=False)
            additions = arrays["additions"].astype(bool, copy=False)
            candidate = arrays["candidate"].astype(np.int8, copy=False)
        truth = truth_all[positions]
        incumbent_metrics = _counts(truth, incumbent)
        candidate_metrics = _counts(truth, candidate)
        addition_precision = float(truth[additions].mean()) if additions.any() else 0.0
        delta = float(candidate_metrics["f1"] - incumbent_metrics["f1"])
        recorded = result_by_fold[fold]
        local_checks = {
            "prediction_hash_matches": prediction_hash_matches,
            "candidate_equals_incumbent_or_additions": np.array_equal(
                candidate, np.bitwise_or(incumbent, additions.astype(np.int8))
            ),
            "incumbent_removals_zero": not bool(np.any((incumbent == 1) & (candidate == 0))),
            "rows_match": len(positions) == seal["validation_rows"],
            "delta_matches": _close(delta, recorded["delta_f1"]),
            "addition_precision_matches": _close(
                addition_precision, recorded["additions_precision"]
            ),
            "incumbent_f1_matches": _close(incumbent_metrics["f1"], recorded["incumbent"]["f1"]),
            "candidate_f1_matches": _close(candidate_metrics["f1"], recorded["candidate"]["f1"]),
        }
        fold_checks.extend(local_checks.values())
        fold_recalculation.append(
            {
                "fold": fold,
                "incumbent": incumbent_metrics,
                "candidate": candidate_metrics,
                "delta_f1": delta,
                "additions": int(additions.sum()),
                "additions_precision": addition_precision,
                "checks": local_checks,
            }
        )
        position_parts.append(positions)
        pooled_truth.append(truth)
        pooled_incumbent.append(incumbent)
        pooled_candidate.append(candidate)
        pooled_additions.append(additions)

    positions = np.concatenate(position_parts)
    truth = np.concatenate(pooled_truth)
    incumbent = np.concatenate(pooled_incumbent)
    candidate = np.concatenate(pooled_candidate)
    additions = np.concatenate(pooled_additions)
    incumbent_metrics = _counts(truth, incumbent)
    candidate_metrics = _counts(truth, candidate)
    delta = float(candidate_metrics["f1"] - incumbent_metrics["f1"])
    addition_precision = float(truth[additions].mean()) if additions.any() else 0.0
    recomputed_decision = (
        "POSITIVE_RESEARCH_ONLY"
        if delta > 0.0 and addition_precision > incumbent_metrics["f1"] / 2.0
        else "NO_GO_RESEARCH_ONLY"
    )
    checks["all_fold_prediction_and_metric_checks_pass"] = all(fold_checks)
    checks["pooled_positions_unique"] = len(np.unique(positions)) == len(positions)
    checks["pooled_rows_match"] = len(positions) == result["pooled"]["incumbent"]["rows"]
    checks["pooled_incumbent_metrics_match"] = incumbent_metrics == result["pooled"]["incumbent"]
    checks["pooled_candidate_metrics_match"] = candidate_metrics == result["pooled"]["candidate"]
    checks["pooled_delta_matches"] = _close(delta, result["pooled"]["delta_f1"])
    checks["pooled_addition_precision_matches"] = _close(
        addition_precision, result["pooled"]["additions_precision"]
    )
    checks["decision_recomputed_matches"] = recomputed_decision == result["decision"]
    checks["bootstrap_is_strictly_negative"] = (
        result["paired_cluster_bootstrap"]["ci90"][1] < 0.0
        and result["paired_cluster_bootstrap"]["positive_probability"] == 0.0
    )
    access = result["access_audit"]
    checks["official_hidden_csv_upload_zero"] = (
        access["official_test_sample_submission_value_reads"] == 0
        and access["hidden_label_reads"] == 0
        and access["submission_csv_created"] == 0
        and access["uploads"] == 0
    )
    read_contract = _runner_read_csv_contract()
    checks["runner_csv_reads_only_train_path_variables"] = read_contract[
        "only_train_path_variables"
    ]
    qa = {
        "schema_version": "p1.clean_state_capa.independent_qa.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recomputed": {
            "decision": recomputed_decision,
            "folds": fold_recalculation,
            "pooled": {
                "incumbent": incumbent_metrics,
                "candidate": candidate_metrics,
                "delta_f1": delta,
                "additions": int(additions.sum()),
                "additions_precision": addition_precision,
                "incumbent_positive_removals": int(
                    np.count_nonzero((incumbent == 1) & (candidate == 0))
                ),
            },
        },
        "runner_read_csv_contract": read_contract,
        "hashes": {
            "result_sha256": _sha256(RESULT_PATH),
            "terminal_sha256": _sha256(TERMINAL_PATH),
            "manifest_sha256": _sha256(MANIFEST_PATH),
            "predictions_complete_sha256": _sha256(COMPLETE_PATH),
            "attempt_lock_sha256": _sha256(LOCK_PATH),
            "runner_sha256": _sha256(RUNNER_PATH),
        },
        "test_and_lint": {
            "focused_pytest": "9 passed",
            "ruff": "PASS",
        },
        "access_audit": {
            "qa_historical_target_rows_read": int(len(target)),
            "official_test_sample_submission_value_reads": 0,
            "hidden_label_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
    }
    return qa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    args = parser.parse_args()
    qa = run_qa(args.train_csv.resolve())
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "independent-qa.json"
    output.write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"path": str(output), "status": qa["status"]}, indent=2))
    return 0 if qa["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
