"""Independent aggregate-only QA for the frozen trial_18 Q3/Q4 confirmation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "p1_mstcn_sobol_trial18_frozen_confirmation_20260830_v3"
ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / f"run_{EXPERIMENT_ID}.py"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
ATTEMPT_LOCK = ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
QA_PATH = ROOT / "reports" / EXPERIMENT_ID / "independent-qa.json"


class QAError(RuntimeError):
    """Raised when the sealed result cannot be independently admitted."""


def _load_runner() -> Any:
    name = f"{EXPERIMENT_ID}_runner_for_qa"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise QAError("cannot load reviewed runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QAError(f"JSON object required: {path}")
    return value


def _binary_metrics(truth: Any, prediction: Any) -> dict[str, float | int]:
    import numpy as np

    y = np.asarray(truth, dtype=np.int8)
    pred = np.asarray(prediction, dtype=np.int8)
    if y.shape != pred.shape or not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise QAError("binary metric inputs are invalid")
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _same_metrics(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for name in ("tp", "fp", "fn"):
        if int(left[name]) != int(right[name]):
            return False
    for name in ("precision", "recall", "f1"):
        if not math.isclose(float(left[name]), float(right[name]), rel_tol=0.0, abs_tol=1.0e-15):
            return False
    return True


def qa(*, root: Path = ROOT, write: bool = True) -> dict[str, Any]:
    runner = _load_runner()
    config = runner._config(root=root)
    terminal_path = root / "artifacts" / EXPERIMENT_ID / "terminal_result.json"
    artifact_dir = terminal_path.parent
    attempt_lock = root / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
    if not terminal_path.is_file() or not attempt_lock.is_file():
        raise QAError("sealed terminal result and attempt lock are required")
    terminal = _load_json(terminal_path)
    lock = _load_json(attempt_lock)
    metrics_path = artifact_dir / "confirmatory_metrics.json"
    manifest_path = artifact_dir / "artifact_manifest.json"
    metrics = _load_json(metrics_path)
    manifest = _load_json(manifest_path)
    checks: dict[str, bool] = {}
    config_path = root / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
    runner_path = root / "scripts" / f"run_{EXPERIMENT_ID}.py"
    checks["experiment_identity"] = (
        terminal.get("experiment_id") == EXPERIMENT_ID
        and lock.get("experiment_id") == EXPERIMENT_ID
        and manifest.get("experiment_id") == EXPERIMENT_ID
    )
    checks["config_hash_consistent"] = (
        terminal.get("config_sha256") == runner._sha256(config_path) == lock.get("config_sha256")
    )
    checks["runner_hash_consistent"] = (
        terminal.get("runner_sha256") == runner._sha256(runner_path) == lock.get("runner_sha256")
    )
    checks["recipe_hash_consistent"] = (
        terminal.get("recipe_sha256")
        == lock.get("recipe_sha256")
        == __import__("hashlib").sha256(runner._json_bytes(config["frozen_recipe"])).hexdigest()
    )
    checks["attempt_lock_hash_consistent"] = terminal.get("attempt_lock_sha256") == runner._sha256(
        attempt_lock
    )
    checks["terminal_is_research_only_state"] = terminal.get("status") in {
        "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY",
        "EXPLORATORY_CHALLENGER_RESEARCH_ONLY",
        "PRIMARY_HARM_RESEARCH_ONLY",
        "INCONCLUSIVE_RESEARCH_ONLY",
    }
    checks["fit_count_exactly_6"] = terminal.get("fit_count") == 6
    checks["q2_fit_and_search_counts_zero"] = (
        terminal.get("q2_fits") == 0
        and terminal.get("q2_search_replayed") is False
        and terminal.get("threshold_search_replayed") is False
    )
    checks["selected_cell_exact"] = (
        terminal.get("selected_trial") == "trial_18"
        and terminal.get("threshold") == 0.8
        and terminal.get("epoch") == 150
        and terminal.get("seeds") == [20260827, 20260839, 20260863]
    )
    checks["manifest_hash_consistent"] = terminal.get("artifact_manifest") == {
        "path": "artifact_manifest.json",
        "bytes": manifest_path.stat().st_size,
        "sha256": runner._sha256(manifest_path),
    }
    manifest_ok = True
    for row in manifest.get("files", []):
        path = artifact_dir / row["path"]
        manifest_ok &= (
            path.is_file()
            and path.stat().st_size == row["bytes"]
            and runner._sha256(path) == row["sha256"]
        )
    checks["manifest_member_hashes_consistent"] = bool(manifest_ok)
    source_pins_ok = True
    try:
        runner._verify_source_pins(config, root=root)
        runner._verify_selection_lineage(config, root=root)
    except BaseException:
        source_pins_ok = False
    checks["source_and_q2_lineage_still_pinned"] = source_pins_ok
    config_sha256 = runner._sha256(config_path)
    recipe_sha256 = terminal["recipe_sha256"]
    receipts: dict[str, Any] = {}
    for phase in ("q3", "q4"):
        receipt_path = artifact_dir / f"{phase}_confirmatory_blind_receipt.json"
        receipt = _load_json(receipt_path)
        receipts[phase] = receipt
    checks["both_blind_receipts_precede_truth"] = (
        all(
            row.get("same_fold_holdout_truth_columns_opened_before_receipt") == 0
            for row in receipts.values()
        )
        and metrics.get("truth_metrics_opened_after_both_blind_seals") is True
        and bool(metrics.get("both_blind_phase_receipts_sealed_at_utc"))
    )
    fit_receipts = [row for phase in ("q3", "q4") for row in receipts[phase]["fit_receipts"]]
    checks["fit_receipts_exact"] = (
        len(fit_receipts) == 6
        and all(row.get("trial_id") == "trial_18" for row in fit_receipts)
        and [row.get("seed") for row in fit_receipts]
        == [20260827, 20260839, 20260863, 20260827, 20260839, 20260863]
        and all(row.get("epochs") == 150 for row in fit_receipts)
        and all(row.get("nonfinite_count_total") == 0 for row in fit_receipts)
        and all(row.get("checkpoint_persisted") is False for row in fit_receipts)
        and terminal.get("optimizer_steps") == sum(row["optimizer_steps"] for row in fit_receipts)
    )
    base = runner._load_base(root=root)
    source_config = base._canonical_config(
        root / "configs" / "experiments" / f"{runner.BASE_EXPERIMENT_ID}.json"
    )
    surfaces = base.load_blind_surfaces(source_config, root=root)
    controls = runner._load_controls(base, config, surfaces, root=root)
    candidates: dict[str, Any] = {}
    holds: dict[str, Any] = {}
    truths: dict[str, Any] = {}
    bootstrap_folds: list[tuple[Any, Any, Any, Any]] = []
    import numpy as np

    for phase in ("q3", "q4"):
        _encoder, _training, holdout, _split = base._prepare_phase_surfaces(
            surfaces, source_config, phase, root=root
        )
        fold = {"q3": "2025_q3", "q4": "2025_q4"}[phase]
        receipt_path = artifact_dir / f"{phase}_confirmatory_blind_receipt.json"
        runner._verify_blind_receipt(
            receipt_path,
            config_sha256=config_sha256,
            recipe_sha256=recipe_sha256,
            expected_key_sha256=surfaces.membership_sha256[fold],
        )
        with np.load(artifact_dir / receipts[phase]["score_path"], allow_pickle=False) as archive:
            candidates[phase] = archive["candidate"].astype(np.int8, copy=True)
        holds[phase] = holdout
    for phase in ("q3", "q4"):
        fold = {"q3": "2025_q3", "q4": "2025_q4"}[phase]
        truths[phase] = runner._load_truth(
            base,
            source_config,
            holds[phase],
            [
                artifact_dir / "q3_confirmatory_blind_receipt.json",
                artifact_dir / "q4_confirmatory_blind_receipt.json",
            ],
            fold=fold,
            config_sha256=config_sha256,
            recipe_sha256=recipe_sha256,
            root=root,
        )
    y_parts: list[Any] = []
    incumbent_parts: list[Any] = []
    candidate_parts: list[Any] = []
    anchor_removed = 0
    for phase in ("q3", "q4"):
        y = truths[phase]["label"].to_numpy(dtype=np.int8)
        incumbent = controls[phase]
        candidate = candidates[phase]
        y_parts.append(y)
        incumbent_parts.append(incumbent)
        candidate_parts.append(candidate)
        anchor_removed += int(np.sum((holds[phase].surface.anchor == 1) & (candidate == 0)))
        bootstrap_folds.append((holds[phase].surface.keys, y, incumbent, candidate))
    y_all = np.concatenate(y_parts)
    incumbent_all = np.concatenate(incumbent_parts)
    candidate_all = np.concatenate(candidate_parts)
    incumbent_metrics = _binary_metrics(y_all, incumbent_all)
    candidate_metrics = _binary_metrics(y_all, candidate_all)
    delta_f1 = float(candidate_metrics["f1"] - incumbent_metrics["f1"])
    primary = terminal["primary"]
    checks["primary_counts_and_f1_recomputed"] = (
        primary.get("rows") == len(y_all)
        and _same_metrics(primary["incumbent"], incumbent_metrics)
        and _same_metrics(primary["candidate"], candidate_metrics)
        and math.isclose(primary["delta_f1"], delta_f1, rel_tol=0.0, abs_tol=1.0e-15)
    )
    uncertainty = config["evaluation_contract"]["level_2_uncertainty"]
    bootstrap = base._paired_day_block_bootstrap(
        bootstrap_folds,
        replicates=int(uncertainty["replicates"]),
        block_days=int(uncertainty["block_days"]),
        seed=int(uncertainty["seed"]),
    )
    checks["bootstrap_recomputed"] = all(
        terminal["uncertainty"].get(name) == bootstrap.get(name) for name in bootstrap
    )
    expected_state = runner.classify_evidence_state(
        delta_f1=delta_f1,
        ci90_lower=float(bootstrap["ci90_lower"]),
        ci90_upper=float(bootstrap["ci90_upper"]),
        level_0_pass=anchor_removed == 0,
    )
    checks["evidence_state_arithmetic"] = terminal.get("status") == expected_state
    checks["raw_anchor_preserved"] = anchor_removed == 0
    checks["legacy_arbitrary_veto_unused"] = (
        terminal.get("legacy_fixed_delta_or_all_slice_veto_used_for_decision") is False
        and metrics.get("legacy_fixed_delta_or_all_slice_veto_used_for_decision") is False
    )
    own_files = [path for path in artifact_dir.rglob("*") if path.is_file()]
    checks["no_checkpoint_or_csv_artifact"] = not any(
        path.suffix.casefold() in {".pt", ".pth", ".ckpt", ".csv"} for path in own_files
    )
    checks["official_csv_upload_outlier_counters_zero"] = (
        terminal.get("official_test_sample_submission_hidden_rows_read") == 0
        and terminal.get("csv_created") is False
        and terminal.get("upload_performed") is False
        and terminal.get("outlier_rows_hard_deleted") == 0
        and terminal.get("label_1_or_anomaly_events_deleted") == 0
        and manifest.get("official_test_sample_submission_hidden_rows_read") == 0
        and manifest.get("csv_created") is False
        and manifest.get("upload_performed") is False
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "schema_version": "p1.mstcn_sobol_trial18_frozen_confirmation.independent_qa.v3",
        "experiment_id": EXPERIMENT_ID,
        "decision": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "recomputed": {
            "primary": {
                "rows": len(y_all),
                "incumbent": incumbent_metrics,
                "candidate": candidate_metrics,
                "delta_f1": delta_f1,
            },
            "uncertainty": bootstrap,
            "evidence_state": expected_state,
            "raw_anchor_positive_removed_rows": anchor_removed,
            "fit_count": len(fit_receipts),
            "optimizer_steps": sum(row["optimizer_steps"] for row in fit_receipts),
        },
        "official_test_sample_submission_hidden_rows_read": 0,
        "csv_created": False,
        "upload_performed": False,
    }
    if write:
        qa_path = root / "reports" / EXPERIMENT_ID / "independent-qa.json"
        runner._exclusive_json(qa_path, result)
    if failed:
        raise QAError(f"independent QA failed: {failed}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    return parser.parse_args()


def main() -> int:
    _parse_args()
    result = qa()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
