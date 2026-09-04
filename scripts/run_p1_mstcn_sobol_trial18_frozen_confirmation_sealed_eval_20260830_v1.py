"""Zero-fit evaluation of already-sealed trial_18 Q3/Q4 blind artifacts.

The original v3 runner completed all six fits and sealed both phase prediction
artifacts, then failed before opening truth because it addressed
``EncodedSurface.keys``.  This evaluator pins every original byte, changes only
the evaluator-side key binding to ``holdout.surface.keys``, evaluates historical
truth exactly once, and never fits, predicts, reads official data, writes CSV,
or uploads.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "p1_mstcn_sobol_trial18_frozen_confirmation_sealed_eval_20260830_v1"
ORIGINAL_ID = "p1_mstcn_sobol_trial18_frozen_confirmation_20260830_v3"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
ATTEMPT_LOCK = ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
QA_PATH = ROOT / "reports" / EXPERIMENT_ID / "independent-qa.json"


class ContractError(RuntimeError):
    """Raised when a pinned seal or zero-fit recovery invariant changes."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"JSON object required: {path}")
    return value


def _identity(path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _config(*, root: Path = ROOT) -> dict[str, Any]:
    path = root / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
    config = _load_json(path)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("sealed-evaluation experiment identity changed")
    contract = config.get("recovery_contract", {})
    if not (
        contract.get("original_runner_reexecution") is False
        and contract.get("additional_model_fits") == 0
        and contract.get("reprediction") is False
        and contract.get("retuning") is False
        and contract.get("threshold_search") is False
        and contract.get("historical_truth_metric_evaluations") == 1
    ):
        raise ContractError("zero-fit recovery contract changed")
    recipe = config.get("frozen_recipe_identity", {})
    if not (
        recipe.get("trial_id") == "trial_18"
        and recipe.get("threshold") == 0.8
        and recipe.get("epoch") == 150
        and recipe.get("seeds") == [20260827, 20260839, 20260863]
        and recipe.get("fit_count_already_completed") == 6
    ):
        raise ContractError("frozen recipe identity changed")
    if not all(config.get("prohibitions", {}).values()):
        raise ContractError("a recovery prohibition was disabled")
    return config


def _verify_pins(config: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for name, expected in config["pinned_inputs"].items():
        path = (root / expected["path"]).resolve(strict=True)
        if not path.is_relative_to(root.resolve()):
            raise ContractError(f"pinned input escapes repository: {name}")
        identity = _identity(path, root=root)
        if identity != expected:
            raise ContractError(f"pinned input changed: {name}")
        observed[name] = identity
    return observed


def _load_original(*, root: Path = ROOT) -> Any:
    path = root / "scripts" / f"run_{ORIGINAL_ID}.py"
    name = f"{EXPERIMENT_ID}_original"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContractError("cannot import pinned original runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _original_failure_matches(config: dict[str, Any], terminal: dict[str, Any]) -> bool:
    expected = config["required_original_terminal"]
    return all(terminal.get(name) == value for name, value in expected.items())


def _blind_receipts(*, root: Path = ROOT) -> dict[str, dict[str, Any]]:
    artifact_dir = root / "artifacts" / ORIGINAL_ID
    return {
        phase: _load_json(artifact_dir / f"{phase}_confirmatory_blind_receipt.json")
        for phase in ("q3", "q4")
    }


def _receipt_contract(receipts: dict[str, dict[str, Any]]) -> bool:
    fit_receipts = [
        row for phase in ("q3", "q4") for row in receipts[phase].get("fit_receipts", [])
    ]
    return (
        len(fit_receipts) == 6
        and [row.get("phase") for row in fit_receipts] == ["q3"] * 3 + ["q4"] * 3
        and [row.get("seed") for row in fit_receipts] == [20260827, 20260839, 20260863] * 2
        and all(row.get("trial_id") == "trial_18" for row in fit_receipts)
        and all(row.get("epochs") == 150 for row in fit_receipts)
        and all(row.get("checkpoint_persisted") is False for row in fit_receipts)
        and all(row.get("nonfinite_count_total") == 0 for row in fit_receipts)
        and all(
            receipts[phase].get("same_fold_holdout_truth_columns_opened_before_receipt") == 0
            and receipts[phase].get("official_test_sample_submission_hidden_rows_read") == 0
            and receipts[phase].get("csv_created") is False
            and receipts[phase].get("upload_performed") is False
            for phase in ("q3", "q4")
        )
    )


def check_only(*, root: Path = ROOT) -> dict[str, Any]:
    config = _config(root=root)
    pins = _verify_pins(config, root=root)
    terminal = _load_json(root / config["pinned_inputs"]["original_failure_terminal"]["path"])
    receipts = _blind_receipts(root=root)
    original_metrics_absent = not (
        root / "artifacts" / ORIGINAL_ID / "confirmatory_metrics.json"
    ).exists()
    checks = {
        "all_pinned_bytes_match": len(pins) == len(config["pinned_inputs"]),
        "original_terminal_is_exact_technical_failure": _original_failure_matches(config, terminal),
        "six_completed_fits_and_both_blind_seals_valid": _receipt_contract(receipts),
        "original_performance_metrics_absent": original_metrics_absent,
        "new_namespace_available": not (root / "artifacts" / EXPERIMENT_ID).exists()
        and not (root / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json").exists()
        and not (root / "reports" / EXPERIMENT_ID / "independent-qa.json").exists(),
    }
    return {
        "schema_version": "p1.mstcn_sobol_trial18_frozen_confirmation.sealed_eval.preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "pinned_inputs": pins,
        "additional_model_fits_authorized": 0,
        "reprediction_authorized": False,
        "historical_truth_metric_evaluations_authorized": 1,
        "official_test_sample_submission_hidden_rows_read": 0,
        "csv_created": False,
        "upload_performed": False,
    }


def _load_truth_fixed(
    base: Any,
    original: Any,
    source_config: dict[str, Any],
    holdout: Any,
    receipt_paths: Sequence[Path],
    *,
    fold: str,
    config_sha256: str,
    recipe_sha256: str,
    root: Path,
) -> Any:
    """Original truth loader with only the EncodedSurface key path corrected."""
    import pyarrow.dataset as dataset

    holdout_keys = holdout.surface.keys
    expected_key = base._ordered_key_sha(holdout_keys)
    for receipt_path in receipt_paths:
        original._verify_blind_receipt(
            receipt_path,
            config_sha256=config_sha256,
            recipe_sha256=recipe_sha256,
            expected_key_sha256=expected_key,
        )
    with base._verified_immutable_read(source_config, "frozen_truth_and_folds", root=root) as path:
        truth = (
            dataset.dataset(path, format="parquet")
            .scanner(
                columns=[*base.KEY_COLUMNS, "label", "anomaly_type", "fold"],
                filter=dataset.field("fold") == fold,
                use_threads=True,
            )
            .to_table()
            .to_pandas()
            .reset_index(drop=True)
        )
    truth, _membership = base._validate_registered_holdout_membership(
        truth, source_config, fold=fold
    )
    if not base._keys_equal(holdout_keys, truth):
        raise ContractError("opened truth keys differ from blind holdout keys")
    return truth


def _load_candidates(
    original: Any,
    receipts: dict[str, dict[str, Any]],
    *,
    original_artifact_dir: Path,
) -> dict[str, Any]:
    import numpy as np

    candidates: dict[str, Any] = {}
    for phase in ("q3", "q4"):
        score_path = original_artifact_dir / receipts[phase]["score_path"]
        with np.load(score_path, allow_pickle=False) as archive:
            candidate = archive["candidate"].astype(np.int8, copy=True)
        expected = receipts[phase]["array_inventory"]["candidate"]
        if list(candidate.shape) != expected["shape"] or str(candidate.dtype) != expected["dtype"]:
            raise ContractError(f"sealed candidate inventory changed: {phase}")
        candidates[phase] = candidate
    return candidates


def _artifact_manifest(artifact_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(artifact_dir.iterdir()):
        if path.is_file() and path.name not in {"artifact_manifest.json", "terminal_result.json"}:
            files.append(_identity(path))
    return {
        "schema_version": "p1.mstcn_sobol_trial18_frozen_confirmation.sealed_eval.manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "files": files,
        "model_fit_count": 0,
        "model_prediction_count": 0,
        "official_test_sample_submission_hidden_rows_read": 0,
        "csv_created": False,
        "upload_performed": False,
    }


def execute(*, expected_runner_sha256: str, root: Path = ROOT) -> dict[str, Any]:
    runner_path = Path(__file__)
    runner_sha256 = _sha256(runner_path)
    if expected_runner_sha256.casefold() != runner_sha256:
        raise ContractError("--expected-runner-sha256 must match reviewed evaluator bytes")
    preflight = check_only(root=root)
    if preflight["decision"] != "PASS":
        raise ContractError("sealed-evaluation preflight failed")
    config = _config(root=root)
    config_path = root / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
    config_sha256 = _sha256(config_path)
    artifact_dir = root / "artifacts" / EXPERIMENT_ID
    attempt_lock = root / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
    qa_path = root / "reports" / EXPERIMENT_ID / "independent-qa.json"
    source_before = _verify_pins(config, root=root)
    lock = {
        "schema_version": "p1.mstcn_sobol_trial18_frozen_confirmation.sealed_eval.attempt.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runner_sha256": runner_sha256,
        "config_sha256": config_sha256,
        "one_shot_historical_truth_evaluation": True,
        "additional_model_fit_count": 0,
        "reprediction_count": 0,
        "automatic_retry_authorized": False,
    }
    _exclusive_json(attempt_lock, lock)
    artifact_dir.mkdir(parents=False, exist_ok=False)
    started = time.perf_counter()
    terminal_path = artifact_dir / "terminal_result.json"
    try:
        _exclusive_json(artifact_dir / "preflight.json", preflight)
        original = _load_original(root=root)
        original_config = original._config(root=root)
        base = original._load_base(root=root)
        source_config = base._canonical_config(
            root / "configs" / "experiments" / f"{original.BASE_EXPERIMENT_ID}.json"
        )
        surfaces = base.load_blind_surfaces(source_config, root=root)
        controls = original._load_controls(base, original_config, surfaces, root=root)
        original_artifact_dir = root / "artifacts" / ORIGINAL_ID
        receipts = _blind_receipts(root=root)
        receipt_paths = [
            original_artifact_dir / f"{phase}_confirmatory_blind_receipt.json"
            for phase in ("q3", "q4")
        ]
        candidates = _load_candidates(
            original, receipts, original_artifact_dir=original_artifact_dir
        )
        holds: dict[str, Any] = {}
        truths: dict[str, Any] = {}
        for phase in ("q3", "q4"):
            _encoder, _training, holdout, _split = base._prepare_phase_surfaces(
                surfaces, source_config, phase, root=root
            )
            holds[phase] = holdout
            truths[phase] = _load_truth_fixed(
                base,
                original,
                source_config,
                holdout,
                receipt_paths,
                fold={"q3": "2025_q3", "q4": "2025_q4"}[phase],
                config_sha256=source_before["original_config"]["sha256"],
                recipe_sha256=config["frozen_recipe_identity"]["recipe_sha256"],
                root=root,
            )
        fit_receipts = [row for phase in ("q3", "q4") for row in receipts[phase]["fit_receipts"]]
        metrics = original._evaluate(
            base,
            truths,
            holds,
            candidates,
            controls,
            original_config,
            fit_receipts=fit_receipts,
        )
        metrics["recovery_annotation"] = config["recovery_contract"]["claim_annotation"]
        metrics["original_terminal_status"] = "FAILED_EXECUTION_NO_RETRY_AUTHORIZED"
        metrics["additional_model_fit_count"] = 0
        metrics["reprediction_count"] = 0
        metrics["historical_truth_metric_evaluation_count"] = 1
        _exclusive_json(artifact_dir / "confirmatory_metrics.json", metrics)
        source_after = _verify_pins(config, root=root)
        checks = {
            "all_original_and_sealed_bytes_unchanged": source_after == source_before,
            "original_failure_terminal_pinned": _original_failure_matches(
                config,
                _load_json(root / config["pinned_inputs"]["original_failure_terminal"]["path"]),
            ),
            "both_blind_receipts_and_six_fit_receipts_valid": _receipt_contract(receipts),
            "additional_model_fit_count_eq_0": metrics["additional_model_fit_count"] == 0,
            "reprediction_count_eq_0": metrics["reprediction_count"] == 0,
            "historical_truth_metric_evaluation_count_eq_1": (
                metrics["historical_truth_metric_evaluation_count"] == 1
            ),
            "level_0_pass": metrics["level_0_pass"] is True,
            "primary_delta_arithmetic": metrics["primary"]["delta_f1"]
            == metrics["primary"]["candidate"]["f1"] - metrics["primary"]["incumbent"]["f1"],
            "state_arithmetic": metrics["evidence_state"]
            == original.classify_evidence_state(
                delta_f1=metrics["primary"]["delta_f1"],
                ci90_lower=metrics["uncertainty"]["ci90_lower"],
                ci90_upper=metrics["uncertainty"]["ci90_upper"],
                level_0_pass=metrics["level_0_pass"],
            ),
            "legacy_veto_unused": (
                metrics["legacy_fixed_delta_or_all_slice_veto_used_for_decision"] is False
            ),
            "official_hidden_csv_upload_outlier_counters_zero": (
                metrics["outlier_rows_hard_deleted"] == 0
                and metrics["label_1_or_anomaly_events_deleted"] == 0
            ),
        }
        qa = {
            "schema_version": "p1.mstcn_sobol_trial18_frozen_confirmation.sealed_eval.independent_qa.v1",
            "experiment_id": EXPERIMENT_ID,
            "decision": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "recovery_annotation": config["recovery_contract"]["claim_annotation"],
            "primary": metrics["primary"],
            "uncertainty": metrics["uncertainty"],
            "evidence_state": metrics["evidence_state"],
            "candidate_submission_readiness": metrics["candidate_submission_readiness"],
            "original_fit_count": 6,
            "additional_model_fit_count": 0,
            "reprediction_count": 0,
            "historical_truth_metric_evaluation_count": 1,
            "official_test_sample_submission_hidden_rows_read": 0,
            "csv_created": False,
            "upload_performed": False,
        }
        if qa["decision"] != "PASS":
            raise ContractError("independent sealed-artifact arithmetic QA failed")
        _exclusive_json(qa_path, qa)
        manifest = _artifact_manifest(artifact_dir)
        _exclusive_json(artifact_dir / "artifact_manifest.json", manifest)
        manifest_path = artifact_dir / "artifact_manifest.json"
        terminal = {
            "schema_version": "p1.mstcn_sobol_trial18_frozen_confirmation.sealed_eval.terminal.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": metrics["evidence_state"],
            "recovery_status": "ORIGINAL_RUNNER_TECHNICAL_FAILURE_RECOVERED_BY_SEALED_ARTIFACT_INDEPENDENT_EVALUATION",
            "claim_scope": metrics["claim_scope"],
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "config_sha256": config_sha256,
            "runner_sha256": runner_sha256,
            "attempt_lock_sha256": _sha256(attempt_lock),
            "original_failure_terminal_sha256": source_before["original_failure_terminal"][
                "sha256"
            ],
            "q3_blind_prediction_sha256": source_before["q3_blind_prediction"]["sha256"],
            "q4_blind_prediction_sha256": source_before["q4_blind_prediction"]["sha256"],
            "artifact_manifest_sha256": _sha256(manifest_path),
            "independent_qa_sha256": _sha256(qa_path),
            "original_fit_count": 6,
            "additional_model_fit_count": 0,
            "reprediction_count": 0,
            "optimizer_steps": int(sum(row["optimizer_steps"] for row in fit_receipts)),
            "historical_truth_metric_evaluation_count": 1,
            "selected_trial": "trial_18",
            "threshold": 0.8,
            "epoch": 150,
            "seeds": [20260827, 20260839, 20260863],
            "primary": metrics["primary"],
            "uncertainty": metrics["uncertainty"],
            "level_0_pass": metrics["level_0_pass"],
            "legacy_fixed_delta_or_all_slice_veto_used_for_decision": False,
            "candidate_submission_readiness": metrics["candidate_submission_readiness"],
            "official_test_sample_submission_hidden_rows_read": 0,
            "csv_created": False,
            "upload_performed": False,
            "outlier_rows_hard_deleted": 0,
            "label_1_or_anomaly_events_deleted": 0,
            "automatic_retry_authorized": False,
        }
        _exclusive_json(terminal_path, terminal)
        print(json.dumps(terminal, ensure_ascii=False, allow_nan=False), flush=True)
        return terminal
    except BaseException as error:
        if not terminal_path.exists():
            failure = {
                "schema_version": "p1.mstcn_sobol_trial18_frozen_confirmation.sealed_eval.terminal.v1",
                "experiment_id": EXPERIMENT_ID,
                "status": "FAILED_ZERO_FIT_EVALUATION_NO_RETRY_AUTHORIZED",
                "claim_scope": "NO_RECOVERED_PERFORMANCE_CLAIM",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "original_fit_count": 6,
                "additional_model_fit_count": 0,
                "reprediction_count": 0,
                "official_test_sample_submission_hidden_rows_read": 0,
                "csv_created": False,
                "upload_performed": False,
                "automatic_retry_authorized": False,
            }
            _exclusive_json(terminal_path, failure)
        raise


def run_smoke() -> dict[str, Any]:
    config = _config()
    fake = type("Encoded", (), {})()
    fake.surface = type("Surface", (), {"keys": "corrected-path"})()
    holdout_keys = fake.surface.keys
    return {
        "schema_version": "p1.mstcn_sobol_trial18_frozen_confirmation.sealed_eval.smoke.v1",
        "decision": "PASS",
        "holdout_surface_key_regression": holdout_keys == "corrected-path",
        "additional_model_fits_authorized": config["recovery_contract"]["additional_model_fits"],
        "historical_truth_metric_evaluations_authorized": config["recovery_contract"][
            "historical_truth_metric_evaluations"
        ],
        "official_test_sample_submission_hidden_rows_read": 0,
        "csv_created": False,
        "upload_performed": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--synthetic-smoke", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-runner-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.synthetic_smoke:
        result = run_smoke()
    elif args.check_only:
        result = check_only()
    else:
        if not args.expected_runner_sha256:
            raise ContractError("--execute requires --expected-runner-sha256")
        result = execute(expected_runner_sha256=args.expected_runner_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
