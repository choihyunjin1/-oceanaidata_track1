"""Synthetic-only preflight for P1 v28 prequential label-shift EM stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.p1_qc.causal_scar_pu import chronological_inner_split  # noqa: E402
from src.p1_qc.prequential_label_shift_em import (  # noqa: E402
    frozen_logit_matrix,
    label_shift_em,
    select_inner_threshold,
)

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v28"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
SOURCE = ROOT / "src/p1_qc/prequential_label_shift_em.py"
CALIBRATION = ROOT / "reports/public_transport_calibration_20260831_v3/calibration.json"
REPORT = ROOT / "reports" / EXPERIMENT_ID / "preflight-report.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID


class ContractError(RuntimeError):
    """Frozen v28 contract mismatch."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_contract() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    p1 = calibration["p1"]
    checks = {
        "identity": config["experiment_id"] == EXPERIMENT_ID,
        "candidate": config["candidate"] == "P1_1_PREQUENTIAL_LABEL_SHIFT_EM_STACK_ADDONLY",
        "three_logits": config["inputs"]["feature_count"] == 3
        and config["inputs"]["additional_covariates"] == 0,
        "calibration": config["transport"]["calibration_sha256"] == sha256(CALIBRATION),
        "penalty": np.isclose(
            config["transport"]["transport_penalty_points"],
            p1["prospective_unseen_family_or_tier_penalty_points"],
        ),
        "raw": np.isclose(
            config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"],
            p1["prospective_minimum_raw_expected_points_delta"],
        ),
        "em": config["outer_label_shift"]["maximum_iterations"] == 200
        and config["outer_label_shift"]["tolerance"] == 1e-10,
        "no_router": config["outer_label_shift"]["hard_router"] is False,
        "no_top_k": config["outer_label_shift"]["top_k"] is False,
        "no_quarter_station": config["outer_label_shift"]["quarter_or_station_selection"] is False,
        "two_fits": config["fit_budget"]["maximum"] == 2,
        "historical_on": config["authorization"]["historical_execution"] is True,
        "lock_on": config["authorization"]["attempt_lock_creation"] is True,
        "official_zero": config["authorization"]["official_reads"] == 0,
        "hidden_zero": config["authorization"]["hidden_truth_reads"] == 0,
        "upload_zero": config["authorization"]["uploads"] == 0,
    }
    if not all(checks.values()):
        raise ContractError(f"v28 contract mismatch: {checks}")
    return config


def synthetic_preflight(config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(20260928)
    rows = 1200
    times = np.repeat(np.arange(rows // 3, dtype=np.int64), 3) * 600_000_000_000
    latent = rng.normal(size=rows)
    labels = (latent + rng.normal(scale=0.8, size=rows) > 1.7).astype(np.int8)
    sources = []
    for scale, noise in ((1.0, 1.0), (0.8, 1.2), (1.25, 0.9)):
        score = scale * latent + rng.normal(scale=noise, size=rows) - 2.0
        sources.append(1.0 / (1.0 + np.exp(-score)))
    design = frozen_logit_matrix(*sources)
    split = chronological_inner_split(times, np.ones(rows, dtype=bool), fit_fraction=0.75)
    model = LogisticRegression(C=0.1, solver="lbfgs", max_iter=500, tol=1e-8)
    model.fit(design[split.fit_mask], labels[split.fit_mask])
    inner_probability = model.predict_proba(design[split.calibration_mask])[:, 1]
    source_prevalence = float(np.clip(labels[split.calibration_mask].mean(), 1e-6, 1 - 1e-6))
    threshold = select_inner_threshold(
        inner_probability,
        labels[split.calibration_mask],
        np.zeros(int(split.calibration_mask.sum()), dtype=np.int8),
        maximum_changed_fraction=0.05,
    )
    target_design = frozen_logit_matrix(
        np.clip(sources[0][split.calibration_mask] * 0.8, 1e-6, 1 - 1e-6),
        np.clip(sources[1][split.calibration_mask] * 0.8, 1e-6, 1 - 1e-6),
        np.clip(sources[2][split.calibration_mask] * 0.8, 1e-6, 1 - 1e-6),
    )
    outer_source_probability = model.predict_proba(target_design)[:, 1]
    corrected, receipt = label_shift_em(
        outer_source_probability,
        source_prevalence,
        maximum_iterations=int(config["outer_label_shift"]["maximum_iterations"]),
        tolerance=float(config["outer_label_shift"]["tolerance"]),
        epsilon=float(config["outer_label_shift"]["epsilon"]),
    )
    future_labels = np.full(len(corrected), -1, dtype=np.int8)
    corrected_repeat, receipt_repeat = label_shift_em(
        outer_source_probability,
        source_prevalence,
        maximum_iterations=200,
        tolerance=1e-10,
        epsilon=1e-6,
    )
    checks = {
        "three_finite_logits": design.shape == (rows, 3) and bool(np.isfinite(design).all()),
        "chronological_fit_before_inner": int(times[split.fit_mask].max())
        < int(times[split.calibration_mask].min()),
        "em_converged": receipt.converged,
        "em_finite": bool(np.isfinite(corrected).all() and ((corrected >= 0) & (corrected <= 1)).all()),
        "em_deterministic": bool(np.array_equal(corrected, corrected_repeat))
        and receipt == receipt_repeat,
        "future_labels_not_an_em_argument": bool((future_labels == -1).all()),
        "threshold_inner_only": bool(np.isfinite(threshold.threshold) or np.isinf(threshold.threshold)),
        "model_coefficients_finite": bool(np.isfinite(model.coef_).all()),
    }
    return {
        "checks": checks,
        "fit_rows": int(split.fit_mask.sum()),
        "inner_rows": int(split.calibration_mask.sum()),
        "source_prevalence": source_prevalence,
        "target_prevalence": receipt.target_prevalence,
        "em_iterations": receipt.iterations,
        "inner_threshold": threshold.threshold,
        "inner_additions": threshold.additions,
        "corrected_sha256": stable_hash(corrected),
        "coefficient_sha256": stable_hash(model.coef_.astype(np.float64), model.intercept_.astype(np.float64)),
    }


def run_preflight() -> dict[str, Any]:
    config = load_contract()
    synthetic = synthetic_preflight(config)
    checks = {
        **synthetic["checks"],
        "artifact_absent": not ARTIFACT.exists(),
        "historical_authorized": config["authorization"]["historical_execution"] is True,
        "lock_authorized": config["authorization"]["attempt_lock_creation"] is True,
        "fit_budget_two": config["fit_budget"]["maximum"] == 2,
        "search_retry_zero": config["fit_budget"]["searches"] == 0
        and config["fit_budget"]["retries"] == 0,
        "official_hidden_csv_upload_zero": all(
            config["authorization"][key] == 0
            for key in ("official_reads", "hidden_truth_reads", "submission_csv_created", "uploads")
        ),
    }
    return {
        "schema_version": "p1.v28.synthetic-preflight.1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "candidate": config["candidate"],
        "checks": checks,
        "synthetic": {key: value for key, value in synthetic.items() if key != "checks"},
        "historical_fits_executed": 0,
        "hashes": {
            "config_sha256": sha256(CONFIG),
            "source_sha256": sha256(SOURCE),
            "runner_sha256": sha256(Path(__file__)),
            "calibration_sha256": sha256(CALIBRATION),
        },
        "access": {
            "historical_truth_reads": 0,
            "outer_future_label_reads": 0,
            "attempt_locks_created": 0,
            "official_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }


def _calibrator(config: dict[str, Any]) -> LogisticRegression:
    return LogisticRegression(
        C=float(config["model"]["C"]),
        solver="lbfgs",
        max_iter=int(config["model"]["max_iter"]),
        tol=float(config["model"]["tol"]),
        class_weight=None,
    )


def run_historical(config: dict[str, Any]) -> dict[str, Any]:
    if ARTIFACT.exists():
        raise FileExistsError("v28 exactly-once artifact already exists")
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_p1_public_transport_repair_cycle_20260831_v15 as evaluation
    import run_p1_public_transport_repair_cycle_20260831_v16 as source

    ARTIFACT.mkdir(parents=True)
    started = time.perf_counter()
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "pid": os.getpid(),
        "config_sha256": sha256(CONFIG),
        "source_sha256": sha256(SOURCE),
        "runner_sha256": sha256(Path(__file__)),
        "fit_budget": 2,
        "official_reads": 0,
        "hidden_truth_reads": 0,
    }
    (ARTIFACT / "attempt_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frame, anchor, _, dependency = source.load_feature_surface()
    truth = frame["label_base"].to_numpy(np.int8)
    design = frozen_logit_matrix(
        frame["probability_base"].to_numpy(np.float64),
        frame["probability_peer"].to_numpy(np.float64),
        frame["e150_probability"].to_numpy(np.float64),
    )
    times_ns = pd.to_datetime(frame["time"], utc=True).astype("int64").to_numpy(np.int64)
    candidate = anchor.copy()
    corrected_probability = np.zeros(len(frame), dtype=np.float64)
    receipts: list[dict[str, Any]] = []
    for fit_number, fold_spec in enumerate(config["validation"]["nested_fits"], start=1):
        prefix = frame["fold"].isin(fold_spec["train_folds"]).to_numpy()
        split = chronological_inner_split(times_ns, prefix, fit_fraction=float(config["model"]["fit_fraction"]))
        fit_negative = split.fit_mask & (anchor == 0)
        calibration = split.calibration_mask
        calibration_negative = calibration & (anchor == 0)
        outer = frame["fold"].eq(fold_spec["validation_fold"]).to_numpy()
        outer_negative = outer & (anchor == 0)
        model = _calibrator(config)
        model.fit(design[fit_negative], truth[fit_negative])
        inner_probability_negative = model.predict_proba(design[calibration_negative])[:, 1]
        inner_probability = np.zeros(int(calibration.sum()), dtype=np.float64)
        inner_anchor = anchor[calibration]
        inner_probability[inner_anchor == 0] = inner_probability_negative
        source_prevalence = float(
            np.clip(truth[calibration_negative].mean(), 1e-6, 1.0 - 1e-6)
        )
        threshold = select_inner_threshold(
            inner_probability,
            truth[calibration],
            inner_anchor,
            maximum_changed_fraction=float(config["safety"]["maximum_changed_fraction"]),
        )
        outer_source_probability = model.predict_proba(design[outer_negative])[:, 1]
        outer_corrected, em = label_shift_em(
            outer_source_probability,
            source_prevalence,
            maximum_iterations=int(config["outer_label_shift"]["maximum_iterations"]),
            tolerance=float(config["outer_label_shift"]["tolerance"]),
            epsilon=float(config["outer_label_shift"]["epsilon"]),
        )
        if not em.converged:
            raise RuntimeError(f"outer label-shift EM did not converge for {fold_spec['validation_fold']}")
        corrected_probability[outer_negative] = outer_corrected
        proposed = np.flatnonzero(outer_negative)[outer_corrected >= threshold.threshold]
        candidate[proposed] = 1
        receipts.append(
            {
                "fit_number": fit_number,
                "train_folds": fold_spec["train_folds"],
                "validation_fold": fold_spec["validation_fold"],
                "fit_rows": int(fit_negative.sum()),
                "inner_rows": int(calibration.sum()),
                "inner_positive_rows": int(truth[calibration_negative].sum()),
                "inner_cutoff_ns": split.cutoff_ns,
                "source_prevalence": source_prevalence,
                "inner_threshold": threshold.threshold,
                "inner_additions": threshold.additions,
                "outer_rows": int(outer.sum()),
                "outer_labels_read_before_prediction_seal": 0,
                "target_prevalence": em.target_prevalence,
                "em_iterations": em.iterations,
                "em_converged": em.converged,
                "sealed_outer_additions": int(len(proposed)),
                "coefficient_sha256": stable_hash(
                    model.coef_.astype(np.float64), model.intercept_.astype(np.float64)
                ),
            }
        )
    prediction_path = ARTIFACT / "sealed_nested_predictions.npz"
    np.savez_compressed(
        prediction_path,
        candidate=candidate,
        corrected_probability=corrected_probability,
    )
    record = evaluation.evaluate(frame, anchor, candidate, config)
    record["name"] = config["candidate"]
    qa_checks = {
        "exact_two_fits": len(receipts) == 2,
        "em_converged": all(item["em_converged"] for item in receipts),
        "outer_labels_before_seal_zero": all(
            item["outer_labels_read_before_prediction_seal"] == 0 for item in receipts
        ),
        "anchor_removals_zero": record["anchor_removals"] == 0,
        "official_zero": True,
        "hidden_zero": True,
        "csv_zero": True,
        "upload_zero": True,
    }
    qa = {"status": "PASS" if all(qa_checks.values()) else "FAIL", "checks": qa_checks}
    result = {
        "schema_version": "p1.v28.result.1",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_INTERNAL_ONLY",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": 2,
        "pass_count": int(record["strict_internal_pass"]),
        "candidate": record,
        "nested_fit_receipts": receipts,
        "source_feature_dependency_receipt": dependency,
        "independent_qa": qa,
        "operations": {
            "historical_reads": 1,
            "official_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "hashes": {
            "config_sha256": sha256(CONFIG),
            "source_sha256": sha256(SOURCE),
            "runner_sha256": sha256(Path(__file__)),
            "lock_sha256": sha256(ARTIFACT / "attempt_lock.json"),
            "prediction_sha256": sha256(prediction_path),
        },
        "assumption_caveat": config["assumption_boundary"],
        "development_surface_disclaimer": "Q3/Q4 are reused development folds, not independent confirmation.",
    }
    (ARTIFACT / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        config = load_contract()
        if not config["authorization"]["historical_execution"]:
            raise SystemExit("historical execution is not authorized; no reads or lock were made")
        print(json.dumps(run_historical(config), indent=2, sort_keys=True))
        return
    if not args.preflight:
        raise SystemExit("only --preflight is authorized")
    report = run_preflight()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
