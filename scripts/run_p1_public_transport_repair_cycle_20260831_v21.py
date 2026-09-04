"""Presealed SCAR-PU residual candidate for P1 v21.

Only synthetic preflight is authorized.  The historical path is implemented so
its leakage boundary can be tested, but it aborts before reads or lock creation
while the sealed config keeps historical_execution=false.
"""

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

from src.p1_qc.causal_scar_pu import (  # noqa: E402
    chronological_inner_split,
    correct_selection_probability,
    estimate_scar_propensity,
    select_add_only_threshold,
)

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v21"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
SOURCE = ROOT / "src/p1_qc/causal_scar_pu.py"
REPORT = ROOT / "reports" / EXPERIMENT_ID / "preflight-report.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
CALIBRATION = ROOT / "reports/public_transport_calibration_20260831_v2/calibration.json"


class ContractError(RuntimeError):
    """Frozen v21 registration mismatch."""


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
    tier = calibration["tier_gates"]["SMOOTH_LEARNED_PROFILE"]
    checks = {
        "identity": config["experiment_id"] == EXPERIMENT_ID,
        "candidate": config["candidate"] == "P1_1_CAUSAL_SCAR_PU_LINEAR_ADDONLY",
        "family": config["transport_family"]["family_id"] == "P1_SCAR_PU_SMOOTH_LINEAR_RESIDUAL",
        "tier": config["transport_family"]["tier_id"] == "SMOOTH_LEARNED_PROFILE",
        "calibration_hash": config["transport_family"]["selected_penalty_provenance_sha256"]
        == sha256(CALIBRATION),
        "penalty": np.isclose(config["decision_policy"]["transport_penalty_points"], tier["transport_penalty_points"]),
        "raw_gate": np.isclose(
            config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"],
            tier["minimum_raw_expected_points_delta"],
        ),
        "two_fits": config["fit_budget"]["maximum"] == 2,
        "inner_only": config["inner_calibration"]["outer_labels_used"] is False,
        "no_retune": config["validation"]["outer_result_based_tuning"] is False,
        "no_delete": config["model"]["row_deletion"] is False,
        "historical_on": config["authorization"]["historical_execution"] is True,
        "lock_on": config["authorization"]["attempt_lock_creation"] is True,
        "official_zero": config["authorization"]["official_reads"] == 0,
        "hidden_zero": config["authorization"]["hidden_truth_reads"] == 0,
        "upload_zero": config["authorization"]["uploads"] == 0,
    }
    if not all(checks.values()):
        raise ContractError(f"v21 contract mismatch: {checks}")
    return config


def synthetic_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Exercise exactly the fit/calibrate/threshold order on synthetic rows."""
    rng = np.random.default_rng(20260926)
    rows = 960
    width = 12
    times = np.repeat(np.arange(rows // 3, dtype=np.int64), 3) * 600_000_000_000
    design = rng.normal(size=(rows, width))
    latent_probability = 1.0 / (1.0 + np.exp(-(design[:, 0] * 1.4 - design[:, 1] * 0.7 - 2.7)))
    latent = rng.binomial(1, latent_probability).astype(np.int8)
    selected = latent & (rng.random(rows) < 0.62)
    labels = selected.astype(np.int8)
    incumbent = np.zeros(rows, dtype=np.int8)
    incumbent[(design[:, 0] > 2.4) & (labels == 1)] = 1
    split = chronological_inner_split(times, np.ones(rows, dtype=bool), fit_fraction=0.75)
    model = LogisticRegression(
        C=float(config["model"]["C"]),
        solver="lbfgs",
        max_iter=int(config["model"]["max_iter"]),
        tol=float(config["model"]["tol"]),
        class_weight=None,
    )
    model.fit(design[split.fit_mask], labels[split.fit_mask])
    calibration_selection = model.predict_proba(design[split.calibration_mask])[:, 1]
    propensity = estimate_scar_propensity(
        calibration_selection,
        labels[split.calibration_mask],
        minimum_positive_support=int(config["model"]["minimum_inner_positive_support"]),
        lower_clip=float(config["model"]["propensity_lower_clip"]),
    )
    corrected = correct_selection_probability(calibration_selection, propensity)
    threshold = select_add_only_threshold(
        corrected,
        labels[split.calibration_mask],
        incumbent[split.calibration_mask],
        maximum_changed_fraction=float(config["safety"]["maximum_changed_fraction"]),
    )
    future_design = np.vstack([design, np.full((2, width), 999.0)])
    repeated = model.predict_proba(future_design[:rows][split.calibration_mask])[:, 1]
    checks = {
        "fit_before_calibration": int(times[split.fit_mask].max()) < int(times[split.calibration_mask].min()),
        "outer_or_future_rows_in_fit_zero": bool(np.array_equal(repeated, calibration_selection)),
        "propensity_finite_unit_interval": bool(0 < propensity <= 1),
        "corrected_probability_finite_unit_interval": bool(
            np.isfinite(corrected).all() and ((corrected >= 0) & (corrected <= 1)).all()
        ),
        "threshold_inner_only": bool(np.isfinite(threshold.threshold) or np.isinf(threshold.threshold)),
        "add_only": threshold.additions >= 0,
        "coefficient_finite": bool(np.isfinite(model.coef_).all()),
    }
    return {
        "checks": checks,
        "fit_rows": int(split.fit_mask.sum()),
        "calibration_rows": int(split.calibration_mask.sum()),
        "calibration_positive_rows": int(labels[split.calibration_mask].sum()),
        "propensity": propensity,
        "threshold": threshold.threshold,
        "threshold_additions": threshold.additions,
        "coefficient_sha256": stable_hash(model.coef_.astype(np.float64), model.intercept_.astype(np.float64)),
        "corrected_probability_sha256": stable_hash(corrected),
    }


def run_preflight() -> dict[str, Any]:
    config = load_contract()
    synthetic = synthetic_pipeline(config)
    checks = {
        **synthetic["checks"],
        "synthetic_only_authorized": config["authorization"]["synthetic_preflight"] is True,
        "historical_execution_authorized": config["authorization"]["historical_execution"] is True,
        "attempt_lock_creation_authorized": config["authorization"]["attempt_lock_creation"] is True,
        "artifact_absent": not ARTIFACT.exists(),
        "fit_budget_two": config["fit_budget"]["maximum"] == 2,
        "search_and_retry_zero": config["fit_budget"]["hyperparameter_searches"] == 0
        and config["fit_budget"]["retries"] == 0,
        "outer_result_retuning_zero": config["validation"]["outer_result_based_tuning"] is False,
        "official_hidden_csv_upload_zero": all(
            config["authorization"][name] == 0
            for name in ("official_reads", "hidden_truth_reads", "submission_csv_created", "uploads")
        ),
    }
    return {
        "schema_version": "p1.v21.synthetic-preflight.1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "candidate": config["candidate"],
        "checks": checks,
        "synthetic": {key: value for key, value in synthetic.items() if key != "checks"},
        "fit_budget_if_later_authorized": 2,
        "historical_fits_executed": 0,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "hashes": {
            "config_sha256": sha256(CONFIG),
            "source_sha256": sha256(SOURCE),
            "runner_sha256": sha256(Path(__file__)),
            "calibration_sha256": sha256(CALIBRATION),
        },
        "access": {
            "historical_truth_reads": 0,
            "official_reads": 0,
            "hidden_truth_reads": 0,
            "attempt_locks_created": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "limitation": "SCAR is a working model assumption on a reused development surface, not identified label-noise truth or independent confirmation.",
    }


def _selection_model(config: dict[str, Any]) -> LogisticRegression:
    return LogisticRegression(
        C=float(config["model"]["C"]),
        solver="lbfgs",
        max_iter=int(config["model"]["max_iter"]),
        tol=float(config["model"]["tol"]),
        class_weight=None,
    )


def run_historical(config: dict[str, Any]) -> dict[str, Any]:
    """Dormant sealed historical path; authorization is checked by ``main`` first."""
    if ARTIFACT.exists():
        raise FileExistsError("v21 exactly-once artifact already exists")
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
    frame, anchor, numeric_names, dependency = source.load_feature_surface()
    truth = frame["label_base"].to_numpy(np.int8)
    candidate = anchor.copy()
    corrected_probability = np.zeros(len(frame), dtype=np.float64)
    receipts: list[dict[str, Any]] = []
    times_ns = pd.to_datetime(frame["time"], utc=True).astype("int64").to_numpy(np.int64)
    for fit_number, fold_spec in enumerate(config["validation"]["nested_fits"], start=1):
        prefix = frame["fold"].isin(fold_spec["train_folds"]).to_numpy()
        split = chronological_inner_split(times_ns, prefix, fit_fraction=float(config["model"]["fit_fraction"]))
        fit_negative = split.fit_mask & (anchor == 0)
        calibration = split.calibration_mask
        calibration_negative = calibration & (anchor == 0)
        outer = frame["fold"].eq(fold_spec["validation_fold"]).to_numpy()
        outer_negative = outer & (anchor == 0)
        encoder = source.PrefixEncoder.fit(frame, fit_negative, numeric_names)
        fit_design, _ = encoder.transform(frame, fit_negative)
        calibration_design, _ = encoder.transform(frame, calibration_negative)
        model = _selection_model(config)
        model.fit(fit_design, truth[fit_negative])
        inner_selection = model.predict_proba(calibration_design)[:, 1]
        propensity = estimate_scar_propensity(
            inner_selection,
            truth[calibration_negative],
            minimum_positive_support=int(config["model"]["minimum_inner_positive_support"]),
            lower_clip=float(config["model"]["propensity_lower_clip"]),
        )
        inner_corrected = correct_selection_probability(inner_selection, propensity)
        inner_score = np.zeros(int(calibration.sum()), dtype=np.float64)
        inner_anchor = anchor[calibration]
        inner_score[inner_anchor == 0] = inner_corrected
        threshold = select_add_only_threshold(
            inner_score,
            truth[calibration],
            inner_anchor,
            maximum_changed_fraction=float(config["safety"]["maximum_changed_fraction"]),
        )
        outer_design, _ = encoder.transform(frame, outer_negative)
        outer_selection = model.predict_proba(outer_design)[:, 1]
        outer_corrected = correct_selection_probability(outer_selection, propensity)
        corrected_probability[outer_negative] = outer_corrected
        proposed_positions = np.flatnonzero(outer_negative)[outer_corrected >= threshold.threshold]
        candidate[proposed_positions] = 1
        receipts.append(
            {
                "fit_number": fit_number,
                "train_folds": fold_spec["train_folds"],
                "validation_fold": fold_spec["validation_fold"],
                "fit_rows": int(fit_negative.sum()),
                "inner_calibration_rows": int(calibration.sum()),
                "inner_calibration_positive_rows": int(truth[calibration_negative].sum()),
                "inner_cutoff_ns": split.cutoff_ns,
                "outer_rows": int(outer.sum()),
                "outer_labels_read_before_prediction_seal": 0,
                "propensity": propensity,
                "threshold": threshold.threshold,
                "inner_threshold_additions": threshold.additions,
                "sealed_outer_additions": int(len(proposed_positions)),
                "coefficient_sha256": stable_hash(
                    model.coef_.astype(np.float64),
                    model.intercept_.astype(np.float64),
                ),
            }
        )
    np.savez_compressed(
        ARTIFACT / "sealed_nested_predictions.npz",
        candidate=candidate,
        corrected_probability=corrected_probability,
    )
    record = evaluation.evaluate(frame, anchor, candidate, config)
    record["name"] = config["candidate"]
    qa_checks = {
        "exact_two_fits": len(receipts) == 2,
        "outer_labels_before_seal_zero": all(
            item["outer_labels_read_before_prediction_seal"] == 0 for item in receipts
        ),
        "anchor_removals_zero": record["anchor_removals"] == 0,
        "official_zero": True,
        "hidden_zero": True,
        "upload_zero": True,
    }
    qa = {"status": "PASS" if all(qa_checks.values()) else "FAIL", "checks": qa_checks}
    result = {
        "schema_version": "p1.v21.result.1",
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
            "prediction_sha256": sha256(ARTIFACT / "sealed_nested_predictions.npz"),
        },
        "development_surface_disclaimer": "Q3/Q4 are reused adaptive development folds, not an independent confirmation.",
    }
    (ARTIFACT / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
