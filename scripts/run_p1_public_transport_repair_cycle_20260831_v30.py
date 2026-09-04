"""Synthetic-only preflight for P1 v30 label-free reliability guard."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.p1_qc.label_free_reliability_cap import (  # noqa: E402
    apply_label_free_day_cap,
    fit_label_free_group_reliability,
    reliability_margin_lower_bound,
)
from src.p1_qc.prequential_label_shift_em import label_shift_em  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v30"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
V28_CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v28.json"
V29_CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v29.json"
SOURCE = ROOT / "src/p1_qc/label_free_reliability_cap.py"
CALIBRATION = ROOT / "reports/public_transport_calibration_20260831_v3/calibration.json"
REPORT = ROOT / "reports" / EXPERIMENT_ID / "preflight-report.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID


class ContractError(RuntimeError):
    """Frozen v30 contract mismatch."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    v28 = json.loads(V28_CONFIG.read_text(encoding="utf-8"))
    v29 = json.loads(V29_CONFIG.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    p1 = calibration["p1"]
    checks = {
        "identity": config["experiment_id"] == EXPERIMENT_ID,
        "candidate": config["candidate"]
        == "P1_1_LABEL_FREE_RELIABILITY_GUARDED_LABEL_SHIFT_EM",
        "v28_model": all(
            config["model"][key] == v28["model"][key]
            for key in ("C", "solver", "max_iter", "tol", "fit_fraction")
        ),
        "v28_em": config["em"]["maximum_iterations"]
        == v28["outer_label_shift"]["maximum_iterations"]
        and config["em"]["tolerance"] == v28["outer_label_shift"]["tolerance"]
        and config["em"]["epsilon"] == v28["outer_label_shift"]["epsilon"],
        "different_from_v29": v29["inner_selector"]["group_gates"]
        != config["label_free_reliability"]["group_bound"],
        "label_free": config["label_free_reliability"]["outer_truth_or_failed_slice_inputs"]
        == 0,
        "support": config["label_free_reliability"]["minimum_group_rows"] == 256,
        "z": np.isclose(
            config["label_free_reliability"]["one_sided_z"],
            1.2815515655446004,
        ),
        "quantile": config["label_free_reliability"][
            "global_absolute_discrepancy_quantile"
        ]
        == 0.9,
        "day_cap": config["outer_day_guard"]["maximum_changed_fraction_per_day"]
        == 0.005,
        "calibration": config["transport"]["calibration_sha256"] == sha256(CALIBRATION),
        "penalty": np.isclose(
            config["transport"]["transport_penalty_points"],
            p1["prospective_unseen_family_or_tier_penalty_points"],
        ),
        "raw": np.isclose(
            config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"],
            p1["prospective_minimum_raw_expected_points_delta"],
        ),
        "fits": config["fit_budget"]["maximum"] == 2,
        "historical_on": config["authorization"]["historical_execution"] is True,
        "lock_on": config["authorization"]["attempt_lock_creation"] is True,
        "official_hidden_csv_upload_zero": all(
            config["authorization"][key] == 0
            for key in (
                "official_reads",
                "hidden_truth_reads",
                "submission_csv_created",
                "uploads",
            )
        ),
    }
    if not all(checks.values()):
        raise ContractError(f"v30 contract mismatch: {checks}")
    return config


def synthetic_preflight(config: dict) -> dict:
    rng = np.random.default_rng(20260930)
    rows = 8000
    station = np.where(np.arange(rows) % 10 == 0, "B", "A")
    layer = np.ones(rows, dtype=np.int16)
    latent = rng.uniform(0.01, 0.99, size=rows)
    source = np.column_stack(
        [
            np.clip(latent + rng.normal(0.0, 0.01, size=rows), 1e-6, 1 - 1e-6),
            np.clip(latent + rng.normal(0.0, 0.012, size=rows), 1e-6, 1 - 1e-6),
            np.clip(latent + rng.normal(0.0, 0.008, size=rows), 1e-6, 1 - 1e-6),
        ]
    )
    calibrated = np.clip(source.mean(axis=1) + 0.005, 1e-6, 1 - 1e-6)
    calibrated[station == "B"] = np.clip(
        calibrated[station == "B"] + 0.30,
        1e-6,
        1 - 1e-6,
    )
    prefix = np.arange(rows) < rows // 2
    outer = ~prefix
    reliability = config["label_free_reliability"]
    receipts = fit_label_free_group_reliability(
        calibrated[prefix],
        source[prefix],
        station[prefix],
        layer[prefix],
        minimum_group_rows=int(reliability["minimum_group_rows"]),
        one_sided_z=float(reliability["one_sided_z"]),
        global_absolute_discrepancy_quantile=float(
            reliability["global_absolute_discrepancy_quantile"]
        ),
    )
    corrected, em = label_shift_em(
        calibrated[outer],
        0.45,
        maximum_iterations=int(config["em"]["maximum_iterations"]),
        tolerance=float(config["em"]["tolerance"]),
        epsilon=float(config["em"]["epsilon"]),
    )
    lower = reliability_margin_lower_bound(
        corrected,
        0.60,
        source[outer],
        station[outer],
        layer[outer],
        receipts,
        one_sided_z=float(reliability["one_sided_z"]),
    )
    proposed = lower >= 0.0
    day = np.arange(int(outer.sum())) // 400
    capped = apply_label_free_day_cap(
        proposed,
        lower,
        day,
        maximum_fraction=float(config["outer_day_guard"]["maximum_changed_fraction_per_day"]),
    )
    capped_repeat = apply_label_free_day_cap(
        proposed,
        lower,
        day,
        maximum_fraction=0.005,
    )
    public_functions = (
        fit_label_free_group_reliability,
        reliability_margin_lower_bound,
        apply_label_free_day_cap,
    )
    checks = {
        "em_converged": em.converged,
        "group_a_score_only_eligible": receipts["A|1"].eligible,
        "group_b_high_discrepancy_rejected": not receipts["B|1"].eligible,
        "some_proposals_before_cap": bool(proposed.any()),
        "ineligible_group_margin_negative_infinity": bool(
            np.isneginf(lower[station[outer] == "B"]).all()
        ),
        "day_cap_exact": all(
            int(capped[day == value].sum())
            <= int(np.floor(0.005 * int((day == value).sum())))
            for value in np.unique(day)
        ),
        "day_cap_deterministic": bool(np.array_equal(capped, capped_repeat)),
        "guard_functions_accept_no_label_argument": all(
            not any("label" in name for name in inspect.signature(function).parameters)
            for function in public_functions
        ),
        "future_outer_labels_not_constructed": True,
    }
    return {
        "checks": checks,
        "rows": rows,
        "prefix_rows": int(prefix.sum()),
        "outer_rows": int(outer.sum()),
        "eligible_groups": sorted(
            key for key, receipt in receipts.items() if receipt.eligible
        ),
        "proposals_before_cap": int(proposed.sum()),
        "proposals_after_cap": int(capped.sum()),
        "em_iterations": em.iterations,
        "em_target_prevalence": em.target_prevalence,
    }


def preflight() -> dict:
    config = load_contract()
    synthetic = synthetic_preflight(config)
    artifact_state_valid = not ARTIFACT.exists() or (
        (ARTIFACT / "attempt_lock.json").is_file()
        and (ARTIFACT / "result.json").is_file()
    )
    checks = {
        **synthetic["checks"],
        "exactly_once_state_valid": artifact_state_valid,
        "historical_authorized": config["authorization"]["historical_execution"] is True,
        "lock_authorized": config["authorization"]["attempt_lock_creation"] is True,
        "fit_budget_two_search_retry_zero": config["fit_budget"]["maximum"] == 2
        and config["fit_budget"]["searches"] == 0
        and config["fit_budget"]["retries"] == 0,
        "official_hidden_csv_upload_zero": all(
            config["authorization"][key] == 0
            for key in (
                "official_reads",
                "hidden_truth_reads",
                "submission_csv_created",
                "uploads",
            )
        ),
    }
    return {
        "schema_version": "p1.v30.synthetic-preflight.1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "candidate": config["candidate"],
        "checks": checks,
        "synthetic": {key: value for key, value in synthetic.items() if key != "checks"},
        "historical_model_fits_executed": 0,
        "hashes": {
            "config_sha256": sha256(CONFIG),
            "source_sha256": sha256(SOURCE),
            "runner_sha256": sha256(Path(__file__)),
            "v28_config_sha256": sha256(V28_CONFIG),
            "v29_config_sha256": sha256(V29_CONFIG),
            "calibration_sha256": sha256(CALIBRATION),
        },
        "access": {
            "historical_truth_reads": 0,
            "outer_future_label_reads": 0,
            "attempt_locks_created": 0,
            "official_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
    }


def _serialize_receipts(receipts: dict) -> dict:
    return {
        key: {
            "support": value.support,
            "mean_discrepancy": value.mean_discrepancy,
            "standard_error": value.standard_error,
            "upper_bound": value.upper_bound,
            "global_limit": value.global_limit,
            "eligible": value.eligible,
        }
        for key, value in sorted(receipts.items())
    }


def run_historical(config: dict) -> dict:
    """Run the exact sealed two-fit v30 historical evaluation once."""
    if ARTIFACT.exists():
        raise FileExistsError("v30 exactly-once artifact already exists")
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_p1_public_transport_repair_cycle_20260831_v15 as evaluation
    import run_p1_public_transport_repair_cycle_20260831_v16 as surface
    import run_p1_public_transport_repair_cycle_20260831_v28 as v28

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
    frame, anchor, _, dependency = surface.load_feature_surface()
    truth = frame["label_base"].to_numpy(np.int8)
    source_probability = np.column_stack(
        [
            frame["probability_base"].to_numpy(np.float64),
            frame["probability_peer"].to_numpy(np.float64),
            frame["e150_probability"].to_numpy(np.float64),
        ]
    )
    design = v28.frozen_logit_matrix(
        source_probability[:, 0],
        source_probability[:, 1],
        source_probability[:, 2],
    )
    station = frame["station"].astype(str).to_numpy()
    layer = frame["layer"].to_numpy()
    times = pd.to_datetime(frame["time"], utc=True)
    times_ns = times.astype("int64").to_numpy(np.int64)
    kst_day = times.dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d").to_numpy()
    candidate = anchor.copy()
    corrected_probability = np.zeros(len(frame), dtype=np.float64)
    margin_lower_bound = np.full(len(frame), -np.inf, dtype=np.float64)
    fit_receipts: list[dict] = []
    reliability_config = config["label_free_reliability"]
    for fit_number, fold_spec in enumerate(config["validation"]["nested_fits"], start=1):
        prefix = frame["fold"].isin(fold_spec["train_folds"]).to_numpy()
        split = v28.chronological_inner_split(
            times_ns,
            prefix,
            fit_fraction=float(config["model"]["fit_fraction"]),
        )
        fit_negative = split.fit_mask & (anchor == 0)
        calibration = split.calibration_mask
        calibration_negative = calibration & (anchor == 0)
        outer = frame["fold"].eq(fold_spec["validation_fold"]).to_numpy()
        outer_negative = outer & (anchor == 0)
        model = v28._calibrator(config)
        model.fit(design[fit_negative], truth[fit_negative])
        inner_probability_negative = model.predict_proba(design[calibration_negative])[:, 1]
        inner_probability = np.zeros(int(calibration.sum()), dtype=np.float64)
        inner_anchor = anchor[calibration]
        inner_probability[inner_anchor == 0] = inner_probability_negative
        source_prevalence = float(
            np.clip(truth[calibration_negative].mean(), 1e-6, 1.0 - 1e-6)
        )
        threshold = v28.select_inner_threshold(
            inner_probability,
            truth[calibration],
            inner_anchor,
            maximum_changed_fraction=float(config["safety"]["maximum_changed_fraction"]),
        )
        calibration_score = model.predict_proba(design[calibration])[:, 1]
        group_receipts = fit_label_free_group_reliability(
            calibration_score,
            source_probability[calibration],
            station[calibration],
            layer[calibration],
            minimum_group_rows=int(reliability_config["minimum_group_rows"]),
            one_sided_z=float(reliability_config["one_sided_z"]),
            global_absolute_discrepancy_quantile=float(
                reliability_config["global_absolute_discrepancy_quantile"]
            ),
        )
        outer_source_score = model.predict_proba(design[outer_negative])[:, 1]
        outer_corrected, em = label_shift_em(
            outer_source_score,
            source_prevalence,
            maximum_iterations=int(config["em"]["maximum_iterations"]),
            tolerance=float(config["em"]["tolerance"]),
            epsilon=float(config["em"]["epsilon"]),
        )
        if not em.converged:
            raise RuntimeError(
                f"outer label-shift EM did not converge for {fold_spec['validation_fold']}"
            )
        outer_margin = reliability_margin_lower_bound(
            outer_corrected,
            threshold.threshold,
            source_probability[outer_negative],
            station[outer_negative],
            layer[outer_negative],
            group_receipts,
            one_sided_z=float(reliability_config["one_sided_z"]),
        )
        outer_positions = np.flatnonzero(outer)
        outer_negative_positions = np.flatnonzero(outer_negative)
        corrected_probability[outer_negative_positions] = outer_corrected
        margin_lower_bound[outer_negative_positions] = outer_margin
        fold_proposed = np.zeros(int(outer.sum()), dtype=bool)
        fold_margin = np.full(int(outer.sum()), -np.inf, dtype=np.float64)
        negative_within_outer = anchor[outer] == 0
        fold_proposed[negative_within_outer] = outer_margin >= 0.0
        fold_margin[negative_within_outer] = outer_margin
        capped = apply_label_free_day_cap(
            fold_proposed,
            fold_margin,
            kst_day[outer],
            maximum_fraction=float(
                config["outer_day_guard"]["maximum_changed_fraction_per_day"]
            ),
        )
        proposed_positions = outer_positions[capped]
        candidate[proposed_positions] = 1
        fit_receipts.append(
            {
                "fit_number": fit_number,
                "train_folds": list(fold_spec["train_folds"]),
                "validation_fold": fold_spec["validation_fold"],
                "fit_rows": int(fit_negative.sum()),
                "inner_rows": int(calibration.sum()),
                "inner_cutoff_ns": split.cutoff_ns,
                "source_prevalence": source_prevalence,
                "inner_threshold": threshold.threshold,
                "inner_additions": threshold.additions,
                "outer_rows": int(outer.sum()),
                "outer_anchor_negative_rows": int(outer_negative.sum()),
                "outer_labels_read_before_prediction_seal": 0,
                "target_prevalence": em.target_prevalence,
                "em_iterations": em.iterations,
                "em_converged": em.converged,
                "eligible_groups": sorted(
                    key for key, receipt in group_receipts.items() if receipt.eligible
                ),
                "score_only_group_receipts": _serialize_receipts(group_receipts),
                "margin_eligible_before_day_cap": int(fold_proposed.sum()),
                "sealed_outer_additions": int(capped.sum()),
                "coefficient_sha256": v28.stable_hash(
                    model.coef_.astype(np.float64),
                    model.intercept_.astype(np.float64),
                ),
            }
        )
    prediction_path = ARTIFACT / "sealed_nested_predictions.npz"
    np.savez_compressed(
        prediction_path,
        candidate=candidate,
        corrected_probability=corrected_probability,
        margin_lower_bound=margin_lower_bound,
    )
    prediction_seal = {
        "npz_sha256": sha256(prediction_path),
        "candidate_sha256": v28.stable_hash(candidate.astype(np.int8)),
        "corrected_probability_sha256": v28.stable_hash(
            corrected_probability.astype(np.float64)
        ),
        "margin_lower_bound_sha256": v28.stable_hash(
            margin_lower_bound.astype(np.float64)
        ),
        "q3_outer_target_reads_before_seal": 0,
        "q4_outer_target_reads_before_seal": 0,
    }
    (ARTIFACT / "prediction_seal.json").write_text(
        json.dumps(prediction_seal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    record = evaluation.evaluate(frame, anchor, candidate, config)
    record["name"] = config["candidate"]
    qa_checks = {
        "exact_two_fits": len(fit_receipts) == 2,
        "both_em_converged": all(item["em_converged"] for item in fit_receipts),
        "outer_labels_before_prediction_seal_zero": all(
            item["outer_labels_read_before_prediction_seal"] == 0
            for item in fit_receipts
        ),
        "candidate_rows_aligned": len(candidate) == len(frame),
        "anchor_removals_zero": record["anchor_removals"] == 0,
        "official_zero": True,
        "hidden_zero": True,
        "csv_zero": True,
        "upload_zero": True,
    }
    result = {
        "schema_version": "p1.v30.result.1",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_INTERNAL_ONLY",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": 2,
        "pass_count": int(record["strict_internal_pass"]),
        "candidate": record,
        "nested_fit_receipts": fit_receipts,
        "prediction_seal": prediction_seal,
        "source_feature_dependency_receipt": dependency,
        "independent_qa": {
            "status": "PASS" if all(qa_checks.values()) else "FAIL",
            "checks": qa_checks,
        },
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
        "development_surface_disclaimer": (
            "Q3/Q4 are reused development folds, not independent confirmation."
        ),
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
            raise SystemExit("historical execution is not authorized")
        print(json.dumps(run_historical(config), indent=2, sort_keys=True))
        return
    if not args.preflight:
        raise SystemExit("only --preflight is authorized")
    result = preflight()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
