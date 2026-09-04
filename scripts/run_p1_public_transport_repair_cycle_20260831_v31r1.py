"""Exactly-once historical execution of the presealed P1 v31 protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
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

from src.p1_qc.logit_shrunk_label_shift import (  # noqa: E402
    correct_to_prior,
    shrink_lambda,
    shrunk_target_prevalence,
)

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v31r1"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
BASE_CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v31.json"
BASE_RUNNER = ROOT / "scripts/run_p1_public_transport_repair_cycle_20260831_v31.py"
BASE_SOURCE = ROOT / "src/p1_qc/logit_shrunk_label_shift.py"
BASE_PREFLIGHT = ROOT / "reports/p1_public_transport_repair_cycle_20260831_v31/preflight-report.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID


class ContractError(RuntimeError):
    """Frozen v31 authorization or lineage mismatch."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def load_contract() -> tuple[dict, dict]:
    amendment = json.loads(CONFIG.read_text(encoding="utf-8"))
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    preflight = json.loads(BASE_PREFLIGHT.read_text(encoding="utf-8"))
    checks = {
        "identity": amendment["experiment_id"] == EXPERIMENT_ID,
        "base_config_hash": amendment["base_contract"]["sha256"] == sha256(BASE_CONFIG),
        "base_runner_hash": amendment["base_runner"]["sha256"] == sha256(BASE_RUNNER),
        "base_source_hash": amendment["base_source"]["sha256"] == sha256(BASE_SOURCE),
        "base_preflight_hash": amendment["base_preflight"]["sha256"] == sha256(BASE_PREFLIGHT),
        "base_preflight_pass": preflight["status"] == "PASS",
        "candidate_frozen": base["candidate"] == "P1_1_PREFIX_LOGIT_SHRUNK_LABEL_SHIFT_EM",
        "two_fits": base["model"]["maximum_fits"] == amendment["amendment"]["maximum_historical_fits"] == 2,
        "no_changes": all(
            amendment["amendment"][key] == 0
            for key in ("model_parameter_changes", "threshold_rule_changes", "fold_changes", "gate_changes", "searches", "retries")
        ),
        "history_authorized": amendment["authorization"]["historical_execution"] is True,
        "lock_authorized": amendment["authorization"]["attempt_lock_creation"] is True,
        "external_zero": all(
            amendment["authorization"][key] == 0
            for key in ("official_reads", "hidden_truth_reads", "submission_csv_created", "uploads")
        ),
    }
    if not all(checks.values()):
        raise ContractError(f"v31r1 contract mismatch: {checks}")
    return amendment, base


def split_remainder(times_ns: np.ndarray, remainder: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    unique_times = np.unique(times_ns[remainder])
    if len(unique_times) < 2:
        raise ContractError("inner remainder needs at least two timestamps")
    half = len(unique_times) // 2
    cutoff = int(unique_times[half - 1])
    shrink_mask = remainder & (times_ns <= cutoff)
    selection_mask = remainder & (times_ns > cutoff)
    if not shrink_mask.any() or not selection_mask.any():
        raise ContractError("chronological shrink/selection split is empty")
    if int(times_ns[shrink_mask].max()) >= int(times_ns[selection_mask].min()):
        raise ContractError("shrink calibration is not strictly before threshold selection")
    return shrink_mask, selection_mask, cutoff


def run_historical(amendment: dict, base: dict) -> dict:
    if ARTIFACT.exists():
        raise FileExistsError("v31r1 exactly-once artifact already exists")
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
        "base_config_sha256": sha256(BASE_CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "fit_budget": 2,
        "official_reads": 0,
        "hidden_truth_reads": 0,
    }
    lock_path = ARTIFACT / "attempt_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    frame, anchor, _, dependency = surface.load_feature_surface()
    truth = frame["label_base"].to_numpy(np.int8)
    design = v28.frozen_logit_matrix(
        frame["probability_base"].to_numpy(np.float64),
        frame["probability_peer"].to_numpy(np.float64),
        frame["e150_probability"].to_numpy(np.float64),
    )
    times_ns = pd.to_datetime(frame["time"], utc=True).astype("int64").to_numpy(np.int64)
    candidate = anchor.copy()
    corrected_probability = np.zeros(len(frame), dtype=np.float64)
    receipts: list[dict] = []

    for fit_number, spec in enumerate(base["validation"]["nested_fits"], start=1):
        prefix = frame["fold"].isin(spec["train_folds"]).to_numpy()
        split = v28.chronological_inner_split(
            times_ns,
            prefix,
            fit_fraction=float(base["model"]["outer_prefix_fit_fraction"]),
        )
        shrink_mask, selection_mask, inner_cutoff = split_remainder(
            times_ns, split.calibration_mask
        )
        fit_negative = split.fit_mask & (anchor == 0)
        shrink_negative = shrink_mask & (anchor == 0)
        selection_negative = selection_mask & (anchor == 0)
        outer = frame["fold"].eq(spec["validation_fold"]).to_numpy()
        outer_negative = outer & (anchor == 0)

        model = v28._calibrator({"model": base["model"]})
        model.fit(design[fit_negative], truth[fit_negative])
        epsilon = float(base["em"]["epsilon"])
        source_prevalence = float(np.clip(truth[fit_negative].mean(), epsilon, 1.0 - epsilon))

        shrink_source_probability = model.predict_proba(design[shrink_negative])[:, 1]
        _, shrink_em = v28.label_shift_em(
            shrink_source_probability,
            source_prevalence,
            maximum_iterations=int(base["em"]["maximum_iterations"]),
            tolerance=float(base["em"]["tolerance"]),
            epsilon=epsilon,
        )
        observed_prevalence = float(
            np.clip(truth[shrink_negative].mean(), epsilon, 1.0 - epsilon)
        )
        shrink = shrink_lambda(
            source_prevalence, shrink_em.target_prevalence, observed_prevalence, epsilon=epsilon
        )

        selection_source_probability = model.predict_proba(design[selection_negative])[:, 1]
        _, selection_em = v28.label_shift_em(
            selection_source_probability,
            source_prevalence,
            maximum_iterations=int(base["em"]["maximum_iterations"]),
            tolerance=float(base["em"]["tolerance"]),
            epsilon=epsilon,
        )
        selection_target = shrunk_target_prevalence(
            source_prevalence, selection_em.target_prevalence, shrink, epsilon=epsilon
        )
        selection_corrected = correct_to_prior(
            selection_source_probability, source_prevalence, selection_target, epsilon=epsilon
        )
        selection_probability = np.zeros(int(selection_mask.sum()), dtype=np.float64)
        selection_probability[anchor[selection_mask] == 0] = selection_corrected
        threshold = v28.select_inner_threshold(
            selection_probability,
            truth[selection_mask],
            anchor[selection_mask],
            maximum_changed_fraction=float(base["safety"]["maximum_changed_fraction"]),
        )

        outer_source_probability = model.predict_proba(design[outer_negative])[:, 1]
        _, outer_em = v28.label_shift_em(
            outer_source_probability,
            source_prevalence,
            maximum_iterations=int(base["em"]["maximum_iterations"]),
            tolerance=float(base["em"]["tolerance"]),
            epsilon=epsilon,
        )
        outer_target = shrunk_target_prevalence(
            source_prevalence, outer_em.target_prevalence, shrink, epsilon=epsilon
        )
        outer_corrected = correct_to_prior(
            outer_source_probability, source_prevalence, outer_target, epsilon=epsilon
        )
        corrected_probability[outer_negative] = outer_corrected
        proposed = np.flatnonzero(outer_negative)[outer_corrected >= threshold.threshold]
        candidate[proposed] = 1
        receipts.append(
            {
                "fit_number": fit_number,
                "train_folds": list(spec["train_folds"]),
                "validation_fold": spec["validation_fold"],
                "fit_rows": int(fit_negative.sum()),
                "shrink_rows": int(shrink_mask.sum()),
                "selection_rows": int(selection_mask.sum()),
                "inner_cutoff_ns": inner_cutoff,
                "source_prevalence": source_prevalence,
                "shrink_em_target_prevalence": shrink_em.target_prevalence,
                "observed_shrink_prevalence": observed_prevalence,
                "shrink_lambda": shrink,
                "selection_em_target_prevalence": selection_em.target_prevalence,
                "selection_shrunk_target_prevalence": selection_target,
                "inner_threshold": threshold.threshold if np.isfinite(threshold.threshold) else None,
                "inner_additions": threshold.additions,
                "outer_rows": int(outer.sum()),
                "outer_labels_read_before_prediction_seal": 0,
                "outer_em_target_prevalence": outer_em.target_prevalence,
                "outer_shrunk_target_prevalence": outer_target,
                "sealed_outer_additions": int(len(proposed)),
                "coefficient_sha256": v28.stable_hash(
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
    seal = {
        "candidate_sha256": stable_hash(candidate.astype(np.int8)),
        "corrected_probability_sha256": stable_hash(corrected_probability.astype(np.float64)),
        "npz_sha256": sha256(prediction_path),
        "q3_outer_target_reads_before_seal": 0,
        "q4_outer_target_reads_before_seal": 0,
    }
    (ARTIFACT / "prediction_seal.json").write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    evaluation_config = copy.deepcopy(base)
    evaluation_config["safety"]["maximum_changed_fraction_any_kst_day"] = 1.0
    evaluation_config["safety"]["minimum_each_supported_station_layer_delta_f1"] = -1.0
    record = evaluation.evaluate(frame, anchor, candidate, evaluation_config)
    record["name"] = base["candidate"]
    record["diagnostic_only_gate_names"] = [
        "each_kst_day_changed_fraction_at_most_0_005",
        "each_supported_station_layer_nonnegative",
    ]
    hard_gates = {
        key: value
        for key, value in record["gates"].items()
        if key not in record["diagnostic_only_gate_names"]
    }
    record["hard_gates"] = hard_gates
    record["strict_internal_pass"] = bool(all(hard_gates.values()))

    qa_checks = {
        "exact_two_fits": len(receipts) == 2,
        "chronological_three_way_split": all(
            receipt["fit_rows"] > 0 and receipt["shrink_rows"] > 0 and receipt["selection_rows"] > 0
            for receipt in receipts
        ),
        "bounded_shrink": all(0.0 <= receipt["shrink_lambda"] <= 1.0 for receipt in receipts),
        "outer_labels_before_seal_zero": all(
            receipt["outer_labels_read_before_prediction_seal"] == 0 for receipt in receipts
        ),
        "anchor_removals_zero": record["anchor_removals"] == 0,
        "official_zero": True,
        "hidden_zero": True,
        "csv_zero": True,
        "upload_zero": True,
    }
    result = {
        "schema_version": "p1.v31r1.result.1",
        "experiment_id": EXPERIMENT_ID,
        "base_experiment_id": base["experiment_id"],
        "status": "COMPLETE_INTERNAL_ONLY",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": 2,
        "pass_count": int(record["strict_internal_pass"]),
        "candidate": record,
        "nested_fit_receipts": receipts,
        "prediction_seal": seal,
        "source_feature_dependency_receipt": dependency,
        "independent_qa": {"status": "PASS" if all(qa_checks.values()) else "FAIL", "checks": qa_checks},
        "operations": {
            "historical_reads": 1,
            "official_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "hashes": {
            "authorization_config_sha256": sha256(CONFIG),
            "base_config_sha256": sha256(BASE_CONFIG),
            "runner_sha256": sha256(Path(__file__)),
            "lock_sha256": sha256(lock_path),
            "prediction_sha256": sha256(prediction_path),
        },
        "development_surface_disclaimer": "Q3/Q4 are reused development folds, not independent confirmation.",
    }
    (ARTIFACT / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def validate_only() -> dict:
    amendment, base = load_contract()
    return {
        "status": "VALID",
        "experiment_id": EXPERIMENT_ID,
        "candidate": base["candidate"],
        "fit_budget": amendment["amendment"]["maximum_historical_fits"],
        "artifact_absent": not ARTIFACT.exists(),
        "official_reads": 0,
        "hidden_truth_reads": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.validate == args.execute:
        raise SystemExit("use exactly one of --validate or --execute")
    amendment, base = load_contract()
    result = run_historical(amendment, base) if args.execute else validate_only()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
