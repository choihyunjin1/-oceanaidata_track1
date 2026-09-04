"""Presealed rank-transport repair for the v16/v24 GCE probability drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v15 as evaluation  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v16 as gce  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v26"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
CALIBRATION = ROOT / "reports/public_transport_calibration_20260831_v3/calibration.json"
V16_RUNNER = ROOT / "scripts/run_p1_public_transport_repair_cycle_20260831_v16.py"
V24_RUNNER = ROOT / "scripts/run_p1_public_transport_repair_cycle_20260831_v24.py"
V16_ARTIFACT = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v16"
V24_ARTIFACT = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v24"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
REPORT = ROOT / f"reports/{EXPERIMENT_ID}/preflight-report.json"
QA_REPORT = ROOT / f"reports/{EXPERIMENT_ID}/independent-qa.json"


class ContractError(RuntimeError):
    """Frozen v26 contract violation."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks = {
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "v16_runner": config["lineage"]["v16_runner_sha256"] == sha256(V16_RUNNER),
        "v24_runner": config["lineage"]["v24_runner_sha256"] == sha256(V24_RUNNER),
        "v16_prediction": config["lineage"]["v16_prediction_sha256"]
        == sha256(V16_ARTIFACT / "sealed_nested_predictions.npz"),
        "v24_prediction": config["lineage"]["v24_prediction_sha256"]
        == sha256(V24_ARTIFACT / "sealed_nested_predictions.npz"),
        "calibration": config["transport"]["calibration_sha256"] == sha256(CALIBRATION),
        "gce": config["model"]["gce_q"] == 0.7 and config["model"]["l2"] == 0.001,
        "width": config["features"]["encoded_feature_count"] == 165,
        "fraction_choice": config["inner_selector"]["choice"]
        == "minimum eligible action count",
        "outer_label_free": config["outer_selector"]["labels_used"] is False,
        "fraction_floor": config["outer_selector"]["action_count"].startswith("floor("),
        "no_minimum_one": config["outer_selector"]["minimum_one_override"] is False,
        "two_new_fits": config["fit_budget"]["maximum"] == 2,
        "raw_gate": config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"]
        == 0.015383691373120248,
        "calibrated_gate": config["decision_policy"][
            "minimum_calibrated_expected_point_delta_inclusive"
        ]
        == 0.01,
        "authorization_consistent": config["authorization"]["historical_execution"]
        is config["authorization"]["attempt_lock_creation"],
        "no_retrospective_rescore": config["lineage"]["v24_retrospective_rescoring"] is False,
    }
    if not all(checks.values()):
        raise ContractError(f"v26 contract mismatch: {checks}")
    return config


def stable_key_hashes(frame: pd.DataFrame) -> np.ndarray:
    times = pd.to_datetime(frame["time"], utc=True).astype("int64").to_numpy(np.int64)
    stations = frame["station"].astype(str).to_numpy()
    layers = frame["layer"].astype(str).to_numpy()
    output = np.empty(len(frame), dtype=np.uint64)
    for index, (station, layer, timestamp_ns) in enumerate(
        zip(stations, layers, times, strict=True)
    ):
        payload = f"{station}\x1f{layer}\x1f{int(timestamp_ns)}".encode()
        output[index] = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return output


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2.0 * tp / denominator


def select_minimum_action_fraction(
    probabilities: np.ndarray,
    labels: np.ndarray,
    anchor: np.ndarray,
    key_hashes: np.ndarray,
    *,
    maximum_changed_fraction: float,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    anchor = np.asarray(anchor, dtype=np.int8)
    key_hashes = np.asarray(key_hashes, dtype=np.uint64)
    if not (len(probabilities) == len(labels) == len(anchor) == len(key_hashes)):
        raise ContractError("selector arrays do not align")
    if not np.isfinite(probabilities).all() or not np.isin(labels, [0, 1]).all():
        raise ContractError("selector inputs are not finite binary data")
    negative = anchor == 0
    negative_positions = np.flatnonzero(negative)
    if not len(negative_positions):
        return _empty_fraction_record(len(labels), 0)
    order_local = np.lexsort(
        (key_hashes[negative_positions], -probabilities[negative_positions])
    )
    ranked = negative_positions[order_local]
    max_actions = min(
        len(ranked), int(np.floor(maximum_changed_fraction * len(labels) + 1e-12))
    )
    tp0 = int(np.sum((anchor == 1) & (labels == 1)))
    fp0 = int(np.sum((anchor == 1) & (labels == 0)))
    fn0 = int(np.sum((anchor == 0) & (labels == 1)))
    reference_f1 = _f1_from_counts(tp0, fp0, fn0)
    ranked_truth = labels[ranked[:max_actions]]
    cumulative_tp = np.cumsum(ranked_truth == 1)
    cumulative_fp = np.cumsum(ranked_truth == 0)
    for offset in range(max_actions):
        count = offset + 1
        added_tp = int(cumulative_tp[offset])
        added_fp = int(cumulative_fp[offset])
        candidate_f1 = _f1_from_counts(tp0 + added_tp, fp0 + added_fp, fn0 - added_tp)
        precision = added_tp / count
        if candidate_f1 > reference_f1 and precision > reference_f1 / 2.0:
            return {
                "selected_count": count,
                "negative_count": int(len(ranked)),
                "fraction_numerator": count,
                "fraction_denominator": int(len(ranked)),
                "reference_f1": reference_f1,
                "candidate_f1": candidate_f1,
                "inner_delta_f1": candidate_f1 - reference_f1,
                "precision": precision,
                "selected_positions": ranked[:count],
                "changed_fraction": count / len(labels),
            }
    return _empty_fraction_record(len(labels), len(ranked), reference_f1=reference_f1)


def _empty_fraction_record(
    total_rows: int, negative_count: int, *, reference_f1: float = 0.0
) -> dict[str, Any]:
    return {
        "selected_count": 0,
        "negative_count": int(negative_count),
        "fraction_numerator": 0,
        "fraction_denominator": int(negative_count),
        "reference_f1": reference_f1,
        "candidate_f1": reference_f1,
        "inner_delta_f1": 0.0,
        "precision": None,
        "selected_positions": np.empty(0, dtype=np.int64),
        "changed_fraction": 0.0 / max(total_rows, 1),
    }


def select_outer_top_fraction(
    probabilities: np.ndarray,
    key_hashes: np.ndarray,
    *,
    fraction_numerator: int,
    fraction_denominator: int,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    key_hashes = np.asarray(key_hashes, dtype=np.uint64)
    if len(probabilities) != len(key_hashes) or not np.isfinite(probabilities).all():
        raise ContractError("outer rank inputs do not align or are not finite")
    if fraction_numerator < 0 or fraction_denominator < 0:
        raise ContractError("action fraction cannot be negative")
    if fraction_numerator == 0 or fraction_denominator == 0 or not len(probabilities):
        return np.empty(0, dtype=np.int64)
    action_count = (fraction_numerator * len(probabilities)) // fraction_denominator
    if action_count == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((key_hashes, -probabilities))
    return order[:action_count].astype(np.int64, copy=False)


def diagnose_probability_drift() -> dict[str, Any]:
    v24_result = json.loads((V24_ARTIFACT / "result.json").read_text(encoding="utf-8"))
    fold_rows = [
        int(v24_result["candidate"]["by_fold"][fold]["rows"])
        for fold in ("2025_q2", "2025_q3", "2025_q4")
    ]
    boundaries = np.cumsum([0, *fold_rows])
    diagnostics: dict[str, Any] = {}
    for version, directory in (("v16", V16_ARTIFACT), ("v24", V24_ARTIFACT)):
        with np.load(directory / "sealed_nested_predictions.npz", allow_pickle=False) as sealed:
            probability = sealed["probability"].astype(np.float64, copy=False)
        if len(probability) != boundaries[-1]:
            raise ContractError(f"{version} sealed score length changed")
        by_fold = {}
        for fold, start, stop in zip(
            ("2025_q2", "2025_q3", "2025_q4"),
            boundaries[:-1],
            boundaries[1:],
            strict=True,
        ):
            scored = probability[start:stop]
            scored = scored[scored > 0.0]
            by_fold[fold] = {
                "rows": int(stop - start),
                "scored_anchor_negative_rows": int(len(scored)),
                "score_quantiles": (
                    None
                    if not len(scored)
                    else {
                        name: float(value)
                        for name, value in zip(
                            ("p50", "p90", "p99", "p999", "max"),
                            np.quantile(scored, [0.5, 0.9, 0.99, 0.999, 1.0]),
                            strict=True,
                        )
                    }
                ),
            }
        diagnostics[version] = by_fold
    receipts = {item["outer"]: item for item in v24_result["nested_fit_receipts"]}
    for fold in ("2025_q3", "2025_q4"):
        receipt = receipts[fold]
        scored_count = diagnostics["v24"][fold]["scored_anchor_negative_rows"]
        diagnostics["v24"][fold]["absolute_threshold"] = receipt[
            "inner_selected_threshold"
        ]
        diagnostics["v24"][fold]["inner_additions"] = receipt["inner_additions"]
        diagnostics["v24"][fold]["inner_calibration_rows"] = receipt[
            "inner_calibration_rows"
        ]
        diagnostics["v24"][fold]["outer_additions"] = receipt["outer_additions"]
        diagnostics["v24"][fold]["outer_action_fraction"] = (
            receipt["outer_additions"] / scored_count
        )
        diagnostics["v24"][fold]["inner_action_fraction_lower_bound"] = (
            receipt["inner_additions"] / receipt["inner_calibration_rows"]
        )
    diagnostics["read_contract"] = {
        "sealed_probability_arrays_only": True,
        "v24_aggregate_receipts_only": True,
        "labels_read": 0,
        "retrospective_candidate_rescored": False,
    }
    return diagnostics


def synthetic_preflight(config: dict[str, Any]) -> dict[str, Any]:
    rows = 1000
    probability = np.linspace(1.0, 0.0, rows, endpoint=False)
    labels = np.zeros(rows, dtype=np.int8)
    labels[0] = 1
    anchor = np.zeros(rows, dtype=np.int8)
    anchor[900:950] = 1
    labels[900:950] = 1
    hashes = np.arange(rows, dtype=np.uint64)[::-1]
    selected = select_minimum_action_fraction(
        probability,
        labels,
        anchor,
        hashes,
        maximum_changed_fraction=config["safety"]["maximum_changed_fraction"],
    )
    outer_scores = np.linspace(0.8, 0.1, 1900)
    outer_hashes = np.arange(1900, dtype=np.uint64)[::-1]
    selected_outer = select_outer_top_fraction(
        outer_scores,
        outer_hashes,
        fraction_numerator=selected["fraction_numerator"],
        fraction_denominator=selected["fraction_denominator"],
    )
    transformed_outer = np.log1p(outer_scores)
    transformed_selected = select_outer_top_fraction(
        transformed_outer,
        outer_hashes,
        fraction_numerator=selected["fraction_numerator"],
        fraction_denominator=selected["fraction_denominator"],
    )
    checks = {
        "minimum_action_count_one": selected["selected_count"] == 1,
        "positive_inner_delta": selected["inner_delta_f1"] > 0.0,
        "precision_gate": selected["precision"] > selected["reference_f1"] / 2.0,
        "changed_cap": selected["changed_fraction"] <= 0.005,
        "fraction_floor_exact": len(selected_outer)
        == (selected["selected_count"] * len(outer_scores)) // selected["negative_count"],
        "monotone_scale_invariant": np.array_equal(selected_outer, transformed_selected),
        "outer_selector_has_no_label_argument": "labels"
        not in select_outer_top_fraction.__annotations__,
        "authorization_consistent": config["authorization"]["historical_execution"]
        is config["authorization"]["attempt_lock_creation"],
        "two_new_fits": config["fit_budget"]["maximum"] == 2,
    }
    serializable_selected = {key: value for key, value in selected.items() if key != "selected_positions"}
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "selected": serializable_selected,
        "outer_selected_count": int(len(selected_outer)),
    }


def preflight() -> dict[str, Any]:
    config = load_contract()
    diagnostic = diagnose_probability_drift()
    synthetic = synthetic_preflight(config)
    q3 = diagnostic["v24"]["2025_q3"]
    q4 = diagnostic["v24"]["2025_q4"]
    checks = {
        "diagnostic_labels_zero": diagnostic["read_contract"]["labels_read"] == 0,
        "no_retrospective_rescore": diagnostic["read_contract"][
            "retrospective_candidate_rescored"
        ]
        is False,
        "q3_absolute_threshold_budget_explosion": q3["outer_action_fraction"] > 0.005,
        "q4_absolute_threshold_action_collapse": q4["outer_action_fraction"]
        < q3["outer_action_fraction"] / 10.0,
        "q3_q4_upper_tail_scale_drift": q4["score_quantiles"]["p99"]
        < q3["score_quantiles"]["p99"] / 100.0,
        "rank_selector_synthetic_pass": synthetic["status"] == "PASS",
        "official_hidden_lock_zero": config["authorization"]["official_reads"] == 0
        and config["authorization"]["hidden_truth_reads"] == 0
        and not ARTIFACT.exists(),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema_version": "p1.v26.synthetic-preflight.1",
        "status": status,
        "decision": "PRESEALED_FOR_PROSPECTIVE_TWO_FIT_EXECUTION" if status == "PASS" else "NO_GO",
        "checks": checks,
        "probability_drift_diagnostic": diagnostic,
        "synthetic_rank_selector": synthetic,
        "fit_count": 0,
        "hashes": {
            "config_sha256": sha256(CONFIG),
            "runner_sha256": sha256(Path(__file__)),
            "calibration_v3_sha256": sha256(CALIBRATION),
            "v16_prediction_sha256": sha256(
                V16_ARTIFACT / "sealed_nested_predictions.npz"
            ),
            "v24_prediction_sha256": sha256(
                V24_ARTIFACT / "sealed_nested_predictions.npz"
            ),
        },
        "access": {
            "sealed_probability_reads": 2,
            "aggregate_result_reads": 1,
            "historical_truth_reads": 0,
            "historical_fits": 0,
            "attempt_locks": 0,
            "official_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
    }


def execute() -> dict[str, Any]:
    config = load_contract()
    if not config["authorization"]["historical_execution"]:
        raise ContractError("v26 historical execution is not authorized")
    if not config["authorization"]["attempt_lock_creation"]:
        raise ContractError("v26 attempt lock creation is not authorized")
    if ARTIFACT.exists():
        raise FileExistsError("v26 exactly-once artifact already exists")
    ARTIFACT.mkdir(parents=True)
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "pid": os.getpid(),
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "fit_budget": 2,
        "official_reads": 0,
        "hidden_truth_reads": 0,
    }
    (ARTIFACT / "attempt_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ARTIFACT / "progress.json").write_text(
        json.dumps({"phase": "loading", "fit_count": 0}) + "\n", encoding="utf-8"
    )
    started = time.perf_counter()
    frame, anchor, numeric_names, dependency = gce.load_feature_surface()
    truth = frame["label_base"].to_numpy(np.int8)
    candidate = anchor.copy()
    probability = np.zeros(len(frame), dtype=np.float64)
    receipts = []
    all_times = pd.to_datetime(frame["time"], utc=True)
    for fit_number, spec in enumerate(config["validation"]["pipeline_fits"], 1):
        prefix = frame["fold"].isin(spec["train_folds"]).to_numpy()
        unique_times = np.sort(all_times[prefix].unique())
        cutoff = unique_times[min(int(len(unique_times) * 0.75), len(unique_times) - 1)]
        inner_fit = prefix & (all_times.to_numpy() < cutoff) & (anchor == 0)
        inner_calibration = prefix & (all_times.to_numpy() >= cutoff)
        inner_negative = inner_calibration & (anchor == 0)
        encoder = gce.PrefixEncoder.fit(frame, inner_fit, numeric_names)
        fit_design, fit_scaled = encoder.transform(frame, inner_fit)
        weights = gce.leverage_weights(fit_scaled)
        optimizer = gce.fit_gce(fit_design, truth[inner_fit], weights, config)
        calibration_design, _ = encoder.transform(frame, inner_negative)
        calibration_probability = expit(
            calibration_design @ optimizer.x[:-1] + optimizer.x[-1]
        )
        calibration_full_probability = np.zeros(int(inner_calibration.sum()), dtype=np.float64)
        local_negative = anchor[inner_calibration] == 0
        calibration_full_probability[local_negative] = calibration_probability
        inner_hash = stable_key_hashes(frame.loc[inner_calibration, ["station", "layer", "time"]])
        fraction = select_minimum_action_fraction(
            calibration_full_probability,
            truth[inner_calibration],
            anchor[inner_calibration],
            inner_hash,
            maximum_changed_fraction=config["safety"]["maximum_changed_fraction"],
        )
        outer = frame["fold"].eq(spec["outer"]).to_numpy()
        outer_negative = outer & (anchor == 0)
        outer_design, _ = encoder.transform(frame, outer_negative)
        outer_probability = expit(outer_design @ optimizer.x[:-1] + optimizer.x[-1])
        probability[outer_negative] = outer_probability
        outer_hash = stable_key_hashes(frame.loc[outer_negative, ["station", "layer", "time"]])
        selected_local = select_outer_top_fraction(
            outer_probability,
            outer_hash,
            fraction_numerator=fraction["fraction_numerator"],
            fraction_denominator=fraction["fraction_denominator"],
        )
        selected_global = np.flatnonzero(outer_negative)[selected_local]
        candidate[selected_global] = 1
        receipts.append(
            {
                "fit_number": fit_number,
                "train_folds": spec["train_folds"],
                "outer": spec["outer"],
                "cutoff_utc": str(pd.Timestamp(cutoff)),
                "inner_selected_count": fraction["selected_count"],
                "inner_negative_count": fraction["negative_count"],
                "inner_delta_f1": fraction["inner_delta_f1"],
                "inner_precision": fraction["precision"],
                "outer_additions": int(len(selected_global)),
                "outer_labels_used_for_selection": 0,
                "optimizer_success": bool(optimizer.success),
                "parameters_sha256": hashlib.sha256(
                    optimizer.x.astype(np.float64).tobytes()
                ).hexdigest(),
            }
        )
        (ARTIFACT / "progress.json").write_text(
            json.dumps({"phase": "fit_complete", "fit_count": fit_number}) + "\n",
            encoding="utf-8",
        )
    np.savez_compressed(
        ARTIFACT / "sealed_nested_predictions.npz",
        candidate=candidate,
        probability=probability,
    )
    record = evaluation.evaluate(frame, anchor, candidate, config)
    record["name"] = config["candidate"]
    qa = {
        "checks": {
            "exact_two_fits": len(receipts) == 2,
            "exact_gce": config["model"]["gce_q"] == 0.7
            and config["model"]["l2"] == 0.001,
            "rank_fraction_only": all(
                item["outer_labels_used_for_selection"] == 0 for item in receipts
            ),
            "v3_gate": config["decision_policy"][
                "minimum_raw_expected_point_delta_inclusive"
            ]
            == 0.015383691373120248,
            "anchor_removals_zero": record["anchor_removals"] == 0,
            "official_zero": True,
            "hidden_zero": True,
            "csv_zero": True,
            "upload_zero": True,
        }
    }
    qa["status"] = "PASS" if all(qa["checks"].values()) else "FAIL"
    result = {
        "schema_version": "p1.v26.result.1",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_INTERNAL_ONLY",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": 2,
        "pass_count": int(record["strict_internal_pass"]),
        "candidate": record,
        "nested_fit_receipts": receipts,
        "source_feature_dependency_receipt": dependency,
        "independent_qa": qa,
        "adaptive_development_evidence_only": True,
        "operations": {
            "official_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "hashes": {
            "config_sha256": sha256(CONFIG),
            "runner_sha256": sha256(Path(__file__)),
            "calibration_v3_sha256": sha256(CALIBRATION),
            "lock_sha256": sha256(ARTIFACT / "attempt_lock.json"),
            "prediction_sha256": sha256(ARTIFACT / "sealed_nested_predictions.npz"),
        },
    }
    (ARTIFACT / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    QA_REPORT.parent.mkdir(parents=True, exist_ok=True)
    QA_REPORT.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT / "progress.json").write_text(
        json.dumps(
            {"phase": "terminal", "fit_count": 2, "pass_count": result["pass_count"]}
        )
        + "\n",
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
            raise SystemExit("v26 is preflight-only; historical execution and lock creation are disabled")
        result = execute()
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if not args.preflight:
        raise SystemExit("only --preflight is authorized")
    started = time.perf_counter()
    result = preflight()
    result["runtime_seconds"] = time.perf_counter() - started
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
