"""Exactly-once metric-only bidirectional replay of frozen P1 CIF probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v13 as truth_source  # noqa: E402

EXPERIMENT_ID = "p1_causal_cif_selective_bidirectional_20260831_v32d"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
KEYS = ["station", "year", "layer", "time"]


class ContractError(RuntimeError):
    """Raised when a frozen v32d contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")


def resolve_source(spec: dict[str, str]) -> Path:
    path = (ROOT / spec["path"]).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ContractError(f"source is absent or escapes repository: {path}")
    if sha256_file(path) != spec["sha256"]:
        raise ContractError(f"source hash changed: {path}")
    return path


def load_contract() -> tuple[dict[str, Any], dict[str, Path]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    checks = {
        "schema": config["schema_version"]
        == "p1.causal_cif_selective_bidirectional.20260831.v32d",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "threshold": config["decoder"]["probability_threshold_inclusive"] == 0.5,
        "fraction": config["decoder"]["maximum_changed_fraction_per_fold"] == 0.005,
        "tie_keys": config["decoder"]["tie_keys"] == KEYS,
        "folds": config["validation"]["outer_folds"] == ["2025_q3", "2025_q4"],
        "fits": config["validation"]["model_fits"] == 0,
        "no_retry": config["decoder"]["outer_result_based_retry_or_retuning"] is False,
        "official_zero": config["authorization"]["official_reads"] == 0,
        "hidden_zero": config["authorization"]["hidden_truth_reads"] == 0,
        "test_zero": config["authorization"]["test_reads"] == 0,
        "sample_zero": config["authorization"]["sample_reads"] == 0,
        "csv_zero": config["authorization"]["submission_csv_created"] == 0,
        "upload_zero": config["authorization"]["uploads"] == 0,
    }
    if not all(checks.values()):
        raise ContractError(f"v32d config contract changed: {checks}")
    paths = {name: resolve_source(spec) for name, spec in config["source"].items()}
    source_result = json.loads(paths["result"].read_text(encoding="utf-8"))
    if source_result["status"] != "COMPLETE_INTERNAL_ONLY" or source_result["fit_count"] != 2:
        raise ContractError("v32b source is not terminal with exactly two fits")
    if source_result["hashes"]["prediction"] != config["source"]["predictions"]["sha256"]:
        raise ContractError("v32b result does not bind the frozen prediction file")
    return config, paths


def build_action_plan(
    anchor_frame: pd.DataFrame,
    probability: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    required = {*KEYS, "fold", "current_router_prediction"}
    if required.difference(anchor_frame.columns):
        raise ContractError("anchor schema is incomplete")
    if probability.shape != (len(anchor_frame),):
        raise ContractError("anchor/probability row count differs")
    if not np.isfinite(probability).all():
        raise ContractError("source probability is nonfinite")
    anchor = anchor_frame["current_router_prediction"].to_numpy(np.int8)
    if not set(np.unique(anchor)).issubset({0, 1}):
        raise ContractError("anchor is not binary")
    threshold = float(config["decoder"]["probability_threshold_inclusive"])
    full_prediction = (probability >= threshold).astype(np.int8)
    candidate = anchor.copy()
    action = np.zeros(len(anchor), dtype=bool)
    receipts: list[dict[str, Any]] = []
    for fold in config["validation"]["outer_folds"]:
        fold_mask = anchor_frame["fold"].eq(fold).to_numpy()
        fold_rows = int(fold_mask.sum())
        budget = int(
            np.floor(
                fold_rows
                * float(config["decoder"]["maximum_changed_fraction_per_fold"])
            )
        )
        eligible = np.flatnonzero(fold_mask & (full_prediction != anchor))
        ranking = anchor_frame.iloc[eligible][KEYS].copy()
        ranking["margin"] = np.abs(probability[eligible] - threshold)
        ranking["position"] = eligible
        ranking.sort_values(
            ["margin", *KEYS],
            ascending=[False, True, True, True, True],
            kind="stable",
            inplace=True,
        )
        selected = ranking["position"].to_numpy(np.int64)[:budget]
        action[selected] = True
        candidate[selected] = full_prediction[selected]
        receipts.append(
            {
                "fold": fold,
                "rows": fold_rows,
                "eligible_disagreements": int(len(eligible)),
                "budget": budget,
                "selected_changes": int(len(selected)),
                "selected_fraction": float(len(selected) / fold_rows),
                "threshold": threshold,
                "truth_reads_before_selection": 0,
            }
        )
    return candidate, action, receipts


def seal_action_plan(
    config: dict[str, Any],
    paths: dict[str, Path],
    candidate: np.ndarray,
    action: np.ndarray,
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    action_path = ARTIFACT / "sealed_action_mask.npz"
    np.savez_compressed(action_path, action=action, candidate=candidate)
    seal = {
        "schema_version": "p1.causal_cif_selective_bidirectional.action_seal.v32d",
        "experiment_id": EXPERIMENT_ID,
        "rule": config["decoder"],
        "receipts": receipts,
        "rows": int(len(candidate)),
        "changes": int(action.sum()),
        "action_sha256": sha256_array(action.astype(np.uint8)),
        "candidate_sha256": sha256_array(candidate),
        "npz_sha256": sha256_file(action_path),
        "source_prediction_sha256": sha256_file(paths["predictions"]),
        "source_anchor_sha256": sha256_file(paths["anchor"]),
        "truth_reads_before_action_seal": 0,
        "official_reads": 0,
        "hidden_truth_reads": 0,
        "test_reads": 0,
        "sample_reads": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }
    write_json_new(ARTIFACT / "action-seal.json", seal)
    return seal


def evaluate(
    frame: pd.DataFrame,
    anchor: np.ndarray,
    candidate: np.ndarray,
    action: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    truth = frame["label_base"].to_numpy(np.int8)
    folds = config["validation"]["outer_folds"]
    evaluated = frame["fold"].isin(folds).to_numpy()
    changes = evaluated & action
    additions = changes & (anchor == 0) & (candidate == 1)
    removals = changes & (anchor == 1) & (candidate == 0)
    reference_f1 = float(f1_score(truth[evaluated], anchor[evaluated]))
    candidate_f1 = float(f1_score(truth[evaluated], candidate[evaluated]))
    delta_f1 = candidate_f1 - reference_f1
    by_fold: dict[str, dict[str, Any]] = {}
    for fold in folds:
        mask = frame["fold"].eq(fold).to_numpy()
        fold_changes = changes & mask
        fold_additions = additions & mask
        fold_removals = removals & mask
        fold_reference = float(f1_score(truth[mask], anchor[mask]))
        fold_candidate = float(f1_score(truth[mask], candidate[mask]))
        by_fold[fold] = {
            "rows": int(mask.sum()),
            "reference_f1": fold_reference,
            "candidate_f1": fold_candidate,
            "delta_f1": fold_candidate - fold_reference,
            "changes": int(fold_changes.sum()),
            "changed_fraction": float(fold_changes.sum() / mask.sum()),
            "additions": int(fold_additions.sum()),
            "true_positive_additions": int((fold_additions & (truth == 1)).sum()),
            "false_positive_additions": int((fold_additions & (truth == 0)).sum()),
            "removals": int(fold_removals.sum()),
            "incumbent_true_positive_removals": int(
                (fold_removals & (truth == 1)).sum()
            ),
            "incumbent_false_positive_removals": int(
                (fold_removals & (truth == 0)).sum()
            ),
        }
    bootstrap = truth_source.base.day_bootstrap(frame, anchor, candidate, config)
    tp_additions = int((additions & (truth == 1)).sum())
    fp_additions = int((additions & (truth == 0)).sum())
    addition_count = int(additions.sum())
    addition_precision = tp_additions / addition_count if addition_count else None
    true_positive_removals = int((removals & (truth == 1)).sum())
    false_positive_removals = int((removals & (truth == 0)).sum())
    changed = frame.loc[changes, ["station", "layer", "fold"]]
    concentration_counts = changed.groupby(
        ["station", "layer", "fold"], observed=True
    ).size()
    concentration = (
        float(concentration_counts.max() / len(changed)) if len(changed) else 0.0
    )
    local_day = (
        pd.to_datetime(frame["time"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
    )
    day_frame = pd.DataFrame(
        {"day": local_day[evaluated].to_numpy(), "changed": changes[evaluated]}
    )
    day_rows = day_frame.groupby("day", observed=True).size()
    day_changes = day_frame.groupby("day", observed=True)["changed"].sum()
    maximum_day_changed_fraction = float((day_changes / day_rows).max())
    policy = config["decision_policy"]
    raw_points = delta_f1 * float(policy["score_points_per_f1"])
    calibrated_points = raw_points - float(policy["transport_penalty_points"])
    overall_changed_fraction = float(changes.sum() / evaluated.sum())
    gates = {
        "positive_changes": int(changes.sum()) > 0,
        "top_fraction_per_fold_contract": max(
            item["changed_fraction"] for item in by_fold.values()
        )
        <= float(config["safety"]["maximum_changed_fraction_per_fold"]),
        "overall_changed_fraction_at_most_0_005": overall_changed_fraction
        <= float(config["safety"]["maximum_changed_fraction_overall"]),
        "change_concentration_at_most_0_5": concentration
        <= float(
            config["safety"]["maximum_change_concentration_any_station_layer_quarter"]
        ),
        "pooled_delta_f1_positive": delta_f1
        > float(policy["pooled_delta_f1_strictly_above"]),
        "q3_q4_each_nonnegative": min(item["delta_f1"] for item in by_fold.values())
        >= float(policy["minimum_each_outer_fold_delta_f1_inclusive"]),
        "day_block_ci90_low_positive": bootstrap["ci90_low"]
        > float(policy["bootstrap_ci90_low_strictly_above"]),
        "bootstrap_probability_improved_at_least_0_8": bootstrap[
            "probability_improved"
        ]
        >= float(policy["bootstrap_probability_improved_minimum_inclusive"]),
        "raw_expected_points_at_least_0_015383691": raw_points
        >= float(policy["minimum_raw_expected_point_delta_inclusive"]),
        "calibrated_expected_points_at_least_0_01": calibrated_points
        >= float(policy["minimum_calibrated_expected_point_delta_inclusive"]),
        "incumbent_true_positive_removals_zero": true_positive_removals
        == int(policy["incumbent_true_positive_removals_required"]),
        "positive_additions": addition_count > 0,
        "addition_precision_above_incumbent_f1_half": addition_precision is not None
        and addition_precision > reference_f1 / 2.0,
    }
    return {
        "name": config["candidate"],
        "fit_count": 0,
        "reference_f1": reference_f1,
        "candidate_f1": candidate_f1,
        "delta_f1": delta_f1,
        "raw_expected_points_delta": raw_points,
        "transport_penalty_points": float(policy["transport_penalty_points"]),
        "calibrated_conservative_expected_points_delta": calibrated_points,
        "changes": int(changes.sum()),
        "changed_fraction": overall_changed_fraction,
        "additions": addition_count,
        "true_positive_additions": tp_additions,
        "false_positive_additions": fp_additions,
        "additions_precision": addition_precision,
        "anchor_f1_divided_by_two": reference_f1 / 2.0,
        "removals": int(removals.sum()),
        "incumbent_true_positive_removals": true_positive_removals,
        "incumbent_false_positive_removals": false_positive_removals,
        "maximum_change_concentration_station_layer_quarter": concentration,
        "maximum_kst_day_changed_fraction": maximum_day_changed_fraction,
        "by_fold": by_fold,
        "day_block_bootstrap": bootstrap,
        "gates": gates,
        "strict_internal_pass": bool(all(gates.values())),
    }


def execute() -> dict[str, Any]:
    if ARTIFACT.exists() or REPORT.exists():
        raise FileExistsError("v32d exactly-once namespace already exists")
    started = time.perf_counter()
    config, paths = load_contract()
    ARTIFACT.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "pid": os.getpid(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "model_fit_budget": 0,
        "retry_budget": 0,
        "official_reads": 0,
        "hidden_truth_reads": 0,
        "test_reads": 0,
        "sample_reads": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }
    write_json_new(ARTIFACT / "attempt_lock.json", lock)

    # Phase 1 is target-blind: only the frozen prediction and anchor/key metadata are read.
    anchor_frame = pd.read_parquet(paths["anchor"], columns=[*KEYS, "fold", "current_router_prediction"])
    with np.load(paths["predictions"], allow_pickle=False) as frozen:
        probability = frozen["probability"].astype(np.float32, copy=True)
    if len(anchor_frame) != 421_032 or probability.shape != (421_032,):
        raise ContractError("frozen row count changed")
    candidate, action, receipts = build_action_plan(anchor_frame, probability, config)
    action_seal = seal_action_plan(config, paths, candidate, action, receipts)

    # Phase 2 begins only after action-seal.json exists and is hashed.
    historical, candidate = truth_source.attach_truth(anchor_frame, candidate)
    historical_anchor = historical["current_router_prediction"].to_numpy(np.int8)
    if not np.array_equal(historical_anchor, anchor_frame["current_router_prediction"].to_numpy(np.int8)):
        raise ContractError("truth attachment changed anchor order")
    record = evaluate(historical, historical_anchor, candidate, action, config)
    result = {
        "schema_version": "p1.causal_cif_selective_bidirectional.result.v32d",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_INTERNAL_CANDIDATE"
        if record["strict_internal_pass"]
        else "NO_GO",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": 0,
        "pass_count": int(record["strict_internal_pass"]),
        "candidate": record,
        "action_seal": action_seal,
        "operations": {
            "official_reads": 0,
            "hidden_truth_reads": 0,
            "test_reads": 0,
            "sample_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
            "retries": 0,
        },
        "hashes": {
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "attempt_lock_sha256": sha256_file(ARTIFACT / "attempt_lock.json"),
            "action_seal_sha256": sha256_file(ARTIFACT / "action-seal.json"),
            "sealed_action_mask_sha256": sha256_file(
                ARTIFACT / "sealed_action_mask.npz"
            ),
        },
    }
    write_json_new(ARTIFACT / "result.json", result)
    write_json_new(REPORT / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-metric-only", action="store_true")
    args = parser.parse_args()
    if not args.execute_metric_only:
        raise SystemExit("--execute-metric-only required")
    try:
        result = execute()
    except Exception as exc:
        failure = {
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "official_reads": 0,
            "hidden_truth_reads": 0,
            "test_reads": 0,
            "sample_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        }
        if ARTIFACT.exists() and not (ARTIFACT / "terminal_failure.json").exists():
            write_json_new(ARTIFACT / "terminal_failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
