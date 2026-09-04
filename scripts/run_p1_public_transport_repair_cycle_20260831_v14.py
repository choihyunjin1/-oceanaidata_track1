"""Zero-fit target-blind vertical-bracket falsification for P1."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v13 as base  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v14"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
FOLDS = base.FOLDS


class ContractError(RuntimeError):
    """Frozen vertical-bracket contract violation."""


def load_contract() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    calibration = json.loads(base.CALIBRATION_PATH.read_text(encoding="utf-8"))
    family = config["transport_family"]
    policy = config["decision_policy"]
    observed = next(
        item for item in calibration["observed_pairs"] if item["family_id"] == family["family_id"]
    )
    candidate = config["candidate"]
    checks = {
        "calibration_sha": family["calibration_sha256"] == base.sha256_file(base.CALIBRATION_PATH),
        "fixed_family": family["family_id"] == "P1_FIXED_ADD_ONLY_UNION",
        "low_dof": family["tier_id"] == "LOW_DOF_FIXED",
        "fixed_output": family["representation_changed"] is False and family["routing_discontinuous"] is False,
        "penalty": np.isclose(policy["transport_penalty_points"], observed["adverse_penalty_points"], atol=1e-15),
        "raw": np.isclose(
            policy["minimum_raw_expected_point_delta_inclusive"],
            observed["adverse_penalty_points"] + calibration["minimum_calibrated_expected_points_delta"],
            atol=1e-15,
        ),
        "exact_step": candidate["exact_layer_rank_step"] == 1,
        "known_ranks": candidate["known_physical_depth_ranks"] == list(range(1, 9)),
        "zero_fit": config["fit_budget"]["maximum"] == 0,
        "no_trimming": config["safety"]["proposal_caps_are_gates_not_trimming"] is True,
    }
    if not all(checks.values()):
        raise ContractError(f"v14 contract mismatch: {checks}")
    return config


def vertical_bracket_additions(anchor_frame: pd.DataFrame, *, known_ranks: tuple[int, ...]) -> np.ndarray:
    required = {"station", "year", "layer", "time", "current_router_prediction"}
    if not required.issubset(anchor_frame.columns):
        raise ContractError("anchor frame lacks bracket keys")
    work = pd.DataFrame(
        {
            "station": anchor_frame["station"].astype(str),
            "year": anchor_frame["year"].astype(int),
            "layer": anchor_frame["layer"].astype(int),
            "time": pd.to_datetime(anchor_frame["time"], utc=True),
            "anchor": anchor_frame["current_router_prediction"].to_numpy(np.int8),
            "position": np.arange(len(anchor_frame), dtype=np.int64),
        }
    )
    additions = np.zeros(len(anchor_frame), dtype=bool)
    known = set(known_ranks)
    for _, group in work.groupby(["station", "year", "time"], sort=False, observed=True):
        layers = group["layer"].to_numpy(np.int64)
        if len(group) < 3 or len(np.unique(layers)) != len(layers) or any(int(x) not in known for x in layers):
            continue
        ordered = group.sort_values(["layer", "position"], kind="stable")
        ordered_layers = ordered["layer"].to_numpy(np.int64)
        if np.any(np.diff(ordered_layers) <= 0):
            continue
        by_layer = {int(row.layer): row for row in ordered.itertuples(index=False)}
        for layer, row in by_layer.items():
            shallower = by_layer.get(layer - 1)
            deeper = by_layer.get(layer + 1)
            if shallower is None or deeper is None or int(row.anchor) != 0:
                continue
            if int(shallower.anchor) == 1 and int(deeper.anchor) == 1:
                additions[int(row.position)] = True
    return additions


def seal_proposal(config: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    anchor_frame = pd.read_parquet(base.ANCHOR_PATH)
    if len(anchor_frame) != 421_032 or anchor_frame.duplicated(["station", "year", "layer", "time"]).any():
        raise ContractError("anchor key contract changed")
    anchor = anchor_frame["current_router_prediction"].to_numpy(np.int8)
    ranks = tuple(int(value) for value in config["candidate"]["known_physical_depth_ranks"])
    additions = vertical_bracket_additions(anchor_frame, known_ranks=ranks)
    candidate = np.maximum(anchor, additions.astype(np.int8))
    path = ARTIFACT / "proposal_blind.npz"
    np.savez_compressed(path, additions=additions, candidate=candidate)
    receipt = {
        "schema_version": "p1.v14.target_blind_proposal_seal",
        "candidate": config["candidate"]["name"],
        "rows": len(anchor_frame),
        "additions": int(additions.sum()),
        "additions_sha256": base.sha256_bool(additions),
        "candidate_sha256": base.sha256_bool(candidate),
        "npz_sha256": base.sha256_file(path),
        "target_columns_read_before_seal": 0,
        "raw_feature_columns_read": 0,
        "official_covariate_reads": 0,
    }
    base.write_json(ARTIFACT / "proposal_seal.json", receipt)
    return anchor_frame, additions, candidate, receipt


def evaluate(frame: pd.DataFrame, candidate: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    truth = frame["label_base"].to_numpy(np.int8)
    anchor = frame["current_router_prediction"].to_numpy(np.int8)
    evaluated = frame["fold"].isin(FOLDS[1:]).to_numpy()
    additions = evaluated & (candidate == 1) & (anchor == 0)
    removals = evaluated & (candidate == 0) & (anchor == 1)
    reference = float(f1_score(truth[evaluated], anchor[evaluated]))
    value = float(f1_score(truth[evaluated], candidate[evaluated]))
    delta = value - reference
    by_fold: dict[str, dict[str, Any]] = {}
    fold_precision_gates: dict[str, bool] = {}
    for fold in FOLDS:
        mask = frame["fold"].eq(fold).to_numpy()
        fold_add = mask & (candidate == 1) & (anchor == 0)
        base_score = float(f1_score(truth[mask], anchor[mask]))
        candidate_score = float(f1_score(truth[mask], candidate[mask]))
        tp = int((fold_add & (truth == 1)).sum())
        fp = int((fold_add & (truth == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else None
        by_fold[fold] = {
            "rows": int(mask.sum()),
            "reference_f1": base_score,
            "candidate_f1": candidate_score,
            "delta_f1": candidate_score - base_score,
            "additions": tp + fp,
            "true_positive_additions": tp,
            "false_positive_additions": fp,
            "additions_precision": precision,
            "anchor_f1_divided_by_2": base_score / 2.0,
            "anchor_removals": int((mask & removals).sum()),
        }
        if fold in FOLDS[1:]:
            fold_precision_gates[fold] = precision is not None and precision > base_score / 2.0
    bootstrap = base.base.day_bootstrap(
        frame,
        anchor,
        candidate,
        {
            "validation": {
                "bootstrap_replicates": config["validation"]["bootstrap_replicates"],
                "bootstrap_seed": config["validation"]["bootstrap_seed"],
            }
        },
    )
    tp = int((additions & (truth == 1)).sum())
    fp = int((additions & (truth == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    precision_lcb = (
        float(beta_distribution.ppf(float(config["safety"]["precision_lcb_quantile"]), tp, fp + 1))
        if tp
        else 0.0
    )
    local_day = pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    outer = frame.loc[evaluated, ["station", "layer", "fold"]].copy()
    outer["day"] = local_day[evaluated].to_numpy()
    outer["addition"] = additions[evaluated]
    day_rows = outer.groupby("day", observed=True).size()
    day_additions = outer.groupby("day", observed=True)["addition"].sum()
    maximum_day_changed_fraction = float((day_additions / day_rows).max()) if len(day_rows) else 0.0
    changed = outer.loc[outer["addition"], ["station", "layer", "fold"]]
    concentration = changed.groupby(["station", "layer", "fold"], observed=True).size()
    maximum_concentration = float(concentration.max() / len(changed)) if len(changed) else 0.0
    station_layer: dict[str, float] = {}
    evaluated_positions = np.flatnonzero(evaluated)
    local = frame.loc[evaluated, ["station", "layer"]].reset_index(drop=True)
    for (station, layer), positions in local.groupby(["station", "layer"], observed=True).indices.items():
        index = evaluated_positions[np.asarray(positions, dtype=np.int64)]
        station_layer[f"{station}|{layer}"] = float(
            f1_score(truth[index], candidate[index]) - f1_score(truth[index], anchor[index])
        )
    policy = config["decision_policy"]
    raw_points = delta * float(policy["score_points_per_f1"])
    calibrated = raw_points - float(policy["transport_penalty_points"])
    gates = {
        "positive_additions": int(additions.sum()) > 0,
        "anchor_removals_zero": int(removals.sum()) == 0,
        "q3_q4_each_nonnegative": min(by_fold[fold]["delta_f1"] for fold in FOLDS[1:]) >= 0.0,
        "pooled_delta_positive": delta > 0.0,
        "day_block_ci90_low_strictly_positive": bootstrap["ci90_low"] > 0.0,
        "each_confirmatory_fold_precision_above_anchor_f1_half": all(fold_precision_gates.values()),
        "raw_expected_points_at_least_0_015383691": raw_points
        >= float(policy["minimum_raw_expected_point_delta_inclusive"]),
        "calibrated_expected_points_at_least_0_01": calibrated
        >= float(policy["minimum_calibrated_expected_point_delta_inclusive"]),
        "changed_fraction_at_most_0_005": float(additions.sum() / evaluated.sum())
        <= float(config["safety"]["maximum_changed_fraction"]),
        "each_kst_day_changed_fraction_at_most_0_005": maximum_day_changed_fraction
        <= float(config["safety"]["maximum_changed_fraction_any_kst_day"]),
        "station_layer_quarter_concentration_at_most_0_5": maximum_concentration
        <= float(config["safety"]["maximum_addition_concentration_any_station_layer_quarter"]),
        "each_supported_station_layer_nonnegative": min(station_layer.values())
        >= float(config["safety"]["minimum_each_supported_station_layer_delta_f1"]),
    }
    return {
        "name": config["candidate"]["name"],
        "fit_count": 0,
        "reference_f1": reference,
        "candidate_f1": value,
        "delta_f1": delta,
        "raw_expected_points_delta": raw_points,
        "transport_penalty_points": float(policy["transport_penalty_points"]),
        "calibrated_conservative_expected_points_delta": calibrated,
        "additions": int(additions.sum()),
        "true_positive_additions": tp,
        "false_positive_additions": fp,
        "additions_precision": precision,
        "additions_precision_lcb90": precision_lcb,
        "anchor_f1_divided_by_2": reference / 2.0,
        "anchor_removals": int(removals.sum()),
        "changed_fraction": float(additions.sum() / evaluated.sum()),
        "maximum_kst_day_changed_fraction": maximum_day_changed_fraction,
        "maximum_addition_concentration_station_layer_quarter": maximum_concentration,
        "station_layer_delta_f1": station_layer,
        "by_fold": by_fold,
        "fold_precision_gates": fold_precision_gates,
        "day_block_bootstrap": bootstrap,
        "gates": gates,
        "strict_internal_pass": bool(all(gates.values())),
    }


def independent_qa(result: dict[str, Any]) -> dict[str, Any]:
    candidate = result["candidate"]
    checks = {
        "one_candidate": result["candidate_count"] == 1,
        "zero_fits": result["fit_count"] == 0,
        "proposal_sealed_before_target": result["proposal_seal"]["target_columns_read_before_seal"] == 0,
        "anchor_bits_only": result["proposal_seal"]["raw_feature_columns_read"] == 0,
        "exact_step_one": result["candidate_contract"]["exact_layer_rank_step"] == 1,
        "no_trimming": result["safety_contract"]["proposal_caps_are_gates_not_trimming"],
        "add_only": candidate["anchor_removals"] == 0,
        "calibration_hash": result["transport_family"]["calibration_sha256"]
        == result["hashes"]["root_calibration_sha256"],
        "official_reads_zero": result["operations"]["official_covariate_reads"] == 0,
        "hidden_reads_zero": result["operations"]["hidden_truth_reads"] == 0,
        "csv_zero": result["operations"]["submission_csv_created"] == 0,
        "uploads_zero": result["operations"]["uploads"] == 0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def validate_only() -> dict[str, Any]:
    load_contract()
    for path in (base.CALIBRATION_PATH, base.AUTHORITATIVE_PATH, base.ANCHOR_PATH):
        if not path.is_file():
            raise ContractError(f"missing frozen input: {path}")
    return {
        "status": "VALID",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": base.sha256_file(CONFIG_PATH),
        "runner_sha256": base.sha256_file(Path(__file__)),
        "candidate_count": 1,
        "fit_budget": 0,
    }


def execute() -> dict[str, Any]:
    if ARTIFACT.exists():
        raise FileExistsError("exactly-once artifact path exists")
    config = load_contract()
    ARTIFACT.mkdir(parents=True)
    started = time.perf_counter()
    base.write_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "pid": os.getpid(),
            "config_sha256": base.sha256_file(CONFIG_PATH),
            "runner_sha256": base.sha256_file(Path(__file__)),
            "fit_budget": 0,
        },
    )
    anchor_frame, _additions, candidate, proposal_seal = seal_proposal(config)
    base.write_json(
        ARTIFACT / "progress.json",
        {"phase": "proposal_sealed_target_reads_zero", "fit_count": 0},
        exclusive=False,
    )
    frame, candidate = base.attach_truth(anchor_frame, candidate)
    record = evaluate(frame, candidate, config)
    result: dict[str, Any] = {
        "schema_version": "p1.public_transport_repair_cycle.20260831.v14.result",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_INTERNAL_ONLY",
        "runtime_seconds": time.perf_counter() - started,
        "transport_family": config["transport_family"],
        "candidate_contract": config["candidate"],
        "safety_contract": config["safety"],
        "decision_policy": config["decision_policy"],
        "candidate_count": 1,
        "pass_count": int(record["strict_internal_pass"]),
        "fit_count": 0,
        "candidate": record,
        "outputs": [],
        "proposal_seal": proposal_seal,
        "adaptive_surface_disclaimer": "This is newly preregistered development evidence on reused historical Q3/Q4 surfaces, not an independent confirmation.",
        "operations": {
            "official_covariate_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "hashes": {
            "config_sha256": base.sha256_file(CONFIG_PATH),
            "runner_sha256": base.sha256_file(Path(__file__)),
            "root_calibration_sha256": base.sha256_file(base.CALIBRATION_PATH),
            "authoritative_results_sha256": base.sha256_file(base.AUTHORITATIVE_PATH),
            "anchor_sha256": base.sha256_file(base.ANCHOR_PATH),
        },
    }
    result["independent_qa"] = independent_qa(result)
    base.write_json(ARTIFACT / "result.json", result)
    base.write_json(REPORT / "independent-qa.json", result["independent_qa"])
    base.write_json(
        ARTIFACT / "progress.json",
        {"phase": "terminal", "fit_count": 0, "pass_count": result["pass_count"]},
        exclusive=False,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.validate_only == args.execute:
        parser.error("choose exactly one mode")
    try:
        payload = validate_only() if args.validate_only else execute()
        print(json.dumps(base.native(payload), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        payload = {
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "official_covariate_reads": 0,
            "hidden_truth_reads": 0,
            "uploads": 0,
        }
        if ARTIFACT.exists() and not (ARTIFACT / "terminal_failure.json").exists():
            base.write_json(ARTIFACT / "terminal_failure.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
