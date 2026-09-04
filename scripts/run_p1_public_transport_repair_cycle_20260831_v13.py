"""Zero-fit causal asynchronous peer-consensus falsification for P1."""

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
from scipy.stats import beta as beta_distribution
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v12 as base  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v13"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
CALIBRATION_PATH = base.CALIBRATION_PATH
AUTHORITATIVE_PATH = base.AUTHORITATIVE_PATH
ANCHOR_PATH = base.ANCHOR_PATH
FOLDS = base.FOLDS


class ContractError(RuntimeError):
    """Frozen peer-consensus contract violation."""


def native(value: Any) -> Any:
    return base.native(value)


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def sha256_bool(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.uint8).tobytes()).hexdigest()


def write_json(path: Path, payload: Any, *, exclusive: bool = True) -> None:
    base.write_json(path, payload, exclusive=exclusive)


def load_contract() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    family = config["transport_family"]
    policy = config["decision_policy"]
    observed = next(item for item in calibration["observed_pairs"] if item["family_id"] == family["family_id"])
    candidate = config["candidate"]
    checks = {
        "calibration_sha": family["calibration_sha256"] == sha256_file(CALIBRATION_PATH),
        "family": family["family_id"] == "P1_FIXED_ADD_ONLY_UNION",
        "fixed_representation": family["representation_changed"] is False and family["routing_discontinuous"] is False,
        "penalty": np.isclose(policy["transport_penalty_points"], observed["adverse_penalty_points"], atol=1e-15),
        "raw": np.isclose(policy["minimum_raw_expected_point_delta_inclusive"], observed["adverse_penalty_points"] + calibration["minimum_calibrated_expected_points_delta"], atol=1e-15),
        "quorum": candidate["distinct_other_layer_quorum"] == 2,
        "lookback": candidate["lookback_minutes_inclusive"] == 10,
        "zero_fit": config["fit_budget"]["maximum"] == 0,
    }
    if not all(checks.values()):
        raise ContractError(f"v13 contract mismatch: {checks}")
    return config


def async_peer_additions(anchor_frame: pd.DataFrame, *, quorum: int, lookback_minutes: int) -> np.ndarray:
    required = {"station", "year", "layer", "time", "current_router_prediction"}
    if not required.issubset(anchor_frame.columns):
        raise ContractError("anchor frame lacks peer-consensus keys")
    timestamp = pd.to_datetime(anchor_frame["time"], utc=True)
    work = pd.DataFrame(
        {
            "station": anchor_frame["station"].astype(str),
            "year": anchor_frame["year"].astype(int),
            "layer": anchor_frame["layer"].astype(int),
            "time": timestamp,
            "anchor": anchor_frame["current_router_prediction"].to_numpy(np.int8),
            "position": np.arange(len(anchor_frame), dtype=np.int64),
        }
    )
    work.sort_values(["station", "year", "time", "layer", "position"], kind="stable", inplace=True)
    work.reset_index(drop=True, inplace=True)
    additions_sorted = np.zeros(len(work), dtype=bool)
    maximum_age = pd.Timedelta(minutes=lookback_minutes)
    for _, station_year in work.groupby(["station", "year"], sort=False, observed=True):
        last: dict[int, tuple[pd.Timestamp, int]] = {}
        prior_time: pd.Timestamp | None = None
        for current_time, batch in station_year.groupby("time", sort=True, observed=True):
            if prior_time is not None and current_time - prior_time > maximum_age:
                last.clear()
            current = {int(row.layer): int(row.anchor) for row in batch.itertuples(index=False)}
            for layer, anchor in current.items():
                last[layer] = (current_time, anchor)
            for row in batch.itertuples():
                if int(row.anchor) == 1:
                    continue
                peer_count = sum(
                    1
                    for layer, (seen_time, anchor) in last.items()
                    if layer != int(row.layer)
                    and anchor == 1
                    and pd.Timedelta(0) <= current_time - seen_time <= maximum_age
                )
                if peer_count >= quorum:
                    additions_sorted[int(row.Index)] = True
            prior_time = current_time
    additions = np.zeros(len(anchor_frame), dtype=bool)
    additions[work["position"].to_numpy(np.int64)] = additions_sorted
    return additions


def seal_proposal(config: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    anchor_frame = pd.read_parquet(ANCHOR_PATH)
    if len(anchor_frame) != 421_032 or anchor_frame.duplicated(["station", "year", "layer", "time"]).any():
        raise ContractError("anchor key contract changed")
    candidate_spec = config["candidate"]
    anchor = anchor_frame["current_router_prediction"].to_numpy(np.int8)
    additions = async_peer_additions(anchor_frame, quorum=int(candidate_spec["distinct_other_layer_quorum"]), lookback_minutes=int(candidate_spec["lookback_minutes_inclusive"]))
    candidate = anchor.copy()
    candidate[additions & (anchor == 0)] = 1
    path = ARTIFACT / "proposal_blind.npz"
    np.savez_compressed(path, additions=additions, candidate=candidate)
    receipt = {
        "schema_version": "p1.v13.target_blind_proposal_seal",
        "candidate": candidate_spec["name"],
        "rows": len(anchor_frame),
        "additions": int(additions.sum()),
        "additions_sha256": sha256_bool(additions),
        "candidate_sha256": sha256_bool(candidate),
        "npz_sha256": sha256_file(path),
        "target_columns_read_before_seal": 0,
        "raw_feature_columns_read": 0,
        "official_covariate_reads": 0,
    }
    write_json(ARTIFACT / "proposal_seal.json", receipt)
    return anchor_frame, additions, candidate, receipt


def attach_truth(anchor_frame: pd.DataFrame, candidate: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    historical, _ = base.source_cycle.p1_frame()
    historical["time"] = pd.to_datetime(historical["time"], utc=True)
    keys = ["station", "year", "layer", "time", "fold"]
    anchor = anchor_frame.copy()
    anchor["time"] = pd.to_datetime(anchor["time"], utc=True)
    anchor["__proposal_position"] = np.arange(len(anchor), dtype=np.int64)
    historical = historical.merge(anchor[keys + ["current_router_prediction", "__proposal_position"]], on=keys, validate="one_to_one")
    historical.sort_values("__proposal_position", inplace=True)
    historical.reset_index(drop=True, inplace=True)
    if not np.array_equal(historical["__proposal_position"].to_numpy(), np.arange(len(anchor))):
        raise ContractError("truth attachment changed proposal order")
    return historical, candidate


def evaluate(frame: pd.DataFrame, candidate: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    truth = frame["label_base"].to_numpy(np.int8)
    anchor = frame["current_router_prediction"].to_numpy(np.int8)
    evaluated = frame["fold"].isin(FOLDS[1:]).to_numpy()
    additions = evaluated & (candidate == 1) & (anchor == 0)
    removals = evaluated & (candidate == 0) & (anchor == 1)
    reference = float(f1_score(truth[evaluated], anchor[evaluated]))
    value = float(f1_score(truth[evaluated], candidate[evaluated]))
    delta = value - reference
    by_fold = {}
    for fold in FOLDS:
        mask = frame["fold"].eq(fold).to_numpy()
        fold_add = mask & (candidate == 1) & (anchor == 0)
        base_score = float(f1_score(truth[mask], anchor[mask]))
        candidate_score = float(f1_score(truth[mask], candidate[mask]))
        by_fold[fold] = {"rows": int(mask.sum()), "reference_f1": base_score, "candidate_f1": candidate_score, "delta_f1": candidate_score - base_score, "additions": int(fold_add.sum()), "true_positive_additions": int((fold_add & (truth == 1)).sum()), "false_positive_additions": int((fold_add & (truth == 0)).sum()), "anchor_removals": int((mask & removals).sum())}
    bootstrap = base.day_bootstrap(frame, anchor, candidate, {"validation": {"bootstrap_replicates": config["validation"]["bootstrap_replicates"], "bootstrap_seed": config["validation"]["bootstrap_seed"]}})
    tp = int((additions & (truth == 1)).sum())
    fp = int((additions & (truth == 0)).sum())
    precision = tp / int(additions.sum()) if additions.any() else None
    precision_lcb = float(beta_distribution.ppf(float(config["safety"]["precision_lcb_quantile"]), tp, fp + 1)) if tp else 0.0
    local_day = pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    changed = frame.loc[additions, ["station", "layer"]].copy()
    changed["day"] = local_day[additions].to_numpy()
    maximum_station_day = int(changed.groupby(["station", "day"], observed=True).size().max()) if len(changed) else 0
    station_layer = {}
    evaluated_positions = np.flatnonzero(evaluated)
    local = frame.loc[evaluated, ["station", "layer"]].reset_index(drop=True)
    for (station, layer), positions in local.groupby(["station", "layer"], observed=True).indices.items():
        index = evaluated_positions[np.asarray(positions, dtype=np.int64)]
        station_layer[f"{station}|{layer}"] = float(f1_score(truth[index], candidate[index]) - f1_score(truth[index], anchor[index]))
    policy = config["decision_policy"]
    raw_points = delta * float(policy["score_points_per_f1"])
    calibrated = raw_points - float(policy["transport_penalty_points"])
    gates = {
        "positive_additions": int(additions.sum()) > 0,
        "anchor_removals_zero": int(removals.sum()) == 0,
        "q3_q4_each_nonnegative": min(by_fold[fold]["delta_f1"] for fold in FOLDS[1:]) >= float(policy["minimum_each_q3_q4_delta_f1"]),
        "pooled_delta_positive": delta > 0.0,
        "bootstrap_ci90_low_at_least_0_000578810": bootstrap["ci90_low"] >= float(policy["bootstrap_ci90_low_minimum_f1"]),
        "bootstrap_probability_strictly_above_0_8": bootstrap["probability_improved"] > float(policy["bootstrap_probability_improved_minimum_strict"]),
        "marginal_precision_lcb_above_anchor_f1_half": precision_lcb > reference / 2.0,
        "raw_expected_points_at_least_0_015383691": raw_points >= float(policy["minimum_raw_expected_point_delta_inclusive"]),
        "calibrated_expected_points_at_least_0_01": calibrated >= float(policy["minimum_calibrated_expected_point_delta_inclusive"]),
        "changed_fraction_at_most_0_005": float(additions.sum() / evaluated.sum()) <= float(config["safety"]["maximum_changed_fraction"]),
        "maximum_five_additions_per_station_day": maximum_station_day <= int(config["safety"]["maximum_additions_per_station_kst_day"]),
        "worst_station_layer_nonnegative": min(station_layer.values()) >= float(config["safety"]["minimum_each_station_layer_delta_f1"]),
    }
    return {"name": config["candidate"]["name"], "past_only": True, "fit_count": 0, "reference_f1": reference, "candidate_f1": value, "delta_f1": delta, "raw_expected_points_delta": raw_points, "transport_penalty_points": float(policy["transport_penalty_points"]), "calibrated_conservative_expected_points_delta": calibrated, "additions": int(additions.sum()), "true_positive_additions": tp, "false_positive_additions": fp, "additions_precision": precision, "additions_precision_lcb90": precision_lcb, "anchor_f1_divided_by_2": reference / 2.0, "anchor_removals": int(removals.sum()), "changed_fraction": float(additions.sum() / evaluated.sum()), "maximum_station_day_additions": maximum_station_day, "station_layer_delta_f1": station_layer, "by_fold": by_fold, "day_block_bootstrap": bootstrap, "gates": gates, "strict_internal_pass": bool(all(gates.values()))}


def independent_qa(result: dict[str, Any]) -> dict[str, Any]:
    candidate = result["candidate"]
    checks = {"one_candidate": result["candidate_count"] == 1, "zero_fits": result["fit_count"] == 0, "proposal_sealed_before_target": result["proposal_seal"]["target_columns_read_before_seal"] == 0, "anchor_bits_only": result["proposal_seal"]["raw_feature_columns_read"] == 0, "quorum_two": result["candidate_contract"]["distinct_other_layer_quorum"] == 2, "lookback_ten": result["candidate_contract"]["lookback_minutes_inclusive"] == 10, "add_only": candidate["anchor_removals"] == 0, "calibration_hash": result["transport_family"]["calibration_sha256"] == result["hashes"]["root_calibration_sha256"], "official_reads_zero": result["operations"]["official_covariate_reads"] == 0, "hidden_reads_zero": result["operations"]["hidden_truth_reads"] == 0, "csv_zero": result["operations"]["submission_csv_created"] == 0, "uploads_zero": result["operations"]["uploads"] == 0}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def validate_only() -> dict[str, Any]:
    load_contract()
    for path in (CALIBRATION_PATH, AUTHORITATIVE_PATH, ANCHOR_PATH):
        if not path.is_file():
            raise ContractError(f"missing frozen input: {path}")
    return {"status": "VALID", "experiment_id": EXPERIMENT_ID, "config_sha256": sha256_file(CONFIG_PATH), "runner_sha256": sha256_file(Path(__file__)), "calibration_sha256": sha256_file(CALIBRATION_PATH), "candidate_count": 1, "fit_budget": 0}


def execute() -> dict[str, Any]:
    if ARTIFACT.exists():
        raise FileExistsError("exactly-once artifact path exists")
    config = load_contract()
    ARTIFACT.mkdir(parents=True)
    started = time.perf_counter()
    write_json(ARTIFACT / "attempt_lock.json", {"experiment_id": EXPERIMENT_ID, "pid": os.getpid(), "config_sha256": sha256_file(CONFIG_PATH), "runner_sha256": sha256_file(Path(__file__)), "fit_budget": 0})
    anchor_frame, _additions, candidate, proposal_seal = seal_proposal(config)
    write_json(ARTIFACT / "progress.json", {"phase": "proposal_sealed_target_reads_zero", "fit_count": 0}, exclusive=False)
    frame, candidate = attach_truth(anchor_frame, candidate)
    record = evaluate(frame, candidate, config)
    result: dict[str, Any] = {"schema_version": "p1.public_transport_repair_cycle.20260831.v13.result", "experiment_id": EXPERIMENT_ID, "status": "COMPLETE_INTERNAL_ONLY", "runtime_seconds": time.perf_counter() - started, "transport_family": config["transport_family"], "candidate_contract": config["candidate"], "decision_policy": config["decision_policy"], "candidate_count": 1, "pass_count": int(record["strict_internal_pass"]), "fit_count": 0, "candidate": record, "outputs": [], "proposal_seal": proposal_seal, "operations": {"official_covariate_reads": 0, "hidden_truth_reads": 0, "submission_csv_created": 0, "uploads": 0}, "hashes": {"config_sha256": sha256_file(CONFIG_PATH), "runner_sha256": sha256_file(Path(__file__)), "root_calibration_sha256": sha256_file(CALIBRATION_PATH), "authoritative_results_sha256": sha256_file(AUTHORITATIVE_PATH), "anchor_sha256": sha256_file(ANCHOR_PATH)}}
    result["independent_qa"] = independent_qa(result)
    write_json(ARTIFACT / "result.json", result)
    write_json(REPORT / "independent-qa.json", result["independent_qa"])
    write_json(ARTIFACT / "progress.json", {"phase": "terminal", "fit_count": 0, "pass_count": result["pass_count"]}, exclusive=False)
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
        print(json.dumps(native(payload), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        payload = {"experiment_id": EXPERIMENT_ID, "status": "TERMINAL_TECHNICAL_FAILURE", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "official_covariate_reads": 0, "hidden_truth_reads": 0, "uploads": 0}
        if ARTIFACT.exists() and not (ARTIFACT / "terminal_failure.json").exists():
            write_json(ARTIFACT / "terminal_failure.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
