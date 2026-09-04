"""Zero-fit evaluation of two preregistered fixed P1 add-only masks."""

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

import run_full_internal_submission_cycle_20260831_v2 as source_cycle  # noqa: E402
import run_p1_parallel_candidate_cycle_20260831_v4 as feature_cycle  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v12"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
CALIBRATION_PATH = ROOT / "reports/public_transport_calibration_20260831_v2/calibration.json"
AUTHORITATIVE_PATH = ROOT / "reports/parallel_internal_pass_registry_20260831_v1/official-submission-results-20260831.json"
ANCHOR_PATH = ROOT / "artifacts/p1_current_router_oof_anchor_v1/anchor.parquet"
FOLDS = ("2025_q2", "2025_q3", "2025_q4")


class ContractError(RuntimeError):
    """Frozen rule or access contract violation."""


def native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        json.dump(native(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_contract() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    policy = config["decision_policy"]
    family = config["transport_family"]
    observed = next(item for item in calibration["observed_pairs"] if item["family_id"] == family["family_id"])
    checks = {
        "calibration_sha": family["calibration_sha256"] == sha256_file(CALIBRATION_PATH),
        "family": family["family_id"] == "P1_FIXED_ADD_ONLY_UNION",
        "representation": family["representation_changed"] is False,
        "routing": family["routing_discontinuous"] is False,
        "penalty": np.isclose(policy["transport_penalty_points"], observed["adverse_penalty_points"], atol=1e-15),
        "raw": np.isclose(policy["minimum_raw_expected_point_delta_inclusive"], observed["adverse_penalty_points"] + calibration["minimum_calibrated_expected_points_delta"], atol=1e-15),
    }
    if not all(checks.values()):
        raise ContractError(f"exact fixed-union calibration mismatch: {checks}")
    return config


def load_frame() -> pd.DataFrame:
    frame, _ = source_cycle.p1_frame()
    frame, _ = feature_cycle.add_causal_features(frame)
    anchor = pd.read_parquet(ANCHOR_PATH)
    anchor["time"] = pd.to_datetime(anchor["time"], utc=True)
    keys = ["station", "year", "layer", "time", "fold"]
    frame = frame.merge(anchor, on=keys, validate="one_to_one")
    if len(frame) != 421_032:
        raise ContractError("historical frame row contract changed")
    return frame


def causal_lag1_mask(frame: pd.DataFrame, anchor: np.ndarray) -> np.ndarray:
    timestamp = pd.to_datetime(frame["time"], utc=True)
    work = pd.DataFrame({"station": frame["station"].astype(str), "year": frame["year"].astype(int), "layer": frame["layer"].astype(int), "time": timestamp, "position": np.arange(len(frame)), "anchor": anchor})
    work.sort_values(["station", "year", "layer", "time", "position"], kind="stable", inplace=True)
    grouped = work.groupby(["station", "year", "layer"], sort=False, observed=True)
    prior_anchor = grouped["anchor"].shift(1).fillna(0).to_numpy(np.int8)
    exact_cadence = grouped["time"].diff().eq(pd.Timedelta(minutes=10)).to_numpy()
    additions_sorted = (prior_anchor == 1) & exact_cadence & work["anchor"].eq(0).to_numpy()
    additions = np.zeros(len(frame), dtype=bool)
    additions[work["position"].to_numpy(np.int64)] = additions_sorted
    return additions


def candidate_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    anchor = frame["current_router_prediction"].to_numpy(np.int8)
    candidate = anchor.copy()
    candidate[causal_lag1_mask(frame, anchor)] = 1
    return {
        "P1_1_ONE_ROW_TRAILING_EDGE_DILATION_ADD_ONLY": candidate,
    }


def f1_counts(truth: np.ndarray, prediction: np.ndarray) -> tuple[int, int, int]:
    return (int(((truth == 1) & (prediction == 1)).sum()), int(((truth == 0) & (prediction == 1)).sum()), int(((truth == 1) & (prediction == 0)).sum()))


def f1_from_counts(values: np.ndarray) -> float:
    tp, fp, fn = values
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def day_bootstrap(frame: pd.DataFrame, anchor: np.ndarray, candidate: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    truth = frame["label_base"].to_numpy(np.int8)
    evaluated = frame["fold"].isin(FOLDS[1:]).to_numpy()
    local_day = pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    table = pd.DataFrame({"fold": frame["fold"], "day": local_day, "truth": truth, "anchor": anchor, "candidate": candidate, "evaluated": evaluated})
    blocks = []
    for _, group in table.loc[table.evaluated].groupby(["fold", "day"], sort=True):
        y = group.truth.to_numpy(np.int8)
        blocks.append((*f1_counts(y, group.anchor.to_numpy(np.int8)), *f1_counts(y, group.candidate.to_numpy(np.int8))))
    counts = np.asarray(blocks, dtype=np.int64)
    rng = np.random.default_rng(int(config["validation"]["bootstrap_seed"]))
    delta = np.empty(int(config["validation"]["bootstrap_replicates"]), dtype=float)
    for index in range(len(delta)):
        total = counts[rng.integers(0, len(counts), size=len(counts))].sum(axis=0)
        delta[index] = f1_from_counts(total[3:6]) - f1_from_counts(total[0:3])
    return {"blocks": len(blocks), "replicates": len(delta), "mean_delta_f1": float(delta.mean()), "ci90_low": float(np.quantile(delta, 0.05)), "ci90_high": float(np.quantile(delta, 0.95)), "probability_improved": float(np.mean(delta > 0.0))}


def evaluate(name: str, frame: pd.DataFrame, candidate: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
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
        base = float(f1_score(truth[mask], anchor[mask]))
        score = float(f1_score(truth[mask], candidate[mask]))
        fold_add = mask & (candidate == 1) & (anchor == 0)
        by_fold[fold] = {"rows": int(mask.sum()), "reference_f1": base, "candidate_f1": score, "delta_f1": score - base, "additions": int(fold_add.sum()), "true_positive_additions": int((fold_add & (truth == 1)).sum()), "false_positive_additions": int((fold_add & (truth == 0)).sum()), "anchor_removals": int((mask & removals).sum())}
    bootstrap = day_bootstrap(frame, anchor, candidate, config)
    local_day = pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    addition_frame = frame.loc[additions, ["station", "layer"]].copy()
    addition_frame["day"] = local_day[additions].to_numpy()
    max_station_layer_day_additions = int(addition_frame.groupby(["station", "layer", "day"], observed=True).size().max()) if len(addition_frame) else 0
    station_layer = {}
    for (station, layer), positions in frame.loc[evaluated].groupby(["station", "layer"], observed=True).indices.items():
        index = np.asarray(positions, dtype=np.int64)
        global_index = np.flatnonzero(evaluated)[index]
        base = float(f1_score(truth[global_index], anchor[global_index]))
        score = float(f1_score(truth[global_index], candidate[global_index]))
        station_layer[f"{station}|{layer}"] = score - base
    tp_additions = int((additions & (truth == 1)).sum())
    fp_additions = int((additions & (truth == 0)).sum())
    precision = tp_additions / int(additions.sum()) if additions.any() else None
    precision_lcb = float(beta_distribution.ppf(float(config["safety"]["precision_lcb_quantile"]), tp_additions, fp_additions + 1)) if tp_additions else 0.0
    policy = config["decision_policy"]
    raw_points = delta * float(policy["score_points_per_f1"])
    calibrated = raw_points - float(policy["transport_penalty_points"])
    gates = {
        "positive_additions": int(additions.sum()) > 0,
        "anchor_removals_zero": int(removals.sum()) == 0,
        "q3_q4_each_nonnegative": min(by_fold[fold]["delta_f1"] for fold in FOLDS[1:]) >= float(policy["minimum_each_q3_q4_delta_f1"]),
        "bootstrap_ci90_low_at_least_0_000578810": bootstrap["ci90_low"] >= float(policy["bootstrap_ci90_low_minimum_f1"]),
        "bootstrap_probability_strictly_above_0_8": bootstrap["probability_improved"] > float(policy["bootstrap_probability_improved_minimum"]),
        "raw_expected_points_at_least_0_015383691": raw_points >= float(policy["minimum_raw_expected_point_delta_inclusive"]),
        "calibrated_expected_points_at_least_0_01": calibrated >= float(policy["minimum_calibrated_expected_point_delta_inclusive"]),
        "marginal_precision_lcb_above_anchor_f1_half": precision_lcb > reference / 2.0,
        "changed_fraction_at_most_0_005": float(additions.sum() / evaluated.sum()) <= float(config["safety"]["maximum_changed_fraction"]),
        "maximum_five_additions_per_station_layer_day": max_station_layer_day_additions <= int(config["safety"]["maximum_additions_per_station_layer_kst_day"]),
        "worst_station_layer_nonnegative": min(station_layer.values()) >= float(config["safety"]["minimum_each_station_layer_delta_f1"]),
    }
    return {"name": name, "family": "P1_FIXED_ADD_ONLY_UNION", "past_only": True, "learned_parameters": 0, "fit_count": 0, "reference_f1": reference, "candidate_f1": value, "delta_f1": delta, "raw_expected_points_delta": raw_points, "transport_penalty_points": float(policy["transport_penalty_points"]), "calibrated_conservative_expected_points_delta": calibrated, "additions": int(additions.sum()), "true_positive_additions": tp_additions, "false_positive_additions": fp_additions, "additions_precision": precision, "additions_precision_lcb90": precision_lcb, "anchor_f1_divided_by_2": reference / 2.0, "anchor_removals": int(removals.sum()), "changed_fraction": float(additions.sum() / evaluated.sum()), "maximum_station_layer_day_additions": max_station_layer_day_additions, "station_layer_delta_f1": station_layer, "by_fold": by_fold, "day_block_bootstrap": bootstrap, "gates": gates, "strict_internal_pass": bool(all(gates.values()))}


def independent_qa(result: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "one_candidate": len(result["candidates"]) == 1,
        "zero_fits": result["fit_count"] == 0,
        "exact_fixed_family": result["transport_family"]["family_id"] == "P1_FIXED_ADD_ONLY_UNION",
        "representation_unchanged": result["transport_family"]["representation_changed"] is False,
        "routing_continuous": result["transport_family"]["routing_discontinuous"] is False,
        "calibration_hash": result["transport_family"]["calibration_sha256"] == result["hashes"]["root_calibration_sha256"],
        "all_add_only": all(item["anchor_removals"] == 0 for item in result["candidates"]),
        "official_reads_zero": result["operations"]["official_covariate_reads"] == 0,
        "hidden_reads_zero": result["operations"]["hidden_truth_reads"] == 0,
        "csv_zero": result["operations"]["submission_csv_created"] == 0,
        "uploads_zero": result["operations"]["uploads"] == 0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def validate_only() -> dict[str, Any]:
    config = load_contract()
    for path in (CALIBRATION_PATH, AUTHORITATIVE_PATH, ANCHOR_PATH):
        if not path.is_file():
            raise ContractError(f"missing frozen input: {path}")
    return {"status": "VALID", "experiment_id": EXPERIMENT_ID, "config_sha256": sha256_file(CONFIG_PATH), "runner_sha256": sha256_file(Path(__file__)), "calibration_sha256": sha256_file(CALIBRATION_PATH), "candidate_count": len(config["candidates"]), "fit_budget": 0}


def execute() -> dict[str, Any]:
    if ARTIFACT.exists():
        raise FileExistsError("exactly-once artifact path exists")
    config = load_contract()
    ARTIFACT.mkdir(parents=True)
    started = time.perf_counter()
    write_json(ARTIFACT / "attempt_lock.json", {"experiment_id": EXPERIMENT_ID, "pid": os.getpid(), "config_sha256": sha256_file(CONFIG_PATH), "runner_sha256": sha256_file(Path(__file__)), "fit_budget": 0})
    frame = load_frame()
    masks = candidate_masks(frame)
    records = [evaluate(item["name"], frame, masks[item["name"]], config) for item in config["candidates"]]
    result: dict[str, Any] = {"schema_version": "p1.public_transport_repair_cycle.20260831.v12.result", "experiment_id": EXPERIMENT_ID, "status": "COMPLETE_INTERNAL_ONLY", "runtime_seconds": time.perf_counter() - started, "transport_family": config["transport_family"], "decision_policy": config["decision_policy"], "candidate_count": 1, "pass_count": sum(item["strict_internal_pass"] for item in records), "fit_count": 0, "candidates": records, "outputs": [], "operations": {"official_covariate_reads": 0, "hidden_truth_reads": 0, "submission_csv_created": 0, "uploads": 0}, "hashes": {"config_sha256": sha256_file(CONFIG_PATH), "runner_sha256": sha256_file(Path(__file__)), "root_calibration_sha256": sha256_file(CALIBRATION_PATH), "authoritative_results_sha256": sha256_file(AUTHORITATIVE_PATH), "anchor_sha256": sha256_file(ANCHOR_PATH)}}
    result["independent_qa"] = independent_qa(result)
    write_json(ARTIFACT / "result.json", result)
    write_json(REPORT / "independent-qa.json", result["independent_qa"])
    write_json(ARTIFACT / "progress.json", {"phase": "terminal", "fit_count": 0, "pass_count": result["pass_count"]})
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
