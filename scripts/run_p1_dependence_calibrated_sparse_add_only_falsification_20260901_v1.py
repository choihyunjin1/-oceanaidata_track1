"""Exactly-once null-calibrated sparse add-only P1 historical falsification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p1_qc.clean_state_capa import (
    INPUT_ONLY_COLUMNS,
    KEY_COLUMNS,
    apply_clean_state,
    fit_clean_state,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_dependence_calibrated_sparse_add_only_falsification_20260901_v1"
CONFIG_PATH = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK_PATH = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
LEGACY_SCORER = ROOT / "scripts/run_p1_clean_state_capa_falsification_20260831_v1.py"
PART_COLUMNS = [*KEY_COLUMNS, "row_position", "baseline_probability", "baseline_prediction"]
POINTS_PER_F1 = 0.6778 / 0.0255


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path.name}")
    return value


def _create_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode()
        os.write(descriptor, raw + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_scorer():
    spec = importlib.util.spec_from_file_location("p1_capa_historical_scorer", LEGACY_SCORER)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load pinned historical scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_train(path: Path, expected: str) -> Path:
    resolved = path.resolve(strict=True)
    if resolved.name != "train.csv" or any(token in str(resolved).casefold() for token in ("test.csv", "sample_submission", "submission")):
        raise RuntimeError("only historical train.csv is allowed")
    if _sha(resolved) != expected:
        raise RuntimeError("historical train hash changed")
    return resolved


def preflight(train_csv: Path) -> dict[str, Any]:
    if ARTIFACT_DIR.exists() or LOCK_PATH.exists():
        raise FileExistsError("performance namespace is consumed")
    config = _read_json(CONFIG_PATH)
    calibration_path = ROOT / config["calibration_authority"]["path"]
    if _sha(calibration_path) != config["calibration_authority"]["sha256"]:
        raise RuntimeError("label-free calibration authority hash changed")
    calibration = _read_json(calibration_path)
    if calibration["decision"] != config["calibration_authority"]["required_decision"] or not calibration["performance_stage_authorized"]:
        raise RuntimeError("label-free audit did not authorize historical falsification")
    observed_thresholds = {}
    for fold in calibration["folds"]:
        primary = next(item for item in fold["sensitivity"] if item["block_length_rows"] == 144)
        observed_thresholds[fold["fold"]] = primary["threshold"]
    if observed_thresholds != config["sealed_proposal_rule"]["thresholds"]:
        raise RuntimeError("sealed proposal thresholds differ from label-free audit")
    train = _validate_train(train_csv, config["input_contract"]["train_csv_sha256"])
    parts = {}
    for fold, item in config["incumbent_parts"].items():
        path = ROOT / item["path"]
        audit = _read_json(ROOT / item["audit_path"])
        if _sha(path) != item["sha256"] or audit.get("target_fold_validation_labels_read_before_prediction") != 0:
            raise RuntimeError(f"incumbent binding changed: {fold}")
        parts[fold] = {"sha256": item["sha256"], "audit_sha256": _sha(ROOT / item["audit_path"]), "cutoff": audit["adjusted_cutoff_utc"]}
    return {
        "schema_version": "p1.dependence_sparse.preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_FRESH_EXACTLY_ONCE_LABEL_SEALED_READINESS",
        "train_path": str(train),
        "config_sha256": _sha(CONFIG_PATH),
        "runner_sha256": _sha(Path(__file__)),
        "calibration_result_sha256": _sha(calibration_path),
        "parts": parts,
        "thresholds": observed_thresholds,
        "operation_counters": {"claims": 0, "supervised_fits": 0, "target_rows_read": 0, "prediction_rows": 0, "official_reads": 0, "csv_files": 0, "uploads": 0},
    }


def _sparse_segments(frame: pd.DataFrame, signal: np.ndarray, threshold: float, min_layers: int, min_rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    working = frame.loc[:, ["station", "layer", "time"]].reset_index(drop=True).copy()
    working["position"] = np.arange(len(working), dtype=np.int64)
    working["raw"] = np.abs(signal) > threshold
    concurrent = working.groupby(["station", "time"], sort=False, observed=True)["raw"].transform("sum")
    working["coherent"] = working["raw"] & concurrent.ge(min_layers)
    additions = np.zeros(len(frame), dtype=bool)
    segments = 0
    for _key, group in working.groupby(["station", "layer"], sort=True, observed=True):
        group = group.sort_values("time", kind="stable")
        parsed = pd.to_datetime(group["time"], utc=True, errors="raise", format="mixed")
        flags = group["coherent"].to_numpy(dtype=bool)
        positions = group["position"].to_numpy(dtype=np.int64)
        start = 0
        while start < len(group):
            if not flags[start]:
                start += 1
                continue
            end = start + 1
            while end < len(group) and flags[end] and parsed.iloc[end] - parsed.iloc[end - 1] <= pd.Timedelta(minutes=30):
                end += 1
            if end - start >= min_rows:
                additions[positions[start:end]] = True
                segments += 1
            start = end
    return additions, {"raw_exceedance_rows": int(working["raw"].sum()), "coherent_rows": int(working["coherent"].sum()), "accepted_segments": segments, "addition_rows": int(additions.sum())}


def execute(train_csv: Path) -> dict[str, Any]:
    started = time.monotonic()
    ready = preflight(train_csv)
    config = _read_json(CONFIG_PATH)
    scorer = _load_scorer()
    _create_json(LOCK_PATH, {"experiment_id": EXPERIMENT_ID, "status": "CONSUMED_EXACTLY_ONCE", "config_sha256": ready["config_sha256"], "runner_sha256": ready["runner_sha256"]})
    ARTIFACT_DIR.mkdir(exist_ok=False)
    _create_json(ARTIFACT_DIR / "preflight.json", ready)
    frame = pd.read_csv(Path(ready["train_path"]), usecols=list(INPUT_ONLY_COLUMNS)).loc[:, list(INPUT_ONLY_COLUMNS)]
    parsed = pd.to_datetime(frame["time"], utc=True, errors="raise", format="mixed")
    seals = []
    for fold in config["folds"]:
        item = config["incumbent_parts"][fold]
        audit = _read_json(ROOT / item["audit_path"])
        part = pd.read_parquet(ROOT / item["path"], columns=PART_COLUMNS)
        positions = part["row_position"].to_numpy(dtype=np.int64)
        cutoff = pd.Timestamp(audit["adjusted_cutoff_utc"])
        prefix = frame.loc[parsed <= cutoff].reset_index(drop=True)
        validation = frame.iloc[positions].reset_index(drop=True)
        state = fit_clean_state(prefix)
        signal = apply_clean_state(validation, state)["decoder_signal"].to_numpy(dtype=np.float64)
        additions, proposal_audit = _sparse_segments(validation, signal, ready["thresholds"][fold], config["sealed_proposal_rule"]["minimum_concurrent_layers_same_station_time"], config["sealed_proposal_rule"]["minimum_segment_rows"])
        if additions.mean() > config["sealed_proposal_rule"]["maximum_total_proposal_row_share"]:
            raise RuntimeError("label-free proposal share ceiling exceeded before target access")
        incumbent = part["baseline_prediction"].to_numpy(dtype=np.int8)
        candidate = np.bitwise_or(incumbent, additions.astype(np.int8))
        prediction_path = ARTIFACT_DIR / f"{fold}_sealed_predictions.npz"
        np.savez_compressed(prediction_path, row_position=positions, incumbent=incumbent, additions=additions, candidate=candidate)
        seal = {"fold": fold, "prediction_path": str(prediction_path.relative_to(ROOT)).replace("\\", "/"), "prediction_sha256": _sha(prediction_path), "rows": len(positions), "threshold": ready["thresholds"][fold], "clean_state_sha256": state.sha256, "proposal_audit": proposal_audit, "target_rows_read_before_seal": 0}
        seal_path = ARTIFACT_DIR / f"{fold}_seal.json"
        _create_json(seal_path, seal)
        seal["seal_sha256"] = _sha(seal_path)
        seals.append(seal)
    completion = {"experiment_id": EXPERIMENT_ID, "status": "ALL_Q2_Q4_PREDICTIONS_SEALED", "seals": seals, "target_rows_read": 0, "supervised_fits": 0, "official_reads": 0}
    _create_json(ARTIFACT_DIR / "predictions_complete.json", completion)
    targets = pd.read_csv(Path(ready["train_path"]), usecols=["label", "anomaly_type"])
    truth_all = targets["label"].to_numpy(dtype=np.int8)
    fold_scores = []
    pools = {key: [] for key in ("truth", "incumbent", "candidate", "additions", "types", "metadata")}
    for seal in seals:
        with np.load(ROOT / seal["prediction_path"], allow_pickle=False) as arrays:
            positions = arrays["row_position"]
            incumbent = arrays["incumbent"]
            additions = arrays["additions"]
            candidate = arrays["candidate"]
        truth = truth_all[positions]
        metadata = frame.iloc[positions].loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
        anomaly = targets.iloc[positions]["anomaly_type"].reset_index(drop=True)
        score = scorer._score_surface(truth, incumbent, candidate, additions, anomaly, metadata)
        fold_scores.append({"fold": seal["fold"], **score, "addition_share": float(additions.mean())})
        for key, value in (("truth", truth), ("incumbent", incumbent), ("candidate", candidate), ("additions", additions), ("types", anomaly), ("metadata", metadata)):
            pools[key].append(value)
    truth = np.concatenate(pools["truth"])
    incumbent = np.concatenate(pools["incumbent"])
    candidate = np.concatenate(pools["candidate"])
    additions = np.concatenate(pools["additions"])
    types = pd.concat(pools["types"], ignore_index=True)
    metadata = pd.concat(pools["metadata"], ignore_index=True)
    pooled = scorer._score_surface(truth, incumbent, candidate, additions, types, metadata)
    bootstrap = scorer._paired_cluster_bootstrap(truth, incumbent, candidate, metadata, replicates=config["decision"]["bootstrap_replicates"], seed=config["decision"]["bootstrap_seed"])
    fold_nonnegative = all(item["delta_f1"] >= 0 for item in fold_scores)
    passed = pooled["delta_f1"] > 0 and bootstrap["ci90"][0] >= 0 and pooled["additions_precision"] > pooled["incumbent"]["f1"] / 2 and fold_nonnegative
    diagnostics = pooled["station_layer_diagnostics"]
    worst = sorted(diagnostics, key=lambda item: (item["delta_f1"], item["station"], item["layer"]))[:5]
    result = {
        "schema_version": "p1.dependence_sparse.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_HISTORICAL_FALSIFICATION",
        "decision": config["decision"]["pass"] if passed else config["decision"]["fail"],
        "fold_scores": fold_scores,
        "pooled": pooled,
        "paired_block_bootstrap": bootstrap,
        "worst_station_layer_slices": worst,
        "nominal_point_conversion": {"basis": "planning map, not official guarantee", "expected_points": pooled["delta_f1"] * POINTS_PER_F1},
        "decision_checks": {"pooled_delta_positive": pooled["delta_f1"] > 0, "ci90_lower_nonnegative": bootstrap["ci90"][0] >= 0, "addition_precision_gate": pooled["additions_precision"] > pooled["incumbent"]["f1"] / 2, "all_folds_nonnegative": fold_nonnegative},
        "operation_counters": {"executions": 1, "input_only_clean_state_fits": 3, "supervised_fits": 0, "target_rows_read_after_all_seals": len(targets), "official_reads": 0, "csv_files": 0, "uploads": 0},
        "runtime_seconds": time.monotonic() - started,
        "hashes": {"config_sha256": ready["config_sha256"], "runner_sha256": ready["runner_sha256"], "calibration_result_sha256": ready["calibration_result_sha256"], "predictions_complete_sha256": _sha(ARTIFACT_DIR / "predictions_complete.json"), "attempt_lock_sha256": _sha(LOCK_PATH)},
    }
    _create_json(ARTIFACT_DIR / "result.json", result)
    return result


def qa(train_csv: Path) -> dict[str, Any]:
    ready = preflight(train_csv)
    config = _read_json(CONFIG_PATH)
    checks = {"zero_operation": all(value == 0 for value in ready["operation_counters"].values()), "audit_pass_bound": ready["calibration_result_sha256"] == config["calibration_authority"]["sha256"], "label_score_tuning_zero": config["sealed_proposal_rule"]["label_score_based_tuning_count"] == 0, "add_only": config["sealed_proposal_rule"]["operation"] == "bitwise_or_with_champion_anchor_no_removals", "forbidden_family_repetition_zero": not config["duplication_audit"]["gaussian_capa_exact_family_repeated"] and not config["duplication_audit"]["long_event_rescore_exact_family_repeated"]}
    return {"schema_version": "p1.dependence_sparse.independent_qa.v1", "experiment_id": EXPERIMENT_ID, "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--qa", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = preflight(args.train_csv) if args.preflight else qa(args.train_csv) if args.qa else execute(args.train_csv)
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False), end="")


if __name__ == "__main__":
    main()
