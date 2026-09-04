"""Exactly-once label-free dependence-calibration feasibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p1_qc.clean_state_capa import INPUT_ONLY_COLUMNS, apply_clean_state, fit_clean_state

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_dependence_aware_empirical_null_feasibility_20260901_v1"
CONFIG_PATH = ROOT / "configs/experiments/p1_dependence_aware_empirical_null_feasibility_20260901_v1.json"
ARTIFACT_DIR = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK_PATH = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


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
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        os.write(descriptor, raw + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_train(path: Path, expected_sha: str) -> Path:
    resolved = path.resolve(strict=True)
    if resolved.name != "train.csv" or not resolved.is_file():
        raise RuntimeError("only historical train.csv is allowed")
    lowered = str(resolved).casefold()
    if any(token in lowered for token in ("test.csv", "sample_submission", "submission")):
        raise RuntimeError("forbidden source path")
    if _sha(resolved) != expected_sha:
        raise RuntimeError("historical train hash changed")
    return resolved


def preflight(train_csv: Path) -> dict[str, Any]:
    if ARTIFACT_DIR.exists() or LOCK_PATH.exists():
        raise FileExistsError("exactly-once audit namespace is consumed")
    config = _read_json(CONFIG_PATH)
    if config["experiment_id"] != EXPERIMENT_ID:
        raise RuntimeError("experiment id changed")
    authority = ROOT / config["authority"]["path"]
    if _sha(authority) != config["authority"]["sha256"]:
        raise RuntimeError("meta-research authority changed")
    train = _validate_train(train_csv, config["input_contract"]["train_csv_sha256"])
    audits = {}
    for fold, relative in config["incumbent_audits"].items():
        audit_path = ROOT / relative
        audit = _read_json(audit_path)
        if audit.get("fold") != fold or audit.get("target_fold_validation_labels_read_before_prediction") != 0:
            raise RuntimeError(f"incumbent audit changed: {fold}")
        audits[fold] = {
            "path": relative,
            "sha256": _sha(audit_path),
            "adjusted_cutoff_utc": audit["adjusted_cutoff_utc"],
            "prefix_rows": audit["prefix_rows"],
        }
    return {
        "schema_version": "p1.dependence_null.preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_ZERO_OPERATION_LABEL_FREE_READINESS",
        "train_path": str(train),
        "train_sha256": config["input_contract"]["train_csv_sha256"],
        "config_sha256": _sha(CONFIG_PATH),
        "runner_sha256": _sha(Path(__file__)),
        "authority_sha256": config["authority"]["sha256"],
        "incumbent_audits": audits,
        "sealed_null_calibration": config["sealed_null_calibration"],
        "sealed_feasibility_ceilings": config["sealed_feasibility_ceilings"],
        "operation_counters": {"claims": 0, "supervised_fits": 0, "input_only_clean_state_projections": 0, "target_rows_read": 0, "prediction_rows_created": 0, "official_reads": 0, "csv_files": 0, "uploads": 0},
    }


def _split_by_time(frame: pd.DataFrame, fractions: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parsed = pd.to_datetime(frame["time"], utc=True, errors="raise", format="mixed")
    unique = np.sort(parsed.unique())
    first = unique[max(1, int(len(unique) * fractions[0])) - 1]
    second = unique[max(2, int(len(unique) * (fractions[0] + fractions[1]))) - 1]
    fit = frame.loc[parsed <= first].reset_index(drop=True)
    calibration = frame.loc[(parsed > first) & (parsed <= second)].reset_index(drop=True)
    heldout = frame.loc[parsed > second].reset_index(drop=True)
    if min(len(fit), len(calibration), len(heldout)) == 0:
        raise RuntimeError("prefix time split is empty")
    return fit, calibration, heldout


def _block_maxima(frame: pd.DataFrame, signal: np.ndarray, block: int) -> np.ndarray:
    maxima: list[float] = []
    working = frame.loc[:, ["station", "layer", "time"]].copy()
    working["signal"] = np.abs(signal)
    working.sort_values(["station", "layer", "time"], kind="stable", inplace=True)
    for _key, group in working.groupby(["station", "layer"], sort=True, observed=True):
        values = group["signal"].to_numpy(dtype=np.float64)
        for start in range(0, len(values) - block + 1, block):
            maxima.append(float(np.max(values[start : start + block])))
    if len(maxima) < 20:
        raise RuntimeError("insufficient moving blocks for empirical null")
    return np.asarray(maxima, dtype=np.float64)


def _evaluate(frame: pd.DataFrame, signal: np.ndarray, threshold: float, block: int, min_layers: int) -> dict[str, Any]:
    working = frame.loc[:, ["station", "layer", "time"]].reset_index(drop=True).copy()
    working["raw"] = np.abs(signal) > threshold
    concurrent = working.groupby(["station", "time"], sort=False, observed=True)["raw"].transform("sum")
    proposal = working["raw"].to_numpy() & concurrent.to_numpy(dtype=np.int64).__ge__(min_layers)
    proposed = int(proposal.sum())
    blocks = _block_maxima(frame, signal, block)
    false_alarm = float(np.mean(blocks > threshold))
    if proposed:
        counts = working.loc[proposal].groupby(["station", "layer"], observed=True).size()
        concentration = float(counts.max() / proposed)
    else:
        concentration = 0.0
    return {
        "rows": len(frame),
        "threshold": threshold,
        "heldout_block_count": len(blocks),
        "realized_false_alarm_rate": false_alarm,
        "proposal_rows": proposed,
        "proposal_row_share": float(proposed / len(frame)),
        "maximum_single_station_layer_proposal_concentration": concentration,
    }


def execute(train_csv: Path) -> dict[str, Any]:
    started = time.monotonic()
    ready = preflight(train_csv)
    config = _read_json(CONFIG_PATH)
    _create_json(LOCK_PATH, {"experiment_id": EXPERIMENT_ID, "status": "CONSUMED_EXACTLY_ONCE", "config_sha256": ready["config_sha256"], "runner_sha256": ready["runner_sha256"]})
    ARTIFACT_DIR.mkdir(exist_ok=False)
    _create_json(ARTIFACT_DIR / "preflight.json", ready)
    frame = pd.read_csv(Path(ready["train_path"]), usecols=list(INPUT_ONLY_COLUMNS)).loc[:, list(INPUT_ONLY_COLUMNS)]
    parsed = pd.to_datetime(frame["time"], utc=True, errors="raise", format="mixed")
    settings = config["sealed_null_calibration"]
    ceilings = config["sealed_feasibility_ceilings"]
    fold_results = []
    all_decisions: list[bool] = []
    for fold in config["folds"]:
        cutoff = pd.Timestamp(ready["incumbent_audits"][fold]["adjusted_cutoff_utc"])
        prefix = frame.loc[parsed <= cutoff].reset_index(drop=True)
        fit, calibration, heldout = _split_by_time(prefix, settings["prefix_time_split"])
        state = fit_clean_state(fit)
        calibration_signal = apply_clean_state(calibration, state)["decoder_signal"].to_numpy(dtype=np.float64)
        heldout_signal = apply_clean_state(heldout, state)["decoder_signal"].to_numpy(dtype=np.float64)
        sensitivity = []
        for block in settings["sensitivity_block_lengths_rows"]:
            null_maxima = _block_maxima(calibration, calibration_signal, int(block))
            threshold = float(np.quantile(null_maxima, settings["empirical_block_max_quantile"], method="higher"))
            evidence = _evaluate(heldout, heldout_signal, threshold, int(block), settings["cross_layer_minimum_concurrent_layers"])
            passed = bool(
                evidence["realized_false_alarm_rate"] <= ceilings["realized_false_alarm_rate_lte"]
                and evidence["proposal_row_share"] <= ceilings["proposal_row_share_lte"]
                and evidence["maximum_single_station_layer_proposal_concentration"] <= ceilings["maximum_single_station_layer_proposal_concentration_lte"]
            )
            evidence.update({"block_length_rows": int(block), "calibration_block_count": len(null_maxima), "decision_pass": passed})
            sensitivity.append(evidence)
            all_decisions.append(passed)
        stable = len({item["decision_pass"] for item in sensitivity}) == 1
        fold_results.append({"fold": fold, "fit_rows": len(fit), "calibration_rows": len(calibration), "heldout_rows": len(heldout), "clean_state_sha256": state.sha256, "sensitivity": sensitivity, "block_length_decision_stable": stable})
    global_stable = all(item["block_length_decision_stable"] for item in fold_results)
    passed = all(all_decisions) and global_stable
    decision = config["decision"]["pass"] if passed else config["decision"]["fail"]
    result = {
        "schema_version": "p1.dependence_null.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_LABEL_FREE_INFORMATION_AUDIT",
        "decision": decision,
        "folds": fold_results,
        "decision_checks": {"all_fold_block_ceiling_checks_pass": all(all_decisions), "all_block_length_decisions_equal_within_fold": global_stable},
        "operation_counters": {"executions": 1, "supervised_fits": 0, "input_only_clean_state_projections": 3, "target_rows_read": 0, "prediction_rows_created": 0, "official_reads": 0, "csv_files": 0, "uploads": 0},
        "runtime_seconds": time.monotonic() - started,
        "hashes": {"config_sha256": ready["config_sha256"], "runner_sha256": ready["runner_sha256"], "attempt_lock_sha256": _sha(LOCK_PATH)},
        "performance_stage_authorized": passed,
    }
    _create_json(ARTIFACT_DIR / "result.json", result)
    return result


def qa(train_csv: Path) -> dict[str, Any]:
    ready = preflight(train_csv)
    config = _read_json(CONFIG_PATH)
    checks = {
        "zero_operation": all(value == 0 for value in ready["operation_counters"].values()),
        "label_free": config["input_contract"]["target_columns_read"] == 0,
        "supervised_fits_zero": config["operation_contract"]["supervised_model_fits"] == 0,
        "ceilings_sealed": set(config["sealed_feasibility_ceilings"]) == {
            "realized_false_alarm_rate_lte",
            "proposal_row_share_lte",
            "maximum_single_station_layer_proposal_concentration_lte",
            "all_block_length_decisions_equal",
        },
        "no_alpha_sweep": config["sealed_null_calibration"]["alpha_sweep"] is False,
        "forbidden_family_repetition_zero": not config["duplication_audit"]["gaussian_capa_exact_family_repeated"]
        and not config["duplication_audit"]["long_event_rescore_exact_family_repeated"],
    }
    return {
        "schema_version": "p1.dependence_null.independent_qa.v1",
        "experiment_id": EXPERIMENT_ID,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "preflight_sha256": hashlib.sha256(
            json.dumps(ready, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--qa", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        value = preflight(args.train_csv)
    elif args.qa:
        value = qa(args.train_csv)
    else:
        value = execute(args.train_csv)
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False), end="")


if __name__ == "__main__":
    main()
