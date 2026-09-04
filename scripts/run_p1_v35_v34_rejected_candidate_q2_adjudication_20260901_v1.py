"""Zero-fit audit-only adjudication of v34's fixed rejected Q2 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v35_v34_rejected_candidate_q2_adjudication_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode() + b"\n"
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def preflight(data_dir: Path) -> dict[str, Any]:
    if ARTIFACT.exists() or LOCK.exists():
        raise FileExistsError("namespace consumed")
    config = _read(CONFIG)
    resolved = data_dir.resolve(strict=True)
    readme = (resolved / "README.md").resolve(strict=True)
    train = (resolved / "train.csv").resolve(strict=True)
    if readme.parent != resolved or train.parent != resolved:
        raise RuntimeError("source path escaped P1_DATA_DIR")
    if config["source"]["allowed_files"] != ["README.md", "train.csv"]:
        raise RuntimeError("source allowlist drifted")
    if _sha(readme) != config["source"]["readme_sha256"] or _sha(train) != config["source"]["train_sha256"]:
        raise RuntimeError("source binding invalid")

    parent = config["sealed_parent"]
    for key in ("config", "result", "manifest", "bundle", "q2_part"):
        path = ROOT / parent[f"{key}_path"]
        if _sha(path) != parent[f"{key}_sha256"]:
            raise RuntimeError(f"sealed parent {key} drifted")
    result = _read(ROOT / parent["result_path"])
    manifest = _read(ROOT / parent["manifest_path"])
    fixed = config["selection_fixed_before_q2_target"]
    candidates = result["threshold_selection"]["candidates"]
    selected_index = int(np.argmax([item["precision_lcb"] for item in candidates]))
    selected = candidates[selected_index]
    if selected_index != fixed["candidate_index"]:
        raise RuntimeError("fixed max-LCB candidate drifted")
    if selected["quantile"] != fixed["quantile"] or selected["threshold"] != fixed["threshold"]:
        raise RuntimeError("fixed candidate value drifted")
    if result["failed_stage"] != "PRE_Q2_CALIBRATION_GATE" or result["counters"]["transport_target_windows_read"] != 0:
        raise RuntimeError("parent lifecycle incompatible")

    with np.load(ROOT / parent["bundle_path"], allow_pickle=False) as bundle:
        positions = bundle["positions"]
        incumbent = bundle["incumbent"]
        actions = bundle[fixed["action_key"]]
        expected_keys = {"positions", "incumbent", "scores", *(item["action_key"] for item in manifest["candidates"])}
        if set(bundle.files) != expected_keys:
            raise RuntimeError("auditability bundle keys drifted")
    part = pd.read_parquet(ROOT / parent["q2_part_path"], columns=["row_position", "baseline_prediction"])
    if not np.array_equal(positions, part["row_position"].to_numpy(np.int64)):
        raise RuntimeError("Q2 positions drifted")
    if not np.array_equal(incumbent, part["baseline_prediction"].to_numpy(np.int8)):
        raise RuntimeError("Q2 incumbent drifted")
    if int(actions.sum()) != fixed["q2_label_blind_action_count"] or np.any(actions & incumbent.astype(bool)):
        raise RuntimeError("fixed Q2 actions drifted or remove anchor")
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_ZERO_FIT_PRETARGET",
        "source": {"readme": str(readme), "train": str(train)},
        "config_sha256": _sha(CONFIG),
        "runner_sha256": _sha(Path(__file__)),
        "sealed_parent_hashes": {key: parent[f"{key}_sha256"] for key in ("config", "result", "manifest", "bundle", "q2_part")},
        "selected_candidate": fixed,
        "bundle_receipt": {"rows": int(len(positions)), "actions": int(actions.sum()), "model_state_hash_count": len(manifest["model_state_hashes"]), "target_values_present": False},
        "counters": {"fits": 0, "refits": 0, "reselection": 0, "q2_target_reads": 0, "q3_target_reads": 0, "q4_target_reads": 0, "official": 0, "csv": 0, "uploads": 0},
    }


def execute(data_dir: Path) -> dict[str, Any]:
    ready = preflight(data_dir)
    config = _read(CONFIG)
    _write(LOCK, {"experiment_id": EXPERIMENT_ID, "status": "CONSUMED_EXACTLY_ONCE", "config_sha256": ready["config_sha256"], "runner_sha256": ready["runner_sha256"]})
    ARTIFACT.mkdir(exist_ok=False)
    _write(ARTIFACT / "preflight.json", ready)
    parent = config["sealed_parent"]
    fixed = config["selection_fixed_before_q2_target"]
    seal = {
        "experiment_id": EXPERIMENT_ID,
        "rule": fixed["rule"],
        "candidate_index": fixed["candidate_index"],
        "quantile": fixed["quantile"],
        "threshold": fixed["threshold"],
        "action_key": fixed["action_key"],
        "bundle_sha256": parent["bundle_sha256"],
        "fits_refits_reselection": [0, 0, 0],
        "q2_q3_q4_target_reads_before_seal": [0, 0, 0],
    }
    _write(ARTIFACT / "fixed_candidate_pretarget_seal.json", seal)

    with np.load(ROOT / parent["bundle_path"], allow_pickle=False) as bundle:
        positions = bundle["positions"].astype(np.int64)
        incumbent = bundle["incumbent"].astype(np.int8)
        actions = bundle[fixed["action_key"]].astype(np.int8)
    labels = pd.read_csv(ready["source"]["train"], usecols=["label"])["label"].to_numpy(np.int8)
    truth = labels[positions]
    candidate = np.bitwise_or(incumbent, actions)
    additions = actions.astype(bool)
    tp_add = int(np.sum(additions & (truth == 1)))
    fp_add = int(np.sum(additions & (truth == 0)))
    precision = float(tp_add / (tp_add + fp_add)) if tp_add + fp_add else 0.0
    baseline_f1 = _f1(truth, incumbent)
    candidate_f1 = _f1(truth, candidate)
    delta_f1 = candidate_f1 - baseline_f1
    false_negative = delta_f1 > config["diagnosis"]["guard_false_negative_if_q2_delta_f1_gt"]
    diagnosis = "GUARD_FALSE_NEGATIVE_ON_THIS_FIXED_REJECT" if false_negative else "GUARD_TRUE_NEGATIVE_ON_THIS_FIXED_REJECT"
    recommendation = (
        "Do not rescue v34. Preserve v28 until more prospectively sampled rejected outcomes establish a stable false-negative rate."
        if false_negative
        else "Keep v28 unchanged prospectively; this fixed reject was harmful, while one adjudication is insufficient to estimate sensitivity."
    )
    completion = {"experiment_id": EXPERIMENT_ID, "status": "TERMINAL_AUDIT_ONLY", "q2_target_reads": 1, "q3_target_reads": 0, "q4_target_reads": 0, "parent_decision_unchanged": True}
    _write(ARTIFACT / "completion.json", completion)
    result = {
        "schema_version": config["result_schema_version"],
        "experiment_id": EXPERIMENT_ID,
        "decision": diagnosis,
        "scope": "AUDIT_ONLY_NO_PROMOTION_RESCUE_RETUNE",
        "fixed_candidate": seal,
        "q2": {"rows": int(len(truth)), "additions": int(additions.sum()), "tp": tp_add, "fp": fp_add, "precision": precision, "baseline_f1": baseline_f1, "candidate_f1": candidate_f1, "delta_f1": delta_f1},
        "guard_diagnosis": {"false_negative": false_negative, "recommendation": recommendation, "parent_v34_decision_unchanged": True},
        "counters": {"fits": 0, "refits": 0, "score_recomputations": 0, "threshold_reselections": 0, "q2_target_reads": 1, "q3_target_reads": 0, "q4_target_reads": 0, "official": 0, "csv": 0, "uploads": 0},
        "hashes": {"config": ready["config_sha256"], "runner": ready["runner_sha256"], "parent_bundle": parent["bundle_sha256"], "selection_seal": _sha(ARTIFACT / "fixed_candidate_pretarget_seal.json"), "completion": _sha(ARTIFACT / "completion.json"), "lock": _sha(LOCK)},
    }
    _write(ARTIFACT / "result.json", result)
    return result


def qa(data_dir: Path) -> dict[str, Any]:
    result_path = ARTIFACT / "result.json"
    config = _read(CONFIG)
    if not result_path.exists():
        ready = preflight(data_dir)
        checks = {
            "zero_fit_pretarget": all(value == 0 for value in ready["counters"].values()),
            "max_lcb_fixed": ready["selected_candidate"]["candidate_index"] == 0,
            "bundle_target_free": not ready["bundle_receipt"]["target_values_present"],
            "q3_q4_unopened": config["contract"]["q3_q4_target_reads"] == 0,
            "no_rescue": config["contract"]["candidate_promotion_rescue_retune"] == 0,
            "access0": config["source"]["official_test_sample_submission_hidden_reads"] == 0,
        }
        return {"experiment_id": EXPERIMENT_ID, "phase": "PRE_EXECUTION", "checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}
    result = _read(result_path)
    completion = _read(ARTIFACT / "completion.json")
    checks = {
        "terminal": completion["status"] == "TERMINAL_AUDIT_ONLY",
        "zero_fit_refit_reselection": result["counters"]["fits"] == result["counters"]["refits"] == result["counters"]["score_recomputations"] == result["counters"]["threshold_reselections"] == 0,
        "only_q2_opened": result["counters"]["q2_target_reads"] == 1 and result["counters"]["q3_target_reads"] == result["counters"]["q4_target_reads"] == 0,
        "parent_unchanged": result["guard_diagnosis"]["parent_v34_decision_unchanged"],
        "access0": result["counters"]["official"] == result["counters"]["csv"] == result["counters"]["uploads"] == 0,
        "config_hash": result["hashes"]["config"] == _sha(CONFIG),
        "runner_hash": result["hashes"]["runner"] == _sha(Path(__file__)),
        "bundle_hash": result["hashes"]["parent_bundle"] == _sha(ROOT / config["sealed_parent"]["bundle_path"]),
        "selection_seal_hash": result["hashes"]["selection_seal"] == _sha(ARTIFACT / "fixed_candidate_pretarget_seal.json"),
        "completion_hash": result["hashes"]["completion"] == _sha(ARTIFACT / "completion.json"),
        "lock_hash": result["hashes"]["lock"] == _sha(LOCK),
    }
    return {"experiment_id": EXPERIMENT_ID, "phase": "POST_TERMINAL_IMMUTABLE_REVALIDATION", "checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL", "result_sha256": _sha(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--qa", action="store_true")
    args = parser.parse_args()
    value = preflight(args.data_dir) if args.preflight else execute(args.data_dir) if args.execute else qa(args.data_dir)
    print(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False), end="")


if __name__ == "__main__":
    main()
