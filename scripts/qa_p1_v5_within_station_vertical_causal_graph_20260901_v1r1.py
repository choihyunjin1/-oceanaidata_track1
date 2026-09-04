"""Independent read-only QA for the completed P1 v5r1 result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v5_within_station_vertical_causal_graph_20260901_v1r1"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
RUNNER = ROOT / f"scripts/run_{EXPERIMENT_ID}.py"
TRAIN = Path("C:/Users/cedis/Downloads/데이터셋_P1/P1_qc_anomaly/train.csv")
INVALID_V5 = ROOT / "artifacts/p1_v5_within_station_vertical_causal_graph_20260901_v1/attempt_journal/9999_failed.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    actual, predicted = truth.astype(bool), prediction.astype(bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    denominator = 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / denominator if denominator else 0.0}


def main() -> None:
    result = _read(ARTIFACT / "result.json")
    preflight = _read(ARTIFACT / "preflight.json")
    completion = _read(ARTIFACT / "predictions_complete.json")
    config = _read(CONFIG)
    invalid = _read(INVALID_V5)
    labels = pd.read_csv(TRAIN, usecols=["label"])["label"].to_numpy(dtype=np.int8)
    truths, incumbents, candidates, additions = [], [], [], []
    seal_hashes, boundaries, model_hashes = [], [], []
    for fold in config["architecture"]["outer_folds"]:
        seal_path = ARTIFACT / f"{fold}_seal.json"
        seal = _read(seal_path)
        with np.load(ROOT / seal["path"], allow_pickle=False) as values:
            positions = values["positions"]
            incumbent = values["incumbent"]
            addition = values["additions"]
            candidate = values["candidate"]
        truths.append(labels[positions])
        incumbents.append(incumbent)
        candidates.append(candidate)
        additions.append(addition)
        seal_hashes.append(_sha(ROOT / seal["path"]) == seal["sha256"])
        boundaries.append(pd.Timestamp(seal["train_boundary_utc"]).value)
        model_hashes.extend(seal["model_hashes"])
    truth = np.concatenate(truths)
    incumbent = np.concatenate(incumbents)
    candidate = np.concatenate(candidates)
    addition = np.concatenate(additions)
    recomputed = _counts(truth, candidate)
    checks = {
        "invalid_v5_quarantined": invalid["status"] == "INVALID_TECHNICAL_TARGET_LEAKAGE_TIME_UNIT"
        and not invalid["scientific_interpretation_allowed"],
        "v5_not_used_for_selection": config["repair"]["predecessor_result_use_for_selection"] == 0,
        "time_contract_ns_distinct": preflight["time_contract"]["status"] == "PASS_NS_CUTOFF_DISTINCT"
        and len(set(boundaries)) == 3
        and min(boundaries) > pd.Timestamp("2024-01-01T00:00:00Z").value,
        "nine_fresh_models": completion["fits"] == 9 and len(model_hashes) == len(set(model_hashes)) == 9,
        "sealed_hashes": all(seal_hashes),
        "add_only": np.array_equal(candidate, np.bitwise_or(incumbent, addition.astype(np.int8)))
        and int((incumbent > candidate).sum()) == 0,
        "counts_match": all(recomputed[key] == result["pooled"]["candidate"][key] for key in ("tp", "fp", "fn"))
        and abs(recomputed["f1"] - result["pooled"]["candidate"]["f1"]) < 1e-15,
        "no_additions": int(addition.sum()) == result["pooled"]["additions"] == 0,
        "ci_match": result["block_bootstrap"]["ci90"] == [0.0, 0.0],
        "hashes_match": result["hashes"]["config"] == _sha(CONFIG)
        and result["hashes"]["runner"] == _sha(RUNNER)
        and result["hashes"]["completion"] == _sha(ARTIFACT / "predictions_complete.json")
        and result["hashes"]["lock"] == _sha(LOCK),
        "access_zero": result["counters"]["official"] == 0
        and result["counters"]["csv"] == 0
        and result["counters"]["uploads"] == 0
        and result["counters"]["outer_target_reads_before_all_seals"] == 0,
    }
    output = {
        "experiment_id": EXPERIMENT_ID,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recomputed": {**recomputed, "additions": int(addition.sum())},
        "result_sha256": _sha(ARTIFACT / "result.json"),
    }
    print(json.dumps(output, sort_keys=True, ensure_ascii=False, allow_nan=False))
    if output["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
