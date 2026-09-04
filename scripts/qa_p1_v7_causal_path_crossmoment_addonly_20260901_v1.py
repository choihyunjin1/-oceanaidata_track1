"""Lifecycle-safe independent post-terminal QA for P1 v7."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v7_causal_path_crossmoment_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
RUNNER = ROOT / f"scripts/run_{EXPERIMENT_ID}.py"
RESULT = ROOT / f"artifacts/{EXPERIMENT_ID}/result.json"
COMPLETE = ROOT / f"artifacts/{EXPERIMENT_ID}/predictions_complete.json"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
TRAIN = Path("C:/Users/cedis/Downloads/데이터셋_P1/P1_qc_anomaly/train.csv")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    actual, predicted = truth.astype(bool), prediction.astype(bool)
    tp, fp, fn = int((actual & predicted).sum()), int((~actual & predicted).sum()), int((actual & ~predicted).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / (2 * tp + fp + fn)}


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    complete = json.loads(COMPLETE.read_text(encoding="utf-8"))
    labels = pd.read_csv(TRAIN, usecols=["label"])["label"].to_numpy(np.int8)
    truths, incumbents, candidates, additions, hashes, models = [], [], [], [], [], []
    for fold in config["parts"]:
        seal = json.loads((ROOT / f"artifacts/{EXPERIMENT_ID}/{fold}_seal.json").read_text(encoding="utf-8"))
        with np.load(ROOT / seal["path"], allow_pickle=False) as values:
            positions, incumbent, addition, candidate = values["positions"], values["incumbent"], values["additions"], values["candidate"]
        truths.append(labels[positions])
        incumbents.append(incumbent)
        candidates.append(candidate)
        additions.append(addition)
        hashes.append(_sha(ROOT / seal["path"]) == seal["sha256"])
        models.extend(seal["model_hashes"])
    truth, incumbent, candidate, addition = (np.concatenate(values) for values in (truths, incumbents, candidates, additions))
    score = _counts(truth, candidate)
    checks = {
        "sealed_hashes": all(hashes),
        "nine_unique_models": complete["fits"] == 9 and len(set(models)) == 9,
        "add_only": np.array_equal(candidate, np.bitwise_or(incumbent, addition.astype(np.int8))) and not (incumbent > candidate).any(),
        "score": all(score[key] == result["pooled"]["candidate"][key] for key in ("tp", "fp", "fn")) and abs(score["f1"] - result["pooled"]["candidate"]["f1"]) < 1e-15,
        "hashes": result["hashes"] == {"config": _sha(CONFIG), "runner": _sha(RUNNER), "completion": _sha(COMPLETE), "lock": _sha(LOCK)},
        "access0": result["counters"]["official"] == result["counters"]["csv"] == result["counters"]["uploads"] == 0,
        "surface": result["surface"] == "EXPLORATORY_REUSED_SURFACE",
    }
    payload = {"experiment_id": EXPERIMENT_ID, "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "recomputed": {**score, "additions": int(addition.sum())}, "result_sha256": _sha(RESULT)}
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    if payload["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
