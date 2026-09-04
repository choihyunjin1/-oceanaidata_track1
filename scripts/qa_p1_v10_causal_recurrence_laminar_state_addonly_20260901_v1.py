"""Lifecycle-safe independent post-terminal QA for P1 v10 recurrence state."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v10_causal_recurrence_laminar_state_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
RUNNER = ROOT / f"scripts/run_{EXPERIMENT_ID}.py"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
RESULT = ARTIFACT / "result.json"
COMPLETE = ARTIFACT / "predictions_complete.json"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _module():
    spec = importlib.util.spec_from_file_location("p1_v10_qa_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("runner load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def _counts(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    actual = truth.astype(bool)
    predicted = prediction.astype(bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "f1": f1}


def verify(data_dir: Path) -> dict[str, object]:
    runner = _module()
    config = _read(CONFIG)
    result = _read(RESULT)
    complete = _read(COMPLETE)
    train = (data_dir.resolve(strict=True) / "train.csv").resolve(strict=True)
    frame = pd.read_csv(train, usecols=["station", "layer", "time", "label"])
    labels = frame["label"].to_numpy(np.int8)
    truths, incumbents, candidates, additions, metadata_parts = [], [], [], [], []
    hashes, isolation, models = [], [], []
    for fold in config["parts"]:
        seal = _read(ARTIFACT / f"{fold}_seal.json")
        path = ROOT / seal["path"]
        with np.load(path, allow_pickle=False) as values:
            positions = values["positions"]
            incumbent = values["incumbent"]
            addition = values["additions"]
            candidate = values["candidate"]
        truths.append(labels[positions])
        incumbents.append(incumbent)
        candidates.append(candidate)
        additions.append(addition)
        metadata_parts.append(
            frame.iloc[positions]
            .loc[:, ["station", "layer", "time"]]
            .reset_index(drop=True)
        )
        hashes.append(_sha(path) == seal["sha256"])
        isolation.append(seal["outer_target_reads_before_seal"] == 0)
        models.extend(seal["model_hashes"])
    truth, incumbent, candidate, addition = (
        np.concatenate(values)
        for values in (truths, incumbents, candidates, additions)
    )
    metadata = pd.concat(metadata_parts, ignore_index=True)
    anchor_score = _counts(truth, incumbent)
    candidate_score = _counts(truth, candidate)
    long_event = runner.base._long_event_interior(
        truth,
        incumbent,
        candidate,
        metadata,
    )
    checks = {
        "sealed_hashes": all(hashes),
        "nine_unique_models": complete["fits"] == 9 and len(set(models)) == 9,
        "outer_isolation": all(isolation)
        and complete["outer_target_reads_before_all_seals"] == 0,
        "add_only": np.array_equal(
            candidate,
            np.bitwise_or(incumbent, addition.astype(np.int8)),
        )
        and not (incumbent > candidate).any(),
        "anchor_score": all(
            anchor_score[key] == result["pooled"]["incumbent"][key]
            for key in ("tp", "fp", "fn", "f1")
        ),
        "candidate_score": all(
            candidate_score[key] == result["pooled"]["candidate"][key]
            for key in ("tp", "fp", "fn", "f1")
        ),
        "delta": abs(
            candidate_score["f1"]
            - anchor_score["f1"]
            - result["pooled"]["delta_f1"]
        )
        < 1e-15,
        "long_event": long_event == result["long_event_interior"],
        "hashes": result["hashes"]
        == {
            "config": _sha(CONFIG),
            "runner": _sha(RUNNER),
            "completion": _sha(COMPLETE),
            "lock": _sha(LOCK),
        },
        "access0": result["counters"]["official"]
        == result["counters"]["csv"]
        == result["counters"]["uploads"]
        == 0,
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recomputed": {
            "anchor": anchor_score,
            "candidate": candidate_score,
            "delta_f1": candidate_score["f1"] - anchor_score["f1"],
            "additions": int(addition.sum()),
            "addition_true_positive": int(truth[addition].sum()),
            "long_event_interior": long_event,
        },
        "result_sha256": _sha(RESULT),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = verify(args.data_dir)
    print(json.dumps(payload, sort_keys=True, ensure_ascii=True))
    if payload["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
