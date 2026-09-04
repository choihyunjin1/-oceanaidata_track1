"""Lifecycle-safe independent post-terminal QA for the P1 v8 latent-state run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v8_missingness_decay_gru_latent_state_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
RUNNER = ROOT / f"scripts/run_{EXPERIMENT_ID}.py"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
RESULT = ARTIFACT / "result.json"
COMPLETE = ARTIFACT / "predictions_complete.json"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _counts(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    actual = truth.astype(bool)
    predicted = prediction.astype(bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "f1": f1}


def _long_event_diagnostics(
    truth: np.ndarray,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    """Recall on interiors of positive 10-minute runs lasting at least three hours."""

    work = metadata.loc[:, ["station", "layer", "time"]].copy()
    work["_position"] = np.arange(len(work))
    work["_ns"] = pd.DatetimeIndex(
        pd.to_datetime(work["time"], utc=True, errors="raise", format="mixed")
    ).as_unit("ns").asi8
    interior: list[int] = []
    run_count = 0
    for _key, group in work.groupby(
        ["station", "layer"],
        sort=True,
        observed=True,
    ):
        ordered = group.sort_values("_ns", kind="stable")
        positions = ordered["_position"].to_numpy(np.int64)
        times = ordered["_ns"].to_numpy(np.int64)
        positive = truth[positions].astype(bool)
        start = 0
        while start < len(positions):
            if not positive[start]:
                start += 1
                continue
            end = start + 1
            while (
                end < len(positions)
                and positive[end]
                and times[end] - times[end - 1] == 600_000_000_000
            ):
                end += 1
            if end - start >= 18:
                run_count += 1
                interior.extend(positions[start + 6 : end - 6].tolist())
            start = end
    selected = np.asarray(interior, dtype=np.int64)
    return {
        "definition": "positive 10-minute runs >=18 rows; exclude 6 boundary rows per side",
        "runs": run_count,
        "interior_rows": len(selected),
        "anchor_recall": float(incumbent[selected].mean()) if len(selected) else 0.0,
        "candidate_recall": float(candidate[selected].mean()) if len(selected) else 0.0,
        "delta_recall": (
            float(candidate[selected].mean() - incumbent[selected].mean())
            if len(selected)
            else 0.0
        ),
    }


def verify(data_dir: Path) -> dict[str, Any]:
    config = _read(CONFIG)
    result = _read(RESULT)
    complete = _read(COMPLETE)
    train = (data_dir.resolve(strict=True) / "train.csv").resolve(strict=True)
    train_frame = pd.read_csv(
        train,
        usecols=["station", "layer", "time", "label"],
    )
    labels = train_frame["label"].to_numpy(np.int8)
    truths = []
    incumbents = []
    candidates = []
    additions = []
    metadata_parts = []
    sealed_hashes = []
    outer_isolation = []
    models = []
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
            train_frame.iloc[positions]
            .loc[:, ["station", "layer", "time"]]
            .reset_index(drop=True)
        )
        sealed_hashes.append(_sha(path) == seal["sha256"])
        outer_isolation.append(seal["outer_target_reads_before_seal"] == 0)
        models.extend(seal["model_hashes"])
    truth, incumbent, candidate, addition = (
        np.concatenate(values)
        for values in (truths, incumbents, candidates, additions)
    )
    anchor_score = _counts(truth, incumbent)
    candidate_score = _counts(truth, candidate)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    long_event = _long_event_diagnostics(
        truth,
        incumbent,
        candidate,
        metadata,
    )
    checks = {
        "sealed_hashes": all(sealed_hashes),
        "nine_unique_models": complete["fits"] == 9 and len(set(models)) == 9,
        "outer_isolation": (
            all(outer_isolation)
            and complete["outer_target_reads_before_all_seals"] == 0
        ),
        "add_only": (
            np.array_equal(
                candidate,
                np.bitwise_or(incumbent, addition.astype(np.int8)),
            )
            and not (incumbent > candidate).any()
        ),
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
        "hashes": result["hashes"]
        == {
            "config": _sha(CONFIG),
            "runner": _sha(RUNNER),
            "completion": _sha(COMPLETE),
            "lock": _sha(LOCK),
        },
        "access0": (
            result["counters"]["official"]
            == result["counters"]["csv"]
            == result["counters"]["uploads"]
            == 0
        ),
        "surface": result["surface"] == "EXPLORATORY_REUSED_SURFACE",
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
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    if payload["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
