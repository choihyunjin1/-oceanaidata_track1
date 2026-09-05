"""Zero-fit metadata-only support audit for a separate two-sided P1 proposal."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import ocean_evaluation_contract_v5 as evaluation
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = "p1_twosided_support_20260906_v1"
REPORT = ROOT / "reports" / ID
KEYS = ["station", "year", "layer", "time"]
TRAIN_SHA = "20b656b0cbd524ad9da0bae8ecb6e0bacfc006e05810b37e83f29a5fa8e65cd2"
PINS = {
    "scripts/ocean_evaluation_contract_v5.py":
        "36581e8ac358463670b368e2da4ded98d5479c8a98c7c3917774c0dd68eb6e39",
    "configs/evaluation/ocean_forward_v5.json":
        "ca6f610aa087c5d2f4c3c25e7af487178c2d344b527d987f0571ddeb178b8a5b",
}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def key_sha(frame):
    array = pd.util.hash_pandas_object(frame[KEYS], index=False).to_numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def split_masks(frame, fold, contract, metadata=None):
    """Same whole-run-owned validation; train outside both 21-day purge margins.

    A positive run is excluded in its entirety when any part intersects the
    excluded interval. No rows are removed because their station/layer is unseen.
    """
    meta = evaluation.p1_run_metadata(frame, contract) if metadata is None else metadata
    times = pd.to_datetime(frame.time, utc=True)
    labels = frame.label.to_numpy()
    start, end = pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
    left, right = start - pd.Timedelta(days=21), end + pd.Timedelta(days=21)
    run_start = (
        pd.Series([stamp.value for stamp in times])
        .groupby(meta["run_id"]).transform("min").to_numpy()
    )
    temporal = ((times < left) | (times >= right)).to_numpy()
    whole_run_safe = (meta["end_ns"] < left.value) | (run_start >= right.value)
    train = temporal & ((labels == 0) | whole_run_safe)
    validation = (
        (labels == 0) & (times >= start).to_numpy() & (times < end).to_numpy()
    ) | ((labels == 1) & (meta["owner"] == fold["id"]))
    if not train.any() or not validation.any() or np.any(train & validation):
        raise ValueError("empty or overlapping proposed split")
    train_runs = set(meta["run_id"][train & (labels == 1)])
    if train_runs & set(meta["run_id"][validation & (labels == 1)]):
        raise ValueError("positive run shared across train/validation")
    for run in train_runs:
        if not train[meta["run_id"] == run].all():
            raise ValueError("partial positive training run")
    return train, validation, temporal, left, right


def main():
    started = time.perf_counter()
    if (REPORT / "result.json").exists():
        raise FileExistsError("preserve previous support receipt; no overwrite")
    for path, expected in PINS.items():
        if sha(ROOT / path) != expected:
            raise ValueError("pinned source changed: " + path)
    source = Path(os.environ["P1_DATA_DIR"]).resolve() / "train.csv"
    if sha(source) != TRAIN_SHA:
        raise ValueError("distributed training hash mismatch")
    # No observation feature values, official keys, old models or predictions.
    data = pd.read_csv(source, usecols=KEYS + ["label"])
    if len(data) != 776706 or data.duplicated(KEYS).any() or not data.label.isin([0, 1]).all():
        raise ValueError("invalid train metadata")
    data = data.sort_values(["station", "layer", "time"], kind="stable").reset_index(drop=True)
    contract = evaluation.load_contract()
    meta = evaluation.p1_run_metadata(data, contract)
    forward = json.loads((ROOT / "reports/p1_bracket_forward_20260906_v1/support.json").read_text(
        encoding="utf-8"
    ))
    if isinstance(forward, dict):
        forward = forward["folds"]
    output = []
    for fold in contract["P1"]["folds"]:
        if fold["warmup"]:
            continue
        train, validation, temporal, left, right = split_masks(data, fold, contract, meta)
        reference = next(x for x in forward if x["fold"] == fold["id"] and x["stage"] == "outer")
        keys_sha = key_sha(data.loc[validation])
        if keys_sha != reference["validation_keys_sha256"]:
            raise ValueError("validation keys/order differ from frozen forward comparison")
        known = set(zip(data.loc[train, "station"], data.loc[train, "layer"], strict=True))
        val = data.loc[validation]
        unseen = np.array([k not in known for k in zip(val.station, val.layer, strict=True)])
        groups = []
        for (station, layer), group in val.groupby(["station", "layer"], sort=True):
            mask = train & data.station.eq(station).to_numpy() & data.layer.eq(layer).to_numpy()
            groups.append({"station": station, "layer": int(layer), "train_rows": int(mask.sum()),
                           "validation_rows": len(group), "unseen": (station, layer) not in known})
        output.append({
            "fold": fold["id"], "train_rows": int(train.sum()), "validation_rows": len(val),
            "train_positive": int(data.loc[train, "label"].sum()),
            "validation_positive": int(val.label.sum()),
            "forward_train_rows": reference["train_rows"],
            "forward_unseen_station_layer_rows": reference["unseen_station_layer_rows"],
            "twosided_unseen_station_layer_rows": int(unseen.sum()),
            "twosided_unseen_fraction": float(unseen.mean()),
            "calendar_purge_or_validation_rows": int((~temporal).sum()),
            "whole_run_additional_excluded_rows": int((temporal & ~train).sum()),
            "train_keys_sha256": key_sha(data.loc[train]), "validation_keys_sha256": keys_sha,
            "same_forward_validation_keys": True, "station_layer_support": groups,
            "train_left_exclusive": left.isoformat(), "train_right_inclusive": right.isoformat(),
        })
    receipt = {
        "audit_id": ID, "status": "SUPPORT_AUDIT_COMPLETE_NOT_EVALUATOR_EXECUTION_READY",
        "proposal": "Separate retrospective two-sided train-label support; not forward forecasting.",
        "fits": 0, "metric_evaluations": 0, "official_rows": 0, "hidden_rows": 0,
        "feature_values_loaded": 0, "old_models_or_predictions_read": 0,
        "csv_written": 0, "uploads": 0, "source_sha256": TRAIN_SHA,
        "source_read_columns": KEYS + ["label"], "source_pins": PINS,
        "runner_sha256": sha(__file__), "folds": output,
        "unresolved_before_fit": [
            "Root must freeze separate prospective contract, including inner policy-selection folds.",
            "Train/inner/outer partition isolation and feature/decoder sentinel tests for two-sided context.",
            "No v1/v5 metric replacement, no automatic training, no score transfer or causal support claim.",
        ],
        "runtime_seconds": time.perf_counter() - started,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": receipt["status"], "fits": 0,
                      "unseen_rows": {x["fold"]: x["twosided_unseen_station_layer_rows"] for x in output}}))


if __name__ == "__main__":
    main()
