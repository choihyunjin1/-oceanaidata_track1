"""Append event-phase features to ignored P3 feature caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p3_wave.event_phase import build_event_phase_features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", default="artifacts/p3/features_all20_v1")
    parser.add_argument("--sequence-cache", default="artifacts/p3/sequences_all20_v1")
    parser.add_argument("--output-dir", default="artifacts/p3/features_event_v1")
    args = parser.parse_args()
    source = Path(args.feature_cache)
    sequence = Path(args.sequence_cache)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(source / "train_features.parquet")
    anchors = pd.read_parquet(source / "train_anchors.parquet")
    test = pd.read_parquet(source / "test_features.parquet")
    train_event = build_event_phase_features(np.load(sequence / "train_values.npy", mmap_mode="r"))
    test_event = build_event_phase_features(np.load(sequence / "test_values.npy", mmap_mode="r"))
    if len(train_event) != len(train) or len(test_event) != len(test):
        raise ValueError("event feature alignment mismatch")
    train = pd.concat([train.reset_index(drop=True), train_event], axis=1)
    test = pd.concat([test.reset_index(drop=True), test_event], axis=1)
    paths = {
        "train_features.parquet": train,
        "train_anchors.parquet": anchors,
        "test_features.parquet": test,
    }
    manifest: dict[str, object] = {
        "event_feature_count": len(train_event.columns),
        "event_columns": list(train_event.columns),
        "files": {},
    }
    for name, frame in paths.items():
        path = output / name
        frame.to_parquet(path, index=False, compression="zstd")
        manifest["files"][name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
