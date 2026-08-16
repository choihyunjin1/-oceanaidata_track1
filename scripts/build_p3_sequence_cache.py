"""Build ignored raw-sequence arrays aligned to the all-20-minute feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p3_wave.data import load_p3_data
from p3_wave.sequences import build_test_sequences, build_train_sequences


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--feature-cache", default="artifacts/p3/features_all20_v1")
    parser.add_argument("--output-dir", default="artifacts/p3/sequences_all20_v1")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = load_p3_data(args.data_dir)
    anchors = pd.read_parquet(Path(args.feature_cache) / "train_anchors.parquet")
    train = build_train_sequences(data, anchors)
    test = build_test_sequences(data)
    files = {
        "train_values.npy": train.values,
        "train_station.npy": train.station_code,
        "test_values.npy": test.values,
        "test_station.npy": test.station_code,
    }
    manifest: dict[str, object] = {"arrays": {}}
    for name, value in files.items():
        path = output / name
        np.save(path, value, allow_pickle=False)
        manifest["arrays"][name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
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
