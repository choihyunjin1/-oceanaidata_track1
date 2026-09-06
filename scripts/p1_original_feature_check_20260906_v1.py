"""Rebuild original P1 covariates from released inputs, compare old cache only in QA."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(root, data, out):
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(root / "src"))
    from p1_qc.config import load_config
    from p1_qc.features import build_features

    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    config = load_config(root / "artifacts/runs/20260813T155254+0900_train_378a4e89/config.toml", env={})
    rows = {}
    for kind, name in [("train", "train_offline_e9fe1eb46cb7431f"),
                       ("test", "test_offline_c2a3877bdecea937")]:
        source = data / f"{kind}.csv"
        # Only the seven released covariates reach the feature builder.
        frame = pd.read_csv(source, usecols=["station", "year", "layer", "time", "temp", "psal", "depth"])
        before = time.time()
        fresh = build_features(frame, config=config)
        dest = out / f"{kind}_features.parquet"
        fresh.frame.to_parquet(dest, index=False, compression="zstd")
        # QA references are opened ONLY after the independent rebuild is saved.
        meta = json.loads((root / "artifacts/cache" / f"{name}.json").read_text())
        reference_path = root / "artifacts/cache" / f"{name}.parquet"
        if sha(source) != meta["source_sha256"] or sha(reference_path) != meta["parquet_sha256"]:
            raise RuntimeError("Original input/reference pin mismatch")
        old = pd.read_parquet(reference_path)
        if list(fresh.feature_columns) != meta["feature_columns"] or len(old) != len(frame):
            raise RuntimeError("Original feature schema/row mismatch")
        differences = {}
        for column in old.columns:
            a, b = fresh.frame[column], old[column]
            if column in fresh.categorical_columns:
                count = int((a.astype(str) != b.astype(str)).sum())
            else:
                av, bv = a.to_numpy(), b.to_numpy()
                count = int((~((av == bv) | (np.isnan(av) & np.isnan(bv)))).sum())
            if count:
                differences[column] = count
        rows[kind] = {"rows": len(frame), "features": len(old.columns),
                      "different_values_by_column": differences,
                      "all_values_exact": not differences, "sha256": sha(dest),
                      "reference_sha256": sha(reference_path), "source_sha256": sha(source),
                      "seconds": time.time() - before}
        print(json.dumps({"kind": kind, **rows[kind]}), flush=True)
    result = {"status": "RAW_TO_ORIGINAL_FEATURES_EXACT_PASS" if all(
        r["all_values_exact"] for r in rows.values()) else "FEATURE_REGEN_MISMATCH",
        "surfaces": rows, "model_fits": 0, "labels_read": 0,
        "cached_values_used_for_feature_generation": 0,
        "reference_cache_read_scope": "QA comparison after fresh features were saved",
        "candidate_csv_writes": 0, "uploads": 0, "runtime_seconds": time.time()-started,
        "script_sha256": sha(Path(__file__))}
    with (out / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(result["status"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.root.resolve(), a.data.resolve(), a.out.resolve())
