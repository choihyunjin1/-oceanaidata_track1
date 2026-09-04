"""Build the unique train-only causal raw-feature cache for P1 Gen4r2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p1_qc.causal_raw_features_v4r2 import (  # noqa: E402
    CAUSAL_FEATURE_COLUMNS,
    assert_future_value_invariance,
    build_causal_raw_features,
)

CANONICAL_ROOT = Path(r"C:\Users\cedis\PycharmProjects\PythonProject")
CANONICAL_DATA_DIR = CANONICAL_ROOT / "데이터셋 원본/데이터셋_P1/P1_qc_anomaly"
CACHE_RELATIVE = "artifacts/cache/train_causal_raw_prefix_safe_v4r2.parquet"
METADATA_RELATIVE = "artifacts/cache/train_causal_raw_prefix_safe_v4r2.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_location(root: Path, data_dir: Path) -> tuple[Path, Path]:
    lexical_root = Path(os.path.abspath(root))
    lexical_data = Path(os.path.abspath(data_dir))
    if lexical_root != CANONICAL_ROOT or root.resolve(strict=True) != CANONICAL_ROOT:
        raise PermissionError("cache build requires the exact canonical workspace")
    if lexical_data != CANONICAL_DATA_DIR or data_dir.resolve(strict=True) != CANONICAL_DATA_DIR:
        raise PermissionError("cache build requires the exact canonical P1 data directory")
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for path in (CANONICAL_ROOT, CANONICAL_DATA_DIR):
        if getattr(os.lstat(path), "st_file_attributes", 0) & reparse:
            raise PermissionError("canonical cache location cannot be a reparse point")
    return CANONICAL_ROOT, CANONICAL_DATA_DIR


def build(*, root: Path, data_dir: Path) -> dict[str, object]:
    root, data_dir = _assert_location(root, data_dir)
    cache = root / CACHE_RELATIVE
    metadata = root / METADATA_RELATIVE
    if cache.exists() or metadata.exists():
        raise FileExistsError("append-only causal cache target already exists")
    raw_path = data_dir / "train.csv"
    frame = pd.read_csv(
        raw_path,
        usecols=["station", "layer", "time", "temp", "psal", "depth"],
        low_memory=False,
    )
    features = build_causal_raw_features(frame)
    group = frame["station"].astype(str) + "|" + frame["layer"].astype(str)
    prefix_ids: list[int] = []
    parsed = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    audit = pd.DataFrame({"group": group, "time": parsed, "row": range(len(frame))})
    audit.sort_values(["group", "time", "row"], kind="mergesort", inplace=True)
    for _, rows in audit.groupby("group", sort=False, observed=True):
        keep = max(1, int(len(rows) * 0.7))
        prefix_ids.extend(rows.iloc[:keep]["row"].astype(int).tolist())
    invariance_sha = assert_future_value_invariance(
        frame, sorted(prefix_ids)
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(cache, index=False)
    value = {
        "schema_version": "p1_causal_raw_feature_cache.v4r2",
        "feature_columns": list(CAUSAL_FEATURE_COLUMNS),
        "row_count": int(len(features)),
        "source_train_sha256": _sha(raw_path),
        "builder_module_sha256": _sha(root / "src/p1_qc/causal_raw_features_v4r2.py"),
        "future_value_perturbation_invariant": True,
        "future_value_perturbation_prefix_feature_sha256": invariance_sha,
        "target_columns_read": 0,
        "forward_or_centered_operations": 0,
    }
    with metadata.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return {
        "cache": CACHE_RELATIVE,
        "cache_sha256": _sha(cache),
        "cache_bytes": cache.stat().st_size,
        "metadata": METADATA_RELATIVE,
        "metadata_sha256": _sha(metadata),
        "metadata_bytes": metadata.stat().st_size,
        "row_count": len(features),
        "feature_count": len(CAUSAL_FEATURE_COLUMNS),
        "future_value_perturbation_invariant": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(root=args.root, data_dir=args.data_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
