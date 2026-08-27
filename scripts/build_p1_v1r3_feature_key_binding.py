"""Create the one-time P1 v1r3 cache-key provenance sidecar.

This utility reads only the immutable P1 training source and the already pinned
training feature cache.  It performs no fit, score, prediction, candidate, or
submission operation.  Publication is create-only so a second invocation fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "20b656b0cbd524ad9da0bae8ecb6e0bacfc006e05810b37e83f29a5fa8e65cd2"
CACHE_RELATIVE = "artifacts/cache/train_offline_e9fe1eb46cb7431f.parquet"
CACHE_SHA256 = "f37c56ff016e90fb9a8d86299b4d9528c8f2e03181d326169b561fe3b27bc912"
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "p1_round_b_nonspike_long_event_residual_v1r3_preexecution"
)
SIDECAR_PATH = OUTPUT_ROOT / "feature_cache_row_keys.parquet"
RECEIPT_PATH = OUTPUT_ROOT / "feature_cache_row_keys_receipt.json"
KEY_COLUMNS = ("station", "year", "layer", "time")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_handle(handle: Any) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    handle.seek(0)
    return digest.hexdigest()


def _read_from_verified_handle(path: Path, expected: str, parser: Any) -> Any:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        first = _hash_handle(handle)
        if first != expected:
            raise RuntimeError(f"input digest mismatch: {path}")
        value = parser(handle)
        if handle.closed:
            raise RuntimeError(f"parser closed held input: {path}")
        second = _hash_handle(handle)
        after = os.fstat(handle.fileno())
    if (
        (before.st_size, before.st_mtime_ns, before.st_ino)
        != (after.st_size, after.st_mtime_ns, after.st_ino)
        or first != second
    ):
        raise RuntimeError(f"input changed during held-handle parse: {path}")
    return value


def _key_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for ordinal, row in enumerate(
        frame.loc[:, list(KEY_COLUMNS)].itertuples(index=False, name=None)
    ):
        values = (str(ordinal), str(row[0]), str(int(row[1])), str(int(row[2])), str(row[3]))
        for value in values:
            encoded = value.encode()
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
    return digest.hexdigest()


def _publish_temp_create_only(temporary: Path, target: Path) -> None:
    try:
        os.link(temporary, target)
    except FileExistsError:
        raise
    except OSError as error:
        raise RuntimeError("hardlink create-only publication is unavailable") from error
    temporary.unlink()


def _write_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            frame.to_parquet(handle, index=False, compression="zstd")
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temp_create_only(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temp_create_only(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    raw = os.environ.get("P1_DATA_DIR")
    if not raw:
        raise RuntimeError("P1_DATA_DIR is required")
    directory = Path(raw).expanduser().resolve(strict=True)
    source = (directory / "train.csv").resolve(strict=True)
    if source.parent != directory or source.name != "train.csv":
        raise RuntimeError("only the P1 training source is permitted")
    cache = (PROJECT_ROOT / CACHE_RELATIVE).resolve(strict=True)
    train = _read_from_verified_handle(
        source,
        SOURCE_SHA256,
        lambda handle: pd.read_csv(
            handle,
            usecols=[*KEY_COLUMNS, "temp", "psal", "depth"],
            low_memory=False,
        ),
    )
    features = _read_from_verified_handle(
        cache,
        CACHE_SHA256,
        lambda handle: pd.read_parquet(
            handle,
            columns=["station", "layer_category", "temp_raw", "psal_raw", "depth_raw"],
        ),
    )
    expected_index = pd.RangeIndex(start=0, stop=len(train), step=1)
    if not train.index.equals(expected_index) or not features.index.equals(expected_index):
        raise RuntimeError("source/cache index is not exact RangeIndex")
    if len(train) != len(features):
        raise RuntimeError("source/cache row count mismatch")
    if not np.array_equal(
        train["station"].astype(str).to_numpy(),
        features["station"].astype(str).to_numpy(),
    ):
        raise RuntimeError("source/cache station order mismatch")
    if not np.array_equal(
        train["layer"].astype(str).to_numpy(),
        features["layer_category"].astype(str).to_numpy(),
    ):
        raise RuntimeError("source/cache layer order mismatch")
    raw_checks: dict[str, bool] = {}
    for source_column, cache_column in (
        ("temp", "temp_raw"),
        ("psal", "psal_raw"),
        ("depth", "depth_raw"),
    ):
        left = pd.to_numeric(train[source_column], errors="coerce").to_numpy(dtype=np.float32)
        right = features[cache_column].to_numpy(dtype=np.float32)
        equal = bool(np.array_equal(left, right, equal_nan=True))
        raw_checks[cache_column] = equal
        if not equal:
            raise RuntimeError(f"source/cache raw order mismatch: {cache_column}")
    if train.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError("source exact keys are duplicated")
    key_digest = _key_digest(train)
    sidecar = train.loc[:, list(KEY_COLUMNS)].copy()
    sidecar.insert(0, "ordinal", np.arange(len(sidecar), dtype=np.int64))
    _write_parquet_new(SIDECAR_PATH, sidecar)
    receipt = {
        "schema_version": "p1_round_b_residual.feature_cache_key_binding.v1r3",
        "status": "SEALED_ZERO_FIT_SOURCE_CACHE_POSITIONAL_BINDING",
        "rows": len(sidecar),
        "source_sha256": SOURCE_SHA256,
        "feature_cache_sha256": CACHE_SHA256,
        "source_train_key_digest": key_digest,
        "feature_cache_key_digest": key_digest,
        "key_digests_equal": True,
        "range_index_exact": True,
        "station_layer_exact": True,
        "raw_float32_nan_aware_exact": raw_checks,
        "sidecar_path": str(SIDECAR_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "sidecar_bytes": SIDECAR_PATH.stat().st_size,
        "sidecar_sha256": _sha256(SIDECAR_PATH),
        "operation_counters": {
            "model_fits": 0,
            "scores": 0,
            "predictions": 0,
            "candidate_files": 0,
            "uploads": 0,
            "official_test_reads": 0,
            "sample_format_reads": 0,
            "submission_candidate_reads": 0,
        },
    }
    _write_json_new(RECEIPT_PATH, receipt)
    print(
        json.dumps(
            {
                "status": "ok",
                "rows": len(sidecar),
                "key_digest": key_digest,
                "sidecar_sha256": receipt["sidecar_sha256"],
                "receipt_sha256": _sha256(RECEIPT_PATH),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
