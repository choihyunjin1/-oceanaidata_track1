"""Build ignored, immutable-source P3 case-local feature caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

from p3_wave.data import audit_p3_data, load_p3_data
from p3_wave.features import build_test_features, build_training_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", default="artifacts/p3/features_all20_v1")
    parser.add_argument("--spacing-minutes", type=int, default=20)
    parser.add_argument("--status-file", default="artifacts/status/p3_recon.json")
    return parser.parse_args()


def _status(path: Path, progress: float, phase: str, detail: str) -> None:
    value = {
        "title": "P3 유의파고 예측 — 정찰 및 검증 구축",
        "phase": phase,
        "progress": progress,
        "eta": "로컬 계산 중",
        "detail": detail,
        "status": "running" if progress < 100 else "complete",
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status = Path(args.status_file)
    _status(status, 52, "20분 전체 eligible anchor 특징 생성", "공식 후보 조건 전체를 캐시")
    data = load_p3_data(args.data_dir)
    audit = audit_p3_data(data)
    start = time.perf_counter()

    def progress(done: int, total: int) -> None:
        fraction = done / max(total, 1)
        _status(
            status,
            52 + 22 * fraction,
            "20분 전체 eligible anchor 특징 생성",
            f"{done:,}/{total:,} contexts",
        )

    train = build_training_features(
        data, dense_spacing_minutes=args.spacing_minutes, progress=progress
    )
    test = build_test_features(data)
    paths = {
        "train_features": output / "train_features.parquet",
        "train_anchors": output / "train_anchors.parquet",
        "test_features": output / "test_features.parquet",
    }
    train.features.to_parquet(paths["train_features"], index=False, compression="zstd")
    train.anchors.to_parquet(paths["train_anchors"], index=False, compression="zstd")
    test.features.to_parquet(paths["test_features"], index=False, compression="zstd")
    manifest = {
        "audit": audit,
        "spacing_minutes": args.spacing_minutes,
        "train_rows": len(train.features),
        "test_rows": len(test.features),
        "feature_count": len(train.feature_columns),
        "elapsed_seconds": time.perf_counter() - start,
        "files": {
            name: {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in paths.items()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _status(status, 76, "정확한 독립 anchor 검증 준비", "전체 feature cache 생성 완료")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
