"""Train the frozen peer-gate P1 model on all train rows and build three submissions.

The old outer-CV selection is treated as a frozen nuisance contract.  This
runner performs one new full-data fit, one official-test inference pass, and
materializes three predeclared candidates without reading hidden labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))

from p1_qc.config import load_config  # noqa: E402
from p1_qc.data import load_train_test  # noqa: E402
from p1_qc.pipeline import (  # noqa: E402
    load_or_build_features,
    predict_submission,
    train_full_model,
)
from p1_qc.stratification import PeerGateConfig, append_stratification_peer_gate  # noqa: E402

EXPERIMENT_ID = "p1_peer_full_train_submission_ladder_20260831_v1"
SELECTION = (
    ROOT
    / "artifacts/runs/20260813T205237+0900_strat_gate_fixed24h_59f6d5c6/selection.json"
)
CONFIG = ROOT / "configs/p1.toml"
KEYS = ["station", "year", "layer", "time"]
EXPECTED_ROWS = 169_011
EXPECTED_SELECTION_SHA256 = "9da6416a5b5c48a5845b2afd00479e35a101b860eb7c76217efad1fb6b1731c5"
GATE = PeerGateConfig(mode="offline", window_hours=24, min_period_fraction=0.5)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def candidate_payload(test: pd.DataFrame, label: np.ndarray) -> bytes:
    frame = test[KEYS].copy()
    frame["label"] = np.asarray(label, dtype=np.int8)
    if len(frame) != EXPECTED_ROWS or frame.duplicated(KEYS).any():
        raise RuntimeError("P1 candidate row/key contract failed")
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise RuntimeError("P1 label domain failed")
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("--execute required")

    data_dir = args.data_dir.resolve(strict=True)
    champion_path = args.champion.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if sha256_file(SELECTION) != EXPECTED_SELECTION_SHA256:
        raise RuntimeError("frozen peer-gate selection changed")

    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection["backend"] != "xgboost" or int(selection["iteration_count"]) != 700:
        raise RuntimeError("unexpected frozen selection")
    config = load_config(CONFIG, env={"P1_DATA_DIR": str(data_dir)})
    started = time.perf_counter()
    train, test = load_train_test(data_dir, audit=True, strict=True)
    raw_test = pd.read_csv(
        data_dir / "test.csv",
        dtype={"station": "string", "time": "string"},
        keep_default_na=False,
    )
    if len(test) != EXPECTED_ROWS:
        raise RuntimeError("official P1 test row count changed")

    champion = pd.read_csv(
        champion_path,
        dtype={"station": "string", "time": "string", "label": "int8"},
        keep_default_na=False,
    )
    if list(champion.columns) != [*KEYS, "label"]:
        raise RuntimeError("champion schema changed")
    if len(champion) != len(raw_test) or not champion[KEYS].equals(raw_test[KEYS]):
        raise RuntimeError("champion key/order differs from official test")
    champion_label = champion["label"].to_numpy(dtype=np.int8)

    train_base = load_or_build_features(train, config, kind="train", use_cache=True)
    test_base = load_or_build_features(test, config, kind="test", use_cache=True)
    train_bundle = append_stratification_peer_gate(
        train_base,
        train,
        config=GATE,
        cadence_minutes=config.data.cadence_minutes,
        group_columns=config.data.group_columns,
    )
    test_bundle = append_stratification_peer_gate(
        test_base,
        test,
        config=GATE,
        cadence_minutes=config.data.cadence_minutes,
        group_columns=config.data.group_columns,
    )

    model = train_full_model(train, train_bundle, config, selection)
    model_submission, probability = predict_submission(model, test, test_bundle)
    model_label = model_submission["label"].to_numpy(dtype=np.int8)
    if not np.isfinite(probability).all():
        raise RuntimeError("non-finite P1 probability")

    specifications = [
        {
            "priority": 1,
            "name": "P1_1_PEER_HIGHCONF_UNION",
            "title": "P1 동적 peer 고신뢰 챔피언 결합",
            "summary": "전체 train 재학습 peer-coherence XGBoost 중 확률 0.50 이상 탐지만 현재 챔피언에 추가합니다.",
            "label": np.maximum(champion_label, model_label * (probability >= 0.50)),
        },
        {
            "priority": 2,
            "name": "P1_2_PEER_FULL_UNION",
            "title": "P1 동적 peer 전체 챔피언 결합",
            "summary": "전체 train 재학습 peer-coherence XGBoost 탐지를 현재 챔피언과 합집합으로 결합합니다.",
            "label": np.maximum(champion_label, model_label),
        },
        {
            "priority": 3,
            "name": "P1_3_PEER_STANDALONE",
            "title": "P1 동적 peer 전체학습 단독",
            "summary": "24시간 cross-layer 변화 일치도 특징을 포함한 XGBoost를 전체 train으로 학습해 단독 추론합니다.",
            "label": model_label,
        },
    ]

    output_dir.mkdir(parents=True, exist_ok=False)
    seen = {sha256_file(champion_path)}
    records: list[dict[str, object]] = []
    for spec in specifications:
        payload = candidate_payload(raw_test, np.asarray(spec.pop("label"), dtype=np.int8))
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen:
            raise RuntimeError(f"duplicate P1 candidate: {spec['name']}")
        seen.add(digest)
        directory = output_dir / str(spec["name"])
        csv_path = directory / "P1_submission.csv"
        write_new(csv_path, payload)
        label = pd.read_csv(csv_path, usecols=["label"])["label"].to_numpy(dtype=np.int8)
        changed = label != champion_label
        additions = (label == 1) & (champion_label == 0)
        removals = (label == 0) & (champion_label == 1)
        record = {
            **spec,
            "path": str(csv_path),
            "sha256": digest,
            "bytes": len(payload),
            "rows": len(label),
            "positive_rows": int(label.sum()),
            "changed_rows_vs_champion": int(changed.sum()),
            "additions_vs_champion": int(additions.sum()),
            "removals_vs_champion": int(removals.sum()),
            "status": "TRAINED_TEST_INFERRED_READY_NOT_UPLOADED",
        }
        note = (
            f"제출물 제목: {spec['title']}\n"
            f"한줄요약(접근방식): {spec['summary']}\n"
            f"파일 SHA-256: {digest}\n"
            "상태: TRAINED_TEST_INFERRED_READY_NOT_UPLOADED\n"
        ).encode("utf-8-sig")
        write_new(directory / "제출정보.txt", note)
        records.append(record)

    manifest = {
        "schema_version": "p1.peer_full_train_submission_ladder.20260831.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "TRAINED_TEST_INFERRED_READY_NOT_UPLOADED",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": 1,
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": len(train_bundle.feature_columns),
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "champion_sha256": sha256_file(champion_path),
        "source_hashes": {
            "train": str(train.attrs.get("source_sha256")),
            "test": str(test.attrs.get("source_sha256")),
        },
        "probability_aggregates": {
            "minimum": float(probability.min()),
            "maximum": float(probability.max()),
            "mean": float(probability.mean()),
        },
        "candidates": records,
        "qa": {
            "full_train_fit": True,
            "official_test_inference": True,
            "hidden_truth_reads": 0,
            "candidate_hashes_distinct": True,
            "schema_key_order_binary": True,
            "uploads": 0,
        },
    }
    write_new(
        output_dir / "SET_MANIFEST.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
