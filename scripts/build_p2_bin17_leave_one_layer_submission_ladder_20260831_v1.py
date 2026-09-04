"""Build three frozen leave-one-layer-out P2 bin17 submission candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))

from p2_restore.profile_projection import (  # noqa: E402
    project_profiles_vectorized,
    public_endpoint_frame,
)

ID = "p2_bin17_leave_one_layer_submission_ladder_20260831_v1"
CONFIG = ROOT / "configs/experiments" / f"{ID}.json"
KEYS = ["station", "layer", "time"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-dir", type=Path, required=True)
    parser.add_argument("--alpha50", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("--execute required")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["experiment_id"] != ID:
        raise RuntimeError("config ID changed")
    paths = {
        "observations": args.p2_dir.resolve(strict=True) / "observations.csv",
        "test_index": args.p2_dir.resolve(strict=True) / "test_index.csv",
        "sample_submission": args.p2_dir.resolve(strict=True) / "sample_submission.csv",
        "alpha50": args.alpha50.resolve(strict=True),
        "champion": args.champion.resolve(strict=True),
    }
    for name, path in paths.items():
        if sha(path) != config["source_pins"][f"{name}_sha256"]:
            raise RuntimeError(f"source pin mismatch: {name}")
    observations = pd.read_csv(
        paths["observations"], dtype={"station": "string", "time": "string"}
    )
    endpoints = public_endpoint_frame(observations)
    test = pd.read_csv(paths["test_index"], dtype={"station": "string", "time": "string"})
    sample = pd.read_csv(paths["sample_submission"], dtype={"station": "string", "time": "string"})
    alpha = pd.read_csv(paths["alpha50"], dtype={"station": "string", "time": "string"})
    champion = pd.read_csv(paths["champion"], dtype={"station": "string", "time": "string"})
    if list(test.columns) != config["contract"]["test_columns"] or len(test) != 26061:
        raise RuntimeError("official P2 test contract changed")
    for name, frame in {"sample": sample, "alpha50": alpha, "champion": champion}.items():
        if list(frame.columns) != config["contract"]["output_columns"]:
            raise RuntimeError(f"schema mismatch: {name}")
        if len(frame) != len(test) or not frame[KEYS].equals(test[KEYS]):
            raise RuntimeError(f"key/order mismatch: {name}")
    times = pd.to_datetime(test["time"], utc=True).dt.tz_convert("Asia/Seoul")
    bins = ((times.dt.dayofyear.to_numpy() - 1) // 14).astype(int)
    layer = test["layer"].to_numpy(dtype=int)
    anchor = alpha["temp"].to_numpy(dtype=np.float64)
    champion_value = champion["temp"].to_numpy(dtype=np.float64)
    correction = champion_value - anchor
    if np.any((np.abs(correction) > 1e-12) & (bins != 17)):
        raise RuntimeError("current champion correction escaped bin17")
    output.mkdir(parents=True, exist_ok=False)
    seen = {sha(paths["champion"]), sha(paths["alpha50"])}
    records: list[dict[str, object]] = []
    for spec in config["recipe"]["candidates"]:
        disabled = np.isin(layer, spec["disabled_layers"]) & (bins == 17)
        unprojected = np.where(disabled, anchor, champion_value)
        projection = project_profiles_vectorized(test, unprojected, endpoints)
        values = projection.prediction
        if not np.isfinite(values).all() or values.min() < -5 or values.max() > 45:
            raise RuntimeError("candidate value contract failed")
        frame = test[KEYS].copy()
        frame["temp"] = values
        payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen:
            raise RuntimeError("duplicate P2 candidate")
        seen.add(digest)
        directory = output / spec["name"]
        csv_path = directory / "P2_submission.csv"
        write_new(csv_path, payload)
        delta = values - champion_value
        record = {
            **spec,
            "path": str(csv_path),
            "sha256": digest,
            "rows": len(frame),
            "changed_rows_vs_champion": int(np.sum(np.abs(delta) > 1e-12)),
            "pava_active_rows": int(projection.active_mask.sum()),
            "rms_change_vs_champion_c": float(np.sqrt(np.mean(np.square(delta)))),
            "minimum_c": float(values.min()),
            "maximum_c": float(values.max()),
            "status": "TRAINED_LINEAGE_TEST_MATERIALIZED_READY_NOT_UPLOADED"
        }
        write_new(
            directory / "제출정보.txt",
            (
                f"제출물 제목: {spec['title']}\n한줄요약(접근방식): {spec['summary']}\n"
                f"파일 SHA-256: {digest}\n상태: TRAINED_LINEAGE_TEST_MATERIALIZED_READY_NOT_UPLOADED\n"
            ).encode("utf-8-sig"),
        )
        records.append(record)
    manifest = {
        "schema_version": "p2.bin17_leave_one_layer_submission_ladder.prepared.20260831.v1",
        "experiment_id": ID,
        "status": "TRAINED_LINEAGE_TEST_MATERIALIZED_READY_NOT_UPLOADED",
        "config_sha256": sha(CONFIG),
        "source_hashes": {name: sha(path) for name, path in paths.items()},
        "candidates": records,
        "qa": {
            "frozen_together_before_score": True,
            "schema_key_order_finite_domain": True,
            "candidate_hashes_distinct": True,
            "hidden_truth_reads": 0,
            "uploads": 0
        }
    }
    write_new(output / "SET_MANIFEST.json", (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
