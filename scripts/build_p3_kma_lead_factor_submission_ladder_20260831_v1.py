"""Materialize three frozen P3 lead-factor candidates from the trained KMA axis."""

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

from p3_wave.kma_alpha_surface import apply_official_correction  # noqa: E402

ID = "p3_kma_lead_factor_submission_ladder_20260831_v1"
CONFIG = ROOT / "configs/experiments" / f"{ID}.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-base", type=Path, required=True)
    parser.add_argument("--axis-old", type=Path, required=True)
    parser.add_argument("--kma-alpha40", type=Path, required=True)
    parser.add_argument("--current-champion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("--execute required")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    current_base_path = args.current_base.resolve(strict=True)
    axis_old_path = args.axis_old.resolve(strict=True)
    kma_path = args.kma_alpha40.resolve(strict=True)
    champion_path = args.current_champion.resolve(strict=True)
    current_base = pd.read_csv(current_base_path)
    axis_old = pd.read_csv(axis_old_path)
    kma = pd.read_csv(kma_path)
    champion = pd.read_csv(champion_path)
    expected = config["contract"]["columns"]
    for name, frame in {
        "current_base": current_base,
        "axis_old": axis_old,
        "kma": kma,
        "champion": champion,
    }.items():
        if list(frame.columns) != expected or len(frame) != config["contract"]["rows"]:
            raise RuntimeError(f"P3 {name} contract changed")
    # Reproduce the official 0.425 champion exactly before deriving any arm.
    reproduced = apply_official_correction(
        current_base,
        axis_old,
        kma,
        alpha_by_lead={18: 0.425, 24: 0.425},
        reference_alpha=float(config["reference_alpha"]),
    )
    reproduced_payload = reproduced.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if hashlib.sha256(reproduced_payload).hexdigest() != sha(champion_path):
        raise RuntimeError("P3 0.425 official champion failed exact reproduction")
    output.mkdir(parents=True, exist_ok=False)
    champion_value = champion["hs_pred"].to_numpy(dtype=np.float64)
    seen = {sha(current_base_path), sha(axis_old_path), sha(kma_path), sha(champion_path)}
    records: list[dict[str, object]] = []
    for spec in config["candidates"]:
        candidate = apply_official_correction(
            current_base,
            axis_old,
            kma,
            alpha_by_lead={18: float(spec["alpha_18"]), 24: float(spec["alpha_24"])},
            reference_alpha=float(config["reference_alpha"]),
        )
        values = candidate["hs_pred"].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 30:
            raise RuntimeError("P3 candidate value contract failed")
        payload = candidate.to_csv(index=False, lineterminator="\n").encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen:
            raise RuntimeError("duplicate P3 candidate")
        seen.add(digest)
        directory = output / spec["name"]
        csv_path = directory / "P3_submission.csv"
        write_new(csv_path, payload)
        delta = values - champion_value
        record = {
            **spec,
            "path": str(csv_path),
            "sha256": digest,
            "rows": len(candidate),
            "changed_rows_vs_champion": int(np.sum(np.abs(delta) > 1e-12)),
            "rms_change_vs_champion_m": float(np.sqrt(np.mean(np.square(delta)))),
            "minimum_m": float(values.min()),
            "maximum_m": float(values.max()),
            "status": "TRAINED_AXIS_TEST_MATERIALIZED_READY_NOT_UPLOADED"
        }
        write_new(
            directory / "제출정보.txt",
            (
                f"제출물 제목: {spec['title']}\n한줄요약(접근방식): {spec['summary']}\n"
                f"파일 SHA-256: {digest}\n상태: TRAINED_AXIS_TEST_MATERIALIZED_READY_NOT_UPLOADED\n"
            ).encode("utf-8-sig"),
        )
        records.append(record)
    manifest = {
        "schema_version": "p3.kma_lead_factor_submission_ladder.prepared.20260831.v1",
        "experiment_id": ID,
        "status": "TRAINED_AXIS_TEST_MATERIALIZED_READY_NOT_UPLOADED",
        "config_sha256": sha(CONFIG),
        "source_hashes": {
            "current_base": sha(current_base_path),
            "axis_old": sha(axis_old_path),
            "kma_alpha40": sha(kma_path),
            "current_champion": sha(champion_path),
        },
        "candidates": records,
        "qa": {
            "official_champion_exactly_reproduced": True,
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
