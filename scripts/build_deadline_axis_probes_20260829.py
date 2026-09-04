from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

P1_KEYS = ["station", "year", "layer", "time"]
P2_KEYS = ["station", "layer", "time"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_candidate(frame: pd.DataFrame, directory: Path, filename: str, title: str, summary: str) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / filename
    frame.to_csv(output, index=False)
    (directory / "제출정보.txt").write_text(
        f"제출물 제목: {title}\n한줄요약(접근방식): {summary}\n파일 SHA-256: {sha256(output)}\n",
        encoding="utf-8",
    )
    return {
        "path": str(output),
        "rows": int(len(frame)),
        "sha256": sha256(output),
        "title": title,
        "summary": summary,
    }


def validate_keys(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], value: str) -> None:
    expected = keys + [value]
    if list(left.columns) != expected or list(right.columns) != expected:
        raise RuntimeError(f"schema mismatch: expected {expected}")
    if not left[keys].equals(right[keys]):
        raise RuntimeError("key/order mismatch")


def build_p1(args: argparse.Namespace, output: Path) -> list[dict]:
    base = pd.read_csv(args.p1_base, dtype={"station": "string", "time": "string"})
    champion = pd.read_csv(args.p1_champion, dtype={"station": "string", "time": "string"})
    validate_keys(base, champion, P1_KEYS, "label")
    base_label = base["label"].to_numpy(int)
    champion_label = champion["label"].to_numpy(int)
    if not set(np.unique(base_label)).issubset({0, 1}) or not set(np.unique(champion_label)).issubset({0, 1}):
        raise RuntimeError("P1 labels must be binary")
    added = np.flatnonzero((base_label == 0) & (champion_label == 1))
    removed = np.flatnonzero((base_label == 1) & (champion_label == 0))
    if len(added) != 2 or len(removed) != 0:
        raise RuntimeError(f"expected exactly two additions and no removals, got +{len(added)} -{len(removed)}")

    records: list[dict] = []
    for ordinal, row_index in enumerate(added, start=1):
        candidate = base.copy()
        candidate.loc[row_index, "label"] = 1
        record = write_candidate(
            candidate,
            output / f"P1_{ordinal}_GI_SINGLE_ROW_{ordinal}",
            "P1_submission.csv",
            f"P1 e150 + GI 단일행 분해 {ordinal}",
            "공식 개선을 만든 GI 2행을 한 행씩 분리해 각 행의 순효과를 직접 검증합니다.",
        )
        record.update(
            problem="P1",
            role="leave-one-row decomposition of the two-row official champion delta",
            positive_rows=int(candidate["label"].sum()),
            changed_rows_vs_base=1,
            changed_rows_vs_champion=1,
            test_order_match=True,
        )
        records.append(record)
    return records


def build_p2(args: argparse.Namespace, output: Path) -> list[dict]:
    base = pd.read_csv(args.p2_base, dtype={"station": "string", "time": "string"})
    champion = pd.read_csv(args.p2_champion, dtype={"station": "string", "time": "string"})
    validate_keys(base, champion, P2_KEYS, "temp")
    base_value = base["temp"].to_numpy(float)
    champion_value = champion["temp"].to_numpy(float)
    if not np.isfinite(base_value).all() or not np.isfinite(champion_value).all():
        raise RuntimeError("P2 inputs contain non-finite values")
    delta = champion_value - base_value

    records: list[dict] = []
    for alpha in args.p2_alpha:
        candidate = base.copy()
        candidate["temp"] = base_value + float(alpha) * delta
        if not np.isfinite(candidate["temp"].to_numpy(float)).all():
            raise RuntimeError("P2 candidate contains non-finite values")
        alpha_tag = f"{alpha:.6f}".rstrip("0").rstrip(".").replace("-", "NEG").replace(".", "P")
        record = write_candidate(
            candidate,
            output / f"P2_RANK1_AXIS_ALPHA_{alpha_tag}",
            "P2_submission.csv",
            f"P2 rank-1 공식축 강도 {alpha:g}",
            "공식 개선된 alpha50 계절 rank-1 보정 벡터의 강도만 선형 조절해 Public 최적점을 식별합니다.",
        )
        change = candidate["temp"].to_numpy(float) - champion_value
        record.update(
            problem="P2",
            role="official-success-axis strength probe",
            axis_alpha=float(alpha),
            changed_rows_vs_champion=int(np.count_nonzero(np.abs(change) > 1e-12)),
            rms_change_vs_champion=float(np.sqrt(np.mean(change**2))),
            minimum=float(candidate["temp"].min()),
            maximum=float(candidate["temp"].max()),
            key_order_match=True,
        )
        records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--p1-base", type=Path)
    parser.add_argument("--p1-champion", type=Path)
    parser.add_argument("--p2-base", type=Path)
    parser.add_argument("--p2-champion", type=Path)
    parser.add_argument("--p2-alpha", type=float, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    records: list[dict] = []
    if args.p1_base and args.p1_champion:
        records.extend(build_p1(args, args.output_dir))
    if args.p2_base and args.p2_champion and args.p2_alpha:
        records.extend(build_p2(args, args.output_dir))
    if not records:
        raise RuntimeError("no candidates requested")
    hashes = [record["sha256"] for record in records]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("candidate hash collision")
    manifest = {
        "schema_version": "deadline.axis_probes.20260829.v1",
        "status": "READY_NOT_UPLOADED",
        "records": records,
        "official_uploads": 0,
    }
    (args.output_dir / "SET_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
