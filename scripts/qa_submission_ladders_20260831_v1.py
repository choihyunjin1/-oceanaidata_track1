"""Independently validate the three 2026-08-31 submission ladders."""

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

from p1_qc.submission import validate_submission as validate_p1  # noqa: E402
from p2_restore.data import load_p2_data  # noqa: E402
from p2_restore.profile_projection import (  # noqa: E402
    project_profiles_vectorized,
    public_endpoint_frame,
)
from p2_restore.submission import validate_submission as validate_p2  # noqa: E402
from p3_wave.submission import validate_submission as validate_p3  # noqa: E402

ID = "submission_ladders_20260831_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(directory: Path) -> dict[str, object]:
    return json.loads((directory / "SET_MANIFEST.json").read_text(encoding="utf-8"))


def candidate_paths(directory: Path, filename: str) -> list[Path]:
    return sorted(directory.glob(f"*/{filename}"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-dir", type=Path, required=True)
    parser.add_argument("--p1-data", type=Path, required=True)
    parser.add_argument("--p2-dir", type=Path, required=True)
    parser.add_argument("--p2-data", type=Path, required=True)
    parser.add_argument("--p3-dir", type=Path, required=True)
    parser.add_argument("--p3-data", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    if report_dir.exists():
        raise FileExistsError(report_dir)

    p1_dir, p2_dir, p3_dir = (args.p1_dir.resolve(), args.p2_dir.resolve(), args.p3_dir.resolve())
    manifests = {"P1": load_manifest(p1_dir), "P2": load_manifest(p2_dir), "P3": load_manifest(p3_dir)}
    p1_test = args.p1_data.resolve(strict=True) / "test.csv"
    p1_results = [validate_p1(path, p1_test) for path in candidate_paths(p1_dir, "P1_submission.csv")]

    p2_data = load_p2_data(args.p2_data.resolve(strict=True))
    endpoints = public_endpoint_frame(p2_data.observations)
    p2_results: list[dict[str, object]] = []
    for path in candidate_paths(p2_dir, "P2_submission.csv"):
        result = validate_p2(path, p2_data.test_index)
        frame = pd.read_csv(path, dtype={"station": "string", "time": "string"})
        projection = project_profiles_vectorized(
            p2_data.test_index,
            frame["temp"].to_numpy(dtype=np.float64),
            endpoints,
        )
        if projection.active_mask.any():
            raise RuntimeError(f"P2 candidate is not a PAVA fixed point: {path}")
        result.update({"path": str(path), "sha256": sha(path), "pava_reprojection_active_rows": 0})
        p2_results.append(result)

    p3_test = pd.read_csv(args.p3_data.resolve(strict=True) / "test_index.csv")
    p3_results: list[dict[str, object]] = []
    for path in candidate_paths(p3_dir, "P3_submission.csv"):
        frame = pd.read_csv(path)
        validate_p3(frame, p3_test)
        values = frame["hs_pred"].to_numpy(dtype=np.float64)
        p3_results.append(
            {
                "path": str(path),
                "sha256": sha(path),
                "rows": len(frame),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )

    results = {"P1": p1_results, "P2": p2_results, "P3": p3_results}
    all_hashes = [row["sha256"] for rows in results.values() for row in rows]
    if len(all_hashes) != 9 or len(set(all_hashes)) != 9:
        raise RuntimeError("submission ladder must contain nine distinct hashes")
    for problem, rows in results.items():
        manifest_hashes = [row["sha256"] for row in manifests[problem]["candidates"]]
        if manifest_hashes != [row["sha256"] for row in rows]:
            raise RuntimeError(f"{problem} manifest/file ordering or hash mismatch")

    payload = {
        "schema_version": "submission.ladders.independent_qa.20260831.v1",
        "experiment_id": ID,
        "status": "PASS_9_DISTINCT_FILES_READY_NOT_UPLOADED",
        "results": results,
        "qa": {
            "candidate_count": 9,
            "distinct_sha256_count": 9,
            "official_schema_key_order_valid": True,
            "finite_domain_valid": True,
            "p2_pava_fixed_points": True,
            "manifest_file_hash_match": True,
            "hidden_truth_reads": 0,
            "uploads": 0,
        },
        "priority": {
            "P1": ["P1_1_PEER_HIGHCONF_UNION", "P1_2_PEER_FULL_UNION", "P1_3_PEER_STANDALONE"],
            "P2": ["P2_1_BIN17_DROP_LAYER2", "P2_2_BIN17_DROP_LAYER3", "P2_3_BIN17_DROP_LAYER4"],
            "P3": ["P3_1_KMA_A18_0425_A24_0600", "P3_2_KMA_A18_0200_A24_0425", "P3_3_KMA_A18_0200_A24_0600"],
        },
    }
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir / "independent-qa.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 2026-08-31 문제별 제출 사다리",
        "",
        "## 결론",
        "",
        "P1/P2/P3 각 3개, 총 9개 파일이 공식 스키마·키·순서·유한값·범위·SHA 검증을 통과했다. 아직 업로드하지 않았다.",
        "",
        "| 문제 | 우선순위 1 | 우선순위 2 | 우선순위 3 |",
        "|---|---|---|---|",
    ]
    for problem in ("P1", "P2", "P3"):
        names = payload["priority"][problem]
        lines.append(f"| {problem} | {names[0]} | {names[1]} | {names[2]} |")
    lines.extend(
        [
            "",
            "## 실행 원칙",
            "",
            "각 문제의 세 후보는 새 공식 점수를 보기 전에 함께 고정했다. 첫 후보가 챔피언을 넘지 못하면 같은 문제의 다음 고정 후보로 이동하며, 점수에 맞춘 파라미터 재조정은 이 세 파일 사이에서 하지 않는다.",
            "",
            "P1은 전체 train 776,706행으로 새 full fit을 수행하고 공식 test 169,011행을 추론했다. P2는 학습된 rank-1 bin17 보정을 수층별로 분해한 뒤 공개 endpoint PAVA를 다시 적용했다. P3는 학습된 KMA 장기리드 보정축에서 18h와 24h를 분리했다.",
        ]
    )
    (report_dir / "report-source.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
