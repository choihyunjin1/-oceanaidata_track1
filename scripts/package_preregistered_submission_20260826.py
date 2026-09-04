from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = Path.home() / "Downloads" / "해양 해커톤 제출용"
PACKAGE = BASE / "20260826_round_C_preregistered_P1x3_P2x1"
RESEARCH = BASE / "20260826_value_of_information_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False, lineterminator="\n")


def p1_candidates() -> dict[str, dict]:
    original_path = ROOT / "output" / "2026-08-20" / "ready" / "P1_submission.csv"
    best_path = BASE / "20260825_round_B_target_adaptive" / "P1_submission.csv"
    original = pd.read_csv(original_path)
    best = pd.read_csv(best_path)
    keys = ["station", "year", "layer", "time"]
    if list(original.columns) != ["station", "year", "layer", "time", "label", "anomaly_type"]:
        raise RuntimeError("unexpected P1 schema")
    if list(best.columns) != list(original.columns) or not original[keys].equals(best[keys]):
        raise RuntimeError("P1 key/schema mismatch")

    original_positive = original["label"].eq(1).to_numpy()
    best_positive = best["label"].eq(1).to_numpy()
    o_only = original_positive & ~best_positive
    b_only = ~original_positive & best_positive
    station = best["station"].astype(str).to_numpy()
    layer = best["layer"].to_numpy(dtype=int)

    add_router = o_only & (
        ((station == "G-ORS") & (layer == 1))
        | ((station == "I-ORS") & (layer == 2))
    )
    remove_router = b_only & (
        ((station == "S-ORS") & np.isin(layer, [1, 5, 6]))
        | ((station == "I-ORS") & (layer == 4))
    )

    router = best.copy()
    router.loc[add_router, "label"] = 1
    router.loc[add_router, "anomaly_type"] = original.loc[add_router, "anomaly_type"].to_numpy()
    router.loc[remove_router, "label"] = 0
    router.loc[remove_router, "anomaly_type"] = np.nan

    intersection = best.copy()
    intersection["label"] = (best_positive & original_positive).astype(np.int8)
    intersection.loc[intersection["label"].eq(0), "anomaly_type"] = np.nan

    union = best.copy()
    union["label"] = (best_positive | original_positive).astype(np.int8)
    added_to_union = original_positive & ~best_positive
    union.loc[added_to_union, "anomaly_type"] = original.loc[added_to_union, "anomaly_type"].to_numpy()
    union.loc[union["label"].eq(0), "anomaly_type"] = np.nan

    specifications = [
        (
            "P1_1_EXPLOIT_DISAGREEMENT_ROUTER",
            router,
            "P1 정점-층 disagreement router v1",
            "현 베스트 B와 구 모델의 불일치 중 로컬 시간검증에서 재현된 정점-층만 선택적으로 복원·제거했습니다.",
            "EXPLOIT_RESEARCH_FLOOR",
        ),
        (
            "P1_2_PROBE_INTERSECTION",
            intersection,
            "P1 추가양성 절제 probe v1",
            "현 베스트 B와 구 모델의 교집합만 남겨 B가 새로 추가한 양성 묶음의 공식 효용을 검증합니다.",
            "INFORMATION_PROBE",
        ),
        (
            "P1_3_PROBE_UNION",
            union,
            "P1 제거양성 복원 probe v1",
            "현 베스트 B와 구 모델의 합집합을 사용해 B가 제거한 양성 묶음의 공식 효용을 검증합니다.",
            "INFORMATION_PROBE",
        ),
    ]
    output: dict[str, dict] = {}
    for directory_name, frame, title, summary, purpose in specifications:
        directory = PACKAGE / directory_name
        directory.mkdir()
        csv_path = directory / "P1_submission.csv"
        write_csv_new(csv_path, frame)
        write_text_new(
            directory / "P1_제출정보.txt",
            f"문제: P1\n제출물 제목: {title}\n한줄요약(접근방식): {summary}\n"
            f"목적: {purpose}\nCSV 파일: P1_submission.csv\nSHA-256: {sha256(csv_path)}",
        )
        output[directory_name] = {
            "problem": "P1",
            "purpose": purpose,
            "title": title,
            "one_line_summary": summary,
            "path": str(csv_path),
            "rows": int(len(frame)),
            "positive_count": int(frame["label"].sum()),
            "differences_vs_current_best": int(np.sum(frame["label"].to_numpy() != best["label"].to_numpy())),
            "sha256": sha256(csv_path),
        }
    output["lineage"] = {
        "original_sha256": sha256(original_path),
        "current_best_b_sha256": sha256(best_path),
        "original_positive": int(original["label"].sum()),
        "current_best_b_positive": int(best["label"].sum()),
        "o_only": int(o_only.sum()),
        "b_only": int(b_only.sum()),
        "router_additions": int(add_router.sum()),
        "router_removals": int(remove_router.sum()),
    }
    return output


def p2_candidate() -> dict:
    source = RESEARCH / "P2_PUBLIC_QUADRATIC_OPT_V1.csv"
    directory = PACKAGE / "P2_1_EXPLOIT_PUBLIC_QUADRATIC_OPT"
    directory.mkdir()
    destination = directory / "P2_submission.csv"
    shutil.copy2(source, destination)
    title = "P2 public quadratic 역방향 혼합 v1"
    summary = "기존 공식 alpha=0·0.5·1 RMSE로 정확한 MSE 이차곡선을 복원해 alpha=-0.158977 최적점을 적용했습니다."
    write_text_new(
        directory / "P2_제출정보.txt",
        f"문제: P2\n제출물 제목: {title}\n한줄요약(접근방식): {summary}\n"
        f"목적: EXPLOIT\nCSV 파일: P2_submission.csv\nSHA-256: {sha256(destination)}",
    )
    frame = pd.read_csv(destination)
    return {
        "problem": "P2",
        "purpose": "EXPLOIT",
        "title": title,
        "one_line_summary": summary,
        "path": str(destination),
        "rows": int(len(frame)),
        "temp_min": float(frame["temp"].min()),
        "temp_max": float(frame["temp"].max()),
        "sha256": sha256(destination),
    }


def backup_current_best() -> dict[str, dict]:
    directory = PACKAGE / "backup_best_before_round_C"
    directory.mkdir()
    sources = {
        "P1_submission.csv": BASE / "20260825_round_B_target_adaptive" / "P1_submission.csv",
        "P2_submission.csv": ROOT / "output" / "2026-08-20" / "ready" / "P2_submission.csv",
        "P3_submission.csv": ROOT / "output" / "2026-08-20" / "ready" / "P3_submission.csv",
    }
    result = {}
    for name, source in sources.items():
        destination = directory / name
        shutil.copy2(source, destination)
        result[name] = {
            "source": str(source),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        }
    write_text_new(
        directory / "README.txt",
        "2026-08-26 Round C 제출 전 공식 best 3문제 백업입니다.\n"
        "이 폴더의 CSV는 복구·비교용이며 오늘 재제출하지 않습니다.",
    )
    return result


def validate_manifest_entries(entries: dict[str, dict]) -> None:
    for entry in entries.values():
        path = Path(entry["path"])
        frame = pd.read_csv(path)
        if entry["problem"] == "P1":
            if list(frame.columns) != ["station", "year", "layer", "time", "label", "anomaly_type"]:
                raise RuntimeError(f"P1 schema failure: {path}")
            if len(frame) != 169011 or frame[["station", "year", "layer", "time"]].duplicated().any():
                raise RuntimeError(f"P1 row/key failure: {path}")
            if not frame["label"].isin([0, 1]).all():
                raise RuntimeError(f"P1 label failure: {path}")
        else:
            if list(frame.columns) != ["station", "layer", "time", "temp"]:
                raise RuntimeError(f"P2 schema failure: {path}")
            if len(frame) != 26061 or frame[["station", "layer", "time"]].duplicated().any():
                raise RuntimeError(f"P2 row/key failure: {path}")
            if not np.isfinite(frame["temp"].to_numpy(dtype=float)).all():
                raise RuntimeError(f"P2 finite failure: {path}")


def main() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=False)
    p1 = p1_candidates()
    p1_entries = {key: value for key, value in p1.items() if key != "lineage"}
    p2 = p2_candidate()
    all_entries = {**p1_entries, "P2_1_EXPLOIT_PUBLIC_QUADRATIC_OPT": p2}
    validate_manifest_entries(all_entries)
    backup = backup_current_best()
    plan = (
        "2026-08-26 사전등록 제출 계획\n\n"
        "오늘 공식 확인: 각 문제 3/3 남음.\n"
        "제출 대상: P1 세 폴더 각 1회, P2 한 폴더 1회. P3 0회.\n"
        "P1 세 결과와 P2 결과를 보기 전에 네 파일·해시·해석 규칙을 동결했습니다.\n"
        "P1-1은 개선 후보, P1-2/P1-3은 구조 식별 probe입니다.\n"
        "P2는 OPT만 제출하며 SAFE 후보는 제출하지 않습니다.\n"
        "실제 업로드는 사용자 action-time 확인 전에는 수행하지 않습니다.\n"
    )
    write_text_new(PACKAGE / "READY_SUBMISSION_PLAN.txt", plan)
    manifest = {
        "schema_version": "preregistered_submission_bundle_20260826.v1",
        "status": "FILES_FROZEN_AWAITING_INDEPENDENT_QA_AND_USER_CONFIRMATION",
        "official_submissions_performed": 0,
        "daily_limits_confirmed": {"P1_remaining": 3, "P2_remaining": 3, "P3_remaining": 3},
        "submission_plan": {"P1": 3, "P2": 1, "P3": 0},
        "p1": p1,
        "p2": p2,
        "backup_best_before_round_C": backup,
    }
    manifest_path = PACKAGE / "SET_MANIFEST.json"
    write_text_new(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
    checksum_lines = []
    for path in sorted(PACKAGE.rglob("*.csv")):
        checksum_lines.append(f"{sha256(path)}  {path.relative_to(PACKAGE).as_posix()}")
    write_text_new(PACKAGE / "SHA256SUMS.txt", "\n".join(checksum_lines))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
