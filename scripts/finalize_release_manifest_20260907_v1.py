"""Finalize local attachment forms only after exact archive replay receipts exist."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from build_final_release_20260907_v1 import read, save, sha

SPECS = {
    "P1": ("P1_submission.csv", 169011, ["station", "year", "layer", "time", "label"], "F1 0.833548", "28.909341", "Original O+B+MS-TCN source-retrained"),
    "P2": ("submission_p2_L120_3seed.csv", 26061, ["station", "layer", "time", "temp"], "RMSE 0.418892 C", "28.077280", "C3 DeepSet L120 3-seed"),
    "P3": ("submission_scored_numeric.csv", 1200, ["case_id", "station", "lead_h", "hs_pred"], "RMSE 0.604351 m", "23.741446", "CatBoost numeric lead + train-OOF router"),
}


def answer_check(path, expected_rows, columns, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError("selected answer mismatch")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader) != columns:
            raise ValueError("answer columns mismatch")
        rows = 0
        keys = set()
        for row in reader:
            if len(row) != len(columns):
                raise ValueError("answer width mismatch")
            key = tuple(row[:-1])
            if key in keys:
                raise ValueError("duplicate answer key")
            keys.add(key)
            value = float(row[-1])
            if not math.isfinite(value):
                raise ValueError("nonfinite prediction")
            if columns[-1] == "label" and row[-1] not in {"0", "1"}:
                raise ValueError("nonbinary label")
            rows += 1
    if rows != expected_rows:
        raise ValueError("answer row count mismatch")
    return {"rows": rows, "columns": columns, "sha256": expected_sha, "duplicate_keys": 0, "nonfinite": 0}


def finalize(root, p3_evidence, output):
    candidates = {}
    for problem in ("P1", "P2"):
        build = read(root / problem / "BUILD_QA.json")
        replay = read(root / f"{problem}_saved_validation-receipt.json")
        if replay["status"] != "EXTRACTED_SAVED_NOTEBOOK_EXACT_ANSWER_PASS":
            raise ValueError("saved notebook replay incomplete")
        candidates[problem] = {"full_cold_fits": build["original_full_cold_fits"],
            "full_cold_seconds": build["original_full_cold_seconds"],
            "saved_inference_seconds": replay["runtime_seconds"],
            "saved_inference_status": replay["status"], "answer_sha256": build["answer_sha256"],
            "archives": [{k: a[k] for k in ("file", "sha256", "bytes", "files")} for a in build["archives"]],
            "split_parts": build["split_parts"]}
    candidates["P3"] = read(p3_evidence)
    if candidates["P3"]["saved_inference_status"] != "EXACT_SAVED_REPLAY_PASS":
        raise ValueError("P3 saved archive evidence incomplete")
    for index, (problem, info) in enumerate(candidates.items(), 5):
        name, count, columns, metric, score, title = SPECS[problem]
        info["answer"] = answer_check(root / problem / "ANSWER" / name, count, columns, info["answer_sha256"])
        info["answer_file"] = f"{problem}/ANSWER/{name}"
        info["official_metric"], info["official_score"] = metric, score
        for record in info["archives"]:
            if sha(root / problem / record["file"]) != record["sha256"]:
                raise ValueError("final archive changed")
        parts = "P1 대용량 저장 ZIP 대신 SAVED_MODEL_PARTS의 모든 part ZIP, manifest, reassemble.py를 함께 첨부할 수 있습니다. 분할은 독립 모델이 아닙니다.\n" if problem == "P1" else ""
        limitation = ("\n주의: P3 SOURCE_ONLY 재학습은 완료됐으나 ff42 채점본과 660행이 다릅니다(최대 0.003506091m). "
                      "동봉 SAVED_MODELS는 ff42 원 채점 가중치이며 정확 재생을 통과했습니다. "
                      "새 cold 파일 NOT_SCORED_whole_cold.csv는 미채점 검증본입니다. 운영진의 재학습 허용오차/판정은 미확인입니다.\n"
                      if problem == "P3" else "")
        text = (f"# {problem} 제출 양식\n\n"
            f"문제 페이지: https://oceanaidata.org/app/problems/{index}\n\n"
            f"제목: {problem} {title}\n\n"
            f"한 줄 요약: 운영진 배포자료만으로 학습한 {title}. 학습 코드·가중치·별도 프로세스 답안 재생 검증 동봉.\n\n"
            f"답안 채점 카드: ANSWER/{name}, {count:,}행, {','.join(columns)}\n\n"
            f"SHA256: {info['answer_sha256']}\n\n"
            f"같은 SHA에 연결된 기존 공식 기록: {metric} / {score}점. 새 채점을 수행한 기록이 아닙니다.\n\n"
            "최종 모델·재현 첨부:\n\n" + "\n".join(f"- {a['file']} — SHA256 {a['sha256']}" for a in info["archives"])
            + "\n- FORM.md 및 상위 RELEASE_MANIFEST.json/START_HERE.md\n\n" + parts
            + "\nSOURCE_ONLY는 새 빈 폴더에서 전체 학습, SAVED_MODELS는 동봉 모델로 새 출력에 추론합니다. 각 ZIP 역할을 혼동하지 마세요.\n" + limitation
            + "\n현재 포털 첨부 제한과 최종 잠금 경고를 확인한 뒤 지정하세요. 이 준비 작업은 공식 최종 제출·업로드를 하지 않았습니다.\n"
            + "\n저장소: https://github.com/choihyunjin1/-oceanaidata_track1/tree/codex/p1-qc\n")
        (root / problem / "FORM.md").write_text(text, encoding="utf-8")
    result = {"status": "LOCAL_RELEASE_ARCHIVES_AND_SAVED_REPLAY_VERIFIED", "candidates": candidates,
              "raw_data_in_github": 0, "weights_answers_archives_in_github": 0, "uploads": 0,
              "final_model_designations": 0, "fresh_venv_OS_offline_test": False,
              "portal_attachment_limits_rechecked": False,
              "portal_read_blocker": "Chrome Debugger unattached; no final action attempted"}
    save(root / "RELEASE_MANIFEST.json", result)
    save(output, result)
    (root / "START_HERE.md").write_text(
        "# 최종 패키지 — 2026-09-07\n\n"
        "P1/P2/P3 각각 FORM.md에서 정확한 답안·첨부 ZIP을 선택하세요. 공식 최종 지정은 아직 하지 않았습니다.\n\n"
        "**P3 주의:** 저장된 원 모델은 ff42 채점본을 정확히 재생하지만 새 전체 재학습은 56e2 답안으로 660행 미세 차이가 있습니다. "
        "SOURCE_ONLY의 재학습 bit 일치/심사 허용오차는 미확정이며 두 파일을 혼동하지 마세요.\n\n"
        "- ANSWER: 채점용 CSV. 기존 공식 점수는 같은 SHA에만 연결됩니다.\n"
        "- SOURCE_ONLY ZIP: 빈 모델부터 전체 학습하는 코드·노트북·환경.\n"
        "- SAVED_MODELS ZIP: 검증된 가중치와 저장 모델 추론. 새 폴더에 풀어 실행합니다.\n"
        "- RELEASE_MANIFEST.json: 전체 해시·실측 시간·재생 검증.\n\n"
        "배포 데이터는 동봉하지 않습니다. P1_DATA_DIR/P2_DATA_DIR/P3_DATA_DIR로 원 배포 폴더를 지정하세요. "
        "모델이나 lock을 지우지 말고 SOURCE_ONLY를 새로운 폴더에 풀어 P1 RUN_ALL, P2/P3 TRAIN→PREDICT를 실행하세요. "
        "GPU는 한 문제씩 사용하세요. P1/P2 SAVED_PREDICT는 학습 0회이며 cold TRAIN을 비어 있지 않은 저장 모델 폴더에서 실행하지 않습니다.\n\n"
        "검증은 기존 Windows/Python3.12.10/RTX5090 환경에서 완료했습니다. 환경 버전은 각 패키지 명세를 따릅니다. "
        "새 venv, OS 차단망, 다른 GPU 재학습 결정론은 미검증입니다. P1 셀 선택 프로그램은 미복구이며 과거 local-OOF 고정 설정을 사용합니다.\n\n"
        "상세 안내: https://github.com/choihyunjin1/-oceanaidata_track1/blob/codex/p1-qc/docs/FINAL_RELEASE_20260907.md\n",
        encoding="utf-8")
    print(json.dumps({"status": result["status"], "problems": list(candidates)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--p3-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.root.resolve(), args.p3_evidence.resolve(), args.output.resolve())
