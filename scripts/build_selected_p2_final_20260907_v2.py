"""Freeze highest-scored reviewed P2 as one final portal attachment, with audit."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from build_final_release_20260907_v1 import read, save, sha

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path("C:/Users/cedis/Documents/OceanFinalDay_20260907/P2_L120_s3_smooth7_proj_v2")
OUT = Path("C:/Users/cedis/Documents/OceanFinalSelected_20260907/P2_v2")
REPORT = REPO / "reports/p2_smooth7_portability_repair_20260907_v2"
EXPECTED = "794268f15a0a7ac18ecd4dc99757083e159c49d639d2a8a72df3f835414cc481"


def main():
    completed = read(SOURCE / "result.json")
    assert completed["status"] == "COLD_AND_SAVED_EXTRACTED_NOTEBOOK_EXACT_PASS"
    assert completed["answer"]["sha256"] == EXPECTED
    official = read(REPO / "reports/p2_l120_s3_smooth7_projection_20260907_v1/official-receipt.json")
    assert official["sha256"] == EXPECTED and official["public_score"] == 28.373869 and official["remaining_after"] == 0
    OUT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(exist_ok=True)
    payload = OUT / "PACKAGE"
    payload.mkdir()
    manifest = {}
    for a in completed["archives"]:
        origin = SOURCE / a["file"]
        assert sha(origin) == a["sha256"]
        role = "SOURCE_ONLY" if "SOURCE_ONLY" in a["file"] else "SAVED_MODELS"
        target = payload / role
        target.mkdir()
        with zipfile.ZipFile(origin) as z:
            assert z.testzip() is None
            for name in z.namelist():
                assert (target / name).resolve().is_relative_to(target.resolve())
                if name.endswith("/"):
                    continue
                assert Path(name).suffix.lower() not in {".csv", ".npz", ".parquet", ".log", ".pyc"}
                assert not any(part in {".git", ".env", "__pycache__"} for part in Path(name).parts)
                with z.open(name) as stream:
                    assert hashlib.file_digest(stream, "sha256").hexdigest() == a["member_sha256"][name]
            z.extractall(target)
        assert not any(p.is_file() for p in (target / "05_answer").rglob("*"))
        if role == "SOURCE_ONLY":
            assert not any(p.is_file() for p in (target / "03_model").rglob("*"))
        manifest[role] = {"origin_zip_sha256": a["sha256"], "files": a["files"]}
    answer = Path(completed["answer"]["path"])
    assert sha(answer) == EXPECTED
    (payload / "ANSWER").mkdir()
    shutil.copy2(answer, payload / "ANSWER" / answer.name)
    evidence = payload / "EVIDENCE"
    evidence.mkdir()
    for name, src in {
        "cold-and-saved-notebook-receipt.json": SOURCE / "result.json",
        "official-public-receipt.json": REPO / "reports/p2_l120_s3_smooth7_projection_20260907_v1/official-receipt.json",
        "internal-result.json": REPO / "reports/p2_l120_s3_smooth7_projection_20260907_v1/result.json",
        "independent-qa.json": REPO / "reports/p2_l120_s3_smooth7_projection_20260907_v1/independent-qa.json",
    }.items():
        shutil.copy2(src, evidence / name)
    guide = """# OCN-02 / P2 최종 재현 제출물 — L120 s3 smooth7 projection

## 최종 선택과 정확한 답안

배포 데이터로 처음부터 학습한 DeepSets C3 L120 3-seed 평균 + 층별 ±30분 평활 + 공개 endpoint clip/PAVA입니다. 2026-09-07 Public RMSE 0.395254℃, 점수28.373869. 정상적인 후보 비교로 선택했으며 반환 점수 역산으로 계수/창/seed를 정하지 않았습니다. Private 점수와 운영진 적격성 심사는 별개입니다.

답안은 ANSWER/submission_p2_L120_s3_smooth7_proj.csv, 26061행, 1253654bytes, UTF-8/LF, 열 station,layer,time,temp입니다. SHA256: 794268f15a0a7ac18ecd4dc99757083e159c49d639d2a8a72df3f835414cc481.

## 환경 차이와 역사적 진입점

기반 L120 답안 SHA가 역사적 fee6118b와 달라도 이번 실행의 무결성 QA를 통과했다면 차이를 기록하고 평활·투영을 계속합니다. 소스·데이터·모델 무결성 및 같은 실행의 별도 PID 재생 검사는 유지합니다. 다른 GPU/드라이버 결과 일치 또는 CPU 실행 지원을 새로 증명한 것은 아닙니다. CUDA는 여전히 필요합니다. 내부 원형 README의 중간 답안 제출 지시와 PROJECT.ipynb는 이력용이며 실행하지 마세요.

## 학습부터 재현하기 — SOURCE_ONLY

1. **SOURCE_ONLY와 SAVED_MODELS는 독립 실행 폴더입니다. 서로 합치지 마세요.** SOURCE_ONLY의 03_model/05_answer는 비어 있습니다. ANSWER의 CSV를 이 폴더로 복사하지 않습니다.
2. 배포 P2_profile_restore 폴더를 환경변수 P2_DATA_DIR로 지정합니다. 데이터 원본은 재배포하지 않습니다. SOURCE_ONLY/requirements.txt의 고정 라이브러리를 미리 설치한 Python3.12.10/CUDA 환경을 사용합니다. 학습·추론 중 모델/자료 다운로드는 없습니다.
3. SOURCE_ONLY를 작업 디렉터리로 Jupyter에서 **TRAIN.ipynb → PREDICT.ipynb → SMOOTH_PROJECT.ipynb**를 순서대로 실행합니다. 각 노트북의 작업 폴더가 SOURCE_ONLY인지 확인합니다. 긴 수동 대기 없이 이어서 실행하세요(원형 코어의 내부1800초 시퀀스 한도 보존).
4. **최종 출력은 SOURCE_ONLY/05_answer/submission_p2_L120_s3_smooth7_proj.csv**입니다. submission_p2_L120_3seed.csv는 중간 산출물이므로 제출/비교 대상이 아닙니다. 생성 SHA를 위 정본과 대조하세요.
5. TRAIN은 seed20260901/02/03의 3개 모델을 새로 학습합니다. 120epochs, batch4096, AdamW lr.001/wd.0001, CPU2/CUDA0, loader0. 모델은03_model, 학습·검증 기록은04_logs, 후처리 재생 receipt는06_docs에 생깁니다. 실패/사용된 폴더의 lock/모델을 삭제해 재시작하지 말고 새 추출 폴더를 사용합니다.

## 저장 모델로 재현하기 — SAVED_MODELS

SAVED_MODELS를 별도 작업 폴더로 두고 같은 환경변수를 지정한 후 **SAVED_PREDICT.ipynb**만 실행합니다. 학습0회, 베이스 추론·별도 PID 재생·고정 평활/투영 및 별도 PID 재생을 수행합니다. 최종 파일명과 SHA는 같습니다.

## 파일·규정·재현 감사

- 01_data=배포 데이터 참조, 02_code=실행 소스, 03_model=학습 산출 모델, 04_logs=실행 후 로그, 05_answer=실행 후 답안, 06_docs=설정·QA 근거. 외부 관측/재분석/예보/사전학습 가중치/hidden truth/과거 답안 입력/리더보드 적합 상수 없음. 일반 모델3개를 배포 관측으로 학습합니다.
- 창7은 이미 공개된 내부 OOF에서 선택됐습니다. fresh confirmatory 검증으로 표현하지 않습니다. 내부 주평가 B3 .463529468→.441853923℃, pooled1.236732066→1.204715055℃. 범위가 다른 Public/Private 개선을 보장하지 않습니다.
- 학습/추론 핵심은 원 저장소 접근 및 네트워크 연결을 Python audit hook으로 막고 별도 경로에서 실제 실행했습니다. OS 차단망/새 가상환경 전체 검증을 주장하지 않습니다. 합성-only 사전학습 예외는 사용하지 않습니다.
- 실제 ZIP 추출 빈 모델 노트북 TRAIN142.218s / PREDICT34.375s / SMOOTH_PROJECT12.500s, 저장 모델48.047s. Windows/Python3.12.10/Ryzen7800X3D/RTX5090에서 측정했습니다. **일반 모델에 공식6시간 제한이 적용되는지는 미확인**이며 이 수치를 운영진 규정 충족 인증으로 표현하지 않습니다.
- 부모의 고정 README/PROJECT/config에는 원형 L120 중간 출력 설명이 보존되어 있습니다. **이 최상위 README와 각 폴더 CURRENT_README가 현재 최종 출력 및 실행 순서에 우선합니다.** 원형 config의 'no projection'은 베이스만 설명하며 새 SMOOTH_PROJECT가 최종 고정 후처리입니다.
- EVIDENCE는 내부집계·실제 cold/saved SHA 재생·공식 Public receipt입니다. 최종 모델 접수는 포털에서 별도로 확인해야 하며, 접수 전에는 완료로 주장하지 않습니다.
"""
    (payload / "README.md").write_text(guide, encoding="utf-8")
    audit = {"status": "LOCAL_FINAL_PACKAGE_REVIEW_PASS_NOT_ORGANIZER_CERTIFICATION",
             "selected_sha256": EXPECTED, "public_score": 28.373869,
             "checks": {"distributed_only_lineage": True, "pretrained_weights_used": False,
                        "hidden_truth_used": False, "Public_fitted_parameters": False,
                        "whole_cold_and_saved_answer_exact": True, "source_only_empty_models": True,
                        "source_zip_members_match": True, "no_raw_or_unrelated_archives": True,
                        "same_fixed_smoothing_and_projection": True},
             "caveats": ["OOF window selection retrospective", "general six-hour official scope unknown",
                         "not OS-offline/fresh-venv tested", "final eligibility decided by organizer"],
             "ancestry": manifest, "uploads": 0, "final_model_receipt": None}
    save(evidence / "FINAL_AUDIT.json", audit)
    pins = {p.relative_to(payload).as_posix(): sha(p) for p in payload.rglob("*") if p.is_file()}
    save(payload / "FINAL_MANIFEST.json", {"files": pins})
    target = OUT / "P2_FINAL_REPRODUCTION_794268f1_v2.zip"
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(payload.rglob("*")):
            z.write(p, p.relative_to(payload).as_posix() + ("/" if p.is_dir() else ""))
    assert target.stat().st_size < 50000000
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name, digest in pins.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    result = {"status": "FINAL_SELECTED_P2_READY", "title": "P2 L120 3-seed smooth7 endpoint projection",
              "attachment": {"path": str(target), "bytes": target.stat().st_size, "sha256": sha(target)},
              "selected_answer_sha256": EXPECTED, "audit": audit, "final_submitted": False}
    save(OUT / "FINAL_READY.json", result)
    save(REPORT / "p2-final-ready.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
