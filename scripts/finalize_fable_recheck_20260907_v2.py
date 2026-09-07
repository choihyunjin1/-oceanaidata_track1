"""Record repaired package and disambiguate outer documents; never submit."""
from __future__ import annotations

import json
from pathlib import Path

from build_final_release_20260907_v1 import read, save, sha

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/p2_smooth7_portability_repair_20260907_v2"
SELECTED = Path("C:/Users/cedis/Documents/OceanFinalSelected_20260907")
LEGACY = Path("C:/Users/cedis/Documents/OceanFinalRelease_20260907")
P3 = Path("C:/Users/cedis/Documents/OceanFinalDay_20260907/P3_numeric_cpudet_noshrink_v2")
P2_NEW = Path("C:/Users/cedis/Documents/OceanFinalDay_20260907/P2_L120_s3_smooth7_proj_v2")


def main():
    ready = read(REPORT / "p2-final-ready.json")
    proof = read(REPORT / "candidate-ready.json")
    expected = "794268f15a0a7ac18ecd4dc99757083e159c49d639d2a8a72df3f835414cc481"
    assert proof["status"] == "COLD_AND_SAVED_EXTRACTED_NOTEBOOK_EXACT_PASS"
    assert proof["saved"]["sha256"] == proof["postprocess_cold"]["sha256"] == expected
    assert proof["answer"]["sha256"] == ready["selected_answer_sha256"] == expected
    assert sha(ready["attachment"]["path"]) == ready["attachment"]["sha256"]
    p3_archives = read(P3 / "archives.json")["archives"]
    unchanged = {
        str(LEGACY / "P1/P1_SOURCE_ONLY.zip"): "ba9fabd4b3d87a6030658dd2826c080d64b48952adf0b28b5f70c545c213507a",
        str(LEGACY / "P1/P1_SAVED_MODELS.zip"): "61e796c2304a7899dd2dd94ec280c99f8937c46b2a1f5b197e9090ac9f870fa8",
        str(SELECTED / "P2/P2_FINAL_REPRODUCTION_794268f1.zip"): "aab30bbe5e244098c1ef379b09077ccd4e98bd6a6cd0ccf778109a7f5282054d",
        **{a["path"]: a["sha256"] for a in p3_archives.values()},
    }
    assert all(sha(path) == digest for path, digest in unchanged.items())
    base_manifest_unchanged = {}
    for role in ["SOURCE_ONLY", "SAVED_MODELS"]:
        old = P2_NEW.with_name("P2_L120_s3_smooth7_proj_v1") / role
        new = P2_NEW / role
        base_manifest_unchanged[role] = sha(old / "PACKAGE_MANIFEST.json") == sha(new / "PACKAGE_MANIFEST.json")
        assert base_manifest_unchanged[role]
    manifest = {
        "status": "USER_HOLD_PENDING_FABLE_RECHECK_NOT_FINAL_SUBMITTED",
        "selection_basis": "Public comparison of independently prepared candidates, not score-derived coefficients",
        "P1": {"answer_sha256": "57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687",
               "points": 28.909341, "root": str(LEGACY / "P1"),
               "whole_cold_fits": 7, "whole_cold_seconds": 6323.356084823608,
               "attachment_note": "SOURCE_ONLY plus all 15 model part ZIPs, reassembly manifest/tool and current FORM; count acceptance unverified",
               "caveat": "historical six-cell selection algorithm unrecovered"},
        "P2": {"answer_sha256": expected, "points": 28.373869, "attachment": ready["attachment"],
               "whole_cold_fits": 3, "notebooks": proof["notebooks"],
               "evidence": "actual cold and saved exact on this PC, historical SHA gate replaced with current-run integrity QA",
               "caveat": "CUDA required; other GPU, fresh venv and OS-offline execution not tested"},
        "P3": {"answer_sha256": "70761affca4d3fc6f1d24ae53467e5b185b23465926ebb4b851f0300872cddbd",
               "points": 23.881592, "root": str(P3), "archives": p3_archives,
               "evidence": "completion_1 prefix plus new fits; saved/independent PID replay exact",
               "fresh_cold": "PENDING: do not claim completion or substitute saved replay for training proof"},
        "historical_archives_unchanged_sha256": unchanged,
        "P2_core_package_manifests_unchanged": base_manifest_unchanged,
        "official_deadline_time": None, "general_six_hour_scope": "UNKNOWN",
        "attachment_count_limit": None, "observed_per_file_limit_bytes": 50000000,
        "final_model_receipt": None,
        "do_not_attach": [str(LEGACY / "START_HERE.md"), str(LEGACY / "RELEASE_MANIFEST.json")],
    }
    save(SELECTED / "FINAL_SELECTION_MANIFEST.json", manifest)
    guide = f"""# 09-07 세 문제 최종 검토용 선택 — 제출 보류

Fable 재검토와 사용자 재개 지시 전 최종 확인을 누르지 않습니다. Git push, 로컬 READY, 답안 채점은 최종 모델 접수가 아닙니다. P2 첫 확인창 OK는 누르지 않았으며 Fable의 15:30 포털 읽기 미접수 관찰은 별도 확인 주체의 증거입니다. 재개 시 서버 상태부터 다시 확인합니다.

| 문제 | 답안 SHA 앞8 | Public 점수 | 현재 파일 |
|---|---|---:|---|
| P1 | 57844ef2 | 28.909341 | {LEGACY.as_posix()}/P1/P1_SOURCE_ONLY.zip + SAVED_MODEL_PARTS 전부 + 재조립 manifest/tool + 갱신 FORM |
| P2 | 794268f1 | 28.373869 | {Path(ready['attachment']['path']).as_posix()} |
| P3 | 70761aff | 23.881592 | {P3.as_posix()}/SOURCE_ONLY.zip + SAVED_MODELS.zip + 갱신 외부 FORM/README |

정확한 전체 SHA/bytes/증거는 같은 폴더 FINAL_SELECTION_MANIFEST.json을 읽습니다. Public 합계 81.164802는 Private/적격성 보장이 아닙니다. 이 문서와 manifest가 이전 OceanFinalRelease의 START_HERE/RELEASE_MANIFEST 대신 현재 선택을 설명합니다. **옛 두 상위 파일은 첨부하지 않습니다.** 데이터는 재배포하지 않습니다.

## 문제별 실행과 보류 사유

- P1: SOURCE_ONLY는 빈 모델부터 7fit, 6323.356초 exact 증거. SAVED_MODELS는 새 추출 추론. 파일당 50MB 때문에 590MB 단일 저장 ZIP을 직접 첨부하지 말고 15분할과 재조립 manifest/tool을 모두 사용합니다. 첨부 개수 상한은 미확인입니다. 여섯 셀은 역사적 OOF 고정값이며 선택 프로그램 미복구; 운영진 판정을 대신하지 않습니다.
- P2: v2 ZIP 최상위 README를 먼저 읽습니다. SOURCE_ONLY에서 TRAIN → PREDICT → SMOOTH_PROJECT; SAVED_MODELS에서는 SAVED_PREDICT. **두 역할 폴더를 합치지 않습니다.** 옛 PROJECT.ipynb 및 내부 원형 README의 중간 답안 제출 지시는 실행 진입점이 아닙니다. 최종 출력은 05_answer/submission_p2_L120_s3_smooth7_proj.csv. 역사적 기반 SHA 차이는 기록하며 현재 실행 무결성/별도 PID 재생은 계속 검사합니다. CUDA 필수 조건은 유지됩니다.
- P3: 두 ZIP은 P2와 달리 source + 모델 확장 묶음이므로 같은 새 루트에 합칩니다. 새 학습은 비어 있는 별도 SOURCE_ONLY에서 RUN_TRAINING; 저장 모델 추론은 RUN_INFERENCE → REPLAY_INFERENCE. 옛 TRAIN/PREDICT/UNBOUNDED는 실행하지 않습니다. 현재 근거는 completion_1 + saved replay이고 fresh_cold_2 및 no-shrink fresh 대조는 대기입니다. 저장 모델 추론은 필수 학습 재현 증거의 대체물이 아닙니다.

P2 새 ZIP {ready['attachment']['bytes']} bytes / SHA256 {ready['attachment']['sha256']}. 기존 P2 v1 ZIP은 보존하지만 최종 폼의 옛 첨부를 그대로 확정하지 마세요. 모든 첨부 선택/확정은 재개 승인 뒤 진행합니다.

모델 마감 시각, 첨부 개수, 재현 대상 답안/허용오차, 일반 모델 6시간 적용 범위는 미확인입니다. 모델 코드·답안·가중치에 Public 역산 조정을 하지 않았습니다. 이 검토 문서는 최종 제출 완료를 선언하지 않습니다.
"""
    (SELECTED / "FINAL_SELECTION_20260907.md").write_text(guide, encoding="utf-8")
    save(REPORT / "final-selection-manifest.json", manifest)
    before = {
        str(LEGACY / "START_HERE.md"): "26eedf295437880d82e6cfe8811284cc34af75b7c6dd87f5d8984149daa4448d",
        str(LEGACY / "RELEASE_MANIFEST.json"): "b47d03d230260f5bf256a7f1bffafbb ae07f858275c053626ecb2a378025f193".replace(" ", ""),
        str(LEGACY / "P1/FORM.md"): "a9236294c9344da155721b714b874eb76362382744621ff9fc6bb2e56f90138a",
        str(P3 / "FORM.md"): "00ec64e3c56b6fb91ca99a56fc2014cc986c0581890cf6d6d6c5131d58e19158",
        str(P3 / "README.md"): "1901f7d50c52c71708d9937e91414e7d3c2106cffa74324181547cd6befdc462",
    }
    save(REPORT / "outer-document-change-sha.json", {path: {"before": old, "after": sha(path)} for path, old in before.items()})
    print(json.dumps({"status": manifest["status"], "P2_attachment": ready["attachment"],
                      "old_archives_unchanged": True, "final_confirmation": False}))


if __name__ == "__main__":
    main()
