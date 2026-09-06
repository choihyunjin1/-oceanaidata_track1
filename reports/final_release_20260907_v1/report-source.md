# 최종 패키지 정리 — 2026-09-07

**세 문제의 로컬 소스·모델·답안 분리와 실제 저장 ZIP 추출 재생 검증을 완료했다.** P1/P2 재학습 답안은 채점본과 같고, P3 채점 당시 저장 모델은 채점본을 정확히 재생한다. 다만 **P3 새 전체 재학습 답안은 채점본과 다르므로 최종 재학습 적격성까지 통과했다고 주장하지 않는다.** 대회 최종 업로드·잠금은 수행하지 않았다.

## 고정된 선택과 기존 재학습 증거

| 문제 | 선택 | 같은 답안의 공식 점수 | 전체 재학습 증거 |
|---|---|---:|---|
| P1 | 원형 O+B+MS-TCN `57844ef2` | 28.909341 | 배포자료→7fit→QA→답안 6323.356초, 역사적 답안 exact; 09-07 서버 중복 응답 |
| P2 | L120 3-seed `fee6118b` | 28.077280 | 새 ZIP 추출→TRAIN/PREDICT notebook→3fit→QA→답안 173.010초, 모델 및 답안 exact |
| P3 | numeric lead `ff42a6a0` | 23.741446 | 새 source-only 12 backbone+5 router=17fit, 실제 TRAIN/PREDICT notebook 완료, 수치 replay까지 1497.613초; **채점본과 다름** |

P1 28.909341은 새 점수를 받은 것이 아니라 같은 SHA에 연결된 기존 공식 값이다. 원형 router는 과거 로컬 OOF 고정 설정으로 복원했으며 셀 선택 프로그램 자체는 복구하지 않았다.

P3 새 cold SHA는 `56e289af18d1ebfe4cf37410a277668a5cc9d502d5e5692ac99ab20ba6828784`이다. 채점본과 660/1200행이 다르고 최대 절대차 0.0035060905563752964m, 평균 절대차 0.0001601851401365397m, **예측 간** RMSE 0.0005459156056845382m이다. 이는 hidden truth 대비 RMSE가 아니다. 자체 모델 fresh-process replay는 통과했지만 채점 점수는 승계하지 않는다. 차이의 원인은 확정하지 않았고 재시작·재튜닝하지 않았다. 채점 당시 모델과 새 cold 모델은 별개로 보존했다.

## 이번 정리의 검증

- P1/P2 패키지: 원 실행은 불변, 별도 SOURCE_ONLY/SAVED_MODELS/ANSWER. P1 saved ZIP은 15개 분할 ZIP과 합산 원 ZIP SHA 검증.
- GitHub 실행 소스: `final_packages/`에 ZIP과 같은 73개 source 파일을 공개. 가중치·raw·CSV·cache 없음. Git index에 저장한 73개 원 바이트 해시도 일치.
- Git이 보존하지 않는 빈 runtime 폴더는 비파괴 `PREPARE_DIRECTORIES.py`로 복구. 새 복사본에서 빈 모델/답안 폴더 복구 테스트 통과.
- 소스 노트북 5개 schema PASS. 이는 실제 P1 RUN_ALL notebook 실행 완료를 의미하지 않으며 P1은 동등 CLI 전체 실행의 증거를 사용한다.
- 최종 focused 검사: release/P1 original/P2 cold/P3 saved 묶음 **57 PASS (4.08초)**, 최종 출판·패키지 Python 9개 Ruff PASS. 작업 중 다른 묶음과 중복되므로 합산하지 않는다. [출판 전 검증 영수증](publication-qa.json).
- 출판 대상 Python 114개 Ruff PASS. 후속 새 코드도 focused Ruff/pytest를 적용한다.
- P1/P2 저장 ZIP 실제 새 추출 notebook은 각각 35.656초/30.891초, P3 채점 모델 ZIP 실제 새 추출 CLI는 6.026초에 채점 답안 SHA exact를 확인했다. 모두 새 학습 0회이다.
- P3 독립 QA 18/18 PASS: 기존/신규 학습 계보 분리, 17fit, 자체 replay, 채점 모델 새 ZIP replay, 수치 경로 AST 동등성, hidden/과거 답안 입력 0 확인. [독립 QA](p3-independent-qa.json).
- 로컬 `START_HERE.md`와 문제별 `FORM.md`가 실제 선택 파일을 지정한다. P3 `NOT_SCORED_whole_cold.csv`와 `P3_COLD_SAVED_MODELS.zip`은 미채점 보조 증거이며 채점 파일 대신 선택하지 않는다. [최종 해시·검증 manifest](release-manifest.json).

## 제외·한계

Git에 원본 데이터·모델·답안·예측 NPZ/parquet·attempt lock·실행 로그·ZIP·credentials를 포함하지 않는다. P1 로컬 saved ZIP에는 불변 MS 모델 integrity guard가 요구하는 TRAIN-derived probe만 포함하며 Git에는 없다. SOURCE_ONLY에는 probe도 없다.

새 venv·OS 차단망·다른 GPU 재학습 결정론은 미검증이다. Python network/file audit hook은 OS 보안 경계가 아니다. 현재 브라우저 탭은 `Debugger unattached`로 읽기 연결 재시도 1회 후 중단했다. 따라서 최신 포털 첨부 용량/개수·최종 잠금 효과 실측은 이번 기록에서 미확인이다. 최종 제출/잠금/업로드는 이번 요청에 포함하지 않는다.

## 근거

- [P1 source package](../p1_original_source_package_20260906_v1/report-source.md), [preupload QA](../p1_original_source_package_20260906_v1/preupload-qa-20260907.json), [공식 중복 응답](../p1_original_source_package_20260906_v1/official-attempt-20260907.json).
- [P2 L120 cold](../p2_L120_portable_cold_20260906_v1/report-source.md).
- [P3 numeric 기존 결과](../p3_numeric_candidate_20260906_v1/result.json), [공식 후보 채점](../official_candidate_submissions_20260906_evening_v1/result.json).
- [실행 전 계획](plan.md), [최종 사용·제출 안내](../../docs/FINAL_RELEASE_20260907.md), [남은 공백](gap-matrix.md).
